//! Landmark (subset-of-regressors) Gaussian process regression on 1-D inputs.
//!
//! Multi-output: one set of landmarks and one factorisation serve every
//! response column, which is what makes fitting a few hundred genes against one
//! pseudotime axis cheap. The responses enter only through `A r` and the final
//! GEMM.
//!
//! The posterior mean follows mellon's `_sparse_solve`, which is the estimator
//! Palantir's gene trends are built on:
//!
//! ```text
//! Lp      = chol(k(u,u) + jitter I)     # m x m, u = landmarks
//! A       = Lp^-1 k(u,x)                # m x n, triangular solve
//! G       = A A^T,   P = A r            # accumulated over chunks of rows
//! B       = G / sigma^2 + I             # m x m
//! L_B     = chol(B)
//! weights = Lp^-T L_B^-T L_B^-1 (P / sigma^2)
//! mean    = k(x_new, u) @ weights
//! ```
//!
//! `A` is never materialised at its full `m x n`: the two accumulators `G` and
//! `P` are both landmark-sized, so training rows stream through in chunks and
//! peak memory is independent of `n`.
//!
//! Everything runs in `f64` regardless of the caller's float type, which is not
//! a style slip. On a dense 1-D grid at a length scale spanning the domain, the
//! landmark Gram is near-constant: one eigenvalue around `m` and the rest
//! collapsing toward zero. At 500 landmarks and the reference jitter of `1e-6`
//! its condition number is roughly `4e8`, which leaves eight digits in `f64`
//! and makes the matrix numerically indefinite in `f32`. The triangular solve
//! then scales trailing rows of `A` up by ~`1/sqrt(jitter)` with mixed signs,
//! and `A A^T` sums `n` of those per entry, so the accumulation wants the
//! headroom too.
//!
//! ### References
//!
//! Quiñonero-Candela and Rasmussen, JMLR, 2005 (subset of regressors).
//! Otto, Zhou, et al., bioRxiv, 2023 (mellon).

use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::{Accum, Mat, MatRef, Par, Side};
use rayon::prelude::*;

use crate::ml::gp::kernels::fill_matern52_cross_1d;
use crate::prelude::*;
use crate::utils::faer_parallelism;

/// Default training rows per accumulation chunk.
///
/// Sets the per-thread footprint together with the landmark and response
/// counts, at `8 * (m * chunk + chunk * p)` bytes for the two scratch buffers.
/// 2048 keeps that in the tens of megabytes at the sizes gene trends produce
/// while still handing the triangular solve and the GEMM enough columns to
/// amortise their own setup.
const GP_DEFAULT_CHUNK: usize = 2048;

/// Minimum landmarks a fit will accept.
///
/// Two points define no curvature at all, so anything below this is a caller
/// error rather than a degenerate-but-valid fit.
const GP_MIN_LANDMARKS: usize = 2;

//////////////
// Params //
//////////////

/// Hyperparameters for [fit_landmark_gp].
///
/// Every field is `f64` because the fit runs in `f64` regardless of the
/// caller's float type; see the module documentation.
///
/// The defaults are **Palantir's**, not mellon's: `presults.py` passes
/// `dict(sigma=1, ls=1)` explicitly. Left to itself mellon derives `ls` from
/// nearest-neighbour distances and defaults `sigma` to zero, so comparing
/// against a bare `mellon.FunctionEstimator()` will not reproduce these curves.
///
/// They are also a strong smoothing prior: with the input scaled to `[0, 1]`,
/// `length_scale = 1.0` spans the whole domain and `sigma = 1.0` puts the noise
/// at roughly the signal scale of log-normalised expression, so the posterior
/// is prior-dominated and will flatten genuine transient structure. Shortening
/// `length_scale` is how a caller checks whether a feature is real.
#[derive(Clone, Copy, Debug)]
pub struct LandmarkGpParams {
    /// Matérn-5/2 length scale. Must be strictly positive.
    ///
    /// Palantir passes `1.0`. mellon's own default is `None`, meaning "derive
    /// it from the data", which is a different estimator.
    pub length_scale: f64,
    /// Noise standard deviation. Must be strictly positive.
    ///
    /// Palantir passes `1.0`. mellon's own default is `0`, which would divide
    /// by zero; we reject non-positive values rather than inherit the bug.
    pub sigma: f64,
    /// Added to the diagonal of `k(u, u)` before its Cholesky. mellon: `1e-6`.
    pub jitter: f64,
    /// Cholesky retries, each multiplying the jitter by ten.
    ///
    /// mellon simply errors and suggests raising the jitter. The ladder does
    /// that automatically and reports what it landed on via
    /// [LandmarkGpFit::jitter_used], so a silently over-regularised fit is
    /// still visible.
    pub max_jitter_retries: usize,
    /// Training rows per accumulation chunk.
    ///
    /// Per-thread scratch is `8 * (n_landmarks * chunk_size + chunk_size *
    /// n_outputs)` bytes. Zero is treated as the default.
    pub chunk_size: usize,
}

impl LandmarkGpParams {
    /// Construct the hyperparameters.
    ///
    /// ### Params
    ///
    /// * `length_scale` - Matérn-5/2 length scale, strictly positive.
    /// * `sigma` - Noise standard deviation, strictly positive.
    /// * `jitter` - Diagonal loading for the landmark Cholesky.
    /// * `max_jitter_retries` - Retries, each multiplying the jitter by ten.
    /// * `chunk_size` - Training rows per accumulation chunk.
    ///
    /// ### Returns
    ///
    /// The parameter set. Values are validated in [fit_landmark_gp], not here.
    pub fn new(
        length_scale: f64,
        sigma: f64,
        jitter: f64,
        max_jitter_retries: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            length_scale,
            sigma,
            jitter,
            max_jitter_retries,
            chunk_size,
        }
    }
}

impl Default for LandmarkGpParams {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            sigma: 1.0,
            jitter: 1e-6,
            max_jitter_retries: 3,
            chunk_size: GP_DEFAULT_CHUNK,
        }
    }
}

/////////////////
// Fitted state //
/////////////////

/// A fitted landmark Gaussian process.
///
/// Holds everything prediction needs, so the training data can be dropped.
#[derive(Clone, Debug)]
pub struct LandmarkGpFit {
    /// Landmark coordinates, `m` of them.
    pub landmarks: Vec<f64>,
    /// Posterior weights, `m` by `n_outputs`.
    pub weights: Mat<f64>,
    /// Length scale the fit used, needed to rebuild the kernel at prediction.
    pub length_scale: f64,
    /// Jitter the landmark Cholesky actually succeeded at.
    ///
    /// Above the requested value means the Gram was singular there and the
    /// retry ladder had to over-regularise, which shows up as a flatter curve.
    pub jitter_used: f64,
    /// `k(u, u)` without the jitter, cached so that predicting on the landmarks
    /// themselves is a single GEMM rather than a rebuild.
    k_uu: Mat<f64>,
}

///////////////
// Fitting //
///////////////

/// Fit the subset-of-regressors posterior mean.
///
/// The prior mean is fixed at zero, matching mellon and therefore Palantir. On
/// log-normalised expression that means the trend is shrunk toward zero, which
/// is part of the estimator, not an oversight.
///
/// Parallelism: rayon splits the training rows into one contiguous slab per
/// thread and every faer call inside a slab runs sequentially, so the outer
/// split owns the machine. The slab partials are summed back in index order
/// rather than by a `reduce`, which keeps the result reproducible across runs.
///
/// The inputs and the landmarks are `f64` while the responses stay generic:
/// the 1-D coordinate is one number per observation and is widened for the
/// kernel regardless, whereas the response block is the large object and is
/// whatever the caller stores it as.
///
/// ### Params
///
/// * `x` - Training inputs, one scalar per row of `y`.
/// * `y` - Training responses, `x.len()` by `n_outputs`.
/// * `landmarks` - Inducing points. For a trend fit these are the prediction
///   grid, which lets [LandmarkGpFit::predict_on_landmarks] reuse `k(u, u)`.
/// * `sample_weights` - Optional per-observation weights, i.e. observation `i`
///   is treated as having noise variance `sigma² / w_i`. `None` weights every
///   observation equally.
/// * `params` - Hyperparameters.
///
/// ### Returns
///
/// The fitted model, or the matching [BixverseErrors] variant if an input is
/// empty, misaligned, non-finite, or a hyperparameter is out of range.
pub fn fit_landmark_gp<T>(
    x: &[f64],
    y: MatRef<'_, T>,
    landmarks: &[f64],
    sample_weights: Option<&[f64]>,
    params: &LandmarkGpParams,
) -> Result<LandmarkGpFit, BixverseErrors>
where
    T: BixverseFloat,
{
    let n = x.len();
    let m = landmarks.len();
    let p = y.ncols();

    validate_inputs(x, y, landmarks, sample_weights, params)?;

    // Landmark Gram and its Cholesky. Both depend only on the grid, so neither
    // sees the training data at all.
    let mut k_uu = Mat::<f64>::zeros(m, m);
    fill_matern52_cross_1d(k_uu.as_mut(), landmarks, landmarks, params.length_scale);

    let (llt_p, jitter_used) = factorise_landmarks(&k_uu, params)?;
    let lp = llt_p.L();

    let chunk = if params.chunk_size == 0 {
        GP_DEFAULT_CHUNK
    } else {
        params.chunk_size
    };

    // One contiguous slab per thread, summed back in index order.
    let n_threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let slab = n.div_ceil(n_threads).max(1);
    let slab_starts: Vec<usize> = (0..n).step_by(slab).collect();

    let partials: Vec<(Mat<f64>, Mat<f64>)> = slab_starts
        .par_iter()
        .map(|&slab_start| {
            let slab_end = (slab_start + slab).min(n);
            // Never allocate scratch for rows this slab does not hold. A branch
            // with a handful of cells would otherwise take the full
            // `chunk_size` buffer on every thread.
            let chunk = chunk.min(slab_end - slab_start).max(1);

            let mut g_local = Mat::<f64>::zeros(m, m);
            let mut p_local = Mat::<f64>::zeros(m, p);
            let mut a = Mat::<f64>::zeros(m, chunk);
            let mut r = Mat::<f64>::zeros(chunk, p);

            for start in (slab_start..slab_end).step_by(chunk) {
                let c = (start + chunk).min(slab_end) - start;

                // k(u, x_chunk), fusing the 1-D distance into the kernel.
                fill_matern52_cross_1d(
                    a.as_mut().subcols_mut(0, c),
                    landmarks,
                    &x[start..start + c],
                    params.length_scale,
                );

                // A <- Lp^-1 k(u, x_chunk)
                solve_lower_triangular_in_place(lp, a.as_mut().subcols_mut(0, c), Par::Seq);

                // Responses widened to f64. The prior mean is zero, so the
                // residual is the response itself.
                for j in 0..p {
                    for i in 0..c {
                        r[(i, j)] = y[(start + i, j)].to_f64().unwrap_or(f64::NAN);
                    }
                }

                // Weighted least squares: scaling column i of A and row i of r
                // by sqrt(w_i) turns the two accumulations below into
                // `A W A^T` and `A W r`.
                if let Some(w) = sample_weights {
                    for i in 0..c {
                        let s = w[start + i].sqrt();
                        for k in 0..m {
                            a[(k, i)] *= s;
                        }
                        for j in 0..p {
                            r[(i, j)] *= s;
                        }
                    }
                }

                // G += A A^T, lower triangle only; the upper is mirrored later.
                triangular_matmul(
                    g_local.as_mut(),
                    BlockStructure::TriangularLower,
                    Accum::Add,
                    a.as_ref().subcols(0, c),
                    BlockStructure::Rectangular,
                    a.as_ref().subcols(0, c).transpose(),
                    BlockStructure::Rectangular,
                    1.0_f64,
                    Par::Seq,
                );

                // P += A r
                matmul(
                    p_local.as_mut(),
                    Accum::Add,
                    a.as_ref().subcols(0, c),
                    r.as_ref().subrows(0, c),
                    1.0_f64,
                    Par::Seq,
                );
            }

            (g_local, p_local)
        })
        .collect();

    let mut g = Mat::<f64>::zeros(m, m);
    let mut weights = Mat::<f64>::zeros(m, p);
    for (g_local, p_local) in &partials {
        for j in 0..m {
            for i in j..m {
                g[(i, j)] += g_local[(i, j)];
            }
        }
        for j in 0..p {
            for i in 0..m {
                weights[(i, j)] += p_local[(i, j)];
            }
        }
    }

    // B = G / sigma^2 + I, mirroring the lower triangle as we go.
    let inv_sigma_sq = 1.0 / (params.sigma * params.sigma);
    let mut b = Mat::<f64>::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            let v = g[(i, j)] * inv_sigma_sq;
            b[(i, j)] = v;
            b[(j, i)] = v;
        }
        b[(j, j)] += 1.0;
    }

    // Cannot fail in exact arithmetic: the smallest eigenvalue is at least one.
    // A failure therefore means a non-finite value survived the input checks.
    let llt_b = b
        .llt(Side::Lower)
        .map_err(|_| BixverseErrors::GpPosteriorCholeskyFailed)?;
    let lb = llt_b.L();

    // weights currently holds P. Scale to A r_l, then the three solves.
    let par = faer_parallelism();
    for j in 0..p {
        for i in 0..m {
            weights[(i, j)] *= inv_sigma_sq;
        }
    }
    solve_lower_triangular_in_place(lb, weights.as_mut(), par);
    solve_upper_triangular_in_place(lb.transpose(), weights.as_mut(), par);
    solve_upper_triangular_in_place(lp.transpose(), weights.as_mut(), par);

    Ok(LandmarkGpFit {
        landmarks: landmarks.to_vec(),
        weights,
        length_scale: params.length_scale,
        jitter_used,
        k_uu,
    })
}

impl LandmarkGpFit {
    /// Number of response columns the fit carries.
    ///
    /// ### Returns
    ///
    /// The output dimension.
    pub fn n_outputs(&self) -> usize {
        self.weights.ncols()
    }

    /// Posterior mean at arbitrary 1-D points.
    ///
    /// ### Params
    ///
    /// * `x_new` - Points to evaluate at.
    ///
    /// ### Returns
    ///
    /// `x_new.len()` by `n_outputs`, cast to `T`.
    pub fn predict<T>(&self, x_new: &[f64]) -> Mat<T>
    where
        T: BixverseFloat,
    {
        let mut k_new = Mat::<f64>::zeros(x_new.len(), self.landmarks.len());
        fill_matern52_cross_1d(k_new.as_mut(), x_new, &self.landmarks, self.length_scale);

        self.apply_weights(k_new.as_ref())
    }

    /// Posterior mean on the landmarks themselves.
    ///
    /// When the landmarks were chosen as the prediction grid, `k(grid, u)` and
    /// `k(u, u)` are the same matrix, so this reuses the cached one instead of
    /// rebuilding it. That is the gene-trends case.
    ///
    /// ### Returns
    ///
    /// `n_landmarks` by `n_outputs`, cast to `T`.
    pub fn predict_on_landmarks<T>(&self) -> Mat<T>
    where
        T: BixverseFloat,
    {
        self.apply_weights(self.k_uu.as_ref())
    }

    /// Multiply a cross-covariance block by the posterior weights.
    ///
    /// ### Params
    ///
    /// * `k_cross` - `n_new` by `n_landmarks` cross-covariance.
    ///
    /// ### Returns
    ///
    /// `n_new` by `n_outputs`, cast to `T`.
    fn apply_weights<T>(&self, k_cross: MatRef<'_, f64>) -> Mat<T>
    where
        T: BixverseFloat,
    {
        let mut out = Mat::<f64>::zeros(k_cross.nrows(), self.weights.ncols());
        matmul(
            out.as_mut(),
            Accum::Replace,
            k_cross,
            self.weights.as_ref(),
            1.0_f64,
            faer_parallelism(),
        );

        Mat::<T>::from_fn(out.nrows(), out.ncols(), |i, j| {
            T::from_f64(out[(i, j)]).unwrap_or_else(T::zero)
        })
    }
}

/////////////
// Helpers //
/////////////

/// Validate everything that would otherwise surface as an unexplained Cholesky
/// failure much later.
///
/// ### Params
///
/// * `x` - Training inputs.
/// * `y` - Training responses.
/// * `landmarks` - Inducing points.
/// * `sample_weights` - Optional per-observation weights.
/// * `params` - Hyperparameters.
///
/// ### Returns
///
/// `Ok(())`, or the matching [BixverseErrors] variant.
fn validate_inputs<T>(
    x: &[f64],
    y: MatRef<'_, T>,
    landmarks: &[f64],
    sample_weights: Option<&[f64]>,
    params: &LandmarkGpParams,
) -> Result<(), BixverseErrors>
where
    T: BixverseFloat,
{
    if x.is_empty() {
        return Err(BixverseErrors::GpEmptyInput { name: "x" });
    }
    if y.ncols() == 0 {
        return Err(BixverseErrors::GpEmptyInput { name: "y" });
    }
    if x.len() != y.nrows() {
        return Err(BixverseErrors::GpDimensionMismatch {
            n_x: x.len(),
            n_y: y.nrows(),
        });
    }
    if landmarks.len() < GP_MIN_LANDMARKS {
        return Err(BixverseErrors::GpEmptyInput { name: "landmarks" });
    }

    // Finiteness first, so the range comparisons that follow are total.
    if !params.length_scale.is_finite() || params.length_scale <= 0.0 {
        return Err(BixverseErrors::GpInvalidHyperparameter {
            name: "length_scale",
            value: params.length_scale,
        });
    }
    if !params.sigma.is_finite() || params.sigma <= 0.0 {
        return Err(BixverseErrors::GpInvalidHyperparameter {
            name: "sigma",
            value: params.sigma,
        });
    }
    if !params.jitter.is_finite() || params.jitter < 0.0 {
        return Err(BixverseErrors::GpInvalidHyperparameter {
            name: "jitter",
            value: params.jitter,
        });
    }

    if x.iter().any(|v| !v.is_finite()) {
        return Err(BixverseErrors::GpNonFiniteInput { name: "x" });
    }
    if landmarks.iter().any(|v| !v.is_finite()) {
        return Err(BixverseErrors::GpNonFiniteInput { name: "landmarks" });
    }
    let y_bad = (0..y.ncols())
        .into_par_iter()
        .any(|j| (0..y.nrows()).any(|i| !y[(i, j)].is_finite()));
    if y_bad {
        return Err(BixverseErrors::GpNonFiniteInput { name: "y" });
    }

    if let Some(w) = sample_weights {
        if w.len() != x.len() {
            return Err(BixverseErrors::GpDimensionMismatch {
                n_x: x.len(),
                n_y: w.len(),
            });
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(BixverseErrors::GpNonFiniteInput {
                name: "sample_weights",
            });
        }
        // A zero weight drops its observation, so all-zero weights leave the
        // fit with no data at all: `G` and `P` stay zero, the weights come out
        // zero, and the trend is the zero prior everywhere. That is
        // indistinguishable from a genuinely flat response, so it has to fail
        // rather than return.
        if w.iter().all(|v| *v == 0.0) {
            return Err(BixverseErrors::GpAllWeightsZero);
        }
    }

    // All landmarks at one coordinate gives an exactly all-ones Gram, which is
    // SPD once jittered but whose "trend" is a single number repeated.
    let first = landmarks[0];
    if landmarks.iter().all(|v| *v == first) {
        return Err(BixverseErrors::GpDegenerateLandmarks);
    }

    Ok(())
}

/// Factorise the landmark Gram, escalating the jitter until it succeeds.
///
/// ### Params
///
/// * `k_uu` - The jitter-free landmark Gram, left untouched.
/// * `params` - Hyperparameters, supplying the starting jitter and the retries.
///
/// ### Returns
///
/// The Cholesky factorisation and the jitter it succeeded at, or
/// [BixverseErrors::GpLandmarkCholeskyFailed] once the ladder is exhausted.
#[allow(clippy::type_complexity)]
fn factorise_landmarks(
    k_uu: &Mat<f64>,
    params: &LandmarkGpParams,
) -> Result<(faer::linalg::solvers::Llt<f64>, f64), BixverseErrors> {
    let m = k_uu.nrows();
    let mut jitter = params.jitter;
    // Tracked separately so the error reports the largest value actually
    // attempted rather than the next rung the loop was about to try.
    let mut last_tried = jitter;

    for _ in 0..=params.max_jitter_retries {
        let mut kp = k_uu.clone();
        for i in 0..m {
            kp[(i, i)] += jitter;
        }
        if let Ok(factorisation) = kp.llt(Side::Lower) {
            return Ok((factorisation, jitter));
        }
        last_tried = jitter;
        // A zero starting jitter has nothing to escalate from.
        jitter = if jitter == 0.0 { 1e-12 } else { jitter * 10.0 };
    }

    Err(BixverseErrors::GpLandmarkCholeskyFailed {
        jitter: last_tried,
        n_landmarks: m,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Evenly spaced points on `[lo, hi]`, both endpoints pinned.
    fn grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        if n == 1 {
            return vec![lo];
        }
        let step = (hi - lo) / (n - 1) as f64;
        let mut out: Vec<f64> = (0..n).map(|i| lo + step * i as f64).collect();
        out[n - 1] = hi;
        out
    }

    /// With the landmarks sitting on the training points and effectively no
    /// noise, the posterior mean has to pass through the data.
    #[test]
    fn test_landmark_gp_interpolates_at_tiny_sigma() {
        let x: Vec<f64> = grid(0.0, 1.0, 25);
        let y = Mat::<f64>::from_fn(x.len(), 1, |i, _| (3.0 * x[i]).sin());

        let params = LandmarkGpParams {
            sigma: 1e-4,
            jitter: 1e-10,
            ..Default::default()
        };
        let fit = fit_landmark_gp(&x, y.as_ref(), &x, None, &params).unwrap();
        let pred: Mat<f64> = fit.predict_on_landmarks();

        for i in 0..x.len() {
            assert_relative_eq!(pred[(i, 0)], y[(i, 0)], epsilon = 1e-3);
        }
    }

    /// A smooth function sampled densely must be recovered on a finer grid,
    /// with a length scale short enough that the prior does not dominate.
    #[test]
    fn test_landmark_gp_recovers_a_smooth_function() {
        let x: Vec<f64> = grid(0.0, 1.0, 200);
        let f = |t: f64| (2.0 * std::f64::consts::PI * t).sin();
        let y = Mat::<f64>::from_fn(x.len(), 1, |i, _| f(x[i]));

        let landmarks = grid(0.0, 1.0, 40);
        let params = LandmarkGpParams {
            length_scale: 0.15,
            sigma: 0.05,
            ..Default::default()
        };
        let fit = fit_landmark_gp(&x, y.as_ref(), &landmarks, None, &params).unwrap();
        let pred: Mat<f64> = fit.predict_on_landmarks();

        for (i, &t) in landmarks.iter().enumerate() {
            assert!(
                (pred[(i, 0)] - f(t)).abs() < 0.05,
                "landmark {} at t={}: got {}, want {}",
                i,
                t,
                pred[(i, 0)],
                f(t)
            );
        }
    }

    /// Chunking is a memory strategy, not a numerical one, so a chunk size
    /// below the row count must agree with one above it.
    #[test]
    fn test_landmark_gp_chunking_is_invariant() {
        let x: Vec<f64> = grid(-1.0, 2.0, 137);
        let y = Mat::<f64>::from_fn(x.len(), 3, |i, j| (x[i] * (j + 1) as f64).cos());
        let landmarks = grid(-1.0, 2.0, 20);

        let base = LandmarkGpParams {
            length_scale: 0.4,
            sigma: 0.2,
            ..Default::default()
        };

        let one_shot = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams {
                chunk_size: 4096,
                ..base
            },
        )
        .unwrap();
        let chunked = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams {
                chunk_size: 7,
                ..base
            },
        )
        .unwrap();

        for j in 0..3 {
            for i in 0..landmarks.len() {
                assert_relative_eq!(
                    one_shot.weights[(i, j)],
                    chunked.weights[(i, j)],
                    epsilon = 1e-9,
                    max_relative = 1e-7
                );
            }
        }
    }

    /// Uniform weights are the unweighted problem, so the two must coincide.
    #[test]
    fn test_landmark_gp_uniform_weights_match_unweighted() {
        let x: Vec<f64> = grid(0.0, 1.0, 61);
        let y = Mat::<f64>::from_fn(x.len(), 2, |i, j| x[i].powi(j as i32 + 1));
        let landmarks = grid(0.0, 1.0, 15);
        let params = LandmarkGpParams {
            length_scale: 0.3,
            sigma: 0.1,
            ..Default::default()
        };

        let plain = fit_landmark_gp(&x, y.as_ref(), &landmarks, None, &params).unwrap();
        let weighted = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            Some(&vec![1.0; x.len()]),
            &params,
        )
        .unwrap();

        for j in 0..2 {
            for i in 0..landmarks.len() {
                assert_relative_eq!(
                    plain.weights[(i, j)],
                    weighted.weights[(i, j)],
                    epsilon = 1e-10,
                    max_relative = 1e-9
                );
            }
        }
    }

    /// Pinned against mellon 1.7.1, the estimator this is a port of.
    ///
    /// Every other test here is self-consistency or qualitative recovery, so a
    /// uniformly-wrong-but-coherent implementation would sail through them.
    /// These numbers came out of
    /// `mellon.FunctionEstimator(sigma=…, ls=…, landmarks=grid).fit_predict(x, y, grid)`
    /// and are reproducible bit-for-bit there: `mu` defaults to zero, the
    /// supplied `ls` is honoured rather than re-derived from the data, and
    /// nothing in the path is seeded.
    ///
    /// Exact agreement is not available and not a defect. mellon adds `1e-12`
    /// to the *squared* distance before its `sqrt`, so its kernel diagonal is
    /// `0.999999999999167` where ours is exactly one. That perturbation is
    /// seven orders of magnitude below the `1e-6` jitter both sides load onto
    /// the diagonal, which is why `1e-8` is a generous tolerance rather than a
    /// fitted one.
    ///
    /// Orientation is grid-by-genes, matching `fit_predict`. (mellon's
    /// `multi_fit_predict` returns the transpose; Palantir transposes again for
    /// its AnnData `varm`, which is why the reference's stored trends are
    /// genes-by-grid.)
    #[test]
    fn test_landmark_gp_matches_mellon_reference() {
        let x: Vec<f64> = grid(0.0, 1.0, 60);
        let landmarks = grid(0.0, 1.0, 12);
        let y = Mat::<f64>::from_fn(60, 2, |i, j| {
            if j == 0 {
                (3.0 * x[i]).sin()
            } else {
                (2.0 * x[i]).cos()
            }
        });

        // mellon.FunctionEstimator(sigma=1.0, ls=1.0, landmarks=grid)
        let want_default: [[f64; 2]; 12] = [
            [0.3539910533495122, 0.9557370626602998],
            [0.45860317497407643, 0.9227324648238153],
            [0.5635334042170455, 0.8661092108664761],
            [0.6594165102705661, 0.7854533705009715],
            [0.7360541752407392, 0.6822244585700202],
            [0.784678516820574, 0.5598666389154093],
            [0.7999122445616568, 0.423657703757],
            [0.7809292078791386, 0.28029708866765085],
            [0.7315603203894544, 0.13725078488221495],
            [0.6593248125430881, 0.0018968648884448041],
            [0.5736043429117991, -0.11942927668599138],
            [0.48342829487977995, -0.22231117542051806],
        ];

        // mellon.FunctionEstimator(sigma=0.2, ls=0.3, landmarks=grid)
        let want_tight: [[f64; 2]; 12] = [
            [0.0234549590946689, 0.9780991676470202],
            [0.26317299032134295, 0.9868910259026406],
            [0.5177675908376118, 0.9339767189083742],
            [0.7295837169477231, 0.8532715614628628],
            [0.8856831320496343, 0.7461492453229132],
            [0.9771156022101442, 0.6138245434522089],
            [0.9963285483204488, 0.46106437022572266],
            [0.9418762515598207, 0.29328314035705316],
            [0.8185313482486412, 0.11657557620418924],
            [0.6331241780975282, -0.06576534988479371],
            [0.396731427365446, -0.24998402840753614],
            [0.16164981869415487, -0.3930057188672961],
        ];

        for (ls, sigma, want) in [(1.0_f64, 1.0_f64, &want_default), (0.3, 0.2, &want_tight)] {
            let fit = fit_landmark_gp(
                &x,
                y.as_ref(),
                &landmarks,
                None,
                &LandmarkGpParams {
                    length_scale: ls,
                    sigma,
                    ..Default::default()
                },
            )
            .unwrap();
            let got: Mat<f64> = fit.predict_on_landmarks();

            assert_eq!(got.nrows(), 12);
            assert_eq!(got.ncols(), 2);
            for (i, row) in want.iter().enumerate() {
                for (j, expected) in row.iter().enumerate() {
                    assert_relative_eq!(got[(i, j)], expected, epsilon = 1e-8, max_relative = 1e-8);
                }
            }
        }
    }

    /// An integer weight must equal repeating the observation that many times.
    ///
    /// This is the test that pins `sqrt(w)` rather than `w`. Every weight is at
    /// least two, so the two scalings are genuinely different (unlike weights in
    /// `{0, 1}`, where `sqrt(w) == w` and any of them passes), and the
    /// duplicated fit is an oracle built without touching the weighting code at
    /// all.
    #[test]
    fn test_landmark_gp_integer_weights_match_duplicated_observations() {
        let x: Vec<f64> = grid(0.0, 1.0, 17);
        let y = Mat::<f64>::from_fn(x.len(), 2, |i, j| (x[i] * (j + 2) as f64).sin());
        let landmarks = grid(0.0, 1.0, 9);
        let params = LandmarkGpParams {
            length_scale: 0.35,
            sigma: 0.25,
            ..Default::default()
        };

        // Deterministic multiplicities in 2..=4.
        let counts: Vec<usize> = (0..x.len()).map(|i| 2 + i % 3).collect();
        let w: Vec<f64> = counts.iter().map(|&c| c as f64).collect();

        let weighted = fit_landmark_gp(&x, y.as_ref(), &landmarks, Some(&w), &params).unwrap();

        // The oracle: physically repeat each observation `counts[i]` times.
        let mut x_rep: Vec<f64> = Vec::new();
        let mut rows: Vec<usize> = Vec::new();
        for (i, &c) in counts.iter().enumerate() {
            for _ in 0..c {
                x_rep.push(x[i]);
                rows.push(i);
            }
        }
        let y_rep = Mat::<f64>::from_fn(rows.len(), 2, |i, j| y[(rows[i], j)]);
        let duplicated =
            fit_landmark_gp(&x_rep, y_rep.as_ref(), &landmarks, None, &params).unwrap();

        for j in 0..2 {
            for i in 0..landmarks.len() {
                assert_relative_eq!(
                    weighted.weights[(i, j)],
                    duplicated.weights[(i, j)],
                    epsilon = 1e-9,
                    max_relative = 1e-7
                );
            }
        }
    }

    /// Scaling every weight by `c` is the same as scaling `sigma^2` by `c`.
    ///
    /// `B = A W A^T / sigma^2 + I` and `P = A W r / sigma^2` are both invariant
    /// under `(W, sigma^2) -> (cW, c sigma^2)`. Uses genuinely fractional
    /// weights, which is the regime `BranchWeighting::FateProbability` actually
    /// feeds in and which the `{0, 1}` tests never reach. The invariant fails if
    /// the scaling is `w` rather than `sqrt(w)`, since `A W^2 A^T` picks up
    /// `c^2` against the denominator's `c`.
    #[test]
    fn test_landmark_gp_weight_scaling_trades_against_sigma() {
        let x: Vec<f64> = grid(0.0, 1.0, 41);
        let y = Mat::<f64>::from_fn(x.len(), 2, |i, j| (x[i] + j as f64).exp());
        let landmarks = grid(0.0, 1.0, 11);

        // Fractional, deterministic, spanning the open unit interval.
        let w: Vec<f64> = (0..x.len())
            .map(|i| 0.05 + 0.9 * ((i * 7 % 13) as f64 / 13.0))
            .collect();
        let c = 4.0_f64;
        let w_scaled: Vec<f64> = w.iter().map(|v| v * c).collect();

        let base = LandmarkGpParams {
            length_scale: 0.3,
            sigma: 0.2,
            ..Default::default()
        };
        let compensated = LandmarkGpParams {
            sigma: base.sigma * c.sqrt(),
            ..base
        };

        let plain = fit_landmark_gp(&x, y.as_ref(), &landmarks, Some(&w), &base).unwrap();
        let scaled =
            fit_landmark_gp(&x, y.as_ref(), &landmarks, Some(&w_scaled), &compensated).unwrap();

        for j in 0..2 {
            for i in 0..landmarks.len() {
                assert_relative_eq!(
                    plain.weights[(i, j)],
                    scaled.weights[(i, j)],
                    epsilon = 1e-9,
                    max_relative = 1e-7
                );
            }
        }
    }

    /// All-zero weights leave the fit with no data, so they must error rather
    /// than return the flat prior.
    #[test]
    fn test_landmark_gp_rejects_all_zero_weights() {
        let x: Vec<f64> = grid(0.0, 1.0, 10);
        let y = Mat::<f64>::from_fn(10, 1, |i, _| i as f64);
        let landmarks = grid(0.0, 1.0, 5);

        assert!(matches!(
            fit_landmark_gp(
                &x,
                y.as_ref(),
                &landmarks,
                Some(&[0.0; 10]),
                &LandmarkGpParams::default()
            ),
            Err(BixverseErrors::GpAllWeightsZero)
        ));
    }

    /// A zero weight must remove an observation entirely, so a fit that zeroes
    /// out the second half has to match one trained only on the first half.
    #[test]
    fn test_landmark_gp_zero_weights_drop_observations() {
        let x: Vec<f64> = grid(0.0, 1.0, 40);
        let y = Mat::<f64>::from_fn(x.len(), 1, |i, _| (4.0 * x[i]).sin());
        let landmarks = grid(0.0, 1.0, 12);
        let params = LandmarkGpParams {
            length_scale: 0.3,
            sigma: 0.2,
            ..Default::default()
        };

        let mut w = vec![1.0; x.len()];
        w[20..].fill(0.0);
        let masked = fit_landmark_gp(&x, y.as_ref(), &landmarks, Some(&w), &params).unwrap();

        let x_head = x[..20].to_vec();
        let y_head = Mat::<f64>::from_fn(20, 1, |i, _| y[(i, 0)]);
        let head = fit_landmark_gp(&x_head, y_head.as_ref(), &landmarks, None, &params).unwrap();

        for i in 0..landmarks.len() {
            assert_relative_eq!(
                masked.weights[(i, 0)],
                head.weights[(i, 0)],
                epsilon = 1e-9,
                max_relative = 1e-7
            );
        }
    }

    /// Fewer training points than landmarks leaves `A A^T` rank deficient. The
    /// `+ I` keeps it factorisable, so this must return a finite prior-dominated
    /// curve rather than failing. Palantir relies on it: it builds a 500-point
    /// grid regardless of how many cells a branch has.
    #[test]
    fn test_landmark_gp_handles_fewer_points_than_landmarks() {
        let x: Vec<f64> = grid(0.0, 1.0, 5);
        let y = Mat::<f64>::from_fn(5, 2, |i, j| (i + j) as f64);
        let landmarks = grid(0.0, 1.0, 50);

        let fit = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams::default(),
        )
        .unwrap();
        let pred: Mat<f64> = fit.predict_on_landmarks();

        for j in 0..2 {
            for i in 0..50 {
                assert!(pred[(i, j)].is_finite());
            }
        }
    }

    /// Predicting at the landmarks through the generic path must agree with the
    /// cached-kernel shortcut.
    #[test]
    fn test_predict_on_landmarks_matches_generic_predict() {
        let x: Vec<f64> = grid(0.0, 1.0, 50);
        let y = Mat::<f64>::from_fn(50, 2, |i, j| (x[i] * (j + 2) as f64).exp());
        let landmarks = grid(0.0, 1.0, 17);

        let fit = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams::default(),
        )
        .unwrap();

        let cached: Mat<f64> = fit.predict_on_landmarks();
        let generic: Mat<f64> = fit.predict(&landmarks);

        for j in 0..2 {
            for i in 0..17 {
                assert_relative_eq!(
                    cached[(i, j)],
                    generic[(i, j)],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }

    /// An `f32` response block must give an `f32` answer that tracks the `f64`
    /// one; the fit itself runs in `f64` either way.
    #[test]
    fn test_landmark_gp_accepts_f32_responses() {
        let x: Vec<f64> = grid(0.0, 1.0, 60);
        let y32 = Mat::<f32>::from_fn(60, 1, |i, _| (2.0 * x[i] as f32).sin());
        let y64 = Mat::<f64>::from_fn(60, 1, |i, _| y32[(i, 0)] as f64);
        let landmarks = grid(0.0, 1.0, 20);
        let params = LandmarkGpParams {
            length_scale: 0.3,
            sigma: 0.1,
            ..Default::default()
        };

        let fit32 = fit_landmark_gp(&x, y32.as_ref(), &landmarks, None, &params).unwrap();
        let fit64 = fit_landmark_gp(&x, y64.as_ref(), &landmarks, None, &params).unwrap();

        let pred32: Mat<f32> = fit32.predict_on_landmarks();
        let pred64: Mat<f64> = fit64.predict_on_landmarks();

        for i in 0..20 {
            assert_relative_eq!(pred32[(i, 0)] as f64, pred64[(i, 0)], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_landmark_gp_rejects_bad_inputs() {
        let x: Vec<f64> = grid(0.0, 1.0, 10);
        let y = Mat::<f64>::from_fn(10, 1, |i, _| i as f64);
        let landmarks = grid(0.0, 1.0, 5);
        let ok = LandmarkGpParams::default();

        // Dimension mismatch.
        assert!(matches!(
            fit_landmark_gp(&x[..5], y.as_ref(), &landmarks, None, &ok),
            Err(BixverseErrors::GpDimensionMismatch { n_x: 5, n_y: 10 })
        ));

        // Empty inputs.
        assert!(matches!(
            fit_landmark_gp::<f64>(&[], Mat::<f64>::zeros(0, 1).as_ref(), &landmarks, None, &ok),
            Err(BixverseErrors::GpEmptyInput { .. })
        ));
        assert!(matches!(
            fit_landmark_gp(&x, y.as_ref(), &landmarks[..1], None, &ok),
            Err(BixverseErrors::GpEmptyInput { .. })
        ));

        // Non-positive sigma, which mellon's own default would hit.
        assert!(matches!(
            fit_landmark_gp(
                &x,
                y.as_ref(),
                &landmarks,
                None,
                &LandmarkGpParams { sigma: 0.0, ..ok }
            ),
            Err(BixverseErrors::GpInvalidHyperparameter { name: "sigma", .. })
        ));
        assert!(matches!(
            fit_landmark_gp(
                &x,
                y.as_ref(),
                &landmarks,
                None,
                &LandmarkGpParams {
                    length_scale: -1.0,
                    ..ok
                }
            ),
            Err(BixverseErrors::GpInvalidHyperparameter {
                name: "length_scale",
                ..
            })
        ));

        // Non-finite data.
        let mut x_nan = x.clone();
        x_nan[3] = f64::NAN;
        assert!(matches!(
            fit_landmark_gp(&x_nan, y.as_ref(), &landmarks, None, &ok),
            Err(BixverseErrors::GpNonFiniteInput { name: "x" })
        ));

        let mut y_inf = y.clone();
        y_inf[(2, 0)] = f64::INFINITY;
        assert!(matches!(
            fit_landmark_gp(&x, y_inf.as_ref(), &landmarks, None, &ok),
            Err(BixverseErrors::GpNonFiniteInput { name: "y" })
        ));

        // Degenerate landmarks.
        assert!(matches!(
            fit_landmark_gp(&x, y.as_ref(), &[0.5; 6], None, &ok),
            Err(BixverseErrors::GpDegenerateLandmarks)
        ));

        // Weight length mismatch.
        assert!(matches!(
            fit_landmark_gp(&x, y.as_ref(), &landmarks, Some(&[1.0; 3]), &ok),
            Err(BixverseErrors::GpDimensionMismatch { .. })
        ));
    }

    /// The jitter ladder must report what it actually used, so an
    /// over-regularised fit is visible rather than silent.
    #[test]
    fn test_jitter_used_is_reported() {
        let x: Vec<f64> = grid(0.0, 1.0, 20);
        let y = Mat::<f64>::from_fn(20, 1, |i, _| i as f64);
        let landmarks = grid(0.0, 1.0, 8);

        let fit = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams {
                jitter: 1e-3,
                ..Default::default()
            },
        )
        .unwrap();

        // Well conditioned at this size, so the first rung succeeds.
        assert_relative_eq!(fit.jitter_used, 1e-3, epsilon = 1e-15);
    }

    /// The jitter ladder must actually climb, and the failure must report the
    /// largest rung that was tried rather than the next one it was about to.
    ///
    /// A repeated landmark coordinate gives `k(u, u)` two identical rows, so it
    /// is exactly singular and the factorisation fails at zero jitter. Loading
    /// the diagonal separates them again: the offending two-by-two becomes
    /// `[[1+j, 1], [1, 1+j]]` with determinant `2j + j²`, positive for any
    /// `j > 0`. So the same input fails without a retry budget and succeeds
    /// with one, which is exactly the pair of behaviours the ladder exists for.
    #[test]
    fn test_jitter_ladder_escalates_and_reports_the_last_rung() {
        let landmarks = vec![0.0_f64, 0.25, 0.25, 0.5, 0.75, 1.0];
        let x: Vec<f64> = grid(0.0, 1.0, 30);
        let y = Mat::<f64>::from_fn(30, 1, |i, _| x[i].sin());

        // No retries and no jitter: must fail, naming the value it did try.
        let err = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams {
                jitter: 0.0,
                max_jitter_retries: 0,
                ..Default::default()
            },
        )
        .unwrap_err();

        match err {
            BixverseErrors::GpLandmarkCholeskyFailed {
                jitter,
                n_landmarks,
            } => {
                assert_eq!(n_landmarks, 6);
                // The only attempt was at the requested 0.0. Reporting the next
                // rung here would advertise a jitter that never ran.
                assert_eq!(jitter, 0.0);
            }
            other => panic!("expected GpLandmarkCholeskyFailed, got {:?}", other),
        }

        // Two rungs of budget: must fail again, and must report the *second*
        // rung rather than the third. This is the off-by-one guard.
        let err = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams {
                jitter: f64::MIN_POSITIVE,
                max_jitter_retries: 1,
                ..Default::default()
            },
        )
        .unwrap_err();

        match err {
            BixverseErrors::GpLandmarkCholeskyFailed { jitter, .. } => {
                assert_eq!(
                    jitter,
                    f64::MIN_POSITIVE * 10.0,
                    "reported jitter is not the last rung actually attempted"
                );
            }
            other => panic!("expected GpLandmarkCholeskyFailed, got {:?}", other),
        }

        // With a real budget the ladder climbs off zero and succeeds.
        let fit = fit_landmark_gp(
            &x,
            y.as_ref(),
            &landmarks,
            None,
            &LandmarkGpParams {
                jitter: 0.0,
                max_jitter_retries: 12,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            fit.jitter_used > 0.0,
            "the ladder never escalated off the requested jitter"
        );
        assert!(
            fit.weights
                .col_iter()
                .all(|c| c.iter().all(|v| v.is_finite())),
            "escalation produced a non-finite fit"
        );
    }
}
