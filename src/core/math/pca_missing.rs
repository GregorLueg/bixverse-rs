//! Probabilistic (PPCA) and Bayesian (BPCA) PCA on data with missing values.
//! Ports of `ppca()` and `bpca()` from the R package pcaMethods: both fit the
//! principal subspace on the observed entries only and return the imputed
//! matrix alongside scores and loadings.
//!
//! Inputs are observations x variables with `NaN` marking a missing value.
//! The fit runs in f64 regardless of `T`: the PPCA residual comes from a trace
//! identity and the BPCA noise precision from a difference of traces, and both
//! cancel badly in f32 near convergence.

use crate::utils::gemm as dense;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Accum, Mat, MatRef, Side};
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::faer_parallelism;

////////////
// Consts //
////////////

/// Standard deviation floor from pcaMethods' `prep()`. Columns below it are
/// left unscaled rather than blown up.
const SCALE_EPS: f64 = 1e-12;

/// PPCA runs at least this many EM iterations before the relative change in
/// the objective may stop it (pcaMethods' `count > 5`).
const PPCA_MIN_ITER: usize = 5;

/// PPCA progress is printed every this many iterations at
/// [Verbosity::Normal]; [Verbosity::Detailed] prints every iteration.
const PPCA_REPORT_EVERY: usize = 25;

/// BPCA: shape of the Gamma prior on the ARD precisions `alpha`.
const BPCA_GALPHA0: f64 = 1e-10;

/// BPCA: scale of the Gamma prior on the ARD precisions `alpha`.
const BPCA_BALPHA0: f64 = 1.0;

/// BPCA: prior precision on the column means.
const BPCA_GMU0: f64 = 0.001;

/// BPCA: shape of the Gamma prior on the noise precision `tau`.
const BPCA_GTAU0: f64 = 1e-10;

/// BPCA: scale of the Gamma prior on the noise precision `tau`.
const BPCA_BTAU0: f64 = 1.0;

/// BPCA: lower clamp on the initial noise precision.
const BPCA_TAU_MIN: f64 = 1e-10;

/// BPCA: upper clamp on the initial noise precision.
const BPCA_TAU_MAX: f64 = 1e10;

/// BPCA: reference `tau` for the first convergence check.
const BPCA_TAU_START: f64 = 1000.0;

/// BPCA: convergence (change in `log10(tau)`) is only checked every this many
/// steps, as in pcaMethods.
const BPCA_CHECK_EVERY: usize = 10;

////////////
// Params //
////////////

/// Parameters for probabilistic PCA.
#[derive(Clone, Copy, Debug)]
pub struct PpcaParams {
    /// Number of principal components
    pub n_pcs: usize,
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Relative change in the objective below which EM stops
    pub tol: f64,
    /// Seed for the random initial loadings
    pub seed: u64,
    /// Subtract the observed column means first
    pub centre: bool,
    /// Divide by the observed column SD first (pcaMethods' `"uv"`)
    pub scale: bool,
}

impl PpcaParams {
    /// Generate new PPCA parameters.
    ///
    /// ### Params
    ///
    /// * `n_pcs` - Number of principal components
    /// * `max_iter` - Maximum number of EM iterations
    /// * `tol` - Relative change in the objective below which EM stops
    /// * `seed` - Seed for the random initial loadings
    /// * `centre` - Subtract the observed column means first
    /// * `scale` - Divide by the observed column SD first
    ///
    /// ### Returns
    ///
    /// The [PpcaParams].
    pub fn new(
        n_pcs: usize,
        max_iter: usize,
        tol: f64,
        seed: u64,
        centre: bool,
        scale: bool,
    ) -> Self {
        Self {
            n_pcs,
            max_iter,
            tol,
            seed,
            centre,
            scale,
        }
    }
}

impl Default for PpcaParams {
    /// pcaMethods' defaults.
    fn default() -> Self {
        Self {
            n_pcs: 2,
            max_iter: 1000,
            tol: 1e-5,
            seed: 42,
            centre: true,
            scale: false,
        }
    }
}

/// Parameters for Bayesian PCA.
#[derive(Clone, Copy, Debug)]
pub struct BpcaParams {
    /// Number of principal components
    pub n_pcs: usize,
    /// Maximum number of variational steps
    pub max_iter: usize,
    /// Change in `log10(tau)` over ten steps below which the fit stops
    pub tol: f64,
    /// Subtract the observed column means first
    pub centre: bool,
    /// Divide by the observed column SD first (pcaMethods' `"uv"`)
    pub scale: bool,
}

impl BpcaParams {
    /// Generate new BPCA parameters.
    ///
    /// ### Params
    ///
    /// * `n_pcs` - Number of principal components
    /// * `max_iter` - Maximum number of variational steps
    /// * `tol` - Change in `log10(tau)` over ten steps below which the fit
    ///   stops
    /// * `centre` - Subtract the observed column means first
    /// * `scale` - Divide by the observed column SD first
    ///
    /// ### Returns
    ///
    /// The [BpcaParams].
    pub fn new(n_pcs: usize, max_iter: usize, tol: f64, centre: bool, scale: bool) -> Self {
        Self {
            n_pcs,
            max_iter,
            tol,
            centre,
            scale,
        }
    }
}

impl Default for BpcaParams {
    /// pcaMethods' defaults.
    fn default() -> Self {
        Self {
            n_pcs: 2,
            max_iter: 100,
            tol: 1e-4,
            centre: true,
            scale: false,
        }
    }
}

////////////////
// Structures //
////////////////

/// Result of [ppca] or [bpca].
#[derive(Clone, Debug)]
pub struct MissingPcaResults<T> {
    /// Scores, N x n_pcs
    pub scores: Mat<T>,
    /// Loadings, D x n_pcs. Orthonormal for PPCA, not for BPCA
    pub loadings: Mat<T>,
    /// Cumulative R^2 per component. PPCA scores it on the completed matrix,
    /// BPCA on the observed entries only, as pcaMethods does
    pub r2_cum: Vec<T>,
    /// Column centres that were subtracted (zeros when not centring)
    pub centre: Vec<T>,
    /// Column scales that were divided out (ones when not scaling)
    pub scale: Vec<T>,
    /// Input with missing entries replaced by `scores %*% t(loadings)` on the
    /// original scale. Observed entries are untouched
    pub completed: Mat<T>,
    /// Residual variance outside the subspace: `ss` for PPCA, `1 / tau` for
    /// BPCA
    pub noise_var: T,
    /// Iterations run
    pub n_iter: usize,
    /// Whether the stopping criterion was met before `max_iter`
    pub converged: bool,
}

/// Centred and scaled input in f64 with its missingness mask.
struct PreppedData {
    /// Row-major N x D values; missing entries hold 0
    y: Vec<f64>,
    /// Row-major N x D mask, true where the input was missing
    missing: Vec<bool>,
    /// Number of missing entries per row
    n_miss_row: Vec<usize>,
    /// Number of observations
    n: usize,
    /// Number of variables
    d: usize,
    /// Total number of missing entries
    n_missing: usize,
    /// Column centres
    centre: Vec<f64>,
    /// Column scales
    scale: Vec<f64>,
}

/// Raw output of one of the fits, before back-transformation.
struct Fit {
    /// Scores, N x k
    scores: Mat<f64>,
    /// Loadings, D x k
    loadings: Mat<f64>,
    /// Cumulative R^2 per component
    r2_cum: Vec<f64>,
    /// Residual variance
    noise_var: f64,
    /// Iterations run
    n_iter: usize,
    /// Whether the stopping criterion was met
    converged: bool,
}

/////////////
// Helpers //
/////////////

/// Validate the input and apply pcaMethods' `prep()`.
///
/// Centres use the observed column means; scales the observed column SD
/// (`sd(na.rm = TRUE)`), with SDs under [SCALE_EPS] or from fewer than two
/// observations left at one.
///
/// ### Params
///
/// * `y` - Input, N x D, `NaN` for missing
/// * `n_pcs` - Requested number of components
/// * `centre` - Whether to centre
/// * `scale` - Whether to scale to unit variance
///
/// ### Returns
///
/// The [PreppedData], or an error on non-finite input, all-missing rows or
/// columns, fewer than two rows, or too many components.
fn prep_data<T>(
    y: MatRef<T>,
    n_pcs: usize,
    centre: bool,
    scale: bool,
) -> Result<PreppedData, BixverseErrors>
where
    T: BixverseFloat,
{
    let (n, d) = y.shape();
    if n_pcs == 0 {
        return Err(BixverseErrors::MustBePositive("n_pcs".to_string()));
    }
    if n < 2 {
        return Err(BixverseErrors::InvalidArgument(
            "PCA with missing values needs at least two rows".to_string(),
        ));
    }
    let max = n.min(d);
    if n_pcs > max {
        return Err(BixverseErrors::PcaTooManyComponents { n_pcs, max });
    }

    let col_stats: Vec<(f64, f64, usize, bool)> = y
        .par_col_iter()
        .map(|col| {
            let mut sum = 0.0;
            let mut n_obs = 0usize;
            let mut finite = true;
            for v in col.iter() {
                let v = v.to_f64().unwrap();
                if v.is_nan() {
                    continue;
                }
                finite &= v.is_finite();
                sum += v;
                n_obs += 1;
            }
            let mean = sum / n_obs as f64;
            let ss: f64 = col
                .iter()
                .map(|v| v.to_f64().unwrap())
                .filter(|v| !v.is_nan())
                .map(|v| (v - mean) * (v - mean))
                .sum();
            let sd = (ss / (n_obs as f64 - 1.0)).sqrt();
            (mean, sd, n_obs, finite)
        })
        .collect();

    if col_stats.iter().any(|s| !s.3) {
        return Err(BixverseErrors::InvalidArgument(
            "PCA input contains infinite values".to_string(),
        ));
    }
    if let Some(j) = col_stats.iter().position(|s| s.2 == 0) {
        return Err(BixverseErrors::PcaAllMissing {
            axis: "column",
            index: j,
        });
    }

    let centre: Vec<f64> = col_stats
        .iter()
        .map(|s| if centre { s.0 } else { 0.0 })
        .collect();
    let scale: Vec<f64> = col_stats
        .iter()
        .map(|s| {
            if scale && s.2 > 1 && s.1 >= SCALE_EPS {
                s.1
            } else {
                1.0
            }
        })
        .collect();

    let mut y_rm = vec![0.0; n * d];
    let mut missing = vec![false; n * d];
    let n_miss_row: Vec<usize> = y_rm
        .par_chunks_mut(d)
        .zip(missing.par_chunks_mut(d))
        .enumerate()
        .map(|(i, (row, m_row))| {
            let mut nm = 0;
            for j in 0..d {
                let v = y[(i, j)].to_f64().unwrap();
                if v.is_nan() {
                    m_row[j] = true;
                    nm += 1;
                } else {
                    row[j] = (v - centre[j]) / scale[j];
                }
            }
            nm
        })
        .collect();

    if let Some(i) = n_miss_row.iter().position(|&nm| nm == d) {
        return Err(BixverseErrors::PcaAllMissing {
            axis: "row",
            index: i,
        });
    }
    let n_missing = n_miss_row.iter().sum();

    Ok(PreppedData {
        y: y_rm,
        missing,
        n_miss_row,
        n,
        d,
        n_missing,
        centre,
        scale,
    })
}

/// Parallel GEMM `a %*% b` into a fresh matrix.
///
/// ### Params
///
/// * `a` - Left factor
/// * `b` - Right factor
///
/// ### Returns
///
/// The product.
fn gemm(a: MatRef<f64>, b: MatRef<f64>) -> Mat<f64> {
    let mut out = Mat::zeros(a.nrows(), b.ncols());
    dense::gemm(out.as_mut(), Accum::Replace, a, b, 1.0, faer_parallelism());
    out
}

/// Frobenius inner product `sum(a * b)`.
///
/// ### Params
///
/// * `a` - First matrix
/// * `b` - Second matrix, same shape
///
/// ### Returns
///
/// The sum of the elementwise product.
fn frob_dot(a: MatRef<f64>, b: MatRef<f64>) -> f64 {
    a.col_iter()
        .zip(b.col_iter())
        .map(|(ca, cb)| ca.iter().zip(cb.iter()).map(|(x, y)| x * y).sum::<f64>())
        .sum()
}

/// Inverse and log-determinant of a small symmetric positive definite matrix
/// via Cholesky.
///
/// ### Params
///
/// * `a` - SPD matrix, k x k
///
/// ### Returns
///
/// `(inverse, log(det(a)))`, or the Cholesky error if `a` is not SPD.
fn spd_inverse(a: MatRef<f64>) -> Result<(Mat<f64>, f64), BixverseErrors> {
    let llt = a.llt(Side::Lower)?;
    let logdet = 2.0
        * llt
            .L()
            .diagonal()
            .column_vector()
            .iter()
            .map(|v| v.ln())
            .sum::<f64>();
    Ok((llt.inverse(), logdet))
}

/// Cumulative R^2 of the rank-1..k reconstructions `scores %*% t(loadings)`.
///
/// ### Params
///
/// * `y` - Row-major N x D data
/// * `mask` - Optional row-major missingness mask; masked entries are skipped
///   in both the residual and the total sum of squares
/// * `d` - Number of columns
/// * `scores` - N x k
/// * `loadings` - D x k
///
/// ### Returns
///
/// `1 - RSS_i / TSS` for each prefix of components.
fn r2_cum(
    y: &[f64],
    mask: Option<&[bool]>,
    d: usize,
    scores: MatRef<f64>,
    loadings: MatRef<f64>,
) -> Vec<f64> {
    let k = scores.ncols();
    let w_rows = loadings.transpose().to_owned();

    // per-row partials summed sequentially: a rayon fold/reduce associates in
    // whatever order work stealing produces, so the result drifts run to run
    let rows: Vec<(f64, Vec<f64>)> = y
        .par_chunks(d)
        .enumerate()
        .map(|(i, row)| {
            let (mut tss, mut rss) = (0.0, vec![0.0; k]);
            for (j, &v) in row.iter().enumerate() {
                if mask.is_some_and(|m| m[i * d + j]) {
                    continue;
                }
                tss += v * v;
                let w = w_rows.col(j);
                let mut rec = 0.0;
                for l in 0..k {
                    rec += scores[(i, l)] * w[l];
                    let r = v - rec;
                    rss[l] += r * r;
                }
            }
            (tss, rss)
        })
        .collect();
    let (mut tss, mut rss) = (0.0, vec![0.0; k]);
    for (t, r) in rows {
        tss += t;
        rss.iter_mut().zip(r).for_each(|(a, b)| *a += b);
    }

    rss.iter().map(|r| 1.0 - r / tss).collect()
}

/// Assemble the public result: back-transform and fill the missing entries.
///
/// ### Params
///
/// * `y` - The original input
/// * `prep` - The prepped data
/// * `fit` - The raw fit
///
/// ### Returns
///
/// The [MissingPcaResults] in `T`.
fn finish<T>(y: MatRef<T>, prep: &PreppedData, fit: Fit) -> MissingPcaResults<T>
where
    T: BixverseFloat,
{
    let (n, d) = (prep.n, prep.d);
    let k = fit.scores.ncols();
    let w_rows = fit.loadings.transpose().to_owned();
    let to_t = |v: f64| T::from_f64(v).unwrap();

    let mut completed_rm = vec![0.0; n * d];
    completed_rm
        .par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..d {
                row[j] = if prep.missing[i * d + j] {
                    let w = w_rows.col(j);
                    let rec: f64 = (0..k).map(|l| fit.scores[(i, l)] * w[l]).sum();
                    rec * prep.scale[j] + prep.centre[j]
                } else {
                    y[(i, j)].to_f64().unwrap()
                };
            }
        });

    MissingPcaResults {
        scores: Mat::from_fn(n, k, |i, j| to_t(fit.scores[(i, j)])),
        loadings: Mat::from_fn(d, k, |i, j| to_t(fit.loadings[(i, j)])),
        r2_cum: fit.r2_cum.into_iter().map(to_t).collect(),
        centre: prep.centre.iter().map(|&v| to_t(v)).collect(),
        scale: prep.scale.iter().map(|&v| to_t(v)).collect(),
        completed: Mat::from_fn(n, d, |i, j| to_t(completed_rm[i * d + j])),
        noise_var: to_t(fit.noise_var),
        n_iter: fit.n_iter,
        converged: fit.converged,
    }
}

//////////
// PPCA //
//////////

/// Overwrite the missing entries of `y` with `x %*% t(c)`.
///
/// ### Params
///
/// * `y` - Row-major N x D data, modified in place
/// * `missing` - Row-major missingness mask
/// * `d` - Number of columns
/// * `x` - Scores, N x k
/// * `c` - Loadings, D x k
///
/// ### Returns
///
/// The sum of squares of the written values.
fn fill_missing(y: &mut [f64], missing: &[bool], d: usize, x: MatRef<f64>, c: MatRef<f64>) -> f64 {
    let k = x.ncols();
    let c_rows = c.transpose().to_owned();
    y.par_chunks_mut(d)
        .zip(missing.par_chunks(d))
        .enumerate()
        .map(|(i, (row, m_row))| {
            let mut sq = 0.0;
            for j in 0..d {
                if m_row[j] {
                    let cj = c_rows.col(j);
                    let v: f64 = (0..k).map(|l| x[(i, l)] * cj[l]).sum();
                    row[j] = v;
                    sq += v * v;
                }
            }
            sq
        })
        .collect::<Vec<f64>>()
        .iter()
        .sum()
}

/// PPCA EM loop on prepped data, after Porta's Matlab `ppca.m` as ported by
/// pcaMethods.
///
/// Per iteration: two N x D x k GEMMs (`Y C` and `Y^T X`), a k-dot per missing
/// entry, and k x k solves. The residual `||Y - X C^T||^2` is taken from
/// `||Y||^2 - 2 <Y^T X, C> + <X^T X, C^T C>` rather than materialised, with
/// `||Y||^2` updated over the refilled entries only.
///
/// ### Params
///
/// * `prep` - Prepped data, consumed; the missing entries are refilled in place
/// * `c` - Initial loadings, D x k
/// * `params` - PPCA parameters
/// * `verbosity` - Progress reporting
///
/// ### Returns
///
/// The raw [Fit], or a Cholesky error if a k x k system goes singular.
///
/// ### References
///
/// Roweis, NIPS, 1998; Tipping & Bishop, J R Stat Soc B, 1999
fn ppca_fit(
    prep: &mut PreppedData,
    mut c: Mat<f64>,
    params: &PpcaParams,
    verbosity: Verbosity,
) -> Result<Fit, BixverseErrors> {
    let (n, d, k) = (prep.n, prep.d, params.n_pcs);
    let (nf, df, n_miss) = (n as f64, d as f64, prep.n_missing as f64);
    let eye = Mat::<f64>::identity(k, k);

    let y_obs_sq: f64 = prep.y.iter().map(|v| v * v).sum();

    // initial X = Y C (C^T C)^-1 and residual on the observed entries. The
    // missing entries of Y are still zero, so the residual over all entries
    // minus the squared reconstruction at the missing ones is the observed
    // residual. The fill written here is exactly the one the first E-step
    // would write, since X and C do not change in between.
    let mut ctc = c.transpose() * &c;
    let yc = gemm(MatRef::from_row_major_slice(&prep.y, n, d), c.as_ref());
    let mut x = &yc * spd_inverse(ctc.as_ref())?.0;
    let xtx = x.transpose() * &x;
    let resid_all =
        y_obs_sq - 2.0 * frob_dot(yc.as_ref(), x.as_ref()) + frob_dot(xtx.as_ref(), ctc.as_ref());
    let miss_sq = if prep.n_missing > 0 {
        fill_missing(&mut prep.y, &prep.missing, d, x.as_ref(), c.as_ref())
    } else {
        0.0
    };
    let mut ss = (resid_all - miss_sq) / (nf * df - n_miss);
    let mut y_sq = y_obs_sq + miss_sq;

    let mut old = f64::INFINITY;
    let mut n_iter = 0;
    let mut converged = false;

    loop {
        let (sx, logdet_a) = spd_inverse((&eye + &ctc * (1.0 / ss)).as_ref())?;
        let logdet_sx = -logdet_a;
        let ss_old = ss;

        if prep.n_missing > 0 {
            y_sq = y_obs_sq + fill_missing(&mut prep.y, &prep.missing, d, x.as_ref(), c.as_ref());
        }
        let y_mat = MatRef::from_row_major_slice(&prep.y, n, d);

        // E-step
        x = gemm(y_mat, c.as_ref()) * (&sx * (1.0 / ss));

        // M-step
        let xtx = gemm(x.transpose(), x.as_ref());
        let ytx = gemm(y_mat.transpose(), x.as_ref());
        c = &ytx * spd_inverse((&xtx + &sx * nf).as_ref())?.0;
        ctc = c.transpose() * &c;

        let resid =
            y_sq - 2.0 * frob_dot(ytx.as_ref(), c.as_ref()) + frob_dot(xtx.as_ref(), ctc.as_ref());
        // pcaMethods sums every entry of CtC %*% Sx where the Matlab original
        // takes the trace. Kept for parity; imputation RMSE agreed to five
        // significant digits between the two on three test shapes.
        let ctc_sx = &ctc * &sx;
        let ctc_sx_sum: f64 = ctc_sx.col_iter().map(|col| col.iter().sum::<f64>()).sum();
        ss = (resid + nf * ctc_sx_sum + n_miss * ss_old) / (nf * df);

        let tr_sx: f64 = sx.diagonal().column_vector().iter().sum();
        let tr_xtx: f64 = xtx.diagonal().column_vector().iter().sum();
        let objective = nf * (df * ss.ln() + tr_sx - logdet_sx) + tr_xtx - n_miss * ss_old.ln();
        let rel_ch = (1.0 - objective / old).abs();
        old = objective;

        n_iter += 1;
        if verbosity.detailed_verbosity()
            || (verbosity.normal_verbosity() && n_iter % PPCA_REPORT_EVERY == 0)
        {
            println!(
                "PPCA: iteration {n_iter}, objective {objective:.6e}, relative change {rel_ch:.3e}"
            );
        }
        if rel_ch < params.tol && n_iter + 1 > PPCA_MIN_ITER {
            converged = true;
            break;
        }
        if n_iter + 1 > params.max_iter {
            break;
        }
    }

    if verbosity.normal_verbosity() {
        let status = if converged {
            "converged"
        } else {
            "hit max_iter"
        };
        println!("PPCA: {status} after {n_iter} iterations, noise variance {ss:.4e}");
    }

    // orthonormalise, then rotate onto the eigenvectors of the score covariance
    let y_mat = MatRef::from_row_major_slice(&prep.y, n, d);
    let c_orth = c
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?
        .U()
        .to_owned();
    let yc = gemm(y_mat, c_orth.as_ref());
    let means: Vec<f64> = yc
        .col_iter()
        .map(|col| col.iter().sum::<f64>() / nf)
        .collect();
    let yc_c = Mat::from_fn(n, k, |i, j| yc[(i, j)] - means[j]);
    let cov = (yc_c.transpose() * &yc_c) * (1.0 / (nf - 1.0));
    let eig = cov
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| BixverseErrors::FaerEigenError)?;
    // faer sorts ascending, R's eigen() descending
    let u = eig.U();
    let vecs = Mat::from_fn(k, k, |i, j| u[(i, k - 1 - j)]);
    let loadings = &c_orth * &vecs;
    let scores = gemm(y_mat, loadings.as_ref());

    let r2 = r2_cum(&prep.y, None, d, scores.as_ref(), loadings.as_ref());

    Ok(Fit {
        scores,
        loadings,
        r2_cum: r2,
        noise_var: ss,
        n_iter,
        converged,
    })
}

/// Probabilistic PCA with missing values, from a random start.
///
/// The initial loadings are standard normal draws from `params.seed`. R's
/// `ppca()` draws from its own RNG, so results match pcaMethods at
/// convergence, not iterate by iterate; use [ppca_from_init] with R's start
/// for the latter.
///
/// ### Params
///
/// * `y` - Observations x variables, `NaN` for missing
/// * `params` - PPCA parameters; defaults to [PpcaParams::default]
/// * `verbosity` - Progress reporting
///
/// ### Returns
///
/// The [MissingPcaResults], or an error on invalid input or a singular
/// k x k system.
///
/// ### References
///
/// Roweis, NIPS, 1998; Tipping & Bishop, J R Stat Soc B, 1999; Stacklies et
/// al., Bioinformatics, 2007
pub fn ppca<T>(
    y: MatRef<T>,
    params: Option<PpcaParams>,
    verbosity: Verbosity,
) -> Result<MissingPcaResults<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let params = params.unwrap_or_default();
    let mut rng = StdRng::seed_from_u64(params.seed);
    let normal = Normal::new(0.0, 1.0).expect("unit normal is valid");
    let c0 = Mat::<T>::from_fn(y.ncols(), params.n_pcs, |_, _| {
        T::from_f64(normal.sample(&mut rng)).unwrap()
    });
    ppca_from_init(y, c0.as_ref(), Some(params), verbosity)
}

/// Probabilistic PCA with missing values, from given initial loadings.
///
/// Doubles as a warm start and as the hook for iterate-level parity with
/// pcaMethods, whose start is `matrix(rnorm(D * k), D, k)` after
/// `set.seed(seed); sample(N)`.
///
/// ### Params
///
/// * `y` - Observations x variables, `NaN` for missing
/// * `c0` - Initial loadings, D x n_pcs, on the prepped scale
/// * `params` - PPCA parameters; defaults to [PpcaParams::default]
/// * `verbosity` - Progress reporting
///
/// ### Returns
///
/// The [MissingPcaResults], or an error on invalid input, a mis-shaped `c0`
/// or a singular k x k system.
pub fn ppca_from_init<T>(
    y: MatRef<T>,
    c0: MatRef<T>,
    params: Option<PpcaParams>,
    verbosity: Verbosity,
) -> Result<MissingPcaResults<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let params = params.unwrap_or_default();
    let mut prep = prep_data(y, params.n_pcs, params.centre, params.scale)?;
    if c0.shape() != (prep.d, params.n_pcs) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "initial loadings are {} x {}, expected {} x {}",
            c0.nrows(),
            c0.ncols(),
            prep.d,
            params.n_pcs
        )));
    }
    let c = Mat::from_fn(prep.d, params.n_pcs, |i, j| c0[(i, j)].to_f64().unwrap());
    let fit = ppca_fit(&mut prep, c, &params, verbosity)?;
    Ok(finish(y, &prep, fit))
}

//////////
// BPCA //
//////////

/// State of the BPCA variational fit.
struct BpcaState {
    /// Loadings (principal axes), D x k
    pa: Mat<f64>,
    /// Noise precision
    tau: f64,
    /// ARD precisions, one per component
    alpha: Vec<f64>,
    /// Posterior covariance of the loadings, k x k
    sig_w: Mat<f64>,
}

/// BPCA initialisation (pcaMethods' `BPCA_initmodel`).
///
/// R eigendecomposes the D x D covariance of the zero-filled matrix. Its top
/// k eigenpairs are the right singular vectors and `s^2 / (N - 1)` of the
/// column-centred N x D matrix, so this never forms the D x D matrix. As in R,
/// the covariance centres by the zero-filled column means while the model
/// later uses the observed means.
///
/// ### Params
///
/// * `prep` - Prepped data
/// * `k` - Number of components
///
/// ### Returns
///
/// The initial [BpcaState], or an SVD error.
fn bpca_init(prep: &PreppedData, k: usize) -> Result<BpcaState, BixverseErrors> {
    let (n, d) = (prep.n, prep.d);
    let nf = n as f64;
    let y = MatRef::from_row_major_slice(&prep.y, n, d);

    let zero_means: Vec<f64> = y
        .par_col_iter()
        .map(|col| col.iter().sum::<f64>() / nf)
        .collect();
    let yc = Mat::from_fn(n, d, |i, j| y[(i, j)] - zero_means[j]);
    let tr_cov = frob_dot(yc.as_ref(), yc.as_ref()) / (nf - 1.0);

    let svd = yc
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
    let s = svd.S().column_vector();
    let v = svd.V();
    let eig: Vec<f64> = (0..k).map(|l| s[l] * s[l] / (nf - 1.0)).collect();

    let pa = Mat::from_fn(d, k, |j, l| v[(j, l)] * eig[l].sqrt());
    let tau = (1.0 / (tr_cov - eig.iter().sum::<f64>())).clamp(BPCA_TAU_MIN, BPCA_TAU_MAX);
    let alpha = pa
        .col_iter()
        .map(|col| {
            let sq: f64 = col.iter().map(|v| v * v).sum();
            (2.0 * BPCA_GALPHA0 + d as f64) / (tau * sq + 2.0 * BPCA_GALPHA0 / BPCA_BALPHA0)
        })
        .collect();

    Ok(BpcaState {
        pa,
        tau,
        alpha,
        sig_w: Mat::identity(k, k),
    })
}

/// One BPCA variational step (pcaMethods' `BPCA_dostep`).
///
/// E-step per row in parallel: complete rows share `Rx^-1`, incomplete rows
/// solve their own `Rx - tau Wm^T Wm`, built from whichever of the missing or
/// observed loadings is shorter. The missing entries of `dy` are overwritten
/// with their conditional means, so `T = dy^T X` is one GEMM; the posterior
/// covariance term R adds into the missing rows of `T` is summed per column
/// afterwards, which needs no reduction across threads.
///
/// ### Params
///
/// * `state` - Current state, updated in place
/// * `dy` - Row-major N x D data minus the observed means; the missing entries
///   are overwritten
/// * `prep` - Prepped data (mask and row counts)
/// * `mean_sq` - Squared norm of the observed column means
///
/// ### Returns
///
/// The scores from this E-step (N x k), or a Cholesky/LU error.
///
/// ### References
///
/// Oba et al., Bioinformatics, 2003
fn bpca_step(
    state: &mut BpcaState,
    dy: &mut [f64],
    prep: &PreppedData,
    mean_sq: f64,
) -> Result<Mat<f64>, BixverseErrors> {
    let (n, d) = (prep.n, prep.d);
    let k = state.pa.ncols();
    let (nf, df) = (n as f64, d as f64);
    let tau = state.tau;
    let eye = Mat::<f64>::identity(k, k);

    let ptp = state.pa.transpose() * &state.pa;
    let rx = &eye + &ptp * tau + &state.sig_w;
    let (rx_inv, _) = spd_inverse(rx.as_ref())?;
    let base = &eye + &state.sig_w;
    // row j of pa_rows is loading row j, contiguous
    let pa_rows = state.pa.transpose().to_owned();

    let rows: Vec<(Vec<f64>, Option<Mat<f64>>, f64)> = dy
        .par_chunks_mut(d)
        .zip(prep.missing.par_chunks(d))
        .zip(prep.n_miss_row.par_iter())
        .map(|((row, m_row), &nm)| {
            let mut ex = vec![0.0; k];
            for j in 0..d {
                if !m_row[j] {
                    let w = pa_rows.col(j);
                    for l in 0..k {
                        ex[l] += w[l] * row[j];
                    }
                }
            }

            if nm == 0 {
                let x: Vec<f64> = (0..k)
                    .map(|a| tau * (0..k).map(|b| rx_inv[(a, b)] * ex[b]).sum::<f64>())
                    .collect();
                let dd: f64 = row.iter().map(|v| v * v).sum();
                return Ok((x, None, dd));
            }

            let use_missing = nm < d - nm;
            let mut g = if use_missing {
                rx.clone()
            } else {
                base.clone()
            };
            let sign = if use_missing { -tau } else { tau };
            for j in 0..d {
                if m_row[j] == use_missing {
                    let w = pa_rows.col(j);
                    for a in 0..k {
                        for b in 0..k {
                            g[(a, b)] += sign * w[a] * w[b];
                        }
                    }
                }
            }
            let (g_inv, _) = spd_inverse(g.as_ref())?;
            let x: Vec<f64> = (0..k)
                .map(|a| tau * (0..k).map(|b| g_inv[(a, b)] * ex[b]).sum::<f64>())
                .collect();

            let mut quad = 0.0;
            for j in 0..d {
                if m_row[j] {
                    let w = pa_rows.col(j);
                    row[j] = (0..k).map(|l| w[l] * x[l]).sum();
                    for a in 0..k {
                        for b in 0..k {
                            quad += w[a] * g_inv[(a, b)] * w[b];
                        }
                    }
                }
            }
            let dd: f64 = row.iter().map(|v| v * v).sum();
            Ok((x, Some(g_inv), dd + nm as f64 / tau + quad))
        })
        .collect::<Result<_, BixverseErrors>>()?;

    let scores = Mat::from_fn(n, k, |i, l| rows[i].0[l]);
    let tr_s = rows.iter().map(|r| r.2).sum::<f64>() / nf;

    let mut t = gemm(
        MatRef::from_row_major_slice(dy, n, d).transpose(),
        scores.as_ref(),
    );

    // T[j, ] += pa_j^T sum_{i: y_ij missing} Rxinv_i
    let incomplete: Vec<usize> = (0..n).filter(|&i| prep.n_miss_row[i] > 0).collect();
    if !incomplete.is_empty() {
        let mut corr = vec![0.0; d * k];
        corr.par_chunks_mut(k).enumerate().for_each(|(j, out)| {
            let w = pa_rows.col(j);
            for &i in &incomplete {
                if prep.missing[i * d + j] {
                    let g_inv = rows[i].1.as_ref().expect("incomplete rows carry Rxinv");
                    for b in 0..k {
                        out[b] += (0..k).map(|a| w[a] * g_inv[(a, b)]).sum::<f64>();
                    }
                }
            }
        });
        for j in 0..d {
            for b in 0..k {
                t[(j, b)] += corr[j * k + b];
            }
        }
    }
    t *= faer::Scale(1.0 / nf);

    // M-step
    let alpha_diag = Mat::from_fn(k, k, |a, b| if a == b { state.alpha[a] / nf } else { 0.0 });
    let dw = &rx_inv + (t.transpose() * &state.pa * &rx_inv) * tau + alpha_diag;
    let dw_inv = dw.partial_piv_lu().inverse();
    state.pa = &t * &dw_inv;

    let tr_tpa = frob_dot(t.as_ref(), state.pa.as_ref());
    state.tau = (df + 2.0 * BPCA_GTAU0 / nf)
        / (tr_s - tr_tpa + (mean_sq * BPCA_GMU0 + 2.0 * BPCA_GTAU0 / BPCA_BTAU0) / nf);
    state.sig_w = dw_inv * (df / nf);
    state.alpha = state
        .pa
        .col_iter()
        .enumerate()
        .map(|(l, col)| {
            let sq: f64 = col.iter().map(|v| v * v).sum();
            (2.0 * BPCA_GALPHA0 + df)
                / (state.tau * sq + state.sig_w[(l, l)] + 2.0 * BPCA_GALPHA0 / BPCA_BALPHA0)
        })
        .collect();

    Ok(scores)
}

/// BPCA loop on prepped data.
///
/// ### Params
///
/// * `prep` - Prepped data
/// * `params` - BPCA parameters
/// * `verbosity` - Progress reporting
///
/// ### Returns
///
/// The raw [Fit]. Scores come from the last E-step and loadings from the last
/// M-step, as pcaMethods returns them.
fn bpca_fit(
    prep: &PreppedData,
    params: &BpcaParams,
    verbosity: Verbosity,
) -> Result<Fit, BixverseErrors> {
    let (n, d, k) = (prep.n, prep.d, params.n_pcs);
    let mut state = bpca_init(prep, k)?;

    let obs_mean: Vec<f64> = (0..d)
        .into_par_iter()
        .map(|j| {
            let (s, c) = (0..n)
                .filter(|&i| !prep.missing[i * d + j])
                .fold((0.0, 0usize), |(s, c), i| (s + prep.y[i * d + j], c + 1));
            s / c as f64
        })
        .collect();
    let mean_sq: f64 = obs_mean.iter().map(|v| v * v).sum();

    let mut dy = prep.y.clone();
    dy.par_chunks_mut(d).for_each(|row| {
        row.iter_mut().zip(&obs_mean).for_each(|(v, m)| *v -= m);
    });

    let mut tau_old = BPCA_TAU_START;
    let mut scores = Mat::zeros(n, k);
    let mut n_iter = 0;
    let mut converged = false;

    for step in 1..=params.max_iter {
        scores = bpca_step(&mut state, &mut dy, prep, mean_sq)?;
        n_iter = step;
        let check = step % BPCA_CHECK_EVERY == 0;
        if verbosity.detailed_verbosity() && !check {
            println!("BPCA: step {step}, tau {:.6e}", state.tau);
        }
        if check {
            let dtau = (state.tau.log10() - tau_old.log10()).abs();
            if verbosity.normal_verbosity() {
                println!(
                    "BPCA: step {step}, tau {:.6e}, change in log10(tau) {dtau:.3e}",
                    state.tau
                );
            }
            if dtau < params.tol {
                converged = true;
                break;
            }
            tau_old = state.tau;
        }
    }

    if verbosity.normal_verbosity() {
        let status = if converged {
            "converged"
        } else {
            "hit max_iter"
        };
        println!(
            "BPCA: {status} after {n_iter} steps, noise variance {:.4e}",
            1.0 / state.tau
        );
    }

    let r2 = r2_cum(
        &prep.y,
        Some(&prep.missing),
        d,
        scores.as_ref(),
        state.pa.as_ref(),
    );

    Ok(Fit {
        scores,
        loadings: state.pa,
        r2_cum: r2,
        noise_var: 1.0 / state.tau,
        n_iter,
        converged,
    })
}

/// Bayesian PCA with missing values.
///
/// Variational Bayes with an ARD prior per component, so superfluous
/// components shrink towards zero rather than fitting noise. Deterministic:
/// the start comes from an SVD. Loadings are not orthonormal.
///
/// ### Params
///
/// * `y` - Observations x variables, `NaN` for missing
/// * `params` - BPCA parameters; defaults to [BpcaParams::default]
/// * `verbosity` - Progress reporting
///
/// ### Returns
///
/// The [MissingPcaResults], or an error on invalid input or a singular
/// k x k system.
///
/// ### References
///
/// Oba et al., Bioinformatics, 2003; Stacklies et al., Bioinformatics, 2007
pub fn bpca<T>(
    y: MatRef<T>,
    params: Option<BpcaParams>,
    verbosity: Verbosity,
) -> Result<MissingPcaResults<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let params = params.unwrap_or_default();
    let prep = prep_data(y, params.n_pcs, params.centre, params.scale)?;
    let fit = bpca_fit(&prep, &params, verbosity)?;
    Ok(finish(y, &prep, fit))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Rank-`k` signal plus `noise`-scaled Gaussian noise, N x D, with a
    /// fraction `miss` of entries set to `NaN` completely at random.
    ///
    /// Returns the masked matrix and the full one.
    fn low_rank(n: usize, d: usize, k: usize, noise: f64, miss: f64) -> (Mat<f64>, Mat<f64>) {
        let mut rng = StdRng::seed_from_u64(1);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = Mat::<f64>::from_fn(n, k, |_, _| normal.sample(&mut rng));
        let w = Mat::<f64>::from_fn(d, k, |_, _| normal.sample(&mut rng));
        let signal = &z * w.transpose();
        let full = Mat::from_fn(n, d, |i, j| {
            signal[(i, j)] + noise * normal.sample(&mut rng)
        });
        let masked = Mat::from_fn(n, d, |i, j| {
            if rng.random::<f64>() < miss {
                f64::NAN
            } else {
                full[(i, j)]
            }
        });
        (masked, full)
    }

    /// Frobenius distance between the projectors onto the column spaces of
    /// two D x k bases.
    fn subspace_gap(a: MatRef<f64>, b: MatRef<f64>) -> f64 {
        let qa = a.qr().compute_thin_Q();
        let qb = b.qr().compute_thin_Q();
        let diff = &qa * qa.transpose() - &qb * qb.transpose();
        frob_dot(diff.as_ref(), diff.as_ref()).sqrt()
    }

    /// Top-k right singular vectors of the column-centred matrix.
    fn svd_basis(y: MatRef<f64>, k: usize) -> Mat<f64> {
        let (n, d) = y.shape();
        let means: Vec<f64> = (0..d)
            .map(|j| (0..n).map(|i| y[(i, j)]).sum::<f64>() / n as f64)
            .collect();
        let yc = Mat::from_fn(n, d, |i, j| y[(i, j)] - means[j]);
        let svd = yc.thin_svd().unwrap();
        svd.V().subcols(0, k).to_owned()
    }

    /// With nothing missing, both fits recover the PCA subspace.
    #[test]
    fn test_complete_data_matches_svd_subspace() {
        let (_, full) = low_rank(80, 15, 3, 0.1, 0.0);
        let basis = svd_basis(full.as_ref(), 3);

        let params = PpcaParams {
            n_pcs: 3,
            ..PpcaParams::default()
        };
        let pp = ppca(full.as_ref(), Some(params), Verbosity::Quiet).unwrap();
        assert!(pp.converged);
        assert!(subspace_gap(pp.loadings.as_ref(), basis.as_ref()) < 1e-4);

        let params = BpcaParams {
            n_pcs: 3,
            ..BpcaParams::default()
        };
        let bp = bpca(full.as_ref(), Some(params), Verbosity::Quiet).unwrap();
        assert!(subspace_gap(bp.loadings.as_ref(), basis.as_ref()) < 1e-4);
    }

    /// Same seed, same fit; the completed matrix leaves observed entries alone.
    #[test]
    fn test_ppca_seed_reproducible_and_observed_untouched() {
        let (masked, _) = low_rank(50, 10, 2, 0.2, 0.2);
        let a = ppca(masked.as_ref(), None, Verbosity::Quiet).unwrap();
        let b = ppca(masked.as_ref(), None, Verbosity::Quiet).unwrap();
        assert_eq!(a.n_iter, b.n_iter);
        for i in 0..50 {
            for j in 0..10 {
                assert_eq!(a.completed[(i, j)], b.completed[(i, j)]);
                if !masked[(i, j)].is_nan() {
                    assert_eq!(a.completed[(i, j)], masked[(i, j)]);
                }
            }
        }
    }

    /// Low-rank data with 30 % MCAR: imputation error sits near the noise
    /// level, far below the column spread, for both fits and with scaling on.
    #[test]
    fn test_imputation_recovers_low_rank_signal() {
        let (masked, full) = low_rank(120, 30, 3, 0.1, 0.3);
        let rmse = |c: &Mat<f64>| {
            let (mut s, mut m) = (0.0, 0usize);
            for i in 0..120 {
                for j in 0..30 {
                    if masked[(i, j)].is_nan() {
                        s += (c[(i, j)] - full[(i, j)]).powi(2);
                        m += 1;
                    }
                }
            }
            (s / m as f64).sqrt()
        };

        for scale in [false, true] {
            let pp = ppca(
                masked.as_ref(),
                Some(PpcaParams {
                    n_pcs: 3,
                    scale,
                    ..PpcaParams::default()
                }),
                Verbosity::Quiet,
            )
            .unwrap();
            let bp = bpca(
                masked.as_ref(),
                Some(BpcaParams {
                    n_pcs: 3,
                    scale,
                    ..BpcaParams::default()
                }),
                Verbosity::Quiet,
            )
            .unwrap();
            // the signal has per-entry SD ~ sqrt(3); noise SD is 0.1
            assert!(
                rmse(&pp.completed) < 0.3,
                "ppca scale={scale}: {}",
                rmse(&pp.completed)
            );
            assert!(
                rmse(&bp.completed) < 0.3,
                "bpca scale={scale}: {}",
                rmse(&bp.completed)
            );
            assert!(pp.r2_cum.windows(2).all(|w| w[0] <= w[1]));
        }
    }

    /// The f32 entry point agrees with the f64 one: the fit itself runs in f64.
    #[test]
    fn test_f32_input_matches_f64() {
        let (masked, _) = low_rank(40, 8, 2, 0.2, 0.2);
        let masked32 = Mat::<f32>::from_fn(40, 8, |i, j| masked[(i, j)] as f32);
        let a = bpca(masked.as_ref(), None, Verbosity::Quiet).unwrap();
        let b = bpca(masked32.as_ref(), None, Verbosity::Quiet).unwrap();
        for i in 0..40 {
            for j in 0..8 {
                assert!((a.completed[(i, j)] - b.completed[(i, j)] as f64).abs() < 1e-3);
            }
        }
    }

    /// Unusable inputs error rather than panic or return NaN.
    #[test]
    fn test_invalid_inputs_error() {
        let (mut masked, _) = low_rank(20, 6, 2, 0.1, 0.0);
        let too_many = PpcaParams {
            n_pcs: 7,
            ..PpcaParams::default()
        };
        assert!(matches!(
            ppca(masked.as_ref(), Some(too_many), Verbosity::Quiet),
            Err(BixverseErrors::PcaTooManyComponents { n_pcs: 7, max: 6 })
        ));

        for i in 0..20 {
            masked[(i, 4)] = f64::NAN;
        }
        assert!(matches!(
            bpca(masked.as_ref(), None, Verbosity::Quiet),
            Err(BixverseErrors::PcaAllMissing {
                axis: "column",
                index: 4
            })
        ));

        let (mut masked, _) = low_rank(20, 6, 2, 0.1, 0.0);
        for j in 0..6 {
            masked[(3, j)] = f64::NAN;
        }
        assert!(matches!(
            ppca(masked.as_ref(), None, Verbosity::Quiet),
            Err(BixverseErrors::PcaAllMissing {
                axis: "row",
                index: 3
            })
        ));

        let c0 = Mat::<f64>::zeros(5, 2);
        let (full, _) = low_rank(20, 6, 2, 0.1, 0.0);
        assert!(matches!(
            ppca_from_init(full.as_ref(), c0.as_ref(), None, Verbosity::Quiet),
            Err(BixverseErrors::InvalidArgument(_))
        ));
    }
}
