//! Implementation of NMF, dense and sparse paths. This version implements
//! HALS with random or NNDSVD initialisation.

use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::{Accum, Mat, MatRef};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::core::math::pca_svd::SvdResults;
use crate::prelude::*;
use crate::utils::faer_parallelism;

pub mod dense;
pub mod sparse;

////////////////////
// Initialisation //
////////////////////

#[derive(Clone, Debug, Copy, Default)]
/// Initialisation strategy for HALS.
pub enum NmfInit {
    /// Boutsidis-Gallopoulos NNDSVD from the top-k truncated SVD.
    /// Deterministic.
    #[default]
    Nndsvd,
    /// Random non-negative draws. Used for restart diversity in consensus.
    Random {
        /// Random seed
        seed: u64,
    },
}

/// Parse the NMF initialisation
///
/// ### Params
///
/// * `s` - String to parse. One of `"nndsvd"`, `"svd"` or `"random"`.
/// * `seed` - Seed for the random initialisation
///
/// ### Returns
///
/// The option of [NmfInit]
pub fn parse_nmf_init(s: &str, seed: usize) -> Option<NmfInit> {
    match s.to_lowercase().as_str() {
        "nndsvd" | "svd" => Some(NmfInit::Nndsvd),
        "random" => Some(NmfInit::Random { seed: seed as u64 }),
        _ => None,
    }
}

/// Random initialisation
///
/// ### Params
///
/// * `m` - Number of features
/// * `n` - Number of samples
/// * `k` - Number of components to return
/// * `sq_frob` - The squared Frobenius norm for scaling
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// Tuple of `(w, h)` matrix
fn random_init<F: BixverseFloat>(
    m: usize,
    n: usize,
    k: usize,
    sq_frob: F,
    seed: u64,
) -> (Mat<F>, Mat<F>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mn_k = F::from_usize(m * n * k).unwrap();
    let denom = if mn_k > F::zero() { mn_k } else { F::one() };
    let scale = (sq_frob / denom).sqrt();

    let w = Mat::<F>::from_fn(m, k, |_, _| {
        let v: f64 = normal.sample(&mut rng).abs();
        F::from_f64(v).unwrap() * scale
    });
    let h = Mat::<F>::from_fn(k, n, |_, _| {
        let v: f64 = normal.sample(&mut rng).abs();
        F::from_f64(v).unwrap() * scale
    });

    (w, h)
}

/// Boutsidis-Gallopoulos NNDSVD from a top-k truncated SVD.
///
/// ### Params
///
/// * `svd` - The Singular Value Decomposition results to use for
///   initialisation.
/// * `k` - Number of components to return
/// * `m` - Number of features
/// * `n` - Number of samples
///
/// ### Returns
///
/// Tuple of `(w, h)` matrix
fn nndsvd_from_svd<F: BixverseFloat>(
    svd: &SvdResults<F>,
    k: usize,
    m: usize,
    n: usize,
) -> Result<(Mat<F>, Mat<F>), BixverseErrors> {
    let avail = svd.s.len();
    if avail < k {
        return Err(BixverseErrors::NmfRankTooLarge {
            requested: k,
            available: avail,
        });
    }

    let mut w = Mat::<F>::zeros(m, k);
    let mut h = Mat::<F>::zeros(k, n);

    let s0_sqrt = svd.s[0].sqrt();
    for i in 0..m {
        w[(i, 0)] = svd.u[(i, 0)].abs() * s0_sqrt;
    }
    for j in 0..n {
        h[(0, j)] = svd.v[(j, 0)].abs() * s0_sqrt;
    }

    for r in 1..k {
        let mut up = vec![F::zero(); m];
        let mut un = vec![F::zero(); m];
        let mut vp = vec![F::zero(); n];
        let mut vn = vec![F::zero(); n];
        let mut up_sq = F::zero();
        let mut un_sq = F::zero();
        let mut vp_sq = F::zero();
        let mut vn_sq = F::zero();

        for i in 0..m {
            let x = svd.u[(i, r)];
            if x > F::zero() {
                up[i] = x;
                up_sq += x * x;
            } else if x < F::zero() {
                un[i] = -x;
                un_sq += x * x;
            }
        }
        for j in 0..n {
            let y = svd.v[(j, r)];
            if y > F::zero() {
                vp[j] = y;
                vp_sq += y * y;
            } else if y < F::zero() {
                vn[j] = -y;
                vn_sq += y * y;
            }
        }

        let up_norm = up_sq.sqrt();
        let un_norm = un_sq.sqrt();
        let vp_norm = vp_sq.sqrt();
        let vn_norm = vn_sq.sqrt();
        let m_pos = up_norm * vp_norm;
        let m_neg = un_norm * vn_norm;

        let (chosen_u, chosen_v, u_norm, v_norm, m_chosen) = if m_pos >= m_neg {
            (&up, &vp, up_norm, vp_norm, m_pos)
        } else {
            (&un, &vn, un_norm, vn_norm, m_neg)
        };

        if u_norm <= F::zero() || v_norm <= F::zero() {
            continue;
        }
        let factor = (svd.s[r] * m_chosen).sqrt();
        let scale_u = factor / u_norm;
        let scale_v = factor / v_norm;
        for i in 0..m {
            w[(i, r)] = chosen_u[i] * scale_u;
        }
        for j in 0..n {
            h[(r, j)] = chosen_v[j] * scale_v;
        }
    }

    Ok((w, h))
}

////////////
// Params //
////////////

/// HALS solver options.
pub struct HalsOpts<F: BixverseFloat> {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Relative objective decrease below which the solver stops.
    pub tol: F,
    /// Floor applied to W and H entries on every sweep.
    pub eps: F,
    /// Cadence (in iterations) for evaluating the Frobenius objective.
    pub check_every: usize,
    /// How W and H are initialised.
    pub init: NmfInit,
}

impl<F: BixverseFloat> HalsOpts<F> {
    /// Generate a new instance of [HalsOpts]
    ///
    /// ### Params
    ///
    /// * `max_iter` - Maximum iterations to run the algorithm for
    /// * `tol` - Tolerance for early stopping
    /// * `eps` - Floor applied to W and H entries on every sweep.
    /// * `check_every` - Cadence (in iters) for evaluating the Frobenius
    ///   objective.
    /// * `init` - How the W and H matrices are initialised, see [NmfInit].
    ///
    /// ### Returns
    ///
    /// The initialised [HalsOpts].
    pub fn new(max_iter: usize, tol: F, eps: F, check_every: usize, init: NmfInit) -> Self {
        Self {
            max_iter,
            tol,
            eps,
            check_every,
            init,
        }
    }

    /// Generate default parameters
    pub fn defaults() -> Self {
        Self {
            max_iter: 250,
            tol: F::from_f64(1e-4).unwrap(),
            eps: F::from_f64(1e-10).unwrap(),
            check_every: 10,
            init: NmfInit::Nndsvd,
        }
    }
}

/////////////
// Results //
/////////////

/// Result of a single NMF run.
pub struct NmfResult<F: BixverseFloat> {
    /// Left factor (m x k).
    pub w: Mat<F>,
    /// Right factor (k x n).
    pub h: Mat<F>,
    /// Frobenius squared error at the last evaluated point.
    pub final_loss: F,
    /// Number of outer iterations actually performed.
    pub n_iter: usize,
    /// True if the relative tolerance was met before `max_iter`.
    pub converged: bool,
}

//////////////
// NmfInput //
//////////////

/// Backend-agnostic view of V for HALS NMF. Implementations write the two
/// data-touching products into caller-owned buffers.
pub trait NmfInput<F: BixverseFloat> {
    /// Returns the shape
    ///
    /// ### Returns
    ///
    /// `(m, n)`.
    fn shape(&self) -> (usize, usize);

    /// The squared Frobenius norm `||V||_F^2`,
    ///
    /// Computed once at construction.
    ///
    /// ### Returns
    ///
    /// The squared Frobenius norm.
    fn sq_frob(&self) -> F;

    /// Compute W^T V.
    ///
    /// Writes the product `W^T V` into a caller-owned buffer.
    ///
    /// ### Params
    ///
    /// * `w` - Left factor, shape `m x k`.
    /// * `out` - Output buffer `k x n`. Overwritten on return.
    fn wt_v(&self, w: MatRef<F>, out: &mut Mat<F>);

    /// Gram matrix W^T W.
    ///
    /// Computes the symmetric Gram matrix using a triangular matmul, then
    /// mirrors the lower triangle into the upper half.
    ///
    /// ### Params
    ///
    /// * `w` - Left factor, shape `m x k`.
    /// * `out` - Output buffer `k x k`. Overwritten on return.
    fn v_ht(&self, h: MatRef<F>, out: &mut Mat<F>);

    /// Top-k truncated SVD of V.
    ///
    /// Used to seed NNDSVD initialisation.
    ///
    /// ### Params
    ///
    /// * `k` - Number of singular triplets to compute.
    ///
    /// ### Returns
    ///
    /// The top-k [`SvdResults`], or a [`BixverseErrors`] if the decomposition
    /// fails.
    fn top_k_svd(&self, k: usize) -> Result<SvdResults<F>, BixverseErrors>;
}

/////////////
// Helpers //
/////////////

/// Gram matrix W^T W.
///
/// Computes the symmetric Gram matrix using a triangular matmul, then mirrors
/// the lower triangle into the upper half.
///
/// ### Params
///
/// * `w` - Left factor, shape `m x k`.
/// * `out` - Output buffer `k x k`. Overwritten on return.
fn gram_wt_w<F: BixverseFloat>(w: MatRef<F>, out: &mut Mat<F>) {
    let k = w.ncols();
    triangular_matmul(
        out.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        w.transpose(),
        BlockStructure::Rectangular,
        w,
        BlockStructure::Rectangular,
        F::one(),
        faer_parallelism(),
    );
    for j in 0..k {
        for i in 0..j {
            out[(i, j)] = out[(j, i)];
        }
    }
}

/// Gram matrix H H^T.
///
/// Computes the symmetric Gram matrix using a triangular matmul, then mirrors
/// the lower triangle into the upper half.
///
/// ### Params
///
/// * `h` - Right factor, shape `k x n`.
/// * `out` - Output buffer `k x k`. Overwritten on return.
fn gram_h_ht<F: BixverseFloat>(h: MatRef<F>, out: &mut Mat<F>) {
    let k = h.nrows();
    triangular_matmul(
        out.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        h,
        BlockStructure::Rectangular,
        h.transpose(),
        BlockStructure::Rectangular,
        F::one(),
        faer_parallelism(),
    );
    for j in 0..k {
        for i in 0..j {
            out[(i, j)] = out[(j, i)];
        }
    }
}

/// HALS row sweep for H.
///
/// For each rank component `r`, updates row `r` of `H` in-place:
///
/// `H[r,j] = max(eps, H[r,j] + (A[r,j] - (BH)[r,j]) / B[r,r])`.
///
/// ### Params
///
/// * `h` - Right factor `k x n`, updated in-place.
/// * `b` - Gram matrix `W^T W`, shape `k x k`.
/// * `a` - Product `W^T V`, shape `k x n`.
/// * `eps` - Non-negativity floor.
fn hals_sweep_rows<F>(h: &mut Mat<F>, b: MatRef<F>, a: MatRef<F>, eps: F)
where
    F: BixverseFloat + Send + Sync,
{
    let k = h.nrows();
    let n = h.ncols();

    let h_ptr = h.as_ptr_mut() as usize;
    let h_row_stride = h.row_stride();
    let h_col_stride = h.col_stride();

    for r in 0..k {
        let brr = b[(r, r)];
        if brr <= F::zero() {
            continue;
        }
        let inv_brr = F::one() / brr;
        let b_row_r: Vec<F> = (0..k).map(|s| b[(r, s)]).collect();
        let a_row_r: Vec<F> = (0..n).map(|j| a[(r, j)]).collect();

        (0..n).into_par_iter().for_each(|j| {
            let base = h_ptr as *mut F;
            let mut bh_rj = F::zero();
            for s in 0..k {
                let h_sj =
                    unsafe { *base.offset(s as isize * h_row_stride + j as isize * h_col_stride) };
                bh_rj += b_row_r[s] * h_sj;
            }
            let off = r as isize * h_row_stride + j as isize * h_col_stride;
            let h_rj = unsafe { *base.offset(off) };
            let new_val = h_rj + (a_row_r[j] - bh_rj) * inv_brr;
            let clamped = if new_val > eps { new_val } else { eps };
            unsafe {
                *base.offset(off) = clamped;
            }
        });
    }
}

/// HALS column sweep for W.
///
/// For each rank component `c`, updates column `c` of `W` in-place:
///
/// `W[i,c] = max(eps, W[i,c] + (C[i,c] - (WD)[i,c]) / D[c,c])`.
///
/// ### Params
///
/// * `w` - Left factor `m x k`, updated in-place.
/// * `d` - Gram matrix `H H^T`, shape `k x k`.
/// * `c_mat` - Product `V H^T`, shape `m x k`.
/// * `eps` - Non-negativity floor.
fn hals_sweep_cols<F>(w: &mut Mat<F>, d: MatRef<F>, c_mat: MatRef<F>, eps: F)
where
    F: BixverseFloat + Send + Sync,
{
    let m = w.nrows();
    let k = w.ncols();

    let w_ptr = w.as_ptr_mut() as usize;
    let w_row_stride = w.row_stride();
    let w_col_stride = w.col_stride();

    for c in 0..k {
        let dcc = d[(c, c)];
        if dcc <= F::zero() {
            continue;
        }
        let inv_dcc = F::one() / dcc;
        let d_col_c: Vec<F> = (0..k).map(|s| d[(s, c)]).collect();

        (0..m).into_par_iter().for_each(|i| {
            let base = w_ptr as *mut F;
            let mut wd_ic = F::zero();
            for s in 0..k {
                let w_is =
                    unsafe { *base.offset(i as isize * w_row_stride + s as isize * w_col_stride) };
                wd_ic += w_is * d_col_c[s];
            }
            let off = i as isize * w_row_stride + c as isize * w_col_stride;
            let w_ic = unsafe { *base.offset(off) };
            let new_val = w_ic + (c_mat[(i, c)] - wd_ic) * inv_dcc;
            let clamped = if new_val > eps { new_val } else { eps };
            unsafe {
                *base.offset(off) = clamped;
            }
        });
    }
}

/// Normalise columns of W to unit L2 norm.
///
/// Absorbs each column's norm into the corresponding row of H, keeping
/// the product W H invariant.
///
/// ### Params
///
/// * `w` - Left factor `m x k`, columns normalised in-place.
/// * `h` - Right factor `k x n`, rows rescaled in-place.
fn normalise_w_cols<F: BixverseFloat>(w: &mut Mat<F>, h: &mut Mat<F>) {
    let m = w.nrows();
    let k = w.ncols();
    let n = h.ncols();
    for c in 0..k {
        let mut sq_norm = F::zero();
        for i in 0..m {
            sq_norm += w[(i, c)] * w[(i, c)];
        }
        let norm = sq_norm.sqrt();
        if norm <= F::zero() {
            continue;
        }
        let inv_norm = F::one() / norm;
        for i in 0..m {
            w[(i, c)] *= inv_norm;
        }
        for j in 0..n {
            h[(c, j)] *= norm;
        }
    }
}

/// Frobenius squared reconstruction error.
///
/// Evaluates `||V - WH||_F^2` via the expansion
/// `||V||_F^2 - 2<W^T V, H> + <W^T W, H H^T>`,
/// avoiding materialising W H.
///
/// ### Params
///
/// * `sq_frob_v` - Precomputed `||V||_F^2`.
/// * `h` - Right factor `k x n`.
/// * `a` - Product `W^T V`, shape `k x n`.
/// * `b` - Gram matrix `W^T W`, shape `k x k`.
/// * `d` - Gram matrix `H H^T`, shape `k x k`.
///
/// ### Returns
///
/// The scalar reconstruction error.
fn compute_objective<F: BixverseFloat + Send + Sync>(
    sq_frob_v: F,
    h: MatRef<F>,
    a: MatRef<F>,
    b: MatRef<F>,
    d: MatRef<F>,
) -> F {
    let k = h.nrows();
    let n = h.ncols();

    let inner_ha = (0..n)
        .into_par_iter()
        .with_min_len(64)
        .map(|j| {
            let mut acc = F::zero();
            for r in 0..k {
                acc += h[(r, j)] * a[(r, j)];
            }
            acc
        })
        .reduce(|| F::zero(), |x, y| x + y);

    let mut inner_bd = F::zero();
    for j in 0..k {
        for i in 0..k {
            inner_bd += b[(i, j)] * d[(i, j)];
        }
    }

    let two = F::from_f64(2.0).unwrap();
    sq_frob_v - two * inner_ha + inner_bd
}

//////////
// Main //
//////////

/// Frobenius HALS NMF.
///
/// Runs HALS on any backend implementing [`NmfInput`]. Initialises W and H via
/// the strategy in `opts.init`, then alternates row and column HALS sweeps
/// until convergence or `max_iter` is reached. W columns are normalised after
/// each sweep.
///
/// ### Params
///
/// * `v` - Input matrix backend.
/// * `k` - Number of components.
/// * `opts` - Solver options; see [`HalsOpts`].
///
/// ### Returns
///
/// An [`NmfResult`] containing W, H, the final reconstruction loss, the iter
/// count, and whether the relative tolerance was met.
pub fn nmf_hals<F, In>(v: &In, k: usize, opts: &HalsOpts<F>) -> Result<NmfResult<F>, BixverseErrors>
where
    F: BixverseFloat + Send + Sync,
    In: NmfInput<F> + Sync,
{
    let (m, n) = v.shape();
    let sq_frob = v.sq_frob();

    let (mut w, mut h) = match &opts.init {
        NmfInit::Nndsvd => {
            let svd = v.top_k_svd(k)?;
            nndsvd_from_svd(&svd, k, m, n)?
        }
        NmfInit::Random { seed } => random_init(m, n, k, sq_frob, *seed),
    };

    let mut wtv = Mat::<F>::zeros(k, n);
    let mut vht = Mat::<F>::zeros(m, k);
    let mut wtw = Mat::<F>::zeros(k, k);
    let mut hht = Mat::<F>::zeros(k, k);

    let mut final_loss = F::infinity();
    let mut last_loss = F::infinity();
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;

        gram_wt_w(w.as_ref(), &mut wtw);
        v.wt_v(w.as_ref(), &mut wtv);
        hals_sweep_rows(&mut h, wtw.as_ref(), wtv.as_ref(), opts.eps);

        gram_h_ht(h.as_ref(), &mut hht);
        v.v_ht(h.as_ref(), &mut vht);
        hals_sweep_cols(&mut w, hht.as_ref(), vht.as_ref(), opts.eps);

        normalise_w_cols(&mut w, &mut h);

        if n_iter.is_multiple_of(opts.check_every) {
            gram_wt_w(w.as_ref(), &mut wtw);
            v.wt_v(w.as_ref(), &mut wtv);
            gram_h_ht(h.as_ref(), &mut hht);

            let loss = compute_objective(
                sq_frob,
                h.as_ref(),
                wtv.as_ref(),
                wtw.as_ref(),
                hht.as_ref(),
            );
            final_loss = loss;

            if n_iter > opts.check_every {
                let denom = if sq_frob > F::one() {
                    sq_frob
                } else {
                    F::one()
                };
                let rel = (last_loss - loss).abs() / denom;
                if rel < opts.tol {
                    converged = true;
                    break;
                }
            }
            last_loss = loss;
        }
    }

    if !n_iter.is_multiple_of(opts.check_every) {
        gram_wt_w(w.as_ref(), &mut wtw);
        v.wt_v(w.as_ref(), &mut wtv);
        gram_h_ht(h.as_ref(), &mut hht);
        final_loss = compute_objective(
            sq_frob,
            h.as_ref(),
            wtv.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );
    }

    Ok(NmfResult {
        w,
        h,
        final_loss,
        n_iter,
        converged,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::nmf_hals::dense::DenseInput;
    use crate::methods::nmf_hals::sparse::SparseInput;
    use faer::Mat;

    #[test]
    fn gram_wt_w_full_symmetric() {
        let (m, k) = (5, 3);
        let w: Mat<f64> = Mat::from_fn(m, k, |i, j| (i + j + 1) as f64);
        let mut out = Mat::<f64>::zeros(k, k);
        gram_wt_w(w.as_ref(), &mut out);
        for r in 0..k {
            for c in 0..k {
                let expected: f64 = (0..m).map(|i| w[(i, r)] * w[(i, c)]).sum();
                assert!((out[(r, c)] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn gram_h_ht_full_symmetric() {
        let (k, n) = (3, 5);
        let h: Mat<f64> = Mat::from_fn(k, n, |i, j| (i + j + 1) as f64);
        let mut out = Mat::<f64>::zeros(k, k);
        gram_h_ht(h.as_ref(), &mut out);
        for r in 0..k {
            for c in 0..k {
                let expected: f64 = (0..n).map(|j| h[(r, j)] * h[(c, j)]).sum();
                assert!((out[(r, c)] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn objective_matches_direct() {
        let (m, n, k) = (4, 3, 2);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| (i + j + 1) as f64);
        let w: Mat<f64> = Mat::from_fn(m, k, |i, j| (i + j + 1) as f64 * 0.1);
        let h: Mat<f64> = Mat::from_fn(k, n, |i, j| (i + j + 1) as f64 * 0.2);

        let mut sq_frob = 0.0;
        for i in 0..m {
            for j in 0..n {
                sq_frob += v_mat[(i, j)].powi(2);
            }
        }

        let mut direct = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut wh = 0.0;
                for r in 0..k {
                    wh += w[(i, r)] * h[(r, j)];
                }
                direct += (v_mat[(i, j)] - wh).powi(2);
            }
        }

        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let mut wtv = Mat::<f64>::zeros(k, n);
        let mut wtw = Mat::<f64>::zeros(k, k);
        let mut hht = Mat::<f64>::zeros(k, k);
        input.wt_v(w.as_ref(), &mut wtv);
        gram_wt_w(w.as_ref(), &mut wtw);
        gram_h_ht(h.as_ref(), &mut hht);

        let via = compute_objective(
            sq_frob,
            h.as_ref(),
            wtv.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );
        assert!((direct - via).abs() < 1e-8);
    }

    #[test]
    fn normalise_preserves_product_and_unit_norms() {
        let (m, n, k) = (4, 3, 2);
        let mut w: Mat<f64> = Mat::from_fn(m, k, |i, j| (i + j + 1) as f64);
        let mut h: Mat<f64> = Mat::from_fn(k, n, |i, j| (i + j + 1) as f64);

        let mut before = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut prod = 0.0;
                for r in 0..k {
                    prod += w[(i, r)] * h[(r, j)];
                }
                before.push(prod);
            }
        }

        normalise_w_cols(&mut w, &mut h);

        for c in 0..k {
            let mut sq = 0.0;
            for i in 0..m {
                sq += w[(i, c)].powi(2);
            }
            assert!((sq.sqrt() - 1.0).abs() < 1e-9);
        }

        let mut after = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut prod = 0.0;
                for r in 0..k {
                    prod += w[(i, r)] * h[(r, j)];
                }
                after.push(prod);
            }
        }
        for (a, b) in after.iter().zip(before.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn hals_recovers_rank_k_dense() {
        let (m, n, k) = (30, 20, 3);
        let w_true: Mat<f64> = Mat::from_fn(m, k, |i, j| ((i * 7 + j * 3) % 5 + 1) as f64);
        let h_true: Mat<f64> = Mat::from_fn(k, n, |i, j| ((i * 5 + j * 2) % 4 + 1) as f64);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| {
            let mut s = 0.0;
            for r in 0..k {
                s += w_true[(i, r)] * h_true[(r, j)];
            }
            s
        });

        let mut sq_frob_v = 0.0;
        for i in 0..m {
            for j in 0..n {
                sq_frob_v += v_mat[(i, j)].powi(2);
            }
        }

        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(400, 1e-9, 1e-12, 5, NmfInit::Nndsvd);
        let res = nmf_hals(&input, k, &opts).unwrap();

        let rel = res.final_loss / sq_frob_v;
        assert!(rel < 1e-3, "rel loss {rel}");

        for i in 0..m {
            for r in 0..k {
                assert!(res.w[(i, r)] >= 0.0);
            }
        }
        for r in 0..k {
            for j in 0..n {
                assert!(res.h[(r, j)] >= 0.0);
            }
        }
    }

    #[test]
    fn hals_random_init_runs_and_decreases() {
        let (m, n, k) = (10, 8, 2);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| (i + j + 1) as f64);

        let mut sq_frob_v = 0.0;
        for i in 0..m {
            for j in 0..n {
                sq_frob_v += v_mat[(i, j)].powi(2);
            }
        }

        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(100, 1e-8, 1e-12, 10, NmfInit::Random { seed: 7 });
        let res = nmf_hals(&input, k, &opts).unwrap();
        assert!(res.final_loss.is_finite());
        assert!(res.final_loss < sq_frob_v);
        assert_eq!(res.w.shape(), (m, k));
        assert_eq!(res.h.shape(), (k, n));
    }

    #[test]
    fn hals_sparse_recovers_rank_k() {
        let (m, n, k) = (20, 15, 2);
        let w_true: Mat<f64> = Mat::from_fn(m, k, |i, j| {
            if (i + j) % 3 == 0 {
                (i + 1) as f64
            } else {
                0.0
            }
        });
        let h_true: Mat<f64> = Mat::from_fn(k, n, |i, j| ((i * 3 + j) % 4 + 1) as f64);
        let v_dense: Mat<f64> = Mat::from_fn(m, n, |i, j| {
            let mut s = 0.0;
            for r in 0..k {
                s += w_true[(i, r)] * h_true[(r, j)];
            }
            s
        });

        let mut sq_frob_v = 0.0;
        for i in 0..m {
            for j in 0..n {
                sq_frob_v += v_dense[(i, j)].powi(2);
            }
        }

        let csr = CompressedSparseData2::<f64, f64>::from_dense_matrix(
            v_dense.as_ref(),
            CompressedSparseFormat::Csr,
        );
        let sparse_in: SparseInput<f64, f64> = SparseInput::from_primary(&csr).unwrap();

        let opts = HalsOpts::<f64>::new(400, 1e-9, 1e-12, 10, NmfInit::Nndsvd);
        let res = nmf_hals(&sparse_in, k, &opts).unwrap();

        let rel = res.final_loss / sq_frob_v;
        assert!(rel < 1e-2, "rel loss {rel}");
    }

    #[test]
    fn hals_converged_flag_set_when_tol_met() {
        let (m, n, k) = (15, 10, 2);
        let w_true: Mat<f64> = Mat::from_fn(m, k, |i, j| ((i + 2 * j) % 4 + 1) as f64);
        let h_true: Mat<f64> = Mat::from_fn(k, n, |i, j| ((2 * i + j) % 3 + 1) as f64);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| {
            (0..k).map(|r| w_true[(i, r)] * h_true[(r, j)]).sum()
        });
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(1000, 1e-6, 1e-12, 5, NmfInit::Nndsvd);
        let res = nmf_hals(&input, k, &opts).unwrap();
        assert!(res.converged);
        assert!(res.n_iter < 1000);
    }
}
