//! Implementation of NMF, dense and sparse paths. This version implements
//! HALS with random or NNDSVD initialisation.

use crate::utils::gemm::gram;
use faer::linalg::matmul::triangular::BlockStructure;
use faer::{Accum, Mat, MatRef};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::core::math::pca_svd::SvdResults;
use crate::prelude::*;
use crate::utils::faer_parallelism;

pub mod consensus;
pub mod dense;
pub mod nmf_preprocessing;
pub mod refit;
pub mod sparse;

/// Columns of H per task in the row sweep.
const HALS_COL_CHUNK: usize = 64;

/// Rows of W per task in the column sweep.
const HALS_ROW_TILE: usize = 256;

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
pub(crate) fn random_init<F: BixverseFloat>(
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
pub(crate) fn nndsvd_from_svd<F: BixverseFloat>(
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
        // f64 accumulators, for the same reason as in `normalise_w_cols`: these
        // run over the sample and feature counts, and they set the scale of the
        // whole initialisation.
        let mut up_sq = 0f64;
        let mut un_sq = 0f64;
        let mut vp_sq = 0f64;
        let mut vn_sq = 0f64;

        for i in 0..m {
            let x = svd.u[(i, r)];
            let d = x.to_f64().unwrap();
            if x > F::zero() {
                up[i] = x;
                up_sq += d * d;
            } else if x < F::zero() {
                un[i] = -x;
                un_sq += d * d;
            }
        }
        for j in 0..n {
            let y = svd.v[(j, r)];
            let d = y.to_f64().unwrap();
            if y > F::zero() {
                vp[j] = y;
                vp_sq += d * d;
            } else if y < F::zero() {
                vn[j] = -y;
                vn_sq += d * d;
            }
        }

        let up_norm = F::from_f64(up_sq.sqrt()).unwrap();
        let un_norm = F::from_f64(un_sq.sqrt()).unwrap();
        let vp_norm = F::from_f64(vp_sq.sqrt()).unwrap();
        let vn_norm = F::from_f64(vn_sq.sqrt()).unwrap();
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
#[derive(Clone, Debug)]
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
}

impl<T> Default for HalsOpts<T>
where
    T: BixverseFloat,
{
    fn default() -> Self {
        Self {
            max_iter: 250,
            tol: T::from_f64(1e-4).unwrap(),
            eps: T::from_f64(1e-10).unwrap(),
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
/// Computes the lower triangle via `gram`, then mirrors it into the upper half.
///
/// ### Params
///
/// * `w` - Left factor, shape `m x k`.
/// * `out` - Output buffer `k x k`. Overwritten on return.
pub(crate) fn gram_wt_w<F: BixverseFloat>(w: MatRef<F>, out: &mut Mat<F>) {
    let k = w.ncols();
    gram(
        out.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        w.transpose(),
        w,
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
/// Computes the lower triangle via `gram`, then mirrors it into the upper half.
///
/// ### Params
///
/// * `h` - Right factor, shape `k x n`.
/// * `out` - Output buffer `k x k`. Overwritten on return.
pub(crate) fn gram_h_ht<F: BixverseFloat>(h: MatRef<F>, out: &mut Mat<F>) {
    let k = h.nrows();
    gram(
        out.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        h,
        h.transpose(),
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
pub(crate) fn hals_sweep_rows<F>(h: &mut Mat<F>, b: MatRef<F>, a: MatRef<F>, eps: F)
where
    F: BixverseFloat + Send + Sync,
{
    let k = h.nrows();

    // Column j of H only reads and writes column j, so the whole r sweep runs
    // inside one pass over the columns.
    let b_rows: Vec<F> = (0..k)
        .flat_map(|r| (0..k).map(move |s| (r, s)))
        .map(|(r, s)| b[(r, s)])
        .collect();

    h.par_col_iter_mut()
        .with_min_len(HALS_COL_CHUNK)
        .enumerate()
        .for_each(|(j, col)| {
            let col = col.try_as_col_major_mut().unwrap().as_slice_mut();
            let a_col = a.col(j);
            for r in 0..k {
                let brr = b_rows[r * k + r];
                if brr <= F::zero() {
                    continue;
                }
                let inv_brr = F::one() / brr;
                let mut bh_rj = F::zero();
                for (&b_rs, &h_sj) in b_rows[r * k..(r + 1) * k].iter().zip(col.iter()) {
                    bh_rj += b_rs * h_sj;
                }
                let new_val = col[r] + (a_col[r] - bh_rj) * inv_brr;
                col[r] = if new_val > eps { new_val } else { eps };
            }
        });
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
pub(crate) fn hals_sweep_cols<F>(w: &mut Mat<F>, d: MatRef<F>, c_mat: MatRef<F>, eps: F)
where
    F: BixverseFloat + Send + Sync,
{
    let m = w.nrows();
    let k = w.ncols();

    let w_ptr = w.as_ptr_mut() as usize;
    let w_row_stride = w.row_stride();
    let w_col_stride = w.col_stride();

    let d_cols: Vec<F> = (0..k)
        .flat_map(|c| (0..k).map(move |s| (s, c)))
        .map(|(s, c)| d[(s, c)])
        .collect();

    // Row i of W only reads and writes row i, so tiles of rows are processed
    // independently, each staged row-major so the k-length dots are contiguous.
    (0..m.div_ceil(HALS_ROW_TILE))
        .into_par_iter()
        .for_each_init(
            || vec![F::zero(); HALS_ROW_TILE * k],
            |scratch, tile| {
                let i0 = tile * HALS_ROW_TILE;
                let rows = HALS_ROW_TILE.min(m - i0);
                let base = w_ptr as *mut F;
                let offset =
                    |i: usize, s: usize| i as isize * w_row_stride + s as isize * w_col_stride;

                for s in 0..k {
                    for ii in 0..rows {
                        // SAFETY: i0 + ii < m and s < k, and tiles cover disjoint rows.
                        scratch[ii * k + s] = unsafe { *base.offset(offset(i0 + ii, s)) };
                    }
                }

                for c in 0..k {
                    let dcc = d_cols[c * k + c];
                    if dcc <= F::zero() {
                        continue;
                    }
                    let inv_dcc = F::one() / dcc;
                    let d_col_c = &d_cols[c * k..(c + 1) * k];
                    for ii in 0..rows {
                        let row = &mut scratch[ii * k..(ii + 1) * k];
                        let mut wd_ic = F::zero();
                        for (&w_is, &d_sc) in row.iter().zip(d_col_c) {
                            wd_ic += w_is * d_sc;
                        }
                        let new_val = row[c] + (c_mat[(i0 + ii, c)] - wd_ic) * inv_dcc;
                        row[c] = if new_val > eps { new_val } else { eps };
                    }
                }

                for s in 0..k {
                    for ii in 0..rows {
                        // SAFETY: as above; this tile is the only writer of its rows.
                        unsafe {
                            *base.offset(offset(i0 + ii, s)) = scratch[ii * k + s];
                        }
                    }
                }
            },
        );
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
pub(crate) fn normalise_w_cols<F: BixverseFloat>(w: &mut Mat<F>, h: &mut Mat<F>) {
    let m = w.nrows();
    let k = w.ncols();
    let n = h.ncols();
    for c in 0..k {
        // f64 accumulator: `m` is the sample count, so on single-cell inputs this
        // is a sum over hundreds of thousands of positive terms. In `F = f32` the
        // norm drifts, and since it is the scale pushed into H it decides whether
        // W's columns really are unit length, which the consensus clustering
        // assumes.
        let mut sq_norm = 0f64;
        for i in 0..m {
            let x = w[(i, c)].to_f64().unwrap();
            sq_norm += x * x;
        }
        let norm = F::from_f64(sq_norm.sqrt()).unwrap();
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
/// avoiding materialising W H. Always in `f64` to avoid issues with
/// catastrophic cancellation.
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
pub(crate) fn compute_objective<F: BixverseFloat + Send + Sync>(
    sq_frob_v: F,
    h: MatRef<F>,
    a: MatRef<F>,
    b: MatRef<F>,
    d: MatRef<F>,
) -> F {
    let k = h.nrows();
    let n = h.ncols();

    let inner_ha: f64 = (0..n)
        .into_par_iter()
        .with_min_len(64)
        .map(|j| {
            let mut acc = 0f64;
            for r in 0..k {
                acc += h[(r, j)].to_f64().unwrap() * a[(r, j)].to_f64().unwrap();
            }
            acc
        })
        .reduce(|| 0f64, |x, y| x + y);

    let mut inner_bd = 0f64;
    for j in 0..k {
        for i in 0..k {
            inner_bd += b[(i, j)].to_f64().unwrap() * d[(i, j)].to_f64().unwrap();
        }
    }

    let result = sq_frob_v.to_f64().unwrap() - 2.0 * inner_ha + inner_bd;
    F::from_f64(result.max(0.0)).unwrap()
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
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// An [`NmfResult`] containing W, H, the final reconstruction loss, the iter
/// count, and whether the relative tolerance was met.
pub fn nmf_hals<F, In>(
    v: &In,
    k: usize,
    opts: &HalsOpts<F>,
    verbose: usize,
) -> Result<NmfResult<F>, BixverseErrors>
where
    F: BixverseFloat + Send + Sync,
    In: NmfInput<F> + Sync,
{
    let (m, n) = v.shape();
    let sq_frob = v.sq_frob();

    let verbosity = parse_verbosity_level(verbose);

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

    // the convergence check leaves wtw and wtv current for the unchanged w
    let mut grams_current = false;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;

        if !grams_current {
            gram_wt_w(w.as_ref(), &mut wtw);
            v.wt_v(w.as_ref(), &mut wtv);
        }
        grams_current = false;
        hals_sweep_rows(&mut h, wtw.as_ref(), wtv.as_ref(), opts.eps);

        gram_h_ht(h.as_ref(), &mut hht);
        v.v_ht(h.as_ref(), &mut vht);
        hals_sweep_cols(&mut w, hht.as_ref(), vht.as_ref(), opts.eps);

        normalise_w_cols(&mut w, &mut h);

        if n_iter.is_multiple_of(opts.check_every) {
            gram_wt_w(w.as_ref(), &mut wtw);
            v.wt_v(w.as_ref(), &mut wtv);
            gram_h_ht(h.as_ref(), &mut hht);
            grams_current = true;

            let loss = compute_objective(
                sq_frob,
                h.as_ref(),
                wtv.as_ref(),
                wtw.as_ref(),
                hht.as_ref(),
            );
            final_loss = loss;

            if n_iter > opts.check_every {
                if verbosity.normal_verbosity() {
                    println!(
                        "  NMF: Iteration {} out of {} - current loss: {:.2?}",
                        iter + 1,
                        opts.max_iter,
                        loss
                    );
                }

                let denom = if sq_frob > F::one() {
                    sq_frob
                } else {
                    F::one()
                };
                let rel = (last_loss - loss).abs() / denom;
                if rel < opts.tol {
                    converged = true;
                    if verbosity.normal_verbosity() {
                        println!("  NMF converged successfully after {} iters", iter + 1)
                    };

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

//////////////////////////
// Multiple run version //
//////////////////////////

/// Result of stabilised NMF across random restarts.
pub struct StabilisedNmfResult<F: BixverseFloat> {
    /// Column-bound W matrices across all runs, shape `m x (k * n_runs)`.
    /// Columns `i*k..(i+1)*k` are run `i`'s components.
    pub w_all: Mat<F>,
    /// Per-run H matrices, each `k x n`.
    pub h_per_run: Vec<Mat<F>>,
    /// Final reconstruction loss for each run.
    pub losses: Vec<F>,
    /// Convergence flag for each run.
    pub converged: Vec<bool>,
    /// Index of the run with the lowest final loss.
    pub best_idx: usize,
}

/// Stabilised NMF via random restarts.
///
/// Runs `nmf_hals` `n_runs` times with random initialisations seeded by
/// `base_seed + i` and column-binds the resulting `W` matrices for downstream
/// consensus clustering. Outer parallelism is capped at
/// `min(n_runs, max(cores / 2, 1))` to leave headroom for the inner
/// parallelism inside HALS. The `init` field of `opts` is ignored; random
/// init is always used.
///
/// ### Params
///
/// * `v` - Input matrix backend.
/// * `k` - Number of components per run.
/// * `n_runs` - Number of random restarts. Must be >= 1.
/// * `base_seed` - Seed offset; run `i` uses `base_seed + i`.
/// * `opts` - HALS options (init field ignored).
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A [`StabilisedNmfResult`] with column-bound `W`, per-run `H`, per-run
/// losses and convergence flags, and the index of the best run.
pub fn stabilised_nmf<F, In>(
    v: &In,
    k: usize,
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<F>,
    verbose: usize,
) -> Result<StabilisedNmfResult<F>, BixverseErrors>
where
    F: BixverseFloat + Send + Sync,
    In: NmfInput<F> + Sync,
{
    assert!(n_runs >= 1, "n_runs must be >= 1");

    let verbosity = parse_verbosity_level(verbose);
    let cores = rayon::current_num_threads();
    let n_outer = n_runs.min((cores / 2).max(1));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_outer)
        .build()
        .expect("failed to build thread pool for stabilised NMF");

    let runs: Vec<NmfResult<F>> = pool.install(|| {
        (0..n_runs)
            .into_par_iter()
            .map(|i| {
                let opts_i = HalsOpts {
                    max_iter: opts.max_iter,
                    tol: opts.tol,
                    eps: opts.eps,
                    check_every: opts.check_every,
                    init: NmfInit::Random {
                        seed: base_seed + i as u64,
                    },
                };
                let inner_verbose = verbose.saturating_sub(1);

                let res = nmf_hals(v, k, &opts_i, inner_verbose);

                if verbosity.normal_verbosity() {
                    println!(" Finished stabilised NMF run: {}", i + 1);
                }

                res
            })
            .collect::<Result<Vec<_>, _>>()
    })?;

    let (m, _) = v.shape();

    let w_all = Mat::<F>::from_fn(m, k * n_runs, |row, col| {
        let run_idx = col / k;
        let comp_idx = col % k;
        runs[run_idx].w[(row, comp_idx)]
    });

    let losses: Vec<F> = runs.iter().map(|r| r.final_loss).collect();
    let converged: Vec<bool> = runs.iter().map(|r| r.converged).collect();
    let h_per_run: Vec<Mat<F>> = runs.into_iter().map(|r| r.h).collect();

    let best_idx = losses
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(StabilisedNmfResult {
        w_all,
        h_per_run,
        losses,
        converged,
        best_idx,
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

    /// The triangular matmul fills only the lower half, so the mirror step must complete W^T W.
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

    /// Same mirror step for H H^T: both triangles must match the direct dot products.
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

    /// The expanded Frobenius objective equals a direct `||V - WH||_F^2` without materialising WH.
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

    /// Column normalisation leaves W H invariant while driving every W column to unit L2 norm.
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

    /// HALS recovers an exactly rank-k dense product with near-zero loss and non-negative factors.
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
        let res = nmf_hals(&input, k, &opts, 0).unwrap();

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

    /// The sparse backend reaches the same low relative loss as the dense path on a rank-k product.
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
        let res = nmf_hals(&sparse_in, k, &opts, 0).unwrap();

        let rel = res.final_loss / sq_frob_v;
        assert!(rel < 1e-2, "rel loss {rel}");
    }

    /// The converged flag is set, and iteration stops early, once the relative tolerance is met.
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
        let res = nmf_hals(&input, k, &opts, 0).unwrap();
        assert!(res.converged);
        assert!(res.n_iter < 1000);
    }

    /// `best_idx` points at the run with the lowest final loss, not merely the first run.
    #[test]
    fn stabilised_nmf_best_idx_is_min_loss() {
        let (m, n, k) = (10, 8, 2);
        let n_runs = 5;
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| (i * 2 + j + 1) as f64);
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(100, 1e-8, 1e-12, 10, NmfInit::Nndsvd);

        let res = stabilised_nmf(&input, k, n_runs, 0, &opts, 0).unwrap();

        let best_loss = res.losses[res.best_idx];
        for &loss in &res.losses {
            assert!(loss >= best_loss - 1e-12);
        }
    }

    /// The same base seed reproduces identical losses and W blocks, despite the outer thread pool.
    #[test]
    fn stabilised_nmf_deterministic_with_same_seed() {
        let (m, n, k) = (10, 8, 2);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| (i + j + 1) as f64);
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(50, 1e-6, 1e-12, 10, NmfInit::Nndsvd);

        let res_a = stabilised_nmf(&input, k, 3, 123, &opts, 0).unwrap();
        let res_b = stabilised_nmf(&input, k, 3, 123, &opts, 0).unwrap();

        assert_eq!(res_a.losses.len(), res_b.losses.len());
        for run in 0..res_a.losses.len() {
            assert!((res_a.losses[run] - res_b.losses[run]).abs() < 1e-9);
        }
        for col in 0..res_a.w_all.ncols() {
            for row in 0..res_a.w_all.nrows() {
                assert!((res_a.w_all[(row, col)] - res_b.w_all[(row, col)]).abs() < 1e-9);
            }
        }
    }

    /// The best of several random restarts still recovers an exactly rank-k product.
    #[test]
    fn stabilised_nmf_recovers_rank_k() {
        let (m, n, k) = (25, 18, 2);
        let w_true: Mat<f64> = Mat::from_fn(m, k, |i, j| ((i * 3 + j) % 4 + 1) as f64);
        let h_true: Mat<f64> = Mat::from_fn(k, n, |i, j| ((i * 2 + j) % 3 + 1) as f64);
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
        let opts = HalsOpts::<f64>::new(300, 1e-8, 1e-12, 10, NmfInit::Nndsvd);

        let res = stabilised_nmf(&input, k, 5, 0, &opts, 0).unwrap();

        let rel = res.losses[res.best_idx] / sq_frob_v;
        assert!(rel < 1e-2, "best rel loss {rel}");
    }

    /// The `init` field is ignored: every restart uses random init, so the per-run W blocks differ.
    #[test]
    fn stabilised_nmf_ignores_init_field() {
        // Passing NmfInit::Nndsvd in opts should not produce identical W columns
        // across runs: random init must override.
        let (m, n, k) = (10, 8, 2);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| (i + j + 1) as f64);
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(20, 1e-3, 1e-12, 10, NmfInit::Nndsvd);

        let res = stabilised_nmf(&input, k, 3, 0, &opts, 0).unwrap();

        // Runs 0 and 1 use different seeds, so their W blocks should differ.
        let mut max_diff = 0.0_f64;
        for col in 0..k {
            for row in 0..m {
                let a = res.w_all[(row, col)];
                let b = res.w_all[(row, col + k)];
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(max_diff > 1e-6, "init field was not overridden to Random");
    }
}
