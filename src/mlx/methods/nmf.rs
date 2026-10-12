//! MLX HALS NMF, dense input only.
//!
//! Mirrors [`crate::methods::nmf_hals::nmf_hals`] and the cubecl port in
//! `gpu::methods_gpu::nmf_gpu` iteration for iteration. Layout follows the
//! cubecl path: `W` is `[m, k]` row-major and `H` is held transposed as
//! `[n, k]`, so both Grams, both data products and both sweeps see the same
//! component-minor layout. The Grams and data products are MLX GEMMs over lazy
//! transposes; the Gauss-Seidel sweep and the normalisation scale are custom
//! Metal kernels. One iteration stays in flight: iteration `t` is submitted
//! asynchronously before the host waits on `t - 1`.

use std::borrow::Cow;

use faer::{Mat, MatRef};

use crate::methods::nmf_hals::consensus::{
    ConsensusNmfResult, ConsensusParams, ConsensusTarget, consensus_from_restarts,
};
use crate::methods::nmf_hals::dense::DenseInput;
use crate::methods::nmf_hals::nmf_preprocessing::{
    NmfPreprocessing, nmf_process_dense, parse_nmf_processing,
};
use crate::methods::nmf_hals::{
    HalsOpts, NmfInit, NmfInput, NmfResult, StabilisedNmfResult, nndsvd_from_svd, random_init,
};
use crate::mlx::ffi::{
    Array, MLX_FLOAT32, MetalKernel, OutputSpec, Stream, clear_cache, eval_all,
    install_error_handler,
};
use crate::prelude::*;

////////////
// Consts //
////////////

/// Largest rank the sweep is compiled for. Same bound as the cubecl sweep: at
/// `k_cap = 128` the threadgroup is down to one SIMD group of 32.
pub const NMF_MAX_RANK: usize = 128;

/// Threads per threadgroup for the flat elementwise scale kernel.
const ELEMENTWISE_TG: usize = 256;

/// Metal source of the HALS sweep. One thread per row; the row is staged in
/// threadgroup memory at `stage[s * WG + tx]` exactly as the cubecl kernel
/// does, `K_CAP * WG = 4096` floats (16 KiB) at every tier.
const SWEEP_SOURCE: &str = r#"
    uint tx = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.x;
    threadgroup float stage[K_CAP * WG];
    if (row >= uint(N_ROWS)) {
        return;
    }
    uint base = row * uint(K);
    for (uint s = 0; s < uint(K); ++s) {
        stage[s * WG + tx] = x[base + s];
    }
    for (uint r = 0; r < uint(K); ++r) {
        float grr = g[r * K + r];
        if (grr > 0.0f) {
            float dot = 0.0f;
            for (uint t = 0; t < uint(K); ++t) {
                dot += g[r * K + t] * stage[t * WG + tx];
            }
            uint slot = r * WG + tx;
            float updated = stage[slot] + (p[base + r] - dot) / grr;
            stage[slot] = updated < eps ? eps : updated;
        }
    }
    for (uint s = 0; s < uint(K); ++s) {
        x_out[base + s] = stage[s * WG + tx];
    }
"#;

/// Metal source of the normalisation scale. Turns a per-column sum of squares
/// into a factor (the norm, or its reciprocal with `INVERT`) and scales every
/// element of a `[rows, k]` buffer by its column's factor. A collapsed column
/// gets a factor of one, as on the CPU and in the cubecl path.
const SCALE_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= uint(TOTAL)) {
        return;
    }
    float v = sq[idx % uint(K)];
    float f = 1.0f;
    if (v > 0.0f) {
        float nrm = sqrt(v);
        f = INVERT ? 1.0f / nrm : nrm;
    }
    x_out[idx] = x[idx] * f;
"#;

/////////////
// Helpers //
/////////////

/// Rank capacity and threadgroup width for a sweep at rank `k`.
///
/// `k_cap * wg` is 4096 in every arm, so the threadgroup footprint does not
/// depend on `k`. Same tiers as the cubecl sweep.
///
/// ### Params
///
/// * `k` - Number of components
///
/// ### Returns
///
/// `Some((k_cap, wg))`, or `None` if `k` is zero or above [`NMF_MAX_RANK`].
fn sweep_tier(k: usize) -> Option<(usize, usize)> {
    match k {
        0 => None,
        1..=16 => Some((16, 256)),
        17..=32 => Some((32, 128)),
        33..=64 => Some((64, 64)),
        65..=NMF_MAX_RANK => Some((128, 32)),
        _ => None,
    }
}

/// Reject a rank the sweep is not compiled for.
///
/// ### Params
///
/// * `k` - Number of components
///
/// ### Returns
///
/// `Ok(())`, or `MlxNmfRankTooLarge`.
fn check_rank(k: usize) -> Result<(), BixverseErrors> {
    match sweep_tier(k) {
        Some(_) => Ok(()),
        None => Err(BixverseErrors::MlxNmfRankTooLarge {
            k,
            max: NMF_MAX_RANK,
        }),
    }
}

/// Convert a size to the `i32` mlx-c takes for shapes, grids and templates.
///
/// ### Params
///
/// * `x` - The size
///
/// ### Returns
///
/// `x` as `i32`. Panics past `i32::MAX`, which MLX cannot index either.
fn to_i32(x: usize) -> i32 {
    i32::try_from(x).expect("size exceeds i32, which MLX cannot address")
}

/// Borrow a column-major faer matrix as one flat buffer when it has no padding
/// between columns, otherwise copy it.
///
/// A column-major `rows x cols` buffer is `[cols, rows]` row-major, i.e. the
/// transpose, so this is how `V^T` and `H^T` reach the device.
///
/// ### Params
///
/// * `x` - The matrix
///
/// ### Returns
///
/// The column-major values, borrowed or owned.
fn col_major_flat<'a>(x: MatRef<'a, f32>) -> Cow<'a, [f32]> {
    let (rows, cols) = (x.nrows(), x.ncols());
    if let Some(cm) = x.try_as_col_major()
        && cm.col_stride() as usize == rows
    {
        // SAFETY: unit row stride from `try_as_col_major` and no padding
        // between columns, so the `rows * cols` elements are one contiguous
        // range of the allocation starting at `as_ptr()`.
        return Cow::Borrowed(unsafe { std::slice::from_raw_parts(cm.as_ptr(), rows * cols) });
    }
    let mut out = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            out.push(x[(i, j)]);
        }
    }
    Cow::Owned(out)
}

/// Flatten a faer `rows x cols` matrix into row-major `[rows, cols]`.
///
/// ### Params
///
/// * `x` - The matrix
///
/// ### Returns
///
/// `rows * cols` values, row `i` contiguous at `i * cols`.
fn to_row_major(x: MatRef<f32>) -> Vec<f32> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut out = vec![0f32; rows * cols];
    for (i, row) in out.chunks_exact_mut(cols.max(1)).enumerate().take(rows) {
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = x[(i, c)];
        }
    }
    out
}

/// Gram of a `[rows, k]` buffer, `X^T X`, via a lazy transpose.
///
/// ### Params
///
/// * `x` - Buffer `[rows, k]`
/// * `s` - Stream
///
/// ### Returns
///
/// The lazy `[k, k]` Gram.
fn gram(x: &Array, s: &Stream) -> Result<Array, BixverseErrors> {
    x.transpose(s)?.matmul(x, s)
}

/// Per-row dot products of two `[rows, k]` buffers, `out[i] = sum_s x[i, s] *
/// y[i, s]`. One partial per row so the host finishes the sum in f64, as the
/// cubecl objective does.
///
/// ### Params
///
/// * `x` - First buffer `[rows, k]`
/// * `y` - Second buffer `[rows, k]`
/// * `s` - Stream
///
/// ### Returns
///
/// The lazy `[rows]` partials.
fn row_dot_partials(x: &Array, y: &Array, s: &Stream) -> Result<Array, BixverseErrors> {
    x.multiply(y, s)?.sum_axis(1, s)
}

/// Evaluate the objective parts and finish `||V||^2 - 2 sum(partials) + <B,
/// D>` on the host in f64.
///
/// ### Params
///
/// * `sq_frob` - `||V||_F^2`
/// * `partials` - Per-row partials of the cross term
/// * `b` - `W^T W`, `[k, k]`
/// * `d` - `H H^T`, `[k, k]`
///
/// ### Returns
///
/// The reconstruction error, clamped at zero.
fn finish_objective(
    sq_frob: f32,
    partials: &Array,
    b: &Array,
    d: &Array,
) -> Result<f32, BixverseErrors> {
    eval_all(&[partials, b, d], false)?;
    let cross: f64 = partials.as_f32()?.iter().map(|&x| x as f64).sum();
    let bd: f64 = b
        .as_f32()?
        .iter()
        .zip(d.as_f32()?)
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    Ok((sq_frob as f64 - 2.0 * cross + bd).max(0.0) as f32)
}

/// The relative-tolerance denominator, floored at one.
///
/// ### Params
///
/// * `sq_frob` - `||V||_F^2`
///
/// ### Returns
///
/// `max(sq_frob, 1)`.
fn relative_denominator(sq_frob: f32) -> f32 {
    if sq_frob > 1.0 { sq_frob } else { 1.0 }
}

/////////////
// Kernels //
/////////////

/// The two custom kernels of the solve, registered once per input.
struct HalsKernels {
    /// Gauss-Seidel sweep over all components of a `[rows, k]` factor
    sweep: MetalKernel,
    /// Column scale by a norm or its reciprocal
    scale: MetalKernel,
}

impl HalsKernels {
    /// Register both kernels.
    ///
    /// ### Returns
    ///
    /// The kernel handles.
    fn new() -> Self {
        Self {
            sweep: MetalKernel::new(
                "bixverse_hals_sweep",
                &["x", "g", "p", "eps"],
                &["x_out"],
                "",
                SWEEP_SOURCE,
            ),
            scale: MetalKernel::new(
                "bixverse_hals_scale",
                &["x", "sq"],
                &["x_out"],
                "",
                SCALE_SOURCE,
            ),
        }
    }

    /// One HALS sweep, `X[i, r] = max(eps, X[i, r] + (P[i, r] - sum_s G[r, s]
    /// X[i, s]) / G[r, r])` for every `r` in order. Covers the H sweep with
    /// `X = H^T`, `G = W^T W`, `P = (W^T V)^T`, and the W sweep with `X = W`,
    /// `G = H H^T`, `P = V H^T`. A component with a non-positive Gram diagonal
    /// is skipped, as on the CPU.
    ///
    /// ### Params
    ///
    /// * `x` - Factor `[n_rows, k]`
    /// * `g` - Symmetric Gram `[k, k]`
    /// * `p` - Data product `[n_rows, k]`
    /// * `eps` - Non-negativity floor, a 0-d array
    /// * `n_rows` - Rows of `x`
    /// * `k` - Number of components
    /// * `s` - Stream
    ///
    /// ### Returns
    ///
    /// The lazy swept factor `[n_rows, k]`.
    #[allow(clippy::too_many_arguments)]
    fn sweep(
        &self,
        x: &Array,
        g: &Array,
        p: &Array,
        eps: &Array,
        n_rows: usize,
        k: usize,
        s: &Stream,
    ) -> Result<Array, BixverseErrors> {
        let (k_cap, wg) = sweep_tier(k).ok_or(BixverseErrors::MlxNmfRankTooLarge {
            k,
            max: NMF_MAX_RANK,
        })?;
        let grid = n_rows.div_ceil(wg) * wg;
        let mut out = self.sweep.apply(
            &[x, g, p, eps],
            &[OutputSpec {
                shape: &[to_i32(n_rows), to_i32(k)],
                dtype: MLX_FLOAT32,
            }],
            [to_i32(grid), 1, 1],
            [to_i32(wg), 1, 1],
            &[
                ("N_ROWS", to_i32(n_rows)),
                ("K", to_i32(k)),
                ("K_CAP", to_i32(k_cap)),
                ("WG", to_i32(wg)),
            ],
            s,
        )?;
        Ok(out.remove(0))
    }

    /// Scale each column of a `[n_rows, k]` buffer by the L2 norm behind `sq`,
    /// or by its reciprocal.
    ///
    /// ### Params
    ///
    /// * `x` - Buffer `[n_rows, k]`
    /// * `sq` - Per-column sums of squares `[k]`
    /// * `n_rows` - Rows of `x`
    /// * `k` - Number of components
    /// * `invert` - Divide by the norm instead of multiplying
    /// * `s` - Stream
    ///
    /// ### Returns
    ///
    /// The lazy scaled buffer.
    fn scale(
        &self,
        x: &Array,
        sq: &Array,
        n_rows: usize,
        k: usize,
        invert: bool,
        s: &Stream,
    ) -> Result<Array, BixverseErrors> {
        let total = n_rows * k;
        let mut out = self.scale.apply(
            &[x, sq],
            &[OutputSpec {
                shape: &[to_i32(n_rows), to_i32(k)],
                dtype: MLX_FLOAT32,
            }],
            [
                to_i32(total.div_ceil(ELEMENTWISE_TG) * ELEMENTWISE_TG),
                1,
                1,
            ],
            [to_i32(ELEMENTWISE_TG), 1, 1],
            &[
                ("TOTAL", to_i32(total)),
                ("K", to_i32(k)),
                ("INVERT", invert as i32),
            ],
            s,
        )?;
        Ok(out.remove(0))
    }
}

//////////////////////
// MlxDenseNmfInput //
//////////////////////

/// Dense `V` on the MLX device, uploaded once as `V^T`.
///
/// Holds the host view alongside for the NNDSVD initialisation SVD, which stays
/// on the CPU so CPU and MLX runs start from identical factors. Also owns the
/// stream and the registered kernels, so every solve against this input reuses
/// them. MLX streams are thread affine, so this is `!Send`.
pub struct MlxDenseNmfInput<'a> {
    /// `V^T` as `[n, m]` row-major
    v_t: Array,
    /// Host-side view, for the initialisation SVD only
    host: DenseInput<'a, f32>,
    /// Logical shape `(m, n)` of `V`
    shape: (usize, usize),
    /// Cached `||V||_F^2`, accumulated in f64
    sq_frob: f32,
    /// Stream every op of every solve runs on
    stream: Stream,
    /// The custom kernels
    kernels: HalsKernels,
}

impl<'a> MlxDenseNmfInput<'a> {
    /// Validate `V`, upload it and cache its Frobenius norm.
    ///
    /// A contiguous column-major `m x n` buffer already is `[n, m]` row-major,
    /// so that case uploads without an intermediate host copy.
    ///
    /// ### Params
    ///
    /// * `v` - The matrix to factorise, `m x n`, samples by features
    ///
    /// ### Returns
    ///
    /// The input, or `NmfNonFinite` / `NmfNonNegativeViolated` if validation
    /// fails.
    pub fn new(v: MatRef<'a, f32>) -> Result<Self, BixverseErrors> {
        install_error_handler();
        let host = DenseInput::new(v)?;
        let (m, n) = (v.nrows(), v.ncols());
        let sq_frob = NmfInput::sq_frob(&host);
        let v_t = Array::from_f32(&col_major_flat(v), &[to_i32(n), to_i32(m)]);
        Ok(Self {
            v_t,
            host,
            shape: (m, n),
            sq_frob,
            stream: Stream::default_gpu(),
            kernels: HalsKernels::new(),
        })
    }

    /// Shape of `V`.
    ///
    /// ### Returns
    ///
    /// `(m, n)`, samples by features.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The squared Frobenius norm `||V||_F^2`.
    ///
    /// ### Returns
    ///
    /// The squared Frobenius norm.
    pub fn sq_frob(&self) -> f32 {
        self.sq_frob
    }

    /// `(W^T V)^T = V^T W`.
    ///
    /// ### Params
    ///
    /// * `w` - Left factor `[m, k]`
    ///
    /// ### Returns
    ///
    /// The lazy `[n, k]` product.
    fn wt_v(&self, w: &Array) -> Result<Array, BixverseErrors> {
        self.v_t.matmul(w, &self.stream)
    }

    /// `V H^T`, reading `V` as a lazy transpose of the stored `V^T`.
    ///
    /// ### Params
    ///
    /// * `h_t` - Right factor transposed, `[n, k]`
    ///
    /// ### Returns
    ///
    /// The lazy `[m, k]` product.
    fn v_ht(&self, h_t: &Array) -> Result<Array, BixverseErrors> {
        self.v_t.transpose(&self.stream)?.matmul(h_t, &self.stream)
    }

    /// Recompute both Grams and `W^T V`, then evaluate `||V - W H||_F^2`.
    ///
    /// ### Params
    ///
    /// * `w` - Left factor `[m, k]`
    /// * `h_t` - Right factor transposed `[n, k]`
    ///
    /// ### Returns
    ///
    /// The reconstruction error, clamped at zero. Blocks.
    fn objective(&self, w: &Array, h_t: &Array) -> Result<f32, BixverseErrors> {
        let s = &self.stream;
        let b = gram(w, s)?;
        let a = self.wt_v(w)?;
        let d = gram(h_t, s)?;
        let partials = row_dot_partials(&a, h_t, s)?;
        finish_objective(self.sq_frob, &partials, &b, &d)
    }

    /// Normalise the columns of `W` to unit L2 and absorb the norms into `H`.
    ///
    /// ### Params
    ///
    /// * `w` - Left factor `[m, k]`
    /// * `h_t` - Right factor transposed `[n, k]`
    /// * `k` - Number of components
    ///
    /// ### Returns
    ///
    /// The lazy pair `(W, H^T)` with `W H` unchanged.
    fn normalise(
        &self,
        w: &Array,
        h_t: &Array,
        k: usize,
    ) -> Result<(Array, Array), BixverseErrors> {
        let s = &self.stream;
        let (m, n) = self.shape;
        let sq = w.multiply(w, s)?.sum_axis(0, s)?;
        let w_new = self.kernels.scale(w, &sq, m, k, true, s)?;
        let h_new = self.kernels.scale(h_t, &sq, n, k, false, s)?;
        Ok((w_new, h_new))
    }

    /// One full HALS iteration: Gram, product, H sweep, Gram, product, W
    /// sweep, normalise.
    ///
    /// ### Params
    ///
    /// * `w` - Left factor `[m, k]`
    /// * `h_t` - Right factor transposed `[n, k]`
    /// * `eps` - Non-negativity floor, a 0-d array
    /// * `k` - Number of components
    ///
    /// ### Returns
    ///
    /// The lazy pair `(W, H^T)` after the iteration.
    fn iteration(
        &self,
        w: &Array,
        h_t: &Array,
        eps: &Array,
        k: usize,
    ) -> Result<(Array, Array), BixverseErrors> {
        let s = &self.stream;
        let (m, n) = self.shape;

        let b = gram(w, s)?;
        let a = self.wt_v(w)?;
        let h_t = self.kernels.sweep(h_t, &b, &a, eps, n, k, s)?;

        let d = gram(&h_t, s)?;
        let c = self.v_ht(&h_t)?;
        let w = self.kernels.sweep(w, &d, &c, eps, m, k, s)?;

        self.normalise(&w, &h_t, k)
    }
}

//////////////////////
// Device-side solve //
//////////////////////

/// A finished solve whose factors are still on the device.
///
/// Splits the readback from the solve so the two can be timed apart; the
/// caller normally goes straight through [`MlxNmfSolve::read`].
pub struct MlxNmfSolve {
    /// Left factor `[m, k]`
    w: Array,
    /// Right factor transposed `[n, k]`
    h_t: Array,
    /// Samples
    m: usize,
    /// Features
    n: usize,
    /// Components
    k: usize,
    /// Reconstruction loss at the last evaluated point
    final_loss: f32,
    /// Iterations performed
    n_iter: usize,
    /// Whether the tolerance was met
    converged: bool,
}

impl MlxNmfSolve {
    /// Copy the factors back into faer matrices.
    ///
    /// ### Returns
    ///
    /// The [`NmfResult`] with `W` (`m x k`) and `H` (`k x n`).
    pub fn read(self) -> Result<NmfResult<f32>, BixverseErrors> {
        eval_all(&[&self.w, &self.h_t], false)?;
        let (m, n, k) = (self.m, self.n, self.k);
        let w = self.w.as_f32()?;
        let h = self.h_t.as_f32()?;
        Ok(NmfResult {
            w: Mat::from_fn(m, k, |i, c| w[i * k + c]),
            h: Mat::from_fn(k, n, |r, j| h[j * k + r]),
            final_loss: self.final_loss,
            n_iter: self.n_iter,
            converged: self.converged,
        })
    }
}

//////////
// Main //
//////////

/// Frobenius HALS NMF on MLX, leaving the factors on the device.
///
/// Same algorithm and iteration order as
/// [`crate::methods::nmf_hals::nmf_hals`]. Initialisation runs on the host.
/// The only readback inside the loop is the objective, once per
/// `check_every` iterations, as per-row partials plus the two `[k, k]` Grams
/// finished in f64.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k` - Number of components
/// * `opts` - Solver options; see [`HalsOpts`]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// An [`MlxNmfSolve`]; call [`MlxNmfSolve::read`] for the factors.
///
/// ### Errors
///
/// * `MlxNmfRankTooLarge` if `k` is zero or above [`NMF_MAX_RANK`].
/// * `NmfRankTooLarge` if NNDSVD is requested and `V` has fewer than `k`
///   singular values.
/// * `MlxError` from the device path.
pub fn nmf_hals_mlx_device(
    v: &MlxDenseNmfInput,
    k: usize,
    opts: &HalsOpts<f32>,
    verbose: usize,
) -> Result<MlxNmfSolve, BixverseErrors> {
    check_rank(k)?;
    let (m, n) = v.shape;
    let sq_frob = v.sq_frob;
    let verbosity = parse_verbosity_level(verbose);

    let (w_host, h_host) = match &opts.init {
        NmfInit::Nndsvd => {
            let svd = NmfInput::top_k_svd(&v.host, k)?;
            nndsvd_from_svd(&svd, k, m, n)?
        }
        NmfInit::Random { seed } => random_init(m, n, k, sq_frob, *seed),
    };

    let mut w = Array::from_f32(&to_row_major(w_host.as_ref()), &[to_i32(m), to_i32(k)]);
    let mut h_t = Array::from_f32(&col_major_flat(h_host.as_ref()), &[to_i32(n), to_i32(k)]);
    let eps = Array::from_f32(&[opts.eps], &[]);

    let denom = relative_denominator(sq_frob);
    let mut final_loss = f32::INFINITY;
    let mut last_loss = f32::INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;

        let (w_next, h_next) = v.iteration(&w, &h_t, &eps, k)?;
        eval_all(&[&w_next, &h_next], true)?;
        eval_all(&[&w, &h_t], false)?;
        w = w_next;
        h_t = h_next;

        if n_iter.is_multiple_of(opts.check_every) {
            let loss = v.objective(&w, &h_t)?;
            final_loss = loss;

            if n_iter > opts.check_every {
                if verbosity.normal_verbosity() {
                    println!(
                        "  NMF MLX: Iteration {} out of {} - current loss: {:.2?}",
                        iter + 1,
                        opts.max_iter,
                        loss
                    );
                }
                if (last_loss - loss).abs() / denom < opts.tol {
                    converged = true;
                    if verbosity.normal_verbosity() {
                        println!("  NMF MLX converged successfully after {} iters", iter + 1)
                    };
                    break;
                }
            }
            last_loss = loss;
        }
    }

    // `n_iter == 0` when `max_iter` is zero, and zero is a multiple of
    // everything, so without the first clause the loss stays at infinity.
    if n_iter == 0 || !n_iter.is_multiple_of(opts.check_every) {
        final_loss = v.objective(&w, &h_t)?;
    }
    eval_all(&[&w, &h_t], false)?;
    clear_cache()?;

    Ok(MlxNmfSolve {
        w,
        h_t,
        m,
        n,
        k,
        final_loss,
        n_iter,
        converged,
    })
}

/// Frobenius HALS NMF on MLX.
///
/// [`nmf_hals_mlx_device`] followed by the readback. Results are not
/// bit-identical to the CPU path; the factors agree to f32 GEMM-ordering
/// differences.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k` - Number of components
/// * `opts` - Solver options; see [`HalsOpts`]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// An [`NmfResult`] with `W` (`m x k`), `H` (`k x n`), the final loss, the
/// iteration count and whether the tolerance was met.
pub fn nmf_hals_mlx(
    v: &MlxDenseNmfInput,
    k: usize,
    opts: &HalsOpts<f32>,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors> {
    nmf_hals_mlx_device(v, k, opts, verbose)?.read()
}

//////////////////////////
// Multiple run version //
//////////////////////////

/// Stabilised MLX NMF via random restarts.
///
/// Runs [`nmf_hals_mlx`] `n_runs` times with `base_seed + i`, reusing the
/// uploaded `V`, and column-binds the resulting `W` matrices. The `init` field
/// of `opts` is ignored and random initialisation is always used.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k` - Number of components per run
/// * `n_runs` - Number of restarts, at least one
/// * `base_seed` - Seed offset; run `i` uses `base_seed + i`
/// * `opts` - Solver options; `init` is ignored
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`StabilisedNmfResult`] with column-bound `W`, per-run `H`, per-run losses
/// and convergence flags, and the index of the lowest-loss run.
pub fn stabilised_nmf_mlx(
    v: &MlxDenseNmfInput,
    k: usize,
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<f32>,
    verbose: usize,
) -> Result<StabilisedNmfResult<f32>, BixverseErrors> {
    assert!(n_runs >= 1, "n_runs must be >= 1");
    let verbosity = parse_verbosity_level(verbose);
    let inner_verbose = verbose.saturating_sub(1);

    let mut runs: Vec<NmfResult<f32>> = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let opts_i = HalsOpts {
            init: NmfInit::Random {
                seed: base_seed + i as u64,
            },
            ..opts.clone()
        };
        runs.push(nmf_hals_mlx(v, k, &opts_i, inner_verbose)?);
        if verbosity.normal_verbosity() {
            println!(" Finished stabilised MLX NMF run: {}", i + 1);
        }
    }

    let (m, _) = v.shape;
    let w_all = Mat::<f32>::from_fn(m, k * n_runs, |row, col| runs[col / k].w[(row, col % k)]);
    let losses: Vec<f32> = runs.iter().map(|r| r.final_loss).collect();
    let converged: Vec<bool> = runs.iter().map(|r| r.converged).collect();
    let h_per_run: Vec<Mat<f32>> = runs.into_iter().map(|r| r.h).collect();
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

////////////
// Refits //
////////////

/// Refit `H` against a frozen `W`.
///
/// `W^T W` and `W^T V` are loop invariants, so `V` is touched once. `H` starts
/// at `opts.eps` and no normalisation is applied.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `w_host` - Frozen left factor `m x k`
/// * `opts` - Solver options; `init` is ignored
///
/// ### Returns
///
/// `(H of shape k x n, final ||V - W H||_F^2)`.
fn nmf_refit_h_mlx(
    v: &MlxDenseNmfInput,
    w_host: MatRef<f32>,
    opts: &HalsOpts<f32>,
) -> Result<(Mat<f32>, f32), BixverseErrors> {
    let (m, n) = v.shape;
    if w_host.nrows() != m {
        return Err(BixverseErrors::NmfDimensionMismatch {
            expected: m,
            found: w_host.nrows(),
        });
    }
    let k = w_host.ncols();
    check_rank(k)?;
    let s = &v.stream;

    let w = Array::from_f32(&to_row_major(w_host), &[to_i32(m), to_i32(k)]);
    let mut h_t = Array::from_f32(&vec![opts.eps; n * k], &[to_i32(n), to_i32(k)]);
    let eps = Array::from_f32(&[opts.eps], &[]);
    let b = gram(&w, s)?;
    let a = v.wt_v(&w)?;
    eval_all(&[&b, &a], false)?;

    let objective = |h_t: &Array| -> Result<f32, BixverseErrors> {
        let d = gram(h_t, s)?;
        let partials = row_dot_partials(&a, h_t, s)?;
        finish_objective(v.sq_frob, &partials, &b, &d)
    };

    let final_loss = refit_loop(&mut h_t, &b, &a, &eps, n, k, opts, v, objective)?;
    eval_all(&[&h_t], false)?;
    let h = h_t.as_f32()?;
    let out = Mat::from_fn(k, n, |r, j| h[j * k + r]);
    clear_cache()?;
    Ok((out, final_loss))
}

/// Refit `W` against a frozen `H`.
///
/// Mirror of [`nmf_refit_h_mlx`]: `H H^T` and `V H^T` are the invariants and
/// the objective pairs `W` against `V H^T`, so a changing `W` never forces
/// another pass over `V`.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `h_host` - Frozen right factor `k x n`
/// * `opts` - Solver options; `init` is ignored
///
/// ### Returns
///
/// `(W of shape m x k, final ||V - W H||_F^2)`.
fn nmf_refit_w_mlx(
    v: &MlxDenseNmfInput,
    h_host: MatRef<f32>,
    opts: &HalsOpts<f32>,
) -> Result<(Mat<f32>, f32), BixverseErrors> {
    let (m, n) = v.shape;
    if h_host.ncols() != n {
        return Err(BixverseErrors::NmfDimensionMismatch {
            expected: n,
            found: h_host.ncols(),
        });
    }
    let k = h_host.nrows();
    check_rank(k)?;
    let s = &v.stream;

    let h_t = Array::from_f32(&col_major_flat(h_host), &[to_i32(n), to_i32(k)]);
    let mut w = Array::from_f32(&vec![opts.eps; m * k], &[to_i32(m), to_i32(k)]);
    let eps = Array::from_f32(&[opts.eps], &[]);
    let d = gram(&h_t, s)?;
    let c = v.v_ht(&h_t)?;
    eval_all(&[&d, &c], false)?;

    let objective = |w: &Array| -> Result<f32, BixverseErrors> {
        let b = gram(w, s)?;
        let partials = row_dot_partials(w, &c, s)?;
        finish_objective(v.sq_frob, &partials, &b, &d)
    };

    let final_loss = refit_loop(&mut w, &d, &c, &eps, m, k, opts, v, objective)?;
    eval_all(&[&w], false)?;
    let wf = w.as_f32()?;
    let out = Mat::from_fn(m, k, |i, c| wf[i * k + c]);
    clear_cache()?;
    Ok((out, final_loss))
}

/// Shared loop of the two refits: sweep the free factor against fixed `G` and
/// `P` until the objective stalls.
///
/// ### Params
///
/// * `x` - Free factor `[n_rows, k]`, replaced by the result
/// * `g` - Fixed Gram `[k, k]`
/// * `p` - Fixed data product `[n_rows, k]`
/// * `eps` - Non-negativity floor, a 0-d array
/// * `n_rows` - Rows of `x`
/// * `k` - Number of components
/// * `opts` - Solver options
/// * `v` - Device-resident `V`, for the stream, kernels and norm
/// * `objective` - Objective of a candidate `x`
///
/// ### Returns
///
/// The final reconstruction error.
#[allow(clippy::too_many_arguments)]
fn refit_loop(
    x: &mut Array,
    g: &Array,
    p: &Array,
    eps: &Array,
    n_rows: usize,
    k: usize,
    opts: &HalsOpts<f32>,
    v: &MlxDenseNmfInput,
    objective: impl Fn(&Array) -> Result<f32, BixverseErrors>,
) -> Result<f32, BixverseErrors> {
    let denom = relative_denominator(v.sq_frob);
    let mut final_loss = f32::INFINITY;
    let mut last_loss = f32::INFINITY;
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;
        let next = v.kernels.sweep(x, g, p, eps, n_rows, k, &v.stream)?;
        eval_all(&[&next], true)?;
        eval_all(&[x], false)?;
        *x = next;

        if n_iter.is_multiple_of(opts.check_every) {
            let loss = objective(x)?;
            final_loss = loss;
            if n_iter > opts.check_every && (last_loss - loss).abs() / denom < opts.tol {
                break;
            }
            last_loss = loss;
        }
    }

    if n_iter == 0 || !n_iter.is_multiple_of(opts.check_every) {
        final_loss = objective(x)?;
    }
    Ok(final_loss)
}

///////////////
// Consensus //
///////////////

/// Consensus NMF on MLX.
///
/// Restarts through [`stabilised_nmf_mlx`], the shared host-side
/// [`consensus_from_restarts`], then a refit of the partner factor against the
/// consensus. Same structure and reported quantities as
/// [`crate::methods::nmf_hals::consensus::nmf_consensus`].
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k` - Number of components, at least two
/// * `n_runs` - Number of restarts, at least two
/// * `base_seed` - Seed offset; run `i` uses `base_seed + i`
/// * `opts` - Solver options; the `init` field is ignored by the restarts
/// * `params` - Optional [`ConsensusParams`], defaulted if `None`
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`ConsensusNmfResult`]. `error` and `run_errors` are relative to
/// `||V||_F^2`.
///
/// ### Errors
///
/// * `NmfConsensusInvalidK` if `k < 2`, `NmfConsensusTooFewRuns` if
///   `n_runs < 2`, `NmfConsensusEmptyCluster` if a cluster came out empty.
/// * Anything the solve or the consensus step raises.
pub fn nmf_consensus_mlx(
    v: &MlxDenseNmfInput,
    k: usize,
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<f32>,
    params: Option<ConsensusParams<f32>>,
    verbose: usize,
) -> Result<ConsensusNmfResult<f32>, BixverseErrors> {
    let params = params.unwrap_or_default();
    if k < 2 {
        return Err(BixverseErrors::NmfConsensusInvalidK { k });
    }
    if n_runs < 2 {
        return Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs });
    }

    let restarts = stabilised_nmf_mlx(v, k, n_runs, base_seed, opts, verbose)?;
    let clusters = consensus_from_restarts(&restarts, k, &params, verbose)?;
    if let Some(cluster) = clusters.sizes.iter().position(|&s| s == 0) {
        return Err(BixverseErrors::NmfConsensusEmptyCluster { cluster, k });
    }

    let (w, h, loss) = match params.target {
        ConsensusTarget::HRows => {
            let h = clusters.consensus.clone();
            let (w, loss) = nmf_refit_w_mlx(v, h.as_ref(), opts)?;
            (w, h, loss)
        }
        ConsensusTarget::WColumns => {
            let w = clusters.consensus.transpose().to_owned();
            let (h, loss) = nmf_refit_h_mlx(v, w.as_ref(), opts)?;
            (w, h, loss)
        }
    };

    let denom = if v.sq_frob > 0.0 { v.sq_frob } else { 1.0 };
    let run_errors: Vec<f32> = restarts.losses.iter().map(|&l| l / denom).collect();

    Ok(ConsensusNmfResult {
        w,
        h,
        error: loss / denom,
        clusters,
        run_errors,
    })
}

//////////////////
// Entry points //
//////////////////

/// Parse the pre-processing string, warning and defaulting on unknown input,
/// as the CPU and cubecl entry points do.
///
/// ### Params
///
/// * `preprocessing` - String to parse, see [`parse_nmf_processing`]
///
/// ### Returns
///
/// The [`NmfPreprocessing`] to apply.
fn resolve_preprocessing(preprocessing: &str) -> NmfPreprocessing {
    parse_nmf_processing(preprocessing).unwrap_or_else(|| {
        println!(
            "Unknown string provided: {:?}. Using the default (No additional pre-processing)",
            preprocessing
        );
        NmfPreprocessing::default()
    })
}

/// Run MLX NMF on a dense matrix (single run).
///
/// ### Params
///
/// * `data` - The matrix to factorise, samples x features
/// * `k` - Number of components
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// An [`NmfResult`], or errors from validation or the device path.
pub fn nmf_single_run_mlx(
    data: MatRef<f32>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<f32>>,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let input = MlxDenseNmfInput::new(processed.as_ref())?;
    nmf_hals_mlx(&input, k, &opts, verbose)
}

/// Run MLX consensus NMF on a dense matrix.
///
/// ### Params
///
/// * `data` - The matrix to factorise, samples x features
/// * `k` - Number of components, at least two
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `consensus_params` - Optional [`ConsensusParams`]
/// * `n_runs` - Number of restarts, at least two
/// * `base_seed` - Seed offset
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`ConsensusNmfResult`], or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_run_mlx(
    data: MatRef<f32>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<f32>>,
    consensus_params: Option<ConsensusParams<f32>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<ConsensusNmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let input = MlxDenseNmfInput::new(processed.as_ref())?;
    nmf_consensus_mlx(
        &input,
        k,
        n_runs,
        base_seed as u64,
        &opts,
        consensus_params,
        verbose,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::nmf_hals::{
        compute_objective, gram_h_ht, gram_wt_w, hals_sweep_cols, nmf_hals,
    };

    /// Deterministic non-negative `m x n` matrix with an exact rank-`k` core.
    fn build_v(m: usize, n: usize, k: usize) -> Mat<f32> {
        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 13 + c * 29) % 17) as f32) * 0.1 + 0.05);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 31 + j * 11) % 19) as f32) * 0.1 + 0.05);
        w * h
    }

    /// Fixed `W` (`m x k`) and `H` (`k x n`) for the product tests.
    fn build_factors(m: usize, n: usize, k: usize) -> (Mat<f32>, Mat<f32>) {
        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 5 + c * 3) % 7) as f32) * 0.2 + 0.1);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 7 + j * 5) % 9) as f32) * 0.2 + 0.1);
        (w, h)
    }

    /// Upload `W` as `[m, k]` and `H` as `[n, k]`.
    fn upload(w: &Mat<f32>, h: &Mat<f32>) -> (Array, Array) {
        let (m, k) = (w.nrows(), w.ncols());
        let n = h.ncols();
        (
            Array::from_f32(&to_row_major(w.as_ref()), &[m as i32, k as i32]),
            Array::from_f32(&col_major_flat(h.as_ref()), &[n as i32, k as i32]),
        )
    }

    /// Evaluate and copy an array out.
    fn read(x: &Array) -> Vec<f32> {
        eval_all(&[x], false).unwrap();
        x.as_f32().unwrap().to_vec()
    }

    /// Relative comparison floored at one, with the all-zeros guard.
    fn assert_close(got: &[f32], want: &[f32], tol: f32) -> f32 {
        assert_eq!(got.len(), want.len(), "length mismatch");
        assert!(
            got.iter().any(|v| v.abs() > 1e-6),
            "output is all zeros, the device did no work"
        );
        let mut worst = 0f32;
        for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
            let rel = (a - b).abs() / b.abs().max(1.0);
            worst = worst.max(rel);
            assert!(rel <= tol, "index {i}: got {a} want {b}");
        }
        worst
    }

    #[test]
    fn test_dense_products_match_cpu() {
        let (m, n, k) = (43usize, 29usize, 6usize);
        let v = build_v(m, n, k);
        let cpu = DenseInput::new(v.as_ref()).unwrap();
        let mlx = MlxDenseNmfInput::new(v.as_ref()).unwrap();
        assert_eq!(mlx.shape(), NmfInput::shape(&cpu));

        let (w, h) = build_factors(m, n, k);
        let (w_d, h_d) = upload(&w, &h);
        let a = mlx.wt_v(&w_d).unwrap();
        let c = mlx.v_ht(&h_d).unwrap();

        let mut wtv = Mat::<f32>::zeros(k, n);
        let mut vht = Mat::<f32>::zeros(m, k);
        cpu.wt_v(w.as_ref(), &mut wtv);
        cpu.v_ht(h.as_ref(), &mut vht);

        let d1 = assert_close(&read(&a), &to_row_major(wtv.transpose()), 1e-4);
        let d2 = assert_close(&read(&c), &to_row_major(vht.as_ref()), 1e-4);
        println!("products max rel diff: {d1:e} {d2:e}");
    }

    #[test]
    fn test_grams_match_cpu() {
        let (m, n, k) = (97usize, 61usize, 8usize);
        let (w, h) = build_factors(m, n, k);
        let (w_d, h_d) = upload(&w, &h);
        let s = Stream::default_gpu();

        let mut wtw = Mat::<f32>::zeros(k, k);
        let mut hht = Mat::<f32>::zeros(k, k);
        gram_wt_w(w.as_ref(), &mut wtw);
        gram_h_ht(h.as_ref(), &mut hht);

        let d1 = assert_close(
            &read(&gram(&w_d, &s).unwrap()),
            &to_row_major(wtw.as_ref()),
            1e-4,
        );
        let d2 = assert_close(
            &read(&gram(&h_d, &s).unwrap()),
            &to_row_major(hht.as_ref()),
            1e-4,
        );
        println!("grams max rel diff: {d1:e} {d2:e}");
    }

    // The sweep against the CPU W sweep at every tier, both edges of the
    // smallest tier and the largest rank. More rows than one threadgroup so
    // the row guard and the threadgroup offset are both exercised.
    #[test]
    fn test_sweep_matches_cpu_across_ranks() {
        let s = Stream::default_gpu();
        let kernels = HalsKernels::new();
        let rows = 300usize;
        let eps = 1e-10f32;
        for k in [1usize, 5, 16, 17, 33, 64, 65, 128] {
            let x = Mat::<f32>::from_fn(rows, k, |i, c| {
                (((i * 7 + c * 13) % 11) as f32) * 0.1 + 0.05
            });
            let y =
                Mat::<f32>::from_fn(rows, k, |i, c| (((i * 3 + c * 5) % 13) as f32) * 0.1 + 0.02);
            let mut g = Mat::<f32>::zeros(k, k);
            gram_wt_w(y.as_ref(), &mut g);
            let p = Mat::<f32>::from_fn(rows, k, |i, c| (((i * 11 + c * 2) % 7) as f32) * 0.5);

            let mut want = x.clone();
            hals_sweep_cols(&mut want, g.as_ref(), p.as_ref(), eps);

            let x_d = Array::from_f32(&to_row_major(x.as_ref()), &[rows as i32, k as i32]);
            let g_d = Array::from_f32(&to_row_major(g.as_ref()), &[k as i32, k as i32]);
            let p_d = Array::from_f32(&to_row_major(p.as_ref()), &[rows as i32, k as i32]);
            let eps_d = Array::from_f32(&[eps], &[]);
            let got = kernels
                .sweep(&x_d, &g_d, &p_d, &eps_d, rows, k, &s)
                .unwrap();
            let d = assert_close(&read(&got), &to_row_major(want.as_ref()), 1e-4);
            println!("sweep k = {k}: max rel diff {d:e}");
        }
    }

    #[test]
    fn test_objective_matches_cpu() {
        let (m, n, k) = (53usize, 41usize, 6usize);
        let v = build_v(m, n, k);
        let cpu = DenseInput::new(v.as_ref()).unwrap();
        let mlx = MlxDenseNmfInput::new(v.as_ref()).unwrap();

        let (w, h) = build_factors(m, n, k);
        let (w_d, h_d) = upload(&w, &h);
        let got = mlx.objective(&w_d, &h_d).unwrap();

        let mut wtv = Mat::<f32>::zeros(k, n);
        let mut wtw = Mat::<f32>::zeros(k, k);
        let mut hht = Mat::<f32>::zeros(k, k);
        cpu.wt_v(w.as_ref(), &mut wtv);
        gram_wt_w(w.as_ref(), &mut wtw);
        gram_h_ht(h.as_ref(), &mut hht);
        let want = compute_objective(
            NmfInput::sq_frob(&cpu),
            h.as_ref(),
            wtv.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );

        assert!(got > 0.0, "objective is zero, the device did no work");
        let rel = (got - want).abs() / want.abs().max(1.0);
        println!("objective rel diff: {rel:e}");
        assert!(rel <= 1e-3, "got {got} want {want}");
    }

    /// One CPU and one MLX solve from the same random initialisation.
    fn solve_pair(
        m: usize,
        n: usize,
        k: usize,
        max_iter: usize,
    ) -> (NmfResult<f32>, NmfResult<f32>, f32) {
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter,
            tol: 1e-9,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Random { seed: 7 },
        };
        let cpu_in = DenseInput::new(v.as_ref()).unwrap();
        let cpu = nmf_hals(&cpu_in, k, &opts, 0).unwrap();
        let mlx_in = MlxDenseNmfInput::new(v.as_ref()).unwrap();
        let mlx = nmf_hals_mlx(&mlx_in, k, &opts, 0).unwrap();
        (cpu, mlx, NmfInput::sq_frob(&cpu_in))
    }

    // Pins the whole loop order: Gram, product, sweep, Gram, product, sweep,
    // normalise.
    #[test]
    fn test_one_iteration_matches_cpu_elementwise() {
        let (cpu, mlx, _) = solve_pair(47, 31, 5, 1);
        assert_eq!(cpu.n_iter, 1);
        assert_eq!(mlx.n_iter, 1);
        let dw = assert_close(
            &to_row_major(mlx.w.as_ref()),
            &to_row_major(cpu.w.as_ref()),
            1e-4,
        );
        let dh = assert_close(
            &to_row_major(mlx.h.as_ref()),
            &to_row_major(cpu.h.as_ref()),
            1e-4,
        );
        println!("one iteration max rel diff: W {dw:e} H {dh:e}");
    }

    #[test]
    fn test_converged_loss_matches_cpu() {
        let (cpu, mlx, sq_frob) = solve_pair(47, 31, 5, 300);
        let cpu_rel = cpu.final_loss / sq_frob;
        let mlx_rel = mlx.final_loss / sq_frob;
        println!("converged relative losses: cpu {cpu_rel:e} mlx {mlx_rel:e}");
        assert!(
            (cpu_rel - mlx_rel).abs() < 1e-3,
            "relative losses disagree: cpu {cpu_rel} mlx {mlx_rel}"
        );
    }

    #[test]
    fn test_recovers_rank_k_dense() {
        let (m, n, k) = (60usize, 40usize, 3usize);
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter: 400,
            tol: 1e-9,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Nndsvd,
        };
        let input = MlxDenseNmfInput::new(v.as_ref()).unwrap();
        let res = nmf_hals_mlx(&input, k, &opts, 0).unwrap();

        let rel = res.final_loss / input.sq_frob();
        println!("rank-k recovery relative loss: {rel:e}");
        assert!(rel < 1e-3, "relative loss {rel} is too high");
        assert!((0..m).all(|i| (0..k).all(|c| res.w[(i, c)] >= 0.0)));
        assert!((0..k).all(|r| (0..n).all(|j| res.h[(r, j)] >= 0.0)));
    }

    #[test]
    fn test_rejects_out_of_range_ranks() {
        let v = build_v(20, 10, 2);
        let input = MlxDenseNmfInput::new(v.as_ref()).unwrap();
        for k in [0usize, NMF_MAX_RANK + 1] {
            assert!(matches!(
                nmf_hals_mlx(&input, k, &HalsOpts::default(), 0),
                Err(BixverseErrors::MlxNmfRankTooLarge { .. })
            ));
        }
    }

    #[test]
    fn test_zero_iterations_reports_a_finite_loss() {
        let (m, n, k) = (30usize, 20usize, 3usize);
        let v = build_v(m, n, k);
        let input = MlxDenseNmfInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f32> {
            max_iter: 0,
            init: NmfInit::Random { seed: 1 },
            ..HalsOpts::default()
        };
        let res = nmf_hals_mlx(&input, k, &opts, 0).unwrap();
        assert_eq!(res.n_iter, 0);
        assert!(res.final_loss.is_finite());
    }

    /// Exact rank-k product with clearly separated components.
    fn rank_k_input(m: usize, n: usize, k: usize) -> Mat<f32> {
        let w = Mat::<f32>::from_fn(m, k, |i, j| if i % k == j { 5.0 } else { 0.3 });
        let h = Mat::<f32>::from_fn(k, n, |i, j| if j % k == i { 4.0 } else { 0.2 });
        w * h
    }

    // Both refit arms through the consensus entry point.
    #[test]
    fn test_consensus_returns_usable_factors() {
        let (m, n, k, n_runs) = (25usize, 18usize, 2usize, 4usize);
        let v = rank_k_input(m, n, k);
        let opts = HalsOpts::<f32>::new(200, 1e-7, 1e-10, 10, NmfInit::Nndsvd);
        for target in [ConsensusTarget::HRows, ConsensusTarget::WColumns] {
            let params = ConsensusParams::<f32>::new(target, None, None, 100, 3, 42);
            let res = nmf_consensus_run_mlx(
                v.as_ref(),
                k,
                "none",
                Some(opts.clone()),
                Some(params),
                n_runs,
                0,
                0,
            )
            .unwrap();
            assert_eq!((res.w.nrows(), res.w.ncols()), (m, k));
            assert_eq!((res.h.nrows(), res.h.ncols()), (k, n));
            assert_eq!(res.run_errors.len(), n_runs);
            assert!(
                res.error < 1e-4,
                "consensus error {} is too high",
                res.error
            );
        }
    }

    #[test]
    fn test_single_run_entry_point() {
        let v = build_v(40, 30, 3);
        let res = nmf_single_run_mlx(v.as_ref(), 3, "none", None, 0).unwrap();
        assert_eq!((res.w.nrows(), res.w.ncols()), (40, 3));
        assert_eq!((res.h.nrows(), res.h.ncols()), (3, 30));
        assert!(res.final_loss.is_finite());
    }
}
