//! Randomised SVD of a sparse matrix on MLX, with implicit mean centring,
//! column scaling and optional row offsets.
//!
//! Mirrors [`bixverse_core::core::math::pca_svd::randomised_sparse_svd`] and the
//! cubecl `randomised_sparse_svd_gpu` step for step:
//!
//! 1. Host: CSR from the CSC (`transform_single_layer`), `Omega` drawn
//!    standard normal and pre-divided by `sigma`. Both sparse orientations go
//!    up once; the CSC is the CSR of `A^T`.
//! 2. `Y = A Omega - 1 c^T - r x_sum^T`, the SpMM kernel in
//!    [`super::spmm`], with `c = mu^T Omega` and `x_sum = 1^T Omega` as MLX
//!    GEMVs and reductions.
//! 3. CholeskyQR2: Gram `Y^T Y` by `matmul`, host f64 Cholesky and `R^-1`,
//!    then `Y R^-1` by `matmul`, twice.
//! 4. Power iterations: `Z = (A^T Q - mu q_sum^T - 1 (r^T Q)) / sigma^2`,
//!    i.e. `D^-1 X^T Q`, then step 2 on `Z` and CholeskyQR2 again.
//! 5. Final `Z` as in step 4 but over `sigma`, which is `B^T` for
//!    `B = Q^T X`. Host thin SVD of the small `[m, s]` `Z` in f64, then
//!    `U = Q V_z` by `matmul`.
//!
//! f32 on the device. Every GEMM keeps the long `n` (or `m`) axis as the
//! outer axis of both operands, with a lazy transpose on the left only.

use std::time::{Duration, Instant};

use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Mat, Side};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::ffi::{Array, MetalKernel, Stream, clear_cache, eval_all, install_error_handler};
use crate::linalg::spmm::{MlxCsr, spmm_affine, spmm_kernel};
use crate::prelude::*;
use bixverse_core::core::math::pca_svd::SvdResults;
use bixverse_core::core::math::{DEFAULT_N_POWER_ITERS_RAND_SVD, DEFAULT_OVERSAMPLING_RAND_SVD};
use bixverse_core::utils::faer_parallelism;

////////////
// Params //
////////////

/// Parameters of [`randomised_sparse_svd_mlx`]. The MLX twin of the cubecl
/// `RandSvdGpuParams`, which lives behind the `gpu` feature.
#[derive(Clone, Copy, Debug)]
pub struct RandSvdMlxParams {
    /// Number of power iterations. The single-cell callers use
    /// [`bixverse_core::core::math::N_POWER_ITERS_SINGLE_CELL`].
    pub n_power_iters: usize,
    /// Extra columns sampled on top of `n_components` to stabilise the
    /// estimate of the leading singular subspace.
    pub oversampling: usize,
}

impl RandSvdMlxParams {
    /// New params.
    ///
    /// ### Params
    ///
    /// * `n_power_iters` - Number of power iterations
    /// * `oversampling` - Extra sketch columns
    ///
    /// ### Returns
    ///
    /// The params
    pub fn new(n_power_iters: usize, oversampling: usize) -> Self {
        Self {
            n_power_iters,
            oversampling,
        }
    }
}

impl Default for RandSvdMlxParams {
    fn default() -> Self {
        Self {
            n_power_iters: DEFAULT_N_POWER_ITERS_RAND_SVD,
            oversampling: DEFAULT_OVERSAMPLING_RAND_SVD,
        }
    }
}

/// Wall time of each stage of one [`randomised_sparse_svd_mlx_staged`] call.
/// Stages accumulate over the power iterations.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxRandSvdStages {
    /// Host side: f32 cast, CSR build, `Omega` draw
    pub prep: Duration,
    /// Hand-over to MLX: the CSR and CSC go across without a copy, the small
    /// vectors are copied
    pub upload: Duration,
    /// Forward SpMM `A X` with its correction vectors
    pub spmm_forward: Duration,
    /// Transpose SpMM `A^T Q` with its correction vectors
    pub spmm_transpose: Duration,
    /// CholeskyQR2, host Cholesky included
    pub qr: Duration,
    /// `Z` readback, host thin SVD, `U = Q V_z` and its readback
    pub finalise: Duration,
}

impl MlxRandSvdStages {
    /// Sum of all stages.
    ///
    /// ### Returns
    ///
    /// The end-to-end wall time
    pub fn total(&self) -> Duration {
        self.prep + self.upload + self.spmm_forward + self.spmm_transpose + self.qr + self.finalise
    }
}

/////////////////
// CholeskyQR2 //
/////////////////

/// `R^-1` from the Gram matrix `G = Y^T Y = R^T R`, in f64 on the host.
///
/// ### Params
///
/// * `g` - Row-major `G`, length `s * s`
/// * `s` - Side length of `G`
///
/// ### Returns
///
/// Row-major upper-triangular `R^-1`, cast to f32, or `FaerCholeskyError` if
/// `G` is not SPD.
fn r_inverse_from_gram(g: &[f32], s: usize) -> Result<Vec<f32>, MlxErrors> {
    let g = Mat::<f64>::from_fn(s, s, |i, j| g[i * s + j] as f64);
    let chol = g.llt(Side::Lower).map_err(CoreErrors::from)?;
    let r = chol.L().transpose().to_owned();
    let mut r_inv = Mat::<f64>::identity(s, s);
    solve_upper_triangular_in_place(r.as_ref(), r_inv.as_mut(), faer_parallelism());
    Ok((0..s * s).map(|k| r_inv[(k / s, k % s)] as f32).collect())
}

/// One CholeskyQR pass: `Q = Y R^-1` with `R^T R = Y^T Y`. Blocks on the
/// Gram, which the host Cholesky needs.
///
/// ### Params
///
/// * `y` - Tall-skinny input `[n, s]`, row-major
/// * `s_width` - Number of columns
/// * `st` - Stream to run on
///
/// ### Returns
///
/// The (lazy) `Q` `[n, s]`
fn cholesky_qr_pass(y: &Array, s_width: usize, st: &Stream) -> Result<Array, MlxErrors> {
    let g = y.transpose(st)?.matmul(y, st)?;
    eval_all(&[&g], false)?;
    let r_inv = r_inverse_from_gram(g.as_f32()?, s_width)?;
    let r_inv = Array::from_f32(&r_inv, &[s_width as i32, s_width as i32]);
    y.matmul(&r_inv, st)
}

/// CholeskyQR2: two passes, the second recovers the precision the first
/// loses to the squared condition number of `Y^T Y`.
///
/// ### Params
///
/// * `y` - Tall-skinny input `[n, s]`, row-major, full column rank
/// * `s_width` - Number of columns
/// * `st` - Stream to run on
///
/// ### Returns
///
/// The (lazy) orthonormal `Q` `[n, s]`, or `FaerCholeskyError` if a Gram
/// matrix is not SPD.
///
/// ### References
///
/// Fukaya et al., ScalA, 2014
fn cholesky_qr2(y: &Array, s_width: usize, st: &Stream) -> Result<Array, MlxErrors> {
    let q1 = cholesky_qr_pass(y, s_width, st)?;
    cholesky_qr_pass(&q1, s_width, st)
}

/////////////
// Helpers //
/////////////

/// Close one stage: with `staged`, block on `arrays` first so the time lands
/// on the stage that queued the work.
///
/// ### Params
///
/// * `arrays` - Outputs of the stage
/// * `staged` - Block on them before reading the clock
/// * `clock` - Start of the stage, reset to now
/// * `slot` - Stage total to add to
///
/// ### Returns
///
/// `Ok(())`, or `MlxError` if the eval fails
fn lap(
    arrays: &[&Array],
    staged: bool,
    clock: &mut Instant,
    slot: &mut Duration,
) -> Result<(), MlxErrors> {
    if staged {
        eval_all(arrays, false)?;
    }
    *slot += clock.elapsed();
    *clock = Instant::now();
    Ok(())
}

/// Device state shared by the SpMM calls of one run.
struct Operator {
    /// The SpMM kernel
    kernel: MetalKernel,
    /// CSR of A `[n, m]`
    csr: MlxCsr,
    /// CSC of A, i.e. the CSR of `A^T` `[m, n]`
    csc: MlxCsr,
    /// Column means `[m, 1]`
    mu: Array,
    /// Row offsets `[n, 1]`, zeros without CLR
    offsets: Array,
    /// Ones `[n]`, the forward divisor
    ones_n: Array,
    /// Zeros `[s]`, the correction vectors without CLR
    zeros_s: Array,
    /// Whether row offsets are in play
    use_clr: bool,
    /// Sketch width
    s_width: usize,
}

impl Operator {
    /// `Y = A X - 1 (mu^T X) - r (1^T X)^T`.
    ///
    /// ### Params
    ///
    /// * `x` - Dense operand `[m, s]`, pre-divided by `sigma`
    /// * `st` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The (lazy) `[n, s]` product
    fn forward(&self, x: &Array, st: &Stream) -> Result<Array, MlxErrors> {
        let c = self.mu.transpose(st)?.matmul(x, st)?;
        let x_sum = if self.use_clr {
            Some(x.sum_axis(0, st)?)
        } else {
            None
        };
        spmm_affine(
            &self.kernel,
            &self.csr,
            x,
            &c,
            &self.offsets,
            x_sum.as_ref().unwrap_or(&self.zeros_s),
            &self.ones_n,
            self.s_width,
            st,
        )
    }

    /// `Z = (A^T Q - mu (1^T Q) - 1 (r^T Q)) / div`.
    ///
    /// ### Params
    ///
    /// * `q` - Dense operand `[n, s]`
    /// * `div` - Per-gene divisor `[m]`, `sigma` or `sigma^2`
    /// * `st` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The (lazy) `[m, s]` product
    fn transpose(&self, q: &Array, div: &Array, st: &Stream) -> Result<Array, MlxErrors> {
        let q_sum = q.sum_axis(0, st)?;
        let m_dot_q = if self.use_clr {
            Some(self.offsets.transpose(st)?.matmul(q, st)?)
        } else {
            None
        };
        spmm_affine(
            &self.kernel,
            &self.csc,
            q,
            m_dot_q.as_ref().unwrap_or(&self.zeros_s),
            &self.mu,
            &q_sum,
            div,
            self.s_width,
            st,
        )
    }
}

//////////
// Main //
//////////

/// Shared body of [`randomised_sparse_svd_mlx`] and
/// [`randomised_sparse_svd_mlx_staged`].
///
/// ### Params
///
/// * `data` - CSC of A `(n, m)` with `data_2` populated
/// * `col_means` - Column means `[m]`
/// * `col_stds` - Column standard deviations `[m]`, floored > 0
/// * `row_offsets` - Optional per-row offsets `[n]` (CLR)
/// * `n_components` - Number of leading singular triples
/// * `params` - Optional [`RandSvdMlxParams`]
/// * `seed` - Seed for the random projection
/// * `staged` - Block at every stage boundary for the stage split
///
/// ### Returns
///
/// The SVD and the stage timings (only meaningful with `staged`)
#[allow(clippy::too_many_arguments)]
fn rand_svd_impl<T: BixverseFloat>(
    data: CompressedSparseData2<T, T>,
    col_means: &[T],
    col_stds: &[T],
    row_offsets: Option<&[T]>,
    n_components: usize,
    params: Option<RandSvdMlxParams>,
    seed: u64,
    staged: bool,
) -> Result<(SvdResults<T>, MlxRandSvdStages), MlxErrors> {
    let params = params.unwrap_or_default();
    if !data.cs_type.is_csc() {
        return Err(MlxErrors::Core(CoreErrors::SparseLayoutMismatch {
            expected: CompressedSparseFormat::Csc,
            got: data.cs_type,
        }));
    }
    let (n, m) = data.shape;
    if col_means.len() != m || col_stds.len() != m {
        return Err(CoreErrors::DimensionMisMatchSparse {
            indices_len: col_means.len(),
            data_len: m,
        }
        .into());
    }
    if let Some(off) = row_offsets
        && off.len() != n
    {
        return Err(CoreErrors::DimensionMisMatchSparse {
            indices_len: off.len(),
            data_len: n,
        }
        .into());
    }
    // Clamped as on the CPU: past min(n, m) the sketch is rank deficient and
    // the Cholesky of its Gram fails.
    let s_width = (n_components + params.oversampling).min(n).min(m);
    if n_components == 0 || n_components > s_width {
        return Err(CoreErrors::InvalidArgument(format!(
            "MLX randomised SVD: n_components = {n_components} must be in 1..={}",
            n.min(m)
        ))
        .into());
    }
    let use_clr = row_offsets.is_some();
    let to_f32 = |x: &T| x.to_f32().expect("finite float casts to f32");

    let mut st_times = MlxRandSvdStages::default();
    let mut clock = Instant::now();
    install_error_handler();
    let st = Stream::default_gpu();

    // -- prep --
    let values: Vec<f32> = data
        .data_2
        .as_ref()
        .ok_or(CoreErrors::Data2NotAvailable)?
        .iter()
        .map(to_f32)
        .collect();
    let csc = CompressedSparseData2::<f32, f32>::from_parts(
        Vec::new(),
        data.indices,
        data.indptr,
        Some(values),
        CompressedSparseFormat::Csc,
        (n, m),
    );
    let csr = csc.transform_single_layer(true)?;
    let mu: Vec<f32> = col_means.iter().map(to_f32).collect();
    let sigma: Vec<f32> = col_stds.iter().map(to_f32).collect();
    // The loop divides by sigma^2, not sigma: the forward SpMM wants its
    // operand pre-divided by sigma, so `D^-1 X^T Q` is what it must receive.
    let sigma_sq: Vec<f32> = sigma.iter().map(|x| x * x).collect();
    let offsets: Vec<f32> = match row_offsets {
        Some(off) => off.iter().map(to_f32).collect(),
        None => vec![0.0; n],
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("unit normal is always valid");
    let omega: Vec<f32> = (0..m * s_width)
        .map(|i| (normal.sample(&mut rng) / sigma[i / s_width] as f64) as f32)
        .collect();
    lap(&[], false, &mut clock, &mut st_times.prep)?;

    // -- upload --
    let (n_i, m_i, s_i) = (n as i32, m as i32, s_width as i32);
    let op = Operator {
        kernel: spmm_kernel(),
        csr: MlxCsr::from_vecs(csr.indptr, csr.indices, csr.data_2.unwrap_or_default(), n)?,
        csc: MlxCsr::from_vecs(csc.indptr, csc.indices, csc.data_2.unwrap_or_default(), m)?,
        mu: Array::from_f32(&mu, &[m_i, 1]),
        offsets: Array::from_f32(&offsets, &[n_i, 1]),
        ones_n: Array::from_f32(&vec![1.0; n], &[n_i]),
        zeros_s: Array::from_f32(&vec![0.0; s_width], &[s_i]),
        use_clr,
        s_width,
    };
    let sigma_dev = Array::from_f32(&sigma, &[m_i]);
    let sigma_sq_dev = Array::from_f32(&sigma_sq, &[m_i]);
    let omega_dev = Array::from_f32(&omega, &[m_i, s_i]);
    lap(&[], false, &mut clock, &mut st_times.upload)?;

    // -- initial sketch --
    let y = op.forward(&omega_dev, &st)?;
    lap(&[&y], staged, &mut clock, &mut st_times.spmm_forward)?;
    let mut q = cholesky_qr2(&y, s_width, &st)?;
    lap(&[&q], staged, &mut clock, &mut st_times.qr)?;

    // -- power iterations --
    for _ in 0..params.n_power_iters {
        let z = op.transpose(&q, &sigma_sq_dev, &st)?;
        lap(&[&z], staged, &mut clock, &mut st_times.spmm_transpose)?;
        let y = op.forward(&z, &st)?;
        lap(&[&y], staged, &mut clock, &mut st_times.spmm_forward)?;
        q = cholesky_qr2(&y, s_width, &st)?;
        lap(&[&q], staged, &mut clock, &mut st_times.qr)?;
    }

    // -- final Z --
    let z = op.transpose(&q, &sigma_dev, &st)?;
    lap(&[&z], staged, &mut clock, &mut st_times.spmm_transpose)?;

    // -- finalise --
    eval_all(&[&z], false)?;
    let z_host = z.as_f32()?;
    let svd = Mat::<f64>::from_fn(m, s_width, |i, j| z_host[i * s_width + j] as f64)
        .thin_svd()
        .map_err(|e| CoreErrors::FaerSvdError(format!("{e:?}")))?;
    let v_z = svd.V();
    let v_z_trunc: Vec<f32> = (0..s_width * n_components)
        .map(|k| v_z[(k / n_components, k % n_components)] as f32)
        .collect();
    let v_z_dev = Array::from_f32(&v_z_trunc, &[s_i, n_components as i32]);
    let u = q.matmul(&v_z_dev, &st)?;
    eval_all(&[&u], false)?;
    let u_host = u.as_f32()?;

    let from = |x: f64| T::from_f64(x).expect("f64 casts to the crate float");
    let res = SvdResults {
        u: Mat::from_fn(n, n_components, |i, j| {
            from(u_host[i * n_components + j] as f64)
        }),
        s: (0..n_components).map(|i| from(svd.S()[i])).collect(),
        v: Mat::from_fn(m, n_components, |i, j| from(svd.U()[(i, j)])),
    };
    drop((u, v_z_dev, z, q, y, omega_dev, sigma_dev, sigma_sq_dev, op));
    clear_cache()?;
    lap(&[], false, &mut clock, &mut st_times.finalise)?;
    Ok((res, st_times))
}

/// Randomised SVD of a sparse matrix on MLX with implicit mean centring and
/// column scaling, and optional row offsets (CLR).
///
/// `data` must be a CSC of shape `(n, m)` with the `data_2` layer populated
/// (the normalised counts in the single-cell pipeline). `col_means` and
/// `col_stds` are precomputed against `data_2`, length `m`; `col_stds` must
/// be strictly positive (floor on the host upstream). The sketch width
/// `n_components + oversampling` is clamped to `min(n, m)`, as on the CPU.
///
/// ### Params
///
/// * `data` - CSC of A `(n, m)`, with `data_2` containing the values to
///   factorise
/// * `col_means` - Column means `[m]`
/// * `col_stds` - Column standard deviations `[m]`, floored > 0
/// * `row_offsets` - Optional per-row offsets `[n]`, subtracted from every
///   entry of the row
/// * `n_components` - Number of leading singular triples to return
/// * `params` - Optional [`RandSvdMlxParams`]; defaults to 2 power iters, 10
///   oversampling
/// * `seed` - Seed for the random projection
/// * `verbose` - `0` silent, `1` or `2` print the total time
///
/// ### Returns
///
/// `SvdResults<T>` with `u` `[n, n_components]`, `s` of length
/// `n_components`, and `v` `[m, n_components]`.
///
/// ### Errors
///
/// * `SparseLayoutMismatch` if `data` is not CSC
/// * `DimensionMisMatchSparse` if `col_means`, `col_stds` or `row_offsets`
///   have the wrong length
/// * `Data2NotAvailable` if `data_2` is missing
/// * `InvalidArgument` if `n_components` is zero or above `min(n, m)`, or a
///   size exceeds what MLX can index
/// * `FaerCholeskyError` if a CholeskyQR2 pass hits a non-SPD Gram matrix
/// * `MlxError` if mlx-c reports a failure
#[allow(clippy::too_many_arguments)]
pub fn randomised_sparse_svd_mlx<T: BixverseFloat>(
    data: CompressedSparseData2<T, T>,
    col_means: &[T],
    col_stds: &[T],
    row_offsets: Option<&[T]>,
    n_components: usize,
    params: Option<RandSvdMlxParams>,
    seed: u64,
    verbose: usize,
) -> Result<SvdResults<T>, MlxErrors> {
    let start = Instant::now();
    let (res, _) = rand_svd_impl(
        data,
        col_means,
        col_stds,
        row_offsets,
        n_components,
        params,
        seed,
        false,
    )?;
    if parse_verbosity_level(verbose).normal_verbosity() {
        println!(
            "  Finished randomised sparse SVD (MLX) in {:.2?}",
            start.elapsed()
        );
    }
    Ok(res)
}

/// [`randomised_sparse_svd_mlx`] with a blocking eval at every stage boundary
/// and the per-stage wall time. The extra syncs remove the overlap between
/// stages, so take end-to-end totals from the unstaged call.
///
/// ### Params
///
/// * `data` - CSC of A `(n, m)`, with `data_2` populated
/// * `col_means` - Column means `[m]`
/// * `col_stds` - Column standard deviations `[m]`, floored > 0
/// * `row_offsets` - Optional per-row offsets `[n]`
/// * `n_components` - Number of leading singular triples to return
/// * `params` - Optional [`RandSvdMlxParams`]
/// * `seed` - Seed for the random projection
///
/// ### Returns
///
/// The SVD as in [`randomised_sparse_svd_mlx`] and the stage timings
///
/// ### Errors
///
/// As [`randomised_sparse_svd_mlx`]
#[allow(clippy::too_many_arguments)]
pub fn randomised_sparse_svd_mlx_staged<T: BixverseFloat>(
    data: CompressedSparseData2<T, T>,
    col_means: &[T],
    col_stds: &[T],
    row_offsets: Option<&[T]>,
    n_components: usize,
    params: Option<RandSvdMlxParams>,
    seed: u64,
) -> Result<(SvdResults<T>, MlxRandSvdStages), MlxErrors> {
    rand_svd_impl(
        data,
        col_means,
        col_stds,
        row_offsets,
        n_components,
        params,
        seed,
        true,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense row-major to CSC.
    fn dense_to_csc(a: &[f32], n: usize, m: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for j in 0..m {
            for i in 0..n {
                let v = a[i * m + j];
                if v != 0.0 {
                    values.push(v);
                    indices.push(i as u32);
                }
            }
            indptr.push(values.len() as u32);
        }
        (values, indices, indptr)
    }

    /// Deterministic, structurally sparse, full column rank.
    fn build_dense(n: usize, m: usize) -> Vec<f32> {
        (0..n * m)
            .map(|k| {
                let (i, j) = (k / m, k % m);
                if (i + 2 * j).is_multiple_of(5) {
                    ((i * 7 + j * 13 + 3) % 19) as f32 * 0.3 + 0.5
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// CSC of a dense row-major matrix, values in both layers.
    fn csc_of(a: &[f32], n: usize, m: usize) -> CompressedSparseData2<f32, f32> {
        let (values, indices, indptr) = dense_to_csc(a, n, m);
        CompressedSparseData2::new_csc(&values, &indices, &indptr, Some(&values), (n, m))
    }

    /// Rank-6 signal plus a dense floor, every fourth entry zeroed, columns
    /// stretched by factors from 0.2 to 5.
    fn scaled_dense(n: usize, m: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                if (i * 3 + j) % 4 == 0 {
                    continue;
                }
                let mut v = 0.0f32;
                for f in 0..6 {
                    let a = ((i * (f + 3) + 7 * f) % 17) as f32 / 17.0 - 0.5;
                    let b = ((j * (2 * f + 5) + 3 * f) % 13) as f32 / 13.0 - 0.5;
                    v += 4.0 * a * b / (1.0 + f as f32);
                }
                v += 0.05 * ((i * 31 + j * 17) % 11) as f32 / 11.0 + 0.3;
                let scale = 0.2 + 4.8 * j as f32 / (m - 1) as f32;
                dense[i * m + j] = v * scale;
            }
        }
        dense
    }

    /// Column means and sample standard deviations, in f64.
    fn col_stats(dense: &[f32], n: usize, m: usize) -> (Vec<f32>, Vec<f32>) {
        let mut mu = vec![0.0f32; m];
        let mut sigma = vec![0.0f32; m];
        for j in 0..m {
            let mean = (0..n).map(|i| dense[i * m + j] as f64).sum::<f64>() / n as f64;
            let var = (0..n)
                .map(|i| (dense[i * m + j] as f64 - mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64;
            mu[j] = mean as f32;
            sigma[j] = var.sqrt() as f32;
        }
        (mu, sigma)
    }

    /// Singular values within `tol` (relative) of `want`, and the smallest
    /// principal cosine between the score subspaces above `cos_tol`.
    fn assert_svd_parity(
        got: &SvdResults<f32>,
        want: &faer::linalg::solvers::Svd<f64>,
        k: usize,
        tol: f64,
        cos_tol: f64,
        label: &str,
    ) {
        let n = got.u.nrows();
        let mut worst = 0.0f64;
        for i in 0..k {
            let rel = (got.s[i] as f64 - want.S()[i]).abs() / want.S()[i];
            worst = worst.max(rel);
        }
        let u_got = Mat::<f64>::from_fn(n, k, |i, j| got.u[(i, j)] as f64);
        let cross = want.U().subcols(0, k).transpose() * &u_got;
        let cos_min = cross
            .thin_svd()
            .unwrap()
            .S()
            .column_vector()
            .iter()
            .fold(f64::MAX, |a, &b| a.min(b));
        eprintln!("{label}: worst rel s err {worst:.2e}, min principal cosine {cos_min:.7}");
        assert!(worst < tol, "{label}: singular value rel err {worst:.2e}");
        assert!(
            cos_min > cos_tol,
            "{label}: score subspace drifted, {cos_min:.5}"
        );
    }

    /// `Q^T Q` of a row-major `[n, s]` `Q`.
    fn gram_host(q: &[f32], n: usize, s: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                out[i * s + j] = (0..n).map(|k| q[k * s + i] * q[k * s + j]).sum();
            }
        }
        out
    }

    /// Max abs deviation from the identity.
    fn max_from_identity(mat: &[f32], s: usize) -> f32 {
        (0..s * s).fold(0.0f32, |d, k| {
            let want = if k / s == k % s { 1.0 } else { 0.0 };
            d.max((mat[k] - want).abs())
        })
    }

    #[test]
    fn test_mlx_r_inverse_from_gram_round_trip() {
        let s = 4;
        let g: Vec<f32> = vec![
            4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.5, 0.5, 0.0, 0.0, 0.5, 1.5,
        ];
        let r_inv = r_inverse_from_gram(&g, s).unwrap();
        for i in 0..s {
            for j in 0..i {
                assert!(r_inv[i * s + j].abs() < 1e-6, "not upper at ({i}, {j})");
            }
        }
        // G (R^-1 R^-T) = I
        let mut prod = vec![0.0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                for l in 0..s {
                    let g_inv: f32 = (0..s).map(|t| r_inv[l * s + t] * r_inv[j * s + t]).sum();
                    prod[i * s + j] += g[i * s + l] * g_inv;
                }
            }
        }
        assert!(max_from_identity(&prod, s) < 1e-4);
    }

    /// `Q` from CholeskyQR2 satisfies `Q^T Q = I` to fp32 tolerance.
    #[test]
    fn test_mlx_cholesky_qr2_orthogonality() {
        install_error_handler();
        let st = Stream::default_gpu();
        let (n, s) = (200, 8);
        let y_host: Vec<f32> = (0..n * s)
            .map(|i| {
                let x = ((i / s) as f32 + 0.5) / n as f32;
                (2.0 * std::f32::consts::PI * ((i % s) + 1) as f32 * x).sin()
            })
            .collect();
        let y = Array::from_f32(&y_host, &[n as i32, s as i32]);
        let q = cholesky_qr2(&y, s, &st).unwrap();
        eval_all(&[&q], false).unwrap();
        let diff = max_from_identity(&gram_host(q.as_f32().unwrap(), n, s), s);
        eprintln!("cholesky_qr2: max |Q^T Q - I| {diff:.3e}");
        assert!(diff < 1e-4);
    }

    /// Top singular values match the dense faer SVD.
    #[test]
    fn test_mlx_randomised_sparse_svd_singular_values() {
        let (n, m, k) = (60usize, 20usize, 3usize);
        let a = build_dense(n, m);
        let svd = randomised_sparse_svd_mlx(
            csc_of(&a, n, m),
            &vec![0.0f32; m],
            &vec![1.0f32; m],
            None,
            k,
            Some(RandSvdMlxParams::new(2, 8)),
            42,
            0,
        )
        .unwrap();

        let want = Mat::<f32>::from_fn(n, m, |i, j| a[i * m + j])
            .thin_svd()
            .unwrap();
        let mut worst = 0.0f32;
        for i in 0..k {
            worst = worst.max((svd.s[i] - want.S()[i]).abs() / want.S()[i].abs());
            assert!(svd.s[i] >= 0.0);
            if i > 0 {
                assert!(svd.s[i - 1] >= svd.s[i], "not sorted at {i}");
            }
        }
        eprintln!("singular values: worst rel err {worst:.2e}");
        assert!(worst < 1e-3);
    }

    /// Parity with a dense SVD of the centred and scaled matrix, column scales
    /// far from 1, so dividing by `sigma` instead of `sigma^2` in the loop
    /// would show.
    #[test]
    fn test_mlx_randomised_sparse_svd_scaled_parity() {
        let (n, m, k) = (400usize, 40usize, 5usize);
        let dense = scaled_dense(n, m);
        let (mu, sigma) = col_stats(&dense, n, m);
        let got = randomised_sparse_svd_mlx(
            csc_of(&dense, n, m),
            &mu,
            &sigma,
            None,
            k,
            Some(RandSvdMlxParams::new(2, 10)),
            42,
            0,
        )
        .unwrap();
        let want = Mat::<f64>::from_fn(n, m, |i, j| {
            (dense[i * m + j] as f64 - mu[j] as f64) / sigma[j] as f64
        })
        .thin_svd()
        .unwrap();
        assert_svd_parity(&got, &want, k, 1e-4, 0.9999, "scaled parity");
    }

    /// Same with row offsets, so the CLR terms of both SpMMs are live. The
    /// reference subtracts the offset of each row before centring and scaling.
    #[test]
    fn test_mlx_randomised_sparse_svd_row_offsets_parity() {
        let (n, m, k) = (400usize, 40usize, 5usize);
        let dense = scaled_dense(n, m);
        let (mu, sigma) = col_stats(&dense, n, m);
        let off: Vec<f32> = (0..n).map(|i| ((i * 7) % 13) as f32 * 0.03).collect();
        // the operator is (A - o 1^T - 1 mu^T) D^-1
        let got = randomised_sparse_svd_mlx(
            csc_of(&dense, n, m),
            &mu,
            &sigma,
            Some(&off),
            k,
            Some(RandSvdMlxParams::new(2, 10)),
            42,
            0,
        )
        .unwrap();
        let want = Mat::<f64>::from_fn(n, m, |i, j| {
            (dense[i * m + j] as f64 - off[i] as f64 - mu[j] as f64) / sigma[j] as f64
        })
        .thin_svd()
        .unwrap();
        assert_svd_parity(&got, &want, k, 1e-4, 0.9999, "row offsets parity");
    }

    /// The staged call returns the same answer as the plain one.
    #[test]
    fn test_mlx_randomised_sparse_svd_staged_matches() {
        let (n, m, k) = (400usize, 40usize, 5usize);
        let dense = scaled_dense(n, m);
        let (mu, sigma) = col_stats(&dense, n, m);
        let params = Some(RandSvdMlxParams::new(2, 10));
        let plain =
            randomised_sparse_svd_mlx(csc_of(&dense, n, m), &mu, &sigma, None, k, params, 42, 0)
                .unwrap();
        let (staged, st) = randomised_sparse_svd_mlx_staged(
            csc_of(&dense, n, m),
            &mu,
            &sigma,
            None,
            k,
            params,
            42,
        )
        .unwrap();
        assert_eq!(plain.s, staged.s);
        assert!(st.total() > Duration::ZERO);
    }

    /// Accuracy against the dense reference where the sketch quality shows,
    /// swept over power iterations.
    #[test]
    // Heavy: three power-iteration arms at n = 3000, plus a dense faer SVD.
    #[cfg(feature = "large-test")]
    fn test_mlx_randomised_sparse_svd_accuracy_vs_dense() {
        let (n, m, k, oversampling) = (3000usize, 300usize, 20usize, 30usize);
        let hash = |a: usize, b: usize| -> f32 {
            let mut h = (a as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            ((h >> 40) as f32) / 16_777_216.0
        };
        let mut dense = vec![0.0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                if (i + j) % 10 != 0 {
                    continue;
                }
                let mut v = 0.0f32;
                for f in 0..24 {
                    v += 6.0 * (hash(i, f) - 0.5) * (hash(j, f + 100) - 0.5)
                        / (1.0 + f as f32).sqrt();
                }
                dense[i * m + j] = v + 0.15 * hash(i, j + 7) + 0.1;
            }
        }
        let mut mu = vec![0.0f32; m];
        let mut sigma = vec![0.0f32; m];
        for j in 0..m {
            let mean: f64 = (0..n).map(|i| dense[i * m + j] as f64).sum::<f64>() / n as f64;
            let var: f64 = (0..n)
                .map(|i| (dense[i * m + j] as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            mu[j] = mean as f32;
            sigma[j] = (var.sqrt() as f32).max(1e-6);
        }
        let want = Mat::<f32>::from_fn(n, m, |i, j| (dense[i * m + j] - mu[j]) / sigma[j])
            .thin_svd()
            .unwrap();

        let mut worst_at = Vec::new();
        for n_power in [0usize, 1, 2] {
            let got = randomised_sparse_svd_mlx(
                csc_of(&dense, n, m),
                &mu,
                &sigma,
                None,
                k,
                Some(RandSvdMlxParams::new(n_power, oversampling)),
                42,
                0,
            )
            .unwrap();
            let worst = (0..k).fold(0.0f32, |w, i| {
                w.max((got.s[i] - want.S()[i]).abs() / want.S()[i].abs())
            });
            eprintln!("n_power_iters={n_power}: worst relative error {worst:.5}");
            worst_at.push(worst);
        }
        assert!(
            worst_at[0] > 4.0 * worst_at[2],
            "error barely moved across power iterations ({:.5} -> {:.5})",
            worst_at[0],
            worst_at[2]
        );
        assert!(worst_at[2] < 0.015, "worst rel error {:.5}", worst_at[2]);
    }

    /// A CSR input is refused up front.
    #[test]
    fn test_mlx_randomised_sparse_svd_csr_input_rejected() {
        let (n, m) = (10usize, 4usize);
        let a = build_dense(n, m);
        let csc = csc_of(&a, n, m);
        let csr = csc.transform();
        let res = randomised_sparse_svd_mlx(
            csr,
            &vec![0.0f32; m],
            &vec![1.0f32; m],
            None,
            2,
            None,
            42,
            0,
        );
        assert!(matches!(
            res,
            Err(MlxErrors::Core(CoreErrors::SparseLayoutMismatch { .. }))
        ));
    }

    /// Wrong-length `mu`, `sigma` or offsets error before any upload.
    #[test]
    fn test_mlx_randomised_sparse_svd_dim_mismatch_rejected() {
        let (n, m) = (10usize, 4usize);
        let a = build_dense(n, m);
        let res = randomised_sparse_svd_mlx(
            csc_of(&a, n, m),
            &vec![0.0f32; m + 1],
            &vec![1.0f32; m + 1],
            None,
            2,
            None,
            42,
            0,
        );
        assert!(matches!(
            res,
            Err(MlxErrors::Core(CoreErrors::DimensionMisMatchSparse { .. }))
        ));
        let res = randomised_sparse_svd_mlx(
            csc_of(&a, n, m),
            &vec![0.0f32; m],
            &vec![1.0f32; m],
            Some(&vec![0.0f32; n - 1]),
            2,
            None,
            42,
            0,
        );
        assert!(matches!(
            res,
            Err(MlxErrors::Core(CoreErrors::DimensionMisMatchSparse { .. }))
        ));
    }
}
