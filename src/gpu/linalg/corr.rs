//! GPU-accelerated pairwise column correlation and covariance.
//!
//! ### Pipeline
//!
//! Spearman rank-transforms the columns on the host first; the other two
//! variants skip straight to the upload. From there everything runs on the
//! device: [`column_stats()`] takes the per-column mean and inverse scale,
//! [`apply_centre_scale()`] applies them, and the whole thing collapses to one
//! symmetric Gram product via [`gram_aat`]. The `1/sqrt(n-1)` factor is folded
//! into the scale, so that single product *is* the correlation or covariance
//! matrix, with no separate normalisation pass.
//!
//! ### Layout
//!
//! Everything on the device side is feature-major: the upload is a faer matrix's
//! column-major allocation reinterpreted as `[n_cols, n_rows]` row-major, so a
//! feature's samples sit contiguously at `data[j * n_rows ..]`. That is the
//! `[d, n]` layout [`crate::gpu::linalg::gram`] documents, spelled with this
//! file's parameter names. Kernels here flat-index throughout, so the tensor
//! shape metadata is descriptive rather than load-bearing.
//!
//! Whether this beats the faer path depends on the shape and the device;
//! `benches/gpu_corr_bench.rs` measures it stage by stage.

// The `#[cube]` macro generates undocumented launcher structs and functions.
#![allow(missing_docs)]

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;
use cubecl::prelude::*;
use faer::{Mat, MatRef};
use std::time::Instant;

use crate::core::math::matrix_helpers::rank_matrix_col;
use crate::gpu::linalg::gram::gram_aat;
use crate::gpu::*;
use crate::prelude::*;

///////////
// Enums //
///////////

/// Which pairwise statistic [`column_pairwise_cor_gpu`] computes.
///
/// Selected host-side: the variant decides whether the columns are
/// rank-transformed before upload and whether the scaling divides by the
/// standard deviation. No kernel branches on it.
#[derive(CubeType, Clone, Copy, Default)]
pub enum GpuCorCov {
    /// Covariance. Columns are centred but not divided by their standard
    /// deviation.
    #[default]
    Covariance,
    /// Pearson correlation. Columns are centred and divided by their standard
    /// deviation.
    Pearson,
    /// Spearman correlation. Pearson over columns rank-transformed on the host.
    Spearman,
}

/// Parse a correlation or covariance variant from its name.
///
/// Case-insensitive. Accepts `"pearson"` or `"cor"`, `"spearman"` or
/// `"ranked"`, and `"cov"` or `"covariance"`.
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// The matching [`GpuCorCov`], or `None` if the string is not one of the names
/// above.
pub fn parse_gpu_cor(s: &str) -> Option<GpuCorCov> {
    match s.to_lowercase().as_str() {
        "pearson" | "cor" => Some(GpuCorCov::Pearson),
        "spearman" | "ranked" => Some(GpuCorCov::Spearman),
        "cov" | "covariance" => Some(GpuCorCov::Covariance),
        _ => None,
    }
}

/////////////
// Kernels //
/////////////

/// Per-column mean and inverse scale.
///
/// One workgroup per column, two-pass: pass 1 tree-reduces the sum to get a
/// stable mean; pass 2 tree-reduces `sum((x - mean)^2)` so we never form
/// `sumsq - n*mu^2`. With `scale_sd`, writes `1/(std * sqrt(n-1))`; otherwise
/// just `1/sqrt(n-1)`. The `1/sqrt(n-1)` factor is folded in so the downstream
/// Gram product produces correlation or covariance directly.
///
/// Both reduction ladders are unrolled for exactly [`WORKGROUP_128`] threads and
/// there is no guard on the workgroup size. Launched at any other width the
/// kernel silently drops or double-counts terms, so the launch in
/// [`scale_matrix_col_gpu`] is not free to pick a different one.
///
/// ### Params
///
/// * `data` - Input matrix `[n_cols, n_rows]` row-major in `F`, feature-major
/// * `means` - Output column means `[n_cols]`
/// * `inv_scales` - Output inverse scales `[n_cols]`; `1/(std * sqrt(n-1))`
///   when `scale_sd`, else `1/sqrt(n-1)`
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `scale_sd` - Whether to divide by the column standard deviation; true
///   for correlation, false for covariance
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> column index
/// * `UNIT_POS_X` -> stride offset over rows within the column
#[cube(launch_unchecked)]
pub fn column_stats<F: Float>(
    data: &Tensor<F>,
    means: &mut Tensor<F>,
    inv_scales: &mut Tensor<F>,
    n_rows: u32,
    n_cols: u32,
    #[comptime] scale_sd: bool,
) {
    let feat = CUBE_POS_X;
    if feat >= n_cols {
        terminate!();
    }
    let tx = UNIT_POS_X;
    let mut shared = SharedMemory::<F>::new(WORKGROUP_128 as usize);

    let base = feat * n_rows;

    // Pass 1: sum -> mean.
    let mut local_sum = F::new(0.0);
    let mut i = tx;
    while i < n_rows {
        local_sum += data[(base + i) as usize];
        i += WORKGROUP_128;
    }
    shared[tx as usize] = local_sum;
    sync_cube();

    if tx < 64u32 {
        let v = shared[(tx + 64u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 32u32 {
        let v = shared[(tx + 32u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 16u32 {
        let v = shared[(tx + 16u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 8u32 {
        let v = shared[(tx + 8u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 4u32 {
        let v = shared[(tx + 4u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 2u32 {
        let v = shared[(tx + 2u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 1u32 {
        let v = shared[(tx + 1u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();

    let n = F::cast_from(n_rows);
    let mean = shared[0] / n;

    // Pass 2: centred sum of squares.
    let mut local_sumsq = F::new(0.0);
    let mut j = tx;
    while j < n_rows {
        let d = data[(base + j) as usize] - mean;
        local_sumsq += d * d;
        j += WORKGROUP_128;
    }
    sync_cube();
    shared[tx as usize] = local_sumsq;
    sync_cube();

    if tx < 64u32 {
        let v = shared[(tx + 64u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 32u32 {
        let v = shared[(tx + 32u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 16u32 {
        let v = shared[(tx + 16u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 8u32 {
        let v = shared[(tx + 8u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 4u32 {
        let v = shared[(tx + 4u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 2u32 {
        let v = shared[(tx + 2u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 1u32 {
        let v = shared[(tx + 1u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();

    if tx == 0u32 {
        let nm1 = n - F::new(1.0);
        let inv_sqrt_nm1 = F::new(1.0) / F::sqrt(nm1);
        means[feat as usize] = mean;
        if scale_sd {
            let std = F::sqrt(shared[0] / nm1);
            let eps = F::new(1e-10);
            let safe_std = if std < eps { F::new(1.0) } else { std };
            inv_scales[feat as usize] = inv_sqrt_nm1 / safe_std;
        } else {
            inv_scales[feat as usize] = inv_sqrt_nm1;
        }
    }
}

/// Element-wise centre and rescale:
/// `out[i,j] = (data[i,j] - means[j]) * inv_scales[j]`.
///
/// Flat-indexed over the full `[n_rows * n_cols]` element space; one workgroup
/// covers a contiguous block of `wg_size` elements. The buffer is feature-major,
/// so a thread recovers its column with `idx / n_rows` and consecutive lanes
/// share it for all but one boundary per column.
///
/// ### Params
///
/// * `data` - Input matrix `[n_cols, n_rows]` row-major in `F`, feature-major
/// * `means` - Column means `[n_cols]` (from [`column_stats()`])
/// * `inv_scales` - Column inverse scales `[n_cols]` (from [`column_stats()`])
/// * `out` - Output matrix `[n_cols, n_rows]` row-major in `F`, i.e. the same
///   feature-major layout as `data`
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> flat workgroup index
/// * `UNIT_POS_X` -> element offset within the workgroup block
#[cube(launch_unchecked)]
pub fn apply_centre_scale<F: Float>(
    data: &Tensor<F>,
    means: &Tensor<F>,
    inv_scales: &Tensor<F>,
    out: &mut Tensor<F>,
    n_rows: u32,
    n_cols: u32,
    #[comptime] wg_size: u32,
) {
    let tx = UNIT_POS_X;
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + tx;
    let total = n_rows * n_cols;
    if idx >= total {
        terminate!();
    }
    let feat = idx / n_rows;
    let v = data[idx as usize];
    let m = means[feat as usize];
    let s = inv_scales[feat as usize];
    out[idx as usize] = (v - m) * s;
}

////////////////
// Dispatcher //
////////////////

/// Centre and (optionally) rescale columns.
///
/// Dispatches [`column_stats()`] followed by [`apply_centre_scale()`]. The returned
/// tensor has `1/sqrt(n-1)` baked in, so `S S^T` over it (`S` being
/// feature-major, so this is the Gram of its rows) yields correlation
/// (`scale_sd = true`) or covariance (`scale_sd = false`).
///
/// The two launches use different workgroup widths on purpose.
/// [`column_stats()`] is one workgroup per column and is pinned to
/// [`WORKGROUP_128`] by its unrolled reduction ladder;
/// [`apply_centre_scale()`] is a flat elementwise pass
/// with no cross-thread communication, so it takes the wider
/// [`WORKGROUP_256`] and fewer blocks.
///
/// ### Params
///
/// * `data` - Input matrix `[n_cols, n_rows]` row-major, feature-major
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `scale_sd` - Whether to divide by the column standard deviation; true
///   for correlation, false for covariance
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// Scaled matrix `[n_cols, n_rows]` row-major, i.e. the same feature-major
/// layout as the input.
///
/// ### Errors
///
/// * `GpuCubeCountExceeded` if either grid is over the device limit.
pub fn scale_matrix_col_gpu<F, R>(
    data: &GpuTensor<R, F>,
    n_rows: usize,
    n_cols: usize,
    scale_sd: bool,
    client: &ComputeClient<R>,
) -> Result<GpuTensor<R, F>, BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let means = GpuTensor::<R, F>::empty(vec![n_cols], client);
    let inv_scales = GpuTensor::<R, F>::empty(vec![n_cols], client);
    let scaled = GpuTensor::<R, F>::empty(vec![n_cols, n_rows], client);

    let stats_count = checked_cube_count::<R>("column_stats", n_cols as u32, 1, 1)?;
    unsafe {
        column_stats::launch_unchecked::<F, R>(
            client,
            stats_count,
            CubeDim::new_1d(WORKGROUP_128),
            data.clone().into_tensor_arg(),
            means.clone().into_tensor_arg(),
            inv_scales.clone().into_tensor_arg(),
            n_rows as u32,
            n_cols as u32,
            scale_sd,
        );
    }

    let total = (n_rows * n_cols) as u32;
    let n_blocks = total.div_ceil(WORKGROUP_256);
    let (gx, gy) = grid_2d(n_blocks);
    let scale_count = checked_cube_count::<R>("apply_centre_scale", gx, gy, 1)?;
    unsafe {
        apply_centre_scale::launch_unchecked::<F, R>(
            client,
            scale_count,
            CubeDim::new_1d(WORKGROUP_256),
            data.clone().into_tensor_arg(),
            means.clone().into_tensor_arg(),
            inv_scales.clone().into_tensor_arg(),
            scaled.clone().into_tensor_arg(),
            n_rows as u32,
            n_cols as u32,
            WORKGROUP_256,
        );
    }

    Ok(scaled)
}

/////////////
// Helpers //
/////////////

/// Borrow a matrix's backing buffer as one flat feature-major slice, if its
/// columns are stored back to back.
///
/// The upload wants `[n_cols, n_rows]` row-major, which for a column-major
/// `MatRef` is exactly the underlying allocation. A `Mat` built by faer
/// normally satisfies this, so the common path can skip building a second copy
/// of the whole matrix; a strided view or a submatrix cannot, and the caller
/// falls back to copying.
///
/// ### Params
///
/// * `mat` - Input matrix `[n_rows, n_cols]`
///
/// ### Returns
///
/// `Some(slice)` of length `n_rows * n_cols` when the matrix is column-major
/// with unit row stride and no gap between columns, `None` otherwise.
fn contiguous_col_major<F: BixverseFloat>(mat: MatRef<'_, F>) -> Option<&'_ [F]> {
    let (n_rows, n_cols) = (mat.nrows(), mat.ncols());
    let cm = mat.try_as_col_major()?;
    if cm.col_stride() as usize != n_rows {
        return None;
    }
    // SAFETY: `try_as_col_major` guarantees unit row stride, and the check
    // above guarantees no padding between columns, so the `n_rows * n_cols`
    // elements of a single allocation are exactly the range starting at
    // `as_ptr()`.
    Some(unsafe { std::slice::from_raw_parts(cm.as_ptr(), n_rows * n_cols) })
}

//////////
// Main //
//////////

/// GPU-accelerated pairwise column correlation or covariance.
///
/// For Spearman, columns are rank-transformed on the CPU before upload. The
/// matrix is then centred and scaled on the GPU via [`scale_matrix_col_gpu`],
/// which leaves it feature-major, and the Gram product of its rows gives the
/// full `[n_cols, n_cols]` output directly.
///
/// The product goes through [`gram_aat`] rather than cubek. cubek's
/// `Strategy::Auto` blows up on Apple devices here, and the `DoubleUnit`
/// fallback it was pinned to runs at 5-7% of peak on ordinary shapes and 0.13%
/// when `n_cols` is small; see the module doc on [`crate::gpu::linalg::gram`].
///
/// ### Params
///
/// * `mat` - Input matrix `[n_rows, n_cols]`, samples by features
/// * `cor_type` - Correlation or covariance variant to compute
/// * `device` - CubeCL device to run on
/// * `verbose` - Print the time to upload and the total elapsed to stdout
///
/// ### Returns
///
/// Pairwise column correlation or covariance matrix `[n_cols, n_cols]`.
///
/// ### Errors
///
/// * `InvalidArgument` if the `[d, d]` output is larger than the device accepts
///   in one binding.
/// * `GpuCubeCountExceeded` if any grid is over the device limit.
pub fn column_pairwise_cor_gpu<F, R>(
    mat: MatRef<F>,
    cor_type: GpuCorCov,
    device: R::Device,
    verbose: bool,
) -> Result<Mat<F>, BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement + BixverseFloat,
{
    let start = Instant::now();
    let scale_sd = !matches!(cor_type, GpuCorCov::Covariance);

    let ranked = matches!(cor_type, GpuCorCov::Spearman).then(|| rank_matrix_col(&mat));
    let mat = ranked.as_ref().map(|m| m.as_ref()).unwrap_or(mat);

    let (n_rows, n_cols) = (mat.nrows(), mat.ncols());
    let owned;
    let data_flat: &[F] = match contiguous_col_major(mat) {
        Some(slice) => slice,
        None => {
            let mut buf: Vec<F> = Vec::with_capacity(n_rows * n_cols);
            for j in 0..n_cols {
                match mat.col(j).try_as_col_major() {
                    Some(col) => buf.extend_from_slice(col.as_slice()),
                    None => buf.extend(mat.col(j).iter().cloned()),
                }
            }
            owned = buf;
            &owned
        }
    };
    let client = R::client(&device);

    // The output is quadratic in `n_cols` and is a single binding. Past the
    // device limit `launch_unchecked` does no work, returns zeros and reports
    // nothing, so refuse up front rather than hand back a plausible-looking
    // matrix of zeros.
    let out_bytes = n_cols * n_cols * size_of::<F>();
    let max_binding = client.properties().memory.max_page_size as usize;
    if out_bytes > max_binding {
        return Err(BixverseErrors::InvalidArgument(format!(
            "GPU correlation: the {n_cols} x {n_cols} output needs {} MB but the \
             device accepts at most {} MB per binding. Reduce the feature set.",
            out_bytes / (1024 * 1024),
            max_binding / (1024 * 1024),
        )));
    }

    let data_gpu = GpuTensor::<R, F>::from_slice(data_flat, vec![n_cols, n_rows], &client);

    if verbose {
        println!("Upload to GPU done: {:.2?}", start.elapsed());
    }

    let scaled = scale_matrix_col_gpu(&data_gpu, n_rows, n_cols, scale_sd, &client)?;
    let result = GpuTensor::<R, F>::empty(vec![n_cols, n_cols], &client);

    gram_aat::<R, F>(&client, &scaled, &result, n_rows, n_cols)?;

    let result_flat = result.read(&client)?;

    if verbose {
        println!(" ... done in {:.2?}", start.elapsed());
    }

    // `result_flat` is row-major and `Mat::from_fn` fills column-major, so the
    // obvious `[i * n_cols + j]` strides by `n_cols` on every read. The output
    // is symmetric, so reading the transpose is the same matrix and walks the
    // buffer sequentially. Worth 80% of wall-clock at d = 8000.
    Ok(Mat::from_fn(n_cols, n_cols, |i, j| {
        result_flat[j * n_cols + i]
    }))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    /////////////
    // Helpers //
    /////////////

    // The CPU references below take `data` sample-major `[n, d]`, unlike the
    // feature-major layout the device side uses. They mirror the caller's view,
    // not the upload's.

    /// The default device, or `None` when the machine has no usable GPU, in
    /// which case every test here returns early rather than failing.
    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Assert two matrices match elementwise to an absolute tolerance.
    fn assert_mat_close(got: &Mat<f32>, want: &Mat<f32>, tol: f32) {
        assert_eq!(got.nrows(), want.nrows());
        assert_eq!(got.ncols(), want.ncols());
        for i in 0..got.nrows() {
            for j in 0..got.ncols() {
                assert!(
                    (got[(i, j)] - want[(i, j)]).abs() < tol,
                    "({},{}) got {} want {} diff {}",
                    i,
                    j,
                    got[(i, j)],
                    want[(i, j)],
                    (got[(i, j)] - want[(i, j)]).abs()
                );
            }
        }
    }

    /// Column means of a sample-major `[n, d]` buffer.
    fn cpu_col_means(data: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut means = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..d {
                means[j] += data[i * d + j];
            }
        }
        for j in 0..d {
            means[j] /= n as f32;
        }
        means
    }

    /// Pearson correlation on the host, `[d, d]`. Deliberately the naive
    /// centre-scale-then-dot route, so it shares no code with the kernels.
    fn cpu_pearson(data: &[f32], n: usize, d: usize) -> Mat<f32> {
        let means = cpu_col_means(data, n, d);
        let mut stds = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..d {
                let diff = data[i * d + j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for j in 0..d {
            stds[j] = (stds[j] / (n - 1) as f32).sqrt().max(1e-10);
        }
        let inv_sqrt_nm1 = 1.0 / ((n - 1) as f32).sqrt();
        let mut scaled = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                scaled[i * d + j] = (data[i * d + j] - means[j]) * inv_sqrt_nm1 / stds[j];
            }
        }
        Mat::from_fn(d, d, |a, b| {
            (0..n)
                .map(|i| scaled[i * d + a] * scaled[i * d + b])
                .sum::<f32>()
        })
    }

    /// Covariance on the host, `[d, d]`. As `cpu_pearson` but without the
    /// division by the standard deviation.
    fn cpu_covariance(data: &[f32], n: usize, d: usize) -> Mat<f32> {
        let means = cpu_col_means(data, n, d);
        let inv_sqrt_nm1 = 1.0 / ((n - 1) as f32).sqrt();
        let mut scaled = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                scaled[i * d + j] = (data[i * d + j] - means[j]) * inv_sqrt_nm1;
            }
        }
        Mat::from_fn(d, d, |a, b| {
            (0..n)
                .map(|i| scaled[i * d + a] * scaled[i * d + b])
                .sum::<f32>()
        })
    }

    /// Rank-transform each column independently, ties taking the average of the
    /// 1-based ranks they span. Must agree with `rank_matrix_col`, or the
    /// Spearman test compares two different problems.
    fn cpu_rank_cols(data: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut ranked = vec![0.0f32; n * d];
        for j in 0..d {
            let mut col: Vec<(usize, f32)> = (0..n).map(|i| (i, data[i * d + j])).collect();
            col.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut i = 0;
            while i < n {
                let val = col[i].1;
                let mut end = i + 1;
                while end < n && col[end].1 == val {
                    end += 1;
                }
                // 1-based average rank for the tied block
                let avg_rank = (i + end + 1) as f32 / 2.0;
                for k in i..end {
                    ranked[col[k].0 * d + j] = avg_rank;
                }
                i = end;
            }
        }
        ranked
    }

    /// Spearman on the host, `[d, d]`: Pearson over the ranked columns.
    fn cpu_spearman(data: &[f32], n: usize, d: usize) -> Mat<f32> {
        let ranked = cpu_rank_cols(data, n, d);
        cpu_pearson(&ranked, n, d)
    }

    ///////////
    // Tests //
    ///////////

    #[test]
    fn test_pearson_matches_cpu() {
        let Some(device) = try_device() else { return };
        let (n, d) = (80, 6);
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.2 - 1.0)
            .collect();
        let mat = Mat::from_fn(n, d, |i, j| data[i * d + j]);

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            GpuCorCov::Pearson,
            device,
            false,
        )
        .unwrap();

        assert_mat_close(&got, &cpu_pearson(&data, n, d), 1e-4);
    }

    #[test]
    fn test_covariance_matches_cpu() {
        let Some(device) = try_device() else { return };
        let (n, d) = (80, 6);
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 11 + 5) % 17) as f32 * 0.3 - 2.0)
            .collect();
        let mat = Mat::from_fn(n, d, |i, j| data[i * d + j]);

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            GpuCorCov::Covariance,
            device,
            false,
        )
        .unwrap();

        assert_mat_close(&got, &cpu_covariance(&data, n, d), 1e-4);
    }

    #[test]
    fn test_spearman_matches_cpu() {
        let Some(device) = try_device() else { return };
        let (n, d) = (60, 5);
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 13 + 7) % 19) as f32 * 0.5)
            .collect();
        let mat = Mat::from_fn(n, d, |i, j| data[i * d + j]);

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            GpuCorCov::Spearman,
            device,
            false,
        )
        .unwrap();

        assert_mat_close(&got, &cpu_spearman(&data, n, d), 1e-4);
    }

    // Pearson diagonal must be ~1.
    #[test]
    fn test_pearson_diagonal_ones() {
        let Some(device) = try_device() else { return };
        let (n, d) = (100, 8);
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 9 + 1) % 31) as f32 * 0.1)
            .collect();
        let mat = Mat::from_fn(n, d, |i, j| data[i * d + j]);

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            GpuCorCov::Pearson,
            device,
            false,
        )
        .unwrap();

        for j in 0..d {
            assert!(
                (got[(j, j)] - 1.0).abs() < 1e-4,
                "diagonal[{}] = {} != 1.0",
                j,
                got[(j, j)]
            );
        }
    }

    // All three variants must produce a symmetric matrix.
    #[test]
    fn test_output_symmetric() {
        let Some(device) = try_device() else { return };
        let (n, d) = (80, 6);
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 7 + 5) % 13) as f32 * 0.4)
            .collect();

        for cor_type in [
            GpuCorCov::Pearson,
            GpuCorCov::Covariance,
            GpuCorCov::Spearman,
        ] {
            let mat = Mat::from_fn(n, d, |i, j| data[i * d + j]);
            let got = column_pairwise_cor_gpu::<f32, WgpuRuntime>(
                mat.as_ref(),
                cor_type,
                device.clone(),
                false,
            )
            .unwrap();

            for i in 0..d {
                for j in 0..d {
                    assert!(
                        (got[(i, j)] - got[(j, i)]).abs() < 1e-5,
                        "not symmetric at ({},{}) vs ({},{}): {} vs {}",
                        i,
                        j,
                        j,
                        i,
                        got[(i, j)],
                        got[(j, i)]
                    );
                }
            }
        }
    }
}
