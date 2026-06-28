//! GPU-accelerated correlations and co-variance calculations, leveraging the
//! fast GEMM on the GPU. Should be faster than the faer stuff on very large
//! data sets.

#![allow(missing_docs)]

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;
use ann_search_rs::utils::matrix_to_flat;
use cubecl::prelude::*;
use cubek::matmul::definition::MatmulPrecision;
use cubek::matmul::launch::Strategy;
use faer::{Mat, MatRef};
use std::time::Instant;

use crate::core::math::matrix_helpers::rank_matrix_col;
use crate::gpu::linalg::cholesky_gpu::dense_gemm;
use crate::gpu::*;
use crate::prelude::*;

///////////
// Enums //
///////////

/// GPU-accelerated correlations, covariance Enum for the dispatch
#[derive(CubeType, Clone, Copy, Default)]
pub enum GpuCorCov {
    #[default]
    /// GPU-accelerated Covariance calculation
    Covariance,
    /// GPU-accelerated Pearson correlation
    Pearson,
    /// GPU-accelerated Spearman correlation
    Spearman,
}

/// Parse the desired correlation type
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// The GPU-accelerated correlation or covariance to parse
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
/// `S^T S` GEMM produces correlation or covariance directly.
///
/// ### Params
///
/// * `data` - Input matrix `[n_rows, n_cols]` row-major in `F`
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
    let col = CUBE_POS_X;
    if col >= n_cols {
        terminate!();
    }
    let tx = UNIT_POS_X;
    let mut shared = SharedMemory::<F>::new(WORKGROUP_128 as usize);

    // Pass 1: sum -> mean.
    let mut local_sum = F::new(0.0);
    let mut i = tx;
    while i < n_rows {
        local_sum += data[(i * n_cols + col) as usize];
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
        let d = data[(j * n_cols + col) as usize] - mean;
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
        means[col as usize] = mean;
        if scale_sd {
            let std = F::sqrt(shared[0] / nm1);
            let eps = F::new(1e-10);
            let safe_std = if std < eps { F::new(1.0) } else { std };
            inv_scales[col as usize] = inv_sqrt_nm1 / safe_std;
        } else {
            inv_scales[col as usize] = inv_sqrt_nm1;
        }
    }
}

/// Element-wise centre and rescale:
/// `out[i,j] = (data[i,j] - means[j]) * inv_scales[j]`.
///
/// Flat-indexed over the full `[n_rows * n_cols]` element space; one
/// workgroup covers a contiguous block of `wg_size` elements, with each
/// thread striding to its element and recovering its column via `idx % n_cols`.
///
/// ### Params
///
/// * `data` - Input matrix `[n_rows, n_cols]` row-major in `F`
/// * `means` - Column means `[n_cols]` (from [`column_stats`])
/// * `inv_scales` - Column inverse scales `[n_cols]` (from [`column_stats`])
/// * `out` - Output matrix `[n_rows, n_cols]` row-major in `F`
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
    let col = idx % n_cols;
    let v = data[idx as usize];
    let m = means[col as usize];
    let s = inv_scales[col as usize];
    out[idx as usize] = (v - m) * s;
}

////////////////
// Dispatcher //
////////////////

/// Centre and (optionally) rescale columns.
///
/// Dispatches [`column_stats`] followed by [`apply_centre_scale`]. The returned
/// tensor has `1/sqrt(n-1)` baked in, so `S^T S` over it yields correlation
/// (`scale_sd = true`) or covariance (`scale_sd = false`).
///
/// ### Params
///
/// * `data` - Input matrix `[n_rows, n_cols]` row-major
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `scale_sd` - Whether to divide by the column standard deviation; true
///   for correlation, false for covariance
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// Scaled matrix `[n_rows, n_cols]` row-major
pub fn scale_matrix_col_gpu<F, R>(
    data: &GpuTensor<R, F>,
    n_rows: usize,
    n_cols: usize,
    scale_sd: bool,
    client: &ComputeClient<R>,
) -> GpuTensor<R, F>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let means = GpuTensor::<R, F>::empty(vec![n_cols], client);
    let inv_scales = GpuTensor::<R, F>::empty(vec![n_cols], client);
    let scaled = GpuTensor::<R, F>::empty(vec![n_rows, n_cols], client);

    unsafe {
        column_stats::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(n_cols as u32, 1, 1),
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
    unsafe {
        apply_centre_scale::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
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

    scaled
}

//////////
// Main //
//////////

/// GPU-accelerated pairwise column correlation or covariance.
///
/// For Spearman, columns are rank-transformed on the CPU before upload.
/// The matrix is then centred and scaled on the GPU via
/// [`scale_matrix_col_gpu`], and the result `S^T S` is computed via a
/// single GEMM, yielding the full `[n_cols, n_cols]` output directly.
///
/// ### Params
///
/// * `mat` - Input matrix (N x d)
/// * `cor_type` - Correlation or covariance variant to compute
/// * `device` - CubeCL device to run on
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// Pairwise column correlation or covariance matrix (d x d), or a
/// `BixverseErrors` on GPU failure.
pub fn column_pairwise_cor_gpu<F, R, MP>(
    mat: MatRef<F>,
    cor_type: GpuCorCov,
    device: R::Device,
    verbose: bool,
) -> Result<Mat<F>, BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement + BixverseFloat,
    MP: MatmulPrecision,
{
    let start = Instant::now();
    let scale_sd = !matches!(cor_type, GpuCorCov::Covariance);

    let t = Instant::now();
    let mat = match cor_type {
        GpuCorCov::Spearman => rank_matrix_col(&mat),
        _ => mat.to_owned(),
    };
    if verbose {
        println!("rank: {:.2?}", t.elapsed());
    }

    let t = Instant::now();
    let (data_flat, n_rows, n_cols) = matrix_to_flat(mat.as_ref());
    if verbose {
        println!("flat: {:.2?}", t.elapsed());
    }

    let t = Instant::now();
    let client = R::client(&device);
    if verbose {
        println!("client: {:.2?}", t.elapsed());
    }

    let t = Instant::now();
    let data_gpu = GpuTensor::<R, F>::from_slice(&data_flat, vec![n_rows, n_cols], &client);
    if verbose {
        println!("upload: {:.2?}", t.elapsed());
    }

    let scaled = scale_matrix_col_gpu(&data_gpu, n_rows, n_cols, scale_sd, &client);
    let result = GpuTensor::<R, F>::empty(vec![n_cols, n_cols], &client);

    // S is [n_rows, n_cols] row-major; S^T view is [n_cols, n_rows] with
    // swapped strides (handled by `a_transposed = true` in dense_gemm).
    // Both inputs are read-only, so the same buffer is passed twice.
    dense_gemm::<R, MP>(
        scaled.handle(),
        [n_cols, n_rows],
        true,
        scaled.handle(),
        [n_rows, n_cols],
        result.handle(),
        [n_cols, n_cols],
        // otherwise, this can blow up on Apple devices
        Some(Strategy::DoubleUnit(Default::default())),
        &client,
    )?;

    let result_flat = result.read(&client)?;

    if verbose {
        println!(" ... done in {:.2?}", start.elapsed());
    }

    Ok(Mat::from_fn(n_cols, n_cols, |i, j| {
        result_flat[i * n_cols + j]
    }))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

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

    /////////////
    // Helpers //
    /////////////

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

    fn cpu_spearman(data: &[f32], n: usize, d: usize) -> Mat<f32> {
        let ranked = cpu_rank_cols(data, n, d);
        cpu_pearson(&ranked, n, d)
    }

    //////////////////
    // Actual tests //
    //////////////////

    #[test]
    fn test_pearson_matches_cpu() {
        let Some(device) = try_device() else { return };
        let (n, d) = (80, 6);
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.2 - 1.0)
            .collect();
        let mat = Mat::from_fn(n, d, |i, j| data[i * d + j]);

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime, f32>(
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

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime, f32>(
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

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime, f32>(
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

        let got = column_pairwise_cor_gpu::<f32, WgpuRuntime, f32>(
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
            let got = column_pairwise_cor_gpu::<f32, WgpuRuntime, f32>(
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
