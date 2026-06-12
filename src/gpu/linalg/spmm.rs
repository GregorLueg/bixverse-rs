//! GPU SpMM kernels for sparse-by-dense products with mean centering and
//! column scaling folded in.
//!
//! Both directions of the operator A appearing in randomised SVD are
//! supported:
//!
//! * [`spmm_csr_forward`] computes `Y = A * X - 1 * c^T - r * x_sum^T` where
//!   `c` is a precomputed correction vector and `r` is a per-row offset vector.
//! * [`spmm_csc_transpose`] computes
//!   `Z = (A^T * Q - mu * q_sum^T - 1 * d^T) / sigma` where `d` is a
//!   precomputed column offset vector.
//!
//! Two small reduction kernels precompute the correction vectors:
//!
//! * [`dense_column_weighted_sum`] computes `c = mu^T * X_scaled` for the
//!   forward SpMM.
//! * [`dense_column_sum`] computes `q_sum = 1^T * Q` for the transpose SpMM.
//!
//! ### Threading
//!
//! One workgroup per output row in both SpMM kernels. Threads within a
//! workgroup stride over the `s` output columns of their assigned row,
//! iterating the row's nnz serially. The forward case has many short
//! segments (e.g. ~1M cells, ~200 nnz each); the transpose case has few
//! long segments (e.g. ~2000 genes, ~100k nnz each). Total work is
//! identical; the asymmetry is in how many workgroups exist and how long
//! each runs.
//!
//! ### Mixed precision
//!
//! Storage type `S` and accumulator type `A` are generic, mirroring the
//! k-means convention. Values are cast from `S` to `A` on load; all
//! arithmetic and writes happen in `A`. fp32 throughout is the default;
//! fp16 storage with fp32 accumulation is the opt-in quantisation path.
//!
//! ### Numerical care
//!
//! The reduction kernels use pairwise tree reduction in shared memory so
//! the correction vectors are accurate to O(log n) error rather than the
//! O(n) of naive serial summation. The per-row SpMM accumulation is naive
//! over ~100-200 nnz per row in the single-cell case, which is well within
//! fp32 tolerance. `sigma` is assumed to be floored on the host so the
//! transpose kernel's division never sees zero.

#![allow(missing_docs)]

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;
use cubecl::prelude::*;

use crate::gpu::WORKGROUP_128;
use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::prelude::*;

/////////////
// Kernels //
/////////////

/// Forward SpMM with mean correction: `Y = A * X - 1 * c^T`.
///
/// One workgroup per output row (cell). Threads in the workgroup stride
/// over the `s` output columns; each thread iterates its assigned row's
/// nnz serially, then subtracts the precomputed correction once before
/// writing.
///
/// ### Params
///
/// * `indptr` - CSR row pointers `[n + 1]`
/// * `indices` - Column indices of nnz `[nnz]`
/// * `values` - Values of nnz `[nnz]` in storage precision `S`
/// * `x` - Dense RHS `[m, s]` row-major in accumulator precision `A`
/// * `correction` - Precomputed correction vector `[s]` in `A`
/// * `y` - Dense output `[n, s]` row-major in `A`
/// * `n_rows` - Number of output rows
/// * `s_width` - Output width (rank + oversampling)
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> output row index
/// * `UNIT_POS_X` -> stride offset over output columns
#[cube(launch_unchecked)]
pub fn spmm_csr_forward<S: Float, A: Float>(
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    values: &Tensor<S>,
    x: &Tensor<A>,
    correction: &Tensor<A>,
    row_offsets: &Tensor<A>,
    x_sum: &Tensor<A>,
    y: &mut Tensor<A>,
    n_rows: u32,
    s_width: u32,
    #[comptime] wg_size: u32,
) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if row >= n_rows {
        terminate!();
    }

    let tx = UNIT_POS_X;

    let seg_start = indptr[row as usize];
    let seg_end = indptr[(row + 1u32) as usize];

    let m_row = row_offsets[row as usize];

    let mut col = tx;
    while col < s_width {
        let mut acc = A::new(0.0);
        let mut idx = seg_start;
        while idx < seg_end {
            let j = indices[idx as usize];
            let v = A::cast_from(values[idx as usize]);
            acc += v * x[j as usize * s_width as usize + col as usize];
            idx += 1u32;
        }
        acc -= correction[col as usize];
        acc -= m_row * x_sum[col as usize];
        y[row as usize * s_width as usize + col as usize] = acc;
        col += wg_size;
    }
}

/// Transpose SpMM with mean correction and column scaling:
/// `Z = (A^T * Q - mu * q_sum^T) / sigma`.
///
/// Uses CSC of A, which is structurally a CSR of A^T. One workgroup per
/// output row (gene). Threads stride over the `s` output columns and
/// iterate the column's nnz serially. After the sparse dot product, each
/// output element subtracts `mu[j] * q_sum[col]`, then divides by
/// `sigma[j]`. `mu_j` and `sigma_j` are loaded once per workgroup; the GPU
/// L1/constant cache services the uniform reads across threads.
///
/// ### Params
///
/// * `indptr` - CSC column pointers `[m + 1]`
/// * `indices` - Row indices of nnz `[nnz]`
/// * `values` - Values of nnz `[nnz]` in storage precision `S`
/// * `q` - Dense RHS `[n, s]` row-major in accumulator precision `A`
/// * `q_sum` - Precomputed column sums of Q `[s]` in `A`
/// * `mu` - Column means of A `[m]` in `A`
/// * `sigma` - Column standard deviations of A `[m]` in `A`. Must be > 0;
///   floor on the host.
/// * `z` - Dense output `[m, s]` row-major in `A`
/// * `m_rows` - Number of output rows
/// * `s_width` - Output width
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> output row index
/// * `UNIT_POS_X` -> stride offset over output columns
#[cube(launch_unchecked)]
pub fn spmm_csc_transpose<S: Float, A: Float>(
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    values: &Tensor<S>,
    q: &Tensor<A>,
    q_sum: &Tensor<A>,
    mu: &Tensor<A>,
    sigma: &Tensor<A>,
    m_dot_q: &Tensor<A>,
    z: &mut Tensor<A>,
    m_rows: u32,
    s_width: u32,
    #[comptime] wg_size: u32,
) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if row >= m_rows {
        terminate!();
    }

    let tx = UNIT_POS_X;

    let seg_start = indptr[row as usize];
    let seg_end = indptr[(row + 1u32) as usize];

    let mu_j = mu[row as usize];
    let sigma_j = sigma[row as usize];

    let mut col = tx;
    while col < s_width {
        let mut acc = A::new(0.0);
        let mut idx = seg_start;
        while idx < seg_end {
            let i = indices[idx as usize];
            let v = A::cast_from(values[idx as usize]);
            acc += v * q[i as usize * s_width as usize + col as usize];
            idx += 1u32;
        }
        acc -= mu_j * q_sum[col as usize];
        acc -= m_dot_q[col as usize];
        acc /= sigma_j;
        z[row as usize * s_width as usize + col as usize] = acc;
        col += wg_size;
    }
}

/// Column sums of a dense matrix: `out[col] = sum_i M[i, col]`.
///
/// One workgroup per output column. Each thread accumulates its strided
/// share of the rows into a partial sum; partial sums are then pairwise
/// tree-reduced through shared memory, bounding error growth at O(log n)
/// instead of O(n).
///
/// ### Params
///
/// * `matrix` - Input matrix `[n_rows, s_width]` row-major
/// * `out` - Output column sums `[s_width]`
/// * `n_rows` - Number of rows to sum over
/// * `s_width` - Number of columns
/// * `wg_size` - Workgroup size (comptime, must be a power of two)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> output column index
#[cube(launch_unchecked)]
pub fn dense_column_sum<A: Float>(
    matrix: &Tensor<A>,
    out: &mut Tensor<A>,
    n_rows: u32,
    s_width: u32,
    #[comptime] wg_size: u32,
) {
    let col = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if col >= s_width {
        terminate!();
    }

    let tx = UNIT_POS_X;

    let mut acc = A::new(0.0);
    let mut i = tx;
    while i < n_rows {
        acc += matrix[i as usize * s_width as usize + col as usize];
        i += wg_size;
    }

    let mut shared = SharedMemory::<A>::new(WORKGROUP_128 as usize);
    shared[tx as usize] = acc;
    sync_cube();

    // Pairwise tree reduction for REDUCE_WG = 128. If REDUCE_WG changes,
    // update the unrolled steps to match.
    if tx < 64u32 {
        let other = shared[(tx + 64u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 32u32 {
        let other = shared[(tx + 32u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 16u32 {
        let other = shared[(tx + 16u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 8u32 {
        let other = shared[(tx + 8u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 4u32 {
        let other = shared[(tx + 4u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 2u32 {
        let other = shared[(tx + 2u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 1u32 {
        let other = shared[(tx + 1u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();

    if tx == 0u32 {
        out[col as usize] = shared[0];
    }
}

/// Weighted column sums of a dense matrix:
/// `out[col] = sum_i w[i] * M[i, col]`.
///
/// Used to precompute the forward-SpMM correction `c = mu^T * X_scaled`.
/// Pairwise tree reduction in shared memory bounds error growth at
/// O(log n); this matters because `c` then cancels against the sparse dot
/// product inside SpMM, so any precision loss here shows up amplified.
///
/// ### Params
///
/// * `weights` - Weight vector `[n_rows]` (the column means in PCA use)
/// * `matrix` - Input matrix `[n_rows, s_width]` row-major
/// * `out` - Output `[s_width]`
/// * `n_rows` - Number of rows
/// * `s_width` - Number of columns
/// * `wg_size` - Workgroup size (comptime, must be a power of two)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> output column index
#[cube(launch_unchecked)]
pub fn dense_column_weighted_sum<A: Float>(
    weights: &Tensor<A>,
    matrix: &Tensor<A>,
    out: &mut Tensor<A>,
    n_rows: u32,
    s_width: u32,
    #[comptime] wg_size: u32,
) {
    let col = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if col >= s_width {
        terminate!();
    }

    let tx = UNIT_POS_X;

    let mut acc = A::new(0.0);
    let mut i = tx;
    while i < n_rows {
        acc += weights[i as usize] * matrix[i as usize * s_width as usize + col as usize];
        i += wg_size;
    }

    let mut shared = SharedMemory::<A>::new(WORKGROUP_128 as usize);
    shared[tx as usize] = acc;
    sync_cube();

    // Pairwise tree reduction for REDUCE_WG = 128. If REDUCE_WG changes,
    // update the unrolled steps to match.
    if tx < 64u32 {
        let other = shared[(tx + 64u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 32u32 {
        let other = shared[(tx + 32u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 16u32 {
        let other = shared[(tx + 16u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 8u32 {
        let other = shared[(tx + 8u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 4u32 {
        let other = shared[(tx + 4u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 2u32 {
        let other = shared[(tx + 2u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();
    if tx < 1u32 {
        let other = shared[(tx + 1u32) as usize];
        shared[tx as usize] += other;
    }
    sync_cube();

    if tx == 0u32 {
        out[col as usize] = shared[0];
    }
}

///////////////
// Launchers //
///////////////

/// Dispatch [`spmm_csr_forward`] with shape and layout checks on the
/// sparse matrix.
///
/// The dense tensors are not shape-checked here because `GpuTensor` does
/// not expose its shape. The caller (typically the SVD driver) is
/// responsible for allocating `x`, `correction`, and `y` with the correct
/// dimensions.
///
/// ### Params
///
/// * `sparse` - CSR of A, shape `(n, m)`
/// * `x` - Dense RHS `[m, s_width]` row-major
/// * `correction` - Precomputed correction `[s_width]` (e.g. from
///   [`launch_dense_column_weighted_sum`])
/// * `y` - Dense output `[n, s_width]` row-major
/// * `s_width` - Output width
/// * `client` - CubeCL compute client
///
/// ### Errors
///
/// * `GpuSparseLayoutMismatch` if `sparse.cs_type` is not CSR.
#[allow(clippy::too_many_arguments)]
pub fn launch_spmm_csr_forward<R, S, A>(
    sparse: &GpuCompressedSparseData<R, S>,
    x: &GpuTensor<R, A>,
    correction: &GpuTensor<R, A>,
    row_offsets: &GpuTensor<R, A>,
    x_sum: &GpuTensor<R, A>,
    y: &GpuTensor<R, A>,
    s_width: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement,
{
    if !sparse.cs_type.is_csr() {
        return Err(BixverseErrors::SparseLayoutMismatch {
            expected: CompressedSparseFormat::Csr,
            got: sparse.cs_type,
        });
    }

    let (n, _m) = sparse.shape;
    let (gx, gy) = grid_2d(n as u32);

    unsafe {
        spmm_csr_forward::launch_unchecked::<S, A, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            sparse.indptr.clone().into_tensor_arg(),
            sparse.indices.clone().into_tensor_arg(),
            sparse.values.clone().into_tensor_arg(),
            x.clone().into_tensor_arg(),
            correction.clone().into_tensor_arg(),
            row_offsets.clone().into_tensor_arg(),
            x_sum.clone().into_tensor_arg(),
            y.clone().into_tensor_arg(),
            n as u32,
            s_width as u32,
            WORKGROUP_128,
        );
    }

    Ok(())
}

/// Dispatch [`spmm_csc_transpose`] with shape and layout checks on the
/// sparse matrix.
///
/// As with the forward launcher, dense tensors are not shape-checked here.
///
/// ### Params
///
/// * `sparse` - CSC of A, shape `(n, m)`
/// * `q` - Dense RHS `[n, s_width]` row-major
/// * `q_sum` - Precomputed column sums of Q `[s_width]` (e.g. from
///   [`launch_dense_column_sum`])
/// * `mu` - Column means of A `[m]`
/// * `sigma` - Column standard deviations of A `[m]`, floored > 0
/// * `z` - Dense output `[m, s_width]` row-major
/// * `s_width` - Output width
/// * `client` - CubeCL compute client
///
/// ### Errors
///
/// * `GpuSparseLayoutMismatch` if `sparse.cs_type` is not CSC.
#[allow(clippy::too_many_arguments)]
pub fn launch_spmm_csc_transpose<R, S, A>(
    sparse: &GpuCompressedSparseData<R, S>,
    q: &GpuTensor<R, A>,
    q_sum: &GpuTensor<R, A>,
    mu: &GpuTensor<R, A>,
    sigma: &GpuTensor<R, A>,
    m_dot_q: &GpuTensor<R, A>,
    z: &GpuTensor<R, A>,
    s_width: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement,
{
    if !sparse.cs_type.is_csc() {
        return Err(BixverseErrors::SparseLayoutMismatch {
            expected: CompressedSparseFormat::Csc,
            got: sparse.cs_type,
        });
    }

    let (_n, m) = sparse.shape;
    let (gx, gy) = grid_2d(m as u32);

    unsafe {
        spmm_csc_transpose::launch_unchecked::<S, A, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            sparse.indptr.clone().into_tensor_arg(),
            sparse.indices.clone().into_tensor_arg(),
            sparse.values.clone().into_tensor_arg(),
            q.clone().into_tensor_arg(),
            q_sum.clone().into_tensor_arg(),
            mu.clone().into_tensor_arg(),
            sigma.clone().into_tensor_arg(),
            m_dot_q.clone().into_tensor_arg(),
            z.clone().into_tensor_arg(),
            m as u32,
            s_width as u32,
            WORKGROUP_128,
        );
    }

    Ok(())
}

/// Dispatch [`dense_column_sum`]. One workgroup per output column.
pub fn launch_dense_column_sum<R, A>(
    matrix: &GpuTensor<R, A>,
    out: &GpuTensor<R, A>,
    n_rows: usize,
    s_width: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    A: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(s_width as u32);

    unsafe {
        dense_column_sum::launch_unchecked::<A, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            matrix.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            n_rows as u32,
            s_width as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`dense_column_weighted_sum`]. One workgroup per output column.
pub fn launch_dense_column_weighted_sum<R, A>(
    weights: &GpuTensor<R, A>,
    matrix: &GpuTensor<R, A>,
    out: &GpuTensor<R, A>,
    n_rows: usize,
    s_width: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    A: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(s_width as u32);

    unsafe {
        dense_column_weighted_sum::launch_unchecked::<A, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            weights.clone().into_tensor_arg(),
            matrix.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            n_rows as u32,
            s_width as u32,
            WORKGROUP_128,
        );
    }
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

    /////////////
    // Helpers //
    /////////////

    fn dense_to_csr(a: &[f32], n: usize, m: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for i in 0..n {
            for j in 0..m {
                let v = a[i * m + j];
                if v != 0.0 {
                    values.push(v);
                    indices.push(j as u32);
                }
            }
            indptr.push(values.len() as u32);
        }
        (values, indices, indptr)
    }

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

    #[allow(clippy::too_many_arguments)]
    fn cpu_spmm_csr_forward(
        a: &[f32],
        x: &[f32],
        correction: &[f32],
        row_offsets: &[f32],
        x_sum: &[f32],
        n: usize,
        m: usize,
        s: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; n * s];
        for i in 0..n {
            for col in 0..s {
                let mut acc = 0.0f32;
                for j in 0..m {
                    acc += a[i * m + j] * x[j * s + col];
                }
                y[i * s + col] = acc - correction[col] - row_offsets[i] * x_sum[col];
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_spmm_csc_transpose(
        a: &[f32],
        q: &[f32],
        q_sum: &[f32],
        mu: &[f32],
        sigma: &[f32],
        m_dot_q: &[f32],
        n: usize,
        m: usize,
        s: usize,
    ) -> Vec<f32> {
        let mut z = vec![0.0f32; m * s];
        for j in 0..m {
            for col in 0..s {
                let mut acc = 0.0f32;
                for i in 0..n {
                    acc += a[i * m + j] * q[i * s + col];
                }
                z[j * s + col] = (acc - mu[j] * q_sum[col] - m_dot_q[col]) / sigma[j];
            }
        }
        z
    }

    fn cpu_column_sum(matrix: &[f32], n_rows: usize, s: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; s];
        for i in 0..n_rows {
            for col in 0..s {
                out[col] += matrix[i * s + col];
            }
        }
        out
    }

    fn cpu_column_weighted_sum(w: &[f32], matrix: &[f32], n_rows: usize, s: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; s];
        for i in 0..n_rows {
            for col in 0..s {
                out[col] += w[i] * matrix[i * s + col];
            }
        }
        out
    }

    fn assert_vec_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for j in 0..got.len() {
            assert!(
                (got[j] - want[j]).abs() < tol,
                "elem {}: {} != {} (diff {})",
                j,
                got[j],
                want[j],
                (got[j] - want[j]).abs()
            );
        }
    }

    fn run_column_sum(
        matrix: &[f32],
        n_rows: usize,
        s_width: usize,
        device: &WgpuDevice,
    ) -> Vec<f32> {
        let client = WgpuRuntime::client(device);
        let m_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(matrix, vec![n_rows, s_width], &client);
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; s_width],
            vec![s_width],
            &client,
        );
        launch_dense_column_sum(&m_gpu, &out_gpu, n_rows, s_width, &client);
        out_gpu.read(&client).unwrap()
    }

    fn run_column_weighted_sum(
        weights: &[f32],
        matrix: &[f32],
        n_rows: usize,
        s_width: usize,
        device: &WgpuDevice,
    ) -> Vec<f32> {
        let client = WgpuRuntime::client(device);
        let w_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(weights, vec![n_rows], &client);
        let m_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(matrix, vec![n_rows, s_width], &client);
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; s_width],
            vec![s_width],
            &client,
        );
        launch_dense_column_weighted_sum(&w_gpu, &m_gpu, &out_gpu, n_rows, s_width, &client);
        out_gpu.read(&client).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_spmm_csr_forward(
        a: &[f32],
        x: &[f32],
        correction: &[f32],
        row_offsets: &[f32],
        x_sum: &[f32],
        n: usize,
        m: usize,
        s: usize,
        device: &WgpuDevice,
    ) -> Vec<f32> {
        let client = WgpuRuntime::client(device);
        let (values, indices, indptr) = dense_to_csr(a, n, m);
        let sparse = GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
            &values,
            &indices,
            &indptr,
            CompressedSparseFormat::Csr,
            (n, m),
            &client,
        );
        let x_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(x, vec![m, s], &client);
        let corr_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(correction, vec![s], &client);
        let row_offsets_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(row_offsets, vec![n], &client);
        let x_sum_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(x_sum, vec![s], &client);
        let y_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client);

        launch_spmm_csr_forward(
            &sparse,
            &x_gpu,
            &corr_gpu,
            &row_offsets_gpu,
            &x_sum_gpu,
            &y_gpu,
            s,
            &client,
        )
        .unwrap();
        y_gpu.read(&client).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_spmm_csc_transpose(
        a: &[f32],
        q: &[f32],
        q_sum: &[f32],
        mu: &[f32],
        sigma: &[f32],
        m_dot_q: &[f32],
        n: usize,
        m: usize,
        s: usize,
        device: &WgpuDevice,
    ) -> Vec<f32> {
        let client = WgpuRuntime::client(device);
        let (values, indices, indptr) = dense_to_csc(a, n, m);
        let sparse = GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
            &values,
            &indices,
            &indptr,
            CompressedSparseFormat::Csc,
            (n, m),
            &client,
        );
        let q_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(q, vec![n, s], &client);
        let qsum_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(q_sum, vec![s], &client);
        let mu_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(mu, vec![m], &client);
        let sigma_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(sigma, vec![m], &client);
        let m_dot_q_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(m_dot_q, vec![s], &client);
        let z_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m * s], vec![m, s], &client);

        launch_spmm_csc_transpose(
            &sparse,
            &q_gpu,
            &qsum_gpu,
            &mu_gpu,
            &sigma_gpu,
            &m_dot_q_gpu,
            &z_gpu,
            s,
            &client,
        )
        .unwrap();
        z_gpu.read(&client).unwrap()
    }

    ///////////
    // Tests //
    ///////////

    #[test]
    fn test_dense_column_sum() {
        let Some(device) = try_device() else { return };
        let (n, s) = (250, 16);
        let matrix: Vec<f32> = (0..n * s)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.1)
            .collect();

        let got = run_column_sum(&matrix, n, s, &device);
        let want = cpu_column_sum(&matrix, n, s);

        assert_vec_close(&got, &want, 1e-3);
    }

    #[test]
    fn test_dense_column_weighted_sum() {
        let Some(device) = try_device() else { return };
        let (n, s) = (250, 16);
        let matrix: Vec<f32> = (0..n * s)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.1)
            .collect();
        let weights: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) % 17) as f32 * 0.05).collect();

        let got = run_column_weighted_sum(&weights, &matrix, n, s, &device);
        let want = cpu_column_weighted_sum(&weights, &matrix, n, s);

        assert_vec_close(&got, &want, 1e-3);
    }

    #[test]
    fn test_spmm_csr_forward() {
        let Some(device) = try_device() else { return };
        let (n, m, s) = (100, 60, 16);
        let a: Vec<f32> = (0..n * m)
            .map(|i| {
                if i % 5 == 0 {
                    ((i * 7 + 3) % 23) as f32 * 0.2
                } else {
                    0.0
                }
            })
            .collect();
        let x: Vec<f32> = (0..m * s)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.1)
            .collect();
        let correction: Vec<f32> = (0..s).map(|i| (i as f32) * 0.05).collect();
        let row_offsets = vec![0.0f32; n];
        let x_sum = vec![0.0f32; s];

        let got = run_spmm_csr_forward(&a, &x, &correction, &row_offsets, &x_sum, n, m, s, &device);
        let want = cpu_spmm_csr_forward(&a, &x, &correction, &row_offsets, &x_sum, n, m, s);

        assert_vec_close(&got, &want, 1e-3);
    }

    #[test]
    fn test_spmm_csc_transpose() {
        let Some(device) = try_device() else { return };
        let (n, m, s) = (120, 50, 16);
        let a: Vec<f32> = (0..n * m)
            .map(|i| {
                if i % 5 == 0 {
                    ((i * 7 + 3) % 23) as f32 * 0.2
                } else {
                    0.0
                }
            })
            .collect();
        let q: Vec<f32> = (0..n * s)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.1)
            .collect();
        let q_sum = cpu_column_sum(&q, n, s);
        let mu: Vec<f32> = (0..m).map(|j| ((j * 13 + 1) % 11) as f32 * 0.1).collect();
        let sigma: Vec<f32> = (0..m)
            .map(|j| 0.5 + ((j * 5 + 2) % 7) as f32 * 0.1)
            .collect();
        let m_dot_q = vec![0.0f32; s];

        let got = run_spmm_csc_transpose(&a, &q, &q_sum, &mu, &sigma, &m_dot_q, n, m, s, &device);
        let want = cpu_spmm_csc_transpose(&a, &q, &q_sum, &mu, &sigma, &m_dot_q, n, m, s);

        assert_vec_close(&got, &want, 1e-3);
    }

    // Empty rows must emit -correction[col] verbatim.
    #[test]
    fn test_spmm_csr_forward_empty_row() {
        let Some(device) = try_device() else { return };
        let (n, m, s) = (8, 4, 4);
        let mut a = vec![0.0f32; n * m];
        a[1] = 1.5;
        a[7 * m + 3] = -2.0;
        let x: Vec<f32> = (0..m * s).map(|i| (i + 1) as f32 * 0.1).collect();
        let correction: Vec<f32> = (0..s).map(|i| (i + 1) as f32 * 0.25).collect();
        let row_offsets = vec![0.0f32; n];
        let x_sum = vec![0.0f32; s];

        let got = run_spmm_csr_forward(&a, &x, &correction, &row_offsets, &x_sum, n, m, s, &device);
        let want = cpu_spmm_csr_forward(&a, &x, &correction, &row_offsets, &x_sum, n, m, s);
        assert_vec_close(&got, &want, 1e-6);

        for i in 1..7 {
            for col in 0..s {
                assert!(
                    (got[i * s + col] + correction[col]).abs() < 1e-6,
                    "empty row {} col {} not equal to -correction",
                    i,
                    col
                );
            }
        }
    }

    #[test]
    fn test_spmm_csr_forward_layout_mismatch() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);
        let (n, m, s) = (4, 3, 2);
        let mut a = vec![0.0f32; n * m];
        a[0] = 1.0;
        let (values, indices, indptr) = dense_to_csc(&a, n, m);
        let sparse = GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
            &values,
            &indices,
            &indptr,
            CompressedSparseFormat::Csc,
            (n, m),
            &client,
        );
        let x_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m * s], vec![m, s], &client);
        let corr_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client);
        let y_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client);

        let res = launch_spmm_csr_forward(
            &sparse,
            &x_gpu,
            &corr_gpu,
            &GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n], vec![n], &client),
            &GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client),
            &y_gpu,
            s,
            &client,
        );
        assert!(matches!(
            res,
            Err(BixverseErrors::SparseLayoutMismatch { .. })
        ));
    }

    #[test]
    fn test_spmm_csc_transpose_layout_mismatch() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);
        let (n, m, s) = (4, 3, 2);
        let mut a = vec![0.0f32; n * m];
        a[0] = 1.0;
        let (values, indices, indptr) = dense_to_csr(&a, n, m);
        let sparse = GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
            &values,
            &indices,
            &indptr,
            CompressedSparseFormat::Csr,
            (n, m),
            &client,
        );
        let q_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client);
        let qsum_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client);
        let mu_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m], vec![m], &client);
        let sigma_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![1.0f32; m], vec![m], &client);
        let z_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m * s], vec![m, s], &client);

        let res = launch_spmm_csc_transpose(
            &sparse,
            &q_gpu,
            &qsum_gpu,
            &mu_gpu,
            &sigma_gpu,
            &GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client),
            &z_gpu,
            s,
            &client,
        );
        assert!(matches!(
            res,
            Err(BixverseErrors::SparseLayoutMismatch { .. })
        ));
    }
}
