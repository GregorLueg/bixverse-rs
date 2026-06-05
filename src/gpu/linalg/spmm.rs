//! GPU SpMM kernels for sparse-by-dense products with mean centering and
//! column scaling folded in.
//!
//! Both directions of the operator A appearing in randomised SVD are
//! supported:
//!
//! * [`spmm_csr_forward`] computes `Y = A * X - 1 * c^T` where `c` is a
//!   precomputed correction vector. The sparse matrix is held as CSR of A;
//!   the dense RHS X is row-major `[m, s]`; the output Y is row-major
//!   `[n, s]`.
//! * [`spmm_csc_transpose`] computes `Z = (A^T * Q - mu * q_sum^T) / sigma`
//!   using CSC of A (structurally a CSR of A^T). The dense RHS Q is
//!   row-major `[n, s]`; the output Z is row-major `[m, s]`.
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

use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Workgroup size for the SpMM kernels.
const SPMM_WG: u32 = 128;

/// Workgroup size for the reduction kernels. Must be a power of two.
const REDUCE_WG: u32 = 128;

/// Shared-memory size for the reduction kernels. Must equal `REDUCE_WG`;
/// kept as a separate constant because the cubecl macro does not expand
/// comptime function parameters inside `SharedMemory::new` calls.
const REDUCE_SMEM: u32 = REDUCE_WG;

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

    let mut shared = SharedMemory::<A>::new(REDUCE_SMEM as usize);
    shared[tx as usize] = acc;
    sync_cube();

    let mut stride = REDUCE_SMEM / 2u32;
    while stride > 0u32 {
        if tx < stride {
            let other = shared[(tx + stride) as usize];
            shared[tx as usize] += other;
        }
        sync_cube();
        stride /= 2u32;
    }

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

    let mut shared = SharedMemory::<A>::new(REDUCE_SMEM as usize);
    shared[tx as usize] = acc;
    sync_cube();

    let mut stride = REDUCE_SMEM / 2u32;
    while stride > 0u32 {
        if tx < stride {
            let other = shared[(tx + stride) as usize];
            shared[tx as usize] += other;
        }
        sync_cube();
        stride /= 2u32;
    }

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
pub fn launch_spmm_csr_forward<R, S, A>(
    sparse: &GpuCompressedSparseData<R, S>,
    x: &GpuTensor<R, A>,
    correction: &GpuTensor<R, A>,
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
            CubeDim::new_1d(SPMM_WG),
            sparse.indptr.clone().into_tensor_arg(),
            sparse.indices.clone().into_tensor_arg(),
            sparse.values.clone().into_tensor_arg(),
            x.clone().into_tensor_arg(),
            correction.clone().into_tensor_arg(),
            y.clone().into_tensor_arg(),
            n as u32,
            s_width as u32,
            SPMM_WG,
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
            CubeDim::new_1d(SPMM_WG),
            sparse.indptr.clone().into_tensor_arg(),
            sparse.indices.clone().into_tensor_arg(),
            sparse.values.clone().into_tensor_arg(),
            q.clone().into_tensor_arg(),
            q_sum.clone().into_tensor_arg(),
            mu.clone().into_tensor_arg(),
            sigma.clone().into_tensor_arg(),
            z.clone().into_tensor_arg(),
            m as u32,
            s_width as u32,
            SPMM_WG,
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
            CubeDim::new_1d(REDUCE_WG),
            matrix.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            n_rows as u32,
            s_width as u32,
            REDUCE_WG,
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
            CubeDim::new_1d(REDUCE_WG),
            weights.clone().into_tensor_arg(),
            matrix.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            n_rows as u32,
            s_width as u32,
            REDUCE_WG,
        );
    }
}
