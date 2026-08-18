//! GPU SpMM kernels for sparse-by-dense products with mean centering and
//! column scaling folded in.
//!
//! Both directions of the operator A appearing in randomised SVD are
//! supported:
//!
//! * [`fn@spmm_csr_forward`] computes `Y = A * X - 1 * c^T - r * x_sum^T` where
//!   `c` is a precomputed correction vector and `r` is a per-row offset vector.
//! * [`fn@spmm_csc_transpose`] computes
//!   `Z = (A^T * Q - mu * q_sum^T - 1 * d^T) / sigma` where `d` is a
//!   precomputed column offset vector.
//!
//! [`fn@spmm_csr_plain`] and [`fn@spmm_csc_transpose_plain`] are the same two
//! kernels without any correction terms, for callers whose operator must not be
//! centred or scaled at all. Non-negative matrix factorisation is the case that
//! motivated them.
//!
//! Three small reduction kernels sit alongside:
//!
//! * [`fn@dense_column_weighted_sum`] computes `c = mu^T * X_scaled` for the
//!   forward SpMM.
//! * [`fn@dense_column_sum`] computes `q_sum = 1^T * Q` for the transpose SpMM.
//! * [`fn@dense_column_sq_norm`] computes per-column sums of squares, for
//!   callers needing column L2 norms.
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

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use crate::gpu::{WORKGROUP_64, WORKGROUP_128, WORKGROUP_256};

/////////////
// Helpers //
/////////////

/// Pick the SpMM workgroup width from the dense width `s`.
///
/// Both SpMM kernels stride their column loop by the workgroup width, so a
/// width below `s` re-streams the whole non-zero segment of every row once per
/// extra pass. At the single-cell PCA default `s` is 130 against a 128-wide
/// workgroup, which costs a second full pass of the indices and values for two
/// columns of useful work.
///
/// Rounding up leaves threads that never enter the loop. That is fine: idle
/// threads cost nothing, and the SIMD groups they sit in still help hide
/// memory latency.
///
/// ### Params
///
/// * `s_width` - Number of dense columns
///
/// ### Returns
///
/// Workgroup width, one of 64, 128 or 256.
fn spmm_workgroup(s_width: usize) -> u32 {
    match s_width {
        0..=64 => WORKGROUP_64,
        65..=128 => WORKGROUP_128,
        _ => WORKGROUP_256,
    }
}
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

/// Plain forward SpMM: `Y = A * X`.
///
/// [`fn@spmm_csr_forward`] with the centring and offset terms dropped. The
/// same result is reachable by passing zeroed correction vectors, but that
/// costs three dummy allocations and two subtractions in the innermost loop of
/// what is the dominant kernel of an NMF sweep. Non-negative factorisation must
/// not centre its input at all, so the plain form is also the honest API.
///
/// ### Params
///
/// * `indptr` - CSR row pointers `[n + 1]`
/// * `indices` - Column indices of nnz `[nnz]`
/// * `values` - Values of nnz `[nnz]` in storage precision `S`
/// * `x` - Dense RHS `[m, s]` row-major in accumulator precision `A`
/// * `y` - Dense output `[n, s]` row-major in `A`
/// * `n_rows` - Number of output rows
/// * `s_width` - Output width
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> output row index
/// * `UNIT_POS_X` -> stride offset over output columns
#[cube(launch_unchecked)]
pub fn spmm_csr_plain<S: Float, A: Float>(
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    values: &Tensor<S>,
    x: &Tensor<A>,
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
        y[row as usize * s_width as usize + col as usize] = acc;
        col += wg_size;
    }
}

/// Plain transpose SpMM: `Z = A^T * Q`.
///
/// [`fn@spmm_csc_transpose`] with the centring and scaling terms dropped, for
/// the same reasons as [`fn@spmm_csr_plain`]. Uses the CSC of A, which is
/// structurally a CSR of `A^T`.
///
/// ### Params
///
/// * `indptr` - CSC column pointers `[m + 1]`
/// * `indices` - Row indices of nnz `[nnz]`
/// * `values` - Values of nnz `[nnz]` in storage precision `S`
/// * `q` - Dense RHS `[n, s]` row-major in accumulator precision `A`
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
pub fn spmm_csc_transpose_plain<S: Float, A: Float>(
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    values: &Tensor<S>,
    q: &Tensor<A>,
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

/// Column sums of squares of a dense matrix: `out[col] = sum_i M[i, col]^2`.
///
/// Sibling of [`fn@dense_column_sum`], same threading and the same pinned
/// 128-wide reduction ladder. Squaring on load rather than pre-squaring the
/// matrix keeps this to one pass and needs no scratch buffer. Callers wanting
/// an L2 norm take the square root on the consuming side.
///
/// ### Params
///
/// * `matrix` - Input matrix `[n_rows, s_width]` row-major
/// * `out` - Output column sums of squares `[s_width]`
/// * `n_rows` - Number of rows to sum over
/// * `s_width` - Number of columns
/// * `wg_size` - Workgroup size (comptime, must be a power of two)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> output column index
#[cube(launch_unchecked)]
pub fn dense_column_sq_norm<A: Float>(
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
        let v = matrix[i as usize * s_width as usize + col as usize];
        acc += v * v;
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

/// Dispatch [`fn@spmm_csr_forward`] with shape and layout checks on the
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
/// * `SparseLayoutMismatch` if `sparse.cs_type` is not CSR.
/// * `CubeclUtils` if the grid is over the device's cube-count limit.
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
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n as u32, &limits)?;
    let count = CubeCount::Static(gx, gy, 1);

    macro_rules! dispatch {
        ($wg:expr) => {
            unsafe {
                spmm_csr_forward::launch_unchecked::<S, A, R>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
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
                    $wg,
                );
            }
        };
    }

    match spmm_workgroup(s_width) {
        WORKGROUP_64 => dispatch!(WORKGROUP_64),
        WORKGROUP_128 => dispatch!(WORKGROUP_128),
        _ => dispatch!(WORKGROUP_256),
    }

    Ok(())
}

/// Dispatch [`fn@spmm_csc_transpose`] with shape and layout checks on the
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
/// * `SparseLayoutMismatch` if `sparse.cs_type` is not CSC.
/// * `CubeclUtils` if the grid is over the device's cube-count limit.
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
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(m as u32, &limits)?;
    let count = CubeCount::Static(gx, gy, 1);

    macro_rules! dispatch {
        ($wg:expr) => {
            unsafe {
                spmm_csc_transpose::launch_unchecked::<S, A, R>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
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
                    $wg,
                );
            }
        };
    }

    match spmm_workgroup(s_width) {
        WORKGROUP_64 => dispatch!(WORKGROUP_64),
        WORKGROUP_128 => dispatch!(WORKGROUP_128),
        _ => dispatch!(WORKGROUP_256),
    }

    Ok(())
}

/// Dispatch [`fn@spmm_csr_plain`] with a layout check on the sparse matrix.
///
/// As with the correcting launchers, dense tensors are not shape-checked here.
///
/// ### Params
///
/// * `sparse` - CSR of A, shape `(n, m)`
/// * `x` - Dense RHS `[m, s_width]` row-major
/// * `y` - Dense output `[n, s_width]` row-major
/// * `s_width` - Output width
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, with `y` holding `A * X`.
///
/// ### Errors
///
/// * `SparseLayoutMismatch` if `sparse.cs_type` is not CSR.
/// * `CubeclUtils` if the grid is over the device's cube-count limit.
pub fn launch_spmm_csr_plain<R, S, A>(
    sparse: &GpuCompressedSparseData<R, S>,
    x: &GpuTensor<R, A>,
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
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n as u32, &limits)?;
    let count = checked_cube_count("spmm_csr_plain", gx, gy, 1, &limits)?;

    macro_rules! dispatch {
        ($wg:expr) => {
            unsafe {
                spmm_csr_plain::launch_unchecked::<S, A, R>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
                    sparse.indptr.clone().into_tensor_arg(),
                    sparse.indices.clone().into_tensor_arg(),
                    sparse.values.clone().into_tensor_arg(),
                    x.clone().into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    n as u32,
                    s_width as u32,
                    $wg,
                );
            }
        };
    }

    match spmm_workgroup(s_width) {
        WORKGROUP_64 => dispatch!(WORKGROUP_64),
        WORKGROUP_128 => dispatch!(WORKGROUP_128),
        _ => dispatch!(WORKGROUP_256),
    }

    Ok(())
}

/// Dispatch [`fn@spmm_csc_transpose_plain`] with a layout check on the sparse
/// matrix.
///
/// As with the correcting launchers, dense tensors are not shape-checked here.
///
/// ### Params
///
/// * `sparse` - CSC of A, shape `(n, m)`
/// * `q` - Dense RHS `[n, s_width]` row-major
/// * `z` - Dense output `[m, s_width]` row-major
/// * `s_width` - Output width
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, with `z` holding `A^T * Q`.
///
/// ### Errors
///
/// * `SparseLayoutMismatch` if `sparse.cs_type` is not CSC.
/// * `CubeclUtils` if the grid is over the device's cube-count limit.
pub fn launch_spmm_csc_transpose_plain<R, S, A>(
    sparse: &GpuCompressedSparseData<R, S>,
    q: &GpuTensor<R, A>,
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
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(m as u32, &limits)?;
    let count = checked_cube_count("spmm_csc_transpose_plain", gx, gy, 1, &limits)?;

    macro_rules! dispatch {
        ($wg:expr) => {
            unsafe {
                spmm_csc_transpose_plain::launch_unchecked::<S, A, R>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
                    sparse.indptr.clone().into_tensor_arg(),
                    sparse.indices.clone().into_tensor_arg(),
                    sparse.values.clone().into_tensor_arg(),
                    q.clone().into_tensor_arg(),
                    z.clone().into_tensor_arg(),
                    m as u32,
                    s_width as u32,
                    $wg,
                );
            }
        };
    }

    match spmm_workgroup(s_width) {
        WORKGROUP_64 => dispatch!(WORKGROUP_64),
        WORKGROUP_128 => dispatch!(WORKGROUP_128),
        _ => dispatch!(WORKGROUP_256),
    }

    Ok(())
}

/// Dispatch [`fn@dense_column_sq_norm`]. One workgroup per output column.
///
/// ### Params
///
/// * `matrix` - Dense input `[n_rows, s_width]` row-major
/// * `out` - Dense output `[s_width]`, the per-column sum of squares
/// * `n_rows` - Number of rows in `matrix`
/// * `s_width` - Number of columns in `matrix`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtils` if the grid busts the device's cube-count limit.
pub fn launch_dense_column_sq_norm<R, A>(
    matrix: &GpuTensor<R, A>,
    out: &GpuTensor<R, A>,
    n_rows: usize,
    s_width: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    A: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(s_width as u32, &limits)?;
    let count = checked_cube_count("dense_column_sq_norm", gx, gy, 1, &limits)?;

    unsafe {
        dense_column_sq_norm::launch_unchecked::<A, R>(
            client,
            count,
            CubeDim::new_1d(WORKGROUP_128),
            matrix.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            n_rows as u32,
            s_width as u32,
            WORKGROUP_128,
        );
    }

    Ok(())
}

/// Dispatch [`fn@dense_column_sum`]. One workgroup per output column.
///
/// ### Params
///
/// * `matrix` - Dense input `[n_rows, s_width]` row-major
/// * `out` - Dense output `[s_width]`
/// * `n_rows` - Number of rows in `matrix`
/// * `s_width` - Number of columns in `matrix`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtils` if the grid busts the device's cube-count limit.
pub fn launch_dense_column_sum<R, A>(
    matrix: &GpuTensor<R, A>,
    out: &GpuTensor<R, A>,
    n_rows: usize,
    s_width: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    A: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(s_width as u32, &limits)?;

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

    Ok(())
}

/// Dispatch [`fn@dense_column_weighted_sum`]. One workgroup per output column.
///
/// ### Params
///
/// * `weights` - Per-row weights `[n_rows]`
/// * `matrix` - Dense input `[n_rows, s_width]` row-major
/// * `out` - Dense output `[s_width]`
/// * `n_rows` - Number of rows in `matrix`
/// * `s_width` - Number of columns in `matrix`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtils` if the grid busts the device's cube-count limit.
pub fn launch_dense_column_weighted_sum<R, A>(
    weights: &GpuTensor<R, A>,
    matrix: &GpuTensor<R, A>,
    out: &GpuTensor<R, A>,
    n_rows: usize,
    s_width: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    A: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(s_width as u32, &limits)?;

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

    Ok(())
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
            GpuTensor::<WgpuRuntime, f32>::from_slice(matrix, vec![n_rows, s_width], &client)
                .unwrap();
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; s_width],
            vec![s_width],
            &client,
        )
        .unwrap();
        launch_dense_column_sum(&m_gpu, &out_gpu, n_rows, s_width, &client).unwrap();
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
        let w_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(weights, vec![n_rows], &client).unwrap();
        let m_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(matrix, vec![n_rows, s_width], &client)
                .unwrap();
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; s_width],
            vec![s_width],
            &client,
        )
        .unwrap();
        launch_dense_column_weighted_sum(&w_gpu, &m_gpu, &out_gpu, n_rows, s_width, &client)
            .unwrap();
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
        )
        .unwrap();
        let x_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(x, vec![m, s], &client).unwrap();
        let corr_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(correction, vec![s], &client).unwrap();
        let row_offsets_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(row_offsets, vec![n], &client).unwrap();
        let x_sum_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(x_sum, vec![s], &client).unwrap();
        let y_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client)
                .unwrap();

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
        )
        .unwrap();
        let q_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(q, vec![n, s], &client).unwrap();
        let qsum_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(q_sum, vec![s], &client).unwrap();
        let mu_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(mu, vec![m], &client).unwrap();
        let sigma_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(sigma, vec![m], &client).unwrap();
        let m_dot_q_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(m_dot_q, vec![s], &client).unwrap();
        let z_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m * s], vec![m, s], &client)
                .unwrap();

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

    /// Column-sum reduction against the host, past a single workgroup of rows.
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

    /// Same reduction with per-row weights, a separate kernel that can drift.
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

    /// Fused CSR forward product against the host, corrections and all.
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

    /// Transposed CSC product against the host, with non-trivial mu and sigma.
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

    /// A CSC operand into the CSR launcher must error, not be reinterpreted.
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
        )
        .unwrap();
        let x_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m * s], vec![m, s], &client)
                .unwrap();
        let corr_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client).unwrap();
        let y_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client)
                .unwrap();

        let res = launch_spmm_csr_forward(
            &sparse,
            &x_gpu,
            &corr_gpu,
            &GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n], vec![n], &client).unwrap(),
            &GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client).unwrap(),
            &y_gpu,
            s,
            &client,
        );
        assert!(matches!(
            res,
            Err(BixverseErrors::SparseLayoutMismatch { .. })
        ));
    }

    /// The mirror guard: a CSR operand into the CSC transpose launcher.
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
        )
        .unwrap();
        let q_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client)
                .unwrap();
        let qsum_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client).unwrap();
        let mu_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m], vec![m], &client).unwrap();
        let sigma_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![1.0f32; m], vec![m], &client).unwrap();
        let z_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; m * s], vec![m, s], &client)
                .unwrap();

        let res = launch_spmm_csc_transpose(
            &sparse,
            &q_gpu,
            &qsum_gpu,
            &mu_gpu,
            &sigma_gpu,
            &GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; s], vec![s], &client).unwrap(),
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
