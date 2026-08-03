//! GPU kernels for the SEACells Frank-Wolfe updates.
//!
//! [fw_argmin_b] replaces `fw_argmins_b`, the per-archetype gradient scan that
//! dominates the CPU runtime. The remaining phases stay on the host.
//!
//! Layout convention: every `n × k` matrix (`K²B`, `K²Aᵀ`, `B`) is passed as CSR
//! over cells, the untransposed output of `k_squared_matmul`, so the GPU path
//! drops the transposes the CPU scan needs rather than adding any. `t1 = A Aᵀ`
//! is dense `[k, k]`: it runs close to dense in practice and at `4k²` bytes stays
//! cache-resident.
//!
//! Nothing here allocates an `n × k` dense buffer. The gradient is never
//! materialised: each thread owns a strided slice of the `k` columns in
//! registers and reduces it to a running `(min, idx)` as it goes.

#![allow(missing_docs)]

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;
use cubecl::prelude::*;

use crate::errors::BixverseErrors;
use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::gpu::*;
use crate::prelude::CompressedSparseFormat;

////////////
// Consts //
////////////

/// Workgroups for the B-gradient argmin, grid-striding over cells.
///
/// Fixed rather than derived from `n`, so the partial buffer stays
/// `B_ARGMIN_BLOCKS * k` at every dataset size instead of growing with it.
pub const B_ARGMIN_BLOCKS: u32 = 1024;

/// Workgroup width for [fw_argmin_b].
///
/// Sets how many columns each thread owns, `ceil(k / B_ARGMIN_WG)`. Wider means
/// fewer register slots per thread but more idle threads at small `k`.
pub const B_ARGMIN_WG: u32 = 256;

/// Workgroup width for [reduce_argmin_blocks]. One thread per output column,
/// grid-striding.
pub const B_REDUCE_WG: u32 = 256;

/////////////
// Kernels //
/////////////

/// Fused Frank-Wolfe gradient and column-wise argmin for the B update.
///
/// Computes, without ever storing it,
///
/// ```text
/// G[i, c] = sum_m K²B[i, m] * t1[m, c] - K²Aᵀ[i, c]
/// ```
///
/// and reduces it to `argmin_i G[i, c]` for every archetype `c`, plus the
/// `sum(B * G)` half of the Frank-Wolfe duality gap. Work is `nnz(K²B) * k`
/// fused multiply-adds, the same count as the CPU scan it replaces; the win is
/// throughput, not a better algorithm.
///
/// Thread `tx` owns columns `tx, tx + wg, tx + 2 wg, ...`, held in registers
/// rather than shared memory, which keeps the `t1` row read contiguous across a
/// workgroup and leaves no `k`-dependent shared memory budget to gate on. Every
/// register-array index is comptime, since a dynamically indexed local array is
/// backed by global memory on Metal.
///
/// Rows are visited in increasing order with a strict `<`, so the lowest row
/// index wins a tie within a block. Blocks grid-stride, so block order does not
/// follow row order and [reduce_argmin_blocks] has to break ties by index
/// explicitly.
///
/// ### Params
///
/// * `k2b_indptr` - CSR row pointers of `K²B` `[n + 1]`
/// * `k2b_indices` - Archetype indices of its non-zeros `[nnz]`
/// * `k2b_values` - Values of its non-zeros `[nnz]`
/// * `t1` - `A Aᵀ` dense `[k, k]` row-major
/// * `t2_indptr` - CSR row pointers of `K²Aᵀ` `[n + 1]`
/// * `t2_indices` - Archetype indices of its non-zeros `[nnz]`
/// * `t2_values` - Values of its non-zeros `[nnz]`
/// * `b_indptr` - CSR row pointers of `B` `[n + 1]`
/// * `b_indices` - Archetype indices of its non-zeros `[nnz]`
/// * `b_values` - Values of its non-zeros `[nnz]`
/// * `best_val` - Output partial minima `[n_blocks, k]`
/// * `best_idx` - Output partial argmins `[n_blocks, k]`
/// * `gap_partial` - Output partial `sum(B * G)` `[n_blocks]`
/// * `n` - Number of cells
/// * `k` - Number of archetypes (comptime)
/// * `slots` - Columns owned per thread, `ceil(k / wg_size)` (comptime)
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> block index, then grid-strides
///   over cells
/// * `UNIT_POS_X` -> first owned column, then strides by `wg_size`
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
pub fn fw_argmin_b<F: Float>(
    k2b_indptr: &Tensor<u32>,
    k2b_indices: &Tensor<u32>,
    k2b_values: &Tensor<F>,
    t1: &Tensor<F>,
    t2_indptr: &Tensor<u32>,
    t2_indices: &Tensor<u32>,
    t2_values: &Tensor<F>,
    b_indptr: &Tensor<u32>,
    b_indices: &Tensor<u32>,
    b_values: &Tensor<F>,
    best_val: &mut Tensor<F>,
    best_idx: &mut Tensor<u32>,
    gap_partial: &mut Tensor<F>,
    n: u32,
    #[comptime] k: usize,
    #[comptime] slots: usize,
    #[comptime] wg_size: u32,
) {
    let block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let tx = UNIT_POS_X;

    let mut acc = Array::<F>::new(slots);
    let mut run_val = Array::<F>::new(slots);
    let mut run_idx = Array::<u32>::new(slots);

    #[unroll]
    for s in 0..slots {
        run_val[s] = F::new(f32::MAX);
        run_idx[s] = 0u32;
    }

    let mut gap = F::new(0.0);

    // Grid-stride over the cells this block owns.
    let mut row = block;
    while row < n {
        let row_us = row as usize;

        #[unroll]
        for s in 0..slots {
            acc[s] = F::new(0.0);
        }

        // acc[c] = sum_m K²B[row, m] * t1[m, c]
        let k2b_start = k2b_indptr[row_us];
        let k2b_end = k2b_indptr[row_us + 1];
        let mut p = k2b_start;
        while p < k2b_end {
            let m = k2b_indices[p as usize] as usize;
            let v = k2b_values[p as usize];
            let t1_base = m * k;

            #[unroll]
            for s in 0..slots {
                let col = tx + comptime!(s as u32) * wg_size;
                if col < comptime!(k as u32) {
                    acc[s] += v * t1[t1_base + col as usize];
                }
            }
            p += 1u32;
        }

        // acc[c] -= K²Aᵀ[row, c]. The owning thread applies each non-zero; the
        // slot is found by comparison rather than by dividing the column index,
        // which would be a dynamic index into a register array.
        let t2_start = t2_indptr[row_us];
        let t2_end = t2_indptr[row_us + 1];
        let mut q = t2_start;
        while q < t2_end {
            let col = t2_indices[q as usize];
            let v = t2_values[q as usize];

            #[unroll]
            for s in 0..slots {
                let owned = tx + comptime!(s as u32) * wg_size;
                if owned == col {
                    acc[s] -= v;
                }
            }
            q += 1u32;
        }

        // gap += sum_c B[row, c] * G[row, c]
        let b_start = b_indptr[row_us];
        let b_end = b_indptr[row_us + 1];
        let mut r = b_start;
        while r < b_end {
            let col = b_indices[r as usize];
            let v = b_values[r as usize];

            #[unroll]
            for s in 0..slots {
                let owned = tx + comptime!(s as u32) * wg_size;
                if owned == col {
                    gap += v * acc[s];
                }
            }
            r += 1u32;
        }

        // Running per-column minimum. Strict `<` keeps the lowest row index.
        #[unroll]
        for s in 0..slots {
            let col = tx + comptime!(s as u32) * wg_size;
            if col < comptime!(k as u32) && acc[s] < run_val[s] {
                run_val[s] = acc[s];
                run_idx[s] = row;
            }
        }

        row += CUBE_COUNT_X * CUBE_COUNT_Y;
    }

    // Write this block's partial minima, coalesced across threads.
    let out_base = block as usize * k;
    #[unroll]
    for s in 0..slots {
        let col = tx + comptime!(s as u32) * wg_size;
        if col < comptime!(k as u32) {
            best_val[out_base + col as usize] = run_val[s];
            best_idx[out_base + col as usize] = run_idx[s];
        }
    }

    // Workgroup tree reduction of the gap partial.
    let mut shared = SharedMemory::<F>::new(wg_size as usize);
    shared[tx as usize] = gap;
    sync_cube();

    // Forced runtime, or cubecl unrolls the loop at expansion.
    let mut stride = (wg_size / 2u32).runtime();
    while stride > 0u32 {
        if tx < stride {
            let other = shared[(tx + stride) as usize];
            shared[tx as usize] += other;
        }
        sync_cube();
        stride /= 2u32;
    }

    if tx == 0u32 {
        gap_partial[block as usize] = shared[0];
    }
}

/// Second stage of the B-gradient argmin: reduce per-block partials to one
/// `(min, idx)` per archetype.
///
/// Blocks grid-stride over cells in [fw_argmin_b], so block order does not
/// follow row order. Ties are therefore broken on the row index explicitly,
/// reproducing the CPU scan's lowest-index-wins behaviour.
///
/// ### Params
///
/// * `part_val` - Partial minima `[n_blocks, k]`
/// * `part_idx` - Partial argmins `[n_blocks, k]`
/// * `out_val` - Output minima `[k]`
/// * `out_idx` - Output argmins `[k]`
/// * `n_blocks` - Number of blocks that contributed
/// * `k` - Number of archetypes
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> column
#[cube(launch_unchecked)]
pub fn reduce_argmin_blocks<F: Float>(
    part_val: &Tensor<F>,
    part_idx: &Tensor<u32>,
    out_val: &mut Tensor<F>,
    out_idx: &mut Tensor<u32>,
    n_blocks: u32,
    k: u32,
    #[comptime] wg_size: u32,
) {
    let col = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if col >= k {
        terminate!();
    }

    let mut best = F::new(f32::MAX);
    let mut best_row = u32::MAX.runtime();

    let mut block = 0u32;
    while block < n_blocks {
        let offset = (block * k + col) as usize;
        let val = part_val[offset];
        let idx = part_idx[offset];
        if val < best || (val == best && idx < best_row) {
            best = val;
            best_row = idx;
        }
        block += 1u32;
    }

    out_val[col as usize] = best;
    out_idx[col as usize] = best_row;
}

///////////////
// Launchers //
///////////////

/// Dispatch [fw_argmin_b] followed by [reduce_argmin_blocks].
///
/// ### Params
///
/// * `k2b` - `K²B` as CSR `n × k`
/// * `t1` - `A Aᵀ` dense `[k, k]` row-major
/// * `t2` - `K²Aᵀ` as CSR `n × k`
/// * `b_mat` - `B` as CSR `n × k`
/// * `part_val` - Scratch `[B_ARGMIN_BLOCKS, k]`
/// * `part_idx` - Scratch `[B_ARGMIN_BLOCKS, k]`
/// * `gap_partial` - Scratch `[B_ARGMIN_BLOCKS]`
/// * `out_val` - Output minima `[k]`
/// * `out_idx` - Output argmins `[k]`
/// * `n` - Number of cells
/// * `k` - Number of archetypes
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `GpuCubeCountExceeded` if a dispatch busts the device limit.
#[allow(clippy::too_many_arguments)]
pub fn launch_fw_argmin_b<R, F>(
    k2b: &GpuCompressedSparseData<R, F>,
    t1: &GpuTensor<R, F>,
    t2: &GpuCompressedSparseData<R, F>,
    b_mat: &GpuCompressedSparseData<R, F>,
    part_val: &GpuTensor<R, F>,
    part_idx: &GpuTensor<R, u32>,
    gap_partial: &GpuTensor<R, F>,
    out_val: &GpuTensor<R, F>,
    out_idx: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    for mat in [k2b, t2, b_mat] {
        if !mat.cs_type.is_csr() {
            return Err(BixverseErrors::SparseLayoutMismatch {
                expected: CompressedSparseFormat::Csr,
                got: mat.cs_type,
            });
        }
    }

    // An over-sized binding is rejected silently: the kernel does no work and
    // returns zeros. Check on the host so it surfaces as an error instead.
    let limit = client.properties().memory.max_page_size as usize;
    let checked = [
        ("K2B values", k2b.nnz * size_of::<F>()),
        ("K2B indices", k2b.nnz * size_of::<u32>()),
        ("K2At values", t2.nnz * size_of::<F>()),
        ("t1", k * k * size_of::<F>()),
        ("argmin partials", part_val.len() * size_of::<F>()),
    ];
    for (buffer, bytes) in checked {
        if bytes > limit {
            return Err(BixverseErrors::GpuBindingTooLarge {
                buffer,
                bytes,
                limit,
            });
        }
    }

    let slots = k.div_ceil(B_ARGMIN_WG as usize);
    let blocks = B_ARGMIN_BLOCKS.min(n.max(1) as u32);

    let (gx, gy) = grid_2d(blocks);
    let count = checked_cube_count::<R>("fw_argmin_b", gx, gy, 1)?;

    unsafe {
        fw_argmin_b::launch_unchecked::<F, R>(
            client,
            count,
            CubeDim::new_1d(B_ARGMIN_WG),
            k2b.indptr.clone().into_tensor_arg(),
            k2b.indices.clone().into_tensor_arg(),
            k2b.values.clone().into_tensor_arg(),
            t1.clone().into_tensor_arg(),
            t2.indptr.clone().into_tensor_arg(),
            t2.indices.clone().into_tensor_arg(),
            t2.values.clone().into_tensor_arg(),
            b_mat.indptr.clone().into_tensor_arg(),
            b_mat.indices.clone().into_tensor_arg(),
            b_mat.values.clone().into_tensor_arg(),
            part_val.clone().into_tensor_arg(),
            part_idx.clone().into_tensor_arg(),
            gap_partial.clone().into_tensor_arg(),
            n as u32,
            k,
            slots,
            B_ARGMIN_WG,
        );
    }

    let reduce_cubes = (k as u32).div_ceil(B_REDUCE_WG);
    let (rx, ry) = grid_2d(reduce_cubes);
    let reduce_count = checked_cube_count::<R>("reduce_argmin_blocks", rx, ry, 1)?;

    unsafe {
        reduce_argmin_blocks::launch_unchecked::<F, R>(
            client,
            reduce_count,
            CubeDim::new_1d(B_REDUCE_WG),
            part_val.clone().into_tensor_arg(),
            part_idx.clone().into_tensor_arg(),
            out_val.clone().into_tensor_arg(),
            out_idx.clone().into_tensor_arg(),
            blocks,
            k as u32,
            B_REDUCE_WG,
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
    use approx::assert_relative_eq;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    /// Skip rather than fail where no GPU is available, matching the other
    /// kernel test modules.
    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Row-major dense to CSR parts.
    ///
    /// ### Params
    ///
    /// * `a` - Dense `[n, m]` row-major
    /// * `n` - Rows
    /// * `m` - Columns
    ///
    /// ### Returns
    ///
    /// `(values, indices, indptr)`.
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

    /// CPU reference for [fw_argmin_b], mirroring `fw_argmins_b` in
    /// `single_cell::mc_generation::seacells`.
    ///
    /// ### Params
    ///
    /// * `k2b` - Dense `[n, k]` row-major
    /// * `t1` - Dense `[k, k]` row-major
    /// * `t2` - Dense `[n, k]` row-major
    /// * `b_mat` - Dense `[n, k]` row-major
    /// * `n` - Rows
    /// * `k` - Columns
    ///
    /// ### Returns
    ///
    /// `(min values [k], argmin rows [k], sum(B * G))`.
    fn cpu_fw_argmin_b(
        k2b: &[f32],
        t1: &[f32],
        t2: &[f32],
        b_mat: &[f32],
        n: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<u32>, f32) {
        let mut best_val = vec![f32::MAX; k];
        let mut best_idx = vec![0u32; k];
        let mut gap = 0.0f32;

        for c in 0..k {
            for i in 0..n {
                let mut g = 0.0f32;
                for m in 0..k {
                    g += k2b[i * k + m] * t1[m * k + c];
                }
                g -= t2[i * k + c];
                if b_mat[i * k + c] != 0.0 {
                    gap += b_mat[i * k + c] * g;
                }
                if g < best_val[c] {
                    best_val[c] = g;
                    best_idx[c] = i as u32;
                }
            }
        }

        (best_val, best_idx, gap)
    }

    /// Run the kernel pair over dense fixtures and return its output.
    ///
    /// ### Params
    ///
    /// * `k2b` - Dense `[n, k]` row-major
    /// * `t1` - Dense `[k, k]` row-major
    /// * `t2` - Dense `[n, k]` row-major
    /// * `b_mat` - Dense `[n, k]` row-major
    /// * `n` - Rows
    /// * `k` - Columns
    /// * `device` - Device to run on
    ///
    /// ### Returns
    ///
    /// `(min values [k], argmin rows [k], sum(B * G))`.
    fn gpu_fw_argmin_b(
        k2b: &[f32],
        t1: &[f32],
        t2: &[f32],
        b_mat: &[f32],
        n: usize,
        k: usize,
        device: &WgpuDevice,
    ) -> (Vec<f32>, Vec<u32>, f32) {
        let client = WgpuRuntime::client(device);

        let upload = |dense: &[f32]| {
            let (values, indices, indptr) = dense_to_csr(dense, n, k);
            GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
                &values,
                &indices,
                &indptr,
                CompressedSparseFormat::Csr,
                (n, k),
                &client,
            )
        };

        let k2b_gpu = upload(k2b);
        let t2_gpu = upload(t2);
        let b_gpu = upload(b_mat);
        let t1_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(t1, vec![k * k], &client);

        let blocks = B_ARGMIN_BLOCKS.min(n.max(1) as u32) as usize;
        let part_val = GpuTensor::<WgpuRuntime, f32>::empty(vec![blocks * k], &client);
        let part_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![blocks * k], &client);
        let gap_partial = GpuTensor::<WgpuRuntime, f32>::empty(vec![blocks], &client);
        let out_val = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client);
        let out_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![k], &client);

        launch_fw_argmin_b(
            &k2b_gpu,
            &t1_gpu,
            &t2_gpu,
            &b_gpu,
            &part_val,
            &part_idx,
            &gap_partial,
            &out_val,
            &out_idx,
            n,
            k,
            &client,
        )
        .expect("launch failed");

        let vals = out_val.read(&client).expect("read failed");
        let idx = out_idx.read(&client).expect("read failed");
        let gaps = gap_partial.read(&client).expect("read failed");
        let gap: f32 = gaps.iter().sum();

        (vals, idx, gap)
    }

    /// Deterministic pseudo-random dense fixture with a controlled zero rate.
    ///
    /// ### Params
    ///
    /// * `len` - Number of values
    /// * `seed` - Offsets the sequence
    /// * `zero_every` - Every n-th value is zeroed, giving a sparse pattern
    ///
    /// ### Returns
    ///
    /// The values.
    fn fixture(len: usize, seed: usize, zero_every: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                if (i + seed).is_multiple_of(zero_every) {
                    0.0
                } else {
                    (((i * 37 + seed * 17) % 23) as f32 - 11.0) * 0.1
                }
            })
            .collect()
    }

    /// The fused kernel must reproduce the CPU scan it replaces: same argmin per
    /// archetype, same minimum, same duality-gap term.
    #[test]
    fn test_fw_argmin_b_matches_cpu() {
        let Some(device) = try_device() else { return };

        for (n, k) in [(40usize, 7usize), (257, 33), (1000, 130)] {
            let k2b = fixture(n * k, 1, 3);
            let t1 = fixture(k * k, 5, 4);
            let t2 = fixture(n * k, 9, 5);
            let b_mat = fixture(n * k, 13, 11);

            let (want_val, want_idx, want_gap) = cpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k);
            let (got_val, got_idx, got_gap) =
                gpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k, &device);

            for c in 0..k {
                assert_relative_eq!(got_val[c], want_val[c], max_relative = 1e-4, epsilon = 1e-5);
                assert_eq!(got_idx[c], want_idx[c], "argmin differs at column {}", c);
            }
            assert_relative_eq!(got_gap, want_gap, max_relative = 1e-3, epsilon = 1e-4);
        }
    }

    /// Rows with no non-zeros in `K²B` still take part in the argmin: their
    /// gradient is `-K²Aᵀ[i, :]`, not a skipped row.
    #[test]
    fn test_fw_argmin_b_handles_empty_rows() {
        let Some(device) = try_device() else { return };

        let (n, k) = (64usize, 9usize);
        let mut k2b = fixture(n * k, 1, 3);
        // Blank every third row of K²B entirely.
        for i in (0..n).step_by(3) {
            for c in 0..k {
                k2b[i * k + c] = 0.0;
            }
        }
        let t1 = fixture(k * k, 5, 4);
        let t2 = fixture(n * k, 9, 5);
        let b_mat = fixture(n * k, 13, 11);

        let (want_val, want_idx, want_gap) = cpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k);
        let (got_val, got_idx, got_gap) = gpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k, &device);

        for c in 0..k {
            assert_relative_eq!(got_val[c], want_val[c], max_relative = 1e-4, epsilon = 1e-5);
            assert_eq!(got_idx[c], want_idx[c], "argmin differs at column {}", c);
        }
        assert_relative_eq!(got_gap, want_gap, max_relative = 1e-3, epsilon = 1e-4);
    }

    /// Ties must resolve to the lowest row index, matching the CPU's strict `<`.
    /// Blocks grid-stride, so block order does not follow row order and this only
    /// holds because `reduce_argmin_blocks` compares indices explicitly.
    #[test]
    fn test_fw_argmin_b_tie_breaks_on_lowest_row() {
        let Some(device) = try_device() else { return };

        let (n, k) = (2048usize, 4usize);
        // Identity t1 and zero t2 make G == K²B exactly, so the gradient is
        // whatever is written here.
        let mut t1 = vec![0.0f32; k * k];
        for c in 0..k {
            t1[c * k + c] = 1.0;
        }
        let t2 = vec![0.0f32; n * k];
        let b_mat = vec![0.0f32; n * k];

        // Every row holds the same value per column, so every row ties.
        let mut k2b = vec![0.0f32; n * k];
        for i in 0..n {
            for c in 0..k {
                k2b[i * k + c] = -1.0 - c as f32;
            }
        }

        let (_, got_idx, _) = gpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k, &device);
        for c in 0..k {
            assert_eq!(got_idx[c], 0, "column {} did not tie-break to row 0", c);
        }
    }
}
