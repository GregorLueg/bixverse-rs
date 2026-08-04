//! GPU kernels for the SEACells Frank-Wolfe updates.
//!
//! [fw_argmin_b] replaces `fw_argmins_b`, the per-archetype gradient scan in the
//! B update, and [fw_columns_a_gpu] replaces `fw_columns_a`, the per-cell column
//! solve in the A update. Kernel construction, archetype initialisation and the
//! RSS stay on the host.
//!
//! Layout convention: every `n × k` matrix (`K²B`, `K²Aᵀ`, `B`) is passed as
//! CSR over cells, the untransposed output of `k_squared_matmul`, so the GPU
//! path drops the transposes the CPU scan needs rather than adding any.
//! `t1 = A Aᵀ` is passed dense `[k, k]`. It is not a dense matrix: 1-7% across
//! every shape benchmarked. It is densified so each thread can index the one
//! column it owns instead of scanning the row, which would put an
//! `nnz(t1 row)`-long search inside the `nnz(K²B)` loop. The cost is `4k²`
//! bytes, so the caller has to bound `k`.
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

/// Workgroups for the A-column solve, grid-striding over cells.
///
/// One workgroup owns one cell at a time and runs that cell's whole Frank-Wolfe
/// loop, so this is a residency knob rather than a partition: the shared state
/// is reset per cell and nothing scales with it.
pub const A_COLUMNS_BLOCKS: u32 = 1024;

/// Workgroup width for [fw_columns_a_gpu].
///
/// Doubles as the atom capacity ceiling, since thread `i` owns atom slot `i`,
/// and a column wider than this falls back to the CPU. Must be a power of two
/// and a whole number of planes.
///
/// Measured at 50k cells and 666 archetypes: 256 gave ~210 ms per call and 128
/// gave ~115 ms. Narrower is better here because the kernel is bound by the
/// reductions rather than the arithmetic, and halving the width doubles the
/// columns each thread owns while halving the planes to combine. 64 would
/// continue the trend but cannot hold the ~100 atoms that shape needs without
/// giving each thread several atom slots.
pub const A_COLUMNS_WG: u32 = 128;

/// Register slots per thread beyond which [fw_columns_a_gpu] declines the work.
///
/// `w` and `k2b_row` are each `Array::<F>::new(slots)` with
/// `slots = ceil(k / A_COLUMNS_WG)`, so the pair costs `2 * slots` floats per
/// thread. On Metal a spilled register array is backed by global memory, which
/// turns the kernel's whole reason for holding the gradient in registers into a
/// slow scatter. At `k = 6666` the pair is 106 floats and will certainly spill.
///
/// This is a conservative bound, not a measured knee: the crossover has not been
/// swept, and doing so is the obvious next step before raising it. Above the
/// bound the caller falls back to the CPU.
pub const A_COLUMNS_MAX_SLOTS: usize = 16;

/// Below this the L1 renormalisation is skipped, mirroring `FW_RENORM_FLOOR` on
/// the CPU side so a fully-pruned column behaves the same on both paths.
///
/// The CPU compares in `f64` and this compares in `f32`; at 1e-15 against
/// weights that sum to 1 the two only disagree for columns that have already
/// collapsed to nothing.
pub const A_RENORM_FLOOR: f32 = 1e-15;

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
        run_val[s] = F::max_value();
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
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` ->
///   column
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

    let mut best = F::max_value();
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

/// Frank-Wolfe column solve for the A update, one workgroup per cell.
///
/// Reproduces `fw_columns_a`: for each cell it seeds the atom set and the
/// gradient state `w = t1 · A_prev[:, cell]`, then runs `n_iters` Frank-Wolfe
/// steps of
///
/// ```text
/// amin = argmin_c (w[c] - K²B[cell, c])
/// w    <- (1 - γ) w + γ t1[amin, :]        γ = 2 / (t + 2)
/// ```
///
/// with the atom weights following the same recurrence. The gradient is never
/// materialised across cells: thread `tx` owns columns `tx, tx + wg, ...` in
/// `slots` registers, so nothing here scales shared memory with `k`.
///
/// Atoms live one per thread, which is why `cap` may not exceed `wg_size`.
/// Pruning marks an atom's weight zero rather than compacting the list, so a
/// slot is never reclaimed and the count is bounded by `seed + n_iters`. That
/// is also why an index that gets pruned and later re-selected lands back in
/// its old slot at weight zero and receives `+= γ`, which is exactly the value
/// the CPU path produces by pushing a fresh atom.
///
/// Ties go to the lowest column index at every level: each thread's slots are
/// visited in increasing column order under a strict `<`, and the tree
/// reduction compares indices explicitly.
///
/// ### Params
///
/// * `t1` - `Bᵀ K² B` dense `[k, k]` row-major
/// * `ap_indptr` - CSR row pointers of `A_prevᵀ` `[n + 1]`
/// * `ap_indices` - Archetype indices of its non-zeros `[nnz]`
/// * `ap_values` - Values of its non-zeros `[nnz]`
/// * `k2b_indptr` - CSR row pointers of `K²B` `[n + 1]`
/// * `k2b_indices` - Archetype indices of its non-zeros `[nnz]`
/// * `k2b_values` - Values of its non-zeros `[nnz]`
/// * `atom_idx` - Output atom archetype indices `[n, cap]`
/// * `atom_val` - Output atom weights `[n, cap]`, zero for a pruned slot
/// * `atom_cnt` - Output atom count per cell `[n]`
/// * `threshold` - Pruning threshold as a one-element tensor. A buffer rather
///   than a scalar argument because a comptime `f32` is not `Hash` and a
///   runtime float scalar would need a `ScalarArgSettings` bound this module
///   otherwise has no use for.
/// * `n` - Number of cells
/// * `n_iters` - Frank-Wolfe iterations per column
/// * `k` - Number of archetypes (comptime)
/// * `slots` - Columns owned per thread, `ceil(k / wg_size)` (comptime)
/// * `wg_size` - Workgroup width, a power of two (comptime)
/// * `cap` - Atom capacity per cell, the output stride, at most `wg_size`.
///   Runtime rather than comptime so a changing capacity does not recompile the
///   shader; the shared arrays are sized at `wg_size` instead.
/// * `pruning` - Whether to prune and renormalise (comptime)
/// * `use_plane` - Reduce with plane primitives instead of a shared-memory
///   tree. Halving trees cost `log2(wg_size)` barriers each and this kernel
///   runs two of them per Frank-Wolfe iteration, which dominates at small
///   `slots`. Gated because a workgroup straddling two planes would reduce over
///   only part of the columns (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> block, then grid-strides over
///   cells
/// * `UNIT_POS_X` -> first owned column, and the owned atom slot
// `use_plane` is comptime so exactly one reduction arm survives expansion, but
// the macro expands both at the Rust level and the sentinel initialiser reads as
// a dead store.
#[allow(clippy::too_many_arguments, unused_assignments)]
#[cube(launch_unchecked)]
pub fn fw_columns_a_gpu<F: Float>(
    t1: &Tensor<F>,
    ap_indptr: &Tensor<u32>,
    ap_indices: &Tensor<u32>,
    ap_values: &Tensor<F>,
    k2b_indptr: &Tensor<u32>,
    k2b_indices: &Tensor<u32>,
    k2b_values: &Tensor<F>,
    atom_idx: &mut Tensor<u32>,
    atom_val: &mut Tensor<F>,
    atom_cnt: &mut Tensor<u32>,
    threshold: &Tensor<F>,
    n: u32,
    n_iters: u32,
    cap: u32,
    #[comptime] k: usize,
    #[comptime] slots: usize,
    #[comptime] wg_size: u32,
    #[comptime] pruning: bool,
    #[comptime] use_plane: bool,
) {
    let block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let tx = UNIT_POS_X;

    let mut w = Array::<F>::new(slots);
    let mut k2b = Array::<F>::new(slots);

    let mut s_atom_idx = SharedMemory::<u32>::new(wg_size as usize);
    let mut s_atom_val = SharedMemory::<F>::new(wg_size as usize);
    let mut s_dropped = SharedMemory::<F>::new(wg_size as usize);
    let mut s_val = SharedMemory::<F>::new(wg_size as usize);
    let mut s_idx = SharedMemory::<u32>::new(wg_size as usize);
    let mut s_count = SharedMemory::<u32>::new(1usize);
    let mut s_found = SharedMemory::<u32>::new(1usize);
    let mut s_any_drop = SharedMemory::<u32>::new(1usize);

    let mut cell = block;
    while cell < n {
        let cell_us = cell as usize;

        // Stage this cell's K²B row. The owning thread claims each non-zero by
        // comparison; dividing the column index would be a dynamic index into a
        // register array, which Metal backs with global memory.
        #[unroll]
        for s in 0..slots {
            k2b[s] = F::new(0.0);
            w[s] = F::new(0.0);
        }

        let mut p = k2b_indptr[cell_us];
        let p_end = k2b_indptr[cell_us + 1];
        while p < p_end {
            let col = k2b_indices[p as usize];
            let v = k2b_values[p as usize];
            #[unroll]
            for s in 0..slots {
                let owned = tx + comptime!(s as u32) * wg_size;
                if owned == col {
                    k2b[s] = v;
                }
            }
            p += 1u32;
        }

        // Seed the atoms from A_prev's column, one entry per thread.
        let seed_start = ap_indptr[cell_us];
        let seed_end = ap_indptr[cell_us + 1];
        let seed_len = seed_end - seed_start;

        s_atom_idx[tx as usize] = u32::MAX.runtime();
        s_atom_val[tx as usize] = F::new(0.0);
        if tx < seed_len {
            s_atom_idx[tx as usize] = ap_indices[(seed_start + tx) as usize];
            s_atom_val[tx as usize] = ap_values[(seed_start + tx) as usize];
        }
        if tx == 0u32 {
            s_count[0] = seed_len;
        }
        sync_cube();

        // w = t1 · A_prev[:, cell]
        let mut q = seed_start;
        while q < seed_end {
            let row = ap_indices[q as usize] as usize;
            let weight = ap_values[q as usize];
            let base = row * k;
            #[unroll]
            for s in 0..slots {
                let col = tx + comptime!(s as u32) * wg_size;
                if col < comptime!(k as u32) {
                    w[s] += weight * t1[base + col as usize];
                }
            }
            q += 1u32;
        }

        let mut t = 0u32;
        while t < n_iters {
            // Per-thread argmin over its own columns, then a workgroup tree.
            let mut best = F::max_value();
            let mut best_col = 0u32;
            #[unroll]
            for s in 0..slots {
                let col = tx + comptime!(s as u32) * wg_size;
                if col < comptime!(k as u32) {
                    let grad = w[s] - k2b[s];
                    if grad < best {
                        best = grad;
                        best_col = col;
                    }
                }
            }
            // Sentinel: both branches below assign it before it is read.
            let mut amin = u32::MAX.runtime();
            if use_plane {
                // Plane reduce, then one barrier to combine the per-plane
                // winners. The tie-break has to run on the column index, not
                // the lane: thread `tx` owns columns `tx, tx + wg, ...`, so the
                // lowest lane holding the minimum can be holding a high column.
                // Reducing the candidate columns directly gets it right and
                // needs no shuffle.
                let plane_best = plane_min(best);
                let mut candidate = u32::MAX.runtime();
                if best == plane_best {
                    candidate = best_col;
                }
                let plane_col = plane_min(candidate);

                let plane_id = tx / PLANE_DIM;
                if UNIT_POS_PLANE == 0u32 {
                    s_val[plane_id as usize] = plane_best;
                    s_idx[plane_id as usize] = plane_col;
                }
                sync_cube();

                let n_planes = CUBE_DIM_X / PLANE_DIM;
                let mut acc_val = s_val[0];
                let mut acc_idx = s_idx[0];
                let mut pl = 1u32;
                while pl < n_planes {
                    let other_val = s_val[pl as usize];
                    let other_idx = s_idx[pl as usize];
                    if other_val < acc_val || (other_val == acc_val && other_idx < acc_idx) {
                        acc_val = other_val;
                        acc_idx = other_idx;
                    }
                    pl += 1u32;
                }
                amin = acc_idx;
            } else {
                s_val[tx as usize] = best;
                s_idx[tx as usize] = best_col;
                sync_cube();

                // Forced runtime, or cubecl unrolls the loop at expansion.
                let mut stride = (wg_size / 2u32).runtime();
                while stride > 0u32 {
                    if tx < stride {
                        let other_val = s_val[(tx + stride) as usize];
                        let other_idx = s_idx[(tx + stride) as usize];
                        let mine_val = s_val[tx as usize];
                        let mine_idx = s_idx[tx as usize];
                        if other_val < mine_val || (other_val == mine_val && other_idx < mine_idx) {
                            s_val[tx as usize] = other_val;
                            s_idx[tx as usize] = other_idx;
                        }
                    }
                    sync_cube();
                    stride /= 2u32;
                }
                amin = s_idx[0];
            }

            let gamma = F::new(2.0) / (F::cast_from(t) + F::new(2.0));
            let retain = F::new(1.0) - gamma;

            // Scale every atom, then add γ at `amin`, matching FwAtoms::step.
            let count = s_count[0];
            if tx < count {
                s_atom_val[tx as usize] *= retain;
            }
            if tx == 0u32 {
                s_found[0] = u32::MAX.runtime();
            }
            sync_cube();

            if tx < count && s_atom_idx[tx as usize] == amin {
                s_found[0] = tx;
            }
            sync_cube();

            if tx == 0u32 {
                let found = s_found[0];
                if found == u32::MAX.runtime() {
                    s_atom_idx[count as usize] = amin;
                    s_atom_val[count as usize] = gamma;
                    s_count[0] = count + 1u32;
                } else {
                    s_atom_val[found as usize] += gamma;
                }
            }

            let base = amin as usize * k;
            #[unroll]
            for s in 0..slots {
                let col = tx + comptime!(s as u32) * wg_size;
                if col < comptime!(k as u32) {
                    w[s] = w[s] * retain + gamma * t1[base + col as usize];
                }
            }
            sync_cube();

            if pruning {
                let live = s_count[0];
                if tx == 0u32 {
                    s_any_drop[0] = 0u32;
                }
                s_dropped[tx as usize] = F::new(0.0);
                sync_cube();

                // Drop below threshold. `abs` is spelled out as a pair of
                // comparisons so this stays on the Float ops the backend is
                // guaranteed to have.
                let thr = threshold[0];
                if tx < live {
                    let weight = s_atom_val[tx as usize];
                    let above = weight > thr || weight < -thr;
                    // The `!= 0` guard is what keeps the fast path below alive.
                    // At `t = 0` γ is 1, so every seed atom is scaled to exactly
                    // zero, and a slot is never reclaimed: without this those
                    // slots would fail the threshold test on every subsequent
                    // iteration and pin `s_any_drop` high for the whole column.
                    // Re-dropping an already-zero slot is a no-op anyway, since
                    // its correction term is `-0 · K²[:, j]`.
                    if !above && weight != F::new(0.0) {
                        s_dropped[tx as usize] = weight;
                        s_atom_val[tx as usize] = F::new(0.0);
                        s_any_drop[0] = 1u32;
                    }
                }
                sync_cube();

                // Each dropped atom takes its rank-1 term back out of w. Guarded
                // because at the default threshold nothing drops and this loop
                // would otherwise cost `cap` t1 rows every iteration.
                if s_any_drop[0] == 1u32 {
                    let mut a = 0u32;
                    while a < live {
                        let weight = s_dropped[a as usize];
                        if weight != F::new(0.0) {
                            let drop_base = s_atom_idx[a as usize] as usize * k;
                            #[unroll]
                            for s in 0..slots {
                                let col = tx + comptime!(s as u32) * wg_size;
                                if col < comptime!(k as u32) {
                                    w[s] -= weight * t1[drop_base + col as usize];
                                }
                            }
                        }
                        a += 1u32;
                    }
                }

                // Renormalise the survivors back to a convex combination.
                let mut mass = F::new(0.0);
                if tx < live {
                    mass = s_atom_val[tx as usize];
                }
                let mut total = F::new(0.0);
                if use_plane {
                    let plane_mass = plane_sum(mass);
                    let plane_id = tx / PLANE_DIM;
                    if UNIT_POS_PLANE == 0u32 {
                        s_val[plane_id as usize] = plane_mass;
                    }
                    sync_cube();

                    let n_planes = CUBE_DIM_X / PLANE_DIM;
                    let mut pl = 0u32;
                    while pl < n_planes {
                        total += s_val[pl as usize];
                        pl += 1u32;
                    }
                } else {
                    s_val[tx as usize] = mass;
                    sync_cube();

                    let mut sum_stride = (wg_size / 2u32).runtime();
                    while sum_stride > 0u32 {
                        if tx < sum_stride {
                            let other = s_val[(tx + sum_stride) as usize];
                            s_val[tx as usize] += other;
                        }
                        sync_cube();
                        sum_stride /= 2u32;
                    }
                    total = s_val[0];
                }
                let mut renorm = F::new(1.0);
                if total > F::new(A_RENORM_FLOOR) {
                    renorm = F::new(1.0) / total;
                }
                if tx < live {
                    s_atom_val[tx as usize] *= renorm;
                }
                #[unroll]
                for s in 0..slots {
                    w[s] *= renorm;
                }
                sync_cube();
            }

            t += 1u32;
        }

        let final_count = s_count[0];
        let out_base = cell_us * cap as usize;
        if tx < final_count {
            atom_idx[out_base + tx as usize] = s_atom_idx[tx as usize];
            atom_val[out_base + tx as usize] = s_atom_val[tx as usize];
        }
        if tx == 0u32 {
            atom_cnt[cell_us] = final_count;
        }
        // Guards the shared state against the next cell overwriting it while
        // slower threads are still reading this one.
        sync_cube();

        cell += CUBE_COUNT_X * CUBE_COUNT_Y;
    }
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
    // returns zeros.
    let limit = client.properties().memory.max_page_size as usize;
    let indptr_bytes = (n + 1) * size_of::<u32>();
    let checked = [
        ("K2B values", k2b.nnz * size_of::<F>()),
        ("K2B indices", k2b.nnz * size_of::<u32>()),
        ("K2B indptr", indptr_bytes),
        ("K2At values", t2.nnz * size_of::<F>()),
        ("K2At indices", t2.nnz * size_of::<u32>()),
        ("K2At indptr", indptr_bytes),
        ("B values", b_mat.nnz * size_of::<F>()),
        ("B indices", b_mat.nnz * size_of::<u32>()),
        ("B indptr", indptr_bytes),
        ("t1", k * k * size_of::<F>()),
        ("argmin partial values", part_val.len() * size_of::<F>()),
        ("argmin partial indices", part_idx.len() * size_of::<u32>()),
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

/// Whether the plane-reduction path in [fw_columns_a_gpu] is safe on this device.
///
/// The kernel derives a plane id from `UNIT_POS_X / PLANE_DIM` and reduces
/// within each plane before combining. If the device reports a plane-size range
/// rather than an exact size, or the width is not a whole number of planes, a
/// plane could be partially populated and `plane_min` would silently reduce over
/// only part of the columns. Apple Silicon reports 32/32, so the plane path is
/// taken there.
///
/// ### Params
///
/// * `client` - CubeCL compute client, queried for hardware properties
/// * `wg_size` - Workgroup width the kernel will be launched at
///
/// ### Returns
///
/// `true` to take the plane path, `false` for the shared-memory tree.
pub fn plane_reduce_viable<R: Runtime>(client: &ComputeClient<R>, wg_size: u32) -> bool {
    let hw = &client.properties().hardware;
    let plane = hw.plane_size_min;
    plane == hw.plane_size_max && plane > 0 && wg_size.is_multiple_of(plane)
}

/// Dispatch [fw_columns_a_gpu].
///
/// The atom capacity is a hard constraint rather than a tuning knob: thread `i`
/// owns atom slot `i`, so `cap` cannot exceed the workgroup width. `cap` is
/// `max_seed + n_iters`, where `max_seed` is the widest column of `A_prev`, and
/// the caller is expected to check [a_columns_capacity] first and fall back to
/// the CPU when it does not fit.
///
/// ### Params
///
/// * `t1` - `Bᵀ K² B` dense `[k, k]` row-major
/// * `a_prev_t` - `A_prevᵀ` as CSR `n × k`
/// * `k2b` - `K²B` as CSR `n × k`
/// * `atom_idx` - Output `[n, cap]`
/// * `atom_val` - Output `[n, cap]`
/// * `atom_cnt` - Output `[n]`
/// * `threshold` - Pruning threshold as a one-element tensor
/// * `n` - Number of cells
/// * `k` - Number of archetypes
/// * `n_iters` - Frank-Wolfe iterations per column
/// * `cap` - Atom capacity per cell
/// * `pruning` - Pruning threshold, or `None` to skip pruning
/// * `use_plane` - Force the plane or shared-memory reduction, or `None` to pick
///   by [plane_reduce_viable]. A parameter only so tests can reach the arm this
///   device does not select; production callers pass `None`.
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `GpuBindingTooLarge` / `GpuCubeCountExceeded` if the dispatch
/// would bust a device limit, or `InvalidArgument` if `cap` exceeds the
/// workgroup width.
#[allow(clippy::too_many_arguments)]
pub fn launch_fw_columns_a<R, F>(
    t1: &GpuTensor<R, F>,
    a_prev_t: &GpuCompressedSparseData<R, F>,
    k2b: &GpuCompressedSparseData<R, F>,
    atom_idx: &GpuTensor<R, u32>,
    atom_val: &GpuTensor<R, F>,
    atom_cnt: &GpuTensor<R, u32>,
    threshold: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    n_iters: usize,
    cap: u32,
    pruning: Option<f32>,
    use_plane: Option<bool>,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    for mat in [a_prev_t, k2b] {
        if !mat.cs_type.is_csr() {
            return Err(BixverseErrors::SparseLayoutMismatch {
                expected: CompressedSparseFormat::Csr,
                got: mat.cs_type,
            });
        }
    }

    if cap > A_COLUMNS_WG {
        return Err(BixverseErrors::InvalidArgument(format!(
            "fw_columns_a: atom capacity {} exceeds the workgroup width {}",
            cap, A_COLUMNS_WG
        )));
    }

    // An over-sized binding is rejected silently: the kernel does no work and
    // returns zeros.
    let limit = client.properties().memory.max_page_size as usize;
    let indptr_bytes = (n + 1) * size_of::<u32>();
    let checked = [
        ("t1", k * k * size_of::<F>()),
        ("A_prev^T values", a_prev_t.nnz * size_of::<F>()),
        ("A_prev^T indices", a_prev_t.nnz * size_of::<u32>()),
        ("A_prev^T indptr", indptr_bytes),
        ("K2B values", k2b.nnz * size_of::<F>()),
        ("K2B indices", k2b.nnz * size_of::<u32>()),
        ("K2B indptr", indptr_bytes),
        ("atom indices", n * cap as usize * size_of::<u32>()),
        ("atom weights", n * cap as usize * size_of::<F>()),
        ("atom counts", n * size_of::<u32>()),
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

    let slots = k.div_ceil(A_COLUMNS_WG as usize);
    let blocks = A_COLUMNS_BLOCKS.min(n.max(1) as u32);

    let (gx, gy) = grid_2d(blocks);
    let count = checked_cube_count::<R>("fw_columns_a_gpu", gx, gy, 1)?;

    unsafe {
        fw_columns_a_gpu::launch_unchecked::<F, R>(
            client,
            count,
            CubeDim::new_1d(A_COLUMNS_WG),
            t1.clone().into_tensor_arg(),
            a_prev_t.indptr.clone().into_tensor_arg(),
            a_prev_t.indices.clone().into_tensor_arg(),
            a_prev_t.values.clone().into_tensor_arg(),
            k2b.indptr.clone().into_tensor_arg(),
            k2b.indices.clone().into_tensor_arg(),
            k2b.values.clone().into_tensor_arg(),
            atom_idx.clone().into_tensor_arg(),
            atom_val.clone().into_tensor_arg(),
            atom_cnt.clone().into_tensor_arg(),
            threshold.clone().into_tensor_arg(),
            n as u32,
            n_iters as u32,
            cap,
            k,
            slots,
            A_COLUMNS_WG,
            pruning.is_some(),
            use_plane.unwrap_or_else(|| plane_reduce_viable::<R>(client, A_COLUMNS_WG)),
        );
    }

    Ok(())
}

/// Atom capacity the A-column kernel needs for a given `A_prev`.
///
/// Thread `i` owns atom slot `i` and a slot is never reclaimed, so the capacity
/// is the widest column of `A_prev` plus one append per Frank-Wolfe iteration.
///
/// ### Params
///
/// * `a_prev_t` - `A_prevᵀ` as CSR `n × k`, rows are columns of `A_prev`
/// * `n_iters` - Frank-Wolfe iterations per column
///
/// ### Returns
///
/// The required capacity, which the caller must compare against
/// [A_COLUMNS_WG] before dispatching.
pub fn a_columns_capacity(
    a_prev_t: &crate::prelude::CompressedSparseData2<f32>,
    n_iters: usize,
) -> usize {
    let widest = (0..a_prev_t.shape.0)
        .map(|row| (a_prev_t.indptr[row + 1] - a_prev_t.indptr[row]) as usize)
        .max()
        .unwrap_or(0);
    widest + n_iters
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

    /// One entry of the gradient `G[i, c]`, for checking an argmin that came back
    /// different from the reference.
    ///
    /// ### Params
    ///
    /// * `k2b` - Dense `[n, k]` row-major
    /// * `t1` - Dense `[k, k]` row-major
    /// * `t2` - Dense `[n, k]` row-major
    /// * `k` - Columns
    /// * `i` - Row
    /// * `c` - Column
    ///
    /// ### Returns
    ///
    /// `sum_m K²B[i, m] * t1[m, c] - K²Aᵀ[i, c]`.
    fn cpu_grad_at(k2b: &[f32], t1: &[f32], t2: &[f32], k: usize, i: usize, c: usize) -> f32 {
        let mut g = 0.0f32;
        for m in 0..k {
            g += k2b[i * k + m] * t1[m * k + c];
        }
        g - t2[i * k + c]
    }

    /// Assert the kernel's argmins against the reference, tolerating near-ties.
    ///
    /// The two paths sum the same products in different orders, so columns whose
    /// two best rows sit within a few last bits of each other can legitimately
    /// resolve either way. An exact index match is demanded wherever the minimum
    /// is unambiguous; where it is not, what has to hold is that the row the
    /// kernel picked really is a minimum.
    ///
    /// ### Params
    ///
    /// * `k2b` - Dense `[n, k]` row-major
    /// * `t1` - Dense `[k, k]` row-major
    /// * `t2` - Dense `[n, k]` row-major
    /// * `got_idx` - Kernel argmins `[k]`
    /// * `want_idx` - Reference argmins `[k]`
    /// * `want_val` - Reference minima `[k]`
    /// * `n` - Rows
    /// * `k` - Columns
    #[allow(clippy::too_many_arguments)]
    fn assert_argmins_agree(
        k2b: &[f32],
        t1: &[f32],
        t2: &[f32],
        got_idx: &[u32],
        want_idx: &[u32],
        want_val: &[f32],
        n: usize,
        k: usize,
    ) {
        let mut ties = 0usize;
        for c in 0..k {
            if got_idx[c] == want_idx[c] {
                continue;
            }
            ties += 1;
            let got = cpu_grad_at(k2b, t1, t2, k, got_idx[c] as usize, c);
            assert_relative_eq!(got, want_val[c], max_relative = 1e-5, epsilon = 1e-6);
        }
        // A handful of ties is the arithmetic; a flood of them is a broken scan.
        assert!(
            ties * 50 <= k,
            "{} / {} columns disagree at ({}, {}), well past a tie-break rate",
            ties,
            k,
            n,
            k
        );
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
    ///
    /// The shapes are picked to cover the two things the kernel's structure turns
    /// on, both of which are degenerate at small sizes:
    ///
    /// - `slots = ceil(k / B_ARGMIN_WG)`, the strided column ownership held in
    ///   registers. It is 1 for every `k <= 256`, so anything smaller than that
    ///   never exercises the multi-slot unrolling or the "find the owning slot by
    ///   comparison" trick in the `K²Aᵀ` and gap loops. `k = 300` gives 2 slots,
    ///   `k = 1100` gives 5.
    /// - the grid stride. Blocks are capped at `B_ARGMIN_BLOCKS`, so below
    ///   `n = 1024` each block owns exactly one row and the per-block running
    ///   minimum never actually accumulates. `n = 3000` gives about three rows per
    ///   block.
    #[test]
    fn test_fw_argmin_b_matches_cpu() {
        let Some(device) = try_device() else { return };

        for (n, k) in [
            (40usize, 7usize),
            (257, 33),
            (1000, 130),
            (3000, 300),
            (300, 1100),
        ] {
            let k2b = fixture(n * k, 1, 3);
            let t1 = fixture(k * k, 5, 4);
            let t2 = fixture(n * k, 9, 5);
            let b_mat = fixture(n * k, 13, 11);

            let (want_val, want_idx, want_gap) = cpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k);
            let (got_val, got_idx, got_gap) =
                gpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k, &device);

            for c in 0..k {
                assert_relative_eq!(got_val[c], want_val[c], max_relative = 1e-4, epsilon = 1e-5);
            }
            assert_argmins_agree(&k2b, &t1, &t2, &got_idx, &want_idx, &want_val, n, k);
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
