//! GPU kernels for the SEACells Frank-Wolfe updates.
//!
//! [fw_argmin_b()] replaces `fw_argmins_b`, the per-archetype gradient scan in
//! the B update, and [fw_columns_a_gpu()] replaces `fw_columns_a`, the per-cell
//! column solve in the A update. Kernel construction, archetype initialisation
//! and the RSS stay on the host.
//!
//! Layout convention: every `n × k` matrix (`K²B`, `K²Aᵀ`, `B`) is passed as
//! CSR over cells, the untransposed output of `k_squared_matmul`, so the GPU
//! path drops the transposes the CPU scan needs rather than adding any. The
//! `k × k` `t1` is CSR too, with column indices ascending within a row.
//!
//! Nothing here allocates a buffer quadratic in `k`, and no `n × k` matrix is
//! ever dense. The gradient is never materialised: each thread owns a
//! contiguous block of `ceil(k / wg)` columns in registers and reduces it to a
//! running `(min, idx)` as it goes. Contiguous rather than strided ownership is
//! what lets `t1` stay sparse: a thread's columns form one run of any sorted
//! row, bracketed by a precomputed `[k, wg + 1]` segment table instead of
//! searched for. See [a_columns_segments()] and [accumulate_segmented_row()].
//!
//! The per-thread register arrays are what bound `k`. Exceeding the bound is a
//! rejected dispatch that reports success and does no work, so the widths live
//! in [A_COLUMNS_WG_TIERS] and [B_ARGMIN_WG_TIERS] with per-device caps, and
//! the callers verify the output rather than trusting them.

#![allow(missing_docs)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use rayon::prelude::*;

use crate::errors::BixverseErrors;
use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::prelude::CompressedSparseFormat;

////////////
// Consts //
////////////

/// Workgroups for the B-gradient argmin, grid-striding over cells.
///
/// Fixed rather than derived from `n`, so the partial buffer stays
/// `B_ARGMIN_BLOCKS * k` at every dataset size instead of growing with it.
pub const B_ARGMIN_BLOCKS: u32 = 1024;

/// Workgroup width for [reduce_argmin_blocks()]. One thread per output column,
/// grid-striding.
pub const B_REDUCE_WG: u32 = 256;

/// Workgroup widths [fw_argmin_b()] is instantiated at, narrowest first, each
/// with the most register slots per thread it can carry.
///
/// Same structure and the same hazard as [A_COLUMNS_WG_TIERS], but the widest
/// tier's cap is lower: this kernel holds `acc`, `run_val` and `run_idx` per
/// thread, three register arrays of `slots` against the A solve's two. Past the
/// cap the dispatch is rejected and the kernel silently does no work.
///
/// The caps are device-specific and each sits at the last width observed to
/// work rather than at a bisected boundary, so they are conservative. The
/// widest tier fixes the ceiling on `k`; above it [GpuFwArgminB] runs this
/// solve on the CPU.
///
/// The silent failure is easy to miss here because `fw_argmin_b` is followed by
/// `reduce_argmin_blocks`, which runs regardless and fills the output with
/// whatever the partial buffers held. Check the partials, not the output.
///
/// [GpuFwArgminB]: crate::gpu::sc_gpu::seacells_gpu::GpuFwArgminB
pub const B_ARGMIN_WG_TIERS: [(u32, usize); 3] = [(128, 32), (256, 32), (512, 20)];

/// Workgroup tier for a B-gradient argmin, or `None` if `k` is too large.
///
/// ### Params
///
/// * `k` - Number of archetypes
///
/// ### Returns
///
/// The workgroup width, or `None` when no tier covers the shape.
pub fn b_argmin_workgroup(k: usize) -> Option<u32> {
    B_ARGMIN_WG_TIERS
        .iter()
        .find(|&&(wg, max_slots)| k.div_ceil(wg as usize) <= max_slots)
        .map(|&(wg, _)| wg)
}

/// Workgroups for the A-column solve, grid-striding over cells.
///
/// One workgroup owns one cell at a time and runs that cell's whole Frank-Wolfe
/// loop, so this is a residency knob rather than a partition: the shared state
/// is reset per cell and nothing scales with it.
pub const A_COLUMNS_BLOCKS: u32 = 1024;

/// Narrowest workgroup width [fw_columns_a_gpu()] is instantiated at, and the
/// tier the selector prefers whenever the shape fits it.
///
/// The width doubles as the atom capacity ceiling, since thread `i` owns atom
/// slot `i`. Must be a power of two and a whole number of planes.
///
/// Narrower is better wherever both fit, because the kernel is bound by the
/// reductions rather than the arithmetic, and halving the width doubles the
/// columns each thread owns while halving the planes to combine. 64 would
/// continue the trend but cannot hold the atom counts these shapes need without
/// giving each thread several atom slots. See [a_columns_workgroup()] for the
/// tier the wider shapes take.
pub const A_COLUMNS_WG: u32 = 128;

/// Register slots per thread beyond which [fw_columns_a_gpu()] declines the
/// work.
///
/// `w` and `k2b_row` are each `Array::<F>::new(slots)`, so the pair costs
/// `2 * slots` floats per thread. On Metal a spilled register array is backed by
/// global memory, which turns the kernel's whole reason for holding the gradient
/// in registers into a slow scatter.
///
/// The value sits at the knee of the spill: throughput is flat up to it and
/// falls several-fold immediately past it, so this is a cliff rather than a
/// slope and there is nothing to be gained by creeping over it. Retune by
/// sweeping `k` at a fixed workgroup width and watching for the drop.
///
/// This is the performance ceiling. [A_COLUMNS_WG_TIERS] carries a second,
/// per-width ceiling which is a correctness bound; the effective limit is the
/// lower of the two.
pub const A_COLUMNS_MAX_SLOTS: usize = 32;

/// Workgroup widths [fw_columns_a_gpu()] is instantiated at, narrowest first,
/// each paired with the most register slots per thread it can carry.
///
/// The width trades register slots against reduction width: `slots` falls as
/// `k / wg`, while the plane count the reduction combines rises as `wg / 32`.
/// Narrower measured better wherever both fit, so [a_columns_workgroup()] takes
/// the first tier that fits.
///
/// **The per-tier slot cap is a correctness bound, not a tuning knob.** Past it
/// the launch is rejected and the kernel does no work while reporting success,
/// which is the `launch_unchecked` failure mode. Each cap sits at the last
/// width observed to work at that tier. 1024 is not listed at all: its ceiling
/// covers less `k` than 512's, so the narrower tier reaches every shape it
/// could. The caps are device-specific, so [FwArgminB::columns_a] verifies the
/// kernel actually wrote output rather than trusting them.
///
/// [FwArgminB::columns_a]: crate::single_cell::mc_generation::seacells::FwArgminB::columns_a
pub const A_COLUMNS_WG_TIERS: [(u32, usize); 3] = [(128, 32), (256, 32), (512, 24)];

/// Narrowest plane the plane-reduction arm supports, used to size the per-plane
/// reduction scratch.
///
/// This is a floor the arm *enforces*, not an observation about hardware. Metal
/// and hardware Vulkan report 32 or 64, but lavapipe reports 4 or 8 and Intel
/// integrated graphics can report 8 or 16. Since the scratch is sized by this
/// constant and indexed by the runtime `PLANE_DIM`, anything narrower overruns
/// it, so [plane_reduce_viable()] rejects those devices and they take the
/// shared-memory tree arm instead.
pub const MIN_PLANE_WIDTH: u32 = 32;

/// Shared-memory entries the argmin and renormalisation reductions need.
///
/// The plane arm only ever stores one entry per plane, so it needs
/// `wg_size / plane`; the shared-memory tree arm indexes by thread and needs the
/// full width. Sizing this by arm rather than by workgroup width is what makes
/// the wide tiers affordable.
///
/// The plane arm relies on two invariants, both of which
/// [plane_reduce_viable()] and [A_COLUMNS_WG_TIERS] hold up between them:
///
/// - `plane >= MIN_PLANE_WIDTH`, so `wg_size / plane` never exceeds the
///   `wg_size / MIN_PLANE_WIDTH` slots allocated here;
/// - `wg_size <= plane * plane`, so the level-2 combine, which reduces the
///   per-plane winners with a single `plane_min` over plane 0, covers every
///   plane. The widest tier is 512 and the narrowest accepted plane is 32, so
///   the worst case is 16 planes reduced by 32 lanes.
///
/// ### Params
///
/// * `wg_size` - Workgroup width
/// * `use_plane` - Whether the plane reduction arm is taken
///
/// ### Returns
///
/// The number of entries to allocate.
pub const fn reduce_scratch_len(wg_size: u32, use_plane: bool) -> usize {
    if use_plane {
        let planes = wg_size / MIN_PLANE_WIDTH;
        if planes == 0 { 1 } else { planes as usize }
    } else {
        wg_size as usize
    }
}

/// Below this the L1 renormalisation is skipped, mirroring `FW_RENORM_FLOOR` on
/// the CPU side so a fully-pruned column behaves the same on both paths.
///
/// The CPU compares in `f64` and this compares in `f32`; at 1e-15 against
/// weights that sum to 1 the two only disagree for columns that have already
/// collapsed to nothing.
pub const A_RENORM_FLOOR: f32 = 1e-15;

/////////////
// Helpers //
/////////////

/// First position in `indices[start..end]` whose value is at least `target`.
///
/// The range must be ascending, which every CSR this kernel is fed satisfies:
/// `csr_matmul_csr` sorts each row in its accumulator, `transpose_sparse`
/// scatters in ascending major order, and `sparse_add_csr` merges two sorted
/// rows.
///
/// ### Params
///
/// * `indices` - Ascending column indices
/// * `start` - Inclusive range start
/// * `end` - Exclusive range end
/// * `target` - Value to bound
///
/// ### Returns
///
/// The lower-bound position, `end` if every entry is below `target`.
#[cube]
pub fn lower_bound(indices: &Tensor<u32>, start: u32, end: u32, target: u32) -> u32 {
    let mut lo = start;
    let mut hi = end;
    while lo < hi {
        let mid = lo + (hi - lo) / 2u32;
        if indices[mid as usize] < target {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Accumulate `scale * row` into the caller's register slots, using a
/// precomputed per-thread segment table instead of searching.
///
/// `seg[row * (wg + 1) + tx]` is where thread `tx`'s contiguous column block
/// starts in that row, so the bounds are two loads at adjacent addresses, which
/// coalesce across the workgroup. Searching instead costs `log2(nnz(row))`
/// divergent probes in *every* thread, which measured slower than reading a
/// dense row outright on the shapes this kernel sees.
///
/// ### Params
///
/// * `acc` - Register slots to accumulate into
/// * `indices` - Row column indices, ascending
/// * `values` - Row values, parallel to `indices`
/// * `seg` - Segment table, `[k, wg_size + 1]` row-major
/// * `seg_base` - `row * (wg_size + 1)`
/// * `tx` - Thread index within the workgroup
/// * `base` - First column this thread owns
/// * `scale` - Factor applied to every value
/// * `slots` - Columns owned per thread (comptime)
#[cube]
pub fn accumulate_segmented_row<F: Float>(
    acc: &mut Array<F>,
    indices: &Tensor<u32>,
    values: &Tensor<F>,
    seg: &Tensor<u32>,
    seg_base: u32,
    tx: u32,
    base: u32,
    scale: F,
    #[comptime] slots: usize,
) {
    let mut p = seg[(seg_base + tx) as usize];
    let hi = seg[(seg_base + tx + 1u32) as usize];
    while p < hi {
        let col = indices[p as usize];
        let v = values[p as usize] * scale;
        #[unroll]
        for s in 0..slots {
            if col == base + comptime!(s as u32) {
                acc[s] += v;
            }
        }
        p += 1u32;
    }
}

/// Accumulate `scale * row` into the caller's register slots, sparsely.
///
/// Assumes **blocked** column ownership: thread `tx` owns the contiguous range
/// `[base, base + slots)`, so the entries of a sorted row it needs form one
/// contiguous run. One binary search finds where that run starts and the walk
/// stops at the first column past the block, which costs about one load more
/// than the number of entries actually owned.
///
/// ### Params
///
/// * `acc` - Register slots to accumulate into
/// * `indices` - Row column indices, ascending
/// * `values` - Row values, parallel to `indices`
/// * `row_start` - Inclusive start of the row in `indices`
/// * `row_end` - Exclusive end of the row in `indices`
/// * `base` - First column this thread owns
/// * `scale` - Factor applied to every value
/// * `slots` - Columns owned per thread (comptime)
#[cube]
pub fn accumulate_sparse_row<F: Float>(
    acc: &mut Array<F>,
    indices: &Tensor<u32>,
    values: &Tensor<F>,
    row_start: u32,
    row_end: u32,
    base: u32,
    scale: F,
    #[comptime] slots: usize,
) {
    let stop = base + comptime!(slots as u32);
    let mut p = lower_bound(indices, row_start, row_end, base);
    // Walked rather than bounded by a second binary search: a thread owns
    // `nnz(row) * slots / k` entries, which is well under one on these shapes,
    // so the walk is cheaper than another `log2(nnz)` probes.
    let mut walking = p < row_end;
    while walking {
        let col = indices[p as usize];
        if col < stop {
            let v = values[p as usize] * scale;
            #[unroll]
            for s in 0..slots {
                if col == base + comptime!(s as u32) {
                    acc[s] += v;
                }
            }
            p += 1u32;
            walking = p < row_end;
        } else {
            walking = false;
        }
    }
}

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
/// Thread `tx` owns the contiguous column block
/// `[tx * slots, (tx + 1) * slots)`, held in registers rather than shared
/// memory, so there is no `k`-dependent shared memory budget to gate on. Every
/// register-array index is comptime, since a dynamically indexed local array is
/// backed by global memory on Metal.
///
/// Contiguous ownership is what lets `t1` stay sparse: the entries of a sorted
/// row that a thread needs form one run, bracketed by the precomputed segment
/// table rather than searched for. See [accumulate_segmented_row()].
///
/// The per-thread register arrays are what bound `k` here; see
/// [B_ARGMIN_WG_TIERS].
///
/// Rows are visited in increasing order with a strict `<`, so the lowest row
/// index wins a tie within a block. Blocks grid-stride, so block order does not
/// follow row order and [reduce_argmin_blocks()] has to break ties by index
/// explicitly.
///
/// ### Params
///
/// * `k2b_indptr` - CSR row pointers of `K²B` `[n + 1]`
/// * `k2b_indices` - Archetype indices of its non-zeros `[nnz]`
/// * `k2b_values` - Values of its non-zeros `[nnz]`
/// * `t1_indices` - Archetype indices of `A Aᵀ`'s non-zeros, ascending per row
/// * `t1_values` - Values of its non-zeros `[nnz]`
/// * `t1_seg` - Per-thread column segments of `t1`, `[k, wg_size + 1]` row-major
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
/// * `UNIT_POS_X * slots` -> first owned column, then `slots` contiguous
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
pub fn fw_argmin_b<F: Float>(
    k2b_indptr: &Tensor<u32>,
    k2b_indices: &Tensor<u32>,
    k2b_values: &Tensor<F>,
    t1_indices: &Tensor<u32>,
    t1_values: &Tensor<F>,
    t1_seg: &Tensor<u32>,
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

    // Blocked ownership: thread `tx` owns the contiguous column range
    // `[base, base + slots)`, which is what makes a sorted `t1` row splittable
    // per thread without a search. See [accumulate_segmented_row()].
    let base = tx * comptime!(slots as u32);

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
            let m = k2b_indices[p as usize];
            accumulate_segmented_row::<F>(
                &mut acc,
                t1_indices,
                t1_values,
                t1_seg,
                m * comptime!(wg_size + 1),
                tx,
                base,
                k2b_values[p as usize],
                slots,
            );
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
                if base + comptime!(s as u32) == col {
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
                if base + comptime!(s as u32) == col {
                    gap += v * acc[s];
                }
            }
            r += 1u32;
        }

        // Running per-column minimum. Strict `<` keeps the lowest row index.
        #[unroll]
        for s in 0..slots {
            let col = base + comptime!(s as u32);
            if col < comptime!(k as u32) && acc[s] < run_val[s] {
                run_val[s] = acc[s];
                run_idx[s] = row;
            }
        }

        row += CUBE_COUNT_X * CUBE_COUNT_Y;
    }

    // Write this block's partial minima. Blocked ownership makes adjacent
    // threads write `slots` apart rather than contiguously, which costs
    // coalescing here, but this runs once per block against the `n / blocks`
    // cells the loop above just walked.
    let out_base = block as usize * k;
    #[unroll]
    for s in 0..slots {
        let col = base + comptime!(s as u32);
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
/// Blocks grid-stride over cells in [fw_argmin_b()], so block order does not
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
/// materialised across cells: thread `tx` owns the contiguous column block
/// `[tx * slots, (tx + 1) * slots)` in `slots` registers, so nothing here
/// scales shared memory with `k`.
///
/// Contiguous ownership is what lets `t1` stay sparse. The entries of a sorted
/// row that a thread needs form one run, bracketed by the precomputed segment
/// table rather than searched for; see [accumulate_segmented_row()]. `K²B` is
/// staged once per cell by search instead, via [accumulate_sparse_row()],
/// because a segment table for it would be `n × (wg + 1)` rather than
/// `k × (wg + 1)`, and it is amortised over all `n_iters` iterations anyway.
///
/// The per-thread register arrays are what bound `k` here; see
/// [A_COLUMNS_WG_TIERS] and [A_COLUMNS_MAX_SLOTS].
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
/// * `t1_indices` - Archetype indices of `Bᵀ K² B`'s non-zeros, ascending per
///   row
/// * `t1_values` - Values of its non-zeros `[nnz]`
/// * `t1_seg` - Per-thread column segments of `t1`, `[k, wg_size + 1]`
///   row-major
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
///   shader; the shared arrays are sized at `cap_pad` instead.
/// * `cap_pad` - `cap` rounded up to a power of two, sizing the atom arrays in
///   shared memory. Bucketed rather than exact so a capacity that drifts
///   between outer iterations does not recompile the shader (comptime)
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
/// * `UNIT_POS_X * slots` -> first owned column, then `slots` contiguous
/// * `UNIT_POS_X` -> the owned atom slot
// `use_plane` is comptime so exactly one reduction arm survives expansion, but
// the macro expands both at the Rust level and the sentinel initialiser reads as
// a dead store.
#[allow(clippy::too_many_arguments, unused_assignments)]
#[cube(launch_unchecked)]
pub fn fw_columns_a_gpu<F: Float>(
    t1_indices: &Tensor<u32>,
    t1_values: &Tensor<F>,
    t1_seg: &Tensor<u32>,
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
    #[comptime] cap_pad: usize,
    #[comptime] pruning: bool,
    #[comptime] use_plane: bool,
) {
    let block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let tx = UNIT_POS_X;

    let mut w = Array::<F>::new(slots);
    let mut k2b = Array::<F>::new(slots);

    // Sized by what is actually indexed, not by the workgroup width: the atom
    // arrays never go past `cap`, and the reduction scratch holds one entry per
    // plane on the plane arm. Shared memory is a residency knob, not just a
    // ceiling: sizing every array at `wg_size` starves the wide tiers of
    // resident workgroups.
    let mut s_atom_idx = SharedMemory::<u32>::new(cap_pad);
    let mut s_atom_val = SharedMemory::<F>::new(cap_pad);
    let mut s_dropped = SharedMemory::<F>::new(cap_pad);
    let mut s_val = SharedMemory::<F>::new(comptime!(reduce_scratch_len(wg_size, use_plane)));
    let mut s_idx = SharedMemory::<u32>::new(comptime!(reduce_scratch_len(wg_size, use_plane)));
    let mut s_count = SharedMemory::<u32>::new(1usize);
    let mut s_found = SharedMemory::<u32>::new(1usize);
    let mut s_any_drop = SharedMemory::<u32>::new(1usize);
    // Broadcast slots for the second reduction level. Kept separate from
    // `s_val`/`s_idx` so the level-2 write cannot race the level-2 reads.
    let mut s_amin = SharedMemory::<u32>::new(1usize);
    let mut s_total = SharedMemory::<F>::new(1usize);

    // Blocked ownership: thread `tx` owns the contiguous column range
    // `[base, base + slots)`. Contiguity is what makes a sorted sparse row
    // searchable, see [accumulate_sparse_row()].
    let base = tx * comptime!(slots as u32);

    let mut cell = block;
    while cell < n {
        let cell_us = cell as usize;

        #[unroll]
        for s in 0..slots {
            k2b[s] = F::new(0.0);
            w[s] = F::new(0.0);
        }

        // Stage this cell's K²B row into the owned block.
        accumulate_sparse_row::<F>(
            &mut k2b,
            k2b_indices,
            k2b_values,
            k2b_indptr[cell_us],
            k2b_indptr[cell_us + 1],
            base,
            F::new(1.0),
            slots,
        );

        // Seed the atoms from A_prev's column, one entry per thread.
        let seed_start = ap_indptr[cell_us];
        let seed_end = ap_indptr[cell_us + 1];
        let seed_len = seed_end - seed_start;

        // Guarded because the atom arrays are `cap_pad` wide, not `wg_size`.
        if tx < cap {
            s_atom_idx[tx as usize] = u32::MAX.runtime();
            s_atom_val[tx as usize] = F::new(0.0);
        }
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
            let row = ap_indices[q as usize];
            accumulate_segmented_row::<F>(
                &mut w,
                t1_indices,
                t1_values,
                t1_seg,
                row * comptime!(wg_size + 1),
                tx,
                base,
                ap_values[q as usize],
                slots,
            );
            q += 1u32;
        }

        let mut t = 0u32;
        while t < n_iters {
            // Per-thread argmin over its own columns, then a workgroup tree.
            let mut best = F::max_value();
            let mut best_col = 0u32;
            #[unroll]
            for s in 0..slots {
                let col = base + comptime!(s as u32);
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
                // winners. The tie-break runs on the column index rather than
                // the lane: a second `plane_min` over the candidate columns
                // picks the lowest-index winner directly, where selecting by
                // lane would need a ballot or shuffle primitive.
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

                // Second level, on plane 0. Combining the per-plane winners with
                // a serial loop instead costs `n_planes` shared reads in *every*
                // thread, which is what made the wide tiers unusable.
                let n_planes = CUBE_DIM_X / PLANE_DIM;
                let mut lvl2_val = F::max_value();
                let mut lvl2_idx = u32::MAX.runtime();
                if tx < n_planes {
                    lvl2_val = s_val[tx as usize];
                    lvl2_idx = s_idx[tx as usize];
                }
                let lvl2_best = plane_min(lvl2_val);
                let mut lvl2_cand = u32::MAX.runtime();
                if lvl2_val == lvl2_best {
                    lvl2_cand = lvl2_idx;
                }
                let lvl2_col = plane_min(lvl2_cand);
                if tx == 0u32 {
                    s_amin[0] = lvl2_col;
                }
                sync_cube();
                amin = s_amin[0];
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

            // Scale, then add the γ-weighted `t1` row sparsely. Splitting the
            // fused multiply-add costs one extra pass over `slots` registers and
            // saves reading `k` floats of a dense row.
            #[unroll]
            for s in 0..slots {
                w[s] *= retain;
            }
            accumulate_segmented_row::<F>(
                &mut w,
                t1_indices,
                t1_values,
                t1_seg,
                amin * comptime!(wg_size + 1),
                tx,
                base,
                gamma,
                slots,
            );
            sync_cube();

            if pruning {
                let live = s_count[0];
                if tx == 0u32 {
                    s_any_drop[0] = 0u32;
                }
                // Only `[0, cap)` is cleared, which covers every slot the scan
                // below reads since `live <= cap`.
                if tx < cap {
                    s_dropped[tx as usize] = F::new(0.0);
                }
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
                            let row = s_atom_idx[a as usize];
                            accumulate_segmented_row::<F>(
                                &mut w,
                                t1_indices,
                                t1_values,
                                t1_seg,
                                row * comptime!(wg_size + 1),
                                tx,
                                base,
                                F::new(0.0) - weight,
                                slots,
                            );
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

                    // Second level on plane 0, for the same reason as the argmin
                    // above. Summing in a different order than the CPU is fine:
                    // this feeds a renormalisation factor, not a comparison.
                    let n_planes = CUBE_DIM_X / PLANE_DIM;
                    let mut lvl2 = F::new(0.0);
                    if tx < n_planes {
                        lvl2 = s_val[tx as usize];
                    }
                    let lvl2_total = plane_sum(lvl2);
                    if tx == 0u32 {
                        s_total[0] = lvl2_total;
                    }
                    sync_cube();
                    total = s_total[0];
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

/// Dispatch [fw_argmin_b()] followed by [reduce_argmin_blocks()].
///
/// The workgroup width is chosen by [b_argmin_workgroup()] and each width is a
/// separately compiled shader, so a shape no tier covers is an error here rather
/// than a silently rejected dispatch. Callers that can fall back should check
/// the tier themselves first; [GpuFwArgminB] does.
///
/// ### Params
///
/// * `k2b` - `K²B` as CSR `n × k`
/// * `t1` - `A Aᵀ` as CSR `k × k`, column indices ascending per row
/// * `t1_seg` - Its per-thread column segments, from [a_columns_segments()]
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
/// `Ok(())`, or `GpuBindingTooLarge` / `CubeclUtils` if a dispatch
/// busts a device limit, or `InvalidArgument` if no workgroup tier covers `k`.
///
/// [GpuFwArgminB]: crate::gpu::sc_gpu::seacells_gpu::GpuFwArgminB
#[allow(clippy::too_many_arguments)]
pub fn launch_fw_argmin_b<R, F>(
    k2b: &GpuCompressedSparseData<R, F>,
    t1: &GpuCompressedSparseData<R, F>,
    t1_seg: &GpuTensor<R, u32>,
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
    for mat in [k2b, t1, t2, b_mat] {
        if !mat.cs_type.is_csr() {
            return Err(BixverseErrors::SparseLayoutMismatch {
                expected: CompressedSparseFormat::Csr,
                got: mat.cs_type,
            });
        }
    }

    // An over-sized binding is rejected silently: the kernel does no work and
    // returns zeros.
    let limits = GpuLimits::from_client(client);
    let limit = limits.max_binding_bytes as usize;
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
        ("t1 values", t1.nnz * size_of::<F>()),
        ("t1 indices", t1.nnz * size_of::<u32>()),
        ("t1 segments", t1_seg.len() * size_of::<u32>()),
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

    let wg = b_argmin_workgroup(k).ok_or_else(|| {
        BixverseErrors::InvalidArgument(format!(
            "fw_argmin_b: no workgroup tier covers k = {}, tiers {:?}",
            k, B_ARGMIN_WG_TIERS
        ))
    })?;
    let blocks = B_ARGMIN_BLOCKS.min(n.max(1) as u32);

    let (gx, gy) = grid_2d(blocks, &limits)?;
    let count = checked_cube_count("fw_argmin_b", gx, gy, 1, &limits)?;

    // One arm per tier: `wg_size` is comptime, so each width is its own shader.
    macro_rules! dispatch {
        ($wg:expr) => {{
            unsafe {
                fw_argmin_b::launch_unchecked::<F, R>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
                    k2b.indptr.clone().into_tensor_arg(),
                    k2b.indices.clone().into_tensor_arg(),
                    k2b.values.clone().into_tensor_arg(),
                    t1.indices.clone().into_tensor_arg(),
                    t1.values.clone().into_tensor_arg(),
                    t1_seg.clone().into_tensor_arg(),
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
                    k.div_ceil($wg as usize),
                    $wg,
                );
            }
        }};
    }

    match wg {
        128 => dispatch!(128u32),
        256 => dispatch!(256u32),
        512 => dispatch!(512u32),
        other => {
            return Err(BixverseErrors::InvalidArgument(format!(
                "fw_argmin_b: workgroup width {} is not one of the compiled tiers {:?}",
                other, B_ARGMIN_WG_TIERS
            )));
        }
    }

    let reduce_cubes = (k as u32).div_ceil(B_REDUCE_WG);
    let (rx, ry) = grid_2d(reduce_cubes, &limits)?;
    let reduce_count = checked_cube_count("reduce_argmin_blocks", rx, ry, 1, &limits)?;

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

/// Whether the plane-reduction path in [fw_columns_a_gpu()] is safe on this
/// device.
///
/// The kernel derives a plane id from `UNIT_POS_X / PLANE_DIM` and reduces
/// within each plane before combining. If the device reports a plane-size range
/// rather than an exact size, or the width is not a whole number of planes, a
/// plane could be partially populated and `plane_min` would silently reduce
/// over only part of the columns. Apple Silicon reports 32/32, so the plane
/// path is taken there.
///
/// The width floor is load-bearing, not defensive. [reduce_scratch_len()] sizes
/// the per-plane scratch by [MIN_PLANE_WIDTH], while the kernel indexes it by
/// the runtime `PLANE_DIM`, so a narrower plane means more planes than slots and
/// the reduction reads past the end of the scratch into the neighbouring array.
/// Software Vulkan (lavapipe reports 8/8) and Intel integrated graphics both
/// land there, which is why an exact-size check alone is not enough.
///
/// ### Params
///
/// * `wg_size` - Workgroup width the kernel will be launched at
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `true` to take the plane path, `false` for the shared-memory tree.
pub fn plane_reduce_viable(wg_size: u32, limits: &GpuLimits) -> bool {
    // `plane_partitions` guarantees `n_planes * plane == wg_size`, so bounding
    // the plane count by `wg_size / MIN_PLANE_WIDTH` is exactly the width floor.
    plane_partitions(wg_size, limits).is_some_and(|n_planes| n_planes <= wg_size / MIN_PLANE_WIDTH)
}

/// Dispatch [fw_columns_a_gpu()].
///
/// The atom capacity is a hard constraint rather than a tuning knob: thread `i`
/// owns atom slot `i`, so `cap` cannot exceed the workgroup width. `cap` is
/// `max_seed + n_iters`, where `max_seed` is the widest column of `A_prev`, and
/// the caller is expected to check [a_columns_capacity()] first and fall back
/// to the CPU when it does not fit.
///
/// The width defaults to [a_columns_workgroup()] and each width is a separately
/// compiled shader, so a shape no tier covers is an error here rather than a
/// silently rejected dispatch.
///
/// ### Params
///
/// * `t1` - `Bᵀ K² B` as CSR `k × k`, column indices ascending per row
/// * `t1_seg` - Its per-thread column segments, from [a_columns_segments()]
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
///   by [plane_reduce_viable()]. A parameter only so tests can reach the arm
///   this device does not select; production callers pass `None`.
/// * `wg_override` - Force a workgroup width, or `None` to take the tier
///   [a_columns_workgroup()] selects. A parameter only so benchmarks can sweep
///   the width; production callers pass `None`.
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `GpuBindingTooLarge` / `CubeclUtils` if the dispatch
/// would bust a device limit, or `InvalidArgument` if no tier covers `k` or
/// `cap` exceeds the chosen workgroup width.
#[allow(clippy::too_many_arguments)]
pub fn launch_fw_columns_a<R, F>(
    t1: &GpuCompressedSparseData<R, F>,
    t1_seg: &GpuTensor<R, u32>,
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
    wg_override: Option<u32>,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    for mat in [t1, a_prev_t, k2b] {
        if !mat.cs_type.is_csr() {
            return Err(BixverseErrors::SparseLayoutMismatch {
                expected: CompressedSparseFormat::Csr,
                got: mat.cs_type,
            });
        }
    }

    let wg = match wg_override.or_else(|| a_columns_workgroup(k, cap as usize)) {
        Some(wg) => wg,
        None => {
            return Err(BixverseErrors::InvalidArgument(format!(
                "fw_columns_a: no workgroup tier covers k = {} at atom capacity {}, tiers {:?} \
                 within {} slots",
                k, cap, A_COLUMNS_WG_TIERS, A_COLUMNS_MAX_SLOTS
            )));
        }
    };
    if cap > wg {
        return Err(BixverseErrors::InvalidArgument(format!(
            "fw_columns_a: atom capacity {} exceeds the workgroup width {}",
            cap, wg
        )));
    }

    // An over-sized binding is rejected silently: the kernel does no work and
    // returns zeros.
    let limits = GpuLimits::from_client(client);
    let limit = limits.max_binding_bytes as usize;
    let indptr_bytes = (n + 1) * size_of::<u32>();
    let checked = [
        ("t1 values", t1.nnz * size_of::<F>()),
        ("t1 indices", t1.nnz * size_of::<u32>()),
        ("t1 segments", t1_seg.len() * size_of::<u32>()),
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

    let blocks = A_COLUMNS_BLOCKS.min(n.max(1) as u32);
    let (gx, gy) = grid_2d(blocks, &limits)?;
    let count = checked_cube_count("fw_columns_a_gpu", gx, gy, 1, &limits)?;
    let cap_pad = (cap as usize).next_power_of_two().max(1);
    let planes = use_plane.unwrap_or_else(|| plane_reduce_viable(wg, &limits));

    // One arm per tier: `wg_size` is comptime, so each width is its own shader.
    macro_rules! dispatch {
        ($wg:expr) => {{
            unsafe {
                fw_columns_a_gpu::launch_unchecked::<F, R>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
                    t1.indices.clone().into_tensor_arg(),
                    t1.values.clone().into_tensor_arg(),
                    t1_seg.clone().into_tensor_arg(),
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
                    k.div_ceil($wg as usize),
                    $wg,
                    cap_pad,
                    pruning.is_some(),
                    planes,
                );
            }
        }};
    }

    match wg {
        128 => dispatch!(128u32),
        256 => dispatch!(256u32),
        512 => dispatch!(512u32),
        1024 => dispatch!(1024u32),
        other => {
            return Err(BixverseErrors::InvalidArgument(format!(
                "fw_columns_a: workgroup width {} is not one of the compiled tiers {:?}",
                other, A_COLUMNS_WG_TIERS
            )));
        }
    }

    Ok(())
}

/// Workgroup tier for an A-column solve, or `None` if the shape does not fit.
///
/// Picks the narrowest tier in [A_COLUMNS_WG_TIERS] whose register slot count
/// `ceil(k / wg)` is within both that tier's cap and [A_COLUMNS_MAX_SLOTS], and
/// which is wide enough to hold `cap` atom slots, since thread `i` owns atom
/// slot `i`. Narrower wins because the reduction still combines `wg / plane`
/// partials, just no longer serially.
///
/// ### Params
///
/// * `k` - Number of archetypes
/// * `cap` - Atom capacity, from [a_columns_capacity()]
///
/// ### Returns
///
/// The workgroup width, or `None` when the caller should fall back to the CPU.
pub fn a_columns_workgroup(k: usize, cap: usize) -> Option<u32> {
    A_COLUMNS_WG_TIERS
        .iter()
        .find(|&&(wg, max_slots)| {
            k.div_ceil(wg as usize) <= max_slots.min(A_COLUMNS_MAX_SLOTS) && cap <= wg as usize
        })
        .map(|&(wg, _)| wg)
}

/// Per-thread segment bounds for every row of `t1`.
///
/// Entry `row * (wg + 1) + t` is the position in `t1.indices` of the first
/// entry of `row` whose column is at least `t * slots`, so thread `t` reads its
/// run as `[seg[.. + t], seg[.. + t + 1])` with no search. Built once per A
/// update in `O(nnz + k * wg)`, against a `k * wg` search cost paid every
/// Frank-Wolfe iteration otherwise. The table is `k * (wg + 1)` `u32`, `slots`
/// times smaller than a dense `k * k` row store.
///
/// ### Params
///
/// * `t1` - `Bᵀ K² B` as CSR `k × k`, column indices ascending per row
/// * `wg` - Workgroup width the kernel will launch at
/// * `slots` - Columns owned per thread, `ceil(k / wg)`
///
/// ### Returns
///
/// The segment table, row-major `[k, wg + 1]`.
pub fn a_columns_segments(
    t1: &crate::prelude::CompressedSparseData2<f32>,
    wg: u32,
    slots: usize,
) -> Vec<u32> {
    let k = t1.shape.0;
    let stride = wg as usize + 1;
    let mut seg = vec![0u32; k * stride];

    seg.par_chunks_mut(stride)
        .enumerate()
        .for_each(|(row, out)| {
            let start = t1.indptr[row] as usize;
            let end = t1.indptr[row + 1] as usize;
            // One pass: advance through the row, closing off each thread's
            // block as its upper column bound is passed.
            let mut p = start;
            for (t, slot) in out.iter_mut().enumerate() {
                let bound = (t * slots) as u32;
                while p < end && t1.indices[p] < bound {
                    p += 1;
                }
                *slot = p as u32;
            }
            // The last boundary is `wg * slots >= k`, so it closes the row.
            out[wg as usize] = end as u32;
        });

    seg
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
/// The required capacity. Feed it to [a_columns_workgroup()], which returns
/// `None` when no tier is wide enough to give every atom its own thread.
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

    /// CPU reference for [fw_argmin_b()], mirroring `fw_argmins_b` in
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
            .unwrap()
        };

        let k2b_gpu = upload(k2b);
        let t2_gpu = upload(t2);
        let b_gpu = upload(b_mat);
        let (t1_vals, t1_idx, t1_ptr) = dense_to_csr(t1, k, k);
        let t1_csr = crate::prelude::CompressedSparseData2::<f32>::new_csr(
            &t1_vals,
            &t1_idx,
            &t1_ptr,
            None,
            (k, k),
        );
        let t1_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
            &t1_vals,
            &t1_idx,
            &t1_ptr,
            CompressedSparseFormat::Csr,
            (k, k),
            &client,
        )
        .unwrap();
        let b_wg = b_argmin_workgroup(k).expect("no tier for k");
        let seg_host = a_columns_segments(&t1_csr, b_wg, k.div_ceil(b_wg as usize));
        let t1_seg =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&seg_host, vec![seg_host.len()], &client)
                .unwrap();

        let blocks = B_ARGMIN_BLOCKS.min(n.max(1) as u32) as usize;
        let part_val = GpuTensor::<WgpuRuntime, f32>::empty(vec![blocks * k], &client).unwrap();
        let part_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![blocks * k], &client).unwrap();
        let gap_partial = GpuTensor::<WgpuRuntime, f32>::empty(vec![blocks], &client).unwrap();
        let out_val = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client).unwrap();
        let out_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![k], &client).unwrap();

        launch_fw_argmin_b(
            &k2b_gpu,
            &t1_gpu,
            &t1_seg,
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
    /// The two paths sum the same products in different orders, so columns
    /// whose two best rows sit within a few last bits of each other can
    /// legitimately resolve either way. An exact index match is demanded
    /// wherever the minimum is unambiguous; where it is not, what has to hold
    /// is that the row the kernel picked really is a minimum.
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

    /// One shape of the fused-kernel parity check.
    ///
    /// The kernel must reproduce the CPU scan it replaces: same argmin per
    /// archetype, same minimum, same duality-gap term.
    ///
    /// ### Params
    ///
    /// * `n` - Cells
    /// * `k` - Archetypes
    /// * `device` - Device to launch on
    fn check_argmin_arm(n: usize, k: usize, device: &WgpuDevice) {
        let k2b = fixture(n * k, 1, 3);
        let t1 = fixture(k * k, 5, 4);
        let t2 = fixture(n * k, 9, 5);
        let b_mat = fixture(n * k, 13, 11);

        let (want_val, want_idx, want_gap) = cpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k);
        let (got_val, got_idx, got_gap) = gpu_fw_argmin_b(&k2b, &t1, &t2, &b_mat, n, k, device);

        for c in 0..k {
            assert_relative_eq!(got_val[c], want_val[c], max_relative = 1e-4, epsilon = 1e-5);
        }
        assert_argmins_agree(&k2b, &t1, &t2, &got_idx, &want_idx, &want_val, n, k);
        assert_relative_eq!(got_gap, want_gap, max_relative = 1e-3, epsilon = 1e-4);
    }

    /// Parity at shapes cheap enough to run on every push.
    ///
    /// The two structural properties the kernel turns on, multi-slot ownership
    /// and the grid stride, are both degenerate at these sizes. They are covered
    /// by `test_fw_argmin_b_matches_cpu_large`.
    #[test]
    fn test_fw_argmin_b_matches_cpu() {
        let Some(device) = try_device() else { return };

        for (n, k) in [(40usize, 7usize), (257, 33), (1000, 130)] {
            check_argmin_arm(n, k, &device);
        }
    }

    /// Parity at the shapes that actually exercise the kernel's structure.
    ///
    /// - `slots = ceil(k / wg)`, the contiguous column ownership held in
    ///   registers. It is 1 at small `k`, so those shapes never exercise the
    ///   multi-slot unrolling or the "find the owning slot by comparison" trick
    ///   in the `K²Aᵀ` and gap loops. `k = 300` and `k = 1100` do.
    /// - the grid stride. Blocks are capped at `B_ARGMIN_BLOCKS`, so below
    ///   `n = 1024` each block owns exactly one row and the per-block running
    ///   minimum never actually accumulates. `n = 3000` gives about three rows
    ///   per block.
    #[test]
    // Heavy: these are also the two arms that cost anything, because the CPU
    // reference is O(n k²).
    #[cfg(feature = "large_scale_diagnostics")]
    fn test_fw_argmin_b_matches_cpu_large() {
        let Some(device) = try_device() else { return };

        for (n, k) in [(3000usize, 300usize), (300, 1100)] {
            check_argmin_arm(n, k, &device);
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
