//! GPU implementation of the multi-output tree regression from
//! `sc_analysis/scenic.rs`: wave-scheduled, multi-tree, multi-batch
//! ExtraTrees and RandomForest, dispatched per level via
//! [`evaluate_splits_et`] (random threshold) or [`evaluate_splits_rf`]
//! (exhaustive threshold) on `config.random_threshold()`. Six-kernel
//! level-synchronous BFS pipeline, all kernels running at full
//! `WORKGROUP_128` width using the atomic-free segmented pattern from
//! `gpu/ml/k_means_gpu.rs::segmented_centroid_update` and the SMEM tree
//! reduction from `gpu/sc_gpu/kernels/harmony_kernels.rs::objective_partials`.
//!
//! ### Level-BFS driver
//!
//! [`run_wave_bfs`] walks every tree in a wave depth-first-free, i.e. level
//! by level: [`sample_node_features`], [`build_hist_privatised`],
//! [`merge_hist`], [`prefix_sum_bins`], [`evaluate_splits_et`] /
//! [`evaluate_splits_rf`], [`accumulate_importance`], then
//! [`compute_child_ids`] and [`reassign_samples`] to seed the next level.
//! [`init_sample_to_node`] runs once before the level loop. `n_active_nodes`
//! per level is a static upper bound (`min(2^depth, max_active_nodes)`), so
//! there is no per-level host readback; phantom nodes with no samples cost a
//! cheap kernel launch and no atomics.
//!
//! ### Wave scheduler
//!
//! Trees are processed in waves of `W` at a time; per-tree tensors carry a
//! leading `tree_in_wave` dimension and every kernel grid has an extra `Z`
//! dim for the tree. `W` is chosen from a VRAM budget to stay under a
//! byte-cost ceiling; if the batch doesn't fit at `W = 8` the wave shrinks
//! by halves until it does. Falls back to `W = 1` and errors if even that
//! busts the budget.
//!
//! ### On-device PRNG
//!
//! Per-tree seeds are computed on the host via `tree_seed(seed, tree_idx)`
//! and uploaded once per wave. Feature draws and threshold draws happen on
//! device via `hash_mix(tree_seed ^ level ^ node ^ slot [^ thr_idx])`. Feature
//! draws allow rare duplicates -- at n_features > 3 * k_feats the collision
//! probability is <10% and duplicates just re-evaluate the same feature under
//! different thresholds. Determinism across runs (same seed -> same
//! importances) is preserved; determinism across CPU/GPU is not required.
//!
//! ### Multi-batch loop
//!
//! `fit_multi_trees_gpu` splits the target list into batches of at most
//! `MULTI_OUTPUT_BATCH = 64` targets. Wave-sized scratch buffers are
//! allocated once per batch (they depend on n_targets) and reused across all
//! trees in the ensemble. Feature bins are uploaded once for the whole call.
//! Per-target importances are extracted and normalised at batch end.

#![allow(missing_docs)]
#![cfg(all(feature = "single-cell", feature = "gpu"))]

use ann_search_rs::gpu::grid_2d;
use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::gpu::{WORKGROUP_32, WORKGROUP_128};
use crate::gpu::sc_gpu::scenic_gpu_params::ScenicGpuParams;
use crate::prelude::*;
use crate::single_cell::sc_analysis::scenic::*;
use crate::single_cell::sc_utils::utils_tree::*;

////////////
// Consts //
////////////

/// Number of quantisation bins (must match CPU's u8 layout).
const N_BINS: u32 = 256;

/// Default wave size. Halved until the batch fits the caller's byte budget
/// (see [`ScenicGpuParams::wave_byte_budget`]).
const DEFAULT_WAVE_SIZE: usize = 8;

/// Sentinel for "no node" / "no valid split" / "no child" in the u32-typed
/// device buffers (`split_feature`, `sample_to_node`, `left_child_id`,
/// `right_child_id`). Referenced directly inside `#[cube]` kernel bodies,
/// same as [`N_BINS`].
const INVALID_NODE: u32 = u32::MAX;

/////////////
// Kernels //
/////////////

/// Draw `k_feats` feature ids per (tree, node) into `node_features`. One
/// workgroup per (node, tree), `WORKGROUP_128` wide. Thread `tx` owns slots
/// `tx, tx+wg, ...` and independently hashes `(tree_seed, level, node, slot)`
/// into a feature id in `[0, n_features)`. Duplicates within a node's k_feats
/// are allowed -- at k_feats << n_features they are rare and the algorithm
/// tolerates them (a duplicated feature just gets re-evaluated).
///
/// ### Params
///
/// * `tree_seeds` - Per-tree base seed `[wave_size]`
/// * `node_features` - Output feature ids
///   `[wave_size, n_active_nodes, k_feats]`
/// * `wave_size` - Number of trees in this wave
/// * `n_active_nodes` - Active nodes at this level
/// * `k_feats` - Features per node
/// * `n_features` - Total feature count (draw range)
/// * `level` - Depth being processed
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> feature-slot stride offset
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn sample_node_features(
    tree_seeds: &Tensor<u32>,
    node_features: &mut Tensor<u32>,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_features: u32,
    level: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let tree = CUBE_POS_Z;
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let seed = tree_seeds[tree as usize];
    let base = ((tree * n_active_nodes + node) * k_feats) as usize;

    let mut slot: u32 = tx;
    while slot < k_feats {
        // hash chain matches the threshold-draw pattern in evaluate_splits_et
        let mut h = seed;
        h = hash_mix(h ^ level);
        h = hash_mix(h ^ node);
        h = hash_mix(h ^ slot);
        node_features[base + slot as usize] = h % n_features;
        slot += wg_size;
    }
}

/// CAS-loop atomic f32 add on a `Atomic<u32>` slot holding f32 bits. WGSL
/// has no native atomic f32 op; we bit-reinterpret and retry until the
/// observed old value matches our compare value. Cubecl's
/// `compare_exchange_weak` is defined on `Atomic<u32>` only, hence the u32
/// storage of what the caller conceptually treats as f32.
#[cube]
#[allow(clippy::assign_op_pattern)]
fn atomic_add_f32_bits(ptr: &Atomic<u32>, delta: f32) {
    let mut old_bits: u32 = Atomic::load(ptr);
    let mut done: u32 = 0u32;
    while done == 0u32 {
        let old_f = f32::from_bits(old_bits);
        let new_bits = (old_f + delta).to_bits();
        let observed = Atomic::compare_exchange_weak(ptr, old_bits, new_bits);
        if observed == old_bits {
            done = 1u32;
        } else {
            old_bits = observed;
        }
    }
}

/// Build per-(tree, node, feature-slot) histograms, sample-parallel with
/// atomic accumulation into a per-workgroup private histogram slice. One
/// workgroup per (slot, node, tree), `WORKGROUP_128` wide. Each thread strides
/// over samples `s = tx, tx+wg, ...`; every active sample bumps its owning
/// bin. Since each workgroup owns its own histogram slice
/// `[wave, node, slot, bin, target]`, atomics contend only *within* a
/// workgroup, never across workgroups.
///
/// Y-sum accumulation goes through a CAS-loop on `Atomic<u32>` treating the
/// stored bits as f32 (WGSL has no native atomic f32). Counts are
/// `Atomic::fetch_add` on `Atomic<u32>` which lowers natively.
///
/// Compared to a bin-per-thread design (one thread per bin walks all N
/// samples per bin), this cuts `feature_data` reads by roughly 256x: each
/// sample is read once instead of once per owned bin per workgroup.
///
/// ### Params
///
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `sy_offsets` - Sparse Y offsets `[n_samples + 1]`
/// * `sy_target_indices` - Sparse Y target ids `[nnz]` (u8 as u32)
/// * `sy_values` - Sparse Y values `[nnz]`
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`
/// * `sample_multiplicity` - Per-tree sample multiplier `[wave_size, n_samples]`
/// * `node_features` - Selected feature ids per (tree, node)
///   `[wave_size, n_active_nodes, k_feats]`
/// * `hist_counts` - Output counts as atomic u32
///   `[wave_size, n_active_nodes, k_feats, N_BINS]`
/// * `hist_y_sums` - Output Y sums as atomic u32 (f32 bits)
///   `[wave_size, n_active_nodes, k_feats, N_BINS, n_targets]`
/// * `hist_y_sum_sqs` - Output Y sum-of-squares as atomic u32 (f32 bits, same layout)
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> feature slot index
/// * `CUBE_POS_Y` -> node index (max 65535, so `grid_2d` is not needed here)
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> sample stripe (thread `tx` walks `s = tx, tx+wg, ...`)
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn build_hist_privatised(
    feature_data: &Tensor<u32>,
    sy_offsets: &Tensor<u32>,
    sy_target_indices: &Tensor<u32>,
    sy_values: &Tensor<f32>,
    sample_to_node: &Tensor<u32>,
    sample_multiplicity: &Tensor<u32>,
    node_features: &Tensor<u32>,
    hist_counts: &mut Tensor<Atomic<u32>>,
    hist_y_sums: &mut Tensor<Atomic<u32>>,
    hist_y_sum_sqs: &mut Tensor<Atomic<u32>>,
    n_samples: u32,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    #[comptime] wg_size: u32,
) {
    let slot = CUBE_POS_X;
    let node = CUBE_POS_Y;
    let tree = CUBE_POS_Z;
    if slot >= k_feats {
        terminate!();
    }
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let feat = node_features[((tree * n_active_nodes + node) * k_feats + slot) as usize];

    let count_base =
        (((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) as usize;
    let sum_base =
        ((((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) * n_targets) as usize;
    let feat_base = (feat * n_samples) as usize;
    let s2n_base = (tree * n_samples) as usize;

    // Cooperative zero of this workgroup's private histogram slice. Threads
    // stride over the N_BINS counts and the N_BINS * n_targets sum slots.
    let mut zb: u32 = tx;
    while zb < N_BINS {
        Atomic::store(&hist_counts[count_base + zb as usize], 0u32);
        zb += wg_size;
    }
    let sum_slots = N_BINS * n_targets;
    let mut zs: u32 = tx;
    while zs < sum_slots {
        Atomic::store(&hist_y_sums[sum_base + zs as usize], 0u32);
        Atomic::store(&hist_y_sum_sqs[sum_base + zs as usize], 0u32);
        zs += wg_size;
    }
    sync_cube();

    // Sample-parallel accumulation. Each thread walks its stripe of samples,
    // atomically bumping the bin owned by that sample. Multiplicity 0 samples
    // (bootstrap unselected, or subsample rejected) skip cleanly; sample_to_node
    // set to INVALID_NODE for those (see init_sample_to_node) also fails the
    // node-match test.
    let mut s: u32 = tx;
    while s < n_samples {
        if sample_to_node[s2n_base + s as usize] == node {
            let mult = sample_multiplicity[s2n_base + s as usize];
            if mult > 0u32 {
                let bin = feature_data[feat_base + s as usize];
                Atomic::fetch_add(&hist_counts[count_base + bin as usize], mult);
                let mult_f = f32::cast_from(mult);
                let bin_base = sum_base + (bin * n_targets) as usize;
                let off_s = sy_offsets[s as usize];
                let off_e = sy_offsets[(s + 1u32) as usize];
                let mut j = off_s;
                while j < off_e {
                    let k = sy_target_indices[j as usize];
                    let v = sy_values[j as usize];
                    let mv = mult_f * v;
                    atomic_add_f32_bits(&hist_y_sums[bin_base + k as usize], mv);
                    atomic_add_f32_bits(&hist_y_sum_sqs[bin_base + k as usize], mv * v);
                    j += 1u32;
                }
            }
        }
        s += wg_size;
    }
}

/// Compute per-node totals from the slot-0 histogram. One workgroup per
/// (node, tree), `WORKGROUP_128` wide. Thread 0 does the counts scan;
/// thread `tx` owns targets `tx, tx+wg, ...` for the y-sum totals.
///
/// `hist_y_sums` and `hist_y_sum_sqs` are stored as `u32` (f32 bits written
/// by [`build_hist_privatised`] via atomic CAS); reads reinterpret via
/// `f32::from_bits`.
///
/// ### Params
///
/// * `hist_counts` - Per-slot histogram counts `[wave_size, n_active_nodes,
///   k_feats, N_BINS]`; only slot 0 is read (identical across slots)
/// * `hist_y_sums` - Per-slot Y sums, u32 f32 bits, same layout with a
///   trailing `n_targets`
/// * `hist_y_sum_sqs` - Per-slot Y sum-of-squares, same layout as
///   `hist_y_sums`
/// * `node_counts` - Output per-node sample totals `[wave_size,
///   n_active_nodes]`
/// * `node_y_sums` - Output per-node Y sums `[wave_size, n_active_nodes,
///   n_targets]`, native f32
/// * `node_y_sum_sqs` - Output per-node Y sum-of-squares, same layout as
///   `node_y_sums`
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> target stride offset for the y-sum scan
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn merge_hist(
    hist_counts: &Tensor<u32>,
    hist_y_sums: &Tensor<u32>,
    hist_y_sum_sqs: &Tensor<u32>,
    node_counts: &mut Tensor<u32>,
    node_y_sums: &mut Tensor<f32>,
    node_y_sum_sqs: &mut Tensor<f32>,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let tree = CUBE_POS_Z;
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let count_base = (((tree * n_active_nodes + node) * k_feats) * N_BINS) as usize;
    let sum_base = ((((tree * n_active_nodes + node) * k_feats) * N_BINS) * n_targets) as usize;
    let stats_out = ((tree * n_active_nodes + node) * n_targets) as usize;

    if tx == 0u32 {
        let mut total: u32 = 0u32;
        let mut b: u32 = 0u32;
        while b < N_BINS {
            total += hist_counts[count_base + b as usize];
            b += 1u32;
        }
        node_counts[(tree * n_active_nodes + node) as usize] = total;
    }

    let mut k: u32 = tx;
    while k < n_targets {
        let mut sum: f32 = 0f32;
        let mut ssq: f32 = 0f32;
        let mut b: u32 = 0u32;
        while b < N_BINS {
            let bin_base = sum_base + (b * n_targets) as usize;
            sum += f32::from_bits(hist_y_sums[bin_base + k as usize]);
            ssq += f32::from_bits(hist_y_sum_sqs[bin_base + k as usize]);
            b += 1u32;
        }
        node_y_sums[stats_out + k as usize] = sum;
        node_y_sum_sqs[stats_out + k as usize] = ssq;
        k += wg_size;
    }
}

/// Inclusive prefix sum over 256 bins per (tree, node, slot). One workgroup
/// per (slot, node, tree), `WORKGROUP_128` wide. Thread 0 runs the counts
/// scan and, in the same pass, computes the per-slot informative bin range
/// `[min_bin, max_bin]` (first and last bins with nonzero counts) into
/// `slot_min_bin` / `slot_max_bin`. Downstream `evaluate_splits_*` read
/// these two u32s per slot instead of rescanning all 256 bins per candidate.
///
/// Thread `tx` owns targets `tx, tx+wg, ...` for y-sum scans. Each scan only
/// touches its own history so no cross-thread ordering is needed.
///
/// Empty-slot encoding: `min_bin = 0, max_bin = 0` when the slot has no
/// samples in any bin. `evaluate_splits_*` reject via `max_bin > min_bin`.
///
/// `hist_y_sums` and `hist_y_sum_sqs` are u32-typed (f32 bits written by
/// [`build_hist_privatised`] via atomic CAS); reads reinterpret via
/// `f32::from_bits`. Outputs (`cum_*`) stay in native f32.
///
/// ### Params
///
/// * `hist_counts` - Per-slot histogram counts `[wave_size, n_active_nodes,
///   k_feats, N_BINS]`
/// * `hist_y_sums` - Per-slot Y sums, u32 f32 bits, same layout with a
///   trailing `n_targets`
/// * `hist_y_sum_sqs` - Per-slot Y sum-of-squares, same layout as
///   `hist_y_sums`
/// * `cum_counts` - Output inclusive prefix sums, same layout as
///   `hist_counts`
/// * `cum_y_sums` - Output inclusive prefix sums, native f32, same layout as
///   `hist_y_sums`
/// * `cum_y_sum_sqs` - Output inclusive prefix sums, native f32, same layout
///   as `hist_y_sum_sqs`
/// * `slot_min_bin` - Output first informative bin per slot `[wave_size,
///   n_active_nodes, k_feats]`
/// * `slot_max_bin` - Output last informative bin per slot, same layout as
///   `slot_min_bin`
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> feature slot index
/// * `CUBE_POS_Y` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> target stride offset for the y-sum scans
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn prefix_sum_bins(
    hist_counts: &Tensor<u32>,
    hist_y_sums: &Tensor<u32>,
    hist_y_sum_sqs: &Tensor<u32>,
    cum_counts: &mut Tensor<u32>,
    cum_y_sums: &mut Tensor<f32>,
    cum_y_sum_sqs: &mut Tensor<f32>,
    slot_min_bin: &mut Tensor<u32>,
    slot_max_bin: &mut Tensor<u32>,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    #[comptime] wg_size: u32,
) {
    let slot = CUBE_POS_X;
    let node = CUBE_POS_Y;
    let tree = CUBE_POS_Z;
    if slot >= k_feats {
        terminate!();
    }
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let slot_flat = ((tree * n_active_nodes + node) * k_feats + slot) as usize;
    let count_base = slot_flat * N_BINS as usize;
    let sum_base =
        ((((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) * n_targets) as usize;

    if tx == 0u32 {
        // Fused: inclusive prefix sum + per-slot min/max informative bin
        // scan. Both run in one 256-iter loop so we don't pay two passes.
        let mut min_b: u32 = 0u32;
        let mut has_min: u32 = 0u32;
        let mut max_b: u32 = 0u32;

        let first = hist_counts[count_base];
        cum_counts[count_base] = first;
        if first > 0u32 {
            min_b = 0u32;
            has_min = 1u32;
            max_b = 0u32;
        }
        let mut b: u32 = 1u32;
        while b < N_BINS {
            let prev = count_base + (b - 1u32) as usize;
            let curr = count_base + b as usize;
            let hc = hist_counts[curr];
            cum_counts[curr] = cum_counts[prev] + hc;
            if hc > 0u32 {
                if has_min == 0u32 {
                    min_b = b;
                    has_min = 1u32;
                }
                max_b = b;
            }
            b += 1u32;
        }

        if has_min == 1u32 {
            slot_min_bin[slot_flat] = min_b;
            slot_max_bin[slot_flat] = max_b;
        } else {
            slot_min_bin[slot_flat] = 0u32;
            slot_max_bin[slot_flat] = 0u32;
        }
    }

    let mut k: u32 = tx;
    while k < n_targets {
        cum_y_sums[sum_base + k as usize] = f32::from_bits(hist_y_sums[sum_base + k as usize]);
        cum_y_sum_sqs[sum_base + k as usize] =
            f32::from_bits(hist_y_sum_sqs[sum_base + k as usize]);

        let mut b: u32 = 1u32;
        while b < N_BINS {
            let prev_s = sum_base + ((b - 1u32) * n_targets) as usize + k as usize;
            let curr_s = sum_base + (b * n_targets) as usize + k as usize;
            cum_y_sums[curr_s] =
                cum_y_sums[prev_s] + f32::from_bits(hist_y_sums[curr_s]);
            cum_y_sum_sqs[curr_s] =
                cum_y_sum_sqs[prev_s] + f32::from_bits(hist_y_sum_sqs[curr_s]);
            b += 1u32;
        }
        k += wg_size;
    }
}

/// Cheap on-device hash for feature/threshold selection. Multiplies wrap in
/// shader arithmetic (WGSL and SPIR-V both define `*` on `u32` mod 2^32).
#[cube]
#[allow(clippy::assign_op_pattern)]
fn hash_mix(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16u32;
    h = h * 2246822507u32;
    h ^= h >> 13u32;
    h = h * 3266489909u32;
    h ^= h >> 16u32;
    h
}

/// Evaluate ExtraTrees random-threshold splits. One workgroup per (node,
/// tree), `WORKGROUP_128` wide. Thread `tx` handles candidates
/// `tx, tx+wg, ...` (candidate `c` decodes as `slot = c / n_thresholds`,
/// `thr_idx = c % n_thresholds`), keeps its running best in registers, then
/// participates in a manually-unrolled SMEM tree argmax (128 -> 64 -> ... -> 1).
/// Thread 0 writes the winning split; all threads then fan out again on the
/// target dim to copy the winning bin's left-child Y stats.
///
/// ### Params
///
/// * `cum_counts` - Inclusive prefix-sum counts from [`prefix_sum_bins`]
/// * `cum_y_sums` - Inclusive prefix-sum Y sums from [`prefix_sum_bins`]
/// * `cum_y_sum_sqs` - Inclusive prefix-sum Y sum-of-squares from
///   [`prefix_sum_bins`]
/// * `node_counts` - Per-node sample totals from [`merge_hist`]
/// * `node_y_sums` - Per-node Y sums from [`merge_hist`]
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares from [`merge_hist`]
/// * `node_features` - Selected feature ids per (tree, node, slot) from
///   [`sample_node_features`]
/// * `tree_seeds` - Per-tree base seed `[wave_size]`
/// * `slot_min_bin` - First informative bin per slot from [`prefix_sum_bins`]
/// * `slot_max_bin` - Last informative bin per slot from [`prefix_sum_bins`]
/// * `split_feature` - Output winning feature id per node, or `u32::MAX` if
///   no valid split
/// * `split_threshold` - Output winning threshold bin per node
/// * `split_n_left` - Output left-child sample count per node
/// * `split_y_sums_l` - Output left-child Y sums per (node, target)
/// * `split_y_sum_sqs_l` - Output left-child Y sum-of-squares per (node,
///   target)
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_thresholds` - Random thresholds drawn per feature slot
/// * `min_samples_leaf` - Minimum samples required on both sides of a split
/// * `level` - Depth being processed
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> candidate stride offset (`slot * n_thresholds + thr_idx`)
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn evaluate_splits_et(
    cum_counts: &Tensor<u32>,
    cum_y_sums: &Tensor<f32>,
    cum_y_sum_sqs: &Tensor<f32>,
    node_counts: &Tensor<u32>,
    node_y_sums: &Tensor<f32>,
    node_y_sum_sqs: &Tensor<f32>,
    node_features: &Tensor<u32>,
    tree_seeds: &Tensor<u32>,
    slot_min_bin: &Tensor<u32>,
    slot_max_bin: &Tensor<u32>,
    split_feature: &mut Tensor<u32>,
    split_threshold: &mut Tensor<u32>,
    split_n_left: &mut Tensor<u32>,
    split_y_sums_l: &mut Tensor<f32>,
    split_y_sum_sqs_l: &mut Tensor<f32>,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    n_thresholds: u32,
    min_samples_leaf: u32,
    level: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let tree = CUBE_POS_Z;
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let node_flat = (tree * n_active_nodes + node) as usize;
    let n = node_counts[node_flat];
    let stats_base = node_flat * n_targets as usize;
    let nf = f32::cast_from(n);
    let tree_seed_val = tree_seeds[tree as usize];

    // parent variance sum: thread 0 computes and broadcasts
    let mut s_parent_var = SharedMemory::<f32>::new(1usize);
    if tx == 0u32 {
        let mut pv: f32 = 0f32;
        if n >= 2u32 {
            let mut k: u32 = 0u32;
            while k < n_targets {
                let s = node_y_sums[stats_base + k as usize];
                let ss = node_y_sum_sqs[stats_base + k as usize];
                let mean = s / nf;
                let v = ss / nf - mean * mean;
                pv += f32::max(v, 0f32);
                k += 1u32;
            }
        }
        s_parent_var[0] = pv;
    }
    sync_cube();
    let parent_var_sum = s_parent_var[0];

    if n < 2u32 * min_samples_leaf {
        if tx == 0u32 {
            split_feature[node_flat] = INVALID_NODE;
            split_threshold[node_flat] = 0u32;
            split_n_left[node_flat] = 0u32;
        }
        terminate!();
    }
    if parent_var_sum <= 0f32 {
        if tx == 0u32 {
            split_feature[node_flat] = INVALID_NODE;
            split_threshold[node_flat] = 0u32;
            split_n_left[node_flat] = 0u32;
        }
        terminate!();
    }

    // -- candidate evaluation --
    let mut best_score: f32 = 0f32;
    let mut best_slot: u32 = 0u32;
    let mut best_thr: u32 = 0u32;
    let mut best_n_left: u32 = 0u32;
    let mut best_valid: u32 = 0u32;

    let n_candidates = k_feats * n_thresholds;
    let mut c: u32 = tx;
    while c < n_candidates {
        let slot = c / n_thresholds;
        let ti = c % n_thresholds;

        let count_base =
            (((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) as usize;
        let sum_base =
            ((((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) * n_targets) as usize;

        // Precomputed per-slot informative range (populated in
        // prefix_sum_bins). Empty slots have min_bin == max_bin == 0.
        let slot_flat = ((tree * n_active_nodes + node) * k_feats + slot) as usize;
        let min_bin = slot_min_bin[slot_flat];
        let max_bin = slot_max_bin[slot_flat];

        if max_bin > min_bin {
            let mut seed_mix = tree_seed_val;
            seed_mix = hash_mix(seed_mix ^ level);
            seed_mix = hash_mix(seed_mix ^ node);
            seed_mix = hash_mix(seed_mix ^ slot);
            seed_mix = hash_mix(seed_mix ^ ti);
            // shift threshold hash off the feature-draw hash so the two
            // don't correlate; XOR with a fixed salt is cheap.
            seed_mix = hash_mix(seed_mix ^ 2654435769u32);
            let range = max_bin - min_bin;
            let thr = min_bin + (seed_mix % range);

            let n_left = cum_counts[count_base + thr as usize];
            let n_right = n - n_left;

            let ok_left = n_left >= min_samples_leaf;
            let ok_right = n_right >= min_samples_leaf;
            if ok_left && ok_right {
                let nl = f32::cast_from(n_left);
                let nr = f32::cast_from(n_right);
                let inv_nl = 1f32 / nl;
                let inv_nr = 1f32 / nr;
                let wl = nl / nf;
                let wr = nr / nf;

                let mut score: f32 = 0f32;
                let bin_base = sum_base + (thr * n_targets) as usize;
                let mut k: u32 = 0u32;
                while k < n_targets {
                    let sy = node_y_sums[stats_base + k as usize];
                    let ssq = node_y_sum_sqs[stats_base + k as usize];
                    let mean_p = sy / nf;
                    let var_p = ssq / nf - mean_p * mean_p;

                    let syl = cum_y_sums[bin_base + k as usize];
                    let ssyl = cum_y_sum_sqs[bin_base + k as usize];
                    let mean_l = syl * inv_nl;
                    let var_l = ssyl * inv_nl - mean_l * mean_l;

                    let syr = sy - syl;
                    let ssyr = ssq - ssyl;
                    let mean_r = syr * inv_nr;
                    let var_r = ssyr * inv_nr - mean_r * mean_r;

                    let vp = f32::max(var_p, 0f32);
                    let vl = f32::max(var_l, 0f32);
                    let vr = f32::max(var_r, 0f32);
                    score += vp - wl * vl - wr * vr;
                    k += 1u32;
                }

                if score > best_score {
                    best_score = score;
                    best_slot = slot;
                    best_thr = thr;
                    best_n_left = n_left;
                    best_valid = 1u32;
                }
            }
        }

        c += wg_size;
    }

    // -- argmax reduction (32-wide, 5 halving stages 16->8->4->2->1) --
    let mut s_score = SharedMemory::<f32>::new(WORKGROUP_32 as usize);
    let mut s_slot = SharedMemory::<u32>::new(WORKGROUP_32 as usize);
    let mut s_thr = SharedMemory::<u32>::new(WORKGROUP_32 as usize);
    let mut s_nl = SharedMemory::<u32>::new(WORKGROUP_32 as usize);
    let mut s_valid = SharedMemory::<u32>::new(WORKGROUP_32 as usize);

    s_score[tx as usize] = best_score;
    s_slot[tx as usize] = best_slot;
    s_thr[tx as usize] = best_thr;
    s_nl[tx as usize] = best_n_left;
    s_valid[tx as usize] = best_valid;
    sync_cube();

    if tx < 16u32 {
        let mate = tx + 16u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 8u32 {
        let mate = tx + 8u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 4u32 {
        let mate = tx + 4u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 2u32 {
        let mate = tx + 2u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 1u32 {
        let mate = tx + 1u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();

    let winner_valid = s_valid[0];
    let winner_slot = s_slot[0];
    let winner_thr = s_thr[0];
    let winner_n_left = s_nl[0];

    if tx == 0u32 {
        if winner_valid == 1u32 {
            let feat = node_features
                [((tree * n_active_nodes + node) * k_feats + winner_slot) as usize];
            split_feature[node_flat] = feat;
            split_threshold[node_flat] = winner_thr;
            split_n_left[node_flat] = winner_n_left;
        } else {
            split_feature[node_flat] = INVALID_NODE;
            split_threshold[node_flat] = 0u32;
            split_n_left[node_flat] = 0u32;
        }
    }

    if winner_valid == 1u32 {
        let bin_base = ((((tree * n_active_nodes + node) * k_feats + winner_slot) * N_BINS)
            * n_targets) as usize
            + (winner_thr * n_targets) as usize;
        let mut k: u32 = tx;
        while k < n_targets {
            split_y_sums_l[stats_base + k as usize] = cum_y_sums[bin_base + k as usize];
            split_y_sum_sqs_l[stats_base + k as usize] = cum_y_sum_sqs[bin_base + k as usize];
            k += wg_size;
        }
    }
}

/// Argmax reduction decision: returns 1 iff mate slot should overwrite
/// current slot. Ties resolve to the lower-indexed thread (strict `>`).
#[cube]
fn argmax_takes_mate(cur_valid: u32, cur_score: f32, mate_valid: u32, mate_score: f32) -> u32 {
    let mut take: u32 = 0u32;
    if mate_valid == 1u32 {
        if cur_valid == 0u32 {
            take = 1u32;
        } else {
            if mate_score > cur_score {
                take = 1u32;
            }
        }
    }
    take
}

/// Evaluate RandomForest exhaustive-threshold splits. One workgroup per
/// (node, tree), `WORKGROUP_128` wide. Candidate space is the flattened
/// `(slot x threshold)` grid with 255 thresholds per slot (bins 0..254 --
/// bin 255 as a threshold always sends every sample left, gets rejected
/// by min_samples_leaf). Thresholds outside `[min_bin, max_bin)` for a
/// slot are naturally rejected by the `n_left / n_right >= min_samples_leaf`
/// gate (they produce n_left = 0 or n_right = 0), matching CPU's implicit
/// pruning without an explicit bin scan.
///
/// Same SMEM argmax reduction and left-child copy fan-out as
/// `evaluate_splits_et`. Reuses the shared `argmax_takes_mate` decision.
///
/// ### Params
///
/// * `cum_counts` - Inclusive prefix-sum counts from [`prefix_sum_bins`]
/// * `cum_y_sums` - Inclusive prefix-sum Y sums from [`prefix_sum_bins`]
/// * `cum_y_sum_sqs` - Inclusive prefix-sum Y sum-of-squares from
///   [`prefix_sum_bins`]
/// * `node_counts` - Per-node sample totals from [`merge_hist`]
/// * `node_y_sums` - Per-node Y sums from [`merge_hist`]
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares from [`merge_hist`]
/// * `node_features` - Selected feature ids per (tree, node, slot) from
///   [`sample_node_features`]
/// * `slot_min_bin` - First informative bin per slot from [`prefix_sum_bins`]
/// * `slot_max_bin` - Last informative bin per slot from [`prefix_sum_bins`]
/// * `split_feature` - Output winning feature id per node, or `u32::MAX` if
///   no valid split
/// * `split_threshold` - Output winning threshold bin per node
/// * `split_n_left` - Output left-child sample count per node
/// * `split_y_sums_l` - Output left-child Y sums per (node, target)
/// * `split_y_sum_sqs_l` - Output left-child Y sum-of-squares per (node,
///   target)
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `min_samples_leaf` - Minimum samples required on both sides of a split
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> candidate stride offset (`slot * (N_BINS - 1) + thr`)
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn evaluate_splits_rf(
    cum_counts: &Tensor<u32>,
    cum_y_sums: &Tensor<f32>,
    cum_y_sum_sqs: &Tensor<f32>,
    node_counts: &Tensor<u32>,
    node_y_sums: &Tensor<f32>,
    node_y_sum_sqs: &Tensor<f32>,
    node_features: &Tensor<u32>,
    slot_min_bin: &Tensor<u32>,
    slot_max_bin: &Tensor<u32>,
    split_feature: &mut Tensor<u32>,
    split_threshold: &mut Tensor<u32>,
    split_n_left: &mut Tensor<u32>,
    split_y_sums_l: &mut Tensor<f32>,
    split_y_sum_sqs_l: &mut Tensor<f32>,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    min_samples_leaf: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let tree = CUBE_POS_Z;
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let node_flat = (tree * n_active_nodes + node) as usize;
    let n = node_counts[node_flat];
    let stats_base = node_flat * n_targets as usize;
    let nf = f32::cast_from(n);

    // parent variance broadcast (same shape as ET)
    let mut s_parent_var = SharedMemory::<f32>::new(1usize);
    if tx == 0u32 {
        let mut pv: f32 = 0f32;
        if n >= 2u32 {
            let mut k: u32 = 0u32;
            while k < n_targets {
                let s = node_y_sums[stats_base + k as usize];
                let ss = node_y_sum_sqs[stats_base + k as usize];
                let mean = s / nf;
                let v = ss / nf - mean * mean;
                pv += f32::max(v, 0f32);
                k += 1u32;
            }
        }
        s_parent_var[0] = pv;
    }
    sync_cube();
    let parent_var_sum = s_parent_var[0];

    if n < 2u32 * min_samples_leaf {
        if tx == 0u32 {
            split_feature[node_flat] = INVALID_NODE;
            split_threshold[node_flat] = 0u32;
            split_n_left[node_flat] = 0u32;
        }
        terminate!();
    }
    if parent_var_sum <= 0f32 {
        if tx == 0u32 {
            split_feature[node_flat] = INVALID_NODE;
            split_threshold[node_flat] = 0u32;
            split_n_left[node_flat] = 0u32;
        }
        terminate!();
    }

    // Exhaustive candidate space: k_feats slots x (N_BINS - 1) thresholds.
    // 255 thresholds per slot covers every valid split; bin 255 as a
    // threshold gives n_right = 0 and is uniformly rejected downstream.
    let thresholds_per_slot = N_BINS - 1u32;
    let n_candidates = k_feats * thresholds_per_slot;

    let mut best_score: f32 = 0f32;
    let mut best_slot: u32 = 0u32;
    let mut best_thr: u32 = 0u32;
    let mut best_n_left: u32 = 0u32;
    let mut best_valid: u32 = 0u32;

    let mut c: u32 = tx;
    while c < n_candidates {
        let slot = c / thresholds_per_slot;
        let thr = c % thresholds_per_slot;

        // Skip thresholds outside the slot's informative range. Below
        // `min_bin` all cum_counts are 0 (n_left = 0). At or above `max_bin`
        // every sample sits on the left (n_right = 0). Both cases would be
        // gated out by the min_samples_leaf check below anyway, so this is
        // a pure early-out that avoids the target-loop scoring cost.
        // (cubecl has no `continue`, so the rest of the body sits under a
        // single `if in_range` guard instead.)
        let slot_flat = ((tree * n_active_nodes + node) * k_feats + slot) as usize;
        let min_bin = slot_min_bin[slot_flat];
        let max_bin = slot_max_bin[slot_flat];
        let in_range = max_bin > min_bin && thr >= min_bin && thr < max_bin;

        if in_range {
        let count_base =
            (((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) as usize;
        let sum_base =
            ((((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) * n_targets) as usize;

        let n_left = cum_counts[count_base + thr as usize];
        let n_right = n - n_left;

        let ok_left = n_left >= min_samples_leaf;
        let ok_right = n_right >= min_samples_leaf;
        if ok_left && ok_right {
            let nl = f32::cast_from(n_left);
            let nr = f32::cast_from(n_right);
            let inv_nl = 1f32 / nl;
            let inv_nr = 1f32 / nr;
            let wl = nl / nf;
            let wr = nr / nf;

            let mut score: f32 = 0f32;
            let bin_base = sum_base + (thr * n_targets) as usize;
            let mut k: u32 = 0u32;
            while k < n_targets {
                let sy = node_y_sums[stats_base + k as usize];
                let ssq = node_y_sum_sqs[stats_base + k as usize];
                let mean_p = sy / nf;
                let var_p = ssq / nf - mean_p * mean_p;

                let syl = cum_y_sums[bin_base + k as usize];
                let ssyl = cum_y_sum_sqs[bin_base + k as usize];
                let mean_l = syl * inv_nl;
                let var_l = ssyl * inv_nl - mean_l * mean_l;

                let syr = sy - syl;
                let ssyr = ssq - ssyl;
                let mean_r = syr * inv_nr;
                let var_r = ssyr * inv_nr - mean_r * mean_r;

                let vp = f32::max(var_p, 0f32);
                let vl = f32::max(var_l, 0f32);
                let vr = f32::max(var_r, 0f32);
                score += vp - wl * vl - wr * vr;
                k += 1u32;
            }

            if score > best_score {
                best_score = score;
                best_slot = slot;
                best_thr = thr;
                best_n_left = n_left;
                best_valid = 1u32;
            }
        }
        }

        c += wg_size;
    }

    // -- argmax reduction (WORKGROUP_128: RF's 7905 candidates saturate it) --
    let mut s_score = SharedMemory::<f32>::new(WORKGROUP_128 as usize);
    let mut s_slot = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_thr = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_nl = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_valid = SharedMemory::<u32>::new(WORKGROUP_128 as usize);

    s_score[tx as usize] = best_score;
    s_slot[tx as usize] = best_slot;
    s_thr[tx as usize] = best_thr;
    s_nl[tx as usize] = best_n_left;
    s_valid[tx as usize] = best_valid;
    sync_cube();

    if tx < 64u32 {
        let mate = tx + 64u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 32u32 {
        let mate = tx + 32u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 16u32 {
        let mate = tx + 16u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 8u32 {
        let mate = tx + 8u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 4u32 {
        let mate = tx + 4u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 2u32 {
        let mate = tx + 2u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();
    if tx < 1u32 {
        let mate = tx + 1u32;
        let take = argmax_takes_mate(
            s_valid[tx as usize],
            s_score[tx as usize],
            s_valid[mate as usize],
            s_score[mate as usize],
        );
        if take == 1u32 {
            s_score[tx as usize] = s_score[mate as usize];
            s_slot[tx as usize] = s_slot[mate as usize];
            s_thr[tx as usize] = s_thr[mate as usize];
            s_nl[tx as usize] = s_nl[mate as usize];
            s_valid[tx as usize] = s_valid[mate as usize];
        }
    }
    sync_cube();

    let winner_valid = s_valid[0];
    let winner_slot = s_slot[0];
    let winner_thr = s_thr[0];
    let winner_n_left = s_nl[0];

    if tx == 0u32 {
        if winner_valid == 1u32 {
            let feat = node_features
                [((tree * n_active_nodes + node) * k_feats + winner_slot) as usize];
            split_feature[node_flat] = feat;
            split_threshold[node_flat] = winner_thr;
            split_n_left[node_flat] = winner_n_left;
        } else {
            split_feature[node_flat] = INVALID_NODE;
            split_threshold[node_flat] = 0u32;
            split_n_left[node_flat] = 0u32;
        }
    }

    if winner_valid == 1u32 {
        let bin_base = ((((tree * n_active_nodes + node) * k_feats + winner_slot) * N_BINS)
            * n_targets) as usize
            + (winner_thr * n_targets) as usize;
        let mut k: u32 = tx;
        while k < n_targets {
            split_y_sums_l[stats_base + k as usize] = cum_y_sums[bin_base + k as usize];
            split_y_sum_sqs_l[stats_base + k as usize] = cum_y_sum_sqs[bin_base + k as usize];
            k += wg_size;
        }
    }
}

/// Update sample -> node assignment for the next level. One thread per
/// (sample, tree). Sample whose parent had no valid split becomes inactive.
///
/// ### Params
///
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `split_feature` - Winning feature id per node from `evaluate_splits_*`
/// * `split_threshold` - Winning threshold bin per node
/// * `left_child_id` - Left child id per node from [`compute_child_ids`]
/// * `right_child_id` - Right child id per node from [`compute_child_ids`]
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`,
///   updated in place
/// * `n_samples` - Number of samples
/// * `n_features` - Total feature count
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
///
/// ### Grid mapping
///
/// * `CUBE_POS_X * WORKGROUP_128 + UNIT_POS_X + CUBE_POS_Y * CUBE_COUNT_X *
///   WORKGROUP_128` -> sample index
/// * `CUBE_POS_Z` -> tree_in_wave
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn reassign_samples(
    feature_data: &Tensor<u32>,
    split_feature: &Tensor<u32>,
    split_threshold: &Tensor<u32>,
    left_child_id: &Tensor<u32>,
    right_child_id: &Tensor<u32>,
    sample_to_node: &mut Tensor<u32>,
    n_samples: u32,
    n_features: u32,
    wave_size: u32,
    n_active_nodes: u32,
) {
    let s = CUBE_POS_X * WORKGROUP_128 + UNIT_POS_X
        + CUBE_POS_Y * CUBE_COUNT_X * WORKGROUP_128;
    let tree = CUBE_POS_Z;
    if s >= n_samples {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let s2n_idx = (tree * n_samples + s) as usize;
    let node = sample_to_node[s2n_idx];
    if node == INVALID_NODE {
        terminate!();
    }

    let node_flat = (tree * n_active_nodes + node) as usize;
    let feat = split_feature[node_flat];
    if feat == INVALID_NODE {
        sample_to_node[s2n_idx] = INVALID_NODE;
        terminate!();
    }
    if feat >= n_features {
        sample_to_node[s2n_idx] = INVALID_NODE;
        terminate!();
    }

    let thr = split_threshold[node_flat];
    let bin = feature_data[(feat * n_samples + s) as usize];
    if bin <= thr {
        sample_to_node[s2n_idx] = left_child_id[node_flat];
    } else {
        sample_to_node[s2n_idx] = right_child_id[node_flat];
    }
}

/// Compute per-(tree, node, target) weighted variance reduction and scatter
/// it directly into the per-batch importance accumulator. One workgroup per
/// (node, tree), `WORKGROUP_128` wide, thread `tx` owns targets
/// `tx, tx+wg, ...`. Contribution is atomically added into
/// `batch_importances[feat * n_targets + k]` (f32 bits stored as `u32`,
/// CAS-loop add). This removes the host-side scatter step; the driver reads
/// `batch_importances` once per batch instead of once per level.
///
/// ### Params
///
/// * `node_counts` - Per-node sample totals from [`merge_hist`]
/// * `node_y_sums` - Per-node Y sums from [`merge_hist`]
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares from [`merge_hist`]
/// * `split_feature` - Winning feature id per node from `evaluate_splits_*`;
///   nodes with `INVALID_NODE` contribute nothing
/// * `split_n_left` - Left-child sample count per node
/// * `split_y_sums_l` - Left-child Y sums per (node, target)
/// * `split_y_sum_sqs_l` - Left-child Y sum-of-squares per (node, target)
/// * `batch_importances` - Output `[n_features, n_targets]` atomic
///   accumulator, f32 bits stored as `u32`
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `n_targets` - Targets in batch
/// * `n_total` - Total sample count used for the node-weight normalisation
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> target stride offset
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn accumulate_importance(
    node_counts: &Tensor<u32>,
    node_y_sums: &Tensor<f32>,
    node_y_sum_sqs: &Tensor<f32>,
    split_feature: &Tensor<u32>,
    split_n_left: &Tensor<u32>,
    split_y_sums_l: &Tensor<f32>,
    split_y_sum_sqs_l: &Tensor<f32>,
    batch_importances: &mut Tensor<Atomic<u32>>,
    wave_size: u32,
    n_active_nodes: u32,
    n_targets: u32,
    n_total: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let tree = CUBE_POS_Z;
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let node_flat = (tree * n_active_nodes + node) as usize;
    let feat = split_feature[node_flat];
    if feat == INVALID_NODE {
        terminate!();
    }

    let n = node_counts[node_flat];
    let n_left = split_n_left[node_flat];
    let n_right = n - n_left;
    let nf = f32::cast_from(n);
    let nl = f32::cast_from(n_left);
    let nr = f32::cast_from(n_right);
    let weight = nf / f32::cast_from(n_total);
    let stats_base = node_flat * n_targets as usize;
    let imp_base = (feat * n_targets) as usize;

    let mut k: u32 = tx;
    while k < n_targets {
        let sy = node_y_sums[stats_base + k as usize];
        let ssq = node_y_sum_sqs[stats_base + k as usize];
        let syl = split_y_sums_l[stats_base + k as usize];
        let ssyl = split_y_sum_sqs_l[stats_base + k as usize];
        let syr = sy - syl;
        let ssyr = ssq - ssyl;

        let mean_p = sy / nf;
        let var_p = ssq / nf - mean_p * mean_p;
        let mean_l = syl / nl;
        let var_l = ssyl / nl - mean_l * mean_l;
        let mean_r = syr / nr;
        let var_r = ssyr / nr - mean_r * mean_r;

        let vp = f32::max(var_p, 0f32);
        let vl = f32::max(var_l, 0f32);
        let vr = f32::max(var_r, 0f32);

        let reduction = f32::max(vp - (nl / nf) * vl - (nr / nf) * vr, 0f32);
        let contribution = weight * reduction;
        atomic_add_f32_bits(
            &batch_importances[imp_base + k as usize],
            contribution,
        );
        k += wg_size;
    }
}

/// Compute per-tree child ids for the next level, entirely on device. One
/// workgroup per tree, single thread. Serial scan over nodes: for each
/// internal node (valid split, `node_counts > 0`), assign consecutive child
/// ids `next_w, next_w+1` and increment. Leaf/invalid nodes emit
/// `INVALID_NODE`. Replaces an earlier host-side scatter that required
/// per-level `.read()` of `split_feature`, `node_counts`, and
/// `importance_delta`.
///
/// Under the "no per-level readback" scheme the driver dispatches every
/// level at `n_active_nodes = min(2^depth, max_active_nodes)` (an upper
/// bound); phantom nodes (no samples) end up with `split_feature = INVALID`
/// naturally from `evaluate_splits` and emit `INVALID_NODE` child ids here.
///
/// ### Params
///
/// * `split_feature` - Winning feature id per node from `evaluate_splits_*`
/// * `node_counts` - Per-node sample totals from [`merge_hist`]
/// * `left_child_id` - Output left child id per node, updated in place
/// * `right_child_id` - Output right child id per node, updated in place
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> tree_in_wave
/// * Single thread per workgroup (`UNIT_POS_X == 0`)
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn compute_child_ids(
    split_feature: &Tensor<u32>,
    node_counts: &Tensor<u32>,
    left_child_id: &mut Tensor<u32>,
    right_child_id: &mut Tensor<u32>,
    wave_size: u32,
    n_active_nodes: u32,
) {
    let tree = CUBE_POS_X;
    if tree >= wave_size {
        terminate!();
    }
    if UNIT_POS_X != 0u32 {
        terminate!();
    }
    let base = (tree * n_active_nodes) as usize;
    let mut next_w: u32 = 0u32;
    let mut node: u32 = 0u32;
    while node < n_active_nodes {
        let idx = base + node as usize;
        let feat = split_feature[idx];
        if feat != INVALID_NODE {
            let n = node_counts[idx];
            if n > 0u32 {
                left_child_id[idx] = next_w;
                right_child_id[idx] = next_w + 1u32;
                next_w += 2u32;
            } else {
                left_child_id[idx] = INVALID_NODE;
                right_child_id[idx] = INVALID_NODE;
            }
        } else {
            left_child_id[idx] = INVALID_NODE;
            right_child_id[idx] = INVALID_NODE;
        }
        node += 1u32;
    }
}

/// Seed the per-tree `sample_to_node` at level 0. Samples with multiplicity
/// 0 (not selected for this tree's subset) are marked `INVALID_NODE` so
/// subsequent kernels skip them; everything else lives at the root (node 0).
/// One thread per (sample, tree).
///
/// ### Params
///
/// * `sample_multiplicity` - Per-tree sample multiplier `[wave_size,
///   n_samples]`
/// * `sample_to_node` - Output per-tree sample assignment `[wave_size,
///   n_samples]`
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
///
/// ### Grid mapping
///
/// * `CUBE_POS_X * WORKGROUP_128 + UNIT_POS_X + CUBE_POS_Y * CUBE_COUNT_X *
///   WORKGROUP_128` -> sample index
/// * `CUBE_POS_Z` -> tree_in_wave
#[cube(launch_unchecked)]
pub fn init_sample_to_node(
    sample_multiplicity: &Tensor<u32>,
    sample_to_node: &mut Tensor<u32>,
    n_samples: u32,
    wave_size: u32,
) {
    let s = CUBE_POS_X * WORKGROUP_128 + UNIT_POS_X
        + CUBE_POS_Y * CUBE_COUNT_X * WORKGROUP_128;
    let tree = CUBE_POS_Z;
    if s >= n_samples {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }
    let idx = (tree * n_samples + s) as usize;
    if sample_multiplicity[idx] > 0u32 {
        sample_to_node[idx] = 0u32;
    } else {
        sample_to_node[idx] = INVALID_NODE;
    }
}

/////////////////////
// Launch wrappers //
/////////////////////

/// Dispatch [`sample_node_features`]. One workgroup per (node, tree).
#[allow(clippy::too_many_arguments)]
fn launch_sample_features<R: Runtime>(
    client: &ComputeClient<R>,
    tree_seeds: &GpuTensor<R, u32>,
    node_features: &GpuTensor<R, u32>,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_features: usize,
    level: u32,
) {
    unsafe {
        sample_node_features::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_active_nodes as u32, 1, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            tree_seeds.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_features as u32,
            level,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`build_hist_privatised`]. One workgroup per (slot, node, tree).
#[allow(clippy::too_many_arguments)]
fn launch_build_hist<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data: &GpuTensor<R, u32>,
    sy_offsets: &GpuTensor<R, u32>,
    sy_target_indices: &GpuTensor<R, u32>,
    sy_values: &GpuTensor<R, f32>,
    sample_to_node: &GpuTensor<R, u32>,
    sample_multiplicity: &GpuTensor<R, u32>,
    node_features: &GpuTensor<R, u32>,
    hist_counts: &GpuTensor<R, u32>,
    hist_y_sums: &GpuTensor<R, u32>,
    hist_y_sum_sqs: &GpuTensor<R, u32>,
    n_samples: usize,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) {
    unsafe {
        build_hist_privatised::launch_unchecked::<R>(
            client,
            CubeCount::Static(k_feats as u32, n_active_nodes as u32, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            sy_offsets.clone().into_tensor_arg(),
            sy_target_indices.clone().into_tensor_arg(),
            sy_values.clone().into_tensor_arg(),
            sample_to_node.clone().into_tensor_arg(),
            sample_multiplicity.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            hist_counts.clone().into_tensor_arg(),
            hist_y_sums.clone().into_tensor_arg(),
            hist_y_sum_sqs.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`merge_hist`]. One workgroup per (node, tree).
#[allow(clippy::too_many_arguments)]
fn launch_merge_hist<R: Runtime>(
    client: &ComputeClient<R>,
    hist_counts: &GpuTensor<R, u32>,
    hist_y_sums: &GpuTensor<R, u32>,
    hist_y_sum_sqs: &GpuTensor<R, u32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) {
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1));
    unsafe {
        merge_hist::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            hist_counts.clone().into_tensor_arg(),
            hist_y_sums.clone().into_tensor_arg(),
            hist_y_sum_sqs.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`prefix_sum_bins`]. One workgroup per (slot, node, tree).
#[allow(clippy::too_many_arguments)]
fn launch_prefix_sum<R: Runtime>(
    client: &ComputeClient<R>,
    hist_counts: &GpuTensor<R, u32>,
    hist_y_sums: &GpuTensor<R, u32>,
    hist_y_sum_sqs: &GpuTensor<R, u32>,
    cum_counts: &GpuTensor<R, u32>,
    cum_y_sums: &GpuTensor<R, f32>,
    cum_y_sum_sqs: &GpuTensor<R, f32>,
    slot_min_bin: &GpuTensor<R, u32>,
    slot_max_bin: &GpuTensor<R, u32>,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) {
    unsafe {
        prefix_sum_bins::launch_unchecked::<R>(
            client,
            CubeCount::Static(k_feats as u32, n_active_nodes as u32, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            hist_counts.clone().into_tensor_arg(),
            hist_y_sums.clone().into_tensor_arg(),
            hist_y_sum_sqs.clone().into_tensor_arg(),
            cum_counts.clone().into_tensor_arg(),
            cum_y_sums.clone().into_tensor_arg(),
            cum_y_sum_sqs.clone().into_tensor_arg(),
            slot_min_bin.clone().into_tensor_arg(),
            slot_max_bin.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`evaluate_splits_et`]. One workgroup per (node, tree).
///
/// Uses `WORKGROUP_32`, not `WORKGROUP_128`: at bench shape
/// `k_feats * n_thresholds` is ~31 candidates, so a 32-wide workgroup
/// saturates cleanly. [`launch_evaluate_splits_rf`] stays at `WORKGROUP_128`
/// (`k_feats * 255` is ~7905 candidates).
#[allow(clippy::too_many_arguments)]
fn launch_evaluate_splits_et<R: Runtime>(
    client: &ComputeClient<R>,
    cum_counts: &GpuTensor<R, u32>,
    cum_y_sums: &GpuTensor<R, f32>,
    cum_y_sum_sqs: &GpuTensor<R, f32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    node_features: &GpuTensor<R, u32>,
    tree_seeds: &GpuTensor<R, u32>,
    slot_min_bin: &GpuTensor<R, u32>,
    slot_max_bin: &GpuTensor<R, u32>,
    split_feature: &GpuTensor<R, u32>,
    split_threshold: &GpuTensor<R, u32>,
    split_n_left: &GpuTensor<R, u32>,
    split_y_sums_l: &GpuTensor<R, f32>,
    split_y_sum_sqs_l: &GpuTensor<R, f32>,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    n_thresholds: usize,
    min_samples_leaf: usize,
    level: u32,
) {
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1));
    unsafe {
        evaluate_splits_et::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_32),
            cum_counts.clone().into_tensor_arg(),
            cum_y_sums.clone().into_tensor_arg(),
            cum_y_sum_sqs.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            tree_seeds.clone().into_tensor_arg(),
            slot_min_bin.clone().into_tensor_arg(),
            slot_max_bin.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_threshold.clone().into_tensor_arg(),
            split_n_left.clone().into_tensor_arg(),
            split_y_sums_l.clone().into_tensor_arg(),
            split_y_sum_sqs_l.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            n_thresholds as u32,
            min_samples_leaf as u32,
            level,
            WORKGROUP_32,
        );
    }
}

/// Dispatch [`evaluate_splits_rf`]. One workgroup per (node, tree).
#[allow(clippy::too_many_arguments)]
fn launch_evaluate_splits_rf<R: Runtime>(
    client: &ComputeClient<R>,
    cum_counts: &GpuTensor<R, u32>,
    cum_y_sums: &GpuTensor<R, f32>,
    cum_y_sum_sqs: &GpuTensor<R, f32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    node_features: &GpuTensor<R, u32>,
    slot_min_bin: &GpuTensor<R, u32>,
    slot_max_bin: &GpuTensor<R, u32>,
    split_feature: &GpuTensor<R, u32>,
    split_threshold: &GpuTensor<R, u32>,
    split_n_left: &GpuTensor<R, u32>,
    split_y_sums_l: &GpuTensor<R, f32>,
    split_y_sum_sqs_l: &GpuTensor<R, f32>,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    min_samples_leaf: usize,
) {
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1));
    unsafe {
        evaluate_splits_rf::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            cum_counts.clone().into_tensor_arg(),
            cum_y_sums.clone().into_tensor_arg(),
            cum_y_sum_sqs.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            slot_min_bin.clone().into_tensor_arg(),
            slot_max_bin.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_threshold.clone().into_tensor_arg(),
            split_n_left.clone().into_tensor_arg(),
            split_y_sums_l.clone().into_tensor_arg(),
            split_y_sum_sqs_l.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            min_samples_leaf as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`reassign_samples`]. One workgroup per stripe of samples, per
/// tree.
#[allow(clippy::too_many_arguments)]
fn launch_reassign<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data: &GpuTensor<R, u32>,
    split_feature: &GpuTensor<R, u32>,
    split_threshold: &GpuTensor<R, u32>,
    left_child_id: &GpuTensor<R, u32>,
    right_child_id: &GpuTensor<R, u32>,
    sample_to_node: &GpuTensor<R, u32>,
    n_samples: usize,
    n_features: usize,
    wave_size: usize,
    n_active_nodes: usize,
) {
    let n_wgs = (n_samples as u32).div_ceil(WORKGROUP_128);
    let (gx, gy) = grid_2d(n_wgs.max(1));
    unsafe {
        reassign_samples::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_threshold.clone().into_tensor_arg(),
            left_child_id.clone().into_tensor_arg(),
            right_child_id.clone().into_tensor_arg(),
            sample_to_node.clone().into_tensor_arg(),
            n_samples as u32,
            n_features as u32,
            wave_size as u32,
            n_active_nodes as u32,
        );
    }
}

/// Dispatch [`accumulate_importance`]. One workgroup per (node, tree).
#[allow(clippy::too_many_arguments)]
fn launch_accumulate_importance<R: Runtime>(
    client: &ComputeClient<R>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    split_feature: &GpuTensor<R, u32>,
    split_n_left: &GpuTensor<R, u32>,
    split_y_sums_l: &GpuTensor<R, f32>,
    split_y_sum_sqs_l: &GpuTensor<R, f32>,
    batch_importances: &GpuTensor<R, u32>,
    wave_size: usize,
    n_active_nodes: usize,
    n_targets: usize,
    n_total: usize,
) {
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1));
    unsafe {
        accumulate_importance::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_n_left.clone().into_tensor_arg(),
            split_y_sums_l.clone().into_tensor_arg(),
            split_y_sum_sqs_l.clone().into_tensor_arg(),
            batch_importances.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            n_targets as u32,
            n_total as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`compute_child_ids`]. One workgroup per tree, single thread.
#[allow(clippy::too_many_arguments)]
fn launch_compute_child_ids<R: Runtime>(
    client: &ComputeClient<R>,
    split_feature: &GpuTensor<R, u32>,
    node_counts: &GpuTensor<R, u32>,
    left_child_id: &GpuTensor<R, u32>,
    right_child_id: &GpuTensor<R, u32>,
    wave_size: usize,
    n_active_nodes: usize,
) {
    unsafe {
        compute_child_ids::launch_unchecked::<R>(
            client,
            CubeCount::Static(wave_size as u32, 1, 1),
            CubeDim::new_1d(1),
            split_feature.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            left_child_id.clone().into_tensor_arg(),
            right_child_id.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
        );
    }
}

/// Dispatch [`init_sample_to_node`]. One thread per (sample, tree).
fn launch_init_sample_to_node<R: Runtime>(
    client: &ComputeClient<R>,
    sample_multiplicity: &GpuTensor<R, u32>,
    sample_to_node: &GpuTensor<R, u32>,
    n_samples: usize,
    wave_size: usize,
) {
    let n_wgs = (n_samples as u32).div_ceil(WORKGROUP_128);
    let (gx, gy) = grid_2d(n_wgs.max(1));
    unsafe {
        init_sample_to_node::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            sample_multiplicity.clone().into_tensor_arg(),
            sample_to_node.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
        );
    }
}

////////////////
// Wave state //
////////////////

/// All wave-sized GPU tensors. Allocated once per batch, reused across all
/// waves in the batch. Sized for `wave_size` trees and `viable_max` active
/// nodes; kernels only touch the prefix indexed by the current level's
/// `n_active_nodes`.
struct WaveState<R: Runtime> {
    // per-tree sample assignment
    sample_to_node: GpuTensor<R, u32>,
    // draw / feature selection
    node_features: GpuTensor<R, u32>,
    // histogram buffers. y_sums / y_sum_sqs store f32 bits as u32 so
    // `build_hist_privatised` can `Atomic<u32>::compare_exchange_weak` on
    // them (WGSL has no native atomic f32). Downstream kernels reinterpret
    // via `f32::from_bits`.
    hist_counts: GpuTensor<R, u32>,
    hist_y_sums: GpuTensor<R, u32>,
    hist_y_sum_sqs: GpuTensor<R, u32>,
    cum_counts: GpuTensor<R, u32>,
    cum_y_sums: GpuTensor<R, f32>,
    cum_y_sum_sqs: GpuTensor<R, f32>,
    // per-slot informative bin range. Populated by `prefix_sum_bins`, read
    // by `evaluate_splits_et` / `_rf` to avoid the per-candidate 512-read
    // min/max scan and to skip out-of-range RF thresholds.
    slot_min_bin: GpuTensor<R, u32>,
    slot_max_bin: GpuTensor<R, u32>,
    // per-node stats
    node_counts: GpuTensor<R, u32>,
    node_y_sums: GpuTensor<R, f32>,
    node_y_sum_sqs: GpuTensor<R, f32>,
    // per-node split decisions
    split_feature: GpuTensor<R, u32>,
    split_threshold: GpuTensor<R, u32>,
    split_n_left: GpuTensor<R, u32>,
    split_y_sums_l: GpuTensor<R, f32>,
    split_y_sum_sqs_l: GpuTensor<R, f32>,
    // persistent per-node child ids, populated by `compute_child_ids` on
    // device; consumed by `reassign_samples`. Replaces an earlier per-level
    // host allocation + upload.
    left_child_id: GpuTensor<R, u32>,
    right_child_id: GpuTensor<R, u32>,
    // per-batch shape
    wave_size: usize,
    max_active_nodes: usize,
    n_samples: usize,
    k_feats: usize,
    n_targets: usize,
}

impl<R: Runtime> WaveState<R> {
    /// Allocate a fresh [`WaveState`] sized for `wave_size` trees and
    /// `max_active_nodes` active nodes per level.
    fn allocate(
        client: &ComputeClient<R>,
        wave_size: usize,
        max_active_nodes: usize,
        n_samples: usize,
        k_feats: usize,
        n_targets: usize,
    ) -> Self {
        let hist_counts_len = wave_size * max_active_nodes * k_feats * N_BINS as usize;
        let hist_sums_len = hist_counts_len * n_targets;
        let node_stats_len = wave_size * max_active_nodes * n_targets;

        Self {
            sample_to_node: GpuTensor::empty(vec![wave_size * n_samples], client),
            node_features: GpuTensor::empty(
                vec![wave_size * max_active_nodes * k_feats],
                client,
            ),
            hist_counts: GpuTensor::empty(vec![hist_counts_len], client),
            hist_y_sums: GpuTensor::empty(vec![hist_sums_len], client),
            hist_y_sum_sqs: GpuTensor::empty(vec![hist_sums_len], client),
            cum_counts: GpuTensor::empty(vec![hist_counts_len], client),
            cum_y_sums: GpuTensor::empty(vec![hist_sums_len], client),
            cum_y_sum_sqs: GpuTensor::empty(vec![hist_sums_len], client),
            slot_min_bin: GpuTensor::empty(
                vec![wave_size * max_active_nodes * k_feats],
                client,
            ),
            slot_max_bin: GpuTensor::empty(
                vec![wave_size * max_active_nodes * k_feats],
                client,
            ),
            node_counts: GpuTensor::empty(vec![wave_size * max_active_nodes], client),
            node_y_sums: GpuTensor::empty(vec![node_stats_len], client),
            node_y_sum_sqs: GpuTensor::empty(vec![node_stats_len], client),
            split_feature: GpuTensor::empty(vec![wave_size * max_active_nodes], client),
            split_threshold: GpuTensor::empty(vec![wave_size * max_active_nodes], client),
            split_n_left: GpuTensor::empty(vec![wave_size * max_active_nodes], client),
            split_y_sums_l: GpuTensor::empty(vec![node_stats_len], client),
            split_y_sum_sqs_l: GpuTensor::empty(vec![node_stats_len], client),
            left_child_id: GpuTensor::empty(vec![wave_size * max_active_nodes], client),
            right_child_id: GpuTensor::empty(vec![wave_size * max_active_nodes], client),
            wave_size,
            max_active_nodes,
            n_samples,
            k_feats,
            n_targets,
        }
    }
}

////////////////////
// Sizing helpers //
////////////////////

/// Widest viable active-node count at any level, taking both the depth cap
/// and the min_samples_leaf cap into account. Undersized allocations blow
/// up as OOB writes into `node_features` / `hist_*`; oversized just wastes
/// memory. We stick with the smaller of the two caps.
///
/// ### Params
///
/// * `max_depth` - Maximum tree depth
/// * `n_samples` - Sample count
/// * `min_samples_leaf` - Minimum samples per leaf
///
/// ### Returns
///
/// Upper bound on active nodes at any single level.
fn viable_max_active_nodes(
    max_depth: usize,
    n_samples: usize,
    min_samples_leaf: usize,
) -> usize {
    let depth_cap = 1usize << max_depth.min(20);
    let leaf_cap = if min_samples_leaf == 0 {
        depth_cap
    } else {
        (n_samples / (2 * min_samples_leaf)).max(1)
    };
    depth_cap.min(leaf_cap).max(1)
}

/// Estimate the total wave-scoped VRAM in bytes for the given shape. Covers
/// only the six big allocations (hist_* and cum_*); the small per-node stats
/// and split tensors round to noise.
///
/// ### Params
///
/// * `wave_size` - Trees in wave
/// * `max_active_nodes` - Widest viable active-node count, see
///   [`viable_max_active_nodes`]
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
///
/// ### Returns
///
/// Estimated byte cost of the wave-scoped histogram and cumulative tensors.
fn wave_byte_cost(
    wave_size: usize,
    max_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) -> usize {
    let counts_slots = wave_size * max_active_nodes * k_feats * N_BINS as usize;
    let sums_slots = counts_slots * n_targets;
    // hist_counts + cum_counts (u32) + hist_y_sums + cum_y_sums +
    // hist_y_sum_sqs + cum_y_sum_sqs (f32)
    (counts_slots * 2 * 4) + (sums_slots * 4 * 4)
}

/// Pick a wave size that fits under `wave_byte_budget`, halving from the
/// default until it does.
///
/// ### Params
///
/// * `max_active_nodes` - Widest viable active-node count, see
///   [`viable_max_active_nodes`]
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_trees_target` - Trees remaining in the ensemble; caps the initial
///   wave size below the default when the ensemble is small
/// * `wave_byte_budget` - VRAM ceiling in bytes, see
///   [`ScenicGpuParams::wave_byte_budget`]
///
/// ### Returns
///
/// A wave size in `[1, DEFAULT_WAVE_SIZE]`.
///
/// ### Errors
///
/// * `InvalidArgument` if even `wave_size = 1` exceeds the budget; the
///   caller should surface this as an actionable error rather than OOM-ing
///   at allocation time.
fn pick_wave_size(
    max_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    n_trees_target: usize,
    wave_byte_budget: usize,
) -> Result<usize, BixverseErrors> {
    let mut w = DEFAULT_WAVE_SIZE.min(n_trees_target.max(1));
    while w > 1 {
        if wave_byte_cost(w, max_active_nodes, k_feats, n_targets) <= wave_byte_budget {
            return Ok(w);
        }
        w /= 2;
    }
    if wave_byte_cost(1, max_active_nodes, k_feats, n_targets) > wave_byte_budget {
        return Err(BixverseErrors::InvalidArgument(format!(
            "GPU SCENIC: even wave_size=1 exceeds the {} MB VRAM budget \
             (nodes={}, k_feats={}, n_targets={}). Reduce max_depth or \
             n_features_split.",
            wave_byte_budget / (1024 * 1024),
            max_active_nodes,
            k_feats,
            n_targets,
        )));
    }
    Ok(1)
}

/////////////////////////////
// Sparse Y upload wrapper //
/////////////////////////////

/// Device-resident sparse Y for one target batch (CSR by sample). Uploaded
/// once per batch and read by [`build_hist_privatised`] across every wave.
struct SparseYGpu<R: Runtime> {
    /// Exclusive prefix sums `[n_samples + 1]`
    offsets: GpuTensor<R, u32>,
    /// Target ids per nonzero, u8 widened to u32 `[nnz]`
    target_indices: GpuTensor<R, u32>,
    /// Values per nonzero `[nnz]`
    values: GpuTensor<R, f32>,
}

impl<R: Runtime> SparseYGpu<R> {
    /// Upload a host [`SparseYBatch`] to device. `nnz == 0` uploads a
    /// single dummy zero element so downstream tensors are never empty.
    fn upload(sy: &SparseYBatch, n_samples: usize, client: &ComputeClient<R>) -> Self {
        let offsets =
            GpuTensor::<R, u32>::from_slice(&sy.offsets, vec![n_samples + 1], client);
        let target_indices_u32: Vec<u32> = sy.target_indices.iter().map(|&i| i as u32).collect();
        let nnz = target_indices_u32.len().max(1);
        let (target_indices, values) = if target_indices_u32.is_empty() {
            (
                GpuTensor::<R, u32>::from_slice(&[0u32], vec![1], client),
                GpuTensor::<R, f32>::from_slice(&[0.0f32], vec![1], client),
            )
        } else {
            (
                GpuTensor::<R, u32>::from_slice(&target_indices_u32, vec![nnz], client),
                GpuTensor::<R, f32>::from_slice(&sy.values, vec![nnz], client),
            )
        };
        Self {
            offsets,
            target_indices,
            values,
        }
    }
}

/////////////////
// Wave driver //
/////////////////

/// Run a wave-synchronous BFS for `wave_size` trees, atomically accumulating
/// per-target importances into `batch_importances_gpu` on device (layout
/// `[n_features, n_targets]`, f32 bits stored as `u32`). Uses the
/// pre-allocated `WaveState` and the once-per-batch sparse Y upload.
///
/// No per-level `.read()` calls. `n_active_nodes` per level is the
/// upper bound `min(2^depth, max_active_nodes)`; phantom nodes (no samples)
/// naturally end up with `split_feature == INVALID` from `evaluate_splits`
/// and are no-ops in `compute_child_ids` and `accumulate_importance`. Child
/// ids for `reassign_samples` come from persistent `state.left_child_id` /
/// `state.right_child_id`, populated on device by `compute_child_ids`.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `feature_data_gpu` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `sy_gpu` - Uploaded sparse Y for this batch
/// * `state` - Pre-allocated wave-sized scratch tensors
/// * `sample_multiplicity_gpu` - Per-tree sample multiplier `[wave_size,
///   n_samples]`
/// * `tree_seeds` - Per-tree base seed `[wave_size]`
/// * `config` - Tree configuration; `random_threshold()` selects ET vs RF
/// * `batch_importances_gpu` - Output `[n_features, n_targets]` atomic
///   accumulator, f32 bits stored as `u32`
/// * `n_features` - Total feature count
/// * `max_depth` - Maximum tree depth
/// * `min_samples_leaf` - Minimum samples required on both sides of a split
/// * `n_thresholds` - Random thresholds drawn per feature slot (ET only)
/// * `n_total` - Total sample count used for the importance node-weight
///   normalisation
///
/// ### Returns
///
/// Always `Ok(())`; the `Result` return type matches the fallible
/// [`fit_multi_trees_gpu`] driver loop that calls this with `?`.
#[allow(clippy::too_many_arguments)]
fn run_wave_bfs<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data_gpu: &GpuTensor<R, u32>,
    sy_gpu: &SparseYGpu<R>,
    state: &WaveState<R>,
    sample_multiplicity_gpu: &GpuTensor<R, u32>,
    tree_seeds: &GpuTensor<R, u32>,
    config: &dyn TreeRegressorConfig,
    batch_importances_gpu: &GpuTensor<R, u32>,
    n_features: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    n_thresholds: usize,
    n_total: usize,
) -> Result<(), BixverseErrors> {
    let wave_size = state.wave_size;
    let n_samples = state.n_samples;
    let k_feats = state.k_feats;
    let n_targets = state.n_targets;
    let max_active_nodes = state.max_active_nodes;
    let use_et = config.random_threshold();

    launch_init_sample_to_node(
        client,
        sample_multiplicity_gpu,
        &state.sample_to_node,
        n_samples,
        wave_size,
    );

    for depth in 0..max_depth {
        // Upper-bound estimate: at depth d, at most 2^d active nodes exist
        // (full binary tree). Clamp to the pre-allocated buffer. Phantom
        // nodes cost cheap kernel launches and no atomics because their
        // sample_to_node never matches.
        let depth_cap = 1usize.checked_shl(depth as u32).unwrap_or(usize::MAX);
        let n_active_nodes = depth_cap.min(max_active_nodes).max(1);

        launch_sample_features(
            client,
            tree_seeds,
            &state.node_features,
            wave_size,
            n_active_nodes,
            k_feats,
            n_features,
            depth as u32,
        );

        launch_build_hist(
            client,
            feature_data_gpu,
            &sy_gpu.offsets,
            &sy_gpu.target_indices,
            &sy_gpu.values,
            &state.sample_to_node,
            sample_multiplicity_gpu,
            &state.node_features,
            &state.hist_counts,
            &state.hist_y_sums,
            &state.hist_y_sum_sqs,
            n_samples,
            wave_size,
            n_active_nodes,
            k_feats,
            n_targets,
        );

        launch_merge_hist(
            client,
            &state.hist_counts,
            &state.hist_y_sums,
            &state.hist_y_sum_sqs,
            &state.node_counts,
            &state.node_y_sums,
            &state.node_y_sum_sqs,
            wave_size,
            n_active_nodes,
            k_feats,
            n_targets,
        );

        launch_prefix_sum(
            client,
            &state.hist_counts,
            &state.hist_y_sums,
            &state.hist_y_sum_sqs,
            &state.cum_counts,
            &state.cum_y_sums,
            &state.cum_y_sum_sqs,
            &state.slot_min_bin,
            &state.slot_max_bin,
            wave_size,
            n_active_nodes,
            k_feats,
            n_targets,
        );

        let at_max_depth = depth + 1 >= max_depth;

        if use_et {
            launch_evaluate_splits_et(
                client,
                &state.cum_counts,
                &state.cum_y_sums,
                &state.cum_y_sum_sqs,
                &state.node_counts,
                &state.node_y_sums,
                &state.node_y_sum_sqs,
                &state.node_features,
                tree_seeds,
                &state.slot_min_bin,
                &state.slot_max_bin,
                &state.split_feature,
                &state.split_threshold,
                &state.split_n_left,
                &state.split_y_sums_l,
                &state.split_y_sum_sqs_l,
                wave_size,
                n_active_nodes,
                k_feats,
                n_targets,
                n_thresholds,
                min_samples_leaf,
                depth as u32,
            );
        } else {
            launch_evaluate_splits_rf(
                client,
                &state.cum_counts,
                &state.cum_y_sums,
                &state.cum_y_sum_sqs,
                &state.node_counts,
                &state.node_y_sums,
                &state.node_y_sum_sqs,
                &state.node_features,
                &state.slot_min_bin,
                &state.slot_max_bin,
                &state.split_feature,
                &state.split_threshold,
                &state.split_n_left,
                &state.split_y_sums_l,
                &state.split_y_sum_sqs_l,
                wave_size,
                n_active_nodes,
                k_feats,
                n_targets,
                min_samples_leaf,
            );
        }

        launch_accumulate_importance(
            client,
            &state.node_counts,
            &state.node_y_sums,
            &state.node_y_sum_sqs,
            &state.split_feature,
            &state.split_n_left,
            &state.split_y_sums_l,
            &state.split_y_sum_sqs_l,
            batch_importances_gpu,
            wave_size,
            n_active_nodes,
            n_targets,
            n_total,
        );

        if at_max_depth {
            continue;
        }

        launch_compute_child_ids(
            client,
            &state.split_feature,
            &state.node_counts,
            &state.left_child_id,
            &state.right_child_id,
            wave_size,
            n_active_nodes,
        );

        launch_reassign(
            client,
            feature_data_gpu,
            &state.split_feature,
            &state.split_threshold,
            &state.left_child_id,
            &state.right_child_id,
            &state.sample_to_node,
            n_samples,
            n_features,
            wave_size,
            n_active_nodes,
        );
    }

    Ok(())
}

//////////////////
// Public entry //
//////////////////

/// Multi-tree, multi-batch ExtraTrees or RandomForest regression fit on GPU.
///
/// Mirrors [`fit_multi_trees_sparse`] (`scenic.rs`), differing only in the
/// explicit `R: Runtime` and `device` parameters that a GPU entry requires.
/// Splits `targets` into batches of at most `MULTI_OUTPUT_BATCH = 64`, drives
/// the wave scheduler over all trees for each batch, and returns per-target
/// normalised importance vectors.
///
/// ### Params
///
/// * `targets` - Sparse target expression columns
/// * `feature_matrix` - Quantised u8 features, column-major
/// * `n_samples` - Total sample count
/// * `config` - Tree configuration; `config.random_threshold()` selects the
///   ExtraTrees or RandomForest split kernel and `bootstrap()` is honoured,
///   all other knobs come through `TreeRegressorConfig`
/// * `seed` - Base seed; per-tree seed is `tree_seed(seed, tree_idx)`
/// * `device` - Runtime device
/// * `params` - GPU-side runtime knobs (currently just the wave VRAM budget)
///
/// ### Returns
///
/// One importance vector per target: `result[target_idx][feature_idx]`, each
/// normalised to sum to 1.0.
///
/// ### Errors
///
/// * Propagates sparse-Y construction failures from [`SparseYBatch::from_targets`]
/// * `InvalidArgument` from [`pick_wave_size`] if even `wave_size = 1` busts
///   `params.wave_byte_budget`
/// * Propagates GPU read-back errors from the per-batch importance readback
pub fn fit_multi_trees_gpu<R: Runtime>(
    targets: &[SparseAxis<u32, f32>],
    feature_matrix: &QuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    seed: usize,
    device: R::Device,
    params: &ScenicGpuParams,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let n_features = feature_matrix.n_features;
    let n_targets_total = targets.len();
    let n_trees = config.n_trees();
    let n_features_split = resolve_n_features_split(config.n_features_split(), n_features);
    let k_feats = n_features_split.min(n_features).max(1);
    let max_depth = config.max_depth().unwrap_or(usize::MAX).min(20);
    let min_samples_leaf = config.min_samples_leaf().max(1);
    let n_thresholds = config.n_thresholds().max(1);
    let max_active_nodes = viable_max_active_nodes(max_depth, n_samples, min_samples_leaf);

    // Effective per-tree sample budget, matching CPU's fit_multi_trees_sparse
    // (scenic.rs). subsample_frac takes precedence over subsample_rate;
    // subsample_rate >= 1.0 means "use all". Result is clamped to at least
    // 2 * min_samples_leaf so a tree can always try a root split.
    let n_sub = if let Some(frac) = config.subsample_frac() {
        ((n_samples as f32 * frac).round() as usize).max(2 * min_samples_leaf)
    } else if config.subsample_rate() >= 1.0 {
        n_samples
    } else {
        ((n_samples as f32 * config.subsample_rate()).round() as usize)
            .max(2 * min_samples_leaf)
    };
    let subsample_needed = n_sub < n_samples;
    let bootstrap = config.bootstrap();

    let mut result: Vec<Vec<f32>> = vec![Vec::new(); n_targets_total];

    if n_targets_total == 0 || n_trees == 0 {
        return Ok(result);
    }

    let client = R::client(&device);

    // upload feature bins once for the whole call
    let feature_bins_u32: Vec<u32> = feature_matrix.data.iter().map(|&b| b as u32).collect();
    let feature_data_gpu = GpuTensor::<R, u32>::from_slice(
        &feature_bins_u32,
        vec![n_features * n_samples],
        &client,
    );

    for (batch_idx, chunk) in targets.chunks(MULTI_OUTPUT_BATCH).enumerate() {
        let batch_n_targets = chunk.len();
        let sparse_y = SparseYBatch::from_targets(chunk, n_samples)?;
        let sy_gpu = SparseYGpu::upload(&sparse_y, n_samples, &client);

        let wave_size = pick_wave_size(
            max_active_nodes,
            k_feats,
            batch_n_targets,
            n_trees,
            params.wave_byte_budget,
        )?;
        let state = WaveState::allocate(
            &client,
            wave_size,
            max_active_nodes,
            n_samples,
            k_feats,
            batch_n_targets,
        );

        // Interleaved [n_features, batch_n_targets] importances, accumulated
        // atomically on device across all trees in this batch. f32 bits
        // stored as u32 so `accumulate_importance` can CAS-add. Zeroed once
        // here; every wave adds to it, host reads once at end of batch.
        let batch_importances_gpu = GpuTensor::<R, u32>::from_slice(
            &vec![0u32; n_features * batch_n_targets],
            vec![n_features * batch_n_targets],
            &client,
        );

        let mut tree_idx = 0usize;
        while tree_idx < n_trees {
            let this_wave = std::cmp::min(wave_size, n_trees - tree_idx);

            // per-tree seeds for this wave
            let seeds_host: Vec<u32> = (tree_idx..tree_idx + this_wave)
                .map(|t| tree_seed(seed, t) as u32)
                .collect();
            let tree_seeds_gpu = GpuTensor::<R, u32>::from_slice(
                &seeds_host,
                vec![this_wave],
                &client,
            );

            // Per-tree sample multiplicity for this wave. For bootstrap we
            // draw n_sub samples with replacement (mult in {0, 1, 2, ...});
            // for non-bootstrap subsample we do Fisher-Yates and take the
            // first n_sub (mult in {0, 1}); for the no-subsample path all
            // samples get mult 1. RNG is seeded off `tree_seed(seed, t)`
            // exactly like the CPU path.
            let mut mult_host = vec![0u32; this_wave * n_samples];
            if subsample_needed {
                for w in 0..this_wave {
                    let mut rng =
                        SmallRng::seed_from_u64(tree_seed(seed, tree_idx + w));
                    let row_base = w * n_samples;
                    if bootstrap {
                        for _ in 0..n_sub {
                            let idx = rng.random_range(0..n_samples);
                            mult_host[row_base + idx] += 1;
                        }
                    } else {
                        // Fisher-Yates via init_and_split; buf[..n_sub] holds
                        // the n_sub chosen indices without replacement.
                        let mut buf: Vec<u32> = vec![0u32; n_samples];
                        init_and_split(&mut buf, n_samples, n_sub, &mut rng);
                        for i in 0..n_sub {
                            let idx = buf[i] as usize;
                            mult_host[row_base + idx] = 1;
                        }
                    }
                }
            } else {
                mult_host.fill(1);
            }
            let mult_gpu = GpuTensor::<R, u32>::from_slice(
                &mult_host,
                vec![this_wave * n_samples],
                &client,
            );

            // If the terminal wave is smaller than `wave_size`, we reuse the
            // over-provisioned `state` and only run the first `this_wave`
            // slots of it. Kernels gate on the `wave_size` param passed to
            // each launch, so the tail slots stay untouched.
            let effective_state = if this_wave == wave_size {
                None
            } else {
                Some(WaveState::allocate(
                    &client,
                    this_wave,
                    max_active_nodes,
                    n_samples,
                    k_feats,
                    batch_n_targets,
                ))
            };
            let use_state = effective_state.as_ref().unwrap_or(&state);

            run_wave_bfs(
                &client,
                &feature_data_gpu,
                &sy_gpu,
                use_state,
                &mult_gpu,
                &tree_seeds_gpu,
                config,
                &batch_importances_gpu,
                n_features,
                max_depth,
                min_samples_leaf,
                n_thresholds,
                n_sub,
            )?;

            tree_idx += this_wave;
        }

        // Single readback per batch. Bits stored as u32, reinterpreted as f32.
        let importances_bits = batch_importances_gpu.clone().read(&client)?;
        let batch_importances: Vec<f32> =
            importances_bits.iter().map(|&b| f32::from_bits(b)).collect();

        for (k, target_offset) in (0..batch_n_targets)
            .zip(batch_idx * MULTI_OUTPUT_BATCH..batch_idx * MULTI_OUTPUT_BATCH + batch_n_targets)
        {
            let mut per_target = vec![0.0f32; n_features];
            for f in 0..n_features {
                per_target[f] = batch_importances[f * batch_n_targets + k];
            }
            normalise_importances(&mut per_target);
            result[target_offset] = per_target;
        }
    }

    Ok(result)
}

/// Backward-compatible single-tree entry used by the early single-tree
/// sanity tests. Thin wrapper around [`fit_multi_trees_gpu`] with
/// `n_trees = 1`.
///
/// The signature takes a `&SparseYBatch` (pre-built) and an
/// `&ExtraTreesConfig` (concrete) rather than `&[SparseAxis]` and
/// `&dyn TreeRegressorConfig`. Callers who already hold a `SparseYBatch` can
/// keep using this; new callers should prefer `fit_multi_trees_gpu`.
///
/// ### Params
///
/// * `sparse_y` - Pre-built sparse Y batch (CSR by sample)
/// * `feature_matrix` - Quantised u8 features, column-major
/// * `n_samples` - Total sample count
/// * `config` - ExtraTrees configuration; `n_trees` is forced to 1
/// * `seed` - Base seed
/// * `device` - Runtime device
/// * `params` - GPU-side runtime knobs (currently just the wave VRAM budget)
///
/// ### Returns
///
/// One importance vector per target, see [`fit_multi_trees_gpu`].
///
/// ### Errors
///
/// * Propagates errors from [`fit_multi_trees_gpu`].
pub fn fit_extra_trees_gpu_single<R: Runtime>(
    sparse_y: &SparseYBatch,
    feature_matrix: &QuantisedStore,
    n_samples: usize,
    config: &ExtraTreesConfig,
    seed: usize,
    device: R::Device,
    params: &ScenicGpuParams,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    // Rebuild the target list from `sparse_y` so we can go through the
    // multi-batch driver. This is only used by tests -- the loss of the
    // sparse-y-direct fast path is a rounding-error cost on tiny inputs.
    let n_targets = sparse_y_infer_n_targets(sparse_y);
    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); n_targets];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); n_targets];
    for cell in 0..n_samples {
        let s = sparse_y.offsets[cell] as usize;
        let e = sparse_y.offsets[cell + 1] as usize;
        for j in s..e {
            let t = sparse_y.target_indices[j] as usize;
            let v = sparse_y.values[j];
            cols_indices[t].push(cell);
            cols_values[t].push(v);
        }
    }
    let axes: Vec<SparseAxis<u32, f32>> = cols_indices
        .into_iter()
        .zip(cols_values)
        .map(|(idx, vs)| SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), n_samples))
        .collect();

    // Force `n_trees = 1` while preserving the caller's other knobs.
    let mut cfg = config.clone();
    cfg.n_trees = 1;
    fit_multi_trees_gpu::<R>(&axes, feature_matrix, n_samples, &cfg, seed, device, params)
}

/////////////
// Helpers //
/////////////

/// Infer the number of targets by scanning the sparse Y target index array.
fn sparse_y_infer_n_targets(sy: &SparseYBatch) -> usize {
    let mut max = 0u32;
    for &t in &sy.target_indices {
        if t as u32 > max {
            max = t as u32;
        }
    }
    (max as usize) + 1
}
