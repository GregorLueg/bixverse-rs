//! GPU implementation of the multi-output tree regression from
//! `sc_analysis/scenic.rs`: wave-scheduled, multi-tree, multi-batch
//! ExtraTrees and RandomForest, dispatched per level via
//! [`evaluate_splits_et`] (random threshold) or [`evaluate_splits_rf`]
//! (exhaustive threshold) on `config.random_threshold()`. Six-kernel
//! level-synchronous BFS pipeline, all kernels running at full
//! `WORKGROUP_128` width using the atomic-free segmented pattern from
//! `gpu/ml/k_means_gpu.rs::segmented_centroid_update` and the SMEM tree
//! reduction from `gpu/sc_gpu/kernels/harmony_kernels.rs::objective_partials`.

#![allow(missing_docs)]
#![cfg(all(feature = "single-cell", feature = "gpu"))]

use ann_search_rs::gpu::grid_2d;
use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;
use faer::Mat;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;
use thousands::Separable;

use crate::gpu::{WORKGROUP_32, WORKGROUP_128};
use crate::prelude::*;
use crate::single_cell::mc_analysis::scenic_metacells::{
    batch_genes_in_memory, build_tf_quantised_store, extract_target_column,
};
use crate::single_cell::sc_analysis::scenic::*;
use crate::single_cell::sc_utils::utils_tree::*;

////////////
// Params //
////////////

/// Parameters for the GPU multi-tree SCENIC driver.
pub struct ScenicGpuParams {
    /// VRAM ceiling (bytes) for the per-wave histogram + cumulative tensors.
    /// The wave scheduler halves the wave size from 8 until its byte cost
    /// fits under this budget; an error is returned only when a single-tree
    /// wave still busts it.
    ///
    /// Default: 4 GB. Shrink on 8 GB adapters that host other workloads;
    /// raise on 16 GB+ adapters to keep the wave at 8.
    pub wave_byte_budget: usize,
}

/// Default implementation.
impl Default for ScenicGpuParams {
    fn default() -> Self {
        Self {
            wave_byte_budget: 4 * 1024 * 1024 * 1024,
        }
    }
}

////////////
// Consts //
////////////

/// Number of quantisation bins.
const N_BINS: u32 = 256;

/// Default wave size.
const DEFAULT_WAVE_SIZE: usize = 8;

/// Sentinel for "no node" / "no valid split" / "no child" in the u32-typed
/// device buffers (`split_feature`, `sample_to_node`, `left_child_id`,
/// `right_child_id`).
const INVALID_NODE: u32 = u32::MAX;

/////////////
// Kernels //
/////////////

/// Draw `k_feats` feature ids per (tree, node) into `node_features`.
///
/// One workgroup per (node, tree), `WORKGROUP_128` wide. Thread `tx` owns slots
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
///
/// ### Returns
///
/// Feature ids written into
/// `node_features[tree * n_active_nodes * k_feats + ...]` in place.
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

/// CAS-loop atomic f32 add on a `Atomic<u32>` slot holding f32 bits.
///
/// WGSL has no native atomic f32 op; we bit-reinterpret and retry until the
/// observed old value matches our compare value.
///
/// ### Workgroup
///
/// Inlined into the calling kernel; no dedicated workgroup.
///
/// ### Params
///
/// * `ptr` - Atomic u32 slot holding the current f32 value in bit-reinterpreted
///   form
/// * `delta` - f32 addend to apply
///
/// ### Returns
///
/// `()` — the updated f32 value is stored back into `ptr` in place.
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
/// atomic accumulation into a per-workgroup private histogram slice.
///
/// One workgroup per (slot, node, tree), `WORKGROUP_128` wide. Each thread
/// strides over samples `s = tx, tx+wg, ...`; every active sample bumps its
/// owning bin. Since each workgroup owns its own histogram slice
/// `[wave, node, slot, bin, target]`, atomics contend only *within* a
/// workgroup, never across workgroups.
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
/// * `hist_y_sum_sqs` - Output Y sum-of-squares as atomic u32 (f32 bits, same
///   layout)
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
///
/// ### Returns
///
/// Histogram counts and Y-stat accumulations written into `hist_counts`,
/// `hist_y_sums`, and `hist_y_sum_sqs` in place.
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

    let count_base = (((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) as usize;
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
///
/// ### Returns
///
/// Per-node totals written into `node_counts`, `node_y_sums`, and
/// `node_y_sum_sqs` in place.
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

/// Inclusive prefix sum over 256 bins per (tree, node, slot).
///
/// One workgroug per (slot, node, tree), `WORKGROUP_128` wide. Thread 0 runs
/// the counts scan and, in the same pass, computes the per-slot informative bin
/// range `[min_bin, max_bin]` (first and last bins with nonzero counts) into
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
///
/// ### Returns
///
/// Inclusive prefix sums written into `cum_counts`, `cum_y_sums`, and
/// `cum_y_sum_sqs`; per-slot informative bin ranges written into
/// `slot_min_bin` and `slot_max_bin`.
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
            cum_y_sums[curr_s] = cum_y_sums[prev_s] + f32::from_bits(hist_y_sums[curr_s]);
            cum_y_sum_sqs[curr_s] = cum_y_sum_sqs[prev_s] + f32::from_bits(hist_y_sum_sqs[curr_s]);
            b += 1u32;
        }
        k += wg_size;
    }
}

/// Cheap on-device hash for feature/threshold selection. Multiplies wrap in
/// shader arithmetic (WGSL and SPIR-V both define `*` on `u32` mod 2^32).
///
/// ### Workgroup
///
/// Inlined into the calling kernel; no dedicated workgroup.
///
/// ### Params
///
/// * `x` - Input u32 seed value to mix
///
/// ### Returns
///
/// A pseudorandom u32 derived from `x` via three xor-shift-multiply rounds.
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

/// Evaluate ExtraTrees random-threshold splits.
///
/// One workgroup per (node, tree), `WORKGROUP_128` wide. Thread `tx` handles
/// candidates
/// `tx, tx+wg, ...` (candidate `c` decodes as `slot = c / n_thresholds`,
/// `thr_idx = c % n_thresholds`), keeps its running best in registers, then
/// participates in a manually-unrolled SMEM tree argmax (128 -> 64 -> ...
/// -> 1). Thread 0 writes the winning split; all threads then fan out again on
/// the target dim to copy the winning bin's left-child Y stats.
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
///
/// ### Returns
///
/// Winning split written into `split_feature`, `split_threshold`, and
/// `split_n_left`; left-child Y stats into `split_y_sums_l` and
/// `split_y_sum_sqs_l`. Nodes with no valid split get
/// `split_feature = u32::MAX`.
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

        let count_base = (((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) as usize;
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

    // argmax reduction (32-wide, 5 halving stages 16 -> 8 -> 4 -> 2 -> 1)
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
            let feat =
                node_features[((tree * n_active_nodes + node) * k_feats + winner_slot) as usize];
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

/// Argmax reduction decision: returns 1 iff mate slot should overwrite current
/// slot. Ties resolve to the lower-indexed thread (strict `>`).
///
/// ### Params
///
/// * `cur_valid` - 1 if the current slot holds a valid candidate, 0 otherwise
/// * `cur_score` - Variance-reduction score for the current slot
/// * `mate_valid` - 1 if the mate slot holds a valid candidate, 0 otherwise
/// * `mate_score` - Variance-reduction score for the mate slot
///
/// ### Returns
///
/// 1 if the mate slot should replace the current slot, 0 to keep the current.
///
/// ### Workgroup
///
/// Inlined into the calling kernel's SMEM reduction; no dedicated workgroup.
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

/// Evaluate RandomForest exhaustive-threshold splits.
///
/// One workgroup per (node, tree), `WORKGROUP_128` wide. Candidate space is the
/// flattened `(slot x threshold)` grid with 255 thresholds per slot
/// (bins 0..254 -- bin 255 as a threshold always sends every sample left, gets
/// rejected by min_samples_leaf). Thresholds outside `[min_bin, max_bin)` for a
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
///
/// ### Returns
///
/// Winning split written into `split_feature`, `split_threshold`, and
/// `split_n_left`; left-child Y stats into `split_y_sums_l` and
/// `split_y_sum_sqs_l`. Nodes with no valid split get `split_feature = u32::MAX`.
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

        let slot_flat = ((tree * n_active_nodes + node) * k_feats + slot) as usize;
        let min_bin = slot_min_bin[slot_flat];
        let max_bin = slot_max_bin[slot_flat];
        let in_range = max_bin > min_bin && thr >= min_bin && thr < max_bin;

        if in_range {
            let count_base = (((tree * n_active_nodes + node) * k_feats + slot) * N_BINS) as usize;
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

    // argmax reduction (WORKGROUP_128: RF's larger candidate pool saturates)
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
            let feat =
                node_features[((tree * n_active_nodes + node) * k_feats + winner_slot) as usize];
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
///
/// ### Returns
///
/// `sample_to_node` updated in place to each sample's child node id for the
/// next level; samples at leaves or with an invalid parent are set to
/// `INVALID_NODE`.
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
    let s = CUBE_POS_X * WORKGROUP_128 + UNIT_POS_X + CUBE_POS_Y * CUBE_COUNT_X * WORKGROUP_128;
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
/// it directly into the per-batch importance accumulator.
///
/// One workgroup per (node, tree), `WORKGROUP_128` wide, thread `tx` owns
/// targets `tx, tx+wg, ...`. Contribution is atomically added into
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
///
/// ### Returns
///
/// `batch_importances` atomically updated in place; each valid split's
/// weighted variance reduction is CAS-added to the feature's importance slot.
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
        atomic_add_f32_bits(&batch_importances[imp_base + k as usize], contribution);
        k += wg_size;
    }
}

/// Compute per-tree child ids for the next level, entirely on device.
///
/// One workgroup per tree, single thread. Serial scan over nodes: for each
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
///
/// ### Returns
///
/// `left_child_id` and `right_child_id` written for each internal node;
/// leaves and phantom nodes (no samples) emit `INVALID_NODE`.
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

/// Seed the per-tree `sample_to_node` at level 0.
///
/// Samples with multiplicity 0 (not selected for this tree's subset) are marked
/// `INVALID_NODE` so subsequent kernels skip them; everything else lives at the
/// root (node 0). One thread per (sample, tree).
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
///
/// ### Returns
///
/// `sample_to_node` written to 0 (root) for samples with `multiplicity > 0`,
/// and `INVALID_NODE` for unselected samples.
#[cube(launch_unchecked)]
pub fn init_sample_to_node(
    sample_multiplicity: &Tensor<u32>,
    sample_to_node: &mut Tensor<u32>,
    n_samples: u32,
    wave_size: u32,
) {
    let s = CUBE_POS_X * WORKGROUP_128 + UNIT_POS_X + CUBE_POS_Y * CUBE_COUNT_X * WORKGROUP_128;
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

/// Dispatch [`sample_node_features`] over `(n_active_nodes, 1, wave_size)`
/// workgroups.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup, each owning
/// feature slots `tx, tx+128, ...`.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `tree_seeds` - Per-tree base seed `[wave_size]`
/// * `node_features` - Output feature ids `[wave_size, n_active_nodes, k_feats]`
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
/// * `k_feats` - Features per node
/// * `n_features` - Total feature count (draw range)
/// * `level` - Depth being processed
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`build_hist_privatised`] over `(k_feats, n_active_nodes, wave_size)`
/// workgroups.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup, each striding
/// over samples `tx, tx+128, ...`.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `sy_offsets` - Sparse Y offsets `[n_samples + 1]`
/// * `sy_target_indices` - Sparse Y target ids `[nnz]` (u8 as u32)
/// * `sy_values` - Sparse Y values `[nnz]`
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`
/// * `sample_multiplicity` - Per-tree sample multiplier `[wave_size, n_samples]`
/// * `node_features` - Selected feature ids `[wave_size, n_active_nodes, k_feats]`
/// * `hist_counts` - Output counts `[wave_size, n_active_nodes, k_feats, N_BINS]`
/// * `hist_y_sums` - Output Y sums as u32 f32 bits, same layout with trailing `n_targets`
/// * `hist_y_sum_sqs` - Output Y sum-of-squares, same layout as `hist_y_sums`
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`merge_hist`] over `(gx, gy, wave_size)` workgroups, where
/// `(gx, gy)` is the `grid_2d` decomposition of `n_active_nodes`.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup; thread 0
/// scans counts, threads `tx` own targets `tx, tx+128, ...` for y-sum totals.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `hist_counts` - Per-slot histogram counts
///   `[wave_size, n_active_nodes, k_feats, N_BINS]`
/// * `hist_y_sums` - Per-slot Y sums, u32 f32 bits, same layout with trailing
///   `n_targets`
/// * `hist_y_sum_sqs` - Per-slot Y sum-of-squares, same layout as `hist_y_sums`
/// * `node_counts` - Output per-node sample totals
///   `[wave_size, n_active_nodes]`
/// * `node_y_sums` - Output per-node Y sums
///   `[wave_size, n_active_nodes, n_targets]`
/// * `node_y_sum_sqs` - Output per-node Y sum-of-squares, same layout as
///   `node_y_sums`
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`prefix_sum_bins`] over `(k_feats, n_active_nodes, wave_size)`
/// workgroups.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup; thread 0
/// runs the fused count prefix sum and bin-range scan, threads `tx` own
/// targets `tx, tx+128, ...` for the y-sum prefix sums.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `hist_counts` - Per-slot histogram counts
///   `[wave_size, n_active_nodes, k_feats, N_BINS]`
/// * `hist_y_sums` - Per-slot Y sums, u32 f32 bits, same layout with trailing
///   `n_targets`
/// * `hist_y_sum_sqs` - Per-slot Y sum-of-squares, same layout as
///   `hist_y_sums`
/// * `cum_counts` - Output inclusive prefix-sum counts, same layout as
///   `hist_counts`
/// * `cum_y_sums` - Output inclusive prefix-sum Y sums, native f32
/// * `cum_y_sum_sqs` - Output inclusive prefix-sum Y sum-of-squares, native f32
/// * `slot_min_bin` - Output first informative bin per slot
///   `[wave_size, n_active_nodes, k_feats]`
/// * `slot_max_bin` - Output last informative bin per slot, same layout as
///   `slot_min_bin`
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`evaluate_splits_et`] over `(gx, gy, wave_size)` workgroups,
/// where `(gx, gy)` is the `grid_2d` decomposition of `n_active_nodes`.
///
/// Uses `WORKGROUP_32`, not `WORKGROUP_128`: at bench shape
/// `k_feats * n_thresholds` is ~31 candidates, so a 32-wide workgroup
/// saturates cleanly. [`launch_evaluate_splits_rf`] stays at `WORKGROUP_128`
/// (`k_feats * 255` is ~7905 candidates).
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_32)` — 32 threads per workgroup, 5-stage SMEM
/// argmax (16→8→4→2→1).
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `cum_counts` - Inclusive prefix-sum counts from [`launch_prefix_sum`]
/// * `cum_y_sums` - Inclusive prefix-sum Y sums from [`launch_prefix_sum`]
/// * `cum_y_sum_sqs` - Inclusive prefix-sum Y sum-of-squares from
///   [`launch_prefix_sum`]
/// * `node_counts` - Per-node sample totals from [`launch_merge_hist`]
/// * `node_y_sums` - Per-node Y sums from [`launch_merge_hist`]
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares from [`launch_merge_hist`]
/// * `node_features` - Selected feature ids from [`launch_sample_features`]
/// * `tree_seeds` - Per-tree base seed `[wave_size]`
/// * `slot_min_bin` - First informative bin per slot from [`launch_prefix_sum`]
/// * `slot_max_bin` - Last informative bin per slot from [`launch_prefix_sum`]
/// * `split_feature` - Output winning feature id per node, or `u32::MAX` if no
///   valid split
/// * `split_threshold` - Output winning threshold bin per node
/// * `split_n_left` - Output left-child sample count per node
/// * `split_y_sums_l` - Output left-child Y sums per (node, target)
/// * `split_y_sum_sqs_l` - Output left-child Y sum-of-squares per (node, target)
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_thresholds` - Random thresholds drawn per feature slot
/// * `min_samples_leaf` - Minimum samples required on both sides of a split
/// * `level` - Depth being processed
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`evaluate_splits_rf`] over `(gx, gy, wave_size)` workgroups,
/// where `(gx, gy)` is the `grid_2d` decomposition of `n_active_nodes`.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup, 7-stage SMEM
/// argmax (64→32→16→8→4→2→1) matching the larger `k_feats * 255` candidate
/// space.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `cum_counts` - Inclusive prefix-sum counts from [`launch_prefix_sum`]
/// * `cum_y_sums` - Inclusive prefix-sum Y sums from [`launch_prefix_sum`]
/// * `cum_y_sum_sqs` - Inclusive prefix-sum Y sum-of-squares from
///   [`launch_prefix_sum`]
/// * `node_counts` - Per-node sample totals from [`launch_merge_hist`]
/// * `node_y_sums` - Per-node Y sums from [`launch_merge_hist`]
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares from [`launch_merge_hist`]
/// * `node_features` - Selected feature ids from [`launch_sample_features`]
/// * `slot_min_bin` - First informative bin per slot from [`launch_prefix_sum`]
/// * `slot_max_bin` - Last informative bin per slot from [`launch_prefix_sum`]
/// * `split_feature` - Output winning feature id per node, or `u32::MAX` if no
///   valid split
/// * `split_threshold` - Output winning threshold bin per node
/// * `split_n_left` - Output left-child sample count per node
/// * `split_y_sums_l` - Output left-child Y sums per (node, target)
/// * `split_y_sum_sqs_l` - Output left-child Y sum-of-squares per (node,
///   target)
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `min_samples_leaf` - Minimum samples required on both sides of a split
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`reassign_samples`] over `(gx, gy, wave_size)` workgroups, where
/// `(gx, gy)` is the `grid_2d` decomposition of the sample-block count
/// `ceil(n_samples / WORKGROUP_128)`.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup, one thread
/// per sample.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `split_feature` - Winning feature id per node from `evaluate_splits_*`
/// * `split_threshold` - Winning threshold bin per node
/// * `left_child_id` - Left child id per node from [`launch_compute_child_ids`]
/// * `right_child_id` - Right child id per node from
///   [`launch_compute_child_ids`]
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`,
///   updated in place
/// * `n_samples` - Number of samples
/// * `n_features` - Total feature count
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`accumulate_importance`] over `(gx, gy, wave_size)` workgroups,
/// where `(gx, gy)` is the `grid_2d` decomposition of `n_active_nodes`.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup; thread `tx`
/// owns targets `tx, tx+128, ...` for the per-target variance-reduction
/// computation and CAS-loop atomic add into `batch_importances`.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `node_counts` - Per-node sample totals from [`launch_merge_hist`]
/// * `node_y_sums` - Per-node Y sums from [`launch_merge_hist`]
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares from [`launch_merge_hist`]
/// * `split_feature` - Winning feature id per node from `evaluate_splits_*`
/// * `split_n_left` - Left-child sample count per node
/// * `split_y_sums_l` - Left-child Y sums per (node, target)
/// * `split_y_sum_sqs_l` - Left-child Y sum-of-squares per (node, target)
/// * `batch_importances` - Output `[n_features, n_targets]` atomic accumulator,
///   f32 bits stored as u32
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
/// * `n_targets` - Targets in batch
/// * `n_total` - Total sample count for node-weight normalisation
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`compute_child_ids`] over `(wave_size, 1, 1)` workgroups, one
/// thread per workgroup.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(1)` — single thread per workgroup; the serial node scan is
/// cheap and avoids needing an atomic counter for the next-child-id cursor.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `split_feature` - Winning feature id per node from `evaluate_splits_*`
/// * `node_counts` - Per-node sample totals from [`launch_merge_hist`]
/// * `left_child_id` - Output left child id per node `[wave_size, n_active_nodes]`
/// * `right_child_id` - Output right child id per node, same layout
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at this level
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// Dispatch [`init_sample_to_node`] over `(gx, gy, wave_size)` workgroups,
/// where `(gx, gy)` is the `grid_2d` decomposition of the sample-block count
/// `ceil(n_samples / WORKGROUP_128)`.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup, one thread
/// per sample.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `sample_multiplicity` - Per-tree sample multiplier `[wave_size, n_samples]`
/// * `sample_to_node` - Output per-tree sample assignment `[wave_size, n_samples]`
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
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

/// All wave-sized GPU tensors.
///
/// Allocated once per batch, reused across all waves in the batch. Sized for
/// `wave_size` trees and `viable_max` active nodes; kernels only touch the
/// prefix indexed by the current level's `n_active_nodes`.
struct WaveState<R: Runtime> {
    /// Per-tree sample assignment `[wave_size, n_samples]`. 0 = root;
    /// `INVALID_NODE` for unselected or leaf-reached samples.
    sample_to_node: GpuTensor<R, u32>,
    /// Selected feature ids per (tree, node, slot)
    /// `[wave_size, max_active_nodes, k_feats]`.
    node_features: GpuTensor<R, u32>,
    /// Raw histogram counts `[wave_size, max_active_nodes, k_feats, N_BINS]`.
    hist_counts: GpuTensor<R, u32>,
    /// Raw histogram Y sums, f32 bits stored as u32 for atomic CAS.
    /// Layout `[wave_size, max_active_nodes, k_feats, N_BINS, n_targets]`.
    /// WGSL has no native atomic f32; `build_hist_privatised` uses CAS-loop
    /// via `Atomic<u32>`. Downstream kernels reinterpret via `f32::from_bits`.
    hist_y_sums: GpuTensor<R, u32>,
    /// Raw histogram Y sum-of-squares, same layout and encoding as
    /// `hist_y_sums`.
    hist_y_sum_sqs: GpuTensor<R, u32>,
    /// Inclusive prefix-sum counts from `prefix_sum_bins`, same layout as
    /// `hist_counts`.
    cum_counts: GpuTensor<R, u32>,
    /// Inclusive prefix-sum Y sums from `prefix_sum_bins`, native f32.
    /// Layout `[wave_size, max_active_nodes, k_feats, N_BINS, n_targets]`.
    cum_y_sums: GpuTensor<R, f32>,
    /// Inclusive prefix-sum Y sum-of-squares from `prefix_sum_bins`, same
    /// layout as `cum_y_sums`.
    cum_y_sum_sqs: GpuTensor<R, f32>,
    /// First informative bin per slot `[wave_size, max_active_nodes, k_feats]`.
    /// Populated by `prefix_sum_bins`; read by `evaluate_splits_et` / `_rf` to
    /// skip empty out-of-range thresholds without a per-candidate 256-bin
    /// rescan.
    slot_min_bin: GpuTensor<R, u32>,
    /// Last informative bin per slot, same layout as `slot_min_bin`.
    slot_max_bin: GpuTensor<R, u32>,
    /// Per-node sample totals `[wave_size, max_active_nodes]` from
    /// `merge_hist`.
    node_counts: GpuTensor<R, u32>,
    /// Per-node Y sums `[wave_size, max_active_nodes, n_targets]` from
    /// `merge_hist`.
    node_y_sums: GpuTensor<R, f32>,
    /// Per-node Y sum-of-squares from `merge_hist`, same layout as
    /// `node_y_sums`.
    node_y_sum_sqs: GpuTensor<R, f32>,
    /// Winning feature id per node `[wave_size, max_active_nodes]`;
    /// `INVALID_NODE`
    /// for leaves and phantom nodes.
    split_feature: GpuTensor<R, u32>,
    /// Winning threshold bin per node, same layout as `split_feature`.
    split_threshold: GpuTensor<R, u32>,
    /// Left-child sample count per node, same layout as `split_feature`.
    split_n_left: GpuTensor<R, u32>,
    /// Left-child Y sums per (node, target)
    /// `[wave_size, max_active_nodes, n_targets]`.
    split_y_sums_l: GpuTensor<R, f32>,
    /// Left-child Y sum-of-squares, same layout as `split_y_sums_l`.
    split_y_sum_sqs_l: GpuTensor<R, f32>,
    /// Left child id per node `[wave_size, max_active_nodes]`. Populated on
    /// device by `compute_child_ids`; consumed by `reassign_samples`.
    /// `INVALID_NODE` for leaves and phantom nodes.
    left_child_id: GpuTensor<R, u32>,
    /// Right child id per node, same layout as `left_child_id`.
    right_child_id: GpuTensor<R, u32>,
    /// Trees in the current wave.
    wave_size: usize,
    /// Upper bound on active nodes at any level; controls the buffer prefix
    /// kernels are allowed to touch.
    max_active_nodes: usize,
    /// Number of samples.
    n_samples: usize,
    /// Features drawn per node.
    k_feats: usize,
    /// Targets in the current batch.
    n_targets: usize,
}

impl<R: Runtime> WaveState<R> {
    /// Allocate a fresh [`WaveState`] sized for `wave_size` trees and
    /// `max_active_nodes` active nodes per level. Buffers are uninitialised;
    /// kernels zero the slices they own before use.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client for buffer allocation
    /// * `wave_size` - Trees in wave
    /// * `max_active_nodes` - Upper bound on active nodes at any level; see
    ///   [`viable_max_active_nodes`]
    /// * `n_samples` - Number of samples
    /// * `k_feats` - Features drawn per node
    /// * `n_targets` - Targets in the current batch
    ///
    /// ### Returns
    ///
    /// A `WaveState` with all GPU tensors allocated and uninitialised.
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
            node_features: GpuTensor::empty(vec![wave_size * max_active_nodes * k_feats], client),
            hist_counts: GpuTensor::empty(vec![hist_counts_len], client),
            hist_y_sums: GpuTensor::empty(vec![hist_sums_len], client),
            hist_y_sum_sqs: GpuTensor::empty(vec![hist_sums_len], client),
            cum_counts: GpuTensor::empty(vec![hist_counts_len], client),
            cum_y_sums: GpuTensor::empty(vec![hist_sums_len], client),
            cum_y_sum_sqs: GpuTensor::empty(vec![hist_sums_len], client),
            slot_min_bin: GpuTensor::empty(vec![wave_size * max_active_nodes * k_feats], client),
            slot_max_bin: GpuTensor::empty(vec![wave_size * max_active_nodes * k_feats], client),
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
/// and the min_samples_leaf cap into account.
///
/// Undersized allocations blow up as OOB writes into
/// `node_features` / `hist_*`; oversized just wastes memory. We stick with the
/// smaller of the two caps.
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
pub fn viable_max_active_nodes(
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

/// Estimate the total wave-scoped VRAM in bytes for the given shape.
///
/// Covers only the six big allocations (hist_* and cum_*); the small per-node
/// stats and split tensors round to noise.
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
pub fn wave_byte_cost(
    wave_size: usize,
    max_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) -> usize {
    let counts_slots = wave_size * max_active_nodes * k_feats * N_BINS as usize;
    let sums_slots = counts_slots * n_targets;

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
pub fn pick_wave_size(
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
    ///
    /// ### Params
    ///
    /// * `sy` - Host sparse Y batch to upload (CSR by sample)
    /// * `n_samples` - Number of samples; sets the `offsets` tensor length to
    ///   `n_samples + 1`
    /// * `client` - CubeCL compute client for buffer allocation and upload
    ///
    /// ### Returns
    ///
    /// A `SparseYGpu` with all three tensors resident on device.
    fn upload(sy: &SparseYBatch, n_samples: usize, client: &ComputeClient<R>) -> Self {
        let offsets = GpuTensor::<R, u32>::from_slice(&sy.offsets, vec![n_samples + 1], client);
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

/// Fit a sequence of pre-sliced gene batches on GPU, sharing one feature
/// upload across every batch and deferring importance readbacks to the end
/// of the call.
///
/// This is the workhorse the top-level SCENIC GPU drivers call once with
/// their full list of cluster-aware batches. Each batch runs its own wave
/// loop, but the ~`n_features * n_samples` u32 feature tensor is uploaded
/// exactly once and every batch's importance tensor stays resident on
/// device until every wave has been submitted. Only then do we walk the
/// stashed handles and `.read()` them one by one, so host prep for later
/// batches overlaps with the GPU chewing on earlier ones.
///
/// [`fit_multi_trees_gpu`] wraps this with a single-batch chunking pass for
/// backward compatibility with existing tests and the bench.
///
/// ### Params
///
/// * `batches` - One slice of `SparseAxis` targets per batch. Batch sizes
///   need not be uniform; each is fed to the wave scheduler independently.
/// * `batch_seeds` - Per-batch base seed. Must have the same length as
///   `batches`. Tree seeds within a wave use `tree_seed(batch_seed[i], t)`.
/// * `feature_matrix` - Quantised u8 features, column-major. Uploaded once.
/// * `n_samples` - Total sample count.
/// * `config` - Tree configuration; `config.random_threshold()` selects the
///   ExtraTrees or RandomForest split kernel and `bootstrap()` is honoured.
/// * `device` - Runtime device.
/// * `params` - GPU-side runtime knobs (currently just the wave VRAM budget).
///
/// ### Returns
///
/// One importance vector per target, flat, in the order `batches.iter()
/// .flatten()`. Each inner vector has `n_features` entries and is
/// normalised to sum to 1.0.
///
/// ### Errors
///
/// * Panics (debug) if `batches.len() != batch_seeds.len()`.
/// * Propagates sparse-Y construction failures from
///   [`SparseYBatch::from_targets`].
/// * `InvalidArgument` from [`pick_wave_size`] if even `wave_size = 1` busts
///   `params.wave_byte_budget` on any batch.
/// * Propagates GPU read-back errors from the deferred importance readback.
pub fn fit_scenic_batches_gpu<R: Runtime>(
    batches: &[&[SparseAxis<u32, f32>]],
    batch_seeds: &[usize],
    feature_matrix: &QuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    device: R::Device,
    params: &ScenicGpuParams,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    debug_assert_eq!(
        batches.len(),
        batch_seeds.len(),
        "fit_scenic_batches_gpu: batches.len() must equal batch_seeds.len()"
    );

    let n_features = feature_matrix.n_features;
    let n_trees = config.n_trees();
    let n_features_split = resolve_n_features_split(config.n_features_split(), n_features);
    let k_feats = n_features_split.min(n_features).max(1);
    let max_depth = config.max_depth().unwrap_or(usize::MAX).min(20);
    let min_samples_leaf = config.min_samples_leaf().max(1);
    let n_thresholds = config.n_thresholds().max(1);
    let max_active_nodes = viable_max_active_nodes(max_depth, n_samples, min_samples_leaf);

    let n_sub = if let Some(frac) = config.subsample_frac() {
        ((n_samples as f32 * frac).round() as usize).max(2 * min_samples_leaf)
    } else if config.subsample_rate() >= 1.0 {
        n_samples
    } else {
        ((n_samples as f32 * config.subsample_rate()).round() as usize).max(2 * min_samples_leaf)
    };
    let subsample_needed = n_sub < n_samples;
    let bootstrap = config.bootstrap();

    let n_targets_total: usize = batches.iter().map(|b| b.len()).sum();
    let mut result: Vec<Vec<f32>> = vec![Vec::new(); n_targets_total];

    if batches.is_empty() || n_targets_total == 0 || n_trees == 0 {
        return Ok(result);
    }

    let client = R::client(&device);

    // Single feature upload amortised across every batch. Old code re-did
    // this per fit call; drivers called it once per cluster batch, so the
    // ~n_features * n_samples * 4 bytes went up N_batches times.
    let feature_bins_u32: Vec<u32> = feature_matrix.data.iter().map(|&b| b as u32).collect();
    let feature_data_gpu =
        GpuTensor::<R, u32>::from_slice(&feature_bins_u32, vec![n_features * n_samples], &client);

    // Per-batch importance tensor handles, kept alive until the deferred
    // readback pass. Each entry is (flat_target_offset, batch_n_targets,
    // importance_gpu). Small in VRAM (n_features * batch_n_targets * 4
    // bytes) so N handles across a whole run add up to a few tens of MB
    // even for hundreds of batches.
    let mut deferred: Vec<(usize, usize, GpuTensor<R, u32>)> = Vec::with_capacity(batches.len());

    let mut flat_offset = 0usize;
    for (chunk, &batch_seed) in batches.iter().zip(batch_seeds.iter()) {
        let batch_n_targets = chunk.len();
        if batch_n_targets == 0 {
            continue;
        }

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
        // atomically on device across every tree in this batch. f32 bits
        // stored as u32 so `accumulate_importance` can CAS-add.
        let batch_importances_gpu = GpuTensor::<R, u32>::from_slice(
            &vec![0u32; n_features * batch_n_targets],
            vec![n_features * batch_n_targets],
            &client,
        );

        let mut tree_idx = 0usize;
        while tree_idx < n_trees {
            let this_wave = std::cmp::min(wave_size, n_trees - tree_idx);

            let seeds_host: Vec<u32> = (tree_idx..tree_idx + this_wave)
                .map(|t| tree_seed(batch_seed, t) as u32)
                .collect();
            let tree_seeds_gpu =
                GpuTensor::<R, u32>::from_slice(&seeds_host, vec![this_wave], &client);

            // Per-tree sample multiplicity for this wave. Bootstrap draws
            // n_sub with replacement (mult in {0, 1, 2, ...}); non-bootstrap
            // subsample is Fisher-Yates and takes the first n_sub (mult in
            // {0, 1}); no-subsample path fills 1s. RNG seeded off
            // `tree_seed(batch_seed, t)` matching the CPU path.
            let mut mult_host = vec![0u32; this_wave * n_samples];
            if subsample_needed {
                for w in 0..this_wave {
                    let mut rng =
                        SmallRng::seed_from_u64(tree_seed(batch_seed, tree_idx + w));
                    let row_base = w * n_samples;
                    if bootstrap {
                        for _ in 0..n_sub {
                            let idx = rng.random_range(0..n_samples);
                            mult_host[row_base + idx] += 1;
                        }
                    } else {
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
            let mult_gpu =
                GpuTensor::<R, u32>::from_slice(&mult_host, vec![this_wave * n_samples], &client);

            // Terminal-wave reuse: kernels gate on the `wave_size` param, so
            // an over-provisioned `state` is safe. Only when `this_wave` does
            // not match do we allocate a smaller shadow state.
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

        // Stash the importance handle; no .read() here. The .read() below
        // flushes the queue at that point, so later batches keep submitting
        // kernels while we wait on the first one.
        deferred.push((flat_offset, batch_n_targets, batch_importances_gpu));
        flat_offset += batch_n_targets;
    }

    for (target_offset, batch_n_targets, imp_gpu) in deferred {
        let bits = imp_gpu.read(&client)?;
        let batch_imp: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        for k in 0..batch_n_targets {
            let mut per_target = vec![0.0f32; n_features];
            for f in 0..n_features {
                per_target[f] = batch_imp[f * batch_n_targets + k];
            }
            normalise_importances(&mut per_target);
            result[target_offset + k] = per_target;
        }
    }

    Ok(result)
}

/// Multi-tree, multi-batch ExtraTrees or RandomForest regression fit on GPU.
///
/// Backward-compatible wrapper around [`fit_scenic_batches_gpu`]: chunks
/// `targets` into `MULTI_OUTPUT_BATCH`-sized batches, uses `seed` for every
/// batch (matching the pre-Phase-B semantics where a single call's internal
/// batches all shared one base seed), and forwards to the batch-aware entry.
///
/// New callers (top-level GPU drivers) should skip this and call
/// [`fit_scenic_batches_gpu`] directly with their own cluster-aware batch
/// slices and per-batch seeds.
///
/// ### Params
///
/// * `targets` - Sparse target expression columns.
/// * `feature_matrix` - Quantised u8 features, column-major.
/// * `n_samples` - Total sample count.
/// * `config` - Tree configuration.
/// * `seed` - Base seed shared by every internal batch.
/// * `device` - Runtime device.
/// * `params` - GPU-side runtime knobs.
///
/// ### Returns
///
/// One importance vector per target: `result[target_idx][feature_idx]`, each
/// normalised to sum to 1.0.
pub fn fit_multi_trees_gpu<R: Runtime>(
    targets: &[SparseAxis<u32, f32>],
    feature_matrix: &QuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    seed: usize,
    device: R::Device,
    params: &ScenicGpuParams,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    let batches: Vec<&[SparseAxis<u32, f32>]> = targets.chunks(MULTI_OUTPUT_BATCH).collect();
    let batch_seeds: Vec<usize> = vec![seed; batches.len()];
    fit_scenic_batches_gpu::<R>(
        &batches,
        &batch_seeds,
        feature_matrix,
        n_samples,
        config,
        device,
        params,
    )
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
///
/// ### Params
///
/// * `sy` - Sparse Y batch; `target_indices` holds the u8 target column ids
///
/// ### Returns
///
/// `max(target_indices) + 1`, or `1` when `target_indices` is empty.
fn sparse_y_infer_n_targets(sy: &SparseYBatch) -> usize {
    let mut max = 0u32;
    for &t in &sy.target_indices {
        if t as u32 > max {
            max = t as u32;
        }
    }
    (max as usize) + 1
}

/////////////////////////
// Top-level GPU entry //
/////////////////////////

/// Knuth's multiplicative constant. Mixes a per-batch counter into the base
/// seed so that adjacent batches get well-separated tree seeds, matching the
/// CPU per-batch stride in `run_scenic_multi_output` and
/// `run_scenic_multi_output_streaming`.
const BATCH_SEED_STRIDE: usize = 2_654_435_761;

/// Human-readable name of the SCENIC regression learner. Used for both error
/// messages when the caller picks a learner the GPU path does not implement
/// and for driver-side verbose printouts.
///
/// ### Params
///
/// * `learner` - Reference to the configured learner.
///
/// ### Returns
///
/// A static string naming the learner variant.
fn learner_name(learner: &RegressionLearner) -> &'static str {
    match learner {
        RegressionLearner::ExtraTrees(_) => "ExtraTrees",
        RegressionLearner::RandomForest(_) => "RandomForest",
        RegressionLearner::GradientBoosting(_) => "GradientBoosting",
    }
}

/// Return a shared reference to the concrete tree config the multi-output
/// path expects. The GBM arm is unreachable in practice: every caller checks
/// for `RegressionLearner::GradientBoosting` and short-circuits with
/// `GpuNotSupportedForLearner` before calling this. The panic exists purely
/// to guard the local invariant.
///
/// ### Params
///
/// * `params` - SCENIC parameters, guaranteed non-GBM by the caller.
///
/// ### Returns
///
/// Trait object over the tree regressor config.
fn tree_config_of(params: &ScenicParams) -> &dyn TreeRegressorConfig {
    match &params.regression_learner {
        RegressionLearner::ExtraTrees(cfg) => cfg,
        RegressionLearner::RandomForest(cfg) => cfg,
        RegressionLearner::GradientBoosting(_) => {
            unreachable!("GBM must be rejected before reaching tree_config_of")
        }
    }
}

/// GPU equivalent of [`run_scenic_grn`]: reads all target-gene columns up
/// front, runs cluster-aware gene batching, and hands the whole batch list
/// to [`fit_scenic_batches_gpu`] in one call. Returns the same
/// `(n_genes, n_tfs)` importance matrix as the CPU function.
///
/// The single fit call amortises the feature-tensor upload across every
/// batch and defers per-batch importance readbacks until after all waves
/// have been submitted to the queue.
///
/// GBM is a GPU non-goal; passing a `GradientBoosting` learner yields
/// [`BixverseErrors::GpuNotSupportedForLearner`]. Fall back to
/// `run_scenic_grn` for that path.
///
/// ### Params
///
/// * `f_path` - Path to the sparse gene expression file.
/// * `cell_indices` - Indices of cells to use.
/// * `gene_indices` - Target gene indices.
/// * `tf_indices` - Transcription factor gene indices (predictors).
/// * `scenic_params` - Reference to the SCENIC parameters.
/// * `gpu_params` - GPU-side runtime knobs (wave VRAM budget).
/// * `seed` - Base random seed for reproducibility.
/// * `device` - CubeCL runtime device.
/// * `verbose` - `0` -> silent, `1` -> normal, `2` -> detailed.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` where entry `[i, j]` is the
/// normalised importance of TF `j` for target gene `i`.
///
/// ### References
///
/// Aibar et al., Nat Methods, 2017.
#[allow(clippy::too_many_arguments)]
pub fn run_scenic_grn_gpu<R>(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    scenic_params: &ScenicParams,
    gpu_params: &ScenicGpuParams,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors>
where
    R: Runtime,
{
    if matches!(
        scenic_params.regression_learner,
        RegressionLearner::GradientBoosting(_)
    ) {
        return Err(BixverseErrors::GpuNotSupportedForLearner {
            learner: learner_name(&scenic_params.regression_learner),
        });
    }

    let verbosity = parse_verbosity_level(verbose);
    let setup = scenic_common_setup(f_path, cell_indices, tf_indices, verbose)?;
    let n_genes = gene_indices.len();

    let n_multi_output = scenic_params
        .gene_batch_size
        .unwrap_or(MULTI_OUTPUT_BATCH)
        .min(MULTI_OUTPUT_BATCH);

    let strategy = parse_gene_batch_strategy(
        &scenic_params.gene_batch_strategy,
        scenic_params.n_pcs,
        scenic_params.n_subsample,
    )
    .unwrap_or(GeneBatchStrategy::Random);

    let batches = batch_genes(
        f_path,
        gene_indices,
        cell_indices,
        n_multi_output,
        &strategy,
        seed,
        verbose,
    )?;
    let ordered_genes: Vec<usize> = batches.iter().flatten().copied().collect();

    let gene_id_to_pos: FxHashMap<usize, usize> = gene_indices
        .iter()
        .enumerate()
        .map(|(pos, &gid)| (gid, pos))
        .collect();

    let start_gene_read = Instant::now();
    let mut all_sparse_cols: Vec<SparseAxis<u32, f32>> = Vec::with_capacity(n_genes);

    for (iter, chunk) in ordered_genes.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        let mut gene_chunks: Vec<CscGeneChunk> = setup.reader.read_gene_parallel(chunk)?;
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(&setup.cell_set);
        });

        for gc in gene_chunks.iter() {
            all_sparse_cols.push(gc.to_sparse_axis(setup.n_cells));
        }

        if verbosity.detailed_verbosity() {
            println!(
                "  Read gene chunk {}/{} ({} genes)",
                iter + 1,
                ordered_genes.len().div_ceil(SCENIC_GENE_CHUNK_SIZE),
                all_sparse_cols.len(),
            );
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "Read and filtered {} target genes in {:.2?}",
            n_genes,
            start_gene_read.elapsed()
        );
    }

    // Cluster-bounded batch slices over the flattened column vector.
    let mut col_batches: Vec<&[SparseAxis<u32, f32>]> = Vec::with_capacity(batches.len());
    let mut offset = 0usize;
    for b in &batches {
        col_batches.push(&all_sparse_cols[offset..offset + b.len()]);
        offset += b.len();
    }
    let total_batches = col_batches.len();

    let learner_name = learner_name(&scenic_params.regression_learner);
    let config = tree_config_of(scenic_params);

    if verbosity.normal_verbosity() {
        println!(
            "Running SCENIC GPU ({}) on {} genes ({} TFs, {} cells, {} batches of up to {})",
            learner_name, n_genes, setup.n_tfs, setup.n_cells, total_batches, n_multi_output,
        );
    }

    let start_fit = Instant::now();
    let batch_seeds: Vec<usize> = (0..total_batches)
        .map(|i| seed.wrapping_add(i.wrapping_mul(BATCH_SEED_STRIDE)))
        .collect();

    let flat_imp = fit_scenic_batches_gpu::<R>(
        &col_batches,
        &batch_seeds,
        &setup.tf_data,
        setup.n_cells,
        config,
        device,
        gpu_params,
    )?;

    let flat_gene_ids: Vec<usize> = batches.iter().flatten().copied().collect();
    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];
    for (imp, gene_id) in flat_imp.into_iter().zip(flat_gene_ids.iter()) {
        let pos = gene_id_to_pos[gene_id];
        importance_scores[pos] = imp;
    }

    if verbosity.normal_verbosity() {
        println!(
            "SCENIC GPU ({}) GRN inference complete in {:.2?} (fit {:.2?})",
            learner_name,
            setup.start_total.elapsed(),
            start_fit.elapsed()
        );
    }

    Ok(Mat::from_fn(n_genes, setup.n_tfs, |i, j| {
        if j < importance_scores[i].len() {
            importance_scores[i][j]
        } else {
            0.0
        }
    }))
}

/// GPU equivalent of [`run_scenic_grn_streaming`]: reads target genes in I/O
/// chunks and dispatches every in-chunk batch through a single
/// [`fit_scenic_batches_gpu`] call. Bounds peak host memory to one chunk of
/// sparse columns (roughly `SCENIC_GENE_CHUNK_SIZE` targets).
///
/// ### Params
///
/// See [`run_scenic_grn_gpu`]. Identical signature and semantics apart from
/// the memory bound.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` where entry `[i, j]` is the
/// normalised importance of TF `j` for target gene `i`.
///
/// ### References
///
/// Aibar et al., Nat Methods, 2017.
#[allow(clippy::too_many_arguments)]
pub fn run_scenic_grn_streaming_gpu<R>(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    scenic_params: &ScenicParams,
    gpu_params: &ScenicGpuParams,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors>
where
    R: Runtime,
{
    if matches!(
        scenic_params.regression_learner,
        RegressionLearner::GradientBoosting(_)
    ) {
        return Err(BixverseErrors::GpuNotSupportedForLearner {
            learner: learner_name(&scenic_params.regression_learner),
        });
    }

    let verbosity = parse_verbosity_level(verbose);
    let setup = scenic_common_setup(f_path, cell_indices, tf_indices, verbose)?;
    let n_genes = gene_indices.len();

    let n_multi_output = scenic_params
        .gene_batch_size
        .unwrap_or(MULTI_OUTPUT_BATCH)
        .min(MULTI_OUTPUT_BATCH);

    let strategy = parse_gene_batch_strategy(
        &scenic_params.gene_batch_strategy,
        scenic_params.n_pcs,
        scenic_params.n_subsample,
    )
    .unwrap_or(GeneBatchStrategy::Random);

    let batches = batch_genes(
        f_path,
        gene_indices,
        cell_indices,
        n_multi_output,
        &strategy,
        seed,
        verbose,
    )?;
    let io_groups = group_batches_for_io(&batches, SCENIC_GENE_CHUNK_SIZE);

    let gene_id_to_pos: FxHashMap<usize, usize> = gene_indices
        .iter()
        .enumerate()
        .map(|(pos, &gid)| (gid, pos))
        .collect();

    let learner_name = learner_name(&scenic_params.regression_learner);
    let config = tree_config_of(scenic_params);

    let total_io_chunks = io_groups.len();
    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];
    let mut global_batch_offset: usize = 0;
    let mut genes_processed: usize = 0;

    if verbosity.normal_verbosity() {
        println!(
            "Running SCENIC GPU ({}, streaming) on {} genes ({} TFs, {} cells, {} I/O chunks, batches of {})",
            learner_name,
            n_genes.separate_with_underscores(),
            setup.n_tfs.separate_with_underscores(),
            setup.n_cells.separate_with_underscores(),
            total_io_chunks,
            n_multi_output,
        );
    }

    for (chunk_idx, &(g_start, g_end)) in io_groups.iter().enumerate() {
        let start_chunk = Instant::now();
        let group = &batches[g_start..g_end];
        let io_chunk: Vec<usize> = group.iter().flatten().copied().collect();

        let start_io = Instant::now();
        let mut gene_chunks: Vec<CscGeneChunk> = setup.reader.read_gene_parallel(&io_chunk)?;
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(&setup.cell_set);
        });

        let sparse_columns: Vec<SparseAxis<u32, f32>> = gene_chunks
            .iter()
            .map(|c| c.to_sparse_axis(setup.n_cells))
            .collect();
        drop(gene_chunks);

        if verbosity.normal_verbosity() {
            println!(
                "  Chunk {}/{}: loaded and filtered {} genes in {:.2?}",
                chunk_idx + 1,
                total_io_chunks,
                io_chunk.len(),
                start_io.elapsed()
            );
        }

        let mut col_offsets = Vec::with_capacity(group.len() + 1);
        let mut off = 0usize;
        col_offsets.push(0);
        for b in group {
            off += b.len();
            col_offsets.push(off);
        }
        let n_batches_this_chunk = group.len();

        let start_fit = Instant::now();

        let chunk_col_batches: Vec<&[SparseAxis<u32, f32>]> = (0..n_batches_this_chunk)
            .map(|i| &sparse_columns[col_offsets[i]..col_offsets[i + 1]])
            .collect();
        let chunk_batch_seeds: Vec<usize> = (0..n_batches_this_chunk)
            .map(|i| {
                seed.wrapping_add((global_batch_offset + i).wrapping_mul(BATCH_SEED_STRIDE))
            })
            .collect();

        let flat_imp = fit_scenic_batches_gpu::<R>(
            &chunk_col_batches,
            &chunk_batch_seeds,
            &setup.tf_data,
            setup.n_cells,
            config,
            device.clone(),
            gpu_params,
        )?;

        let chunk_flat_gene_ids: Vec<usize> = group.iter().flatten().copied().collect();
        for (imp, gene_id) in flat_imp.into_iter().zip(chunk_flat_gene_ids.iter()) {
            let pos = gene_id_to_pos[gene_id];
            importance_scores[pos] = imp;
        }

        global_batch_offset += n_batches_this_chunk;
        genes_processed += io_chunk.len();

        if verbosity.detailed_verbosity() {
            println!(
                "    Chunk {}: fit {} batches in {:.2?}",
                chunk_idx + 1,
                n_batches_this_chunk,
                start_fit.elapsed()
            );
        }

        if verbosity.normal_verbosity() {
            println!(
                "  Chunk {}/{}: {}/{} genes done in {:.2?} (fit: {:.2?})",
                chunk_idx + 1,
                total_io_chunks,
                genes_processed,
                n_genes,
                start_chunk.elapsed(),
                start_fit.elapsed()
            );
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "SCENIC GPU ({}) GRN inference (streaming) complete in {:.2?}",
            learner_name,
            setup.start_total.elapsed()
        );
    }

    Ok(Mat::from_fn(n_genes, setup.n_tfs, |i, j| {
        if j < importance_scores[i].len() {
            importance_scores[i][j]
        } else {
            0.0
        }
    }))
}

/// GPU equivalent of
/// [`run_scenic_grn_in_memory`](crate::single_cell::mc_analysis::scenic_metacells::run_scenic_grn_in_memory):
/// runs SCENIC GRN inference against an in-memory cells x genes CSC matrix,
/// handing every cluster-aware batch to [`fit_scenic_batches_gpu`] in one
/// call. Intended for the meta-cell pipeline where the count matrix already
/// fits comfortably in memory and no streaming layer is needed.
///
/// The single fit call amortises the feature-tensor upload across every
/// batch and defers per-batch importance readbacks until after all waves
/// have been submitted to the queue.
///
/// GBM has no GPU implementation; passing a `GradientBoosting` learner yields
/// [`BixverseErrors::GpuNotSupportedForLearner`]. Use `run_scenic_grn_in_memory`
/// for that path.
///
/// ### Params
///
/// * `expr_csc` - Cells x genes CSC (raw in `data`, normalised in `data_2`).
///   CSR inputs are transformed to CSC internally.
/// * `tf_indices` - Column indices of the TFs (predictors).
/// * `scenic_params` - SCENIC configuration.
/// * `gpu_params` - GPU-side runtime knobs (wave VRAM budget).
/// * `seed` - Base random seed.
/// * `device` - CubeCL runtime device.
/// * `verbose` - `0` -> silent, `1` -> normal, `2` -> detailed.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` where entry `[i, j]` is the
/// normalised importance of TF `j` for target gene `i`.
///
/// ### References
///
/// Aibar et al., Nat Methods, 2017.
#[allow(clippy::too_many_arguments)]
pub fn run_scenic_grn_in_memory_gpu<R, T>(
    expr_csc: &CompressedSparseData2<T, f32>,
    tf_indices: &[usize],
    scenic_params: &ScenicParams,
    gpu_params: &ScenicGpuParams,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors>
where
    R: Runtime,
    T: BixverseNumeric + Copy + Into<u32> + Sync,
{
    if matches!(
        scenic_params.regression_learner,
        RegressionLearner::GradientBoosting(_)
    ) {
        return Err(BixverseErrors::GpuNotSupportedForLearner {
            learner: learner_name(&scenic_params.regression_learner),
        });
    }

    let verbosity = parse_verbosity_level(verbose);

    let csc_owned;
    let csc: &CompressedSparseData2<T, f32> = match expr_csc.cs_type {
        CompressedSparseFormat::Csc => expr_csc,
        CompressedSparseFormat::Csr => {
            csc_owned = expr_csc.transform();
            &csc_owned
        }
    };

    let start_total = Instant::now();
    let n_cells = csc.shape.0;
    let n_genes = csc.shape.1;
    let n_tfs = tf_indices.len();

    let start_quant = Instant::now();
    let tf_data = build_tf_quantised_store(csc, tf_indices, n_cells);
    if verbosity.normal_verbosity() {
        println!(
            "Quantised TF store (n: {}) in: {:.2?}",
            n_tfs.separate_with_underscores(),
            start_quant.elapsed()
        );
    }

    let n_multi_output = scenic_params
        .gene_batch_size
        .unwrap_or(MULTI_OUTPUT_BATCH)
        .min(MULTI_OUTPUT_BATCH);

    let strategy = parse_gene_batch_strategy(
        &scenic_params.gene_batch_strategy,
        scenic_params.n_pcs,
        scenic_params.n_subsample,
    )
    .unwrap_or(GeneBatchStrategy::Random);

    let ordered_genes = batch_genes_in_memory(csc, n_multi_output, &strategy, seed, verbose)?;

    let start_extract = Instant::now();
    let all_sparse_cols: Vec<SparseAxis<u32, f32>> = ordered_genes
        .par_iter()
        .map(|&g| extract_target_column(csc, g, n_cells))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "Extracted {} target columns in {:.2?}",
            n_genes,
            start_extract.elapsed()
        );
    }

    let id_batches: Vec<&[usize]> = ordered_genes.chunks(n_multi_output).collect();
    let col_batches: Vec<&[SparseAxis<u32, f32>]> =
        all_sparse_cols.chunks(n_multi_output).collect();
    let total_batches = col_batches.len();

    let learner_name = learner_name(&scenic_params.regression_learner);
    let config = tree_config_of(scenic_params);

    if verbosity.normal_verbosity() {
        println!(
            "Running SCENIC GPU ({}, in-memory) on {} genes ({} TFs, {} cells, {} batches of up to {})",
            learner_name, n_genes, n_tfs, n_cells, total_batches, n_multi_output,
        );
    }

    let start_fit = Instant::now();
    let batch_seeds: Vec<usize> = (0..total_batches)
        .map(|i| seed.wrapping_add(i.wrapping_mul(BATCH_SEED_STRIDE)))
        .collect();

    let flat_imp = fit_scenic_batches_gpu::<R>(
        &col_batches,
        &batch_seeds,
        &tf_data,
        n_cells,
        config,
        device,
        gpu_params,
    )?;

    let flat_gene_ids: Vec<usize> = id_batches.iter().flat_map(|b| b.iter().copied()).collect();
    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];
    for (imp, &gene_id) in flat_imp.into_iter().zip(flat_gene_ids.iter()) {
        importance_scores[gene_id] = imp;
    }

    if verbosity.normal_verbosity() {
        println!(
            "SCENIC GPU ({}, in-memory) GRN inference complete in {:.2?} (fit {:.2?})",
            learner_name,
            start_total.elapsed(),
            start_fit.elapsed()
        );
    }

    Ok(Mat::from_fn(n_genes, n_tfs, |i, j| {
        if j < importance_scores[i].len() {
            importance_scores[i][j]
        } else {
            0.0
        }
    }))
}
