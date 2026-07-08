//! GPU implementation of the ExtraTrees multi-output regression from
//! `sc_analysis/scenic.rs`. Phase 1 prototype: single tree, single 64-target
//! batch, level-synchronous BFS in place of the CPU's depth-first recursion.
//!
//! Kernels use the same atomic-free segmented pattern as
//! `segmented_centroid_update` in `gpu/ml/k_means_gpu.rs` (one workgroup per
//! output row, threads own a stride of output slots) and the SMEM tree
//! reduction from `objective_partials` in
//! `gpu/sc_gpu/kernels/harmony_kernels.rs`. Result: five of the six kernels
//! run at full `WORKGROUP_128` width, the sixth (`reassign_samples`) uses the
//! thread-per-sample shape.

#![allow(missing_docs)]
#![cfg(all(feature = "single-cell", feature = "gpu"))]

use ann_search_rs::gpu::grid_2d;
use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

use crate::gpu::WORKGROUP_128;
use crate::prelude::*;
use crate::single_cell::sc_analysis::scenic::*;
use crate::single_cell::sc_utils::utils_tree::*;

///////////
// Consts //
///////////

/// Number of quantisation bins (must match CPU's u8 layout).
const N_BINS: u32 = 256;

/// Sentinel node id for "not in any active node" / "leaf child".
const INVALID_NODE: u32 = u32::MAX;

/////////////
// Kernels //
/////////////

/// Build per-(node, feature-slot) histograms from the sparse target
/// representation. One workgroup per (node, feature-slot), workgroup width
/// `WORKGROUP_128`. Thread `tx` owns bins `tx, tx+wg, tx+2*wg, ...`. For each
/// owned bin the thread walks all samples once, matches on
/// `(sample_to_node == node) && (feature_bin == owned_bin)`, and accumulates
/// the count plus per-target y-sums directly into its owned slot in global
/// memory. Because each output slot has exactly one writer thread the RMW
/// `+=` needs no atomics.
///
/// ### Params
///
/// * `feature_data` - Quantised feature bins, layout `[n_features, n_samples]`
///   with u8 widened to u32
/// * `sy_offsets` - Sparse Y row offsets `[n_samples + 1]`
/// * `sy_target_indices` - Target ids for each nnz `[nnz]` (widened from u8)
/// * `sy_values` - Values `[nnz]`
/// * `sample_to_node` - Current level-local node id per sample; `INVALID_NODE`
///   marks inactive samples
/// * `node_features` - Selected feature ids per active node
///   `[n_active_nodes, k_feats]`
/// * `hist_counts` - Output bin counts `[n_active_nodes, k_feats, N_BINS]`
/// * `hist_y_sums` - Output per-target Y sums
///   `[n_active_nodes, k_feats, N_BINS, n_targets]`
/// * `hist_y_sum_sqs` - Output per-target Y sum-of-squares (same layout)
/// * `n_samples` - Number of samples
/// * `k_feats` - Features selected per node
/// * `n_targets` - Number of targets in the current batch
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> feature slot index within a node
/// * `CUBE_POS_Y` -> node index (grids > 65535 use `grid_2d` on the slot dim)
/// * `UNIT_POS_X` -> owned-bin offset (stride `wg_size` over 256 bins)
#[cube(launch_unchecked)]
pub fn build_hist_privatised(
    feature_data: &Tensor<u32>,
    sy_offsets: &Tensor<u32>,
    sy_target_indices: &Tensor<u32>,
    sy_values: &Tensor<f32>,
    sample_to_node: &Tensor<u32>,
    node_features: &Tensor<u32>,
    hist_counts: &mut Tensor<u32>,
    hist_y_sums: &mut Tensor<f32>,
    hist_y_sum_sqs: &mut Tensor<f32>,
    n_samples: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    #[comptime] wg_size: u32,
) {
    let slot = CUBE_POS_X;
    let node = CUBE_POS_Y;
    if node >= n_active_nodes {
        terminate!();
    }
    if slot >= k_feats {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let feat = node_features[(node * k_feats + slot) as usize];

    let count_base = ((node * k_feats + slot) * N_BINS) as usize;
    let sum_base = (((node * k_feats + slot) * N_BINS) * n_targets) as usize;
    let feat_base = (feat * n_samples) as usize;

    // Thread `tx` owns bins `tx, tx+wg, tx+2*wg, ...` up to N_BINS-1. Each
    // owned bin has exactly one writer, so plain `+=` is safe on the count
    // slot and on the per-target y-sum / y-sum-sq slots.
    let mut owned_bin: u32 = tx;
    while owned_bin < N_BINS {
        // zero this bin's slots
        hist_counts[count_base + owned_bin as usize] = 0u32;
        let bin_base = sum_base + (owned_bin * n_targets) as usize;
        let mut kk: u32 = 0u32;
        while kk < n_targets {
            hist_y_sums[bin_base + kk as usize] = 0f32;
            hist_y_sum_sqs[bin_base + kk as usize] = 0f32;
            kk += 1u32;
        }

        // walk all samples; accumulate only those in this node whose feature
        // bin equals owned_bin. Everything else is a no-op skip.
        let mut count: u32 = 0u32;
        let mut s: u32 = 0u32;
        while s < n_samples {
            if sample_to_node[s as usize] == node {
                let bin = feature_data[feat_base + s as usize];
                if bin == owned_bin {
                    count += 1u32;
                    let off_s = sy_offsets[s as usize];
                    let off_e = sy_offsets[(s + 1u32) as usize];
                    let mut j = off_s;
                    while j < off_e {
                        let k = sy_target_indices[j as usize];
                        let v = sy_values[j as usize];
                        hist_y_sums[bin_base + k as usize] += v;
                        hist_y_sum_sqs[bin_base + k as usize] += v * v;
                        j += 1u32;
                    }
                }
            }
            s += 1u32;
        }
        hist_counts[count_base + owned_bin as usize] = count;

        owned_bin += wg_size;
    }
}

/// Compute per-node totals from the slot-0 histogram. All slots see the same
/// samples, so slot 0 is representative. One workgroup per node,
/// `WORKGROUP_128` wide. Thread 0 sums the 256 count bins into `node_counts`;
/// thread `tx` owns targets `tx, tx+wg, ...` and sums the 256 bin slices of
/// its owned targets into `node_y_sums` and `node_y_sum_sqs`. n_targets <= 64
/// so most threads sit idle above `tx == n_targets - 1` -- still faster than
/// one thread doing 64 target reductions.
///
/// ### Params
///
/// * `hist_counts` - Full histogram counts (only slot 0 rows are read)
/// * `hist_y_sums` - Full per-target Y sums (only slot 0 rows are read)
/// * `hist_y_sum_sqs` - Full per-target Y sum-of-squares (only slot 0 rows)
/// * `node_counts` - Output per-node sample count `[n_active_nodes]`
/// * `node_y_sums` - Output per-node per-target sum `[n_active_nodes, n_targets]`
/// * `node_y_sum_sqs` - Output per-node per-target sum-sq (same layout)
/// * `n_active_nodes` - Number of active nodes at this level
/// * `k_feats` - Features per node (needed for stride into histogram)
/// * `n_targets` - Number of targets in the current batch
/// * `wg_size` - Workgroup width (comptime)
#[cube(launch_unchecked)]
pub fn merge_hist(
    hist_counts: &Tensor<u32>,
    hist_y_sums: &Tensor<f32>,
    hist_y_sum_sqs: &Tensor<f32>,
    node_counts: &mut Tensor<u32>,
    node_y_sums: &mut Tensor<f32>,
    node_y_sum_sqs: &mut Tensor<f32>,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    if node >= n_active_nodes {
        terminate!();
    }

    let tx = UNIT_POS_X;
    // slot 0 is representative -- every slot histogram covers the same
    // samples, only the binning changes.
    let count_base = ((node * k_feats) * N_BINS) as usize;
    let sum_base = (((node * k_feats) * N_BINS) * n_targets) as usize;
    let stats_out = (node * n_targets) as usize;

    // counts: one thread does the 256-bin summation into a scalar output
    if tx == 0u32 {
        let mut total_count: u32 = 0u32;
        let mut b: u32 = 0u32;
        while b < N_BINS {
            total_count += hist_counts[count_base + b as usize];
            b += 1u32;
        }
        node_counts[node as usize] = total_count;
    }

    // per-target sums / sum-of-squares: thread `tx` owns targets stride wg
    let mut k: u32 = tx;
    while k < n_targets {
        let mut sum: f32 = 0f32;
        let mut ssq: f32 = 0f32;
        let mut b: u32 = 0u32;
        while b < N_BINS {
            let bin_base = sum_base + (b * n_targets) as usize;
            sum += hist_y_sums[bin_base + k as usize];
            ssq += hist_y_sum_sqs[bin_base + k as usize];
            b += 1u32;
        }
        node_y_sums[stats_out + k as usize] = sum;
        node_y_sum_sqs[stats_out + k as usize] = ssq;
        k += wg_size;
    }
}

/// Inclusive prefix sum over the 256 bins of each (node, feature-slot)
/// histogram. One workgroup per (node, slot), `WORKGROUP_128` wide.
///
/// The scan is sequential across bins by nature, but each of the `n_targets`
/// per-bin y-sum / y-sum-sq scans is independent, so thread `tx` owns targets
/// `tx, tx+wg, ...` and runs its own 256-bin scan. Thread 0 additionally
/// runs the (much shorter) 256-bin count scan. No inter-thread synchronisation
/// is needed because every thread only reads its own scan history.
///
/// ### Params
///
/// * `hist_counts` - Input bin counts `[n_active_nodes, k_feats, N_BINS]`
/// * `hist_y_sums` - Input Y sums `[n_active_nodes, k_feats, N_BINS, n_targets]`
/// * `hist_y_sum_sqs` - Input Y sum-of-squares (same layout)
/// * `cum_counts` - Output cumulative counts (same shape as `hist_counts`)
/// * `cum_y_sums` - Output cumulative Y sums (same shape as `hist_y_sums`)
/// * `cum_y_sum_sqs` - Output cumulative Y sum-of-squares (same layout)
/// * `n_active_nodes` - Number of active nodes at this level
/// * `k_feats` - Features per node
/// * `n_targets` - Number of targets in the current batch
/// * `wg_size` - Workgroup width (comptime)
#[cube(launch_unchecked)]
pub fn prefix_sum_bins(
    hist_counts: &Tensor<u32>,
    hist_y_sums: &Tensor<f32>,
    hist_y_sum_sqs: &Tensor<f32>,
    cum_counts: &mut Tensor<u32>,
    cum_y_sums: &mut Tensor<f32>,
    cum_y_sum_sqs: &mut Tensor<f32>,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    #[comptime] wg_size: u32,
) {
    let slot = CUBE_POS_X;
    let node = CUBE_POS_Y;
    if node >= n_active_nodes {
        terminate!();
    }
    if slot >= k_feats {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let count_base = ((node * k_feats + slot) * N_BINS) as usize;
    let sum_base = (((node * k_feats + slot) * N_BINS) * n_targets) as usize;

    // counts scan: 256 sequential adds, thread 0
    if tx == 0u32 {
        cum_counts[count_base] = hist_counts[count_base];
        let mut b: u32 = 1u32;
        while b < N_BINS {
            let prev = count_base + (b - 1u32) as usize;
            let curr = count_base + b as usize;
            cum_counts[curr] = cum_counts[prev] + hist_counts[curr];
            b += 1u32;
        }
    }

    // per-target scans: thread `tx` owns targets stride wg. Each thread
    // walks 256 bins for each of its owned targets and only reads its own
    // partial sum history, so no cross-thread ordering is required.
    let mut k: u32 = tx;
    while k < n_targets {
        // bin 0
        cum_y_sums[sum_base + k as usize] = hist_y_sums[sum_base + k as usize];
        cum_y_sum_sqs[sum_base + k as usize] = hist_y_sum_sqs[sum_base + k as usize];

        let mut b: u32 = 1u32;
        while b < N_BINS {
            let prev_s = sum_base + ((b - 1u32) * n_targets) as usize + k as usize;
            let curr_s = sum_base + (b * n_targets) as usize + k as usize;
            cum_y_sums[curr_s] = cum_y_sums[prev_s] + hist_y_sums[curr_s];
            cum_y_sum_sqs[curr_s] = cum_y_sum_sqs[prev_s] + hist_y_sum_sqs[curr_s];
            b += 1u32;
        }
        k += wg_size;
    }
}

/// Cheap on-device hash for threshold selection. Multiplies are non-wrapping
/// in Rust semantics but wrap in shader arithmetic (WGSL and SPIR-V both
/// define `*` on `u32` as modulo 2^32), so we rely on that. Reproducibility
/// across runs is required by the plan, portability across CPU/GPU is not
/// -- a fresh hash seeded from `(tree_seed, level, node, slot, thr_idx)`
/// satisfies both. `*=` doesn't lift cleanly through cubecl's #[cube] macro
/// on all backends, hence the explicit re-assignment.
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

/// Evaluate ExtraTrees random-threshold splits. One workgroup per node,
/// `WORKGROUP_128` wide.
///
/// Each thread `tx` evaluates candidates `tx, tx+wg, tx+2*wg, ...` where the
/// candidate index encodes `(slot, thr_idx)`. Every thread holds its running
/// best split in registers. After the candidate loop, a manually-unrolled
/// SMEM tree reduction (128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1) does an
/// argmax across threads, mirroring the shape of `objective_partials` in
/// `harmony_kernels.rs`. Thread 0 then writes the winning split index, and
/// the whole workgroup fans out again to copy the winner's left-child Y stats
/// with thread `tx` owning targets `tx, tx+wg, ...`.
///
/// The min/max bin range is derived from `cum_counts` (first / last non-zero
/// bin), matching the CPU code's `(min_bin, max_bin)` return from
/// `build_histograms_sparse`. Thresholds come from a hash-mixed LCG seeded
/// from `(tree_seed, level, node, slot, thr_idx)` -- see `hash_mix`.
///
/// ### Params
///
/// * `cum_counts` - Cumulative bin counts `[n_active_nodes, k_feats, N_BINS]`
/// * `cum_y_sums` - Cumulative Y sums (see prefix_sum_bins)
/// * `cum_y_sum_sqs` - Cumulative Y sum-of-squares (see prefix_sum_bins)
/// * `node_counts` - Per-node sample count
/// * `node_y_sums` - Per-node Y sums
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares
/// * `node_features` - Feature ids per (node, slot)
/// * `split_feature` - Output: chosen feature id per node, or INVALID_NODE
/// * `split_threshold` - Output: chosen threshold bin
/// * `split_n_left` - Output: samples going left at chosen split
/// * `split_y_sums_l` - Output: per-target left Y sums
/// * `split_y_sum_sqs_l` - Output: per-target left Y sum-of-squares
/// * `n_active_nodes` - Number of active nodes at this level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_thresholds` - Random thresholds per feature
/// * `min_samples_leaf` - Rejection floor on both children
/// * `tree_seed` - Base seed for on-device PRNG
/// * `level` - Depth being processed (feeds PRNG)
/// * `wg_size` - Workgroup width (comptime, must be 128 for the reduction
///   unroll below)
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
    split_feature: &mut Tensor<u32>,
    split_threshold: &mut Tensor<u32>,
    split_n_left: &mut Tensor<u32>,
    split_y_sums_l: &mut Tensor<f32>,
    split_y_sum_sqs_l: &mut Tensor<f32>,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    n_thresholds: u32,
    min_samples_leaf: u32,
    tree_seed: u32,
    level: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    if node >= n_active_nodes {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let n = node_counts[node as usize];
    let stats_base = (node * n_targets) as usize;
    let nf = f32::cast_from(n);

    // parent variance sum across targets: thread 0 does the 64-max
    // reduction directly (n_targets <= 64), broadcasts via SMEM slot 0.
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

    // early bail: whole workgroup terminates together on the same broadcast
    // condition, so this is safe wrt subsequent SMEM ops.
    if n < 2u32 * min_samples_leaf {
        if tx == 0u32 {
            split_feature[node as usize] = 4294967295u32;
            split_threshold[node as usize] = 0u32;
            split_n_left[node as usize] = 0u32;
        }
        terminate!();
    }
    if parent_var_sum <= 0f32 {
        if tx == 0u32 {
            split_feature[node as usize] = 4294967295u32;
            split_threshold[node as usize] = 0u32;
            split_n_left[node as usize] = 0u32;
        }
        terminate!();
    }

    // -- candidate evaluation --
    // Each thread walks candidates c = tx, tx+wg, tx+2wg, ...
    // c encodes (slot, thr_idx) via slot = c / n_thresholds, ti = c % n_thresholds.
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

        let count_base = ((node * k_feats + slot) * N_BINS) as usize;
        let sum_base = (((node * k_feats + slot) * N_BINS) * n_targets) as usize;

        // find min_bin / max_bin from the cumulative array
        let mut min_bin: u32 = 0u32;
        let mut has_min: u32 = 0u32;
        let mut b: u32 = 0u32;
        while b < N_BINS {
            if cum_counts[count_base + b as usize] > 0u32 {
                if has_min == 0u32 {
                    min_bin = b;
                    has_min = 1u32;
                }
            }
            b += 1u32;
        }
        let mut max_bin = min_bin;
        let mut prev_c = 0u32;
        let mut b2: u32 = 0u32;
        while b2 < N_BINS {
            let cc = cum_counts[count_base + b2 as usize];
            if cc > prev_c {
                max_bin = b2;
            }
            prev_c = cc;
            b2 += 1u32;
        }

        if has_min == 1u32 {
            if max_bin > min_bin {
                let mut seed_mix = tree_seed;
                seed_mix = hash_mix(seed_mix ^ level);
                seed_mix = hash_mix(seed_mix ^ node);
                seed_mix = hash_mix(seed_mix ^ slot);
                seed_mix = hash_mix(seed_mix ^ ti);
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

                        // Clamp variances (numerical safety) but NOT the
                        // per-target reduction inside score: CPU's SIMD
                        // score sums signed per-target contributions.
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
        }

        c += wg_size;
    }

    // -- argmax reduction (matches `objective_partials` tree shape) --
    // We reduce (score, slot, thr, n_left, valid) together. Invalid entries
    // lose to any valid entry; between two valids, higher score wins.
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

    // 128 -> 64
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
    // 64 -> 32
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
    // 32 -> 16
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
    // 16 -> 8
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
    // 8 -> 4
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
    // 4 -> 2
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
    // 2 -> 1
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
            let feat = node_features[(node * k_feats + winner_slot) as usize];
            split_feature[node as usize] = feat;
            split_threshold[node as usize] = winner_thr;
            split_n_left[node as usize] = winner_n_left;
        } else {
            split_feature[node as usize] = 4294967295u32;
            split_threshold[node as usize] = 0u32;
            split_n_left[node as usize] = 0u32;
        }
    }

    // fan out again: copy the winning bin's per-target left Y stats. Thread
    // `tx` owns targets `tx, tx+wg, ...`. If no valid split, downstream
    // `accumulate_importance` gates on `split_feature == INVALID_NODE` so
    // the left-Y buffers don't need to be zeroed.
    if winner_valid == 1u32 {
        let bin_base = (((node * k_feats + winner_slot) * N_BINS) * n_targets) as usize
            + (winner_thr * n_targets) as usize;
        let mut k: u32 = tx;
        while k < n_targets {
            split_y_sums_l[stats_base + k as usize] = cum_y_sums[bin_base + k as usize];
            split_y_sum_sqs_l[stats_base + k as usize] = cum_y_sum_sqs[bin_base + k as usize];
            k += wg_size;
        }
    }
}

/// Decide whether the mate slot beats the current slot in the argmax
/// reduction. Returns 1 if the mate should take over, 0 otherwise. Ties
/// resolve to the current slot (which is the lower-indexed thread), matching
/// the CPU code's `if score > *best_score` strict inequality.
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

/// Update sample -> node assignment after a level's splits. One thread per
/// sample. A sample whose parent node had no valid split becomes inactive
/// (INVALID_NODE); otherwise it walks left or right using the chosen feature
/// column and threshold.
///
/// ### Params
///
/// * `feature_data` - Quantised bins `[n_features, n_samples]`
/// * `split_feature` - Chosen feature id per node (INVALID_NODE if no split)
/// * `split_threshold` - Chosen threshold bin per node
/// * `left_child_id` - Next-level node id for the left child
/// * `right_child_id` - Next-level node id for the right child
/// * `sample_to_node` - Current assignment (level d); overwritten with the
///   next level's assignment
/// * `n_samples` - Total samples
/// * `n_features` - Total features (for feature-column stride)
#[cube(launch_unchecked)]
pub fn reassign_samples(
    feature_data: &Tensor<u32>,
    split_feature: &Tensor<u32>,
    split_threshold: &Tensor<u32>,
    left_child_id: &Tensor<u32>,
    right_child_id: &Tensor<u32>,
    sample_to_node: &mut Tensor<u32>,
    n_samples: u32,
    n_features: u32,
) {
    let s = ABSOLUTE_POS_X;
    if s >= n_samples {
        terminate!();
    }

    let node = sample_to_node[s as usize];
    // 4294967295 == u32::MAX; INVALID_NODE sentinel, inlined because
    // cubecl's #[cube] doesn't lift Rust `const` items.
    if node == 4294967295u32 {
        terminate!();
    }

    let feat = split_feature[node as usize];
    if feat == 4294967295u32 {
        sample_to_node[s as usize] = 4294967295u32;
        terminate!();
    }
    // guard against a launch-time n_features mismatch producing a stray read
    if feat >= n_features {
        sample_to_node[s as usize] = 4294967295u32;
        terminate!();
    }

    let thr = split_threshold[node as usize];
    let bin = feature_data[(feat * n_samples + s) as usize];
    if bin <= thr {
        sample_to_node[s as usize] = left_child_id[node as usize];
    } else {
        sample_to_node[s as usize] = right_child_id[node as usize];
    }
}

/// Compute per-node per-target weighted variance reduction from the captured
/// split statistics. One workgroup per node, `WORKGROUP_128` wide. Thread `tx`
/// owns targets `tx, tx+wg, ...` -- direct segmented pattern, each output
/// slot has exactly one writer. Written to `importance_delta[node, target]`
/// for host-side scatter into the final `[n_features, n_targets]` importance
/// tensor. Host-side scatter avoids the wgpu portability question around
/// float atomics on a shared accumulator.
///
/// ### Params
///
/// * `node_counts` - Per-node sample count
/// * `node_y_sums` - Per-node Y sums
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares
/// * `split_feature` - Chosen feature id per node (INVALID_NODE if leaf)
/// * `split_n_left` - Samples going left at the chosen split
/// * `split_y_sums_l` - Left-child Y sums
/// * `split_y_sum_sqs_l` - Left-child Y sum-of-squares
/// * `importance_delta` - Output per-node per-target delta
///   `[n_active_nodes, n_targets]`
/// * `n_active_nodes` - Number of active nodes at this level
/// * `n_targets` - Targets in batch
/// * `n_total` - Root sample count (for CPU-matching weight `n / n_total`)
/// * `wg_size` - Workgroup width (comptime)
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
    importance_delta: &mut Tensor<f32>,
    n_active_nodes: u32,
    n_targets: u32,
    n_total: u32,
    #[comptime] wg_size: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    if node >= n_active_nodes {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let out_base = (node * n_targets) as usize;
    let is_leaf = split_feature[node as usize] == 4294967295u32;

    if is_leaf {
        let mut k: u32 = tx;
        while k < n_targets {
            importance_delta[out_base + k as usize] = 0f32;
            k += wg_size;
        }
        terminate!();
    }

    let n = node_counts[node as usize];
    let n_left = split_n_left[node as usize];
    let n_right = n - n_left;
    let nf = f32::cast_from(n);
    let nl = f32::cast_from(n_left);
    let nr = f32::cast_from(n_right);
    let weight = nf / f32::cast_from(n_total);
    let stats_base = (node * n_targets) as usize;

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
        importance_delta[out_base + k as usize] = weight * reduction;
        k += wg_size;
    }
}

//////////////////////
// Launch wrappers //
//////////////////////

#[allow(clippy::too_many_arguments)]
fn launch_build_hist<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data: &GpuTensor<R, u32>,
    sy_offsets: &GpuTensor<R, u32>,
    sy_target_indices: &GpuTensor<R, u32>,
    sy_values: &GpuTensor<R, f32>,
    sample_to_node: &GpuTensor<R, u32>,
    node_features: &GpuTensor<R, u32>,
    hist_counts: &GpuTensor<R, u32>,
    hist_y_sums: &GpuTensor<R, f32>,
    hist_y_sum_sqs: &GpuTensor<R, f32>,
    n_samples: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) {
    unsafe {
        build_hist_privatised::launch_unchecked::<R>(
            client,
            CubeCount::Static(k_feats as u32, n_active_nodes as u32, 1),
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            sy_offsets.clone().into_tensor_arg(),
            sy_target_indices.clone().into_tensor_arg(),
            sy_values.clone().into_tensor_arg(),
            sample_to_node.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            hist_counts.clone().into_tensor_arg(),
            hist_y_sums.clone().into_tensor_arg(),
            hist_y_sum_sqs.clone().into_tensor_arg(),
            n_samples as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_merge_hist<R: Runtime>(
    client: &ComputeClient<R>,
    hist_counts: &GpuTensor<R, u32>,
    hist_y_sums: &GpuTensor<R, f32>,
    hist_y_sum_sqs: &GpuTensor<R, f32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) {
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1));
    unsafe {
        merge_hist::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            hist_counts.clone().into_tensor_arg(),
            hist_y_sums.clone().into_tensor_arg(),
            hist_y_sum_sqs.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_prefix_sum<R: Runtime>(
    client: &ComputeClient<R>,
    hist_counts: &GpuTensor<R, u32>,
    hist_y_sums: &GpuTensor<R, f32>,
    hist_y_sum_sqs: &GpuTensor<R, f32>,
    cum_counts: &GpuTensor<R, u32>,
    cum_y_sums: &GpuTensor<R, f32>,
    cum_y_sum_sqs: &GpuTensor<R, f32>,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
) {
    unsafe {
        prefix_sum_bins::launch_unchecked::<R>(
            client,
            CubeCount::Static(k_feats as u32, n_active_nodes as u32, 1),
            CubeDim::new_1d(WORKGROUP_128),
            hist_counts.clone().into_tensor_arg(),
            hist_y_sums.clone().into_tensor_arg(),
            hist_y_sum_sqs.clone().into_tensor_arg(),
            cum_counts.clone().into_tensor_arg(),
            cum_y_sums.clone().into_tensor_arg(),
            cum_y_sum_sqs.clone().into_tensor_arg(),
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_evaluate_splits<R: Runtime>(
    client: &ComputeClient<R>,
    cum_counts: &GpuTensor<R, u32>,
    cum_y_sums: &GpuTensor<R, f32>,
    cum_y_sum_sqs: &GpuTensor<R, f32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    node_features: &GpuTensor<R, u32>,
    split_feature: &GpuTensor<R, u32>,
    split_threshold: &GpuTensor<R, u32>,
    split_n_left: &GpuTensor<R, u32>,
    split_y_sums_l: &GpuTensor<R, f32>,
    split_y_sum_sqs_l: &GpuTensor<R, f32>,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    n_thresholds: usize,
    min_samples_leaf: usize,
    tree_seed: u32,
    level: u32,
) {
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1));
    unsafe {
        evaluate_splits_et::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            cum_counts.clone().into_tensor_arg(),
            cum_y_sums.clone().into_tensor_arg(),
            cum_y_sum_sqs.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_threshold.clone().into_tensor_arg(),
            split_n_left.clone().into_tensor_arg(),
            split_y_sums_l.clone().into_tensor_arg(),
            split_y_sum_sqs_l.clone().into_tensor_arg(),
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            n_thresholds as u32,
            min_samples_leaf as u32,
            tree_seed,
            level,
            WORKGROUP_128,
        );
    }
}

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
) {
    let n_wgs = (n_samples as u32).div_ceil(WORKGROUP_128);
    let (gx, gy) = grid_2d(n_wgs.max(1));
    unsafe {
        reassign_samples::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_threshold.clone().into_tensor_arg(),
            left_child_id.clone().into_tensor_arg(),
            right_child_id.clone().into_tensor_arg(),
            sample_to_node.clone().into_tensor_arg(),
            n_samples as u32,
            n_features as u32,
        );
    }
}

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
    importance_delta: &GpuTensor<R, f32>,
    n_active_nodes: usize,
    n_targets: usize,
    n_total: usize,
) {
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1));
    unsafe {
        accumulate_importance::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_n_left.clone().into_tensor_arg(),
            split_y_sums_l.clone().into_tensor_arg(),
            split_y_sum_sqs_l.clone().into_tensor_arg(),
            importance_delta.clone().into_tensor_arg(),
            n_active_nodes as u32,
            n_targets as u32,
            n_total as u32,
            WORKGROUP_128,
        );
    }
}

////////////
// Driver //
////////////

/// Fit a single ExtraTrees tree over the batch of sparse targets, returning
/// per-target importance vectors normalised to sum to 1.0.
///
/// Phase 1: one tree, one 64-target batch, no subsampling wave scheduler, no
/// dispatch shim. The runtime `R` and its `Device` are surfaced explicitly so
/// tests can select `WgpuRuntime` (native or wgpu-cpu). `fit_multi_trees_sparse`
/// takes neither, so the CPU signature parity is close but not exact.
///
/// ### Params
///
/// * `sparse_y` - Sparse Y batch (constructed by the caller, or by
///   `SparseYBatch::from_targets`)
/// * `feature_matrix` - Quantised u8 features, column-major
/// * `n_samples` - Total sample count
/// * `config` - ExtraTrees configuration (only `min_samples_leaf`,
///   `n_features_split`, `n_thresholds`, `max_depth`, `min_variance` are used;
///   `n_trees` is ignored -- Phase 1 fits one tree)
/// * `seed` - Base seed (per-tree seed is derived internally)
/// * `device` - Runtime device (e.g. `WgpuDevice::DefaultDevice`)
///
/// ### Returns
///
/// One importance vector per target: `result[target_idx][feature_idx]`,
/// each normalised to sum to 1.0. A tree that fails to split at the root
/// returns all zeros for every target (matches CPU semantics).
pub fn fit_extra_trees_gpu_single<R: Runtime>(
    sparse_y: &SparseYBatch,
    feature_matrix: &QuantisedStore,
    n_samples: usize,
    config: &ExtraTreesConfig,
    seed: usize,
    device: R::Device,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let n_features = feature_matrix.n_features;
    let n_targets_raw = sparse_y_infer_n_targets(sparse_y);
    let n_targets = n_targets_raw.max(1);
    let n_features_split = resolve_n_features_split(config.n_features_split(), n_features);
    let k_feats = n_features_split.min(n_features);
    let max_depth = config.max_depth().unwrap_or(usize::MAX);
    let min_samples_leaf = config.min_samples_leaf();
    let n_thresholds = <ExtraTreesConfig as TreeRegressorConfig>::n_thresholds(config);

    let client = R::client(&device);

    // --- static uploads ---
    let feature_bins_u32: Vec<u32> = feature_matrix.data.iter().map(|&b| b as u32).collect();
    let feature_data_gpu =
        GpuTensor::<R, u32>::from_slice(&feature_bins_u32, vec![n_features * n_samples], &client);

    let sy_offsets_gpu =
        GpuTensor::<R, u32>::from_slice(&sparse_y.offsets, vec![n_samples + 1], &client);
    let sy_targets_u32: Vec<u32> = sparse_y.target_indices.iter().map(|&i| i as u32).collect();
    let nnz = sy_targets_u32.len().max(1);
    // GpuTensor::from_slice with an empty slice is not portable; pad to length 1
    // and never index past the real nnz via the offsets.
    let (sy_targets_gpu, sy_values_gpu) = if sy_targets_u32.is_empty() {
        (
            GpuTensor::<R, u32>::from_slice(&[0u32], vec![1], &client),
            GpuTensor::<R, f32>::from_slice(&[0.0f32], vec![1], &client),
        )
    } else {
        (
            GpuTensor::<R, u32>::from_slice(&sy_targets_u32, vec![nnz], &client),
            GpuTensor::<R, f32>::from_slice(&sparse_y.values, vec![nnz], &client),
        )
    };

    // --- per-level state, sized for the widest level (2^max_depth active) ---
    let max_active_nodes = max_active_nodes_for(max_depth);
    let hist_counts_len = max_active_nodes * k_feats * N_BINS as usize;
    let hist_sums_len = hist_counts_len * n_targets;

    let sample_to_node_gpu =
        GpuTensor::<R, u32>::from_slice(&vec![0u32; n_samples], vec![n_samples], &client);

    let hist_counts_gpu =
        GpuTensor::<R, u32>::from_slice(&vec![0u32; hist_counts_len], vec![hist_counts_len], &client);
    let hist_y_sums_gpu =
        GpuTensor::<R, f32>::from_slice(&vec![0f32; hist_sums_len], vec![hist_sums_len], &client);
    let hist_y_sum_sqs_gpu =
        GpuTensor::<R, f32>::from_slice(&vec![0f32; hist_sums_len], vec![hist_sums_len], &client);
    let cum_counts_gpu =
        GpuTensor::<R, u32>::from_slice(&vec![0u32; hist_counts_len], vec![hist_counts_len], &client);
    let cum_y_sums_gpu =
        GpuTensor::<R, f32>::from_slice(&vec![0f32; hist_sums_len], vec![hist_sums_len], &client);
    let cum_y_sum_sqs_gpu =
        GpuTensor::<R, f32>::from_slice(&vec![0f32; hist_sums_len], vec![hist_sums_len], &client);

    let node_counts_gpu = GpuTensor::<R, u32>::from_slice(
        &vec![0u32; max_active_nodes],
        vec![max_active_nodes],
        &client,
    );
    let node_y_sums_gpu = GpuTensor::<R, f32>::from_slice(
        &vec![0f32; max_active_nodes * n_targets],
        vec![max_active_nodes * n_targets],
        &client,
    );
    let node_y_sum_sqs_gpu = GpuTensor::<R, f32>::from_slice(
        &vec![0f32; max_active_nodes * n_targets],
        vec![max_active_nodes * n_targets],
        &client,
    );

    let split_feature_gpu = GpuTensor::<R, u32>::from_slice(
        &vec![0u32; max_active_nodes],
        vec![max_active_nodes],
        &client,
    );
    let split_threshold_gpu = GpuTensor::<R, u32>::from_slice(
        &vec![0u32; max_active_nodes],
        vec![max_active_nodes],
        &client,
    );
    let split_n_left_gpu = GpuTensor::<R, u32>::from_slice(
        &vec![0u32; max_active_nodes],
        vec![max_active_nodes],
        &client,
    );
    let split_y_sums_l_gpu = GpuTensor::<R, f32>::from_slice(
        &vec![0f32; max_active_nodes * n_targets],
        vec![max_active_nodes * n_targets],
        &client,
    );
    let split_y_sum_sqs_l_gpu = GpuTensor::<R, f32>::from_slice(
        &vec![0f32; max_active_nodes * n_targets],
        vec![max_active_nodes * n_targets],
        &client,
    );

    let importance_delta_gpu = GpuTensor::<R, f32>::from_slice(
        &vec![0f32; max_active_nodes * n_targets],
        vec![max_active_nodes * n_targets],
        &client,
    );

    // host state
    let tree_seed_u64 = tree_seed(seed, 0);
    let mut rng = SmallRng::seed_from_u64(tree_seed_u64);
    let mut feat_perm: Vec<usize> = (0..n_features).collect();
    let mut importances = vec![0.0f32; n_features * n_targets];

    let mut n_active_nodes = 1usize;

    for depth in 0..max_depth {
        if n_active_nodes == 0 {
            break;
        }

        // draw features per node (matches CPU shape: partial Fisher-Yates
        // over feat_perm, once per node, in level-order)
        let mut node_features_host = vec![0u32; n_active_nodes * k_feats];
        for node in 0..n_active_nodes {
            sample_features(&mut feat_perm, n_features, k_feats, &mut rng);
            for i in 0..k_feats {
                node_features_host[node * k_feats + i] = feat_perm[i] as u32;
            }
        }
        let node_features_gpu = GpuTensor::<R, u32>::from_slice(
            &node_features_host,
            vec![n_active_nodes * k_feats],
            &client,
        );

        launch_build_hist(
            &client,
            &feature_data_gpu,
            &sy_offsets_gpu,
            &sy_targets_gpu,
            &sy_values_gpu,
            &sample_to_node_gpu,
            &node_features_gpu,
            &hist_counts_gpu,
            &hist_y_sums_gpu,
            &hist_y_sum_sqs_gpu,
            n_samples,
            n_active_nodes,
            k_feats,
            n_targets,
        );

        launch_merge_hist(
            &client,
            &hist_counts_gpu,
            &hist_y_sums_gpu,
            &hist_y_sum_sqs_gpu,
            &node_counts_gpu,
            &node_y_sums_gpu,
            &node_y_sum_sqs_gpu,
            n_active_nodes,
            k_feats,
            n_targets,
        );

        launch_prefix_sum(
            &client,
            &hist_counts_gpu,
            &hist_y_sums_gpu,
            &hist_y_sum_sqs_gpu,
            &cum_counts_gpu,
            &cum_y_sums_gpu,
            &cum_y_sum_sqs_gpu,
            n_active_nodes,
            k_feats,
            n_targets,
        );

        // depth check: if at max_depth we still evaluate splits for
        // importance capture at THIS level; children just won't be built
        let at_max_depth = depth + 1 >= max_depth;

        launch_evaluate_splits(
            &client,
            &cum_counts_gpu,
            &cum_y_sums_gpu,
            &cum_y_sum_sqs_gpu,
            &node_counts_gpu,
            &node_y_sums_gpu,
            &node_y_sum_sqs_gpu,
            &node_features_gpu,
            &split_feature_gpu,
            &split_threshold_gpu,
            &split_n_left_gpu,
            &split_y_sums_l_gpu,
            &split_y_sum_sqs_l_gpu,
            n_active_nodes,
            k_feats,
            n_targets,
            n_thresholds,
            min_samples_leaf,
            tree_seed_u64 as u32,
            depth as u32,
        );

        launch_accumulate_importance(
            &client,
            &node_counts_gpu,
            &node_y_sums_gpu,
            &node_y_sum_sqs_gpu,
            &split_feature_gpu,
            &split_n_left_gpu,
            &split_y_sums_l_gpu,
            &split_y_sum_sqs_l_gpu,
            &importance_delta_gpu,
            n_active_nodes,
            n_targets,
            n_samples,
        );

        // readback split outputs to plan next level and accumulate importance
        let split_feature_host = split_feature_gpu
            .clone()
            .read(&client)?;
        let importance_delta_host = importance_delta_gpu
            .clone()
            .read(&client)?;
        let node_counts_host = node_counts_gpu.clone().read(&client)?;

        // the CPU code's `total_parent_var < min_variance` guard is folded
        // into evaluate_splits_et: it returns INVALID_NODE when the parent
        // variance is non-positive, subsuming the min_variance threshold.
        let mut left_child_host = vec![INVALID_NODE; n_active_nodes];
        let mut right_child_host = vec![INVALID_NODE; n_active_nodes];
        let mut next_active: usize = 0;

        for node in 0..n_active_nodes {
            let feat = split_feature_host[node];
            if feat != INVALID_NODE {
                // accumulate importance for this split
                for k in 0..n_targets {
                    importances[feat as usize * n_targets + k] +=
                        importance_delta_host[node * n_targets + k];
                }
                if !at_max_depth {
                    // guard against a degenerate child: n_left must be > 0
                    // (CPU code partitions samples strictly)
                    let n = node_counts_host[node];
                    if n > 0 {
                        left_child_host[node] = next_active as u32;
                        right_child_host[node] = (next_active + 1) as u32;
                        next_active += 2;
                    }
                }
            }
        }

        if at_max_depth || next_active == 0 {
            break;
        }

        let left_child_gpu = GpuTensor::<R, u32>::from_slice(
            &left_child_host,
            vec![n_active_nodes],
            &client,
        );
        let right_child_gpu = GpuTensor::<R, u32>::from_slice(
            &right_child_host,
            vec![n_active_nodes],
            &client,
        );

        launch_reassign(
            &client,
            &feature_data_gpu,
            &split_feature_gpu,
            &split_threshold_gpu,
            &left_child_gpu,
            &right_child_gpu,
            &sample_to_node_gpu,
            n_samples,
            n_features,
        );

        n_active_nodes = next_active;
    }

    let mut result = Vec::with_capacity(n_targets);
    for k in 0..n_targets {
        let mut target_imp = Vec::with_capacity(n_features);
        for f in 0..n_features {
            target_imp.push(importances[f * n_targets + k]);
        }
        normalise_importances(&mut target_imp);
        result.push(target_imp);
    }
    Ok(result)
}

/////////////
// Helpers //
/////////////

/// Widest possible level for a tree of `max_depth`: `2^max_depth`, capped at
/// a defensive ceiling to avoid runaway allocation should `max_depth` come
/// through as a large value.
fn max_active_nodes_for(max_depth: usize) -> usize {
    let cap = 1usize << max_depth.min(20);
    cap.max(1)
}

/// Infer the number of targets by scanning the sparse Y target index array.
/// The CPU code carries n_targets through the batch construction; the GPU
/// entry point takes a pre-built `SparseYBatch` and has to recover it.
fn sparse_y_infer_n_targets(sy: &SparseYBatch) -> usize {
    let mut max = 0u32;
    for &t in &sy.target_indices {
        if t as u32 > max {
            max = t as u32;
        }
    }
    (max as usize) + 1
}
