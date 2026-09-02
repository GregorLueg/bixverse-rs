//! GPU implementation of the multi-output tree regression from
//! `sc_analysis/scenic.rs`: wave-scheduled, multi-tree, multi-batch ExtraTrees
//! and RandomForest as a level-synchronous BFS sharing a common tail of
//! node-bookkeeping kernels.
//!
//! `config.random_threshold()` picks one of two split paths, see
//! [`WaveLayout`]:
//!
//! * ExtraTrees: [`scan_slot_bin_range()`] -> [`accumulate_split_stats_et()`] ->
//!   [`evaluate_splits_et_direct()`]. No histogram at all.
//! * RandomForest: [`build_score_rf_fused()`] -> [`reduce_slot_winners()`],
//!   histogram in threadgroup memory. There is no DRAM-histogram fallback; a
//!   device that cannot spare the shared memory is rejected up front by
//!   `require_fused_rf`.
//!
//! Kernels run [`WORKGROUP_128`] wide, except [`evaluate_splits_et_direct()`],
//! which is [`WORKGROUP_32`] so its argmax fits one plane.

#![allow(missing_docs)]
#![cfg(all(feature = "single-cell", feature = "gpu"))]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
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
const DEFAULT_WAVE_SIZE: usize = 32;

/// Upper bound on planes per workgroup. Sizes the scan-offset scratch in
/// [`accumulate_split_stats_et()`] and gates `plane_compact_viable`. Workgroups
/// are at most `WORKGROUP_128` wide and no shipping backend reports a plane
/// narrower than 4, so 32 slots is a generous ceiling at 128 bytes of shared
/// memory.
const MAX_PLANES_PER_CUBE: u32 = 32;

/// Float slots reserved for the fused RandomForest histogram in threadgroup
/// memory.
///
/// Rows are padded to `n_targets + 1`, so this bounds
/// `n_bins_gpu * (n_targets + 1)`. The pad matters because the scoring phase
/// gives consecutive threads consecutive bins: at an unpadded stride of 64 they
/// all resolve to the same bank and the pass serialises 32 ways.
///
/// This is an occupancy knob, not just a capacity bound. The kernel is latency
/// bound, so what matters is workgroups resident per core (32768 / footprint).
/// Measured on `rf_32t`, letting [`pick_gpu_bins`] fall out of the value:
///
/// | slots | bins @ 64 targets | footprint | resident | ratio |
/// |---|---|---|---|---|
/// | 4352 | 64 | 22024 B | 1 | 10.80x |
/// | **2176** | **32** | **13320 B** | **2** | **18.92x** |
/// | 1088 | 16 | 8968 B | 3 | 22.17x |
/// | 544 | 8 | 6792 B | 4 | 22.40x |
///
/// Nearly all of the win is the step from one resident workgroup to two; past
/// 16 bins the speed saturates and only accuracy keeps falling. 2176 is
/// deliberately not the fastest setting: fidelity measured flat down to 4 bins,
/// which is weak evidence rather than a licence to coarsen (the metric
/// correlates importance vectors, and the synthetic targets are monotone in
/// their informative features, so any mid-range split preserves the ranking).
const SMEM_HIST_SLOTS: u32 = 2176;

/// Samples processed per iteration of the fused accumulate loop.
///
/// A latency knob: the loop is bound on the per-sample dense-Y fetch and only
/// one workgroup fits per core, so the fix is several fetches in flight at once.
/// Measured on `rf_32t` (GPU seconds): 1 -> 0.46, 4 -> 0.26, 8 -> 0.23,
/// 16 -> 0.22, and 24 and 32 give nothing further. 16 over 32 for the smaller
/// register footprint.
const RF_UNROLL: u32 = 16;

/// Sentinel for "no node" / "no valid split" / "no child" in the u32-typed
/// device buffers (`split_feature`, `sample_to_node`, `left_child_id`,
/// `right_child_id`).
const INVALID_NODE: u32 = u32::MAX;

// `build_score_rf_fused`, `finalise_split_stats_rf` and
// `accumulate_split_stats_et` put target `k` on thread `k`, so a batch wider
// than the workgroup would silently drop every target past thread 127. The
// shared-memory budget does not catch it: at 129 targets `pick_gpu_bins` still
// returns a bin count that fits.
const _: () = assert!(MULTI_OUTPUT_BATCH <= WORKGROUP_128 as usize);

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
        // Same hash chain shape as the threshold draw in
        // accumulate_split_stats_et.
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

/// Read one quantised bin out of the packed feature tensor.
///
/// `feature_data` holds four u8 bins per u32 word, indexed by the flat
/// `feature * n_samples + sample` position. Packing is worth 4x on both the
/// device tensor and the host staging buffer: unpacked, a 1000-TF by 1M-cell
/// matrix is 4 GB, which is exactly the per-binding limit on an M1 Max, and it
/// costs a 4 GB host allocation to widen u8 to u32 besides.
///
/// Consecutive samples share a word, so threads walking samples still touch
/// each cache line once.
///
/// ### Workgroup
///
/// Inlined into the calling kernel; no dedicated workgroup.
///
/// ### Params
///
/// * `feature_data` - Packed bins, four per u32 word
/// * `idx` - Flat `feature * n_samples + sample` index of the wanted bin
///
/// ### Returns
///
/// The bin value in `[0, N_BINS)`.
#[cube]
fn feature_bin(feature_data: &Tensor<u32>, idx: u32) -> u32 {
    let word = feature_data[(idx / 4u32) as usize];
    (word >> ((idx % 4u32) * 8u32)) & 255u32
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

/// Recompute the winning split's left-child Y sums and sums-of-squares from the
/// node's actual samples.
///
/// Dropping the sum-of-squares histogram is what buys [`build_score_rf_fused()`]
/// its shared-memory budget, so the winner's `ssyl` has to come from somewhere
/// else. One pass over the node's gathered samples at the already-decided
/// threshold is that somewhere, and it runs per (node, tree) rather than per
/// (slot, node, tree), so it costs `1/k_feats` of a histogram build.
///
/// Reads the dense Y so target sits on the thread axis and the loads coalesce.
/// Values differ in the last f32 place from the histogram-derived ones: this
/// sums in sample order, the histogram sums per bin then prefix-sums across
/// bins. Same terms, different association.
///
/// ### Params
///
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `y_dense` - Dense Y `[n_samples, n_targets]`, target contiguous
/// * `node_sample_offsets` - Node slice bounds `[wave_size, n_active_nodes + 1]`
/// * `node_sample_ids` - Gathered sample ids `[wave_size, n_samples]`
/// * `node_sample_mults` - Gathered multiplicities, same layout
/// * `split_feature` - Winning feature per node; `INVALID_NODE` means leaf
/// * `split_threshold` - Winning threshold per node, in *fine* bin space
/// * `split_y_sums_l` - Output left-child Y sums `[wave_size, n_active_nodes,
///   n_targets]`
/// * `split_y_sum_sqs_l` - Output left-child Y sums-of-squares, same layout
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `n_targets` - Targets in batch
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> target index
///
/// ### Returns
///
/// `split_y_sums_l` and `split_y_sum_sqs_l` written for every node that took a
/// split; leaves are left untouched, matching what `accumulate_importance`
/// already skips.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn finalise_split_stats_rf(
    feature_data: &Tensor<u32>,
    y_dense: &Tensor<f32>,
    node_sample_offsets: &Tensor<u32>,
    node_sample_ids: &Tensor<u32>,
    node_sample_mults: &Tensor<u32>,
    split_feature: &Tensor<u32>,
    split_threshold: &Tensor<u32>,
    split_y_sums_l: &mut Tensor<f32>,
    split_y_sum_sqs_l: &mut Tensor<f32>,
    n_samples: u32,
    wave_size: u32,
    n_active_nodes: u32,
    n_targets: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let tree = CUBE_POS_Z;
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let mut s_tile_id = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_tile_mult = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut ys = Array::<f32>::new(RF_UNROLL as usize);
    let mut ms = Array::<u32>::new(RF_UNROLL as usize);

    let tx = UNIT_POS_X;
    let node_flat = (tree * n_active_nodes + node) as usize;
    let feat = split_feature[node_flat];
    // Uniform across the workgroup, so bailing here cannot strand a sync_cube.
    if feat == INVALID_NODE {
        terminate!();
    }

    let thr = split_threshold[node_flat];
    let obase = (tree * (n_active_nodes + 1u32)) as usize;
    let lo = node_sample_offsets[obase + node as usize];
    let hi = node_sample_offsets[obase + (node + 1u32) as usize];
    let gbase = (tree * n_samples) as usize;

    // Same staging and unrolling as build_score_rf_fused. The threshold test is
    // folded into the staged multiplicity: a sample right of the split gets
    // multiplicity 0, which keeps the branch out of the unrolled body.
    let mut acc_s: f32 = 0f32;
    let mut acc_q: f32 = 0f32;

    let mut base: u32 = lo;
    while base < hi {
        let idx = base + tx;
        if idx < hi {
            let s = node_sample_ids[gbase + idx as usize];
            s_tile_id[tx as usize] = s;
            if feature_bin(feature_data, feat * n_samples + s) <= thr {
                s_tile_mult[tx as usize] = node_sample_mults[gbase + idx as usize];
            } else {
                s_tile_mult[tx as usize] = 0u32;
            }
        }
        sync_cube();

        let mut cnt = hi - base;
        if cnt > CUBE_DIM_X {
            cnt = CUBE_DIM_X;
        }

        if tx < n_targets {
            let mut i: u32 = 0u32;
            while i + (RF_UNROLL - 1u32) < cnt {
                #[unroll]
                for u in 0..RF_UNROLL {
                    ms[u as usize] = s_tile_mult[(i + u) as usize];
                    ys[u as usize] =
                        y_dense[(s_tile_id[(i + u) as usize] * n_targets + tx) as usize];
                }
                #[unroll]
                for u in 0..RF_UNROLL {
                    let mf = f32::cast_from(ms[u as usize]);
                    acc_s += mf * ys[u as usize];
                    acc_q += mf * ys[u as usize] * ys[u as usize];
                }
                i += RF_UNROLL;
            }
            while i < cnt {
                let mf = f32::cast_from(s_tile_mult[i as usize]);
                let y = y_dense[(s_tile_id[i as usize] * n_targets + tx) as usize];
                acc_s += mf * y;
                acc_q += mf * y * y;
                i += 1u32;
            }
        }
        sync_cube();
        base += CUBE_DIM_X;
    }

    if tx < n_targets {
        let out = node_flat * n_targets as usize + tx as usize;
        split_y_sums_l[out] = acc_s;
        split_y_sum_sqs_l[out] = acc_q;
    }
}

//////////////////////////
// RandomForest, fused  //
//////////////////////////

// One kernel replacing the three DRAM-histogram kernels this path used to run
// (build, prefix sum, threshold sweep), which between them were 91% of
// RandomForest GPU time. The histogram lives in threadgroup memory and never
// reaches DRAM, the prefix sum is a register carry, and the threshold sweep
// reads the result in place.
//
// The shared-memory budget is bought by dropping the sum-of-squares histogram
// (see the score note on `build_score_rf_fused`) and coarsening the bin axis to
// whatever fits. Neither touches the shared `QuantisedStore`, still 256 u8 bins.

/// Finest bin count whose padded histogram fits `SMEM_HIST_SLOTS`.
///
/// The GPU is free to bin more coarsely than the shared `QuantisedStore`,
/// because the winning threshold is widened back into fine bin space before
/// anything downstream sees it. Coarsening only where the budget forces it
/// means small-target batches keep the full 256 bins for free: at 8 targets
/// this returns 256, at 32 it returns 128, at the production 64 it returns 64.
///
/// ### Params
///
/// * `n_targets` - Targets in the batch
///
/// ### Returns
///
/// A power-of-two bin count in `1..=N_BINS`.
pub fn pick_gpu_bins(n_targets: usize) -> u32 {
    let mut bins = N_BINS;
    while bins > 1 && (bins as usize) * (n_targets + 1) > SMEM_HIST_SLOTS as usize {
        bins /= 2;
    }
    bins
}

/// Shared-memory bytes [`build_score_rf_fused()`] asks for, independent of shape.
///
/// Everything it declares is comptime-sized, so the ask does not vary with
/// `n_targets` or the bin count.
///
/// ### Returns
///
/// Threadgroup-memory bytes one workgroup of the fused kernel allocates.
pub const fn fused_rf_smem_bytes() -> usize {
    (SMEM_HIST_SLOTS as usize * 4)
        + (N_BINS as usize * 4)
        // s_score, s_bin, s_nl, s_valid for the argmax; s_tile_{id,mult,bin} for
        // the sample staging.
        + (WORKGROUP_128 as usize * 7 * 4)
        + (2 * 4)
}

// 16352 is wgpu's `downlevel_defaults()` threadgroup-memory floor, the tightest
// budget any backend we ship on reports (plain `Limits::defaults()` gives
// 16384). The fused kernel currently asks for 13320, so this is not slack we
// can spend: SMEM_HIST_SLOTS only has to reach 2935, 1.35x its current value,
// to bust it. Busting it means real devices start failing `require_fused_rf`,
// and the DRAM histogram path they used to fall back to has been deleted, so
// raise the slot count only alongside a matching cut elsewhere in the kernel's
// shared memory.
const _: () = assert!(
    fused_rf_smem_bytes() <= 16_352,
    "build_score_rf_fused no longer fits the tightest shipping shared-memory \
     budget; shrink SMEM_HIST_SLOTS or N_BINS"
);

/// Check that this device can run [`build_score_rf_fused()`], the only
/// RandomForest split path on the GPU.
///
/// An oversized `SharedMemory` allocation is rejected when the pipeline is
/// compiled, and that happens on a cubecl background thread where the error
/// does not reach the caller. So the check runs host-side rather than finding
/// out by getting zeros back. Failing it is a hard stop: the module-level
/// assertion above keeps the ask under every shipping backend's floor, so a
/// device that still fails is far enough off the map that the honest answer is
/// the CPU entry point.
///
/// ### Params
///
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())` when the device's threadgroup-memory budget covers
/// [`fused_rf_smem_bytes`].
///
/// ### Errors
///
/// * `GpuScenicFusedRfUnavailable` when it does not.
fn require_fused_rf<R: Runtime>(client: &ComputeClient<R>) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let needed = fused_rf_smem_bytes();
    if needed > limits.max_shared_bytes {
        return Err(BixverseErrors::GpuScenicFusedRfUnavailable {
            bytes: needed,
            limit: limits.max_shared_bytes,
        });
    }
    Ok(())
}

/// Build a slot's histogram in shared memory, sweep every threshold, and emit
/// that slot's best split.
///
/// One workgroup per (slot, node, tree), `WORKGROUP_128` wide, in five phases:
/// zero the shared histogram, accumulate the node's gathered samples into it,
/// prefix-sum it along bins in place, score one candidate bin per thread, then
/// argmax across threads. Nothing reaches DRAM but the four small per-slot
/// outputs.
///
/// **Accumulation needs no atomics.** Every thread walks the *same* sample and
/// thread `k` owns target `k`, so no two threads ever touch the same histogram
/// column. That is what removes the ~1.4e9 CAS retry loops the DRAM path pays.
///
/// **The score drops the sum-of-squares histogram.** Expanding the objective,
/// `Σ_k (wl·vl_k + wr·vr_k) = Q/n − (1/n)(S_L/nl + S_R/nr)` with
/// `S_L = Σ_k syl_k²`, `S_R = Σ_k (sy_k − syl_k)²` and `Q = Σ_k ssq_k` a node
/// constant. So `argmax` over thresholds only needs `G = S_L/nl + S_R/nr`, and
/// `hist_y_sum_sqs` never has to exist. This is cuML's standard MSE
/// formulation.
///
/// Two consequences. The algebraic form does not clamp the per-target child
/// variances at zero the way the CPU does, so results differ wherever f32
/// cancellation drives one negative. And `best_score > 0` is an *acceptance*
/// gate on the CPU, not just a tie-break, so the winner's true score is
/// reconstructed as `P − Q/n + G/n` before that gate applies; `argmax G` alone
/// would accept zero-gain splits wherever no positive split exists.
///
/// Measured negative result: the scoring loop re-reads all `n_targets` node Y
/// totals per candidate bin, ~4000 apparently redundant loads per workgroup.
/// Staging them in shared memory first is worth nothing (10.53x to 10.60x,
/// inside the noise) because every thread wants the same address and the
/// hardware broadcasts it from cache. Occupancy here is shared-memory bound, so
/// do not spend any on it.
///
/// ### Params
///
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `y_dense` - Dense Y `[n_samples, n_targets]`, target contiguous
/// * `node_sample_offsets` - Node slice bounds `[wave_size, n_active_nodes + 1]`
/// * `node_sample_ids` - Gathered sample ids `[wave_size, n_samples]`
/// * `node_sample_mults` - Gathered multiplicities, same layout
/// * `node_features` - Selected feature ids per (tree, node, slot)
/// * `node_counts` - Per-node sample totals `[wave_size, n_active_nodes]`
/// * `node_y_sums` - Per-node Y sums `[wave_size, n_active_nodes, n_targets]`
/// * `node_y_sum_sqs` - Per-node Y sums-of-squares, same layout
/// * `slot_best_score` - Output reconstructed score per slot
/// * `slot_best_thr` - Output threshold per slot, widened to *fine* bin space
/// * `slot_best_n_left` - Output left-child count per slot
/// * `slot_best_valid` - Output 1 when the slot proposes a split, else 0
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_bins_gpu` - Coarsened bin count from [`pick_gpu_bins`]
/// * `bin_shift` - Right shift taking a fine bin to a coarse one
/// * `min_samples_leaf` - Minimum samples per child
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> feature slot
/// * `CUBE_POS_Y` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> target index in the accumulate and prefix phases, candidate
///   bin in the scoring phase
///
/// ### Returns
///
/// The four `slot_best_*` tensors written for every (tree, node, slot).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn build_score_rf_fused(
    feature_data: &Tensor<u32>,
    y_dense: &Tensor<f32>,
    node_sample_offsets: &Tensor<u32>,
    node_sample_ids: &Tensor<u32>,
    node_sample_mults: &Tensor<u32>,
    node_features: &Tensor<u32>,
    node_counts: &Tensor<u32>,
    node_y_sums: &Tensor<f32>,
    node_y_sum_sqs: &Tensor<f32>,
    slot_best_score: &mut Tensor<f32>,
    slot_best_thr: &mut Tensor<u32>,
    slot_best_n_left: &mut Tensor<u32>,
    slot_best_valid: &mut Tensor<u32>,
    n_samples: u32,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    n_bins_gpu: u32,
    bin_shift: u32,
    min_samples_leaf: u32,
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

    let mut s_hist = SharedMemory::<f32>::new(SMEM_HIST_SLOTS as usize);
    let mut s_cnt = SharedMemory::<u32>::new(N_BINS as usize);
    let mut s_score = SharedMemory::<f32>::new(WORKGROUP_128 as usize);
    let mut s_bin = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_nl = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_valid = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_pq = SharedMemory::<f32>::new(2usize);
    let mut s_tile_id = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_tile_mult = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_tile_bin = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    // Register-backed staging, so every load of the unrolled accumulate can be
    // in flight before the first store consumes one.
    let mut ys = Array::<f32>::new(RF_UNROLL as usize);
    let mut bs = Array::<u32>::new(RF_UNROLL as usize);
    let mut ms = Array::<u32>::new(RF_UNROLL as usize);

    let tx = UNIT_POS_X;
    let node_flat = (tree * n_active_nodes + node) as usize;
    let slot_flat = ((tree * n_active_nodes + node) * k_feats + slot) as usize;
    let stats_base = node_flat * n_targets as usize;
    let n = node_counts[node_flat];

    // Nodes that cannot split still have to leave a definite verdict behind, or
    // reduce_slot_winners reads whatever was in the buffer two levels ago.
    if n < 2u32 || n < 2u32 * min_samples_leaf {
        if tx == 0u32 {
            slot_best_score[slot_flat] = 0f32;
            slot_best_thr[slot_flat] = 0u32;
            slot_best_n_left[slot_flat] = 0u32;
            slot_best_valid[slot_flat] = 0u32;
        }
        terminate!();
    }

    // Rows are padded by one float so the scoring phase, where consecutive
    // threads own consecutive bins, does not put all of them on one bank.
    let stride = n_targets + 1u32;
    let feat = node_features[slot_flat];

    let hist_len = n_bins_gpu * stride;
    let mut z: u32 = tx;
    while z < hist_len {
        s_hist[z as usize] = 0f32;
        z += wg_size;
    }
    let mut zc: u32 = tx;
    while zc < n_bins_gpu {
        s_cnt[zc as usize] = 0u32;
        zc += wg_size;
    }
    sync_cube();

    // Accumulate. The whole workgroup walks one sample at a time; thread k adds
    // that sample's target k. Column ownership is what makes this atomic-free,
    // and why the loop runs over samples rather than striding them.
    //
    // Staged a tile at a time: a sample's id, multiplicity and bin are the same
    // for all 128 threads, so one cooperative load per tile replaces three
    // redundant global loads per thread. Only the y_dense fetch is genuinely
    // per-thread.
    let obase = (tree * (n_active_nodes + 1u32)) as usize;
    let lo = node_sample_offsets[obase + node as usize];
    let hi = node_sample_offsets[obase + (node + 1u32) as usize];
    let gbase = (tree * n_samples) as usize;

    let mut base: u32 = lo;
    while base < hi {
        let idx = base + tx;
        if idx < hi {
            let s = node_sample_ids[gbase + idx as usize];
            s_tile_id[tx as usize] = s;
            s_tile_mult[tx as usize] = node_sample_mults[gbase + idx as usize];
            s_tile_bin[tx as usize] = feature_bin(feature_data, feat * n_samples + s) >> bin_shift;
        }
        sync_cube();

        let mut cnt = hi - base;
        if cnt > CUBE_DIM_X {
            cnt = CUBE_DIM_X;
        }
        // Every y_dense load issued before the first is consumed. Repeated bins
        // within a group are safe: thread k owns column k, so they serialise
        // inside one thread rather than racing.
        let mut i: u32 = 0u32;
        while i + (RF_UNROLL - 1u32) < cnt {
            #[unroll]
            for u in 0..RF_UNROLL {
                bs[u as usize] = s_tile_bin[(i + u) as usize];
                ms[u as usize] = s_tile_mult[(i + u) as usize];
            }
            if tx < n_targets {
                #[unroll]
                for u in 0..RF_UNROLL {
                    ys[u as usize] =
                        y_dense[(s_tile_id[(i + u) as usize] * n_targets + tx) as usize];
                }
                #[unroll]
                for u in 0..RF_UNROLL {
                    s_hist[(bs[u as usize] * stride + tx) as usize] +=
                        f32::cast_from(ms[u as usize]) * ys[u as usize];
                }
            }
            if tx == 0u32 {
                #[unroll]
                for u in 0..RF_UNROLL {
                    s_cnt[bs[u as usize] as usize] += ms[u as usize];
                }
            }
            i += RF_UNROLL;
        }
        while i < cnt {
            let b = s_tile_bin[i as usize];
            if tx < n_targets {
                let mf = f32::cast_from(s_tile_mult[i as usize]);
                let y = y_dense[(s_tile_id[i as usize] * n_targets + tx) as usize];
                s_hist[(b * stride + tx) as usize] += mf * y;
            }
            if tx == 0u32 {
                s_cnt[b as usize] += s_tile_mult[i as usize];
            }
            i += 1u32;
        }
        sync_cube();
        base += CUBE_DIM_X;
    }

    // Inclusive prefix along bins, in place. Thread k owns column k throughout,
    // so the running total lives in a register and no barrier is needed.
    if tx < n_targets {
        let mut run: f32 = 0f32;
        let mut b: u32 = 0u32;
        while b < n_bins_gpu {
            run += s_hist[(b * stride + tx) as usize];
            s_hist[(b * stride + tx) as usize] = run;
            b += 1u32;
        }
    }
    if tx == 0u32 {
        let mut runc: u32 = 0u32;
        let mut b: u32 = 0u32;
        while b < n_bins_gpu {
            runc += s_cnt[b as usize];
            s_cnt[b as usize] = runc;
            b += 1u32;
        }

        // Node constants for the score reconstruction: the clamped parent
        // variance sum and the raw sum-of-squares total.
        let nf = f32::cast_from(n);
        let mut p: f32 = 0f32;
        let mut q: f32 = 0f32;
        let mut k: u32 = 0u32;
        while k < n_targets {
            let sy = node_y_sums[stats_base + k as usize];
            let ssq = node_y_sum_sqs[stats_base + k as usize];
            q += ssq;
            let mean = sy / nf;
            p += f32::max(ssq / nf - mean * mean, 0f32);
            k += 1u32;
        }
        s_pq[0] = p;
        s_pq[1] = q;
    }
    sync_cube();

    let parent_var = s_pq[0];
    let q_tot = s_pq[1];
    if parent_var <= 0f32 {
        if tx == 0u32 {
            slot_best_score[slot_flat] = 0f32;
            slot_best_thr[slot_flat] = 0u32;
            slot_best_n_left[slot_flat] = 0u32;
            slot_best_valid[slot_flat] = 0u32;
        }
        terminate!();
    }

    // Score. One thread per candidate bin; splitting at the last bin sends
    // everything left, so the sweep stops one short.
    let mut best_g: f32 = 0f32;
    let mut best_b: u32 = 0u32;
    let mut best_nl: u32 = 0u32;
    let mut best_valid: u32 = 0u32;

    let mut b: u32 = tx;
    while b + 1u32 < n_bins_gpu {
        let n_left = s_cnt[b as usize];
        let n_right = n - n_left;
        if n_left >= min_samples_leaf {
            if n_right >= min_samples_leaf {
                let nl = f32::cast_from(n_left);
                let nr = f32::cast_from(n_right);
                let mut sl: f32 = 0f32;
                let mut sr: f32 = 0f32;
                let mut k: u32 = 0u32;
                while k < n_targets {
                    let cl = s_hist[(b * stride + k) as usize];
                    let cr = node_y_sums[stats_base + k as usize] - cl;
                    sl += cl * cl;
                    sr += cr * cr;
                    k += 1u32;
                }
                let g = sl / nl + sr / nr;
                if best_valid == 0u32 || g > best_g {
                    best_g = g;
                    best_b = b;
                    best_nl = n_left;
                    best_valid = 1u32;
                }
            }
        }
        b += wg_size;
    }

    s_score[tx as usize] = best_g;
    s_bin[tx as usize] = best_b;
    s_nl[tx as usize] = best_nl;
    s_valid[tx as usize] = best_valid;
    sync_cube();

    let mut red: u32 = (wg_size / 2u32).runtime();
    while red > 0u32 {
        if tx < red {
            let mate = tx + red;
            let take = argmax_takes_mate(
                s_valid[tx as usize],
                s_score[tx as usize],
                s_valid[mate as usize],
                s_score[mate as usize],
            );
            if take == 1u32 {
                s_score[tx as usize] = s_score[mate as usize];
                s_bin[tx as usize] = s_bin[mate as usize];
                s_nl[tx as usize] = s_nl[mate as usize];
                s_valid[tx as usize] = s_valid[mate as usize];
            }
        }
        sync_cube();
        red /= 2u32;
    }

    if tx == 0u32 {
        let nf = f32::cast_from(n);
        // argmax on G is argmax on the real score, but the > 0 acceptance gate
        // needs the real thing back.
        let score = parent_var - q_tot / nf + s_score[0] / nf;
        if s_valid[0] == 1u32 && score > 0f32 {
            slot_best_score[slot_flat] = score;
            slot_best_thr[slot_flat] = (s_bin[0] << bin_shift) | ((1u32 << bin_shift) - 1u32);
            slot_best_n_left[slot_flat] = s_nl[0];
            slot_best_valid[slot_flat] = 1u32;
        } else {
            slot_best_score[slot_flat] = 0f32;
            slot_best_thr[slot_flat] = 0u32;
            slot_best_n_left[slot_flat] = 0u32;
            slot_best_valid[slot_flat] = 0u32;
        }
    }
}

/// Pick each node's winning slot from the per-slot proposals.
///
/// Scans slots ascending with a strict `>`, so ties go to the lowest slot,
/// matching how the CPU walks its shuffled feature buffer.
///
/// ### Params
///
/// * `slot_best_score` - Per-slot reconstructed scores
/// * `slot_best_thr` - Per-slot thresholds, fine bin space
/// * `slot_best_n_left` - Per-slot left-child counts
/// * `slot_best_valid` - Per-slot validity flags
/// * `node_features` - Selected feature ids per (tree, node, slot)
/// * `split_feature` - Output winning feature per node, `INVALID_NODE` for leaf
/// * `split_threshold` - Output winning threshold per node
/// * `split_n_left` - Output left-child count per node
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * Single thread per workgroup (`UNIT_POS_X == 0`)
///
/// ### Returns
///
/// `split_feature`, `split_threshold` and `split_n_left` written per node.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn reduce_slot_winners(
    slot_best_score: &Tensor<f32>,
    slot_best_thr: &Tensor<u32>,
    slot_best_n_left: &Tensor<u32>,
    slot_best_valid: &Tensor<u32>,
    node_features: &Tensor<u32>,
    split_feature: &mut Tensor<u32>,
    split_threshold: &mut Tensor<u32>,
    split_n_left: &mut Tensor<u32>,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
) {
    let node = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let tree = CUBE_POS_Z;
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }
    if UNIT_POS_X != 0u32 {
        terminate!();
    }

    let node_flat = (tree * n_active_nodes + node) as usize;
    let base = node_flat * k_feats as usize;

    let mut best_score: f32 = 0f32;
    let mut best_slot: u32 = 0u32;
    let mut found: u32 = 0u32;
    let mut s: u32 = 0u32;
    while s < k_feats {
        if slot_best_valid[base + s as usize] == 1u32 {
            let sc = slot_best_score[base + s as usize];
            if found == 0u32 || sc > best_score {
                best_score = sc;
                best_slot = s;
                found = 1u32;
            }
        }
        s += 1u32;
    }

    if found == 1u32 {
        split_feature[node_flat] = node_features[base + best_slot as usize];
        split_threshold[node_flat] = slot_best_thr[base + best_slot as usize];
        split_n_left[node_flat] = slot_best_n_left[base + best_slot as usize];
    } else {
        split_feature[node_flat] = INVALID_NODE;
        split_threshold[node_flat] = 0u32;
        split_n_left[node_flat] = 0u32;
    }
}

///////////////////////
// ExtraTrees direct //
///////////////////////

// The histogram pipeline is pure waste for ExtraTrees: it builds a full
// 256-bin x n_targets cumulative table per slot, then reads exactly one bin row
// out of it, because ExtraTrees draws `n_thresholds` random thresholds per
// (node, slot) and defaults to 1. Profiling put the histogram build at 69% and
// its prefix sum at 31% of ExtraTrees GPU time.
//
// So: reduce the node's bin range (scan_slot_bin_range), draw the threshold
// from it, and accumulate the left-side statistics at that single threshold
// (accumulate_split_stats_et). Per-level scratch drops from ~6.1 GiB to ~13 MB
// at the reference shape, and the accumulation moves off global CAS-loop float
// atomics (~1.4 billion per fit, which is where build_hist's time went) onto
// registers with a shared-memory reduction.
//
// Needs Y dense rather than CSR-by-sample, which makes every Y read coalesced
// across the target axis. At 64 targets that is ~2.5 MB at 10k cells.

/// Per-(tree, node, slot) min and max occupied bin, by reduction over the
/// node's samples.
///
/// The first and last bin holding at least one active sample. Empty slots emit
/// `min = max = 0`, which the evaluate kernels reject via `max > min`.
///
/// One workgroup per (slot, node, tree), `WORKGROUP_128` wide. Threads stride
/// over samples keeping a register min/max, then reduce through shared memory.
///
/// ### Params
///
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`
/// * `sample_multiplicity` - Per-tree sample multiplier `[wave_size, n_samples]`
/// * `node_features` - Selected feature ids `[wave_size, n_active_nodes, k_feats]`
/// * `node_counts` - Per-node sample totals; nodes at 0 are skipped
/// * `slot_min_bin` - Output first occupied bin per slot
/// * `slot_max_bin` - Output last occupied bin per slot
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> feature slot index
/// * `CUBE_POS_Y` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> sample stripe
///
/// ### Returns
///
/// `slot_min_bin` and `slot_max_bin` written per slot.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn scan_slot_bin_range(
    feature_data: &Tensor<u32>,
    sample_to_node: &Tensor<u32>,
    sample_multiplicity: &Tensor<u32>,
    node_features: &Tensor<u32>,
    node_counts: &Tensor<u32>,
    slot_min_bin: &mut Tensor<u32>,
    slot_max_bin: &mut Tensor<u32>,
    n_samples: u32,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
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

    let slot_flat = ((tree * n_active_nodes + node) * k_feats + slot) as usize;
    let tx = UNIT_POS_X;

    if node_counts[(tree * n_active_nodes + node) as usize] == 0u32 {
        if tx == 0u32 {
            slot_min_bin[slot_flat] = 0u32;
            slot_max_bin[slot_flat] = 0u32;
        }
        terminate!();
    }

    let feat = node_features[slot_flat];
    let s2n_base = (tree * n_samples) as usize;

    // N_BINS is the "nothing seen" sentinel for the min. The max seeds at 0,
    // which is safe: if the only occupied bin is 0 the slot ends up with
    // min == max and gets rejected downstream.
    // N_BINS is a host const; .runtime() lifts it so it can meet runtime
    // values in comparisons and assignments.
    let n_bins = N_BINS.runtime();
    let mut lo: u32 = n_bins;
    let mut hi: u32 = 0u32;
    let mut s: u32 = tx;
    while s < n_samples {
        if sample_to_node[s2n_base + s as usize] == node {
            if sample_multiplicity[s2n_base + s as usize] > 0u32 {
                let b = feature_bin(feature_data, feat * n_samples + s);
                if b < lo {
                    lo = b;
                }
                if b > hi {
                    hi = b;
                }
            }
        }
        s += wg_size;
    }

    let mut s_lo = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_hi = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    s_lo[tx as usize] = lo;
    s_hi[tx as usize] = hi;
    sync_cube();

    let mut stride: u32 = (wg_size / 2u32).runtime();
    while stride > 0u32 {
        if tx < stride {
            let a = s_lo[(tx + stride) as usize];
            if a < s_lo[tx as usize] {
                s_lo[tx as usize] = a;
            }
            let b = s_hi[(tx + stride) as usize];
            if b > s_hi[tx as usize] {
                s_hi[tx as usize] = b;
            }
        }
        sync_cube();
        stride /= 2u32;
    }

    if tx == 0u32 {
        if s_lo[0] >= n_bins {
            slot_min_bin[slot_flat] = 0u32;
            slot_max_bin[slot_flat] = 0u32;
        } else {
            slot_min_bin[slot_flat] = s_lo[0];
            slot_max_bin[slot_flat] = s_hi[0];
        }
    }
}

/// Draw the ExtraTrees threshold for each (tree, node, slot, threshold) and
/// accumulate the left-side statistics at it in one pass over the samples.
///
/// This is the kernel that replaces the histogram. Rather than binning every
/// sample into 256 buckets per target and prefix-summing, it evaluates the
/// single predicate `bin <= thr` and sums the matching samples' Y directly.
///
/// The threshold hash chain (`tree_seed ^ level ^ node ^ slot ^ ti`, salted
/// with `2654435769`) is the one the histogram path used, so the drawn
/// thresholds and hence the trees are unchanged.
///
/// Two decompositions, chosen at comptime by `use_plane`. The compacted path
/// tests one sample per thread per block and compacts survivors through a
/// plane-based scan; the fallback lays threads out as `(stripe, target)` and
/// re-reads the sample metadata once per target. Either way each thread keeps
/// its target's running sums in registers, so there are no atomics, and
/// consecutive threads read consecutive targets out of the dense Y.
///
/// ### Params
///
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `y_dense` - Dense Y `[n_samples, n_targets]`
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`
/// * `sample_multiplicity` - Per-tree sample multiplier `[wave_size, n_samples]`
/// * `node_features` - Selected feature ids `[wave_size, n_active_nodes, k_feats]`
/// * `node_counts` - Per-node sample totals; nodes at 0 are skipped
/// * `tree_seeds` - Per-tree base seed `[wave_size]`
/// * `slot_min_bin` - First occupied bin per slot from [`scan_slot_bin_range()`]
/// * `slot_max_bin` - Last occupied bin per slot, same source
/// * `cand_thr` - Output drawn threshold per candidate
/// * `cand_n_left` - Output left-side sample count per candidate
/// * `cand_y_sums_l` - Output left-side Y sums per (candidate, target)
/// * `cand_y_sum_sqs_l` - Output left-side Y sum-of-squares, same layout
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_thresholds` - Random thresholds drawn per feature slot
/// * `level` - Depth being processed
/// * `use_plane` - Take the plane-compacted path; see `plane_compact_viable`
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> candidate index `slot * n_thresholds + ti`
/// * `CUBE_POS_Y` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> `(stripe, target)` pair
///
/// ### Returns
///
/// `cand_thr`, `cand_n_left`, `cand_y_sums_l` and `cand_y_sum_sqs_l` written
/// per candidate. Rejected candidates (empty or single-valued slots) emit
/// `cand_n_left = 0`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn accumulate_split_stats_et(
    feature_data: &Tensor<u32>,
    y_dense: &Tensor<f32>,
    sample_to_node: &Tensor<u32>,
    sample_multiplicity: &Tensor<u32>,
    node_features: &Tensor<u32>,
    node_counts: &Tensor<u32>,
    tree_seeds: &Tensor<u32>,
    slot_min_bin: &Tensor<u32>,
    slot_max_bin: &Tensor<u32>,
    cand_thr: &mut Tensor<u32>,
    cand_n_left: &mut Tensor<u32>,
    cand_y_sums_l: &mut Tensor<f32>,
    cand_y_sum_sqs_l: &mut Tensor<f32>,
    n_samples: u32,
    wave_size: u32,
    n_active_nodes: u32,
    k_feats: u32,
    n_targets: u32,
    n_thresholds: u32,
    level: u32,
    #[comptime] use_plane: bool,
) {
    let cand = CUBE_POS_X;
    let node = CUBE_POS_Y;
    let tree = CUBE_POS_Z;
    let n_cands = k_feats * n_thresholds;
    if cand >= n_cands {
        terminate!();
    }
    if node >= n_active_nodes {
        terminate!();
    }
    if tree >= wave_size {
        terminate!();
    }

    let slot = cand / n_thresholds;
    let ti = cand % n_thresholds;
    let tx = UNIT_POS_X;
    let node_flat = (tree * n_active_nodes + node) as usize;
    let slot_flat = ((tree * n_active_nodes + node) * k_feats + slot) as usize;
    let cand_flat = ((tree * n_active_nodes + node) * n_cands + cand) as usize;

    let min_bin = slot_min_bin[slot_flat];
    let max_bin = slot_max_bin[slot_flat];

    if node_counts[node_flat] == 0u32 || max_bin <= min_bin {
        if tx == 0u32 {
            cand_thr[cand_flat] = 0u32;
            cand_n_left[cand_flat] = 0u32;
        }
        terminate!();
    }

    let mut seed_mix = tree_seeds[tree as usize];
    seed_mix = hash_mix(seed_mix ^ level);
    seed_mix = hash_mix(seed_mix ^ node);
    seed_mix = hash_mix(seed_mix ^ slot);
    seed_mix = hash_mix(seed_mix ^ ti);
    seed_mix = hash_mix(seed_mix ^ 2654435769u32);
    let thr = min_bin + (seed_mix % (max_bin - min_bin));

    let feat = node_features[slot_flat];
    let s2n_base = (tree * n_samples) as usize;

    let mut s_sum = SharedMemory::<f32>::new(WORKGROUP_128 as usize);
    let mut s_sq = SharedMemory::<f32>::new(WORKGROUP_128 as usize);
    let mut s_cnt = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_id = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_mult = SharedMemory::<u32>::new(WORKGROUP_128 as usize);
    let mut s_ptot = SharedMemory::<u32>::new(MAX_PLANES_PER_CUBE as usize);

    let mut acc_s: f32 = 0f32;
    let mut acc_q: f32 = 0f32;
    let mut acc_n: u32 = 0u32;

    if use_plane {
        // Sample metadata is read once per sample rather than once per
        // (sample, target), and survivors are compacted through a plane-based
        // exclusive scan so the drain loop runs over the match count rather
        // than the block width. That is what makes deep levels cheap: a node
        // holding ~100 samples matches once or twice per 128-wide block.
        let lane = UNIT_POS_PLANE;
        let plane_id = UNIT_POS_X / PLANE_DIM;
        let n_planes = CUBE_DIM_X / PLANE_DIM;
        let n_blocks = n_samples.div_ceil(CUBE_DIM_X);

        let mut blk: u32 = 0u32;
        while blk < n_blocks {
            let s = blk * CUBE_DIM_X + tx;
            let mut hit: u32 = 0u32;
            let mut mult: u32 = 0u32;
            if s < n_samples {
                if sample_to_node[s2n_base + s as usize] == node {
                    let m = sample_multiplicity[s2n_base + s as usize];
                    if m > 0u32 {
                        if feature_bin(feature_data, feat * n_samples + s) <= thr {
                            hit = 1u32;
                            mult = m;
                        }
                    }
                }
            }

            // Workgroup exclusive scan of `hit`: scan within the plane, then
            // offset by the totals of the planes below.
            let p_excl = plane_exclusive_sum(hit);
            let p_tot = plane_sum(hit);
            if lane == 0u32 {
                s_ptot[plane_id as usize] = p_tot;
            }
            sync_cube();

            let mut base: u32 = 0u32;
            let mut cnt: u32 = 0u32;
            let mut i: u32 = 0u32;
            while i < n_planes {
                let t = s_ptot[i as usize];
                if i < plane_id {
                    base += t;
                }
                cnt += t;
                i += 1u32;
            }

            if hit == 1u32 {
                s_id[(base + p_excl) as usize] = s;
                s_mult[(base + p_excl) as usize] = mult;
            }
            sync_cube();

            if tx < n_targets {
                let mut j: u32 = 0u32;
                while j < cnt {
                    let sid = s_id[j as usize];
                    let mf = f32::cast_from(s_mult[j as usize]);
                    let y = y_dense[(sid * n_targets + tx) as usize];
                    acc_s += mf * y;
                    acc_q += mf * y * y;
                    j += 1u32;
                }
            }
            if tx == 0u32 {
                let mut j: u32 = 0u32;
                while j < cnt {
                    acc_n += s_mult[j as usize];
                    j += 1u32;
                }
            }
            sync_cube();
            blk += 1u32;
        }

        if tx < n_targets {
            let out = cand_flat * n_targets as usize + tx as usize;
            cand_y_sums_l[out] = acc_s;
            cand_y_sum_sqs_l[out] = acc_q;
        }
        if tx == 0u32 {
            cand_thr[cand_flat] = thr;
            cand_n_left[cand_flat] = acc_n;
        }
    } else {
        // Portable fallback: (stripe, target) decomposition. Correct but
        // re-reads the sample metadata once per target.
        let n_stripes = (CUBE_DIM_X / n_targets).max(1u32);
        let stripe = tx / n_targets;
        let k = tx % n_targets;

        if stripe < n_stripes {
            let mut s: u32 = stripe;
            while s < n_samples {
                if sample_to_node[s2n_base + s as usize] == node {
                    let mult = sample_multiplicity[s2n_base + s as usize];
                    if mult > 0u32 {
                        if feature_bin(feature_data, feat * n_samples + s) <= thr {
                            let mf = f32::cast_from(mult);
                            let y = y_dense[(s * n_targets + k) as usize];
                            acc_s += mf * y;
                            acc_q += mf * y * y;
                            acc_n += mult;
                        }
                    }
                }
                s += n_stripes;
            }
        }

        s_sum[tx as usize] = acc_s;
        s_sq[tx as usize] = acc_q;
        s_cnt[tx as usize] = acc_n;
        sync_cube();

        // Fold the stripes. Only the first n_targets threads do the work; the
        // count is stripe-invariant across targets so lane k == 0 folds it.
        if tx < n_targets {
            let mut tot_s: f32 = 0f32;
            let mut tot_q: f32 = 0f32;
            let mut p: u32 = 0u32;
            while p < n_stripes {
                tot_s += s_sum[(p * n_targets + tx) as usize];
                tot_q += s_sq[(p * n_targets + tx) as usize];
                p += 1u32;
            }
            let out = cand_flat * n_targets as usize + tx as usize;
            cand_y_sums_l[out] = tot_s;
            cand_y_sum_sqs_l[out] = tot_q;
        }

        if tx == 0u32 {
            let mut tot_n: u32 = 0u32;
            let mut p: u32 = 0u32;
            while p < n_stripes {
                tot_n += s_cnt[(p * n_targets) as usize];
                p += 1u32;
            }
            cand_thr[cand_flat] = thr;
            cand_n_left[cand_flat] = tot_n;
        }
    }
}

/// Score the pre-accumulated ExtraTrees candidates and pick the winner.
///
/// Same variance-reduction objective and tie-break as the RandomForest sweep in
/// [`build_score_rf_fused()`], but reading per-candidate left-side statistics
/// from [`accumulate_split_stats_et()`] instead of a cumulative histogram.
/// One workgroup per (node, tree), `WORKGROUP_32` wide, reduced with the plane
/// argmax when the device allows it.
///
/// ### Params
///
/// * `node_counts` - Per-node sample totals
/// * `node_y_sums` - Per-node Y sums
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares
/// * `node_features` - Selected feature ids per (tree, node, slot)
/// * `cand_thr` - Drawn threshold per candidate
/// * `cand_n_left` - Left-side sample count per candidate
/// * `cand_y_sums_l` - Left-side Y sums per (candidate, target)
/// * `cand_y_sum_sqs_l` - Left-side Y sum-of-squares, same layout
/// * `split_feature` - Output winning feature id, or `u32::MAX`
/// * `split_threshold` - Output winning threshold bin
/// * `split_n_left` - Output left-child sample count
/// * `split_y_sums_l` - Output left-child Y sums
/// * `split_y_sum_sqs_l` - Output left-child Y sum-of-squares
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_thresholds` - Random thresholds drawn per feature slot
/// * `min_samples_leaf` - Minimum samples required on both sides
/// * `wg_size` - Workgroup width (comptime)
/// * `use_plane` - Take the plane argmax; see `plane_argmax_viable`
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> candidate stride offset
///
/// ### Returns
///
/// Winning split written into `split_feature`, `split_threshold` and
/// `split_n_left`; left-child stats into `split_y_sums_l` and
/// `split_y_sum_sqs_l`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if, unused_assignments)]
pub fn evaluate_splits_et_direct(
    node_counts: &Tensor<u32>,
    node_y_sums: &Tensor<f32>,
    node_y_sum_sqs: &Tensor<f32>,
    node_features: &Tensor<u32>,
    cand_thr: &Tensor<u32>,
    cand_n_left: &Tensor<u32>,
    cand_y_sums_l: &Tensor<f32>,
    cand_y_sum_sqs_l: &Tensor<f32>,
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
    #[comptime] wg_size: u32,
    #[comptime] use_plane: bool,
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
    let n_cands = k_feats * n_thresholds;

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

    let mut best_score: f32 = 0f32;
    let mut best_cand: u32 = 0u32;
    let mut best_n_left: u32 = 0u32;
    let mut best_valid: u32 = 0u32;

    let mut c: u32 = tx;
    while c < n_cands {
        let cand_flat = (node_flat as u32 * n_cands + c) as usize;
        let n_left = cand_n_left[cand_flat];
        let n_right = n - n_left;

        if n_left >= min_samples_leaf && n_right >= min_samples_leaf {
            let nl = f32::cast_from(n_left);
            let nr = f32::cast_from(n_right);
            let inv_nl = 1f32 / nl;
            let inv_nr = 1f32 / nr;
            let wl = nl / nf;
            let wr = nr / nf;

            let mut score: f32 = 0f32;
            let cbase = cand_flat * n_targets as usize;
            let mut k: u32 = 0u32;
            while k < n_targets {
                let sy = node_y_sums[stats_base + k as usize];
                let ssq = node_y_sum_sqs[stats_base + k as usize];
                let mean_p = sy / nf;
                let var_p = ssq / nf - mean_p * mean_p;

                let syl = cand_y_sums_l[cbase + k as usize];
                let ssyl = cand_y_sum_sqs_l[cbase + k as usize];
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
                best_cand = c;
                best_n_left = n_left;
                best_valid = 1u32;
            }
        }
        c += wg_size;
    }

    let mut s_score = SharedMemory::<f32>::new(WORKGROUP_32 as usize);
    let mut s_cand = SharedMemory::<u32>::new(WORKGROUP_32 as usize);
    let mut s_nl = SharedMemory::<u32>::new(WORKGROUP_32 as usize);
    let mut s_valid = SharedMemory::<u32>::new(WORKGROUP_32 as usize);

    let mut winner_valid: u32 = 0u32;
    let mut winner_cand: u32 = 0u32;
    let mut winner_n_left: u32 = 0u32;

    if use_plane {
        let winner_score = plane_max(best_score);
        let mut rank: u32 = PLANE_DIM;
        if best_valid == 1u32 && best_score == winner_score {
            rank = UNIT_POS_PLANE;
        }
        let winner_lane = plane_min(rank);
        let mut src: u32 = 0u32;
        if winner_lane < PLANE_DIM {
            src = winner_lane;
            winner_valid = 1u32;
        }
        winner_cand = plane_shuffle(best_cand, src);
        winner_n_left = plane_shuffle(best_n_left, src);
    } else {
        s_score[tx as usize] = best_score;
        s_cand[tx as usize] = best_cand;
        s_nl[tx as usize] = best_n_left;
        s_valid[tx as usize] = best_valid;
        sync_cube();

        let mut stride: u32 = (wg_size / 2u32).runtime();
        while stride > 0u32 {
            if tx < stride {
                let mate = tx + stride;
                let take = argmax_takes_mate(
                    s_valid[tx as usize],
                    s_score[tx as usize],
                    s_valid[mate as usize],
                    s_score[mate as usize],
                );
                if take == 1u32 {
                    s_score[tx as usize] = s_score[mate as usize];
                    s_cand[tx as usize] = s_cand[mate as usize];
                    s_nl[tx as usize] = s_nl[mate as usize];
                    s_valid[tx as usize] = s_valid[mate as usize];
                }
            }
            sync_cube();
            stride /= 2u32;
        }
        winner_valid = s_valid[0];
        winner_cand = s_cand[0];
        winner_n_left = s_nl[0];
    }

    if tx == 0u32 {
        if winner_valid == 1u32 {
            let winner_slot = winner_cand / n_thresholds;
            let feat =
                node_features[((tree * n_active_nodes + node) * k_feats + winner_slot) as usize];
            split_feature[node_flat] = feat;
            split_threshold[node_flat] =
                cand_thr[(node_flat as u32 * n_cands + winner_cand) as usize];
            split_n_left[node_flat] = winner_n_left;
        } else {
            split_feature[node_flat] = INVALID_NODE;
            split_threshold[node_flat] = 0u32;
            split_n_left[node_flat] = 0u32;
        }
    }

    if winner_valid == 1u32 {
        let cbase = (node_flat as u32 * n_cands + winner_cand) as usize * n_targets as usize;
        let mut k: u32 = tx;
        while k < n_targets {
            split_y_sums_l[stats_base + k as usize] = cand_y_sums_l[cbase + k as usize];
            split_y_sum_sqs_l[stats_base + k as usize] = cand_y_sum_sqs_l[cbase + k as usize];
            k += wg_size;
        }
    }
}

/// Seed the root node's sufficient statistics from the dense Y.
///
/// Neither split path materialises a DRAM histogram to fold, so the level-0
/// totals that [`propagate_child_stats()`] carries down the tree have to come
/// from the samples directly. One workgroup per tree, thread `tx` owning target
/// `tx`, so the Y reads coalesce. Runs once per wave.
///
/// ### Params
///
/// * `y_dense` - Dense Y `[n_samples, n_targets]`
/// * `sample_multiplicity` - Per-tree sample multiplier `[wave_size, n_samples]`
/// * `node_counts` - Output per-node sample totals; only node 0 is written
/// * `node_y_sums` - Output per-node Y sums; only node 0 is written
/// * `node_y_sum_sqs` - Output per-node Y sum-of-squares; only node 0
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level 0 (1, but sets the stride)
/// * `n_targets` - Targets in batch
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> tree_in_wave
/// * `UNIT_POS_X` -> target
///
/// ### Returns
///
/// Root totals written into `node_counts`, `node_y_sums` and
/// `node_y_sum_sqs`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn init_root_stats(
    y_dense: &Tensor<f32>,
    sample_multiplicity: &Tensor<u32>,
    node_counts: &mut Tensor<u32>,
    node_y_sums: &mut Tensor<f32>,
    node_y_sum_sqs: &mut Tensor<f32>,
    n_samples: u32,
    wave_size: u32,
    n_active_nodes: u32,
    n_targets: u32,
    #[comptime] wg_size: u32,
) {
    let tree = CUBE_POS_X;
    if tree >= wave_size {
        terminate!();
    }
    let tx = UNIT_POS_X;
    let s2n_base = (tree * n_samples) as usize;
    let node_flat = (tree * n_active_nodes) as usize;

    if tx == 0u32 {
        let mut tot: u32 = 0u32;
        let mut s: u32 = 0u32;
        while s < n_samples {
            tot += sample_multiplicity[s2n_base + s as usize];
            s += 1u32;
        }
        node_counts[node_flat] = tot;
    }

    let mut k: u32 = tx;
    while k < n_targets {
        let mut acc_s: f32 = 0f32;
        let mut acc_q: f32 = 0f32;
        let mut s: u32 = 0u32;
        while s < n_samples {
            let mult = sample_multiplicity[s2n_base + s as usize];
            if mult > 0u32 {
                let mf = f32::cast_from(mult);
                let y = y_dense[(s * n_targets + k) as usize];
                acc_s += mf * y;
                acc_q += mf * y * y;
            }
            s += 1u32;
        }
        node_y_sums[node_flat * n_targets as usize + k as usize] = acc_s;
        node_y_sum_sqs[node_flat * n_targets as usize + k as usize] = acc_q;
        k += wg_size;
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
/// * `left_child_id` - Left child id per node from [`compute_child_ids()`]
/// * `right_child_id` - Right child id per node from [`compute_child_ids()`]
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
    // viable_max_active_nodes counts splits rather than nodes at a level, so a
    // balanced enough tree can be handed a child id past the buffer. Drop those
    // samples rather than reading out of range.
    if node >= n_active_nodes {
        sample_to_node[s2n_idx] = INVALID_NODE;
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
    let bin = feature_bin(feature_data, feat * n_samples + s);
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
/// * `node_counts` - Per-node sample totals for this level
/// * `node_y_sums` - Per-node Y sums for this level
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares for this level
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
/// * `node_counts` - Per-node sample totals for this level
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

/// Zero the next level's node-stat slice so unwritten slots read as empty.
///
/// [`propagate_child_stats()`] only touches slots that an internal parent
/// actually spawned. Child ids are handed out compactly from 0 by
/// [`compute_child_ids()`], so everything above `2 * n_internal` is stale from
/// two levels ago and has to be cleared or it looks like a live node.
///
/// One workgroup per (node, tree), `WORKGROUP_128` wide. Thread 0 clears the
/// count; thread `tx` clears targets `tx, tx+wg, ...`.
///
/// ### Params
///
/// * `node_counts` - Per-node sample totals to clear `[wave_size,
///   n_active_nodes]`
/// * `node_y_sums` - Per-node Y sums to clear `[wave_size, n_active_nodes,
///   n_targets]`
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares to clear, same layout
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Nodes to clear at the next level
/// * `n_targets` - Targets in batch
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
/// `node_counts`, `node_y_sums` and `node_y_sum_sqs` zeroed in place over the
/// `n_active_nodes` prefix.
#[cube(launch_unchecked)]
pub fn zero_node_stats(
    node_counts: &mut Tensor<u32>,
    node_y_sums: &mut Tensor<f32>,
    node_y_sum_sqs: &mut Tensor<f32>,
    wave_size: u32,
    n_active_nodes: u32,
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
    let node_flat = (tree * n_active_nodes + node) as usize;
    if tx == 0u32 {
        node_counts[node_flat] = 0u32;
    }
    let stats_base = node_flat * n_targets as usize;
    let mut k: u32 = tx;
    while k < n_targets {
        node_y_sums[stats_base + k as usize] = 0f32;
        node_y_sum_sqs[stats_base + k as usize] = 0f32;
        k += wg_size;
    }
}

/// Hand each child node its sufficient statistics straight from the parent
/// split, so the next level never rebuilds them from its samples.
///
/// The left child's stats are exactly the winning split's left-side totals
/// that `evaluate_splits_*` already wrote; the right child's are the parent
/// minus the left. This is what the CPU recursion does (it threads
/// `left_y_sums` / `right_y_sums` down through `build_node_multi_sparse`), and
/// it keeps [`init_root_stats()`] a level-0-only cost: re-deriving numbers the
/// parent already holds would be the single largest wasted read in the level
/// loop.
///
/// One workgroup per (parent node, tree), `WORKGROUP_128` wide. Thread 0
/// writes both child counts; thread `tx` writes targets `tx, tx+wg, ...` for
/// both children.
///
/// Child ids at or above `n_active_next` are dropped rather than written, for
/// the same reason [`reassign_samples()`] drops out-of-range nodes. Those samples
/// are already dropped downstream; the guard just keeps the write in bounds.
///
/// ### Params
///
/// * `split_feature` - Winning feature id per parent; `INVALID_NODE` marks a
///   leaf and contributes nothing
/// * `split_n_left` - Left-child sample count per parent
/// * `split_y_sums_l` - Left-child Y sums per (parent, target)
/// * `split_y_sum_sqs_l` - Left-child Y sum-of-squares per (parent, target)
/// * `left_child_id` - Left child id per parent from [`compute_child_ids()`]
/// * `right_child_id` - Right child id per parent, same source
/// * `node_counts` - Current-level per-node sample totals (read)
/// * `node_y_sums` - Current-level per-node Y sums (read)
/// * `node_y_sum_sqs` - Current-level per-node Y sum-of-squares (read)
/// * `next_counts` - Next-level per-node sample totals (written)
/// * `next_y_sums` - Next-level per-node Y sums (written)
/// * `next_y_sum_sqs` - Next-level per-node Y sum-of-squares (written)
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at the current level
/// * `n_active_next` - Active nodes at the next level; bounds the writes
/// * `n_targets` - Targets in batch
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X` -> parent node index
/// * `CUBE_POS_Z` -> tree_in_wave
/// * `UNIT_POS_X` -> target stride offset
///
/// ### Returns
///
/// `next_counts`, `next_y_sums` and `next_y_sum_sqs` filled for every child
/// spawned by an internal parent.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn propagate_child_stats(
    split_feature: &Tensor<u32>,
    split_n_left: &Tensor<u32>,
    split_y_sums_l: &Tensor<f32>,
    split_y_sum_sqs_l: &Tensor<f32>,
    left_child_id: &Tensor<u32>,
    right_child_id: &Tensor<u32>,
    node_counts: &Tensor<u32>,
    node_y_sums: &Tensor<f32>,
    node_y_sum_sqs: &Tensor<f32>,
    next_counts: &mut Tensor<u32>,
    next_y_sums: &mut Tensor<f32>,
    next_y_sum_sqs: &mut Tensor<f32>,
    wave_size: u32,
    n_active_nodes: u32,
    n_active_next: u32,
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

    let node_flat = (tree * n_active_nodes + node) as usize;
    if split_feature[node_flat] == INVALID_NODE {
        terminate!();
    }

    let lid = left_child_id[node_flat];
    let rid = right_child_id[node_flat];
    if lid == INVALID_NODE {
        terminate!();
    }
    if lid >= n_active_next {
        terminate!();
    }
    if rid >= n_active_next {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let n = node_counts[node_flat];
    let n_left = split_n_left[node_flat];

    let l_flat = (tree * n_active_next + lid) as usize;
    let r_flat = (tree * n_active_next + rid) as usize;
    if tx == 0u32 {
        next_counts[l_flat] = n_left;
        next_counts[r_flat] = n - n_left;
    }

    let stats_base = node_flat * n_targets as usize;
    let l_base = l_flat * n_targets as usize;
    let r_base = r_flat * n_targets as usize;

    let mut k: u32 = tx;
    while k < n_targets {
        let sy = node_y_sums[stats_base + k as usize];
        let ssq = node_y_sum_sqs[stats_base + k as usize];
        let syl = split_y_sums_l[stats_base + k as usize];
        let ssyl = split_y_sum_sqs_l[stats_base + k as usize];

        next_y_sums[l_base + k as usize] = syl;
        next_y_sum_sqs[l_base + k as usize] = ssyl;
        next_y_sums[r_base + k as usize] = sy - syl;
        next_y_sum_sqs[r_base + k as usize] = ssq - ssyl;
        k += wg_size;
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
// Per-node gather //
/////////////////////

// Three kernels that turn `sample_to_node` into a per-node contiguous list, so
// the RandomForest histogram build can walk a node's own samples instead of
// testing all `n_samples` for membership. At depth that test is 158x wasteful:
// a node holding ~63 samples is found by scanning 10,000, once per feature slot
// per node per tree.
//
// The counter doubles as the scatter cursor. `scan_node_offsets` zeroes each
// count as it folds it into the offsets, `scatter_node_samples` bumps it back
// up to the total, and `zero_node_sample_counts` clears it again next level.

/// Zero the per-node sample counters ahead of [`count_node_samples()`].
///
/// ### Params
///
/// * `node_sample_counts` - Counters to clear `[wave_size, n_active_nodes]`
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `wg_size` - Workgroup width (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> tree_in_wave
/// * `UNIT_POS_X` -> node stride offset
///
/// ### Returns
///
/// `node_sample_counts` zeroed over the `n_active_nodes` prefix.
#[cube(launch_unchecked)]
pub fn zero_node_sample_counts(
    node_sample_counts: &mut Tensor<Atomic<u32>>,
    wave_size: u32,
    n_active_nodes: u32,
    #[comptime] wg_size: u32,
) {
    let tree = CUBE_POS_X;
    if tree >= wave_size {
        terminate!();
    }
    let base = (tree * n_active_nodes) as usize;
    let mut n: u32 = UNIT_POS_X;
    while n < n_active_nodes {
        Atomic::store(&node_sample_counts[base + n as usize], 0u32);
        n += wg_size;
    }
}

/// Count how many live samples each node holds.
///
/// A sample is live when its node is in range and its multiplicity is nonzero.
/// `INVALID_NODE` is `u32::MAX`, so the single `node < n_active_nodes` test
/// rejects unselected samples and out-of-range child ids together.
///
/// ### Params
///
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`
/// * `sample_multiplicity` - Per-tree sample multiplicity, same layout
/// * `node_sample_counts` - Output counters `[wave_size, n_active_nodes]`
/// * `n_samples` - Number of samples
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
/// `node_sample_counts` atomically incremented once per live sample.
#[cube(launch_unchecked)]
#[allow(clippy::collapsible_if)]
pub fn count_node_samples(
    sample_to_node: &Tensor<u32>,
    sample_multiplicity: &Tensor<u32>,
    node_sample_counts: &mut Tensor<Atomic<u32>>,
    n_samples: u32,
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
    let idx = (tree * n_samples + s) as usize;
    let node = sample_to_node[idx];
    if node < n_active_nodes {
        if sample_multiplicity[idx] > 0u32 {
            Atomic::fetch_add(
                &node_sample_counts[(tree * n_active_nodes + node) as usize],
                1u32,
            );
        }
    }
}

/// Exclusive prefix sum of the per-node counts, one thread per tree.
///
/// Also resets each count to zero on the way past, so
/// [`scatter_node_samples()`] can use it as a per-node write cursor.
///
/// Serial over at most `max_active_nodes` (1024) entries and launched once per
/// level, which puts it in the same class as the other node-bookkeeping
/// kernels: collectively 0.1% of GPU time.
///
/// ### Params
///
/// * `node_sample_counts` - Per-node counts, zeroed in place `[wave_size,
///   n_active_nodes]`
/// * `node_sample_offsets` - Output exclusive prefix `[wave_size,
///   n_active_nodes + 1]`
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
/// `node_sample_offsets` holds the node's slice bounds; `node_sample_counts`
/// is left zeroed.
#[cube(launch_unchecked)]
pub fn scan_node_offsets(
    node_sample_counts: &mut Tensor<Atomic<u32>>,
    node_sample_offsets: &mut Tensor<u32>,
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
    let cbase = (tree * n_active_nodes) as usize;
    let obase = (tree * (n_active_nodes + 1u32)) as usize;
    node_sample_offsets[obase] = 0u32;
    let mut run: u32 = 0u32;
    let mut n: u32 = 0u32;
    while n < n_active_nodes {
        let c = Atomic::load(&node_sample_counts[cbase + n as usize]);
        Atomic::store(&node_sample_counts[cbase + n as usize], 0u32);
        run += c;
        node_sample_offsets[obase + (n + 1u32) as usize] = run;
        n += 1u32;
    }
}

/// Scatter live sample ids into their node's slice.
///
/// Within a node the order is whatever the atomic cursor hands out, so it
/// varies run to run. That makes the f32 summation order in the consumer
/// nondeterministic, which is a property the pipeline already has via the
/// CAS-loop atomic in [`accumulate_importance()`].
///
/// ### Params
///
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`
/// * `sample_multiplicity` - Per-tree sample multiplicity, same layout
/// * `node_sample_offsets` - Node slice bounds from [`scan_node_offsets()`]
/// * `node_sample_counts` - Per-node write cursor, zeroed on entry
/// * `node_sample_ids` - Output sample ids `[wave_size, n_samples]`
/// * `node_sample_mults` - Output multiplicities, same layout
/// * `n_samples` - Number of samples
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
/// `node_sample_ids` and `node_sample_mults` filled over each node's slice.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_node_samples(
    sample_to_node: &Tensor<u32>,
    sample_multiplicity: &Tensor<u32>,
    node_sample_offsets: &Tensor<u32>,
    node_sample_counts: &mut Tensor<Atomic<u32>>,
    node_sample_ids: &mut Tensor<u32>,
    node_sample_mults: &mut Tensor<u32>,
    n_samples: u32,
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
    let idx = (tree * n_samples + s) as usize;
    let node = sample_to_node[idx];
    if node < n_active_nodes {
        let mult = sample_multiplicity[idx];
        if mult > 0u32 {
            let slot = Atomic::fetch_add(
                &node_sample_counts[(tree * n_active_nodes + node) as usize],
                1u32,
            );
            let pos = node_sample_offsets[(tree * (n_active_nodes + 1u32) + node) as usize] + slot;
            let out = (tree * n_samples + pos) as usize;
            node_sample_ids[out] = s;
            node_sample_mults[out] = mult;
        }
    }
}

/////////////////////
// Launch wrappers //
/////////////////////

/// Dispatch [`sample_node_features()`] over `(n_active_nodes, 1, wave_size)`
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
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
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
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let count = checked_cube_count(
        "sample_node_features",
        n_active_nodes as u32,
        1,
        wave_size as u32,
        &limits,
    )?;

    unsafe {
        sample_node_features::launch_unchecked::<R>(
            client,
            count,
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

    Ok(())
}

/// Dispatch the four per-node gather kernels in order.
///
/// They are always run as a unit and each depends on the previous, so they get
/// one wrapper rather than four. Cheap by construction: two sample-wide passes
/// and one serial scan over at most `max_active_nodes` entries per tree.
///
/// ### Workgroup
///
/// `WORKGROUP_128` for the zero and the two sample passes; the offset scan is
/// one thread per tree, matching [`compute_child_ids()`].
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `sample_to_node` - Per-tree sample assignment `[wave_size, n_samples]`
/// * `sample_multiplicity` - Per-tree sample multiplier, same layout
/// * `node_sample_counts` - Scratch counters `[wave_size, n_active_nodes]`
/// * `node_sample_offsets` - Output slice bounds `[wave_size, n_active_nodes + 1]`
/// * `node_sample_ids` - Output sample ids `[wave_size, n_samples]`
/// * `node_sample_mults` - Output multiplicities, same layout
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
///
/// ### Returns
///
/// `Ok(())`. Kernels are dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
#[allow(clippy::too_many_arguments)]
fn launch_gather_node_samples<R: Runtime>(
    client: &ComputeClient<R>,
    sample_to_node: &GpuTensor<R, u32>,
    sample_multiplicity: &GpuTensor<R, u32>,
    node_sample_counts: &GpuTensor<R, u32>,
    node_sample_offsets: &GpuTensor<R, u32>,
    node_sample_ids: &GpuTensor<R, u32>,
    node_sample_mults: &GpuTensor<R, u32>,
    n_samples: usize,
    wave_size: usize,
    n_active_nodes: usize,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let n_wgs = (n_samples as u32).div_ceil(WORKGROUP_128);
    let (gx, gy) = grid_2d(n_wgs.max(1), &limits)?;
    unsafe {
        zero_node_sample_counts::launch_unchecked::<R>(
            client,
            CubeCount::Static(wave_size as u32, 1, 1),
            CubeDim::new_1d(WORKGROUP_128),
            node_sample_counts.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            WORKGROUP_128,
        );
        count_node_samples::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            sample_to_node.clone().into_tensor_arg(),
            sample_multiplicity.clone().into_tensor_arg(),
            node_sample_counts.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
        );
        scan_node_offsets::launch_unchecked::<R>(
            client,
            CubeCount::Static(wave_size as u32, 1, 1),
            CubeDim::new_1d(1),
            node_sample_counts.clone().into_tensor_arg(),
            node_sample_offsets.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
        );
        scatter_node_samples::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            sample_to_node.clone().into_tensor_arg(),
            sample_multiplicity.clone().into_tensor_arg(),
            node_sample_offsets.clone().into_tensor_arg(),
            node_sample_counts.clone().into_tensor_arg(),
            node_sample_ids.clone().into_tensor_arg(),
            node_sample_mults.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
        );
    }

    Ok(())
}

/// Dispatch [`build_score_rf_fused()`] then [`reduce_slot_winners()`].
///
/// Together these are the whole RandomForest split step. The bin count is
/// chosen from `n_targets` by [`pick_gpu_bins`]; the caller must have checked
/// `require_fused_rf` first.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` for both. The fused kernel uses `tx` as a
/// target index while accumulating and as a candidate bin while scoring; the
/// reducer is single-threaded per node.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `y_dense` - Dense Y `[n_samples, n_targets]`
/// * `node_sample_offsets` - Node slice bounds
/// * `node_sample_ids` - Gathered sample ids
/// * `node_sample_mults` - Gathered multiplicities
/// * `node_features` - Selected feature ids per (tree, node, slot)
/// * `node_counts` - Per-node sample totals
/// * `node_y_sums` - Per-node Y sums
/// * `node_y_sum_sqs` - Per-node Y sums-of-squares
/// * `slot_best_score` - Scratch per-slot scores
/// * `slot_best_thr` - Scratch per-slot thresholds
/// * `slot_best_n_left` - Scratch per-slot left counts
/// * `slot_best_valid` - Scratch per-slot validity flags
/// * `split_feature` - Output winning feature per node
/// * `split_threshold` - Output winning threshold per node
/// * `split_n_left` - Output left-child count per node
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `min_samples_leaf` - Minimum samples per child
///
/// ### Returns
///
/// `Ok(())`. Kernels are dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
#[allow(clippy::too_many_arguments)]
fn launch_rf_fused<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data: &GpuTensor<R, u32>,
    y_dense: &GpuTensor<R, f32>,
    node_sample_offsets: &GpuTensor<R, u32>,
    node_sample_ids: &GpuTensor<R, u32>,
    node_sample_mults: &GpuTensor<R, u32>,
    node_features: &GpuTensor<R, u32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    slot_best_score: &GpuTensor<R, f32>,
    slot_best_thr: &GpuTensor<R, u32>,
    slot_best_n_left: &GpuTensor<R, u32>,
    slot_best_valid: &GpuTensor<R, u32>,
    split_feature: &GpuTensor<R, u32>,
    split_threshold: &GpuTensor<R, u32>,
    split_n_left: &GpuTensor<R, u32>,
    n_samples: usize,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    min_samples_leaf: usize,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let count = checked_cube_count(
        "build_score_rf_fused",
        k_feats as u32,
        n_active_nodes as u32,
        wave_size as u32,
        &limits,
    )?;
    let n_bins_gpu = pick_gpu_bins(n_targets);
    let bin_shift = (N_BINS / n_bins_gpu).trailing_zeros();
    // pick_gpu_bins guarantees this, but the kernel indexes shared memory with
    // runtime values against a comptime allocation, so an over-wide batch would
    // write past s_hist with no error anywhere. Cheap to state, and the margin
    // is only 2176 against 2080 at the production shape.
    debug_assert!(
        n_bins_gpu * (n_targets as u32 + 1) <= SMEM_HIST_SLOTS,
        "fused RF histogram wants {} slots of {SMEM_HIST_SLOTS}",
        n_bins_gpu * (n_targets as u32 + 1)
    );
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1), &limits)?;
    unsafe {
        build_score_rf_fused::launch_unchecked::<R>(
            client,
            count,
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            y_dense.clone().into_tensor_arg(),
            node_sample_offsets.clone().into_tensor_arg(),
            node_sample_ids.clone().into_tensor_arg(),
            node_sample_mults.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            slot_best_score.clone().into_tensor_arg(),
            slot_best_thr.clone().into_tensor_arg(),
            slot_best_n_left.clone().into_tensor_arg(),
            slot_best_valid.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            n_bins_gpu,
            bin_shift,
            min_samples_leaf as u32,
            WORKGROUP_128,
        );
        reduce_slot_winners::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(1),
            slot_best_score.clone().into_tensor_arg(),
            slot_best_thr.clone().into_tensor_arg(),
            slot_best_n_left.clone().into_tensor_arg(),
            slot_best_valid.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_threshold.clone().into_tensor_arg(),
            split_n_left.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
        );
    }

    Ok(())
}

/// Dispatch [`finalise_split_stats_rf()`] over `grid_2d(n_active_nodes)` by
/// `wave_size` workgroups.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — thread `tx` owns target `tx`, so the
/// dense-Y loads coalesce. Threads past `n_targets` exit.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `feature_data` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `y_dense` - Dense Y `[n_samples, n_targets]`
/// * `node_sample_offsets` - Node slice bounds
/// * `node_sample_ids` - Gathered sample ids
/// * `node_sample_mults` - Gathered multiplicities
/// * `split_feature` - Winning feature per node
/// * `split_threshold` - Winning threshold per node, fine bin space
/// * `split_y_sums_l` - Output left-child Y sums
/// * `split_y_sum_sqs_l` - Output left-child Y sums-of-squares
/// * `n_samples` - Number of samples
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at level
/// * `n_targets` - Targets in batch
///
/// ### Returns
///
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
#[allow(clippy::too_many_arguments)]
fn launch_finalise_split_stats_rf<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data: &GpuTensor<R, u32>,
    y_dense: &GpuTensor<R, f32>,
    node_sample_offsets: &GpuTensor<R, u32>,
    node_sample_ids: &GpuTensor<R, u32>,
    node_sample_mults: &GpuTensor<R, u32>,
    split_feature: &GpuTensor<R, u32>,
    split_threshold: &GpuTensor<R, u32>,
    split_y_sums_l: &GpuTensor<R, f32>,
    split_y_sum_sqs_l: &GpuTensor<R, f32>,
    n_samples: usize,
    wave_size: usize,
    n_active_nodes: usize,
    n_targets: usize,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1), &limits)?;
    unsafe {
        finalise_split_stats_rf::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            y_dense.clone().into_tensor_arg(),
            node_sample_offsets.clone().into_tensor_arg(),
            node_sample_ids.clone().into_tensor_arg(),
            node_sample_mults.clone().into_tensor_arg(),
            split_feature.clone().into_tensor_arg(),
            split_threshold.clone().into_tensor_arg(),
            split_y_sums_l.clone().into_tensor_arg(),
            split_y_sum_sqs_l.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
            n_targets as u32,
        );
    }

    Ok(())
}

/// Whether the plane-compacted path in [`accumulate_split_stats_et()`] can be
/// used on this device.
///
/// It derives a plane id from `UNIT_POS_X / PLANE_DIM`, so the device must
/// report an exact plane size rather than a range, and that size must divide
/// the workgroup width so no plane is partially populated. The resulting plane
/// count also has to fit [`MAX_PLANES_PER_CUBE`].
///
/// ### Params
///
/// * `wg_size` - Workgroup width the kernel will be launched at
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `true` to take the compacted path, `false` for the portable fallback.
fn plane_compact_viable(wg_size: u32, limits: &GpuLimits) -> bool {
    plane_partitions(wg_size, limits).is_some_and(|n_planes| n_planes <= MAX_PLANES_PER_CUBE)
}

/// Whether a `wg_size`-wide workgroup is guaranteed to be exactly one plane on
/// this device, which is the precondition for the plane-primitive argmax in
/// [`evaluate_splits_et_direct()`].
///
/// Both the min and max reported plane size must equal the workgroup width. If
/// the device reports a range (some backends do), a workgroup could straddle
/// two planes and a bare `plane_max` would silently reduce over only part of
/// the candidate set. Apple Silicon reports 32/32, so the plane path is taken
/// there.
///
/// ### Params
///
/// * `wg_size` - Workgroup width the kernel will be launched at
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// `true` when the plane path is safe, `false` to take the SMEM fallback.
fn plane_argmax_viable(wg_size: u32, limits: &GpuLimits) -> bool {
    plane_uniform(wg_size, limits)
}

/// Dispatch [`reassign_samples()`] over `(gx, gy, wave_size)` workgroups, where
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
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
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
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let n_wgs = (n_samples as u32).div_ceil(WORKGROUP_128);
    let (gx, gy) = grid_2d(n_wgs.max(1), &limits)?;
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

    Ok(())
}

/// Dispatch [`accumulate_importance()`] over `(gx, gy, wave_size)` workgroups,
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
/// * `node_counts` - Per-node sample totals for this level
/// * `node_y_sums` - Per-node Y sums for this level
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares for this level
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
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
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
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1), &limits)?;
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

    Ok(())
}

/// Dispatch [`compute_child_ids()`] over `(wave_size, 1, 1)` workgroups, one
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
/// * `node_counts` - Per-node sample totals for this level
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

/// Dispatch [`scan_slot_bin_range()`] over `(k_feats, n_active_nodes, wave_size)`
/// workgroups of `WORKGROUP_128`.
///
/// ### Params
///
/// See [`scan_slot_bin_range()`]; `client` is the CubeCL compute client.
///
/// ### Returns
///
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
#[allow(clippy::too_many_arguments)]
fn launch_scan_slot_bin_range<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data: &GpuTensor<R, u32>,
    sample_to_node: &GpuTensor<R, u32>,
    sample_multiplicity: &GpuTensor<R, u32>,
    node_features: &GpuTensor<R, u32>,
    node_counts: &GpuTensor<R, u32>,
    slot_min_bin: &GpuTensor<R, u32>,
    slot_max_bin: &GpuTensor<R, u32>,
    n_samples: usize,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let count = checked_cube_count(
        "scan_slot_bin_range",
        k_feats as u32,
        n_active_nodes as u32,
        wave_size as u32,
        &limits,
    )?;
    unsafe {
        scan_slot_bin_range::launch_unchecked::<R>(
            client,
            count,
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            sample_to_node.clone().into_tensor_arg(),
            sample_multiplicity.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            slot_min_bin.clone().into_tensor_arg(),
            slot_max_bin.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            WORKGROUP_128,
        );
    }

    Ok(())
}

/// Dispatch [`accumulate_split_stats_et()`] over
/// `(k_feats * n_thresholds, n_active_nodes, wave_size)` workgroups of
/// `WORKGROUP_128`.
///
/// ### Params
///
/// See [`accumulate_split_stats_et()`]; `client` is the CubeCL compute client.
///
/// ### Returns
///
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
#[allow(clippy::too_many_arguments)]
fn launch_accumulate_split_stats_et<R: Runtime>(
    client: &ComputeClient<R>,
    feature_data: &GpuTensor<R, u32>,
    y_dense: &GpuTensor<R, f32>,
    sample_to_node: &GpuTensor<R, u32>,
    sample_multiplicity: &GpuTensor<R, u32>,
    node_features: &GpuTensor<R, u32>,
    node_counts: &GpuTensor<R, u32>,
    tree_seeds: &GpuTensor<R, u32>,
    slot_min_bin: &GpuTensor<R, u32>,
    slot_max_bin: &GpuTensor<R, u32>,
    cand_thr: &GpuTensor<R, u32>,
    cand_n_left: &GpuTensor<R, u32>,
    cand_y_sums_l: &GpuTensor<R, f32>,
    cand_y_sum_sqs_l: &GpuTensor<R, f32>,
    n_samples: usize,
    wave_size: usize,
    n_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    n_thresholds: usize,
    level: u32,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let count = checked_cube_count(
        "accumulate_split_stats_et",
        (k_feats * n_thresholds) as u32,
        n_active_nodes as u32,
        wave_size as u32,
        &limits,
    )?;

    unsafe {
        accumulate_split_stats_et::launch_unchecked::<R>(
            client,
            count,
            CubeDim::new_1d(WORKGROUP_128),
            feature_data.clone().into_tensor_arg(),
            y_dense.clone().into_tensor_arg(),
            sample_to_node.clone().into_tensor_arg(),
            sample_multiplicity.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            tree_seeds.clone().into_tensor_arg(),
            slot_min_bin.clone().into_tensor_arg(),
            slot_max_bin.clone().into_tensor_arg(),
            cand_thr.clone().into_tensor_arg(),
            cand_n_left.clone().into_tensor_arg(),
            cand_y_sums_l.clone().into_tensor_arg(),
            cand_y_sum_sqs_l.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
            k_feats as u32,
            n_targets as u32,
            n_thresholds as u32,
            level,
            plane_compact_viable(WORKGROUP_128, &limits),
        );
    }

    Ok(())
}

/// Dispatch [`evaluate_splits_et_direct()`] over `(gx, gy, wave_size)`
/// workgroups of `WORKGROUP_32`.
///
/// ### Params
///
/// See [`evaluate_splits_et_direct()`]; `client` is the CubeCL compute client.
///
/// ### Returns
///
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
#[allow(clippy::too_many_arguments)]
fn launch_evaluate_splits_et_direct<R: Runtime>(
    client: &ComputeClient<R>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    node_features: &GpuTensor<R, u32>,
    cand_thr: &GpuTensor<R, u32>,
    cand_n_left: &GpuTensor<R, u32>,
    cand_y_sums_l: &GpuTensor<R, f32>,
    cand_y_sum_sqs_l: &GpuTensor<R, f32>,
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
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1), &limits)?;
    unsafe {
        evaluate_splits_et_direct::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_32),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            node_features.clone().into_tensor_arg(),
            cand_thr.clone().into_tensor_arg(),
            cand_n_left.clone().into_tensor_arg(),
            cand_y_sums_l.clone().into_tensor_arg(),
            cand_y_sum_sqs_l.clone().into_tensor_arg(),
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
            WORKGROUP_32,
            plane_argmax_viable(WORKGROUP_32, &limits),
        );
    }

    Ok(())
}

/// Dispatch [`init_root_stats()`] over `(wave_size, 1, 1)` workgroups of
/// `WORKGROUP_128`.
///
/// ### Params
///
/// See [`init_root_stats()`]; `client` is the CubeCL compute client.
///
/// ### Returns
///
/// `()` — kernel is dispatched asynchronously via the client command queue.
#[allow(clippy::too_many_arguments)]
fn launch_init_root_stats<R: Runtime>(
    client: &ComputeClient<R>,
    y_dense: &GpuTensor<R, f32>,
    sample_multiplicity: &GpuTensor<R, u32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    n_samples: usize,
    wave_size: usize,
    n_active_nodes: usize,
    n_targets: usize,
) {
    unsafe {
        init_root_stats::launch_unchecked::<R>(
            client,
            CubeCount::Static(wave_size as u32, 1, 1),
            CubeDim::new_1d(WORKGROUP_128),
            y_dense.clone().into_tensor_arg(),
            sample_multiplicity.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            n_samples as u32,
            wave_size as u32,
            n_active_nodes as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch [`zero_node_stats()`] over `(gx, gy, wave_size)` workgroups, where
/// `(gx, gy)` is the `grid_2d` decomposition of `n_active_nodes`.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup; thread 0
/// clears the count, threads `tx` clear targets `tx, tx+128, ...`.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `node_counts` - Per-node sample totals to clear
/// * `node_y_sums` - Per-node Y sums to clear
/// * `node_y_sum_sqs` - Per-node Y sum-of-squares to clear
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Nodes to clear
/// * `n_targets` - Targets in batch
///
/// ### Returns
///
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
fn launch_zero_node_stats<R: Runtime>(
    client: &ComputeClient<R>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    wave_size: usize,
    n_active_nodes: usize,
    n_targets: usize,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1), &limits)?;
    unsafe {
        zero_node_stats::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }

    Ok(())
}

/// Dispatch [`propagate_child_stats()`] over `(gx, gy, wave_size)` workgroups,
/// where `(gx, gy)` is the `grid_2d` decomposition of `n_active_nodes`.
///
/// ### Workgroup
///
/// `CubeDim::new_1d(WORKGROUP_128)` — 128 threads per workgroup; thread 0
/// writes both child counts, threads `tx` write targets `tx, tx+128, ...`.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `split_feature` - Winning feature id per parent
/// * `split_n_left` - Left-child sample count per parent
/// * `split_y_sums_l` - Left-child Y sums per (parent, target)
/// * `split_y_sum_sqs_l` - Left-child Y sum-of-squares per (parent, target)
/// * `left_child_id` - Left child id per parent
/// * `right_child_id` - Right child id per parent
/// * `node_counts` - Current-level per-node sample totals
/// * `node_y_sums` - Current-level per-node Y sums
/// * `node_y_sum_sqs` - Current-level per-node Y sum-of-squares
/// * `next_counts` - Next-level per-node sample totals (written)
/// * `next_y_sums` - Next-level per-node Y sums (written)
/// * `next_y_sum_sqs` - Next-level per-node Y sum-of-squares (written)
/// * `wave_size` - Trees in wave
/// * `n_active_nodes` - Active nodes at the current level
/// * `n_active_next` - Active nodes at the next level
/// * `n_targets` - Targets in batch
///
/// ### Returns
///
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
#[allow(clippy::too_many_arguments)]
fn launch_propagate_child_stats<R: Runtime>(
    client: &ComputeClient<R>,
    split_feature: &GpuTensor<R, u32>,
    split_n_left: &GpuTensor<R, u32>,
    split_y_sums_l: &GpuTensor<R, f32>,
    split_y_sum_sqs_l: &GpuTensor<R, f32>,
    left_child_id: &GpuTensor<R, u32>,
    right_child_id: &GpuTensor<R, u32>,
    node_counts: &GpuTensor<R, u32>,
    node_y_sums: &GpuTensor<R, f32>,
    node_y_sum_sqs: &GpuTensor<R, f32>,
    next_counts: &GpuTensor<R, u32>,
    next_y_sums: &GpuTensor<R, f32>,
    next_y_sum_sqs: &GpuTensor<R, f32>,
    wave_size: usize,
    n_active_nodes: usize,
    n_active_next: usize,
    n_targets: usize,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((n_active_nodes as u32).max(1), &limits)?;
    unsafe {
        propagate_child_stats::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, wave_size as u32),
            CubeDim::new_1d(WORKGROUP_128),
            split_feature.clone().into_tensor_arg(),
            split_n_left.clone().into_tensor_arg(),
            split_y_sums_l.clone().into_tensor_arg(),
            split_y_sum_sqs_l.clone().into_tensor_arg(),
            left_child_id.clone().into_tensor_arg(),
            right_child_id.clone().into_tensor_arg(),
            node_counts.clone().into_tensor_arg(),
            node_y_sums.clone().into_tensor_arg(),
            node_y_sum_sqs.clone().into_tensor_arg(),
            next_counts.clone().into_tensor_arg(),
            next_y_sums.clone().into_tensor_arg(),
            next_y_sum_sqs.clone().into_tensor_arg(),
            wave_size as u32,
            n_active_nodes as u32,
            n_active_next as u32,
            n_targets as u32,
            WORKGROUP_128,
        );
    }

    Ok(())
}

/// Dispatch [`init_sample_to_node()`] over `(gx, gy, wave_size)` workgroups,
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
/// `Ok(())`. The kernel is dispatched asynchronously via the client command
/// queue, so a successful return means the launch was accepted, not that
/// it has run. `CubeclUtils` when the dispatch geometry busts a device
/// limit, which is checked on the host before the launch.
fn launch_init_sample_to_node<R: Runtime>(
    client: &ComputeClient<R>,
    sample_multiplicity: &GpuTensor<R, u32>,
    sample_to_node: &GpuTensor<R, u32>,
    n_samples: usize,
    wave_size: usize,
) -> Result<(), BixverseErrors> {
    let limits = GpuLimits::from_client(client);
    let n_wgs = (n_samples as u32).div_ceil(WORKGROUP_128);
    let (gx, gy) = grid_2d(n_wgs.max(1), &limits)?;
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

    Ok(())
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
    /// First informative bin per slot `[wave_size, max_active_nodes, k_feats]`.
    /// Written by `scan_slot_bin_range`; lets the consumers skip out-of-range
    /// thresholds without a per-candidate 256-bin rescan.
    slot_min_bin: GpuTensor<R, u32>,
    /// Last informative bin per slot, same layout as `slot_min_bin`.
    slot_max_bin: GpuTensor<R, u32>,
    /// Scratch counters for the per-node gather `[wave_size,
    /// max_active_nodes]`. Doubles as the scatter write cursor. RandomForest
    /// only; stubbed to length 1 on the ExtraTrees path.
    node_sample_counts: GpuTensor<R, u32>,
    /// Per-node slice bounds into `node_sample_ids` `[wave_size,
    /// max_active_nodes + 1]`. RandomForest only.
    node_sample_offsets: GpuTensor<R, u32>,
    /// Sample ids grouped by owning node `[wave_size, n_samples]`. Only the
    /// live prefix of each node's slice is meaningful. RandomForest only.
    node_sample_ids: GpuTensor<R, u32>,
    /// Multiplicities parallel to `node_sample_ids`. RandomForest only.
    node_sample_mults: GpuTensor<R, u32>,
    /// Per-slot best score from `build_score_rf_fused` `[wave_size,
    /// max_active_nodes, k_feats]`. Fused RandomForest path only.
    slot_best_score: GpuTensor<R, f32>,
    /// Per-slot best threshold in fine bin space, same layout.
    slot_best_thr: GpuTensor<R, u32>,
    /// Per-slot left-child count, same layout.
    slot_best_n_left: GpuTensor<R, u32>,
    /// Per-slot validity flag, same layout.
    slot_best_valid: GpuTensor<R, u32>,
    /// Per-node sample totals `[wave_size, max_active_nodes]`. Written by
    /// `init_root_stats` at level 0 and by `propagate_child_stats` thereafter.
    /// Ping-pongs with `node_counts_alt` on level parity.
    node_counts: GpuTensor<R, u32>,
    /// Per-node Y sums `[wave_size, max_active_nodes, n_targets]`. Same
    /// producers and ping-pong as `node_counts`.
    node_y_sums: GpuTensor<R, f32>,
    /// Per-node Y sum-of-squares, same layout and ping-pong as
    /// `node_y_sums`.
    node_y_sum_sqs: GpuTensor<R, f32>,
    /// Odd-level half of the node-stat ping-pong, same layout as
    /// `node_counts`. `propagate_child_stats` reads the current level's
    /// stats and writes the next level's, and child ids overlap parent ids,
    /// so the two cannot share a buffer.
    node_counts_alt: GpuTensor<R, u32>,
    /// Odd-level half of the ping-pong, same layout as `node_y_sums`.
    node_y_sums_alt: GpuTensor<R, f32>,
    /// Odd-level half of the ping-pong, same layout as `node_y_sum_sqs`.
    node_y_sum_sqs_alt: GpuTensor<R, f32>,
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
    /// Drawn threshold per ExtraTrees candidate
    /// `[wave_size, max_active_nodes, k_feats * n_thresholds]`. Unused (and
    /// allocated as a stub) on the RandomForest path.
    cand_thr: GpuTensor<R, u32>,
    /// Left-side sample count per ExtraTrees candidate, same layout as
    /// `cand_thr`.
    cand_n_left: GpuTensor<R, u32>,
    /// Left-side Y sums per (ExtraTrees candidate, target), layout
    /// `[wave_size, max_active_nodes, k_feats * n_thresholds, n_targets]`.
    cand_y_sums_l: GpuTensor<R, f32>,
    /// Left-side Y sum-of-squares, same layout as `cand_y_sums_l`.
    cand_y_sum_sqs_l: GpuTensor<R, f32>,
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
    /// A `WaveState` with all GPU tensors allocated and uninitialised, or
    /// `CubeclUtils` if one busts the device's per-binding size limit. The
    /// per-candidate ExtraTrees tensors are the ones that reach it:
    /// `cand_stats_len` is
    /// `wave_size * max_active_nodes * k_feats * n_thresholds * n_targets`.
    #[allow(clippy::too_many_arguments)]
    fn allocate(
        client: &ComputeClient<R>,
        wave_size: usize,
        max_active_nodes: usize,
        n_samples: usize,
        k_feats: usize,
        n_targets: usize,
        n_thresholds: usize,
        layout: WaveLayout,
    ) -> Result<Self, BixverseErrors> {
        let use_et = layout == WaveLayout::ExtraTrees;
        // Every tensor a layout does not touch collapses to a length-1 stub.
        let node_stats_len = wave_size * max_active_nodes * n_targets;
        let n_cands = k_feats * n_thresholds;
        let cand_len = if use_et {
            wave_size * max_active_nodes * n_cands
        } else {
            1
        };
        let cand_stats_len = if use_et { cand_len * n_targets } else { 1 };
        let gather_node_len = if use_et {
            1
        } else {
            wave_size * max_active_nodes
        };
        let gather_offset_len = if use_et {
            1
        } else {
            wave_size * (max_active_nodes + 1)
        };
        let gather_id_len = if use_et { 1 } else { wave_size * n_samples };
        let slot_best_len = if use_et {
            1
        } else {
            wave_size * max_active_nodes * k_feats
        };

        Ok(Self {
            sample_to_node: GpuTensor::empty(vec![wave_size * n_samples], client)?,
            node_features: GpuTensor::empty(vec![wave_size * max_active_nodes * k_feats], client)?,
            slot_min_bin: GpuTensor::empty(vec![wave_size * max_active_nodes * k_feats], client)?,
            slot_max_bin: GpuTensor::empty(vec![wave_size * max_active_nodes * k_feats], client)?,
            node_sample_counts: GpuTensor::empty(vec![gather_node_len], client)?,
            node_sample_offsets: GpuTensor::empty(vec![gather_offset_len], client)?,
            node_sample_ids: GpuTensor::empty(vec![gather_id_len], client)?,
            node_sample_mults: GpuTensor::empty(vec![gather_id_len], client)?,
            slot_best_score: GpuTensor::empty(vec![slot_best_len], client)?,
            slot_best_thr: GpuTensor::empty(vec![slot_best_len], client)?,
            slot_best_n_left: GpuTensor::empty(vec![slot_best_len], client)?,
            slot_best_valid: GpuTensor::empty(vec![slot_best_len], client)?,
            node_counts: GpuTensor::empty(vec![wave_size * max_active_nodes], client)?,
            node_y_sums: GpuTensor::empty(vec![node_stats_len], client)?,
            node_y_sum_sqs: GpuTensor::empty(vec![node_stats_len], client)?,
            node_counts_alt: GpuTensor::empty(vec![wave_size * max_active_nodes], client)?,
            node_y_sums_alt: GpuTensor::empty(vec![node_stats_len], client)?,
            node_y_sum_sqs_alt: GpuTensor::empty(vec![node_stats_len], client)?,
            split_feature: GpuTensor::empty(vec![wave_size * max_active_nodes], client)?,
            split_threshold: GpuTensor::empty(vec![wave_size * max_active_nodes], client)?,
            split_n_left: GpuTensor::empty(vec![wave_size * max_active_nodes], client)?,
            split_y_sums_l: GpuTensor::empty(vec![node_stats_len], client)?,
            split_y_sum_sqs_l: GpuTensor::empty(vec![node_stats_len], client)?,
            cand_thr: GpuTensor::empty(vec![cand_len], client)?,
            cand_n_left: GpuTensor::empty(vec![cand_len], client)?,
            cand_y_sums_l: GpuTensor::empty(vec![cand_stats_len], client)?,
            cand_y_sum_sqs_l: GpuTensor::empty(vec![cand_stats_len], client)?,
            left_child_id: GpuTensor::empty(vec![wave_size * max_active_nodes], client)?,
            right_child_id: GpuTensor::empty(vec![wave_size * max_active_nodes], client)?,
            wave_size,
            max_active_nodes,
            n_samples,
            k_feats,
            n_targets,
        })
    }
}

////////////////////
// Sizing helpers //
////////////////////

/// Widest viable active-node count at any level, taking both the depth cap
/// and the min_samples_leaf cap into account.
///
/// Undersized allocations blow up as OOB writes into `node_features` and the
/// per-node stat tensors; oversized just wastes memory. We stick with the
/// smaller of the two caps.
///
/// The result is **not** bounded by the device's grid limit. Six kernels put
/// the active-node count straight on the grid y axis with all three axes
/// occupied, so `grid_2d` cannot fold it, and the depth cap alone reaches
/// 1_048_576 against a wgpu per-dimension limit of 65535. Their launchers run
/// the dispatch through `checked_cube_count` and fail loudly rather than
/// truncating a deep tree here, which needs `n_samples > 131_070` and
/// `max_depth >= 17` to reach because the leaf cap usually binds first.
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

/// Wave-scoped VRAM for the ExtraTrees path, which allocates per candidate
/// `(slot, threshold)` pair rather than per node.
///
/// ### Params
///
/// * `wave_size` - Trees in wave
/// * `max_active_nodes` - Widest viable active-node count
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_thresholds` - Random thresholds drawn per feature slot
///
/// ### Returns
///
/// Estimated byte cost of the per-candidate tensors.
pub fn wave_byte_cost_et(
    wave_size: usize,
    max_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    n_thresholds: usize,
) -> usize {
    let cand_slots = wave_size * max_active_nodes * k_feats * n_thresholds;
    (cand_slots * 2 * 4) + (cand_slots * n_targets * 2 * 4)
}

/// Which split pipeline a wave runs, and therefore which cost model
/// [`pick_wave_size`] applies.
///
/// The two differ enough that one model cannot describe them: ExtraTrees
/// materialises per-candidate left sums, the fused RandomForest path holds only
/// per-slot winners plus the sample gather.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveLayout {
    /// ExtraTrees: per-candidate left sums, no histogram.
    ExtraTrees,
    /// RandomForest with the shared-memory fused kernel.
    RandomForestFused,
}

/// Approximate per-wave byte cost of the fused RandomForest path.
///
/// Dominated by the per-node sample gather at large cell counts and by the
/// per-slot winner arrays at large `k_feats`. Both are small: at the reference
/// shape this is ~335 KB per tree, which is what keeps RandomForest at wave 32.
///
/// ### Params
///
/// * `wave_size` - Trees in the wave
/// * `max_active_nodes` - Node bound per level
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// Estimated bytes held resident for one wave.
pub fn wave_byte_cost_rf_fused(
    wave_size: usize,
    max_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    n_samples: usize,
) -> usize {
    // node_features plus the four slot_best_* arrays.
    let slot_slots = wave_size * max_active_nodes * k_feats;
    // Node stats ping-pong (sums and sum-of-squares, live and alt) plus the two
    // split-stat outputs.
    let node_stats = wave_size * max_active_nodes * n_targets;
    // sample_to_node plus the gather's ids and multiplicities.
    let sample_slots = wave_size * n_samples;

    (slot_slots * 5 * 4) + (node_stats * 6 * 4) + (sample_slots * 3 * 4)
}

/// Largest wave that fits both the VRAM budget and the device's per-binding
/// limit.
///
/// Halves from `DEFAULT_WAVE_SIZE` (capped at the tree count) until both
/// constraints hold. Errors only when a single-tree wave still busts them,
/// which takes an extreme `max_depth` / `n_features_split` combination on
/// either layout.
///
/// ### Params
///
/// * `max_active_nodes` - Node bound per level, see [`viable_max_active_nodes`]
/// * `k_feats` - Features per node
/// * `n_targets` - Targets in batch
/// * `n_trees_target` - Trees to fit; caps the wave
/// * `wave_byte_budget` - VRAM ceiling for the wave-scoped tensors
/// * `n_thresholds` - Random thresholds per feature slot (ExtraTrees only)
/// * `layout` - Which cost model applies
/// * `n_samples` - Number of samples
/// * `limits` - Device limits from [`GpuLimits::from_client`]
///
/// ### Returns
///
/// Trees per wave, in `1..=DEFAULT_WAVE_SIZE`.
///
/// ### Errors
///
/// `InvalidArgument` when even `wave_size = 1` exceeds either constraint.
#[allow(clippy::too_many_arguments)]
pub fn pick_wave_size(
    max_active_nodes: usize,
    k_feats: usize,
    n_targets: usize,
    n_trees_target: usize,
    wave_byte_budget: usize,
    n_thresholds: usize,
    layout: WaveLayout,
    n_samples: usize,
    limits: &GpuLimits,
) -> Result<usize, BixverseErrors> {
    let max_binding_bytes = limits.max_binding_bytes as usize;
    let cost = |w: usize| match layout {
        WaveLayout::ExtraTrees => {
            wave_byte_cost_et(w, max_active_nodes, k_feats, n_targets, n_thresholds)
        }
        WaveLayout::RandomForestFused => {
            wave_byte_cost_rf_fused(w, max_active_nodes, k_feats, n_targets, n_samples)
        }
    };
    // Largest single tensor in the wave: the per-(candidate, target) left-side
    // sums on the ExtraTrees path, the gathered sample ids on the fused one.
    let largest = |w: usize| {
        let slots = match layout {
            WaveLayout::ExtraTrees => w * max_active_nodes * k_feats * n_thresholds * n_targets,
            WaveLayout::RandomForestFused => (w * n_samples).max(w * max_active_nodes * n_targets),
        };
        slots * 4
    };
    let fits = |w: usize| cost(w) <= wave_byte_budget && largest(w) <= max_binding_bytes;

    let mut w = DEFAULT_WAVE_SIZE.min(n_trees_target.max(1));
    while w > 1 {
        if fits(w) {
            return Ok(w);
        }
        w /= 2;
    }
    if !fits(1) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "GPU SCENIC: even wave_size=1 does not fit (needs {} MB total vs a \
             {} MB budget, largest binding {} MB vs a {} MB device limit; \
             nodes={}, k_feats={}, n_targets={}). Reduce max_depth or \
             n_features_split.",
            cost(1) / (1024 * 1024),
            wave_byte_budget / (1024 * 1024),
            largest(1) / (1024 * 1024),
            max_binding_bytes / (1024 * 1024),
            max_active_nodes,
            k_feats,
            n_targets,
        )));
    }
    Ok(1)
}

/////////////////////////////
// Dense Y upload wrapper  //
/////////////////////////////

/// Device-resident dense Y for one target batch, `[n_samples, n_targets]`.
///
/// The ExtraTrees path reads Y with the target on the thread axis, so a dense
/// row-major layout makes every read coalesced and removes the CSR
/// indirection. At `MULTI_OUTPUT_BATCH = 64` targets this is
/// `n_samples * 64 * 4` bytes, about 2.5 MB at 10k cells, so densifying costs
/// nothing worth measuring.
///
/// ### Params
///
/// * `sy` - Host sparse Y batch to densify (CSR by sample)
/// * `n_samples` - Number of samples
/// * `n_targets` - Targets in the batch
/// * `scratch` - Caller-owned staging buffer, resized and reused across batches
/// * `client` - CubeCL compute client for buffer allocation and upload
///
/// ### Returns
///
/// A `[n_samples * n_targets]` f32 tensor with implicit zeros filled in, or
/// `CubeclUtils` if it busts the device's per-binding size limit.
fn upload_dense_y<R: Runtime>(
    sy: &SparseYBatch,
    n_samples: usize,
    n_targets: usize,
    scratch: &mut Vec<f32>,
    client: &ComputeClient<R>,
) -> Result<GpuTensor<R, f32>, BixverseErrors> {
    // Reuse the caller's allocation across batches. At 1M cells this buffer is
    // 256 MB, and a 63-batch run would otherwise allocate and free it 63 times.
    scratch.clear();
    scratch.resize(n_samples * n_targets, 0.0);
    for s in 0..n_samples {
        let lo = sy.offsets[s] as usize;
        let hi = sy.offsets[s + 1] as usize;
        for j in lo..hi {
            scratch[s * n_targets + sy.target_indices[j] as usize] = sy.values[j];
        }
    }
    Ok(GpuTensor::<R, f32>::from_slice(
        scratch,
        vec![n_samples * n_targets],
        client,
    )?)
}

/////////////////
// Wave driver //
/////////////////

/// Run a wave-synchronous BFS for `wave_size` trees, atomically accumulating
/// per-target importances into `batch_importances_gpu` on device (layout
/// `[n_features, n_targets]`, f32 bits stored as `u32`). Uses the
/// pre-allocated `WaveState` and the once-per-batch dense Y upload.
///
/// No per-level `.read()` calls. `n_active_nodes` per level is the upper bound
/// `min(2^depth, max_active_nodes)`; phantom nodes (no samples) end up with
/// `split_feature == INVALID_NODE` from the split kernels and are no-ops in
/// `compute_child_ids` and `accumulate_importance`.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `feature_data_gpu` - Quantised bins `[n_features, n_samples]` (u8 as u32)
/// * `y_dense_gpu` - Dense Y `[n_samples, n_targets]` for this batch
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
    y_dense_gpu: &GpuTensor<R, f32>,
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
    )?;

    for depth in 0..max_depth {
        // At depth d a full binary tree has at most 2^d active nodes. Phantom
        // nodes cost cheap launches and no atomics: their sample_to_node never
        // matches.
        let depth_cap = 1usize.checked_shl(depth as u32).unwrap_or(usize::MAX);
        let n_active_nodes = depth_cap.min(max_active_nodes).max(1);
        let n_active_next = depth_cap.saturating_mul(2).min(max_active_nodes).max(1);

        // Node stats ping-pong on level parity: propagate_child_stats reads
        // the current level's totals and writes the next level's, and child
        // ids overlap parent ids, so they cannot share a buffer.
        let (cur_counts, cur_y_sums, cur_y_sum_sqs) = if depth % 2 == 0 {
            (
                &state.node_counts,
                &state.node_y_sums,
                &state.node_y_sum_sqs,
            )
        } else {
            (
                &state.node_counts_alt,
                &state.node_y_sums_alt,
                &state.node_y_sum_sqs_alt,
            )
        };
        let (nxt_counts, nxt_y_sums, nxt_y_sum_sqs) = if depth % 2 == 0 {
            (
                &state.node_counts_alt,
                &state.node_y_sums_alt,
                &state.node_y_sum_sqs_alt,
            )
        } else {
            (
                &state.node_counts,
                &state.node_y_sums,
                &state.node_y_sum_sqs,
            )
        };

        launch_sample_features(
            client,
            tree_seeds,
            &state.node_features,
            wave_size,
            n_active_nodes,
            k_feats,
            n_features,
            depth as u32,
        )?;

        let at_max_depth = depth + 1 >= max_depth;

        if use_et {
            if depth == 0 {
                launch_init_root_stats(
                    client,
                    y_dense_gpu,
                    sample_multiplicity_gpu,
                    cur_counts,
                    cur_y_sums,
                    cur_y_sum_sqs,
                    n_samples,
                    wave_size,
                    n_active_nodes,
                    n_targets,
                );
            }

            launch_scan_slot_bin_range(
                client,
                feature_data_gpu,
                &state.sample_to_node,
                sample_multiplicity_gpu,
                &state.node_features,
                cur_counts,
                &state.slot_min_bin,
                &state.slot_max_bin,
                n_samples,
                wave_size,
                n_active_nodes,
                k_feats,
            )?;

            launch_accumulate_split_stats_et(
                client,
                feature_data_gpu,
                y_dense_gpu,
                &state.sample_to_node,
                sample_multiplicity_gpu,
                &state.node_features,
                cur_counts,
                tree_seeds,
                &state.slot_min_bin,
                &state.slot_max_bin,
                &state.cand_thr,
                &state.cand_n_left,
                &state.cand_y_sums_l,
                &state.cand_y_sum_sqs_l,
                n_samples,
                wave_size,
                n_active_nodes,
                k_feats,
                n_targets,
                n_thresholds,
                depth as u32,
            )?;

            launch_evaluate_splits_et_direct(
                client,
                cur_counts,
                cur_y_sums,
                cur_y_sum_sqs,
                &state.node_features,
                &state.cand_thr,
                &state.cand_n_left,
                &state.cand_y_sums_l,
                &state.cand_y_sum_sqs_l,
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
            )?;
        } else {
            if depth == 0 {
                launch_init_root_stats(
                    client,
                    y_dense_gpu,
                    sample_multiplicity_gpu,
                    cur_counts,
                    cur_y_sums,
                    cur_y_sum_sqs,
                    n_samples,
                    wave_size,
                    n_active_nodes,
                    n_targets,
                );
            }

            launch_gather_node_samples(
                client,
                &state.sample_to_node,
                sample_multiplicity_gpu,
                &state.node_sample_counts,
                &state.node_sample_offsets,
                &state.node_sample_ids,
                &state.node_sample_mults,
                n_samples,
                wave_size,
                n_active_nodes,
            )?;

            launch_rf_fused(
                client,
                feature_data_gpu,
                y_dense_gpu,
                &state.node_sample_offsets,
                &state.node_sample_ids,
                &state.node_sample_mults,
                &state.node_features,
                cur_counts,
                cur_y_sums,
                cur_y_sum_sqs,
                &state.slot_best_score,
                &state.slot_best_thr,
                &state.slot_best_n_left,
                &state.slot_best_valid,
                &state.split_feature,
                &state.split_threshold,
                &state.split_n_left,
                n_samples,
                wave_size,
                n_active_nodes,
                k_feats,
                n_targets,
                min_samples_leaf,
            )?;

            launch_finalise_split_stats_rf(
                client,
                feature_data_gpu,
                y_dense_gpu,
                &state.node_sample_offsets,
                &state.node_sample_ids,
                &state.node_sample_mults,
                &state.split_feature,
                &state.split_threshold,
                &state.split_y_sums_l,
                &state.split_y_sum_sqs_l,
                n_samples,
                wave_size,
                n_active_nodes,
                n_targets,
            )?;
        }

        launch_accumulate_importance(
            client,
            cur_counts,
            cur_y_sums,
            cur_y_sum_sqs,
            &state.split_feature,
            &state.split_n_left,
            &state.split_y_sums_l,
            &state.split_y_sum_sqs_l,
            batch_importances_gpu,
            wave_size,
            n_active_nodes,
            n_targets,
            n_total,
        )?;

        if at_max_depth {
            continue;
        }

        launch_compute_child_ids(
            client,
            &state.split_feature,
            cur_counts,
            &state.left_child_id,
            &state.right_child_id,
            wave_size,
            n_active_nodes,
        );

        launch_zero_node_stats(
            client,
            nxt_counts,
            nxt_y_sums,
            nxt_y_sum_sqs,
            wave_size,
            n_active_next,
            n_targets,
        )?;

        launch_propagate_child_stats(
            client,
            &state.split_feature,
            &state.split_n_left,
            &state.split_y_sums_l,
            &state.split_y_sum_sqs_l,
            &state.left_child_id,
            &state.right_child_id,
            cur_counts,
            cur_y_sums,
            cur_y_sum_sqs,
            nxt_counts,
            nxt_y_sums,
            nxt_y_sum_sqs,
            wave_size,
            n_active_nodes,
            n_active_next,
            n_targets,
        )?;

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
        )?;
    }

    Ok(())
}

//////////////////
// Public entry //
//////////////////

/// Print a progress line every time the completed fraction crosses a 10%
/// stride, matching the CPU driver's cadence in `run_scenic_multi_output`.
///
/// The counter tracks batches *dispatched*, not batches the GPU has retired.
/// Kernel launches are asynchronous, so the queue trails this line. It cannot
/// trail without bound: each batch allocates its own [`WaveState`] and holds an
/// importance tensor resident, so the allocator ends up waiting on finished
/// work to recycle VRAM and the host loop is pulled back in step. Reporting
/// true completion instead would need a `sync()` per batch, which is precisely
/// the host/device overlap this driver exists to buy.
///
/// ### Params
///
/// * `done` - Batches dispatched so far, `1..=total`
/// * `total` - Non-empty batches in this call
/// * `start` - Timer started before the dispatch loop
fn report_batch_progress(done: usize, total: usize, start: Instant) {
    report_decile_progress(
        done,
        done.saturating_sub(1),
        total,
        "batches dispatched",
        start.elapsed(),
    );
}

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
/// * `verbose` - `0` -> silent, `1` -> per-batch progress, `2` -> detailed.
///   See the note on `report_batch_progress` for what the counter tracks.
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
///   `params.wave_byte_budget` on any batch, or if the packed feature tensor
///   exceeds the device's per-binding limit.
/// * `CubeclUtils` if a wave allocation busts the per-binding limit or a
///   dispatch grid is over the device's cube-count limit. The active-node
///   count on the grid y axis is the one that reaches it on deep trees; see
///   [`viable_max_active_nodes`].
/// * Propagates GPU read-back errors from the deferred importance readback.
#[allow(clippy::too_many_arguments)]
pub fn fit_scenic_batches_gpu<R: Runtime>(
    batches: &[&[SparseAxis<u32, f32>]],
    batch_seeds: &[usize],
    feature_matrix: &QuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    device: R::Device,
    params: &ScenicGpuParams,
    verbose: usize,
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
    let use_et = config.random_threshold();
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

    let verbosity = parse_verbosity_level(verbose);
    let total_batches = batches.iter().filter(|b| !b.is_empty()).count();
    let client = R::client(&device);

    // Decided once per call, and the only place the fused kernel's
    // shared-memory ask is checked against the device.
    let wave_layout = if use_et {
        WaveLayout::ExtraTrees
    } else {
        require_fused_rf(&client)?;
        WaveLayout::RandomForestFused
    };

    // Single feature upload amortised across every batch, packed four u8 bins
    // per u32 word. Widening to u32 instead would cost 4 GB on both host and
    // device at 1000 TFs by 1M cells, which is the per-binding limit on an M1
    // Max; `feature_bin` unpacks on the device side.
    let n_bins_total = n_features * n_samples;
    let n_words = n_bins_total.div_ceil(4);
    let limits = GpuLimits::from_client(&client);
    let max_binding = limits.max_binding_bytes as usize;
    if n_words * 4 > max_binding {
        return Err(BixverseErrors::InvalidArgument(format!(
            "GPU SCENIC: the packed feature tensor needs {} MB but the device \
             accepts at most {} MB per binding (n_features={}, n_samples={}). \
             Subsample cells or reduce the TF set.",
            n_words * 4 / (1024 * 1024),
            max_binding / (1024 * 1024),
            n_features,
            n_samples,
        )));
    }
    let mut packed = vec![0u32; n_words];
    for (i, &b) in feature_matrix.data.iter().enumerate() {
        packed[i / 4] |= (b as u32) << ((i % 4) * 8);
    }
    let feature_data_gpu = GpuTensor::<R, u32>::from_slice(&packed, vec![n_words], &client)?;
    drop(packed);

    // Per-batch importance tensor handles, kept alive until the deferred
    // readback pass. Each entry is (flat_target_offset, batch_n_targets,
    // importance_gpu), n_features * batch_n_targets * 4 bytes apiece, so
    // hundreds of batches still add up to only tens of MB.
    let mut deferred: Vec<(usize, usize, GpuTensor<R, u32>)> = Vec::with_capacity(batches.len());

    // Reused across batches by upload_dense_y.
    let mut dense_scratch: Vec<f32> = Vec::new();

    let start_dispatch = Instant::now();
    let mut batches_done = 0usize;
    let mut flat_offset = 0usize;
    for (chunk, &batch_seed) in batches.iter().zip(batch_seeds.iter()) {
        let batch_n_targets = chunk.len();
        if batch_n_targets == 0 {
            continue;
        }

        let sparse_y = SparseYBatch::from_targets(chunk, n_samples)?;
        let y_dense_gpu = upload_dense_y(
            &sparse_y,
            n_samples,
            batch_n_targets,
            &mut dense_scratch,
            &client,
        )?;

        let wave_size = pick_wave_size(
            max_active_nodes,
            k_feats,
            batch_n_targets,
            n_trees,
            params.wave_byte_budget,
            n_thresholds,
            wave_layout,
            n_samples,
            &limits,
        )?;
        let state = WaveState::allocate(
            &client,
            wave_size,
            max_active_nodes,
            n_samples,
            k_feats,
            batch_n_targets,
            n_thresholds,
            wave_layout,
        )?;

        // Interleaved [n_features, batch_n_targets] importances, accumulated
        // atomically on device across every tree in this batch. f32 bits
        // stored as u32 so `accumulate_importance` can CAS-add.
        let batch_importances_gpu = GpuTensor::<R, u32>::from_slice(
            &vec![0u32; n_features * batch_n_targets],
            vec![n_features * batch_n_targets],
            &client,
        )?;

        let mut tree_idx = 0usize;
        while tree_idx < n_trees {
            let this_wave = std::cmp::min(wave_size, n_trees - tree_idx);

            let seeds_host: Vec<u32> = (tree_idx..tree_idx + this_wave)
                .map(|t| tree_seed(batch_seed, t) as u32)
                .collect();
            let tree_seeds_gpu =
                GpuTensor::<R, u32>::from_slice(&seeds_host, vec![this_wave], &client)?;

            // Bootstrap draws n_sub with replacement; non-bootstrap subsample
            // is Fisher-Yates over the first n_sub. RNG seeded off
            // `tree_seed(batch_seed, t)` to match the CPU path.
            let mut mult_host = vec![0u32; this_wave * n_samples];
            if subsample_needed {
                for w in 0..this_wave {
                    let mut rng = SmallRng::seed_from_u64(tree_seed(batch_seed, tree_idx + w));
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
                GpuTensor::<R, u32>::from_slice(&mult_host, vec![this_wave * n_samples], &client)?;

            // Kernels gate on the `wave_size` param, so an over-provisioned
            // `state` is safe; only the terminal short wave needs a shadow.
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
                    n_thresholds,
                    wave_layout,
                )?)
            };
            let use_state = effective_state.as_ref().unwrap_or(&state);

            run_wave_bfs(
                &client,
                &feature_data_gpu,
                &y_dense_gpu,
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

        // Stash the importance handle; no .read() here, so later batches keep
        // submitting kernels instead of blocking on this one.
        deferred.push((flat_offset, batch_n_targets, batch_importances_gpu));
        flat_offset += batch_n_targets;

        batches_done += 1;
        if verbosity.normal_verbosity() {
            report_batch_progress(batches_done, total_batches, start_dispatch);
        }
    }

    if verbosity.detailed_verbosity() {
        println!(
            "  All {} batches dispatched in {:.2?}, draining the queue",
            total_batches,
            start_dispatch.elapsed()
        );
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
        0,
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
/// * `reader` - Reader over the sparse gene expression store.
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
pub fn run_scenic_grn_gpu<R, S>(
    reader: &S,
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
    S: SingleCellReading,
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
    let setup = scenic_common_setup(reader, cell_indices, tf_indices, verbose)?;
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
        reader,
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
        verbose,
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
pub fn run_scenic_grn_streaming_gpu<R, S>(
    reader: &S,
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
    S: SingleCellReading,
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
    let setup = scenic_common_setup(reader, cell_indices, tf_indices, verbose)?;
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
        reader,
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
            .map(|i| seed.wrapping_add((global_batch_offset + i).wrapping_mul(BATCH_SEED_STRIDE)))
            .collect();

        let flat_imp = fit_scenic_batches_gpu::<R>(
            &chunk_col_batches,
            &chunk_batch_seeds,
            &setup.tf_data,
            setup.n_cells,
            config,
            device.clone(),
            gpu_params,
            verbose,
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
        verbose,
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Device limits with no GPU attached. Only the plane width and the binding
    /// ceiling matter to the host-side choices under test; the rest are set
    /// generously so they never become the binding constraint.
    fn limits(plane_min: u32, plane_max: u32, max_binding_bytes: u64) -> GpuLimits {
        GpuLimits {
            max_shared_bytes: 32 * 1024,
            max_cube_count: (65_535, 65_535, 65_535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 64),
            max_binding_bytes,
            plane_size_min: plane_min,
            plane_size_max: plane_max,
        }
    }

    /// Apple Silicon reports 32/32 and lavapipe 8/8. The argmax needs the
    /// workgroup to be exactly one plane, so only the first takes the plane path.
    /// Neither is reachable from the end-to-end tests, which run one shape on one
    /// device.
    #[test]
    fn plane_argmax_viable_requires_an_exact_single_plane() {
        assert!(plane_argmax_viable(WORKGROUP_32, &limits(32, 32, u64::MAX)));
        assert!(!plane_argmax_viable(WORKGROUP_32, &limits(8, 8, u64::MAX)));
        // A reported range means a workgroup could straddle two planes, so a
        // bare plane_max would reduce over only part of the candidate set.
        assert!(!plane_argmax_viable(
            WORKGROUP_32,
            &limits(16, 32, u64::MAX)
        ));
    }

    /// Compaction only needs the workgroup to split into few enough planes, so a
    /// narrow-plane device can still qualify where the argmax does not.
    #[test]
    fn plane_compact_viable_bounds_the_plane_count() {
        assert!(plane_compact_viable(
            WORKGROUP_128,
            &limits(32, 32, u64::MAX)
        ));
        assert_eq!(
            plane_compact_viable(WORKGROUP_128, &limits(8, 8, u64::MAX)),
            128 / 8 <= MAX_PLANES_PER_CUBE
        );
    }

    /// Bins halve until the padded histogram fits `SMEM_HIST_SLOTS`, so the
    /// resolution a run gets is a step function of the target count.
    #[test]
    fn pick_gpu_bins_halves_until_the_histogram_fits() {
        // 256 * 8 = 2048 fits 2176; 256 * 9 = 2304 does not.
        assert_eq!(pick_gpu_bins(7), 256);
        assert_eq!(pick_gpu_bins(8), 128);
        // 64 * 34 = 2176 lands exactly on the budget.
        assert_eq!(pick_gpu_bins(33), 64);
        assert_eq!(pick_gpu_bins(64), 32);

        for n_targets in [0, 1, 7, 8, 33, 64, 1000] {
            let bins = pick_gpu_bins(n_targets);
            assert!((1..=N_BINS).contains(&bins));
            assert!(
                bins == 1 || (bins as usize) * (n_targets + 1) <= SMEM_HIST_SLOTS as usize,
                "n_targets {n_targets} chose {bins} bins, which busts the budget"
            );
        }
    }

    /// `min_samples_leaf == 0` disables the leaf cap, and the depth cap is
    /// clamped at 2^20 so a large `max_depth` cannot overflow the shift.
    #[test]
    fn viable_max_active_nodes_applies_both_caps() {
        // Depth binds: 2^3 = 8 against a leaf cap of 1000 / 4 = 250.
        assert_eq!(viable_max_active_nodes(3, 1000, 2), 8);
        // Leaf binds: 2^10 = 1024 against 100 / 20 = 5.
        assert_eq!(viable_max_active_nodes(10, 100, 10), 5);
        // No leaf constraint falls back to the depth cap.
        assert_eq!(viable_max_active_nodes(3, 1000, 0), 8);
        // The shift is clamped rather than overflowing.
        assert_eq!(viable_max_active_nodes(64, 1000, 0), 1 << 20);
        // Never zero, whatever the inputs.
        assert_eq!(viable_max_active_nodes(0, 0, 1), 1);
    }

    /// The fused kernel's shared-memory ask is a constant, and the compile-time
    /// guard above it depends on that staying under the tightest shipping budget.
    #[test]
    fn fused_rf_smem_bytes_is_the_expected_constant() {
        let expected =
            SMEM_HIST_SLOTS as usize * 4 + N_BINS as usize * 4 + WORKGROUP_128 as usize * 7 * 4 + 8;
        assert_eq!(fused_rf_smem_bytes(), expected);
        assert_eq!(fused_rf_smem_bytes(), 13_320);
        assert!(fused_rf_smem_bytes() <= 16_352);
    }

    /// Wave cost must grow with the wave size on both layouts, which is what
    /// makes `pick_wave_size`'s halving loop terminate at a fitting value.
    #[test]
    fn wave_byte_costs_grow_with_the_wave() {
        let et = |w| wave_byte_cost_et(w, 8, 4, 2, 4);
        let fused = |w| wave_byte_cost_rf_fused(w, 8, 4, 2, 1000);

        assert!(et(1) < et(2) && et(2) < et(32));
        assert!(fused(1) < fused(2) && fused(2) < fused(32));
    }

    /// A generous budget leaves the default wave intact; the tree target caps it
    /// below that.
    #[test]
    fn pick_wave_size_starts_at_the_default_and_respects_the_tree_target() {
        let l = limits(32, 32, u64::MAX);

        let w = pick_wave_size(
            8,
            4,
            2,
            64,
            usize::MAX / 4,
            4,
            WaveLayout::ExtraTrees,
            1000,
            &l,
        )
        .expect("a generous budget fits");
        assert_eq!(w, DEFAULT_WAVE_SIZE);

        let w = pick_wave_size(
            8,
            4,
            2,
            4,
            usize::MAX / 4,
            4,
            WaveLayout::ExtraTrees,
            1000,
            &l,
        )
        .expect("a generous budget fits");
        assert_eq!(w, 4, "the wave cannot exceed the trees it has to fill");
    }

    /// The VRAM budget and the per-binding ceiling halve the wave independently.
    #[test]
    fn pick_wave_size_halves_under_either_constraint() {
        let generous = limits(32, 32, u64::MAX);

        let budget = wave_byte_cost_et(8, 8, 4, 2, 4);
        let w = pick_wave_size(
            8,
            4,
            2,
            64,
            budget,
            4,
            WaveLayout::ExtraTrees,
            1000,
            &generous,
        )
        .expect("wave 8 fits its own cost");
        assert_eq!(w, 8);

        // Same budget, but a binding ceiling that only wave 2 clears.
        let largest_at_2 = (2 * 8 * 4 * 4 * 2) * 4;
        let w = pick_wave_size(
            8,
            4,
            2,
            64,
            usize::MAX / 4,
            4,
            WaveLayout::ExtraTrees,
            1000,
            &limits(32, 32, largest_at_2 as u64),
        )
        .expect("wave 2 fits the binding ceiling");
        assert_eq!(w, 2);
    }

    /// A budget too small even for a single tree is an error rather than a
    /// silent zero-width wave. This arm has never been observed in production.
    #[test]
    fn pick_wave_size_errors_when_even_one_tree_does_not_fit() {
        let l = limits(32, 32, u64::MAX);
        let res = pick_wave_size(8, 4, 2, 64, 1, 4, WaveLayout::ExtraTrees, 1000, &l);

        assert!(
            matches!(res, Err(BixverseErrors::InvalidArgument(_))),
            "expected InvalidArgument, got {res:?}"
        );
    }
}
