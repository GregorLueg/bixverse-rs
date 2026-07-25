# SCENIC GPU: make the ET path actually fast

## Context

`docs/archive/scenic_gpu_experiment.md` concluded the GPU SCENIC tree regressor
is a hardware loss on Apple Silicon and parked it. Tracing the code says
otherwise. The archive doc gets three things wrong and misses the big one.

**Wrong on facts.** The bench overrides the VRAM budget to 12 GiB
(`benches/gpu_scenic_bench.rs:79`). At the bench shape (`viable_max_active_nodes(10,
10000, 50)` = 100 nodes, `k_feats` = 31, 64 targets) `wave_byte_cost(8, ..)` is
6.55 GB, under 12.88 GB, so `pick_wave_size` returns **8**, not 4. Root cause #3
and the "single highest-leverage experiment left" (in-place prefix sum lets the
wave go back to 8, worth 1.5-2x) both describe a wave size that was never in
force for the numbers in the table.

**The miss.** `ExtraTreesConfig::default().n_thresholds == 1`.
`evaluate_splits_et` draws one threshold per (tree, node, slot) and reads exactly
one row, `cum_y_sums[thr * n_targets .. +n_targets]`. Meanwhile
`build_hist_privatised` writes `2 x 256 x n_targets` floats per slot and
`prefix_sum_bins` reads all of them and writes another `2 x 256 x n_targets`. ET
consumes about **0.4%** of the buffer that dominates its runtime.

**Why the CPU is competitive.** Its 256-bin x 64-target histogram is ~128 KB and
lives in L2, never touching DRAM. The GPU streams ~6.5 GB per level through DRAM
to compute the same thing. The fix is not "less traffic" in the abstract, it is
**get the working set into threadgroup memory**: 512 bytes for ET, an 8 KB tile
for RF, against Metal's 32 KB threadgroup budget.

**Target.** On this M1 Max (8 P-cores + 2 E, 32 GPU cores, 400 GB/s) the CPU e2e
driver fans out over batches with rayon while `fit_multi_trees_sparse` is
strictly sequential (`scenic.rs:1513`). So the GPU must beat **one** CPU core by
more than the effective core count before e2e can flip. Today it beats one core
by 1.53x (ET) / 1.34x (RF). Stop when ET crosses **10x**, confirm on the full
e2e, then decide on RF separately.

Fidelity budget: hold the existing gates in `tests/scenic_gpu.rs` (Pearson >=
0.95 ET/RF vs CPU, >= 0.85 roundtrip, GPU-vs-GPU determinism).

## Prior art worth stealing

Checked against `cubecl-core-0.10.0` rather than assumed.

**Plane (subgroup) primitives are available and completely unused.**
`src/frontend/plane.rs` exposes `plane_sum`, `plane_max`, `plane_min`,
`plane_inclusive_sum`, `plane_exclusive_sum`, `plane_shuffle_down`,
`plane_broadcast`, `plane_ballot`, `plane_elect`. Zero uses across `src/gpu/` or
`ann-search-rs/src/gpu/`. The 5-to-7 stage SMEM argmax ladder in
`evaluate_splits_et` / `_rf`, five SMEM arrays with a `sync_cube()` between every
stage, is what CUDA code looks like without `__shfl_down_sync`. `plane_max` plus
`plane_ballot` does it with no SMEM and no barriers. `plane_inclusive_sum` is
exactly what `prefix_sum_bins` hand-rolls as a 256-long serial dependent chain.

**Fixed-point integer atomics (XGBoost `GradientPairInt64`).** `Atomic<Inner>`
gets native `fetch_add` / `fetch_max` / `fetch_min` for
`Inner::Scalar: Numeric`, so `Atomic<i32>` / `Atomic<u32>` have all three. WGSL
has no float `atomicAdd`, which is why `atomic_add_f32_bits` is a CAS retry loop
in the innermost loop of `build_hist_privatised`. Rescaling y to fixed-point
with a rounding factor kills the retry loop and makes accumulation
order-independent, hence actually deterministic.
`phase2_multi_batch_determinism` currently passes by luck (identical launch
shape gives identical scheduling), not by construction. Constraint: WGSL has no
64-bit atomics, so i32 with a range analysis or a hi/lo split.

**Row partitioning is core, not optional.** XGBoost has a dedicated
`RowPartitioner`; cuML RF does the same. Every serious GPU tree implementation
physically permutes rows so a node's rows are contiguous, precisely to avoid the
`n_active_nodes` read amplification present here.

**`N_BINS` is a tunable.** LightGBM defaults to 255, XGBoost to 256, and both
routinely run at 63/64 with small accuracy cost.

**Confirmations.** cuML RF's MSE gain is `S_L²/n_L + S_R²/n_R`, the same
identity that drops the sum-of-squares histogram, so that is the standard
formulation rather than a risky rewrite. cuML also uses a device-side node work
queue with a bounded batch instead of dispatching `2^d`.

**Histogram subtraction does not apply to restructured ET.** Universal in GBDT
and it applies to RF here, but min/max of a union is not derivable by
subtraction and ET's left-sums sit at a per-node random threshold. Step 1
(right = parent minus left) is the same trick one level up. Noted so nobody
burns a day on it.

**Does not transfer:** EFB / feature bundling (the quantised TF matrix is not
exclusive-sparse), GOSS (GBM-specific), leaf-wise growth (ET/RF is depth-wise),
CUDA dynamic parallelism, 64-bit atomics.

## Iteration loop

One step per commit. For each: implement, run the gate, run the micro-bench,
keep or `git reset --hard HEAD~1`. Commit message records the before/after
GPU-vs-1-core ratio.

**Gate** (~1-2 min):
```bash
cargo test --features gpu,single-cell --test scenic_gpu
```
Record the three printed Pearson means at baseline
(`phase2_multi_tree_pearson`, `phase3_random_forest_pearson`,
`run_scenic_grn_gpu_roundtrip`). Every step must hold them to within 0.005
except where it deliberately changes numerics. That is a far tighter regression
signal than the 0.95 floor and needs no new scaffolding.

**Micro-bench** (new, ~1 min): `benches/gpu_scenic_micro.rs`, registered in
`Cargo.toml` alongside the two existing `[[bench]]` blocks.

- Shape mirrors one e2e batch exactly: 10k cells, 1000 TFs, 64 targets,
  `max_depth = 10`, `min_samples_leaf = 50`, `n_features_split = 0`. Only
  `n_trees` shrinks, 250 to 8 and 32.
- Two tree counts per learner. `n_trees = 8` is one wave, `n_trees = 32` is
  four. The delta isolates per-wave kernel cost from fixed per-batch cost
  (`WaveState::allocate`, feature upload, sparse Y upload).
- Runs `fit_multi_trees_sparse` (one core) and `fit_multi_trees_gpu`, prints
  wall clock and **the ratio**. That ratio is the number that decides each step.
- Prints the chosen `wave_size` and `wave_byte_cost` so the VRAM story is
  visible as it changes.
- Reuse the data builders from `benches/gpu_scenic_bench.rs`
  (`make_features_kernel`, `make_targets_kernel`) rather than writing new ones.

Baseline to beat: ET 1.53x, RF 1.34x.

A 50k-cell variant guards against optimising into the 10k corner (the archive
table shows the ratio improving with cell count, so 10k is the GPU's worst
shape and also the only shape the e2e was ever measured at). Run it at
milestones only, not every step.

## Steps

Everything lives in `src/gpu/sc_gpu/scenic_gpu.rs`. The public API is untouched:
`run_scenic_grn_gpu`, `run_scenic_grn_streaming_gpu`,
`run_scenic_grn_in_memory_gpu` and `fit_scenic_batches_gpu` keep their
signatures. Only the kernels, `WaveState` and `run_wave_bfs` change.

### Step 0: prove the plane API on this backend

Do this first, not because it pays much yet, but because step 3 wants to depend
on it and this is the codebase's first use of plane primitives. If they do not
lower cleanly on wgpu/Metal, better to find out in a 50-line change.

Replace the SMEM argmax ladder in `evaluate_splits_et` (the smaller of the two,
`WORKGROUP_32`, five stages) with `plane_max` on the score plus `plane_ballot` /
`plane_shuffle` to recover the winning lane's slot / threshold / n_left. Drops
five SMEM arrays and five `sync_cube()` barriers.

Expected: small on its own. The output is a yes/no on whether plane ops work
here, plus a reusable pattern for `evaluate_splits_rf`, the `prefix_sum_bins`
scan (`plane_inclusive_sum`), and the SMEM reductions in
`harmony_kernels.rs::objective_partials`.

### Step 1: delete `merge_hist`, propagate node stats from the parent

`merge_hist` reads the whole `hist_y_sums` + `hist_y_sum_sqs` array (3.25 GB per
level at wave 8) to recompute per-node totals the parent level already knows
exactly. The CPU does not do this: `build_node_multi_sparse` passes
`left_y_sums` / `right_y_sums` down the recursion (`scenic.rs:1050-1058`).

- Ping-pong `node_counts` / `node_y_sums` / `node_y_sum_sqs` in `WaveState` (two
  buffers each, ~51k floats, noise).
- New `init_root_stats`: one workgroup per tree, threads stride over samples,
  SMEM reduction. Level 0 only.
- New `propagate_child_stats` running after `compute_child_ids`: for each parent
  with a valid split, scatter `(split_n_left, split_y_sums_l, split_y_sum_sqs_l)`
  to `left_child_id` and `(count - n_left, y_sums - y_sums_l, ...)` to
  `right_child_id`. Zero the write buffer first so unwritten slots read count 0.
- Drop `merge_hist`, `launch_merge_hist` and their call site.

Numerics improve rather than drift: this is what the CPU computes. Expected
~20% on both learners. Also the cheapest possible test of the traffic model. If
it lands near zero, stop and re-measure before attempting step 3.

### Step 2: phantom-node early-out

`run_wave_bfs` dispatches `n_active_nodes = min(2^d, max_active_nodes)` with no
knowledge of which nodes exist. The comment at `scenic_gpu.rs:2687-2690` says
phantoms cost "cheap kernel launches and no atomics". The atomics part is true;
the cooperative zero of a `256 x n_targets x 2` slice (33k stores) and the full
n_samples scan are not.

Step 1 makes `node_counts` available *before* `build_hist_privatised`, so:

- Every per-node kernel terminates immediately when `node_counts[node_flat] == 0`.
- **Invariant to keep**: if `build_hist_privatised` skips the zeroing for a
  node, every downstream kernel must skip that node too, or they read stale
  garbage. Apply the same guard to `prefix_sum_bins`, `evaluate_splits_*`,
  `accumulate_importance`.

Cheap (~20 lines), payoff uncertain. At this shape most of the 100 slots are
probably real at deep levels, so expect 1.1-1.3x rather than anything dramatic.
Measure it, keep it if positive.

### Step 3: the ET restructure (the big one)

Replace the histogram pipeline for `config.random_threshold() == true`. Dispatch
on `use_et` in `run_wave_bfs`; the RF path keeps the existing kernels untouched
for now.

New per-level ET pipeline:

1. `sample_node_features` (unchanged).
2. **New `scan_slot_bin_range`**: one workgroup per (tree, node, slot),
   `WORKGROUP_128`. Threads stride over samples keeping register min/max over
   `feature_data[feat * n_samples + s]` for samples where
   `sample_to_node == node && mult > 0`, then `plane_min` / `plane_max` across
   the plane and a native `Atomic::<u32>::fetch_min` / `fetch_max` across planes.
   No SMEM ladder. Writes 2 u32 per slot. This reproduces exactly what
   `prefix_sum_bins` currently derives from the histogram (first/last bin with
   nonzero count) without building one.
3. **New `draw_and_accumulate_split_stats_et`**: one workgroup per (tree, node,
   slot). Recomputes `thr` with the *identical* hash chain already in
   `evaluate_splits_et` (`tree_seed ^ level ^ node ^ slot ^ ti`, salted with
   `2654435769`), writes it to a new `slot_thr` tensor. SMEM `s_y_sum[n_targets]`,
   `s_y_sumsq[n_targets]`, `s_n_left`. Threads stride over the node's samples;
   those with `bin <= thr` add their sparse Y entries into **shared** memory
   instead of global. One coalesced write of `2 * n_targets + 1` at the end.
   - SMEM cost: `n_thresholds * n_targets * 2 * 4` bytes. At the default
     `n_thresholds = 1` and `n_targets <= MULTI_OUTPUT_BATCH = 64` that is 512
     bytes. Generalise the loop over `n_thresholds`; add a `const` guard for the
     threadgroup-memory ceiling rather than a magic number.
   - **Accumulate in fixed-point i32, not float CAS.** The natural translation
     is `atomic_add_f32_bits` on SMEM, but that keeps the retry loop. Scale y by
     a per-batch rounding factor (max |y| over the batch, computed host-side at
     `SparseYBatch::from_targets` time) and use native `Atomic::<i32>::fetch_add`,
     the `GradientPairInt64` trick. Removes the loop and makes the accumulation
     order-independent. Range check: `n_samples * max|y| * scale` must fit i32,
     so pick the scale from `n_samples` and `max|y|` together and fall back to
     the CAS path if it will not fit. Both paths must hold the recorded Pearson
     means; the fixed-point one additionally makes the determinism test hold by
     construction rather than by luck.
4. **Rewrite `evaluate_splits_et`** to read the per-slot left stats (small) plus
   the propagated node stats, score each candidate, SMEM argmax, write the
   winner and copy the winning slot's left stats into `split_y_sums_l` /
   `split_y_sum_sqs_l`. Hoist `var_p` per target into SMEM once instead of
   recomputing it per (candidate, target) from global as it does today.
5. `accumulate_importance`, `compute_child_ids`, `reassign_samples` unchanged.

Deleted for ET: `build_hist_privatised`, `prefix_sum_bins`, and the
`hist_counts` / `hist_y_sums` / `hist_y_sum_sqs` / `cum_counts` / `cum_y_sums` /
`cum_y_sum_sqs` allocations. Level-scoped VRAM goes from ~6.5 GB to ~13 MB.

`WaveState::allocate` takes an `use_et` flag and skips the six big tensors;
`wave_byte_cost` / `pick_wave_size` get an ET arm so the budget stops binding.

**Why this is lower-risk than it sounds**: the threshold stream is preserved
exactly, so the output should be near-bit-identical to the current GPU path,
differing only in f32 accumulation order in the left sums. The three recorded
Pearson means should barely move. If they drop materially, the hash chain was
not reproduced faithfully.

Expected: removes ~95% of ET's DRAM traffic. The new dominant term is two sample
scans still carrying the `n_active_nodes` amplification, roughly 2 GB per
full-width level. Somewhere in the 10-20x range against one core. This is the
step that either clears the bar or does not.

### Step 4: shrink the sample scan

The CUDA prior art says to budget for this rather than treat it as contingent:
XGBoost's `RowPartitioner` and cuML RF's equivalent both exist precisely because
this amplification is the thing that kills a naive implementation. Skip only if
step 3 already clears 10x.

(a) **u8-pack `feature_data`.** `fit_scenic_batches_gpu` currently widens u8
bins to u32 before upload (40 MB instead of 10 MB at this shape), and every
sample-scanning kernel reads it at 4 bytes per bin with `n_active_nodes`
amplification. Pack 4 bins per u32, unpack with shift/mask. 4x on the term that
is now dominant. Mechanical; touches `scan_slot_bin_range`,
`draw_and_accumulate_split_stats_et`, `build_hist_privatised` and
`reassign_samples`. The archive doc dismisses this as "not where the pressure
is", which is right on VRAM capacity and wrong on bandwidth.

(b) **Sorted sample index.** Every (node, slot) workgroup scans all n_samples
and tests `sample_to_node[s] == node`, so read amplification is
`n_active_nodes` (100 here). The CPU pays none of it: `build_node_multi_sparse`
partitions `sample_slice` in place (`scenic.rs:1064-1078`). Fix is the standard
GPU-hist one, an index array sorted by node with per-node offsets, rebuilt each
level by a counting sort over at most `max_active_nodes` buckets. Kills the
amplification entirely. Bigger change; do (a) first, it is ~10x less work.

### Step 5 (free once step 3 lands): wave-size sweep

`DEFAULT_WAVE_SIZE = 8` was chosen under a VRAM constraint that no longer binds
for ET. Sweep 8 / 16 / 32 / 64 on the micro-bench. More trees per launch means
fewer launches and less latency exposure. Pure measurement, no design work.

Fold in the cheap driver hygiene while here, each individually measurable:

- The terminal-wave shadow `WaveState::allocate` (`scenic_gpu.rs:3037-3048`) is
  a GB-scale allocation per batch that its own comment argues is unnecessary:
  every index derives from `n_active_nodes` / `k_feats`, never `wave_size`.
  Thread `this_wave` into `run_wave_bfs` and delete it.
- `mult_host.fill(1)` re-uploads an all-ones `this_wave x n_samples` buffer every
  wave when subsampling is off (~645 MB across an e2e run). Upload once or skip
  the tensor.
- `compute_child_ids` runs `CubeDim::new_1d(1)`: eight threads on the whole GPU
  doing a serial dependent scan, once per level, ~40k launches across an e2e
  run. Negligible today, relatively larger once everything else is fast. Make it
  a workgroup scan over `is_internal`.

### Decision point

Run the full `benches/gpu_scenic_bench.rs` (both matrices, and add 50k/100k cell
rows to the kernel matrix). If ET e2e beats the 229.28s CPU number, the thesis
holds. Then decide on RF.

### RF, if pursued afterwards

Not planned in detail here, but the levers are known and the fidelity answer
("hold existing gates") unlocks both:

- **Fused scan + evaluate.** The cumulative histogram never needs to exist in
  global memory. Tile over targets in chunks of 8: prefix-sum a 256 x 8 tile in
  SMEM (8 KB) via `plane_inclusive_sum`, accumulate partial `S_L` / `S_R` into a
  256-float accumulator, next chunk. Keep each thread's sample bins in registers
  across chunks so `feature_data` is read once, not once per chunk. Reads the raw
  histogram once, writes one winner per slot. Deletes `prefix_sum_bins` and the
  three `cum_*` buffers for RF too. ~5x on its own. This is XGBoost's
  SMEM-histogram path; keep the existing global path behind a `const` threshold
  for shapes where the tile does not fit, matching how they choose.
- **`N_BINS` as a comptime parameter.** Fixed at 256 today. LightGBM defaults to
  255 and XGBoost to 256, but both run at 63/64 routinely with small accuracy
  cost, and at `min_samples_leaf = 50` on 10k cells a deep node holds ~100
  samples across 256 bins. 4x on the histogram, gated on the 0.95 Pearson floor.
- **Histogram subtraction.** Build the smaller child, subtract from the parent.
  Universal in GBDT and it applies here. Halves the build pass.
- **The sum-of-squares histogram is not needed for the search.** Since
  `wl*vl_k + wr*vr_k = (ssyl_k + ssyr_k)/n - syl_k²/(n·nl) - syr_k²/(n·nr)` and
  `ssyl_k + ssyr_k = ssq_k` is constant in `thr`, the argmax reduces to
  maximising `S_L(thr)/nl + S_R(thr)/nr`. Halves the remaining histogram.
  Caveat: the current code clamps each per-target variance at zero before
  summing and the algebraic form does not, so this is a real behaviour change
  where f32 cancellation drives a variance slightly negative. Gate on
  `phase3_random_forest_pearson >= 0.95`.

Worth noting for motivation: GPU RF (1239s) is already *faster* than GPU ET
(1640s) despite evaluating 255 thresholds per slot instead of 1, and despite
costing the CPU nearly 2x more. The threshold sweep is free on the GPU. The
pipeline is entirely bound by moving the histogram through DRAM.

## Verification

Per step:
```bash
cargo test --features gpu,single-cell --test scenic_gpu
cargo bench --features gpu,single-cell --bench gpu_scenic_micro
```

At milestones:
```bash
cargo bench --features gpu,single-cell --bench gpu_scenic_bench   # full, hours
cargo test --no-default-features                                  # CI pass 1
cargo test --features single-cell,multi-modal                     # CI pass 2
cargo clippy --features gpu,single-cell --all-targets
cargo fmt
```

Success: ET micro ratio >= 10x vs one core, all `tests/scenic_gpu.rs` gates
green, the three recorded Pearson means within 0.005 of baseline, and the full
e2e ET row below 229.28s.

Finally, correct the wave-size claim (root cause #3 and the "highest-leverage
experiment left" paragraph) in `docs/archive/scenic_gpu_experiment.md` and
replace the archive framing with whatever the numbers end up saying.
