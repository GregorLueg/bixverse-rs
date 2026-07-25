# SCENIC GPU: RandomForest, revised plan

Supersedes the step list in `plans/scenic-gpu-randomforest.md`. Fold this into
that file when the work starts; the corrections at the bottom apply to it and to
`plans/scenic-gpu-extratrees.md` and `docs/scenic_gpu.md`.

## Status

**RandomForest on GPU went 1.18x -> 4.83x against one CPU core, and its wave
VRAM went 6.10 GiB -> 0.010 GiB.** All three Pearson gates hold: ET 0.993
unchanged, RF 0.987 against a 0.988 baseline, RF+bootstrap 0.975 against 0.976.

| step | state |
|---|---|
| -1 RF test coverage | **done**, calibrated, 12 tests pass in 7.6s debug |
| 0 `prefix_sum_bins` fixes | **dead end, reverted.** Both hypotheses measured wrong |
| 1 per-node gather | **done.** 1.18x -> 1.21x, as predicted a near-null result |
| 2 fused kernel | **done.** 1.21x -> 4.83x |

| cell | before | after |
|---|---:|---:|
| `rf_8t` | 1.16x | **3.75x** |
| `rf_32t` | 1.18x | **4.83x** |
| `rf_multibatch` | 1.18x | **4.63x** |
| RF wave size | 8 | **32** |
| RF wave VRAM | 6.10 GiB | **0.010 GiB** |
| `phase3_random_forest_pearson` GPU | 5.9s | **2.0s** |

**End to end it still loses.** Full `gpu_scenic_bench` at 20% density: RF
251.35s GPU against 87.62s CPU, i.e. 2.87x slower, down from 6.0x. The GPU side
went 1131.63s -> 251.35s but the bar moved too: the measured rayon fan-out at
this density is **12.8x**, not the 8.25x quoted from the old 50%-density
baseline. RF needs 12.8x per batch and has 4.99x.

So the shipping recommendation is unchanged: RandomForest stays CPU-preferred.
What did change is that it now runs at 1M cells instead of refusing. Remaining
levers are in "What is left" at the bottom.

### What the first measurements settled

**Step 0 was wrong twice, and one half of it was a 31% regression.** Three runs
on `rf_32t`, same machine state:

| variant | GPU | ratio |
|---|---:|---:|
| baseline | 1.88s | 1.18x |
| register-carried prefix sums | 1.91s | 1.17x |
| + `WORKGROUP_64` dispatch | 2.51s | 0.86x |

The 256-deep read-after-write chain through global memory costs nothing:
occupancy hides it, exactly as `plans/scenic-gpu-extratrees.md:110` concluded and
my caveat on it denied. And narrowing the dispatch to stop half the workgroup
idling costs 31%: those threads are free, and four SIMD groups per workgroup hide
memory latency that two cannot. Both are now recorded as negative results in the
`prefix_sum_bins` doc comment so nobody retries them. The register carry is kept
(neutral, bit-identical); the width change is reverted.

The baseline 1.18x reproduces `docs/scenic_gpu.md`'s 1.17x at 20% density
exactly, so the harness is trustworthy.

**The profile, and it validates the fused design by a different route.**
`rf_32t`, 90 launches, `CUBECL_DEBUG_OPTION=profile-medium`:

| kernel | total | share |
|---|---:|---:|
| `PrefixSumBins` | 1829 ms | 46.1% |
| `BuildHistPrivatised` | 1785 ms | 45.0% |
| `EvaluateSplitsRf` | 345 ms | 8.7% |
| the other nine | 6.6 ms | 0.1% |

`prefix_sum_bins` moves 27.3 GB per run in 813 ms, i.e. **34 GB/s against ~400
available**. Not bandwidth bound, not chain bound: issue bound, at four memory
ops per two flops in its inner loop. It cannot be fixed in place, which is the
same conclusion the ET work reached about the same kernel. **91% of RF GPU time
is in the two kernels Step 2 deletes outright.**

**`max_shared_memory_size` is 32768 B on this M1 Max.** A fused 64-bin histogram
at 64 targets needs 16384 B of sums plus counts, per-bin scores and compaction
scratch, about 17.9 KB, so it fits with room. 128 bins needs ~34.8 KB and busts.
`BIN_SHIFT = 2` it is, and the split-target variant is not needed on this device.

**RF runs at wave 8, ET at 32**, confirmed by `report_shape`: 6.10 GiB against
0.048 GiB at the reference shape.

**`phase3_rf_pearson_small` calibrated** to 120 trees, floor 0.95. Measured
cpu-gpu against cpu-cpu at 0.878/0.891 (30 trees), 0.966/0.966 (120), 0.982/0.984
(300): the GPU sits on the noise ceiling at every tree count.

## Context

ExtraTrees on GPU beats the 10-core CPU by 3.67x end to end after the rewrite
recorded in `plans/scenic-gpu-extratrees.md`. RandomForest was left untouched at
1.28x against a single core and loses 6.0x end to end. The rayon fan-out over
gene batches is worth ~8.25x, so that is the bar.

The existing RF plan proposed five stacked steps and recommended doing the first
two. Reviewing the code changed that recommendation. Three things came out of it.

**RF has never run at wave 32.** `viable_max_active_nodes(10, 10000, 50) = 100`,
so `wave_byte_cost` is 0.76 GiB per tree at the reference shape. `pick_wave_size`
lands on **8** under the bench's 12 GiB budget and **4** under the 4 GiB library
default. ET runs at 32. Every RF profile number on record is a wave-8
measurement compared against ET at wave 32, and RF pays 4x the launch count and
gets 4x fewer workgroups per grid for it. Killing the histogram fixes this as a
side effect, and it is worth an independent 2-4x that no projection has priced.

**`prefix_sum_bins` is not bandwidth-bound, it is dependency-bound, and two
one-line fixes may capture most of what "fuse the scan into evaluate" was worth.**
It moves 4H of the 8H total histogram traffic (H = 212 GB for a 250-tree fit at
the reference shape, so 4H = 848 GB, ~2.1 s at 400 GB/s) but measures at 36% of
a 19.24 s run, i.e. ~6.9 s, an effective 120 GB/s. Two reasons, both trivially
fixable. Lines 518/548/549 read `cum_*[prev]`, which is a global load of the
value the same thread wrote one iteration earlier: a 256-deep read-after-write
chain through DRAM that the compiler cannot forward. And `launch_prefix_sum`
dispatches `WORKGROUP_128` with `n_targets = 64`, so half the workgroup falls
straight out of `while k < n_targets` and idles through the whole kernel.

**Step 5 of the old plan (histogram subtraction) is actively harmful** under the
design below, and step 2's proposed reuse of the ET compaction pattern is the
wrong machinery for RF. Details in the corrections section.

## Target design

One kernel per level replaces four. Workgroup owns `(slot, node, tree)`, 128
threads, thread `k` owns target `k`.

1. Walk the node's own sample list (from a per-level gather, see Step 1), not all
   `n_samples`. No membership test, no compaction, no `sync_cube` in the hot loop.
2. Accumulate `s_hist[bin_coarse * n_targets + k] += mult * y_dense[sid * n_targets + k]`
   in **shared memory**. Thread `k` is the sole writer of column `k`, so no
   atomics. All 64 drain threads share one `bin`, so the 64 words are consecutive
   and bank-conflict free.
3. Scan bins 0..N_BINS_COARSE with the running cumulative `c_k` in a **register**.
   No prefix tensor, in DRAM or anywhere else.
4. Score each bin from `G(thr) = S_L/nl + S_R/nr` with `S_L = Σ_k c_k²`,
   `S_R = Σ_k (sy_k − c_k)²`. One cross-target reduction per bin (`plane_sum`
   over two planes plus a 2-entry shared combine). Emit the slot's best.

Then a small argmax over slots per `(node, tree)`, and a `finalise_split_stats_rf`
pass that recomputes the winner's `syl_k` and `ssyl_k` from the actual samples at
**full 256-bin precision**. That last point matters: everything flowing down the
tree is exact, so coarsening changes only which split is chosen, never the
arithmetic of anything downstream.

Three things this design turns on.

**Dropping `hist_y_sum_sqs` from the search.** `wl·vl_k + wr·vr_k =
(ssyl_k + ssyr_k)/n − syl_k²/(n·nl) − syr_k²/(n·nr)` and `ssyl_k + ssyr_k = ssq_k`
is constant in `thr`, so `argmax score ≡ argmax G`. Standard cuML MSE
formulation. Two consequences the old plan missed: it drops the per-target
`max(var, 0)` clamp, and `best_score` starts at `0.0` with strict `>`, which is an
**acceptance** gate, not just a tie-break. `G` carries an unknown per-node offset,
so the winner's true score has to be reconstructed as
`parent_var_sum − Q/n + G_win/n` with `Q = Σ_k node_y_sum_sqs[k]`. Skip that and
you silently accept zero-gain splits at every node where no positive split exists,
which at depth 9 with ~63 samples per node is common.

**GPU-only bin coarsening.** `bin_c = feature_bin(..) >> BIN_SHIFT`, winner
widened back as `(thr_c << BIN_SHIFT) | ((1 << BIN_SHIFT) - 1)`. Exact:
`b >> S ≤ thr_c  ⟺  b ≤ (thr_c << S) | (2^S − 1)`. `QuantisedStore` stays at 256
u8 bins, `reassign_samples` is untouched, the CPU path is untouched. The argument
for 64 bins is not "compromise": at depth 7-9 a node holds 63-200 samples, so at
most that many bins are occupied out of 256, and a 64-bin dense histogram at 64
targets is already the minimum useful size.

**Shared-memory residency.** At `BIN_SHIFT = 2`: `s_hist` 64x64 f32 = 16,384 B,
`s_counts` 256 B, `s_binscore` 256 B, scratch ~512 B. About 17.4 KB. Fits a 32 KB
threadgroup budget; 128 bins does not (32 KB for sums alone). Post-change RF wave
cost is `wave · max_active_nodes · k_feats · 4 arrays · 4 B` plus the gather, i.e.
**~2.9 MB** at the reference shape against 6.10 GiB today, and **~16 MB** at 1M
cells against 8.39 GB at wave 1. RF goes to wave 32 permanently and the 1M-cell
refusal disappears. That capacity result is worth shipping even if the speed work
lands short.

## What was built

Three commits after the gather, replacing four kernels with three.

**`build_score_rf_fused`** (one workgroup per slot/node/tree) builds the slot's
histogram in threadgroup memory, prefix-sums it along bins with a register
carry, scores every candidate bin and emits the slot's winner. Nothing reaches
DRAM but four small per-slot arrays. No atomics anywhere: the whole workgroup
walks one sample at a time and thread `k` owns target `k`, so no two threads ever
touch the same histogram column.

**`reduce_slot_winners`** picks each node's best slot, single-threaded, ascending
with a strict `>` so ties go to the lowest slot.

**`finalise_split_stats_rf`** recomputes the winner's left-child sums and
sums-of-squares from the node's samples at the decided threshold, per (node,
tree) rather than per (slot, node, tree).

Three things made it fit.

**Dropping the sum-of-squares histogram from the search.** `argmax` only needs
`G = S_L/nl + S_R/nr`; `Q = Σ_k ssq_k` is a node constant. The winner's true
score is reconstructed as `P − Q/n + G/n` before the `> 0` acceptance gate,
which is the part the original plan missed: `argmax G` alone silently accepts
zero-gain splits wherever no positive split exists.

**Adaptive bin coarsening, not a fixed `BIN_SHIFT`.** The budget is
`n_bins × (n_targets+1)` floats, so `pick_gpu_bins` returns the finest count that
fits: 256 bins at 8 targets, 128 at 32, 64 at the production 64. Small-target
batches keep exact behaviour for free. The winning threshold widens back to fine
bin space before `reassign_samples` sees it, so the shared `QuantisedStore` and
the CPU path are untouched. Cost at 128 bins: 0.001 Pearson.

**Rows padded to `n_targets + 1`.** The scoring phase gives consecutive threads
consecutive bins; at an unpadded stride of 64 every one of them resolves to the
same shared-memory bank.

Two more things that mattered as much as the kernel:

- **`WaveLayout`** replaced the `use_et` bool in `pick_wave_size` and
  `WaveState::allocate`. Without it the fused path still allocated the 6.10 GiB
  histogram it never touches and stayed pinned at wave 8. This is where most of
  the speedup actually came from.
- **Staging the sample tile in shared memory.** Id, multiplicity and bin are
  workgroup-uniform, so reading them per thread per sample was three redundant
  global loads on the critical path. Worth 6%.

## What is left

Profile after the rewrite, `rf_32t`, 30 launches:

| kernel | share |
|---|---:|
| `BuildScoreRfFused` | 92.7% |
| `FinaliseSplitStatsRf` | 3.9% |
| `InitRootStats` | 2.4% |
| everything else (12 kernels) | 1.0% |

One kernel is the whole cost now. It is latency bound on the per-sample
`y_dense` fetch: 22 KB of shared memory per workgroup against a 32 KB budget
means **one resident workgroup per core**, so 32 cores × 128 threads = 4096
threads in flight and very little to hide a ~400-cycle load behind.

Levers, in the order I would try them:

1. **Unroll the sample loop.** Two or four `y_dense` fetches in flight per thread
   costs nothing in accuracy and directly attacks the stall. Safe: thread `k`
   owns column `k`, so repeated bins just serialise within one thread.
2. **Get under 16 KB for two resident workgroups.** Needs 32 bins at 64 targets,
   which is a real accuracy trade and should be measured against the 0.95 floor
   before being taken seriously.
3. **Split the target axis** so two workgroups each handle 32 targets. Halves
   shared memory per workgroup and doubles occupancy, at the cost of combining
   partial `S_L`/`S_R` across workgroups.

Whether any of this reaches 8.25x is open. 4.83x is already a shipping-worthy
result for a path that was 6.0x behind end to end, and the VRAM collapse means
RandomForest now runs at 1M cells where it previously refused.

## Steps

Each step gates on the two RF Pearson floors (`phase3_random_forest_pearson`
0.988, `phase3_rf_bootstrap_pearson` 0.976, both floored at 0.95) plus the new
tests from Step -1, and is measured on `rf_32t` / `rf_multibatch`.

### Step -1: RF test coverage

RF has zero CI coverage today. Both fidelity tests sit behind
`large_scale_diagnostics`, which no workflow enables. Five kernels are about to
be rewritten on that path.

- `rf_gpu_matches_cpu_top10`, ungated, toy shape, mirroring
  `extra_trees_gpu_matches_cpu_top10` (`tests/scenic_gpu.rs:233`). Runs on
  lavapipe and Metal in seconds.
- `phase3_rf_pearson_small`, ungated, ~2k cells / 100 features / 50 trees, floor
  0.90. The 0.95 versions stay gated as milestone gates; the current shape is far
  too slow for lavapipe.
- `coarse_threshold_roundtrip`, host-only: for `shift in 1..=3`, all `thr_c`, all
  `b in 0..256`, assert `(b >> shift <= thr_c) == (b <= ((thr_c << shift) | ((1 << shift) - 1)))`.
- `benches/gpu_scenic_micro.rs`: add a `SCENIC_MICRO_ONLY=rf_32t` filter (there is
  no per-cell selection today, so an RF iteration costs a full ET run), and print
  `max_shared_memory_size`, the chosen wave size for both learners, and
  `BIN_SHIFT` in `report_shape` (line 403).

### Step 0: dead, see the status section

Both fixes measured wrong and one was a 31% regression. Reverted, with the
negative results recorded in the `prefix_sum_bins` doc comment. The bench
cooldown from that work is kept, and so is the module-doc correction.

The one thing worth carrying forward: `prefix_sum_bins` cannot be repaired in
place. It is issue bound at 34 GB/s of ~400, and every attempt to shorten its
chain or tighten its dispatch has now failed, here and in the ET work. Delete it.

### Step 1: per-node sample gather

Three small kernels after `reassign_samples`, plus `node_sample_offsets` /
`node_sample_ids` in `WaveState`:

- `count_node_samples`: grid (sample-blocks, 1, wave), `Atomic::fetch_add` into
  `node_sample_counts`.
- `scan_node_offsets`: grid (wave), single thread, serial exclusive prefix over
  ≤1024 nodes. Copy the `compute_child_ids:2026` shape.
- `scatter_node_samples`: grid (sample-blocks, 1, wave), atomic cursor into
  `node_sample_ids`.

Wire the gathered list into the existing `build_hist_privatised` sample loop as a
drop-in for `while s < n_samples`. Nothing else changes.

**Be honest about the gate here: expect little or no speedup.** The scan it
removes is 158x wasteful at depth (10,000 samples tested to find ~63), but
`sample_to_node` is 320 KB and lives in L2, so my own arithmetic puts the scan at
low single-digit percent of runtime. The reason to do it is that it makes the
fused kernel of Step 2 **barrier-free and portable**: no compaction, no
`plane_exclusive_sum`, no `sync_cube` in the hot loop, and no plane-viability
fallback branch. A flat measurement is not a failure. A Pearson move is.

Cost: `node_sample_ids` is `wave × n_samples × 4` = 1.28 MB at reference, same
order as `sample_to_node`. One caveat to put in the docstring: the atomic-cursor
scatter makes within-node sample order nondeterministic, so f32 summation order
varies run to run. `accumulate_importance:1987` already does this via its CAS
loop, so it is not a new class of problem.

### Step 2: the fused kernel

Build it with `BIN_SHIFT` and `use_smem` as comptime knobs, and land it in four
substeps so a Pearson move is attributable.

1. **`finalise_split_stats_rf` first**, wired into the *existing*
   `evaluate_splits_rf`, which stops writing `split_y_sums_l` /
   `split_y_sum_sqs_l`. Body is `accumulate_split_stats_et:1370-1396` with the
   threshold supplied rather than drawn, grid `(node, tree)`, so 1/31 the cost of
   the build. Existing tests must not move at all.
2. **`build_score_rf_fused` at `BIN_SHIFT = 0`, `use_smem = false`** (256 bins,
   histogram still in DRAM), plus `reduce_slot_winners`. This is a pure fusion
   with no accuracy change, so it gets a **differential test**
   (`rf_fused_matches_dram_path`, ungated, toy shape) comparing `split_feature`,
   `split_threshold` and `split_n_left` element-wise against the old path. Keep
   the old kernels alive for this. Separating "the fusion is wrong" from
   "coarsening moved the answer" is what stops this costing a week.
3. **Add the `> 0` score reconstruction** and drop `hist_y_sum_sqs` from the
   search. Verify the RF Pearson gates do not move.
4. **Flip to `BIN_SHIFT = 2` and enable shared memory.** Gate on
   `smem_hist_viable()`, host-side, mirroring `plane_compact_viable:2591`, because
   an oversized `SharedMemory` allocation almost certainly fails through
   `launch_unchecked` the same silent-zeros way the binding-limit bug did.
5. Delete the dead kernels and shrink `WaveState`.

**Gate after 4.** Pearson below 0.96, try `BIN_SHIFT = 1` (128 bins) with the
target axis split into two halves of 32 (16 KB, two workgroups per slot; build
that comptime variant anyway as the portability tier for any device reporting the
WebGPU default 16384). Below 0.95 at `BIN_SHIFT = 1`, keep the fused DRAM path at
256 bins: you still get the fusion, the atomics removal and the gather, just not
shared-memory residency or the wave-32 capacity win.

Two occupancy caveats to watch, not to pre-optimise. `SharedMemory::new` is
comptime-sized and `n_targets` is runtime, so the full 16 KB is allocated even on
a trailing 8-target batch. And 17.4 KB out of a 32 KB per-core budget permits one
resident threadgroup per core, which hurts latency hiding on the `y_dense` reads.
The split-target variant is the mitigation for both.

## Files

All in `src/gpu/sc_gpu/scenic_gpu.rs` unless stated.

**Add.** Consts `BIN_SHIFT`, `N_BINS_COARSE`, `SMEM_TARGET_STRIDE` near line 62.
Kernels `count_node_samples`, `scan_node_offsets`, `scatter_node_samples`,
`build_score_rf_fused`, `reduce_slot_winners`, `finalise_split_stats_rf`, each
with a `launch_*` host wrapper in the 2337-3288 block. Helpers
`smem_hist_viable<R>(client, bins, padded_targets)` next to
`plane_compact_viable:2591`, and `wave_byte_cost_rf` next to
`wave_byte_cost_et:3549`.

**Modify.** `WaveState` (3289-3383): drop the six `hist_*` / `cum_*` fields, add
`node_sample_counts`, `node_sample_offsets`, `node_sample_ids`, and
`slot_best_{score,thr,n_left,valid}` at `[wave, max_active_nodes, k_feats]`.
`WaveState::allocate` (3404-3470) follows, stubbing the RF-only fields to length 1
when `use_et`, same pattern as today. `pick_wave_size` (3598-3651): RF arm calls
`wave_byte_cost_rf`, and its `largest` closure becomes
`max(w · max_active_nodes · k_feats · 4, w · n_samples · 4)` because
`node_sample_ids` is the biggest RF binding once the histogram goes. `run_wave_bfs`
(3784-3798) drops `sy_gpu`; hoist `launch_init_root_stats` (3871) out of the
`use_et` block so both paths seed root stats from `y_dense`; replace the RF branch
(3947-4025) with gather / fused / reduce / finalise.
`fit_scenic_batches_gpu` (4240-4244) drops the `SparseYGpu` construction.

**Delete** at substep 2.5: `build_hist_privatised` (229-313), `merge_hist`
(355-407), `prefix_sum_bins` (465-554), `evaluate_splits_rf` (696-1010), their
four `launch_*` wrappers, and `SparseYGpu` (3659-3783). Keep
`atomic_add_f32_bits:169-182`, still the only mechanism `accumulate_importance`
has. No fallback needed: the gather makes `build_score_rf_fused` plane-free, and
its `use_smem = false` variant covers low-shared-memory devices.

## Verification

```bash
# every step
cargo test --release --features gpu,single-cell --test scenic_gpu -- --nocapture
cargo clippy --features gpu,single-cell --all-targets && cargo fmt --check

# milestone gates
cargo test --release --features gpu,single-cell,large_scale_diagnostics --test scenic_gpu -- --nocapture
CUBECL_DEBUG_OPTION=profile-medium CUBECL_DEBUG_LOG=stdout \
  cargo bench --features gpu,single-cell --bench gpu_scenic_micro

# before merge
cargo test --no-default-features
cargo test --features single-cell,multi-modal
```

Watch `rf_32t` / `rf_multibatch` and the two RF Pearson means. `gpu_scenic_bench`
at milestones only (~1h). Note `gpu_scenic_bench` has no checksum guard at all, so
a silently dead kernel reads there as a spectacular win; the micro bench's guard
(`checksum > 0.5 * n_targets`, line 318) is the only liveness check and it does
not compare values against the CPU. Add a Pearson column to the micro bench at
reduced tree count as part of Step -1.

## Corrections to the existing plan files

`plans/scenic-gpu-randomforest.md`:

- **Step 5 (histogram subtraction): delete.** Under a shared-memory-resident
  histogram it needs the parent resident in DRAM, ~1.6 GB at reference, which is
  exactly the traffic the design removes. It is only a win for the DRAM design
  being abandoned.
- **Step 2's compaction reuse: replace with the gather.** The ET pattern costs
  three `sync_cube` per 128-sample block; at 79 blocks × 423k workgroups that is
  ~2e8 barriers in the fused kernel, for a scan the gather removes entirely.
- **Step 3: incomplete.** Add the `> 0` acceptance-gate reconstruction, and note
  that the winner's `ssyl_k` is still needed by `accumulate_importance:1970` and
  `propagate_child_stats`, with `finalise_split_stats_rf` as the answer.
- **Step 4: wrong on one point.** "It changes results for the CPU path too if the
  binning is shared" is false. Coarsening is GPU-only.
- **Lines 36-46:** the 57.4 / 35.8 / 6.6 profile is a wave-8 measurement and the
  file does not say so. ET's is at wave 32. The two are not comparable.
- **Lines 63-66:** the "3.25 GB per level" framing misdirects the whole step. The
  ratio is right; the cost is the global read-after-write chain, not the volume.
- **Lines 177-198:** the projection omits the wave-8 to wave-32 jump, so it
  understates the ceiling.
- **Lines 220-223:** both "cheap ET cleanups" are already done. `sy_gpu` is `None`
  when `use_et` (4240-4244) and `upload_dense_y` takes a reused scratch (3690).
- **Line 22:** the 8.25x bar derives from the 187.39s RF CPU baseline the same
  file calls suspect. Restate after the Step 0 bench fix.

`plans/scenic-gpu-extratrees.md` and `docs/scenic_gpu.md`:

- "The CI gpu job compiles `tests/scenic_gpu.rs` to zero tests" is **wrong**.
  The file gates on `single-cell, gpu` only (line 20); nine tests run in CI on both
  runners. The accurate statement is that **RF** has no CI coverage, because only
  the three Pearson tests sit behind `large_scale_diagnostics`.
- The `viable_max_active_nodes` hazard is recorded as "Not fixed", but
  `reassign_samples:1857-1862` now drops out-of-range nodes defensively. Mark it
  resolved.
- `docs/scenic_gpu.md:141` says "The bench runs `rf_e2e_cpu` straight after
  `et_e2e_gpu`" in the present tense. Both drivers now run CPU rows first. Make it
  past tense.
- `docs/scenic_gpu.md:100-114` should say RF is scheduled at wave 4 (default) or 8
  (bench) against ET at 32. It is a live handicap on the numbers in that section.
- `plans/scenic-gpu-extratrees.md:110`: the plane-scan experiment changed
  coalescing and never isolated the global dependency chain. Add the caveat; Step
  0 settles it.
