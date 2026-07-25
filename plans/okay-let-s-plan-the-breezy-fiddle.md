# SCENIC GPU: make the ET path actually fast

Status: **ExtraTrees done. 1.49x -> 28.33x per batch, and end-to-end it went
from 7.2x slower than the CPU to 3.67x faster. RandomForest untouched.**

This file started as a plan. It is now mostly a record, because measurement
killed four of its six steps and the two that survived worked for reasons the
plan got wrong. Kept in that form deliberately: the dead ends are the expensive
part to rediscover.

## Context

`docs/archive/scenic_gpu_experiment.md` concluded the GPU SCENIC tree regressor
was a hardware loss on Apple Silicon (7.2x slower than CPU end-to-end for ET,
2.9x for RF) and parked it. It attributed the gap to the CPU getting a rayon
fan-out over gene batches while the GPU has one queue, and concluded "the gap is
the hardware, not the code".

The arithmetic was right and the conclusion was wrong. `fit_multi_trees_sparse`
is sequential over trees (`scenic.rs:1513`) and the e2e CPU driver fans out with
`into_par_iter` over batches (`scenic_metacells.rs:480`), so the measured
speedup of the fan-out is ~9.1x and that is the bar the GPU has to clear. It was
at 1.49x. A GPU beating one CPU core by 1.5x is a statement about the kernel,
not the hardware.

## Result

Reference shape, one e2e gene batch: 10k cells, 1000 TFs, 64 targets,
`max_depth` 10, `min_samples_leaf` 50. Ratio is GPU wall clock against one
sequential CPU core.

| cell | start | end |
|---|---|---|
| `et_8t` | 1.55x | **18.29x** |
| `et_32t` | 1.49x | **28.33x** |
| `et_multibatch` (4 batches) | 1.51x | **27.76x** |
| `rf_*` | 1.29x | 1.29x (untouched) |
| ET wave VRAM | 6.10 GiB | **0.012 GiB** |
| `phase2` test, GPU | 9.1s | **1.5s** (CPU 22.2s) |

End-to-end, `gpu_scenic_bench`, 4000 targets in ~63 batches at 10k cells:

| row | archive | now |
|---|---|---|
| ET e2e CPU | 229.28s | 236.46s |
| ET e2e GPU | 1640.50s | **64.44s** |
| **ET verdict** | GPU 7.2x slower | **GPU 3.67x faster** |
| ET kernel matrix, 10k | 1.53x | **32.8x** |
| RF e2e CPU | 427.36s | 187.39s (see below) |
| RF e2e GPU | 1239.24s | 1131.63s |

The archive's central claim is false for ExtraTrees. GPU e2e is 25.5x faster
than it was and beats the 10-core CPU by 3.67x.

**Caveat on the RF CPU baseline.** It moved from 427.36s to 187.39s on a code
path this work never touched, while ET CPU reproduced (229s vs 236s). The
likely cause is thermal: the bench runs `rf_e2e_cpu` straight after
`et_e2e_gpu`, which used to be 27 minutes of saturating GPU work and is now
one minute. If that holds, the archive's RF baseline was throttled and RF is
really 6.0x behind rather than 2.9x. Unproven without an isolated re-run. The
bench has an ordering dependency that should be fixed regardless.

Every gate in `tests/scenic_gpu.rs` holds at its baseline value throughout:
phase2 0.993, RF 0.988, RF+bootstrap 0.976, roundtrips 1.000, et_still_works
0.729. Both CI feature passes green, clippy clean.

## What was actually wrong

Two things, and neither was in the original plan.

**1. ExtraTrees built a histogram it did not read.** `n_thresholds` defaults to
1. The old path built a 256-bin x n_targets cumulative table per (tree, node,
slot) and read exactly one row out of it. `build_hist_privatised` was 69% of ET
GPU time and `prefix_sum_bins` 31%, and both existed only to produce 255 bins
nobody looked at.

Fixed by three kernels that reduce the node's bin range, draw the threshold from
it, and sum the left side at that threshold directly. The threshold hash chain
is reproduced verbatim so the RNG stream, and therefore the trees, are
unchanged.

**2. The replacement kernel re-read its metadata once per target.** With threads
laid out as (stripe, target), all 64 target-threads in a stripe re-read the same
`sample_to_node`, `sample_multiplicity` and `feature_data`. That is
`n_samples * n_targets` metadata reads per workgroup where `n_samples` would do.

Fixed by testing each sample once per 128-wide block and compacting the
survivors into shared memory via a plane-based workgroup exclusive scan, so the
drain loop runs over the match count rather than the block width. This was the
single largest win: 8.06x -> 28.33x.

Supporting changes that mattered: Y uploaded dense as well as sparse (2.5 MB at
this shape) so the target sits on the thread axis and the reads coalesce; wave
size raised from 8 to 32, which only became possible once the histogram stopped
sizing the allocation.

## What the measurements killed

Recorded because each of these looked obviously right beforehand.

**"It is bandwidth-bound on the histogram buffers."** No. Once `prefix_sum_bins`
was removed, `build_hist`'s 3.25 GB of zeroing, its full sample scan and its
global atomics were all unmeasurable. The zeroing is ~0.14s at 400 GB/s and
never mattered.

**"`prefix_sum_bins` is latency-bound on its 256-long serial dependency
chain."** No. Rewriting it with plane primitives to remove the chain made it
**3-5x slower**. The histogram layout is `[bin][target]`, so putting targets on
the thread axis (what the original code did) is perfectly coalesced, and putting
bins on the lane axis is not. Occupancy hides the dependency chain completely at
~100k workgroups per level. The kernel could not be improved in place at all;
it had to be deleted.

**"Deleting `merge_hist` is worth ~20%."** Worth 0%. It reads slot 0 only
(`count_base` carries no `+ slot` term), so it moved ~105 MB per level, not the
3.25 GB claimed. About 1% of the traffic. The change was kept anyway because it
matches the CPU's numerics and removes a launch, but it is perf-neutral.

**"Phantom nodes cost real work."** They do not. All nine non-histogram kernels
together are 0.1% of GPU time across ~700 launches per run.

**"The sample scan needs a sorted index and u8-packed features."** Both dead as
originally framed. The scan was free in the old kernel because atomics swamped
it. It became the bottleneck only *after* the atomics went, and the fix was
compaction within the workgroup, not a global sort.

**"Fixed per-call cost is inflating the single-batch harness."** No. The
4-batch cell gives an identical ratio to the 1-batch cell. Everything is linear
in total wave count.

## What worked as method

Three things paid for themselves repeatedly.

**The cubecl profiler.** `CUBECL_DEBUG_OPTION=profile-medium` plus
`CUBECL_DEBUG_LOG=stdout` gives per-kernel timings with no code changes. Every
correct decision in this work came from it; every wrong one came from reasoning
about traffic on paper. Note it serialises kernels, so it measures isolated cost
rather than pipelined wall clock, and the two can disagree sharply.

**Ablation as a ceiling check, not an attribution.** Deleting a kernel tells you
the best case for optimising it. It does *not* tell you its share, because
removing one kernel changes the memory behaviour of the others. Both readings
were needed and they disagreed for a while.

**A harness that cannot lie.** `benches/gpu_scenic_micro.rs` runs in about a
minute and checksums the returned importances. That guard caught a fictional
275x: kernels launch via `launch_unchecked`, so a binding that busts a device
limit does no work, reports no error, and returns zeros very quickly.

## Latent bugs found and fixed

- `pick_wave_size` budgeted *total* VRAM while wgpu limits each *binding*. A
  wave could fit the byte budget with a single tensor over the device limit,
  which fails silently through `launch_unchecked`. Now both ceilings are checked
  and the error names them. This device reports 4.00 GiB per binding.
- `viable_max_active_nodes` computes `leaf_cap` as `n_samples / (2 *
  min_samples_leaf)`, which counts splits rather than nodes at a level. A level
  can hold up to `n_samples / min_samples_leaf` nodes, so the bound is loose by
  2x in the unsafe direction and `reassign_samples` can index out of range.
  **Not fixed.** Doubling it changes no Pearson value at any tested shape, so it
  does not bite in practice, but it is a real hazard for a perfectly balanced
  tree and deserves a defensive guard.
- `tests/scenic_gpu.rs` is gated on `large_scale_diagnostics` as well as `gpu`
  and `single-cell`. The CI gpu job runs `--features gpu`, under which this file
  compiles to **zero tests**. The SCENIC GPU suite has never run in CI.
  **Not fixed**, needs a decision on whether to relax the gate or change CI.

## What remains

### RandomForest

Untouched at 1.29x. It genuinely needs the cumulative histogram, so the ET
approach does not port, but the levers are known and quantified:

- **Fixed-point integer atomics in `build_hist_privatised`.** Its cost is the
  CAS-retry-loop float add per (sample, target) in global memory, ~1.4e9 per fit
  at the reference shape, which matches its measured 1.36s almost exactly. WGSL
  has no float `atomicAdd`; rescaling Y to fixed-point allows native
  `Atomic::<i32>::fetch_add`, removing the retry loop and making the
  accumulation order-independent (hence deterministic by construction rather
  than by luck). Constraint: no 64-bit atomics in WGSL, so this needs a range
  analysis on `n_samples * max|y| * scale`.
- **Fuse the scan into evaluate.** `cum_*` never needs to exist in DRAM.
  Critically, the fused kernel must keep targets on the thread axis or it hits
  the same coalescing wall that killed the plane scan. Takes prefix+evaluate
  from ~9.75 GB to ~3.25 GB per level.
- **Drop the sum-of-squares histogram.** `wl*vl_k + wr*vr_k = (ssyl_k +
  ssyr_k)/n - syl_k^2/(n*nl) - syr_k^2/(n*nr)` and `ssyl_k + ssyr_k = ssq_k` is
  constant in `thr`, so the argmax reduces to maximising `S_L/nl + S_R/nr`. This
  is cuML's standard MSE formulation, not a novel rewrite. Caveat: the current
  code clamps each per-target variance at zero before summing and the algebraic
  form does not, so it is a real behaviour change where f32 cancellation drives
  a variance negative. Gate on `phase3_random_forest_pearson >= 0.95`.
- **`N_BINS` as a comptime parameter.** Fixed at 256. LightGBM defaults to 255
  and XGBoost to 256, but both run at 63/64 routinely.
- **Histogram subtraction.** Build the smaller child, subtract from the parent.
  Applies to RF. Does *not* apply to restructured ET: min/max of a union is not
  derivable by subtraction and ET's left sums sit at a per-node random
  threshold.

Worth noting GPU RF was already *faster* than GPU ET before this work (1239s vs
1640s e2e) despite evaluating 255 thresholds per slot instead of 1. The
threshold sweep is free; the pipeline was entirely bound by moving the histogram
through DRAM.

### Documentation

`docs/archive/scenic_gpu_experiment.md` needs correcting on three counts, plus
its overall verdict:

1. Root cause #3 and the "single highest-leverage experiment left" paragraph
   describe a wave size of 4. The bench overrides the budget to 12 GiB
   (`gpu_scenic_bench.rs:79`), so the measured runs were at wave 8. The
   in-place prefix sum it recommends would have bought nothing.
2. "The VRAM hog is the sum buffers, not the features" is right on capacity and
   wrong on bandwidth, though this turned out not to matter since neither was
   the bottleneck.
3. "wgpu/Metal has no host knob to run independent batches concurrently" is not
   accurate, and in any case the right lever is fatter launches, not concurrent
   queues.

## Verification

```bash
cargo test --release --features gpu,single-cell,large_scale_diagnostics --test scenic_gpu -- --nocapture
cargo bench --features gpu,single-cell --bench gpu_scenic_micro
```

Watch the three Pearson means (phase2 0.993, RF 0.988, RF+bootstrap 0.976) and
the ratio column. At milestones:

```bash
cargo bench --features gpu,single-cell --bench gpu_scenic_bench   # full, ~1h
cargo test --no-default-features                                  # CI pass 1
cargo test --features single-cell,multi-modal                     # CI pass 2
cargo clippy --features gpu,single-cell --all-targets && cargo fmt
```
