# SCENIC GPU experiment: archive and learnings

Archived synthesis of the two working docs from the `feat-faster-gpu` port
(`docs/plans/gpu-scenic-port.md` and `docs/plans/elegant-bubbling-waffle.md`).
The code stays in the tree. This doc records why it exists, what it bought, and
why it is not the default on Apple Silicon, so nobody re-reads the old speedup
table without the caveat.

Short version: the GPU tree regressor is correct and wins the microbenchmark,
but loses the real multi-batch SCENIC workload to the CPU by 3-7x on Apple
Silicon. Keep it gated, keep CPU as the default, revisit only on high-VRAM CUDA.

## What was built

A wave-scheduled, multi-tree, multi-batch ExtraTrees / RandomForest regressor on
`cubecl` + wgpu (Metal backend), the GPU sibling of the CPU SCENIC code in
`sc_analysis/scenic.rs`. GBM stays on CPU by design: all three GPU top-level
functions reject `RegressionLearner::GradientBoosting` with
`GpuNotSupportedForLearner`.

Entry points (all in `src/gpu/sc_gpu/scenic_gpu.rs`):

- `fit_scenic_batches_gpu` - the driver core. Takes pre-sliced target batches,
  uploads the feature tensor once, defers per-batch importance readbacks.
- `run_scenic_grn_gpu` / `run_scenic_grn_streaming_gpu` / `run_scenic_grn_in_memory_gpu`
  - the three top-level callers, mirroring the CPU signatures modulo `device`
  and `gpu_params`.
- `fit_multi_trees_gpu` - thin backward-compat wrapper.

Design: the CPU code is depth-first recursion; the GPU rewrite is
level-synchronous BFS. All nodes at depth `d` are processed in one set of kernel
launches, then the level advances. No tree structure is stored, importance is
the only output. Six kernels per level per tree per gene batch (feature
sub-sampling, privatised histogram build, prefix-sum over 256 bins, split
evaluation, sample reassignment, importance accumulation). Trees run
concurrently as an outer wave dimension, wave size chosen to fit a VRAM budget.

Tests: `tests/scenic_gpu.rs`. Bench: `benches/gpu_scenic_bench.rs`.

## What worked

**Per-batch kernel throughput.** On a single 64-target batch the GPU beats the
CPU at every shape from 10k cells up. This is the number the original port doc
measured, and it is real. Shape: 1k TFs x 64 targets x 250 trees,
`max_depth = 10`, `min_samples_leaf = 50`, median of 3 after 1 warmup.

| Cells | ET    | RF    |
|-------|------:|------:|
| 10k   | 1.53x | 1.34x |
| 25k   | 1.73x | 1.63x |
| 50k   | 1.77x | 1.84x |
| 75k   | 1.47x | *     |
| 100k  | *     | 2.11x |

\* CPU baseline exceeded the 300s skip threshold at that shape.

Three optimisation rounds got it there from a losing start:

- **4a -> 4b: sample-parallel histogram.** The first cut was accidentally
  bin-parallel with 256x read amplification on `feature_data`. Rewriting the
  build so each thread walks its samples once, plus on-device `next_active` to
  kill per-level host readbacks (one `.read()` per wave instead of ~30), flipped
  GPU from losing everywhere to winning from 25k up.
- **4c: min/max bin precompute + narrower workgroup for ET evaluate.** Folding
  `min_bin`/`max_bin` into the existing counts scan let the evaluate kernels skip
  dead bins; shrinking `evaluate_splits_et` to a 32-wide workgroup pushed thread
  utilisation from ~24% to ~97% at the typical `k_feats ~ 31, n_thresholds = 1`.

**Statistical parity.** The GPU output matches CPU where it needs to: per-target
Pearson 0.988-0.993 for ET/RF, RF bootstrap 0.976, byte-identical determinism
across independent batches. Correctness was never the problem.

## What did not work

The microbenchmark measured the wrong thing. Real SCENIC has thousands of target
genes, not 64. Batched at 64 targets that is ~63 batches per run, and the whole
picture inverts.

End-to-end, ~63 batches, 4000 targets, 10k cells, via the top-level drivers:

| Learner | CPU     | GPU      | Result             |
|---------|--------:|---------:|--------------------|
| ET      | 229.28s | 1640.50s | GPU 7.2x slower    |
| RF      | 427.36s | 1239.24s | GPU 2.9x slower    |

The per-batch win is genuine and the end-to-end loss is genuine. Both are true
at once.

## Root cause

Three things stack up, none of them a kernel bug:

1. **CPU gets an N-core multiplier for free.** All three CPU drivers wrap the
   batch loop in `(0..total_batches).into_par_iter()` over a shared read-only
   `QuantisedStore`. 63 batches across ~10 cores is roughly 7 batches of
   wall-clock.
2. **GPU has one queue.** Batches dispatch sequentially, and inside a batch the
   level-synchronous BFS makes level `d+1` depend on level `d`. wgpu/Metal has no
   host knob to run independent batches concurrently without contending for the
   same VRAM and the same queue.
3. **A single batch already saturates the device.** `wave_byte_cost` at these
   shapes busts the 4 GB budget, so `pick_wave_size` halves the wave from 8 to 4.
   The wave is VRAM-bound and each histogram kernel does real
   `n_samples x k_feats` work plus a node x bin x target scan that grows with
   depth. It is bandwidth-bound compute, not launch latency, so filling bubbles
   with a second batch's work would not help.

Do the arithmetic: GPU per-batch (~22s) beats one CPU core per batch (~36s), the
~1.5x the microbench shows. But the CPU runs ~10 of those at once and the GPU
runs one. `10/36` vs `1/22` is roughly 6x in the CPU's favour, which is exactly
the observed 7x (ET) / 3x (RF). The gap is the hardware, not the code.

## What the final refactor bought

The last change (single feature upload + deferred per-batch readbacks) removed
the ~40 MB `feature_data_gpu` re-upload on every batch and the blocking `.read()`
between batches. Before it, one full GPU pass blew past the 300s warmup skip and
was somewhere between 300s and unmeasurable. After it, the run finishes and we
know it is 1240-1640s. Progress from "unmeasurable" to "measurable and honestly
bad." Ship it as a correctness / measurability fix, not a competitiveness claim.

## Takeaways

**The discriminator for GPU wins in this codebase.** The kernels that beat CPU
(sparse randomised SVD, kNN, Harmony v2) are all GEMM/SVD-shaped: one big dense,
regular, high-arithmetic-intensity compute where the CPU baseline is `faer`
already running multithreaded on every core. GPU beats faer-on-N-cores at large
GEMM even on Metal. SCENIC lost for the opposite reason: it is not GEMM-bound. It
is thousands of small irregular tree-batches that the CPU fans out with rayon, so
the single GPU queue had to beat N cores at a workload with no dense-matmul heart.
When the CPU hot path is `faer::matmul`/SVD, GPU is worth trying. When it is
`par_iter` over independent small/irregular tasks, rayon already owns it.

**The VRAM hog is the sum buffers, not the features.** The wave halves from 8 to
4 because `WaveState` holds four separate f32 sum buffers at 16 bytes/slot
(`hist_y_sums`, `hist_y_sum_sqs`, `cum_y_sums`, `cum_y_sum_sqs`), and
`prefix_sum_bins` writes the cumulative buffers as distinct allocations from the
histograms. The parked-optimisations list points at u8-packing `feature_data`,
which is not where the pressure is. An in-place prefix sum (overwrite the
histogram with its cumulative, since the evaluate kernels only need `cum` plus
`cum[last]`) would drop the sums from 4x to 2x and let the wave go back to 8.
Estimate ~1.5-2x on the GPU floor. It is the single highest-leverage experiment
left, and it directly tests whether lifting the VRAM cap buys anything. It still
would not close a 6-7x gap: 1386s floor -> ~700-900s, still 3-4x behind CPU.

**Widening the target batch does not help at fixed VRAM.** Going from 64 to 128
targets per batch just trades the tree axis for the target axis; total per-launch
parallel work stays flat because it is bounded by the same VRAM. It halves batch
count but doubles per-batch cost.

**Hardware caveat.** This is an Apple Silicon verdict: one queue, unified memory
bandwidth shared with the CPU, and the histogram method's node x bin x target
scan cost that the CPU's depth-first code never pays. On high-VRAM CUDA the wave
stays at 8 with real headroom and the calculus flips. We cannot bench that here,
which is the main reason the code stays in the tree.

## Pointers

- Code: `src/gpu/sc_gpu/scenic_gpu.rs`, tests `tests/scenic_gpu.rs`, bench
  `benches/gpu_scenic_bench.rs`.
- CPU siblings: `run_scenic_grn` / `run_scenic_grn_streaming`
  (`sc_analysis/scenic.rs`), `run_scenic_grn_in_memory`
  (`mc_analysis/scenic_metacells.rs`).
- Original working docs: `docs/plans/gpu-scenic-port.md`,
  `docs/plans/elegant-bubbling-waffle.md`.
- Prior art referenced during the port: `ann_search_rs::gpu::forest_gpu`
  (random-projection tree ensembles, useful shape-level patterns).
