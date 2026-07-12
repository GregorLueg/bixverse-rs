# GPU SCENIC: close the batch-loop gap

## Context

`docs/plans/gpu-scenic-port.md` records the SCENIC GPU port as done and claims a 1.3x to 2.1x GPU win across shapes. Real-world runs contradict that. When the workload actually has thousands of target genes (not 64), the CPU version wins comfortably, sometimes by a large margin.

Root cause is not the GPU kernels. It is that the bench measures the wrong thing and the GPU driver leaves single-thread work on the table:

- **Bench blind spot.** `benches/gpu_scenic_bench.rs` sets `N_TARGETS: usize = 64` as the *total* target count (`benches/gpu_scenic_bench.rs:42`, module doc line 4 says "64 targets (one batch)"). It calls `fit_multi_trees_sparse` / `fit_multi_trees_gpu` directly, bypassing the top-level drivers that own the batch loop. So the per-batch kernel throughput is what gets timed; the multi-batch behaviour that dominates real runs is never measured.
- **CPU fans batches out with rayon.** All three CPU drivers (`sc_analysis/scenic.rs:3144-3173`, `sc_analysis/scenic.rs:3477-3504`, `mc_analysis/scenic_metacells.rs:479-508`) wrap the batch loop in `(0..total_batches).into_par_iter().map(|batch_idx| fit_multi_trees_sparse(...))`. Shared read-only `QuantisedStore` for TFs. With 5000 target genes and batch size 64 that is 79 batches; on a 12-core M-series roughly 7 batches of wall-clock time.
- **GPU processes batches strictly sequentially.** `run_scenic_grn_gpu` (`src/gpu/sc_gpu/scenic_gpu.rs:3377`) loops `for (batch_idx, cols) in col_batches.iter().enumerate()` and calls `fit_multi_trees_gpu` per batch. Comment at line 3239-3241 justifies this: *"the wave scheduler inside `fit_multi_trees_gpu` already saturates the device on one batch, so concurrent host-side batch launches would just contend for VRAM."* That claim is plausible for kernel occupancy but ignores what happens between batches.
- **The `fit_multi_trees_gpu` re-does per-call setup.** Every call re-runs `feature_bins_u32` build and uploads `feature_data_gpu` of size `n_features * n_samples * 4` bytes (`scenic_gpu.rs:2930-2932`). At 1000 TFs x 100k cells that is 400 MB re-uploaded for every 64-target batch. Then per internal batch it allocates a fresh `WaveState` (line 2946), zeroes a fresh `batch_importances_gpu` (line 2959), and ends with a blocking `.read()` (line 3047) before the next batch can even start allocating.

So on a realistic workload, CPU gets a `num_cores` fan-out multiplier that GPU never sees, and GPU stacks per-batch setup + readback stalls on top of already-sequential kernel dispatch.

The goal here: fix the bench so the gap is measurable, then rework the GPU path so it stops giving away throughput to setup and host stalls. Compare honestly. If the ceiling on Apple Silicon still favours CPU at real scale, say so.

## Strategy

Three phases. Ship each as its own commit / PR.

### Phase A: Realistic bench

Rewrite `benches/gpu_scenic_bench.rs` to measure the actual workload shape.

- Drop the "64 total targets" trap. Introduce e.g. `N_TARGETS_TOTAL = 4000` (roughly typical SCENIC), batch size 64. That is ~63 batches, enough that the CPU rayon fan-out and GPU sequential stack both show up.
- Time the top-level drivers, not the low-level fitters: `run_scenic_grn_in_memory` vs `run_scenic_grn_in_memory_gpu` from `mc_analysis/scenic_metacells.rs`. This is what real callers hit. If disk I/O is a nuisance in bench, use the in-memory path.
- Keep the CPU-only tiny-batch timing as a second bench matrix (`bench_kernel` vs `bench_end_to_end`) so the per-batch kernel wins we already measured stay visible and Phase B does not silently regress them.
- Skip threshold stays at 300s per iteration. Some shapes will fall off.
- Rerun the perf table on the current tree before touching GPU code; that number is the baseline for Phase B.

Deliverable: two bench targets, honest numbers table in this doc, no code changes to the library.

### Phase B: Cross-batch amortisation

Rework `fit_multi_trees_gpu` and the top-level drivers so per-batch setup and readback stop dominating. Concrete moves, in order of expected payoff:

1. **Single feature upload + deferred readbacks per driver call** (DONE together as one refactor). New private entry `fit_scenic_batches_gpu` takes a `&[&[SparseAxis]]` of pre-sliced batches and a matching `&[usize]` of per-batch seeds. Uploads `feature_data_gpu` once for the whole call. Runs each batch's wave loop as before, but does not `.read()` per batch: it stashes the per-batch `batch_importances_gpu` handle (small, ~256 KB each). Once every batch has been submitted, walks the stashed handles and reads them one by one. Each `.read()` flushes the queue at that point, but by then later batches have already been submitted, so the pipeline stays full. All three top-level drivers now call this once with their cluster-aware batches. `fit_multi_trees_gpu` stays as a thin backward-compat wrapper (chunks into 64s, same seed for every internal batch, matching the pre-refactor semantics).
2. **Persistent `WaveState`** (optional followup, not shipped in the first cut). Allocate once at max shape, reuse across batches. Requires kernels to accept a runtime `n_batch_targets` param so the smaller-batch case does not read stale slots. Skip unless (1) leaves per-batch VRAM churn dominating the profile.
3. **Overlap host prep with device work** (optional). Two-slot ping-pong for `SparseYBatch::from_targets` + `mult_host` so batch `N+1`'s host work overlaps with batch `N`'s GPU work. Modest gain; only worth it if profiling still shows host stalls after (1).
4. **Widen `MULTI_OUTPUT_BATCH` when VRAM allows** (optional). 64 -> 128 with `wave_size` halved keeps `wave_byte_cost` flat but halves batch count. Guard against `wave_byte_budget`. Only if (1) does not close the gap.

Non-negotiables:

- Statistical parity from `docs/plans/gpu-scenic-port.md` must survive: per-target Pearson ≥ 0.95 vs CPU on the existing `tests/scenic_gpu.rs` cases, byte-identical determinism across independent batches.
- The per-batch kernel micro-benchmark from Phase A must not regress.
- Do not remove the `wave_byte_budget` gate. Reshape it if needed.

The comment at `scenic_gpu.rs:3239-3241` should be updated or dropped once the driver moves to a single feature upload. It is no longer accurate as guidance.

### Phase C: Verify and be honest about the ceiling

- Rerun the Phase A bench on the reworked code. Update the perf table.
- Rerun `tests/scenic_gpu.rs` for parity.
- Rerun on a real dataset (any one you have to hand with thousands of target genes) to confirm the microbench numbers translate. Note the shape, cell count, TF count, target count, backend, wall clock.
- If GPU still trails CPU at typical scale on Apple Silicon, write that down clearly in this doc and update `docs/plans/gpu-scenic-port.md` so nobody else reads the current claim without the caveat. On this hardware CPU has 8-12 cores of rayon fan-out that GPU has no direct answer to; that arithmetic may hold. NVIDIA silicon should look very different, we cannot bench that here.

## Critical files

Ordered by touch weight.

- `src/gpu/sc_gpu/scenic_gpu.rs` - `fit_multi_trees_gpu` (line 2892), `run_scenic_grn_gpu` (3268), `run_scenic_grn_streaming_gpu` (~3560, referenced in exploration), `run_scenic_grn_in_memory_gpu` (~3764), `WaveState::allocate` (2414), `write_batch_into_importances` (3221). This is where Phase B lives.
- `benches/gpu_scenic_bench.rs` - full rewrite for Phase A. Keep the median-of-3 harness, replace shapes and entry points.
- `docs/plans/gpu-scenic-port.md` - append a "Correction" note pointing here once Phase C data is in.
- `tests/scenic_gpu.rs` - existing parity tests. Rerun after Phase B; add a driver-level parity test that exercises the new "single upload" path if Option (a) is chosen for step B.1.

CPU reference points (do not edit, useful to trace):

- `run_scenic_grn` (`src/single_cell/sc_analysis/scenic.rs:3144`), `run_scenic_grn_streaming` (same file, 3477), `run_scenic_grn_in_memory` (`src/single_cell/mc_analysis/scenic_metacells.rs:479`) - the rayon batch loops.
- `fit_multi_trees_sparse` (`src/single_cell/sc_analysis/scenic.rs:1485`) - what runs inside each rayon task.

## Verification

- `cargo test --features gpu,single-cell --test scenic_gpu` after Phase B. All parity tests pass.
- `cargo bench --features gpu,single-cell --bench gpu_scenic_bench` before Phase B (baseline) and after (result). Save the two tables into this doc.
- Manual end-to-end run against a real dataset with several thousand target genes. Capture wall clock for CPU vs GPU top-level drivers. Record the shape and numbers here.
- If a driver-level parity test does not already exist, add one that runs `run_scenic_grn_in_memory` vs `run_scenic_grn_in_memory_gpu` on a synthetic multi-batch shape and asserts per-target Pearson ≥ 0.95.

## Baseline: pre-Phase-B numbers

Median-of-3 wall clock, 10k cells, 1000 TFs, 250 trees, `max_depth = 10`, `min_samples_leaf = 50`. Apple Silicon, wgpu Metal backend. `SKIP_ABOVE_SECS = 300s`.

**Kernel matrix** (single 64-target batch via `fit_multi_trees_sparse` / `fit_multi_trees_gpu`) - this is what `gpu-scenic-port.md` measured. Reproduces the Phase 4c numbers within noise.

| Learner | CPU   | GPU   | GPU speedup |
|---------|------:|------:|------------:|
| ET      | 33.86s | 22.37s | **1.51x**   |
| RF      | 25.00s | 19.25s | **1.30x**   |

**End-to-end matrix** (~63 batches, 4000 targets via `run_scenic_grn_in_memory` / `_gpu`) - this is what real callers see.

| Learner | CPU     | GPU        | GPU speedup |
|---------|--------:|-----------:|------------:|
| ET      | 231.48s | SKIP (>300s warmup) | **worse than 0.77x** |
| RF      | 192.11s | SKIP (>300s warmup) | **worse than 0.64x** |

CPU dominates. GPU per-batch wins get eaten by strictly-sequential batch dispatch and per-batch setup/readback stalls. Every one of the 63 batches re-uploads the ~40 MB `feature_data_gpu`, allocates a fresh `WaveState`, allocates a fresh `batch_importances_gpu`, and blocks on `.read()` before host can prep the next batch. Phase B targets exactly these.

## Post-Phase-B numbers

Same shape. Kernel matrix stays median-of-3; end-to-end matrix uses a single-shot measurement (no warmup) because we needed to see GPU wall clock past the 300s skip.

**Kernel matrix** - no regression. Per-batch kernel throughput held within noise.

| Learner | CPU   | GPU   | GPU speedup |
|---------|------:|------:|------------:|
| ET      | 32.97s | 22.13s | **1.49x**   |
| RF      | 24.46s | 19.04s | **1.28x**   |

**End-to-end matrix** - GPU now runs to completion, but is still miles behind CPU.

| Learner | CPU     | GPU     | GPU speedup |
|---------|--------:|--------:|------------:|
| ET      | 229.28s | 1640.50s | **0.14x** (GPU is 7.2x slower) |
| RF      | 427.36s | 1239.24s | **0.34x** (GPU is 2.9x slower) |

(RF CPU number is a single-shot and looks high vs the pre-Phase-B median. First-run cache / thermal effects, or background load during the sample. The GPU number is the honest one.)

**What Phase B actually bought.** The change removed the per-call `feature_data_gpu` re-upload (one 40 MB upload instead of 63) and eliminated the blocking `.read()` between every batch. That is why the GPU end-to-end run now finishes at all - before, warmup blew past 300s, meaning one full pass was somewhere between 300s and infinity. We now know it is 1240s (RF) to 1640s (ET). Progress: from "unmeasurable" to "measurable and honestly bad."

**Why it did not close the gap.** GPU end-to-end is still ~26s per batch (1640s / 63) vs the kernel-bench 22s per batch, so setup overhead per batch is ~4s. Even if we drove that to zero the GPU floor is 63 x 22s = 1386s. CPU wins because rayon fans those 63 batches across all cores in parallel; effective wall clock is roughly 63 / N_cores x per-core-per-batch = ~229s. GPU has no equivalent knob on wgpu / Metal: submitting concurrent batches from multiple threads contends for the same VRAM and the same single queue.

**Conclusion.** On this hardware, for real SCENIC workloads, **CPU is the right choice**. GPU only wins on the artificial single-batch microbenchmark. Ship Phase B as a correctness / measurability fix, not as a "GPU is competitive now" claim. The `gpu-scenic-port.md` speedup table needs a caveat pointing here.

The remaining optional phases (persistent `WaveState`, host-side pipeline, widen `MULTI_OUTPUT_BATCH`) can each shave a few percent, but none of them can multiply GPU throughput by ~7x. Skip them on this hardware. Revisit only on GPUs with much higher per-batch speedup than Apple Silicon offers.
