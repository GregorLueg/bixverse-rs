# GPU port: SCENIC ET/RF regression — completed

Reference doc for the SCENIC GPU implementation. Phases 1-5 delivered on `feat-faster-gpu`. GBM stays on CPU by design. This doc keeps the perf measurements and design notes worth referencing when touching the GPU tree code in future.

> **Caveat (2026-07-11):** the speedup table below measures a single 64-target batch. It does not model the real SCENIC workload, which processes thousands of target genes across tens of batches. On Apple Silicon the CPU driver fans batches across cores via rayon; the GPU driver processes them sequentially. Result: at 1000 TFs x 4000 targets x 10k cells, CPU end-to-end is **~230s (ET) / ~180s (RF)** while GPU end-to-end is **~1640s (ET) / ~1240s (RF)**. GPU loses by 3-7x on realistic shapes. See `docs/plans/elegant-bubbling-waffle.md` for the diagnosis and the Phase B mitigation that made GPU measurable at all (feature-upload amortisation + deferred importance readbacks). Kernel-level per-batch numbers below are still accurate.

## Entry points

- **Fit driver**: `fit_multi_trees_gpu` in `src/gpu/sc_gpu/scenic_gpu.rs` — multi-tree, multi-batch, ET or RF via `TreeRegressorConfig::random_threshold()`.
- **Top-level GPU functions** in the same file:
  - `run_scenic_grn_gpu<R>` — disk-backed, targets loaded up front.
  - `run_scenic_grn_streaming_gpu<R>` — disk-backed, I/O-chunk streaming.
  - `run_scenic_grn_in_memory_gpu<R, T>` — in-memory CSC (meta-cell path).
- **GPU tuning**: `ScenicGpuParams::wave_byte_budget` (default 4 GiB).
- **CPU siblings**: `run_scenic_grn` / `run_scenic_grn_streaming` (in `sc_analysis/scenic.rs`) and `run_scenic_grn_in_memory` (in `mc_analysis/scenic_metacells.rs`) share the same signatures modulo `device` and `gpu_params`.
- **Tests**: `tests/scenic_gpu.rs`.
- **Bench**: `benches/gpu_scenic_bench.rs`. Run with `cargo bench --features gpu,single-cell --bench gpu_scenic_bench`. Hand-rolled median-of-3 with a 300s skip threshold — not Criterion (Criterion's minimum 10 samples would push CPU RF at 50k+ into hours).

GBM callers hit `BixverseErrors::GpuNotSupportedForLearner` and must fall back to the CPU sibling.

## GPU speedup vs CPU (Apple Silicon, Metal via wgpu)

Shape: 1k TFs × 64 targets × 250 trees, `max_depth = 10`, `min_samples_leaf = 50`. Median of 3 runs after 1 warmup. `SKIP_ABOVE_SECS = 300s`.

| Cells | ET  | RF  |
|-------|----:|----:|
| 10k   | 1.53x | 1.34x |
| 25k   | 1.73x | 1.63x |
| 50k   | 1.77x | 1.84x |
| 75k   | 1.47x | *    |
| 100k  | *     | 2.11x |

\* CPU baseline exceeded the 300s per-iteration skip threshold at that shape, so speedup ratio is not defined; GPU wall-clock itself is available in the tables further down.

GPU beats CPU at every measured shape from 10k up.

## Statistical parity

- ET and RF per-target Pearson correlation vs CPU ≥ 0.95 (measured: 0.988–0.993).
- Multi-batch determinism: byte-identical vs standalone batches.
- RF bootstrap Pearson ≥ 0.95 (measured 0.976).
- Round-trip (top-level entry points vs CPU sibling): per-target Pearson ≥ 0.85 across `run_scenic_grn_gpu`, `run_scenic_grn_streaming_gpu`, `run_scenic_grn_in_memory_gpu`.

## Design notes

**Level-synchronous BFS.** CPU code is depth-first recursion. GPU rewrite processes all nodes at depth `d` in one kernel launch, then advances. No tree structure is stored — importance is the only output, matching the CPU code. Kernels per level per tree per gene batch:

1. Feature sub-sampling (on-device PRNG).
2. Sample-parallel histogram build (privatised, then merged).
3. Prefix-sum over 256 bins.
4. Split evaluation (ET random-threshold or RF exhaustive-threshold).
5. Sample reassignment (per-sample lookup).
6. Importance accumulation.

Trees run concurrently as an outer batch dimension (wave scheduler, budget-sized).

**Non-goals**: bit-exact CPU parity (statistical parity is fine), GRNBoost2 GPU port (GBM stays on CPU), vendor-specific tuning (backend-agnostic via cubecl/wgpu).

**GBM error path.** All three GPU top-level functions reject `RegressionLearner::GradientBoosting` up front and return `BixverseErrors::GpuNotSupportedForLearner`. The plan called GBM out as a non-goal in Phase 1 and Phase 5 kept that stance.

## Perf history

### Phase 4a (baseline, before tuning) — GPU losing at every shape

| Cells | ET CPU | ET GPU | ET speedup | RF CPU | RF GPU | RF speedup |
|-------|--------|--------|-----------:|--------|--------|-----------:|
| 10k   | 34.7s  | 59.4s  | 0.58x      | 24.9s  | 48.0s  | 0.52x      |
| 25k   | 84.0s  | 195.8s | 0.43x      | 64.8s  | 75.8s  | 0.85x      |
| 50k   | 157.1s | 256.0s | 0.61x      | 125.0s | 181.7s | 0.69x      |
| 75k   | 234.5s | skip   | –          | 198.1s | 291.2s | 0.68x      |
| 100k  | skip   | skip   | –          | 296.7s | skip   | –          |

Root cause: sample-parallel access pattern was actually bin-parallel with 256× read amplification on `feature_data`. Metal atomic contention was suspected but not proven.

### Phase 4b — sample-parallel histogram, on-device `next_active`

Two optimisations:
1. `build_hist_privatised` walks samples once each rather than the "one thread per bin, walk N samples per bin" Phase 1 shape. SMEM `Atomic<u32>` for bin counts + private-slice / CAS-loop paths for per-target Y sums.
2. On-device `next_active` + persistent child tensors eliminate per-level host readbacks. One `.read()` per wave rather than ~30.

| Cells | ET CPU | ET GPU | ET speedup | RF CPU | RF GPU | RF speedup |
|-------|--------|--------|-----------:|--------|--------|-----------:|
| 10k   | 35.5s  | 46.2s  | 0.77x      | 25.8s  | 27.9s  | 0.92x      |
| 25k   | 86.5s  | 49.8s  | **1.74x**  | 66.3s  | 82.4s  | 0.80x      |
| 50k   | 167.0s | 94.2s  | **1.77x**  | 131.8s | 71.5s  | **1.84x**  |
| 75k   | 240.9s | 163.6s | **1.47x**  | skip   | 102.8s | –          |
| 100k  | skip   | 174.3s | –          | 293.5s | 139.1s | **2.11x**  |

vs 4a: ET-GPU 25k improved 75%, RF-GPU 100k went from unmeasurable to 2.11x. GPU wins from 25k up.

### Phase 4c — min/max precompute + WORKGROUP_32 for ET evaluate

1. `min_bin` / `max_bin` per slot folded into `merge_hist`'s existing 256-bin counts scan. `evaluate_splits_et` and `evaluate_splits_rf` read the two u32s instead of re-scanning 512 bins per candidate.
2. `evaluate_splits_et` shrunk to `WORKGROUP_32` — thread utilisation from ~24% to ~97% at `k_feats ≈ 31, n_thresholds = 1`. `evaluate_splits_rf` stays at `WORKGROUP_128`.

Re-benched at 10k + 25k only:

| Cells | ET CPU | ET GPU | ET speedup | RF CPU | RF GPU | RF speedup |
|-------|--------|--------|-----------:|--------|--------|-----------:|
| 10k   | 34.3s  | **22.5s**  | **1.53x**  | 26.0s  | **19.4s**  | **1.34x**  |
| 25k   | 86.0s  | 49.8s  | 1.73x      | 66.4s  | **40.7s**  | **1.63x**  |

Fixed the Phase 4b RF-25k regression (recovered fully via item 1). GPU now beats CPU at every measured shape.

## Parked optimisations

Not needed at current scale. Revisit only if a workload actually demands more:

- u8-packed `feature_data` (4 samples per u32) — 4× bandwidth cut, compounds on top of the Phase 4b sample-parallel hist.
- Comptime `use_multiplicity` skip when `subsample_rate = 1.0`.
- Hoist Fisher-Yates + multiplicity buffers to wave scope.
- Parallel prefix on the counts scan in `merge_hist` / `prefix_sum_bins`.

## Prior art referenced during the port

`ann_search_rs::gpu::forest_gpu` builds random-projection tree ensembles on GPU. Algorithmically different (hyperplane splits, no target values, no importance accumulation), but useful shape-level ideas:

- `leaf_pairwise_proposals` — per-tree parallelism pattern.
- `TreeResults<T>` — analogous to our sample→node mapping.
- `SMEM_BUDGET: 32_768` — same Apple Silicon constraint.
- Rayon-per-tree host driver pattern — directly transferable.

## Reference points in the CPU code

- `fit_multi_trees_sparse` — `scenic.rs:1485`, top-level per-tree loop.
- `build_node_multi_sparse` — `scenic.rs:919`, recursive node builder; rewritten as level-synchronous BFS on GPU.
- `TreeBuffers::build_histograms_sparse` — `scenic.rs:651`, CPU histogram accumulation shape.
- `evaluate_split_multi` — `scenic.rs:833`, split-score kernel with SIMD variant.
- `SparseYBatch::from_targets` — `scenic.rs:364`, input format.
- `QuantisedStore` — `sc_utils/utils_tree.rs:28`, u8 column-major feature storage.
- `build_csr_gpu_privatised` — `gpu/ml/k_means_gpu.rs:718`, privatised-then-merged accumulation pattern reused for histograms.
- `count_changed` — `gpu/ml/k_means_gpu.rs:875`, one-sample-per-thread + single atomic pattern.

## Follow-up (not this port)

- **R wrappers** for both the CPU and GPU SCENIC entry points, added together in `sc_r_wrappers.rs` and `gpu/gpu_r_wrappers.rs`. Currently `sc_r_wrappers.rs` only carries parameter converters — the `run_scenic_grn*` functions are not exposed to R yet on either side.
