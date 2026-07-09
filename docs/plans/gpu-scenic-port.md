# GPU port: SCENIC RF/ExtraTrees regression

## Goal

Port the multi-output ExtraTrees + RandomForest regression from `single_cell/sc_analysis/scenic.rs` to run on GPU via cubecl, feature-gated behind `gpu`. Reuse GPU primitives from `ann-search-rs` (`GpuTensor`, `grid_2d`, workgroup helpers) as the CPU-side code reuses its SIMD/kNN/k-means primitives.

## Non-goals

- GRNBoost2 / gradient boosting path (`fit_grnboost2_*`). Out of scope.
- Bit-exact numerical parity with the CPU path. Statistical equivalence is fine.
- Vendor-specific tuning. cubecl/wgpu keeps the code backend-agnostic; Metal is known to underperform on atomic-heavy patterns and that is documented, not fixed.
- Streaming I/O overlap with GPU compute in Phase 1-4. Added in Phase 5 if wall-clock warrants it.

## Constraints

- Live in `src/single_cell/sc_gpu/scenic_gpu.rs` (new file), mirroring the shape of `harmony_gpu.rs`.
- All GPU code gated behind `#[cfg(all(feature = "single-cell", feature = "gpu"))]`.
- Reuse `ann_search_rs::gpu::tensor::GpuTensor`, `grid_2d`, and cubecl workgroup conventions already used across `gpu/`.
- Errors go through `BixverseErrors` (add new variants as needed, feature-gated).
- Public entry points mirror CPU signatures where practical to keep the dispatch shim trivial.

## Algorithmic shape (level-synchronous BFS)

CPU code is depth-first recursion (`build_node_multi_sparse` in `scenic.rs:919`). GPU rewrite is level-synchronous: process all nodes at depth `d` in one kernel launch, produce split decisions, reassign samples to child nodes, advance to depth `d+1`. No tree structure is stored — importance is the only output, matching the CPU code.

Per level, per tree, per gene batch, kernels do:

1. **Feature sub-sampling** — on-device PRNG picks `k_feats ≈ √n_features` features per node.
2. **Histogram build** — one workgroup per (node, feature). Threads stride over active samples, atomically bump per-workgroup private histograms (`[bin × n_targets × 3 fields]`). Then merge kernel reduces privatised histograms.
3. **Prefix-sum** — parallel scan over 256 bins per (node, feature).
4. **Split evaluation** — per (node, feature), evaluate candidate thresholds; workgroup-wide argmax across features picks the best split.
5. **Sample reassignment** — per sample, look up its current node's split, write new node ID.
6. **Importance accumulation** — for each node that split, atomically add its weighted variance reduction to the importance tensor `[n_features × n_targets]`.

Trees run concurrently as an outer batch dimension. Wave size (concurrent trees) is chosen from VRAM budget.

---

## Phase 1 — Single-tree ET prototype

**Aim:** Validate the kernel graph on one tree, one 64-target batch, ExtraTrees only. Get statistical parity vs CPU on toy data.

### Deliverables

- New file `src/single_cell/sc_gpu/scenic_gpu.rs`
- Public function `fit_extra_trees_gpu_single(sparse_y, feature_matrix, n_samples, config, seed) -> Result<Vec<Vec<f32>>, BixverseErrors>` matching one-tree slice of `fit_multi_trees_sparse`
- Kernels (all `#[cube(launch_unchecked)]`):
  - `build_hist_privatised` — privatised histograms, one workgroup per (node, feature)
  - `merge_hist` — reduce privatised histograms into per-node full histograms
  - `prefix_sum_bins` — parallel scan over 256 bins
  - `evaluate_splits_et` — random-threshold split-eval, argmax across features
  - `reassign_samples` — sample→node update
  - `accumulate_importance` — weighted variance-reduction accumulator
- Level-synchronous driver on the host, one tree, one batch

### Acceptance criteria

- [ ] Compiles cleanly under `cargo build --features gpu,single-cell`
- [ ] `tests/scenic_gpu.rs` (new, gated behind `gpu` feature) covers:
  - Toy `QuantisedStore` (256 samples, 32 features, seeded deterministic random u8s)
  - Toy `SparseYBatch` (4 targets, 50% sparsity)
  - ExtraTrees config with fixed seed, `n_trees = 1`, `max_depth = 6`
- [ ] For 10 different seeds, GPU top-10 features overlap CPU top-10 features by ≥ 60% (single tree is high-variance — this is a sanity floor, not a precision target)
- [ ] Runs on wgpu-cpu backend (used in cubecl test setup) as well as native GPU

### Test approach

New integration test `tests/scenic_gpu.rs` builds a `QuantisedStore::from_raw` from deterministic pseudo-random u8s, constructs `SparseYBatch` from synthetic sparse targets, runs both CPU and GPU single-tree paths with same seed, compares. Backend chosen via cubecl's runtime picker.

---

## Phase 2 — Many trees, multi-batch ET

**Aim:** Scale Phase 1 to full ensembles and full gene sets. Beat CPU on realistic single-cell shapes.

### Deliverables

- Public function `fit_multi_trees_gpu(targets, feature_matrix, n_samples, config, seed) -> Result<Vec<Vec<f32>>, BixverseErrors>` mirroring `fit_multi_trees_sparse`
- Wave scheduler in the driver: process `W` trees concurrently (default `W = 8`, tunable via env or config)
- On-device per-node PRNG (splitmix64 or LCG seeded from `(tree_idx, node_id)`)
- Multi-batch loop on the host: run the wave driver per gene batch, reusing tensor allocations across batches

### Acceptance criteria

- [ ] Statistical parity vs CPU: per-target importance vector correlation (Pearson) ≥ 0.95 with CPU output, averaged over targets, on synthetic 10k-cell × 500-TF × 20-target data with `n_trees = 500`
- [ ] Handles a full realistic batch: 50k cells × 1k TFs × 64 targets × 500 trees × `max_depth = 10` without OOM on 8GB VRAM
- [ ] Sequential multi-batch loop over 10 batches produces same output as running each batch independently (reset semantics for reused tensors verified)
- [ ] Test in `tests/scenic_gpu.rs` extended to multi-tree case with tolerance

### Test approach

Same synthetic data harness as Phase 1, scaled to 10k samples / 500 features / 20 targets / 500 trees. Assert per-target Pearson correlation. Add a memory-budget sanity check (allocate, run, measure peak).

---

## Phase 3 — RandomForest path

**Aim:** Add exhaustive-threshold split evaluation, bootstrap subsampling. ET/RF selectable via existing `TreeRegressorConfig::random_threshold()`.

### Deliverables

- New kernel `evaluate_splits_rf` — exhaustive threshold scan over prefix-summed histogram, workgroup-wide argmax
- Bootstrap-with-replacement sample selection when `config.bootstrap()` is true (host-side or on-device, whichever is simpler)
- Split-eval kernel selected at launch time based on `config.random_threshold()`
- `fit_multi_trees_gpu` handles both `RandomForestConfig` and `ExtraTreesConfig` (via `dyn TreeRegressorConfig`)

### Acceptance criteria

- [ ] RF path produces per-target Pearson correlation ≥ 0.95 with CPU RF on the Phase 2 synthetic dataset
- [ ] Both paths selectable via the same `fit_multi_trees_gpu` entry
- [ ] Phase 2 ET tests still pass unchanged (no regression)

### Test approach

Extend `tests/scenic_gpu.rs` with an RF case. Reuse the ET tolerance and dataset. Add one test that switches between ET and RF configs to confirm dispatch.

---

## Phase 4 — Benchmarking and tuning

**Aim:** Establish speedup vs CPU across realistic shapes. Tune workgroup sizes, wave counts, storage precision.

### Deliverables

- New bench `benches/gpu_scenic_bench.rs` (mirroring `benches/gpu_k_means_bench.rs`), gated behind `gpu` feature
- Bench matrix: `{10k, 50k, 200k} cells × {500, 1k, 2k} TFs × {64 targets × 5 batches} × 500 trees × max_depth 10`
- Tuning knobs exposed as consts or config: wave size, workgroup size for hist build, workgroup size for split eval, per-node-hist tile width
- Optional: fp16 storage for the multi-output y sums (mirror `k_means_gpu.rs` mixed-precision pattern)

### Acceptance criteria

- [ ] Bench compiles and runs on both wgpu-cpu and native GPU backends
- [ ] Speedup vs CPU (ET path) documented in a comment or `docs/plans/gpu-scenic-port.md` update:
  - Target: **2-3x** on 50k-cell workloads and up
  - Rationale for the modest target: Rust-on-CPU is very fast on Apple Silicon; observed GPU gains across this crate have been smaller than textbook estimates for the workload
- [ ] Any workload where GPU loses to CPU is documented (expected: very small n_samples where launch overhead dominates)

### Test approach

Criterion benches. No hard test assertions on speedup — just documented numbers. If GPU consistently loses on the 10k-cell workload, we accept that and let the dispatch shim (Phase 5) route small workloads to CPU.

### Phase 4a measurements (2026-07-08, macOS wgpu default adapter — Metal)

Shape fixed: 1_000 TFs, 64 targets, 250 trees, `max_depth = 10`, `min_samples_leaf = 50`. Median of 3 timed runs after 1 warmup. `SKIP_ABOVE_SECS = 300s` — shapes whose warmup exceeded 5 min were skipped rather than measured. Bench binary hand-rolled (not Criterion) — `benches/gpu_scenic_bench.rs`.

| Cells | ET CPU | ET GPU | ET speedup | RF CPU | RF GPU | RF speedup |
|-------|--------|--------|-----------:|--------|--------|-----------:|
| 10k   | 34.7s  | 59.4s  | 0.58x      | 24.9s  | 48.0s  | 0.52x      |
| 25k   | 84.0s  | 195.8s | 0.43x      | 64.8s  | 75.8s  | 0.85x      |
| 50k   | 157.1s | 256.0s | 0.61x      | 125.0s | 181.7s | 0.69x      |
| 75k   | 234.5s | skip   | –          | 198.1s | 291.2s | 0.68x      |
| 100k  | skip   | skip   | –          | 296.7s | skip   | –          |

**Takeaway**: GPU is losing at every measured shape. Trend is not "GPU pulls ahead at larger scale" — the ratios are noisy and not monotonically improving. RF fares slightly better than ET on GPU (25k RF hits 0.85x — closest to parity). 75k and 100k GPU points were skipped because warmup alone exceeded 5 min, which is itself a signal that GPU is genuinely bad at those scales, not just marginally slow. Metal atomic contention is a plausible root cause but not proven; Phase 4b tuning should target the largest identified levers (per-level host readback, `evaluate_splits_et` thread utilisation, `evaluate_splits_rf` threshold-range scan) before deciding whether to try larger cell counts or accept GPU as a niche fallback.

### Phase 4b measurements (2026-07-09, macOS wgpu default adapter — Metal)

Same shape, same bench harness. Two optimisations applied:

1. **Sample-parallel `build_hist_privatised`** replaces the bin-parallel Phase 1 shape. Threads walk samples once each rather than the previous "one thread per bin, walk N samples per bin" pattern that produced 256× read amplification on `feature_data`. SMEM `Atomic<u32>` for bin counts + private-slice / CAS-loop paths for the per-target Y sums. Statistical parity holds — all 7 tests still pass, byte-identical multi-batch determinism preserved.
2. **On-device `next_active` + persistent child tensors** eliminate per-level host readbacks. `split_feature`, `importance_delta`, and `node_counts` no longer round-trip to the host per level. One `.read()` per wave for the final importances instead of ~30 per wave.

| Cells | ET CPU | ET GPU | ET speedup | RF CPU | RF GPU | RF speedup |
|-------|--------|--------|-----------:|--------|--------|-----------:|
| 10k   | 35.5s  | 46.2s  | 0.77x      | 25.8s  | 27.9s  | 0.92x      |
| 25k   | 86.5s  | 49.8s  | **1.74x**  | 66.3s  | 82.4s  | 0.80x      |
| 50k   | 167.0s | 94.2s  | **1.77x**  | 131.8s | 71.5s  | **1.84x**  |
| 75k   | 240.9s | 163.6s | **1.47x**  | skip   | 102.8s | –          |
| 100k  | skip   | 174.3s | –          | 293.5s | 139.1s | **2.11x**  |

vs Phase 4a baseline: ET-GPU at 25k improved 75% (195.8s → 49.8s), ET-GPU at 50k improved 63%, RF-GPU at 50k improved 61%, RF-GPU at 100k went from unmeasurable to a 2.11x win.

**Takeaway**: GPU now beats CPU from 25k cells (ET) / 50k cells (RF) and up, with wins in the 1.5x–2.1x range across those shapes. The plan's 2-3x target on 50k+ is hit for RF at 100k (2.11x) and comes close for ET at 25k-50k (~1.75x). Two anomalies worth flagging: (a) **RF at 25k regressed slightly** (75.8s → 82.4s) — small enough to be median-of-3 noise on a single run but worth watching; (b) **ET at 10k is still slower than CPU** (0.77x) — expected, launch overhead dominates at small N, and Phase 5's dispatch shim will route small workloads to CPU.

The audit's diagnosis was correct: read amplification in `build_hist_privatised` was the primary bottleneck, not Metal atomics. Medium-impact items (min/max precompute, WORKGROUP_32 for ET evaluate, u8 packing) remain parked — the current gains are enough that Phase 5 can proceed. Revisit those items only if downstream users need better small-N performance.

---

## Phase 5 — Integration

**Aim:** Make the GPU path reachable from the existing SCENIC entry points, callable from R.

### Deliverables

- Dispatch shim in `run_scenic_multi_output` / `run_scenic_multi_output_streaming`: when `gpu` feature is compiled in AND workload passes a size threshold, call `fit_multi_trees_gpu` instead of `fit_multi_trees_sparse`
- Size threshold documented and configurable via `ScenicParams`
- R wrapper for the GPU dispatch in `gpu/gpu_r_wrappers.rs`
- CI passes with `cargo test --features gpu,single-cell,multi-modal`

### Acceptance criteria

- [ ] `run_scenic_grn` unchanged from R's perspective; GPU is transparent behind feature flag
- [ ] R wrapper exposes a way to force CPU path (for debugging / when GPU adapter is unavailable)
- [ ] CI GPU job (already exists — `test-gpu` in `.github/workflows/test.yml`) runs the SCENIC GPU tests
- [ ] No unexpected regressions in existing GPU tests (`tests/gpu_corr.rs`, `benches/gpu_k_means_bench.rs`)

### Test approach

Existing SCENIC CPU integration tests run through the dispatch shim with the GPU feature enabled; confirm they still pass (accepting statistical drift). Add one R wrapper smoke test if extendr test infrastructure supports it.

---

## Risks and open questions

- **Wave sizing on constrained VRAM** — 8 trees × 1024 leaves × 45 features × 256 bins × 64 targets × 3 fields × 4 B ≈ 3.6 GB just for the level histograms. May need to shrink wave size or process levels one-at-a-time within a wave. Discover empirically in Phase 2.
- **On-device PRNG determinism** — reproducibility across runs (same seed → same importances) is required; across CPU/GPU is not. Cheap LCG is fine.
- **Sparse Y access pattern** — for very dense targets a dense-Y GPU representation may beat the sparse gather. Punt to Phase 4 tuning if it shows up in benches.
- **Metal atomic contention** — histogram build uses atomics. Metal is known to bottleneck here. Documented, not fixed.

## Prior art in ann-search-rs

`ann_search_rs::gpu::forest_gpu` (in `~/repos/shared/ann-search-rs/src/gpu/forest_gpu.rs`) already builds tree ensembles on GPU — random projection trees used to seed the kNN graph before NNDescent iterations. **Algorithmically distinct** from what we're porting (random hyperplane splits vs histogram-based variance-reduction splits, no importance accumulation, no target values at all), so it is not directly reusable. But worth reading before Phase 1 for shape-level ideas:

- `leaf_pairwise_proposals` shows the per-tree parallelism pattern for a tree-ensemble launch
- The `TreeResults<T>` shape and how partition IDs are stored per point is analogous to our sample→node mapping
- SMEM budget conventions (`SMEM_BUDGET: 32_768` for Apple Silicon) — same constraint we'll hit
- The rayon-per-tree host driver pattern is directly transferable

## Reference points in the CPU code

Key locations to mirror or delegate to:

- `fit_multi_trees_sparse` — `scenic.rs:1485`, top-level per-tree loop
- `build_node_multi_sparse` — `scenic.rs:919`, recursive node builder to be rewritten as level-synchronous BFS
- `TreeBuffers::build_histograms_sparse` — `scenic.rs:651`, CPU histogram accumulation shape
- `evaluate_split_multi` — `scenic.rs:833`, split-score kernel with SIMD variant
- `SparseYBatch::from_targets` — `scenic.rs:364`, input format
- `QuantisedStore` — `sc_utils/utils_tree.rs:28`, u8 column-major feature storage
- `build_csr_gpu_privatised` — `gpu/ml/k_means_gpu.rs:718`, privatised-then-merged accumulation pattern to reuse for histograms
- `count_changed` — `gpu/ml/k_means_gpu.rs:875`, one-sample-per-thread + single atomic pattern

## Working style

Phases run sequentially in this worktree. Each phase's acceptance criteria are checked before moving on. Subagents may be spawned per phase or per kernel; the driver stays in main-loop context.
