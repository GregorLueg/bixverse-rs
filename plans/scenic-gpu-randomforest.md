# SCENIC GPU: RandomForest

## Context

ExtraTrees on GPU now beats the CPU 3.67x end to end (see `docs/scenic_gpu.md`
and `plans/okay-let-s-plan-the-breezy-fiddle.md`). RandomForest is untouched and
loses 6.0x. This plan covers whether and how to close that.

**Read the honest assessment first, before the steps.** RF is a much harder
target than ET was. ET fell to one insight; RF needs three or four stacked
changes to reach break-even, and the payoff at the end is modest.

## The bar

| | value |
|---|---|
| RF e2e CPU | 187.39s |
| RF e2e GPU | 1131.63s |
| RF single batch, CPU 1 core | 24.55s |
| RF single batch, GPU | 19.24s (**1.28x**) |
| Same at 20% density | **1.17x** |
| Effective rayon fan-out | 63 x 24.55 / 187.39 = **8.25x** |

So the GPU must beat one core by 8.25x to draw, from 1.28x today. That is a
**6.4x improvement needed just to break even**, and more to be worth shipping.

For contrast, ET needed 9.1x and got 32.8x, but it got there because 99% of what
it computed was thrown away. RF genuinely consumes its histogram.

Note RF is the *cheaper* learner on CPU because `RandomForestConfig` defaults to
`subsample_rate = 0.632` while ET uses every cell. That also means RF's GPU
kernels do ~63% of ET's sample work, so the gap is not about workload size.

## Where the time goes

Per-kernel profile at the reference shape (`CUBECL_DEBUG_OPTION=profile-medium`):

| kernel | share |
|---|---:|
| `build_hist_privatised` | 57.4% |
| `prefix_sum_bins` | 35.8% |
| `evaluate_splits_rf` | 6.6% |
| everything else | ~0.2% |

The 255-threshold sweep is nearly free. The cost is building the histogram and
moving it through DRAM.

`build_hist_privatised`'s cost is specifically the CAS-retry-loop float add per
(sample, target) in global memory, ~1.4e9 per fit, which matches its measured
time almost exactly. WGSL has no float `atomicAdd`, hence the loop.

## Steps

Ordered by confidence per unit of effort, not by size. Each is independently
measurable on `benches/gpu_scenic_micro.rs` (`rf_32t` and `rf_multibatch`), gated
on `tests/scenic_gpu.rs` holding `phase3_random_forest_pearson >= 0.95` and
`phase3_rf_bootstrap_pearson >= 0.95`, currently 0.988 and 0.976.

### Step 1: fuse the scan into evaluate

Highest confidence, no accuracy risk, targets 42% of runtime.

`prefix_sum_bins` writes `cum_counts` / `cum_y_sums` / `cum_y_sum_sqs` to DRAM
and `evaluate_splits_rf` reads them straight back. Per level that is 3.25 GB
written, 3.25 GB read by prefix, 3.25 GB read by evaluate. Fused it becomes one
3.25 GB read and no write: **3x on that portion**.

Shape, and this part is not optional: **keep targets on the thread axis.**
Thread `k` owns target `k`, walks bins 0..255 in order, and carries the
cumulative in a register. Consecutive threads then read consecutive addresses out
of the `[bin][target]` layout, which is the only coalesced access pattern
available. Putting bins on the lane axis was tried during the ET work and came
out 3-5x slower.

Per bin, the score needs a sum across targets, so each bin costs one cross-thread
reduction. With `n_targets = 64` that spans two planes, so `plane_sum` plus a
small shared-memory combine. 256 reductions per workgroup sounds like a lot but
is ~1280 shuffle ops against the 16384 global writes it replaces.

Delete `prefix_sum_bins` and the three `cum_*` tensors from the RF path
afterwards.

Expected: 42% of runtime to ~14%. Roughly 1.4x overall.

### Step 2: remove the atomics from the histogram build

Largest single item at 57%, and the pattern is already proven: it is what took
ET from 8x to 28x.

Restructure `build_hist_privatised` as (sample-block, target) with compaction:

1. Every thread tests one sample per 128-wide block for node membership.
2. Survivors compact into shared memory via a plane-based workgroup exclusive
   scan, carrying `(sample_id, bin, multiplicity)`.
3. Thread `k` owns target `k` and walks the compacted tile, accumulating into
   `hist[bin, k]`.

The point of step 3 is that **thread `k` is the only writer of column `k`**, so
the accumulation needs no atomics at all. That removes ~1.4e9 global CAS retry
loops.

Compaction is what makes it affordable. Without it, per-thread column ownership
would need each of the 64 target-threads to test every sample for membership,
which is the `n_samples * n_targets` redundancy that made the first ET rewrite
slow.

**Start by reading `accumulate_split_stats_et` in
`src/gpu/sc_gpu/scenic_gpu.rs`.** Its `use_plane` branch is this exact pattern,
already working and gated: block loop, per-thread hit test, `plane_exclusive_sum`
plus `plane_sum` into `s_ptot`, offset by the totals of the planes below, scatter
into `s_id` / `s_mult`, drain over `cnt`. Step 2 is that loop with two changes:
carry the sample's bin into the compacted tile alongside its id and multiplicity,
and have the drain accumulate into `hist[bin, k]` rather than into a register.
Do not re-derive it; the ordering of the two `sync_cube()` calls and the
`lane == 0` guard on the `s_ptot` write are both load-bearing.

`plane_compact_viable` already exists for the launch-side gate, as does the
portable non-compacted fallback, so both can be reused as they stand.

Expected: this is the uncertain one. CAS loops under contention typically cost
2-5x a plain read-modify-write, so 57% to somewhere in 15-30%.

**Decision point.** After steps 1 and 2, measure. If RF is not at 4x or better,
the remaining steps are unlikely to close a 8.25x gap and the honest answer is to
stop and leave RF on CPU.

### Step 3: drop the sum-of-squares histogram

Halves what remains of both hot kernels. Small but real behaviour change.

Expanding the split objective:

```
wl*vl_k + wr*vr_k = (ssyl_k + ssyr_k)/n - syl_k²/(n·nl) - syr_k²/(n·nr)
```

and `ssyl_k + ssyr_k = ssq_k`, the node total, which is constant in `thr`. So the
argmax reduces to maximising

```
G(thr) = S_L(thr)/nl + S_R(thr)/nr,   S_L = Σ_k syl_k²,  S_R = Σ_k (sy_k − syl_k)²
```

`hist_y_sum_sqs` drops out of the search entirely. Node-level `ssq_k` is still
needed for the variance gate (`n_targets` per node, nothing), and the winning
bin's `ssyl_k` for `accumulate_importance`, which is one cheap pass at a known
threshold.

This is cuML's standard MSE gain formulation, not a novel rewrite.

**Caveat.** The current code clamps each per-target variance at zero before
summing and the algebraic form does not, so results differ wherever f32
cancellation drives a variance slightly negative. Gate on the 0.95 floor and
watch whether 0.988 moves.

Expected: ~2x on the remaining histogram terms.

### Step 4: make `N_BINS` a comptime parameter

Currently fixed at 256. LightGBM defaults to 255 and XGBoost to 256, but both
run at 63/64 routinely with small accuracy cost, and at `min_samples_leaf = 50`
on 10k cells a deep node holds ~100 samples spread across 256 bins.

4x on everything histogram-sized. This is an accuracy trade rather than a free
win, so it needs a decision on what floor is acceptable, and it changes results
for the CPU path too if the binning is shared.

### Step 5: histogram subtraction

Build the histogram for the smaller child only and subtract it from the parent
to get the larger. Universal in GBDT (LightGBM, XGBoost, CatBoost all do it).
Halves the build pass at the cost of keeping parent histograms resident.

Does *not* apply to the ET path, for the record: min/max of a union is not
derivable by subtraction and ET's left sums sit at a per-node random threshold.

## Projection, and the recommendation

Stacking optimistically: step 1 takes 42% to 14%, step 2 takes 57% to ~19%,
step 3 halves both, step 4 quarters what is left.

| after | share of current runtime | ratio |
|---|---:|---:|
| today | 100% | 1.28x |
| steps 1+2 | ~40% | ~3.2x |
| + step 3 | ~23% | ~5.5x |
| + step 4 | ~12% | ~10x |

So break-even needs steps 1, 2 and 3, and a comfortable win needs step 4, which
is an accuracy trade. That is three substantial kernel rewrites plus a numerics
decision for a result that lands near parity.

**Recommendation: do steps 1 and 2, then stop and re-measure.** They are the two
with no accuracy cost, they are worth ~2.5x between them, and step 2 reuses
machinery that already exists. If the decision point shows 4x or better, steps 3
and 4 become worth arguing about. If it does not, RF stays on CPU, which is a
perfectly good outcome: RF on CPU is 187s, ET on GPU is 64s, and users who want
speed have a fast path.

Worth being explicit that "leave RF on CPU" is not a failure. `run_scenic_grn_gpu`
already rejects GBM with `GpuNotSupportedForLearner`; extending that posture to
RF, or simply documenting that RF is CPU-preferred, is a legitimate shipping
decision.

## Scaling, if large cell counts matter

Independent of the speed work, and arguably more urgent if anyone wants to run
1M-cell datasets.

- ~~`feature_data_gpu` is u8 widened to u32~~. **Done.** Now packed four bins
  per u32 via `feature_bin`, so `n_features` bytes per cell rather than
  `4 * n_features`, and `fit_scenic_batches_gpu` checks it against
  `max_page_size` with an actionable error instead of failing silently.
- **RF's wave tensors do not scale.** At 1M cells `max_active_nodes` saturates
  at the depth cap (1024) and RF needs ~8.4 GB at **wave 1**, against ET's
  ~530 MB at wave 32. RF will refuse to run at large cell counts long before ET
  is uncomfortable. Steps 3 and 4 above (dropping the sum-of-squares histogram,
  and `N_BINS` at 64) are worth 8x on that footprint between them, so they are
  capacity fixes as much as speed ones.
- Two cheap ET cleanups found while looking at this: `fit_scenic_batches_gpu`
  uploads the sparse Y unconditionally even though `run_wave_bfs` only reads it
  on the RF branch, and `upload_dense_y` reallocates its host buffer per batch
  rather than reusing one scratch allocation.

## Verification

```bash
cargo test --release --features gpu,single-cell,large_scale_diagnostics --test scenic_gpu -- --nocapture
cargo bench --features gpu,single-cell --bench gpu_scenic_micro
```

Watch `rf_32t` / `rf_multibatch` and the two RF Pearson means (0.988, 0.976). At
milestones run the full `gpu_scenic_bench`, but **fix the ordering dependency
first**: it runs `rf_e2e_cpu` immediately after `et_e2e_gpu`, and that is how the
original archive got a thermally throttled 427.36s RF CPU baseline. Either
interleave a cooldown or run the CPU rows in a separate invocation.
