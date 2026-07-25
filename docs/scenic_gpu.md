# SCENIC on GPU

Short version: **ExtraTrees on GPU beats the CPU by 3.7x end to end and is the
path worth using. RandomForest does not and should stay on CPU for now.** GBM
is a GPU non-goal.

This file replaces `docs/archive/scenic_gpu_experiment.md`, which concluded the
whole thing was a hardware loss on Apple Silicon. That conclusion was wrong, and
the section at the bottom records why, because the reasoning failed in ways that
are easy to repeat.

## Current numbers

M1 Max (8 P-cores + 2 E, 32 GPU cores, 400 GB/s unified). `gpu_scenic_bench`,
10k cells, 1000 TFs, 250 trees, `max_depth = 10`, `min_samples_leaf = 50`.

End-to-end, 4000 target genes in ~63 batches:

| learner | CPU | GPU | result |
|---------|----:|----:|--------|
| ET | 236.46s | 64.44s | **GPU 3.67x faster** |
| RF | 187.39s | 1131.63s | GPU 6.0x slower |

Single 64-target batch:

| learner | CPU (1 core) | GPU | ratio |
|---------|-------------:|----:|------:|
| ET | 33.80s | 1.03s | **32.8x** |
| RF | 24.55s | 19.24s | 1.28x |

**These are measured at 50% target density, which flatters the GPU.** The
benches now default to `SPARSITY = 0.2`, the dense end of realistic single-cell
data. At 0.2 the same shape gives:

| cell | @0.5 | @0.2 |
|---|---:|---:|
| `et_32t` | 28.33x | **19.62x** |
| `et_multibatch` | 27.76x | **18.93x** |
| `rf_32t` | 1.29x | 1.17x |

The GPU wall clock is *identical* at both densities (0.15s for `et_32t`); it is
the CPU that gets faster. That is structural: the ET path reads `n_targets`
dense values per matched sample regardless of how many are nonzero, so its cost
is density-independent, while both CPU paths and the RF histogram build do work
per nonzero. Expect the e2e ET win to be ~2.5x at 0.2 rather than the 3.67x
measured at 0.5. Metacells are denser than raw cells, so the metacell path sits
nearer the top of that range.

The CPU e2e driver fans out over gene batches with rayon while
`fit_multi_trees_sparse` is sequential over trees, so the fan-out is worth about
9.1x and that is the bar the GPU has to clear per batch. ET clears it three
times over; RF does not come close.

RF is cheaper than ET on CPU because `RandomForestConfig` defaults to
`subsample_rate = 0.632`, so each tree sees 63% of the cells, while ET uses all
of them. Worth remembering when comparing the two.

## How the ExtraTrees path works

Level-synchronous BFS, `wave_size` trees at a time, no tree structure stored,
importance the only output. There is no histogram.

`ExtraTreesConfig` draws `n_thresholds` random thresholds per (node, feature
slot), defaulting to 1. So instead of binning every sample and prefix-summing,
each level runs:

1. `sample_node_features` picks `k_feats` features per (tree, node).
2. `scan_slot_bin_range` reduces the node's occupied bin range to a min and max.
3. `accumulate_split_stats_et` draws the threshold from that range and sums the
   left-side Y at it in one pass over the samples.
4. `evaluate_splits_et_direct` scores the candidates and picks the winner.
5. `accumulate_importance`, `compute_child_ids`, `propagate_child_stats`,
   `reassign_samples` advance the level.

Two details carry most of the performance.

**Threads are laid out over samples, not over (stripe, target).** Each thread
tests one sample per 128-wide block, so `sample_to_node`, `sample_multiplicity`
and `feature_data` are read once per sample rather than once per (sample,
target). Survivors are compacted into shared memory through a plane-based
workgroup exclusive scan, so the drain loop runs over the match count rather
than the block width. At depth a node holds ~100 samples and matches once or
twice per block, so this is the difference between a 128-iteration loop and a
one-iteration one.

**Y is uploaded dense as well as sparse.** At `MULTI_OUTPUT_BATCH = 64` targets
that is `n_samples * 64 * 4` bytes, ~2.5 MB at 10k cells. It puts the target on
the thread axis, so consecutive threads read consecutive addresses and the
accumulation lives in registers with one shared-memory fold at the end. No
atomics at all.

Node statistics are propagated rather than recomputed: a child's totals are
exactly the parent's split outputs (left gets `split_n_left` / `split_y_sums_l`,
right gets parent minus left), which is what the CPU recursion does. `merge_hist`
survives only at depth 0.

Wave-scoped VRAM is ~13 MB at the reference shape, down from 6.1 GiB, which is
why `DEFAULT_WAVE_SIZE` could go from 8 to 32.

## How the RandomForest path works, and why it loses

Unchanged: `build_hist_privatised` builds a 256-bin x `n_targets` histogram per
(tree, node, slot), `prefix_sum_bins` turns it into a cumulative table, and
`evaluate_splits_rf` sweeps all 255 thresholds against it. RF genuinely needs
that table, so the ET approach does not port.

Per-kernel profile at the reference shape: `build_hist_privatised` ~57%,
`prefix_sum_bins` ~36%, `evaluate_splits_rf` ~7%. The threshold sweep is nearly
free; the cost is moving the histogram through DRAM and the atomics that fill
it. `build_hist_privatised` does one CAS-retry-loop float add per (sample,
target) in global memory, about 1.4e9 of them per fit, which accounts for its
time almost exactly.

See `plans/` for the RF work plan.

## Where the original archive went wrong

Kept deliberately. The measurements in it were real; the inferences were not.

**"The gap is the hardware, not the code."** The arithmetic behind this was
right: the CPU gets ~9x from rayon and the GPU had one queue. But the GPU was
beating a *single* CPU core by only 1.5x on a 32-core GPU with 400 GB/s. That is
a statement about the kernel. The correct reading of "we lose to N cores after
beating one by 1.5x" is "the kernel is leaving an order of magnitude on the
table", not "the hardware is wrong".

**The wave size claim.** Root cause #3 and the "single highest-leverage
experiment left" both describe a wave of 4. The bench overrides the VRAM budget
to 12 GiB, so the measured runs were at wave 8. The in-place prefix sum it
recommended would have bought nothing.

**"The VRAM hog is the sum buffers, not the features."** Right on capacity,
wrong on bandwidth, and irrelevant either way since neither was the bottleneck.

**"wgpu/Metal has no host knob to run independent batches concurrently."** Not
accurate, and in any case the right lever is fatter launches, not concurrent
queues.

**The RF CPU baseline of 427.36s is not reproducible** and should be treated as
bad data. It now measures 187.39s on an untouched code path. The bench runs
`rf_e2e_cpu` straight after `et_e2e_gpu`, which was 27 minutes of saturating GPU
work at the time, so the CPU was almost certainly thermally throttled. RF is
therefore further behind than the archive recorded, not closer.

## The discriminator, revised

The archive's heuristic was that GPU wins in this codebase are GEMM-shaped, and
SCENIC lost because it is "thousands of small irregular tree-batches that the
CPU fans out with rayon". That is not what happened. SCENIC is now one of the
larger GPU wins in the crate and it contains no GEMM at all.

Two better questions to ask of a kernel:

1. **How much of what it computes is actually read?** ET was building 256 bins
   per slot to consume one. That ratio, not the shape of the work, was the whole
   problem.
2. **Is any per-element metadata being re-read once per output channel?** The
   (stripe, target) layout read the same three arrays 64 times over. Test once,
   compact, then fan out.

Both are visible in a five-minute profiler run and invisible to reasoning about
memory traffic on paper.

## Scaling to large cell counts

`n_targets` is hard-capped at `MULTI_OUTPUT_BATCH = 64`, so everything below is
linear in cells only. There is no global cell cap in the config:
`ScenicParams::n_subsample` only feeds the correlated gene-batching PCA, not the
fit.

| tensor | bytes per cell | at 1M cells |
|---|---:|---:|
| dense Y (per batch, freed after) | 256 | 256 MB |
| `feature_data_gpu`, four bins per u32 word | `n_features` | 1 GB at 1000 TFs |
| ET wave tensors, wave 32 | n/a, node-bound | ~530 MB |
| RF wave tensors, wave 1 | n/a, node-bound | ~8.4 GB |

Two things follow.

**`feature_data_gpu` is the binding that breaks first**, which is why it is
packed four bins per u32 word and unpacked on device by `feature_bin`. Unpacked
it would be 4 GB at 1000 TFs and 1M cells, exactly the per-binding limit on this
machine, plus a 4 GB host allocation to do the widening. Packing buys 4x on
both. `fit_scenic_batches_gpu` also checks the packed size against
`max_page_size` up front, because otherwise crossing it fails the silent
`launch_unchecked` way. Note this was investigated for *performance* during the
ET work and correctly rejected, since the sample scan is free; it earns its keep
on capacity alone.

**On larger systems this is mostly fine for ET and not for RF.** Datacentre
cards have far higher per-binding limits, so a 1M-cell run with the u32 feature
tensor would work there, just wastefully. ET's wave tensors stay around 530 MB
because `max_active_nodes` saturates at the depth cap of 1024. RF's are ~8.4 GB
at **wave 1**, so RF would refuse to run at 1M cells under any sane budget long
before ET is uncomfortable. If large-cell-count runs are a target, ET is the
only path that scales.

## Hardware caveat

All of this is measured on Apple Silicon: one queue, unified memory shared with
the CPU, plane size 32, a 4.00 GiB per-binding limit. The per-binding limit in
particular is worth knowing about, because `launch_unchecked` dispatches that
exceed it do no work and report no error. `pick_wave_size` checks both that and
the total budget.

## Pointers

- Code: `src/gpu/sc_gpu/scenic_gpu.rs`
- Tests: `tests/scenic_gpu.rs`. Note this file is gated on
  `large_scale_diagnostics` as well as `gpu` and `single-cell`, so the CI gpu
  job currently compiles it to zero tests.
- Benches: `benches/gpu_scenic_micro.rs` (tight loop, ~1 min),
  `benches/gpu_scenic_bench.rs` (full, ~1 h)
- CPU siblings: `run_scenic_grn` / `run_scenic_grn_streaming`
  (`sc_analysis/scenic.rs`), `run_scenic_grn_in_memory`
  (`mc_analysis/scenic_metacells.rs`)
- Profiling: `CUBECL_DEBUG_OPTION=profile-medium CUBECL_DEBUG_LOG=stdout`
