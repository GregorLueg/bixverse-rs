# SCENIC on GPU: the whole effort

Two rounds of work on the SCENIC tree regressor, both finished. Live numbers and
the current design live in `docs/scenic_gpu.md`; these files are the record of
how it went, kept mainly for the things that turned out to be false.

| | ExtraTrees | RandomForest |
|---|---|---|
| record | `scenic-gpu-extratrees.md` | `scenic-gpu-randomforest.md` |
| per batch, vs one CPU core | 1.49x -> 32.8x | 1.18x -> 18.9x |
| end to end, vs the 10-core CPU | 7.2x slower -> 2.0x faster | 6.0x slower -> 1.8x faster |
| wave VRAM | 6.10 GiB -> 0.012 GiB | 6.10 GiB -> 0.010 GiB |

Both learners are now GPU wins. RandomForest is the faster of the two in
absolute terms, which nobody predicted at any point: it subsamples 0.632 of the
cells per tree, and once the histogram stopped dominating, that showed.

## What the two rounds had in common

**The starting document was wrong both times.** The ExtraTrees round began from
an archived conclusion that the whole thing was a hardware loss on Apple
Silicon. The RandomForest round began from a six-step plan whose largest item
was a 31% regression and whose biggest actual win was not on the list. In both
cases the arithmetic in the document was fine and the conclusion drawn from it
was not.

**Both were won by deleting work, not by tuning it.** ExtraTrees built a 256-bin
histogram and read one row of it. RandomForest built one and pushed it through
DRAM three times. Neither kernel could be improved in place; both had to stop
existing. Every attempt to optimise the histogram kernels *as written* measured
zero or negative, across both rounds.

**The profiler settled every question and paper reasoning lost every one.**
Across the two rounds, the count is roughly seven confident predictions wrong.
Three of those were "this obviously has to help".

## What differed, and it is the more useful half

ExtraTrees was **bandwidth and atomics** bound: the win came from removing
traffic and CAS retry loops, and the compaction pattern that made the drain loop
proportional to the match count.

RandomForest, once fused into shared memory, was **latency** bound and nothing
else. It ran at ~6% of memory-issue capacity, ~9% of bandwidth and 0.3% of peak
FLOPs. In that regime every traffic- or instruction-count optimisation measured
exactly zero, and only two things paid:

- issuing more loads per thread before consuming any (unroll 16: **2.1x**)
- fitting more workgroups per core by cutting shared memory (32 bins: **1.75x**)

The same technique therefore has opposite value in the two rounds. Staging a hot
row in shared memory was worth 3.7x in one crate and 0% here, because here every
thread wanted the same address and the hardware already broadcasts it. **Diagnose
the wall before picking the fix** is the one rule that would have saved the most
time in both rounds.

## Caveats that outlived the work

- Every GPU-vs-CPU test builds its bins with `QuantisedStore::from_raw` and
  uniform random values. The real quantisation path, `from_csc`, has never been
  compared against the GPU. `phase3_rf_pearson_skewed_bins` is a proxy, not a
  substitute.
- The fidelity gates correlate importance *vectors*, i.e. feature ranking. They
  stayed flat while the GPU bin axis was coarsened 64-fold, which says the metric
  is insensitive to threshold resolution, not that resolution is free. The bin
  count was deliberately left above the fastest setting for that reason.
- End-to-end rows are single-shot. Treat the last few percent as noise.
- All numbers are one M1 Max at 20% target density. Denser data favours the GPU
  further; other hardware is unmeasured.

## Also in this directory

`gpu-kmeans-and-sparse-svd.md` is a separate effort, on GPU k-means (5.0-5.2x end
to end) and the sparse randomised SVD (~2.7x). It repeats the pattern above from
a third and fourth angle: the profile contradicted the plan both times, and in
each case the biggest win sat in code the plan had only meant to tidy. It also
adds the case where a library GEMM, rather than anything hand-written, was 84.7%
of GPU time.
