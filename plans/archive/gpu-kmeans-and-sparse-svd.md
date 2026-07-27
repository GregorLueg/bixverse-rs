# GPU k-means and sparse randomised SVD

Done. Archived record of the work, kept for the wrong predictions as much as the
result. Both paths had never been profiled or benchmarked at the shape they
actually run at, so the whole effort started with building the harness.

## Status

**k-means went 5.0-5.2x end to end at the Harmony production shapes, and up to
14x on the device loop alone. Sparse randomised SVD went ~2.7x.**

End to end, `KMeansInit::KMeansParallel`, which is what `k <= 200` selects and
therefore what production runs:

| shape | before | after | |
|---|---:|---:|---:|
| 100k, k=100, d=32 | 415 ms | **79.5 ms** | 5.2x |
| 1M, k=100, d=48 | 5.16 s | **995 ms** | 5.2x |
| 1M, k=200, d=48 | 10.86 s | **2.19 s** | 5.0x |

Device loop only, random init, to isolate the kernel work:

| shape | before | after | |
|---|---:|---:|---:|
| 100k, k=100, d=32 | 249 ms | **54.1 ms** | 4.6x |
| 1M, k=100, d=48 | 4.01 s | **534 ms** | 7.5x |
| 1M, k=200, d=48 | 4.70 s | **751 ms** | 6.3x |
| 10k, k=400, d=128 | 693 ms | **61.9 ms** | 11.2x |
| 10k, k=400, d=512 | 2.86 s | **201 ms** | 14.2x |

Per kernel, at 1M / k=100 / d=48:

| kernel | before | after | |
|---|---:|---:|---:|
| `FlashAssignCosine` | 34.35 ms | **9.44 ms** | 3.6x |
| `SegmentedCentroidUpdate` | 32.78 ms | **2.63 ms** | 12.5x |

Sparse randomised SVD at 200k x 2000, 4e7 non-zeros, s = 130: **2.31 s -> 0.89 s**.
The Gram GEMM inside CholeskyQR2 went **223 ms -> 8.67 ms per launch, 25.7x**.

## What the measurements settled

**The k-means diagnosis was half wrong, and the half that was wrong cost the
most.** The plan said the centroid update dominated. The first profile:

| kernel | % GPU |
|---|---:|
| `FlashAssignCosineTiled` | 51.4% |
| `SegmentedCentroidUpdate` | 48.0% |
| the other six | 0.5% |

Near-equal, so fixing either alone caps the loop at ~2x. Worse, the change that
mattered most for the assignment kernel was ranked in the plan as a
low-priority "measure later" item: **deleting its shared-memory centroid tile**.
Every thread reads the same centroid element at the same time, so the hardware
broadcasts it from cache and the staging bought nothing while forcing the point
to be exploded from `dim_lines` vectors into `dim_lines * LINE_SIZE` scalars.
That spills at high dim. Dropping the tile and keeping the point vectorised was
worth 2.3x at dim 128 and 2.9x at dim 512.

**The SVD bottleneck was in a file the plan only meant to tidy.** The plan
reserved a whole deferred step for `spmm_csc_transpose`, described as the
suspected dominant cost. It is 9.7%. The actual profile:

| kernel | % GPU | ms/launch |
|---|---:|---:|
| `MatmulEntry` (cubek) | **84.7%** | 122.19 |
| `SpmmCscTranspose` | 9.7% | 60.46 |
| `SpmmCsrForward` | 5.4% | 33.61 |
| the two reductions | 0.2% | |

The 26 matmul launches were bimodal: 12 at 223 ms, 12 at 36.5 ms, 2 at 18.5 ms.
The mean told me nothing and the spread identified the culprit immediately. The
223 ms one is `G = Y^T Y`, and cubek's `Strategy::Auto` picks `SimpleUnit` for
it: one thread per output element, no split-K. A `[130, 130]` output is 16 900
elements, so that is almost no parallelism against a 200 000-long reduction. It
ran at **0.3% of peak**.

**Host-side work was 76-86% of wall time and nobody had ever looked.** Once the
kernels were 5-9x faster, `kmeans_parallel_init` on the CPU was roughly 3x the
entire device loop: 1.9 s at k=100 and 5.4 s at k=200, against a device loop
well under a second. It had always been that slow, just hidden behind slower
kernels. Timing host and device separately in the bench is what surfaced it, and
a single end-to-end number would have hidden it permanently.

## The steps that paid

| step | result |
|---|---|
| drop the SMEM centroid tile, vectorise the assign kernels | 1.1-2.7x on the loop |
| widen the centroid update 32 -> 128 threads, unroll 8-wide | 1.4-2.5x on the loop, 12.5x on that kernel |
| score 8 centroids per unrolled step in assign | 1.5-3.4x on the loop, 2.6x on that kernel |
| k-means|| initialisation on the device | 2.5-3.0x on end-to-end wall clock |
| split-K Gram kernel replacing cubek | 2.31 s -> 1.00 s on the SVD |
| tall-skinny GEMM kernel replacing cubek | 0.97 s -> 0.83 s on the SVD |

Two of these are the same lever. The centroid update and the assign kernel both
sat at single-digit percentages of peak FLOPs, bandwidth and memory-issue
capacity, which is the latency-bound signature, and both responded to hoisting
loads above their consumers so several are in flight. That is the third and
fourth time this specific fix has been the answer in this crate.

The device-side k-means|| needed one new kernel (nearest-candidate distance,
which is the assignment kernel keeping the distance instead of the index) plus
one host-side algorithmic fix that has nothing to do with the GPU: the sampling
loop rescanned the whole weight vector per draw, `O(k * oversampling * n)`, which
is 2e8 sequential steps per round at n=1e6, k=200. A prefix sum plus a binary
search per draw is `O(n + count * log n)`.

## Negative results

- **Reusing the CholeskyQR2 `[n, s]` scratch to dodge first-touch page faults:
  0%.** This was the highest-confidence item in the SVD plan, justified by a
  measured 39 ms per 1.4 GB from the ann-search-rs work. At 104 MB it does
  nothing. Effects that are real at one size are not proportionally real at
  another. Kept anyway, it is free and correct.
- **Widening the SpMM workgroup so `s = 130` fits in one column pass: 3.5%.**
  The traffic argument is sound (every row's indices and values were streamed
  twice for two columns of useful work) and the kernels did move 1.36x and
  1.10x, but neither is bound by index streaming.
- **`CENTROID_UNROLL = 16` regressed two of five shapes** after 8 had beaten 4 by
  up to 1.5x. Unroll knees have to be swept, not inherited.
- **The five-kernel privatised CSR pipeline is 0.5% of GPU time.** The plan had a
  step to restructure it. Left alone.
- **A Gaussian sketch did not improve accuracy over the uniform one.** See below.

## The sketch, and why it changed anyway

`omega_scaled` was drawn uniform on [0, 1), so it is not zero-mean and carries a
rank-1 all-ones component that every sketch column shares. The three CPU
randomised SVDs in `core::math::pca_svd` all use `Normal::new(0.0, 1.0)`, so the
GPU path was the odd one out.

The prediction was that this hurt accuracy and was why `n_power_iters` needs to
be 2. It does not, and it is not. Worst relative singular-value error against a
dense faer reference:

| power iters | uniform | Gaussian |
|---|---:|---:|
| 0 | 0.1874 | 0.1843 |
| 1 | 0.0406 | 0.0424 |
| 2 (production) | 0.0085 | 0.0101 |

Indistinguishable. The non-zero mean costs about one dimension of an
s-dimensional sketch, and at s = 130 that is nothing.

What does hold is conditioning. Measured at m = 2000, s = 130:

| sketch | kappa(Omega) |
|---|---:|
| uniform on [0, 1) | 26.58 |
| standard normal | 1.69 |

CholeskyQR forms `G = Y^T Y`, so `kappa(G) = kappa(Y)^2`, and in fp32 it fails
outright above roughly `kappa(Y) = 3000`. It fails as a returned
`NonPositivePivot` error, not as a degraded answer, and there is no fallback.
Since `kappa(Y) <= kappa(X) * kappa(Omega)`, the Gaussian sketch buys about 16x
more headroom. That is the reason it was changed: robustness and consistency,
not accuracy.

This is not hypothetical. Two synthetic bench generators in a row tripped that
failure before the SVD bench ran once, first from a low-rank value model and then
from a sparsity pattern with only ten distinct column supports, which put enough
block structure into the centred matrix to blow up its condition number. If a
linear-algebra bench refuses to run, suspect the data generator before the code
under test.

## Building an accuracy gate that can fail

The first version of the accuracy test returned 6.4% error for the good sketch
and 6.7% for the bad one, and would have "shown" they were equivalent. The
synthetic matrix had no spectral gap, so truncation error swamped the thing the
test existed to detect. Strengthening the low-rank signal turned the same test
into 0.184 / 0.042 / 0.010 across 0, 1 and 2 power iterations.

The committed test asserts on the **sweep**, not just the endpoint: it fails if
the error does not fall at least 4x from zero power iterations to two. A single
tolerance cannot catch an insensitive gate; that assertion can.

## Caveats

- One M1 Max, and nothing else was measured.
- The SVD bench runs 200k x 2000. The 1M x 2000 production shape is behind
  `BIXVERSE_BENCH_BIG=1` and has not been run, so the ~3.9 GB memory path and the
  520 MB first-touch question are both still unmeasured.

  **This caveat was the bug.** `tall_skinny_mm` put `n / TSMM_ROWS` straight on
  the x grid dimension, which is over the 65535-per-dimension dispatch limit
  from 524_280 rows upward. Every shape that was ever benchmarked fit; the
  900k-cell production run did not, and surfaced as an unrelated `CallError`
  once the rejected launch killed the cubecl server thread. Fixed by flattening
  the row-block axis over `(x, y)` via `grid_2d`, plus a `checked_cube_count`
  guard that turns a busted limit into a typed error. The 1M shape now runs at
  4.77 s.
- SVD wall clock includes substantial host work (the CSR transpose of 4e7
  non-zeros, ~1 GB of uploads) which was never touched and is now a large share
  of the total. The `host CSC build` column doubles under load and is the
  invariant to cross-check before trusting any SVD timing.
- k-means numbers use synthetic data with a coarse cluster signal. Cluster
  balance measurably changes the update kernel's cost, so real data will differ.
- The accuracy tolerance in `test_randomised_sparse_svd_gpu_accuracy_vs_dense` is
  calibrated to that specific synthetic matrix.

## Left on the table

- `MatmulEntry` was still 46% of GPU time after the Gram fix and before the
  tall-skinny one. It was never re-profiled after, so the current split is
  unknown.
- The assignment kernel is ~73% of what remains of the k-means loop.
- Host-side CSR transpose and upload in the SVD, now a large share of its wall
  clock.
- Row-blocking `spmm_csc_transpose` so `Q` stays cache-resident. Still the
  largest sparse-kernel item, still never justified by a profile.
