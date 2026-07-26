# Faster GPU correlation: a symmetric Gram kernel to replace cubek

## Context

`src/gpu/linalg/corr.rs` computes pairwise column correlation, covariance and
Spearman on the GPU. After centring and scaling, the whole thing is one product:
`G = S^T S`, dispatched through cubek's `dense_gemm`.

That dispatch is currently pinned to `Strategy::DoubleUnit` because
`Strategy::Auto` blows up on Apple devices (commit `b1f4152`, "otherwise, this
can blow up on Apple Silicon"). `DoubleUnit` is a unit-level strategy: one thread
per output element with a serial reduction over the inner dimension, so it
re-reads both operand rows from global memory for every output element. We
already measured what that costs in the SVD work. Auto picked `SimpleUnit` for
`Y^T Y` at `[130, 200k] x [200k, 130]` and ran at **0.3% of peak**, 223 ms per
call; a hand-written split-K Gram kernel took it to 8.67 ms (25.7x, commit
`dd8c6f4`). corr.rs is the same product with the same wrong algorithm underneath.

So the workaround for a correctness problem left a performance problem behind,
and the fix is the same one that worked for the SVD: stop asking a generic
library GEMM to handle a shape it has no strategy for.

Two secondary costs matter at large `d`. `result.read()` pulls `d * d * 4` bytes
back (1.6 GB at `d = 20000`), and then

```rust
Ok(Mat::from_fn(n_cols, n_cols, |i, j| result_flat[i * n_cols + j]))
```

does a transposing, single-threaded, cache-hostile copy of the whole thing.
`Mat::from_fn` fills column-major, so this strides by `d` on every read. `G` is
symmetric, so that transpose is pure waste.

Intended outcome: corr.rs uses its own kernel on every backend, `Strategy::Auto`
is out of the picture, and the host-side assembly stops paying for a transpose it
does not need.

## Rebase first

This worktree sits on `5e4a69a`. `a6d2b3b fix(svd): tall-skinny GEMM grid busts
the dispatch limit past 524k rows` landed on `fix-scenic-gpu` afterwards and is a
clean fast-forward, no conflicts:

```bash
git rebase fix-scenic-gpu
```

Do this before writing any code. The fix adds `checked_cube_count` and the
`GpuCubeCountExceeded` error variant that the rest of this plan depends on, and it
documents a failure mode that the design below would otherwise have walked
straight into.

## What the operation actually is

Not general dense x dense. It is a **symmetric Gram product**, which is a much
more constrained problem:

- Both operands are the same buffer, so there is one input stream, not two.
- The output is symmetric, so half the tiles are free.
- The layouts are fixed and known, not arbitrary strides.

That is why a specialised kernel can beat a library GEMM here without being a
better GEMM writer than the cubek authors. It is also why **no OS gate**: the
kernel wins on shape structure, not on Metal quirks, and `cfg!(target_os =
"macos")` would leave one path that CI never exercises on the shapes we care
about. cubek's `dense_gemm` stays exactly where it is for `harmony_gpu` and
`sparse_rand_svd_gpu`; corr.rs just stops calling it.

### Layout

`scale_matrix_col_gpu` already produces `scaled` in **feature-major** order,
`[n_cols, n_rows]` row-major, i.e. `A[i * n + r]`. Feature `i`'s data is
contiguous over rows. So the target is

```
G[i, j] = sum_r A[i, r] * A[j, r]        i.e. G = A A^T
```

Keep that layout. It gives `column_stats` a contiguous per-column reduction (the
point of commit `cd38e8d`) and gives the new kernel contiguous staging runs along
`r`. No transpose pass anywhere.

## One kernel, all three regimes

The tile structure is identical for fat, tall and square. The only thing that
changes across regimes is **how many row chunks the reduction is split into**,
and that is a host-side integer. So this is one kernel plus one trivial reduce,
not a family of kernels.

| regime | example | tiles available | chunks |
|---|---|---|---|
| fat, `d >> n` | 20000 x 500 | 49k upper-triangular | 1 |
| tall, `n >> d` | 2000 x 500k | 528 | 1 |
| square | 2000 x 2000 | 528 | 1 |
| narrow, small `d` | 100 x 1e6 | 3 | many |

`chunks == 1` is the common case and writes straight into `G` with no reduce
launch. Only the narrow case needs split-K, and it needs it because three
workgroups cannot fill a 32-core GPU, not because of anything about the memory
traffic.

### `gram_symmetric` kernel

Standard register-tiled GEMM structure, specialised two ways: the two staged
operands come from the same buffer, and each workgroup writes its tile twice.

- Workgroup owns a `GRAM_BM x GRAM_BM` block of `G` and one row chunk.
- 256 threads as 16x16, each accumulating a `GRAM_RT x GRAM_RT` register tile.
- Stages `GRAM_BK` columns of the two feature slices into two shared arrays,
  accumulates outer products, `sync_cube` and advance.
- Grid: `(tiles, tiles, n_chunks)`, with `terminate!()` when `CUBE_POS_Y <
  CUBE_POS_X`. Exactly 50% over-dispatch, and an immediately-exiting workgroup
  costs nothing, so this is the cheapest way to get the symmetry saving. (Section
  19 of the cubecl notes says a cube-level early exit recovered nothing in a case
  with *near-uniform* work; here the skipped half is genuinely half the FLOPs.)
- Off-diagonal tiles write both `G[gi, gj]` and `G[gj, gi]`. Emit the mirror as
  `#[unroll]`-ed `for b { for a { G[(j0 + ..b) * d + i0 + ..a] = acc[a][b] } }`
  so each thread still writes `GRAM_RT` contiguous floats per run and the mirror
  coalesces as well as the primary. Diagonal tiles (`CUBE_POS_X == CUBE_POS_Y`)
  write once.
- No race: each unordered tile pair is owned by exactly one workgroup.

Staging coalescing, which is what drives the `GRAM_BK` choice. Map the load
index as `(feature = lid / GRAM_BK, k = lid % GRAM_BK)` so consecutive lanes read
consecutive `r` within one feature. `GRAM_BK = 8` gives 32-byte runs, `16` gives
64-byte, `32` gives a full 128-byte line but costs occupancy. Sweep it.

Shared-memory footprint, per section 5 of the cubecl notes. Write the formula
down and derive the budget from `client.properties().hardware.max_shared_memory_size`,
never a hardcoded 32768:

```
2 * GRAM_BK * GRAM_BM * size_of::<F>()
```

| BM | BK | RT | footprint | resident at 32 KiB |
|---|---|---|---|---|
| 64 | 8 | 4x4 | 4 KiB | 8 |
| 64 | 16 | 4x4 | 8 KiB | 4 |
| 128 | 8 | 8x8 | 8 KiB | 4 |
| 128 | 16 | 8x8 | 16 KiB | 2 |

Start at `BM = 64, BK = 8, RT = 4x4` (16 accumulators, no spill risk) and sweep
the other three. `128 / 8x8` means 64 accumulator registers per thread, which is
where section 16's scalar-explode-and-spill failure lives, so treat it as a
candidate to measure, not a default.

Balance check for the starting config, per r-step of 8 rows: 1024 global loads,
16384 shared reads, 32768 FMAs. Global traffic is 0.03 ops per FMA (irrelevant),
shared reads are 0.5 (the thing the register tile is buying down). This is why
**vectorised staging is not the lever here** and is deliberately not in scope;
`SharedMemory::<Vector<F, 4>>` to cut the shared-read count is the follow-up if
the profile says shared-memory issue rate is still the wall.

### Dispatch limits: where this design busts them

`a6d2b3b` is a warning about exactly this class of kernel. Putting a
data-dependent block count straight on one grid dimension is over the
65535-per-dimension limit sooner than it looks, and the failure is nasty: the
launch is rejected on the cubecl server thread, **that thread dies**, and every
later call on the client returns an unrelated `CallError` from somewhere else.
That is a third silent-failure mode on top of sections 4 and 5 of the cubecl
notes, and it does not look like a dispatch problem at all from the outside.

Audit of every launch in the corr path, with the limit from `R::max_cube_count()`
and never a hardcoded 65535:

| launch | grid | busts at | verdict |
|---|---|---|---|
| `column_stats` | `(d, 1, 1)` | `d > 65535` | unreachable behind the output guard, still route through `checked_cube_count` |
| `apply_centre_scale` | `grid_2d(n * d / 256)` | already flattened | fine as is |
| `gram_symmetric` | `(tiles, tiles, chunks)` | `d > 4.2M` on the tile axes | unreachable; `chunks` is capped by `GRAM_MAX_CHUNKS` |
| `gram_reduce` | `(d * d / 256, 1, 1)` | **`d > 4096`** | **broken as planned. Must flatten.** |

The reduce is the live bug. `cholesky_gpu`'s version puts `total.div_ceil(256)`
on x, which is fine there because `s = 130` makes it 66. At corr's `d = 20000`
that is 1.56 million, roughly 24x the limit. So the corr copy needs `grid_2d` on
the host **and** a flattened index in the kernel body,
`CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` rather than `ABSOLUTE_POS_X`, the same
change `tall_skinny_mm` just took.

It only bites in the split-K arm, which is the narrow-`d` case where `d` is small
and the reduce grid is tiny. So it would have sat there passing every test and
every bench, and only fired on a shape combining large `d` with a chunk count
above 1. Flatten it anyway and test the boundary rather than the middle, per
section 5.

Every launch goes through `checked_cube_count::<R>(name, x, y, z)?`, so a busted
limit is a typed error naming the kernel instead of a dead server thread.

### `gram_chunks` heuristic

Generalise the existing `gram_chunks` in `cholesky_gpu.rs:86`, which only looks
at `n`. The new one needs three terms:

1. Enough upper-triangular tiles to saturate: `T * (T + 1) / 2` where
   `T = ceil(d / GRAM_BM)`. Below a target workgroup count, split.
2. At least `GRAM_MIN_ROWS_PER_CHUNK` rows per chunk, or the per-chunk fixed cost
   and the reduce dominate.
3. Partials must fit: `chunks * d * d * size_of::<F>()` under both the VRAM
   budget and `client.properties().memory.max_page_size`.

Each term has a `const` with a doc comment giving the reasoning, per house style.

## Files

**New: `src/gpu/linalg/gram.rs`**, registered in `src/gpu/linalg/mod.rs`. Holds
the tuning consts, `gram_chunks`, `gram_symmetric`, `gram_reduce` (`cholesky_gpu`'s
version with the flattened grid index) and a `gram_aat` dispatcher taking a
feature-major `GpuTensor` plus `n`, `d` and the output tensor. Returns
`Result<(), BixverseErrors>` because every launch is `checked_cube_count`-guarded.
A separate module because it is a distinct primitive with its own tuning table and
tests, and corr.rs is already 660 lines.

**`src/gpu/linalg/corr.rs`**
- `column_pairwise_cor_gpu`: replace the `dense_gemm` call with `gram_aat`. Drop
  the `dense_gemm`, `Strategy` and `MatmulPrecision` imports. That removes the
  `MP: MatmulPrecision` generic parameter, so update the four call sites in
  `tests/gpu_corr.rs` and the five in the inline test module.
- Add the output-size guard before allocating `result` (see below).
- Fix the result assembly to read the row-major buffer as column-major, free by
  symmetry: `Mat::from_fn(d, d, |i, j| result_flat[j * d + i])`. `from_fn` fills
  column-major, so this reads `result_flat` sequentially instead of striding by
  `d`.
- `scale_matrix_col_gpu` becomes `Result<GpuTensor<R, F>, BixverseErrors>` once its
  two launches go through `checked_cube_count`. One `?` at the call site.
- Fix the two stale doc comments claiming `[n_rows, n_cols]` row-major on
  `apply_centre_scale`'s `out` and on `scale_matrix_col_gpu`'s return. Both are
  feature-major; `cd38e8d` updated the input docs and missed the outputs.

**`src/errors.rs`**: no change. `GpuCubeCountExceeded` arrives with the rebase, and
the output-too-large-for-a-binding case reuses `InvalidArgument` with a message in
the style of `scenic_gpu.rs:5655`. A dedicated variant for one call site is not
worth it.

**New: `benches/gpu_corr_bench.rs`**, registered in `Cargo.toml` with
`required-features = ["gpu"]`.

## Steps

0. **`git rebase fix-scenic-gpu`.** Fast-forward, no conflicts. Everything below
   assumes `checked_cube_count` and `GpuCubeCountExceeded` are present.

1. **Bench and baseline first.** Write `benches/gpu_corr_bench.rs` with separate
   timings for rank (Spearman), host flatten, upload, stats + scale, the product,
   readback and host assembly. Section 17 of the cubecl notes: a single
   end-to-end number hides which half is host, and the ratio inverts as soon as
   the kernel improves. Shapes: fat `20000x500`, tall `2000x100000`, square
   `2000x2000`, narrow `100x1000000`, behind a `BIXVERSE_BENCH_BIG` env flag for
   the expensive ones as `gpu_sparse_svd_bench.rs` does. Record the `DoubleUnit`
   numbers before touching anything.

2. **Profile it.** `CUBECL_DEBUG_OPTION=profile-medium CUBECL_DEBUG_LOG=stdout`.
   Confirm the GEMM dominates and read the `MatmulEntry` line to see what cubek
   actually chose. Look at the per-launch spread, not the mean. If the profile
   says something other than the GEMM dominates, stop and re-plan: the whole
   premise above is measured on the SVD's shapes, not these.

3. **Write `gram.rs`** with `GRAM_BM = 64`, `GRAM_BK = 8`, `GRAM_RT = 4`, split-K
   present but with the heuristic returning 1 for everything but the narrow case.
   `gram_reduce` gets the flattened `grid_2d` index from the start, and every
   launch in the module and in `scale_matrix_col_gpu` goes through
   `checked_cube_count`.

4. **Cross-check elementwise before timing** (section 3). Compare `gram_aat`
   against the current `dense_gemm` output over every entry at `2000x2000`. Also
   assert `trace(G) == d` for Pearson, which is a free invariant and catches the
   silent all-zeros failure mode.

5. **Sweep the tuning table.** The four `(BM, BK, RT)` configs above across all
   four shapes. Section 19: unroll and tile knees move between kernels, so do not
   inherit `cholesky_gpu`'s `GRAM_TILE = 16` / `GRAM_ROWS_STEP = 8` and assume it
   transfers. Pick per measurement, and if a single config is not best everywhere,
   dispatch on shape with the reasoning in a const doc comment.

6. **Output-size guard and the silent-failure defence.** Before allocating
   `result`, check `d * d * size_of::<F>()` against
   `client.properties().memory.max_page_size` and return the new typed error if it
   busts. `d = 32768` already exceeds 4 GiB in fp32, and per section 4 a
   `launch_unchecked` past a binding limit does no work, returns zeros and reports
   nothing. Do the same check for the split-K partials. Assert the trace invariant
   in the bench too, not just in the tests.

7. **Host-side fixes.** The symmetric result assembly and the doc corrections.

8. **Re-run the bench** and compare against the step 1 baseline, host and device
   split out.

## Verification

```bash
cargo test --features gpu -- gram          # new kernel unit tests
cargo test --features gpu -- corr          # existing corr tests, MP generic dropped
cargo test --features gpu,large_scale_diagnostics -- diag_pearson_sweep
cargo clippy --features gpu --all-targets
cargo fmt
cargo bench --features gpu --bench gpu_corr_bench
```

New tests in `gram.rs`: `gram_aat` against a CPU reference at a small shape; a
non-multiple-of-`GRAM_BM` `d` to exercise the tile tails; symmetry exact
(`G[i,j] == G[j,i]` bitwise, since both come from the same accumulator); the
split-K path forced on via a small `d` so both arms are covered; and the boundary
either side of the predicted partials-memory threshold rather than a value in the
middle of the range.

Plus a grid-limit test in the shape of `test_tall_skinny_grid_within_dispatch_limit`
from `a6d2b3b`: assert against `R::max_cube_count()` that the reduce grid covers
`d * d / 256` for `d` well past 4096 and stays inside every dimension. Pure host
arithmetic, no device needed, and it is the test that would have caught the bug
this plan originally shipped.

Accuracy note worth a test rather than a comment: the per-thread accumulator sums
over all `n` in fp32, so at `n = 1e6` the relative error is roughly
`sqrt(n) * eps ~ 6e-5`. That is fine for correlations and matches what the
existing 1e-4 tolerances assume, but assert it at the tall shape rather than
trusting it.

Do not run two benchmarks concurrently (section 3): three parallel runs moved an
identical CPU baseline from 15.7s to 35.3s in an earlier session.

## Explicitly not in scope

- **Refactoring `cholesky_gpu.rs` onto the new kernel.** `A A^T` on feature-major
  and `A^T A` on row-major are the same code with a transposed index, so a
  comptime flag could unify them. That path is measured, working, and feeding the
  SVD; leave it. Follow-up at most.
- **Fusing centre-and-scale into the Gram staging.** It would save one `n * d`
  buffer, but each element is staged `d / GRAM_BM` times (312 at `d = 20000`), so
  the arithmetic gets redone 312x to save a pass worth ~0.4 ms against a GEMM
  worth tens of ms. Not worth it.
- **Vectorised staging.** Global loads are 0.03 ops per FMA at the starting
  config. Wrong lever, as computed above.
- **The Spearman CPU ranking and the upload path.** `rank_matrix_col` is already
  rayon-parallel over columns. Revisit only if step 2's profile puts real time
  there.
