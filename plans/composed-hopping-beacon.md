# Gate heavy GPU tests, fix the narrow-plane reduction bug

## Context

GPU CI takes too long and fails on Ubuntu in ways that do not reproduce on Apple
Silicon. On a Mac the whole suite is green in ~78s of test time (lib 37.5s,
`scenic_gpu` 33.8s, `seacells_gpu` 7.2s). On the Ubuntu GPU runner, which falls
back to lavapipe software Vulkan, four `seacells_gpu` tests fail with garbage
atom indices and column masses of ~307 instead of 1.0.

Two separate problems are tangled together here.

**One is a real portability bug, not lavapipe flakiness.**
`reduce_scratch_len` (`src/gpu/sc_gpu/kernels/seacells_kernels.rs:167`) sizes the
plane-reduction scratch `s_val`/`s_idx` as `wg_size / MIN_PLANE_WIDTH` with
`MIN_PLANE_WIDTH = 32`, but the kernel indexes those arrays by the device's
*runtime* `PLANE_DIM` (`:822-838` for argmin, `:990-1002` for renormalisation).
`plane_reduce_viable` (`:1246`) gates the plane arm on `plane_size_min ==
plane_size_max && plane > 0 && wg_size.is_multiple_of(plane)`, and never on
`plane >= 32`. Lavapipe reports plane 8/8, passes all three conjuncts, then
writes and reads 12 slots past the end of a 4-entry array at `wg = 128`. The
reads cross into the neighbouring array, so a gradient value gets used as an
atom index: the reported `3171468488` is `0xbd08c4c8`, which is `-0.0334f32`.
Apple is immune only because cubecl hardcodes plane 32/32 on Apple Silicon
(`cubecl-wgpu-0.10.0/src/runtime.rs:296-307`); everywhere else the value comes
from the adapter. Any Intel iGPU or other narrow-plane backend running SEACells
GPU gets silently wrong answers today.

The sibling helper `plane_compact_viable` (`src/gpu/sc_gpu/scenic_gpu.rs:3804`)
already carries the equivalent guard. This kernel inverted it: it sized the
scratch by an assumed minimum plane *width* instead of bounding the plane
*count*. Both defects landed together in `0d9bdd4`, which replaced a serial loop
that was correct for any `n_planes` with a plane primitive that is not.

**The other is cost.** The expensive tests are expensive because of their CPU
reference solves, not the GPU work, and they run on every push.

Outcome: default `cargo test` stays fast on all three runners, the heavy tests
move behind `large_scale_diagnostics`, and the narrow-plane bug is fixed so the
Ubuntu lane exercises the shared-memory tree arm correctly instead of taking a
broken plane arm.

## Decisions taken

- **Gate**: reuse the existing `large_scale_diagnostics` feature with plain
  `#[cfg(feature = "large_scale_diagnostics")]`. No new feature key.
- **No new workflow.** Heavy tests are run locally, on the Mac.
- **Keep Ubuntu in `test-gpu`.** After gating it runs toy sizes only, and it is
  the only CI runner that exercises the shared-memory tree reduction arm, since
  Apple reports plane 32/32 and always takes the plane arm. It also gives a
  second shader codegen path (naga to SPIR-V rather than MSL).

Accepted trade-off: hard `cfg` means the heavy tests do not compile in CI, so
they can bit-rot and `cargo clippy --all-targets` in CI will not lint them. The
verification section below includes a clippy invocation with the feature on;
run it whenever those tests are touched.

## Part 1: fix the narrow-plane bug

### `src/gpu/sc_gpu/kernels/seacells_kernels.rs`

**`plane_reduce_viable` (`:1246`)** gains a fourth conjunct:

```rust
plane == hw.plane_size_max
    && plane > 0
    && plane >= MIN_PLANE_WIDTH
    && wg_size.is_multiple_of(plane)
```

That one line is a *complete* fix, not a partial one. `A_COLUMNS_WG_TIERS`
(`:144`) caps `wg_size` at 512, and `reduce_scratch_len` has exactly one caller
(`:1373`). Given `plane >= 32`:

- sizing: `n_planes = wg/plane <= 512/32 = 16 <= wg/32 = reduce_scratch_len`,
  with equality only at `plane == 32`;
- the level-2 reduction at `:832-844` and `:999-1004` combines per-plane winners
  with a single `plane_min`/`plane_sum` over plane 0, so it needs
  `n_planes <= PLANE_DIM`; that holds as `16 <= 32`.

No structural refactor. Extend the `reduce_scratch_len` doc comment (`:152-166`)
to state both invariants it now relies on (`plane >= MIN_PLANE_WIDTH`, enforced
by `plane_reduce_viable`; `wg_size <= plane * plane`, which holds by
construction from the tier table), and correct the `MIN_PLANE_WIDTH` doc comment
at `:146-150`, which currently claims "Metal and Vulkan both report 32 or 64".
That is false for software Vulkan and for Intel integrated graphics.

Also update the `plane_reduce_viable` doc comment to say why the width floor
exists, mirroring `plane_compact_viable` in `scenic_gpu.rs:3804`.

### `src/gpu/sc_gpu/seacells_gpu.rs:1050-1070`

`test_fw_columns_a_reduction_arms_agree` asserts `assert_eq!(plane_idx,
tree_idx)` over the entire `n * cap` buffer. Both buffers come from
`GpuTensor::empty` (`:981-983`) and the kernel only writes `[0, cnt[cell])` per
cell, so the assertion compares uninitialised device memory and passes only
because the allocator happens to hand both runs identically recycled pages. This
is the one place in the repo that reads uninitialised device memory, so it is
squarely part of making the Ubuntu lane trustworthy.

Restructure into a single per-cell loop: keep the existing
`assert_eq!(plane_cnt, tree_cnt)`, then for each `cell` slice
`[cell * cap .. cell * cap + cnt]` on both `_idx` and `_val` and compare only
that range. This also fixes the value loop's message, which currently labels a
flat buffer index as `cell` and prints `cell % cap` as the slot.

## Part 2: gate the heavy tests

Attribute goes after `#[test]`, with a one-line comment above it giving the
problem size, matching the existing style at `tests/scenic_gpu.rs:564-566`. Line
numbers are pre-edit anchors.

### Inline in `src/`

| file:line | test | size |
|---|---|---|
| `src/gpu/linalg/gram.rs:565` | `test_gram_aat_split_k_path` | n=40000 d=48, host reference is a 92 MFLOP triple loop |
| `src/gpu/linalg/cholesky_gpu.rs:869` | `test_cholesky_qr2_above_dispatch_limit` | n=600000, three 9.6 MB buffers |
| `src/gpu/linalg/sparse_rand_svd_gpu.rs:505` | `test_randomised_sparse_svd_gpu_accuracy_vs_dense` | 3 power-iteration arms at n=3000, plus a dense faer `thin_svd` |
| `src/gpu/ml/k_means_gpu.rs:2362` | `test_segmented_update_long_segments` | n=20000 k=4 dim=48 |
| `src/gpu/ml/k_means_gpu.rs:2391` | `test_kmeans_parallel_init_gpu_recovers_blobs` | 20 Lloyd iterations over n=2000 |
| `src/gpu/sc_gpu/seacells_gpu.rs:675` | `test_fw_columns_a_gpu_matches_cpu` | n=2500 k=300, six full CPU `fw_columns_a` solves |
| `src/gpu/sc_gpu/seacells_gpu.rs:799` | `test_fw_columns_a_large_k_matches_cpu` | n=1500, k in {3000,4500,9000} across every tier |
| `src/gpu/sc_gpu/seacells_gpu.rs:1022` | `test_fw_columns_a_reduction_arms_agree` | n=1500 k=300, four dispatches |

`run_columns_a_raw` (`src/gpu/sc_gpu/seacells_gpu.rs:950`) is called only by
`test_fw_columns_a_reduction_arms_agree`, so it must carry the same
`#[cfg(feature = "large_scale_diagnostics")]` or it becomes `dead_code` inside
the library. It is the only helper in `src/` that orphans: `gram_host`,
`sinusoidal_tall_skinny`, `dense_to_csc`, `build_dense` and `random_csr` all stay
reachable from surviving tests. Confirm `run_assign` and the segmented-update
helper in `k_means_gpu.rs` likewise stay reachable when the two tests there are
gated.

### `src/gpu/sc_gpu/kernels/seacells_kernels.rs:1825` needs splitting, not gating

`test_fw_argmin_b_matches_cpu` sweeps `[(40,7), (257,33), (1000,130), (3000,300),
(300,1100)]`. Its doc comment says the structural properties it exists to cover
are `slots > 1` (needs `k > wg`) and grid stride (needs `n > B_ARGMIN_BLOCKS =
1024`). Only `(3000,300)` and `(300,1100)` reach those, and since the CPU
reference is O(n·k²) those same two arms are the entire cost. Gating the whole
test would drop both properties from CI and orphan `assert_argmins_agree` and
`cpu_grad_at`.

Lift the loop body into a private `fn check_argmin_arm(n: usize, k: usize,
device: &WgpuDevice)`. Keep `test_fw_argmin_b_matches_cpu` on the three cheap
arms; add a gated `test_fw_argmin_b_matches_cpu_large` with the two heavy arms.
Move the `slots`/grid-stride paragraphs of the doc comment onto the new test and
leave a one-line pointer on the cheap one. No arms invented, no assertions
changed, and every helper stays reachable from the surviving test, so nothing
orphans.

### Integration tests

**`tests/seacells_gpu.rs`** holds exactly one test, and it is the heaviest in the
repo (n=3000, two arms, a full CPU SEACells fit and a full GPU fit each). Gate
the whole file rather than the test, matching the `tests/gpu_corr.rs:2` idiom, so
`try_device` does not become dead code:

```rust
#![cfg(all(
    feature = "single-cell",
    feature = "gpu",
    feature = "large_scale_diagnostics"
))]
```

**`tests/scenic_gpu.rs`**, gate these six:

- `:642` `phase2_multi_batch_determinism` (2000x100x130, 50 trees, four GPU ensemble fits)
- `:887` `phase3_rf_pearson_small` (120 RF trees, two CPU fits plus one GPU)
- `:1018` `phase3_rf_pearson_skewed_bins` (1500x50x64, 120 RF trees)
- `:1365`, `:1419`, `:1523` the three `run_scenic_grn_*_gpu_roundtrip` (full
  pipeline; two of them write a sparse binary fixture to `temp_dir`)

Also at `:525-526`, replace `#[ignore]` on `phase2_cpu_baseline` with
`#[cfg(feature = "large_scale_diagnostics")]`. That test has zero assertions and
ends in an `eprintln!` of the mean Pearson, so it is a diagnostic sitting next to
its three already-gated siblings, and `#[ignore]` was the wrong attribute for it.
This leaves zero `#[ignore]` in the repo and one gating mechanism.

The existing `#![allow(dead_code)]` at `:19` stays; its justification is
unchanged.

### Deliberately left running in CI

State these in the commit message so the choices are visible rather than
looking like oversights.

- `src/gpu/sc_gpu/seacells_gpu.rs:1082` `test_fw_columns_a_capacity_boundary`.
  At n=300, k=300 the CPU reference is cheap, and gating it alongside `:675` and
  `:799` would leave CI with zero CPU-vs-GPU parity coverage for the A-column
  kernel. Its doc comment records a real off-by-one at the 128-wide tier ceiling.
- `tests/scenic_gpu.rs:231` `extra_trees_gpu_matches_cpu_top10` and `:346`
  `rf_gpu_matches_cpu_top10`. Both run on `make_toy_quantised` at 256x32 with 1
  and 16 trees; the fit count looks large but each fit is one 6-deep tree. The
  doc comment at `:350` states `rf_gpu_matches_cpu_top10` is the only RF
  fidelity test in CI, so gating it would strip that gate and falsify the comment
  in the same commit.

After this, `cargo test --features gpu,single-cell` still runs from
`tests/scenic_gpu.rs`: `cpu_baseline_seed_variance`,
`extra_trees_gpu_matches_cpu_top10`, `rf_gpu_matches_cpu_top10`,
`coarse_threshold_roundtrip`, `phase3_et_still_works` and the two
`*_rejects_gbm` error paths. ET parity, RF parity and the rejection paths are
all retained.

## Part 3: CI and docs

### `.github/workflows/test.yml`

The test command stays `cargo test --features gpu,single-cell`; the speed-up
comes entirely from the gating. Three small fixes:

1. Add the `actions/cache@v5` step to `test-gpu`, copied verbatim from the
   `test` job (`:31-41`). `test-gpu` has no cache today, so it rebuilds cubecl
   and cubek from scratch on every run. This is the largest single win in that
   job's wall clock.
2. Add `timeout-minutes: 30` to both `test` and `test-gpu`. There is no timeout
   anywhere, so a hung wgpu device currently burns the 6h job limit.
3. Drop the dead `if: runner.os != 'Windows'` on the `test-gpu` run step
   (`:93`); that matrix is ubuntu and macos only.

### `Cargo.toml`

No new feature. Broaden the `large_scale_diagnostics` doc comment (`:19-20`),
since it now covers both unasserted diagnostics and slow-but-asserting tests,
and it renders into the public docs via `document_features` (`src/lib.rs:7`).

### `CLAUDE.md`

Both existing errors are worth fixing here, because anyone following the file
today runs zero SCENIC and zero SEACells GPU tests.

- `:36` documents GPU tests as `cargo test --features gpu`, which silently
  excludes `tests/scenic_gpu.rs` and `tests/seacells_gpu.rs` entirely (both are
  `#![cfg(all(feature = "single-cell", feature = "gpu"))]`). Correct it to
  `--features gpu,single-cell` and add the heavy invocation below it.
- `:93` lists integration tests as only `gpu_corr.rs`, `meta_cells2.rs` and
  `large_scale_diagnostics.rs`. Add `scenic_gpu.rs` and `seacells_gpu.rs`, and
  fix the `gpu_corr.rs` description, which omits that it also needs
  `large_scale_diagnostics`.
- Add a short subsection under "Testing layout" recording the convention: tests
  that take more than a second or two go behind
  `#[cfg(feature = "large_scale_diagnostics")]`; the feature covers expensive
  tests and unasserted diagnostics alike; no workflow enables it; run it in
  release or the CPU references dominate.
- `:28` currently describes the feature as "development-only, expensive
  diagnostic tests". Widen to match.

### `CHANGELOG.md`

One entry under Fixes for the `plane_reduce_viable` guard, since it changes
runtime arm selection on any device reporting a plane narrower than 32 and fixes
silently wrong SEACells GPU output there. The test gating does not need an entry.

## Verification

```bash
# 1. Everything still compiles and lints, including the gated tests.
cargo fmt --check
cargo clippy --features single-cell,multi-modal --all-targets
cargo clippy --features gpu,single-cell --all-targets
cargo clippy --features gpu,single-cell,large_scale_diagnostics --all-targets

# 2. The gate set is exactly what was intended, without running anything.
#    Expect the heavy names absent here and present in the second listing.
cargo test --features gpu,single-cell -- --list | grep -E 'fw_columns_a|split_k|above_dispatch|accuracy_vs_dense|long_segments|recovers_blobs|argmin_b|phase2|phase3|roundtrip'
cargo test --features gpu,single-cell,large_scale_diagnostics -- --list | grep -cE 'fw_columns_a|argmin_b'

# 3. Zero #[ignore] should remain.
cargo test --features gpu,single-cell -- --list --ignored   # expect 0 tests

# 4. capacity_boundary, the two top10 parity tests and the cheap argmin arms
#    must still be in the default run.
cargo test --features gpu,single-cell -- --list | grep -E 'capacity_boundary|top10|test_fw_argmin_b_matches_cpu$'

# 5. The point of the exercise. Compare against the baseline of
#    lib 37.5s / scenic_gpu 33.8s / seacells_gpu 7.2s.
time cargo test --features gpu,single-cell
time cargo test --no-default-features
time cargo test --features single-cell,multi-modal

# 6. The heavy set still passes when actually run. Release, expect ~15 min plus.
#    This also pulls in tests/gpu_corr.rs and tests/large_scale_diagnostics.rs.
cargo test --release --features gpu,single-cell,multi-modal,large_scale_diagnostics

# 7. The two behavioural fixes, in debug so assertions are live.
cargo test --features gpu,single-cell,large_scale_diagnostics -- test_fw_columns_a_reduction_arms_agree
cargo test --features gpu,single-cell,large_scale_diagnostics -- test_fw_argmin_b
cargo test --features gpu,single-cell -- fw_columns

# 8. Docs render, and no new feature key appeared.
cargo doc --features single-cell,multi-modal --no-deps
```

The `plane_reduce_viable` guard **cannot be exercised locally**: Apple Silicon
reports plane 32/32, so the new conjunct is a no-op there and step 7 only
confirms no regression. The Ubuntu lane in CI is the actual test of the fix, and
the four previously-failing tests are now gated out of it, so the signal is that
the surviving `fw_columns_a` coverage (`test_fw_columns_a_capacity_boundary`,
which takes the same kernel through the tree arm) passes on Ubuntu. If a
narrow-plane box is available, `benches/seacells_gpu_bench.rs:806-818` already
prints `plane {min}..{max}` and is the quickest confirmation.
