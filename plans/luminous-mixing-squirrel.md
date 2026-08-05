# Migrate the `gpu` feature onto `cubecl-utils-rs`

## Context

`ann-search-rs` used to carry the GPU primitives that `bixverse-rs` builds on:
`GpuTensor`, `grid_2d`, the layout helpers. That was a layering accident. An
approximate-nearest-neighbour library was the wrong home for a tensor wrapper,
and it meant three crates each held a partial answer to "what does this device
allow". `bixverse-rs` wrote its own `checked_cube_count`, its own binding-size
guards and three near-identical plane-viability functions because upstream had
none.

Those primitives now live in `cubecl-utils-rs` 0.1.0 (published), and
`ann-search-rs` 0.5.0 (local branch `feat-saving-indices`, unpublished) is built
on top of it. `bixverse-rs` should depend on `cubecl-utils-rs` directly for
anything GPU and keep `ann-search-rs` for what it is actually good for: SIMD
distances, kNN indices, k-means utilities. All 30 non-GPU `ann_search_rs` uses in
`src/` are CPU-only, so `ann-search-rs/gpu` comes off the feature list entirely.

Two device-limit bugs already flagged in the blast-radius audit are fixed as part
of this, because the migration touches exactly those dispatches anyway.

Reference: `~/repos/shared/bixverse-project/refactors/gpu_refactor_blast_radius.md`

## What changes at a call site

| before | after |
|---|---|
| `ann_search_rs::gpu::{tensor::GpuTensor, grid_2d, *}` | `cubecl_utils_rs::prelude::*` |
| `GpuTensor::empty(shape, client)` | same arity, now returns `Result` |
| `GpuTensor::from_slice(d, shape, client)` | same arity, now returns `Result` |
| `grid_2d(total)` | `grid_2d(total, &limits) -> Result` |
| `checked_cube_count::<R>(k, x, y, z)` | `checked_cube_count(k, x, y, z, &limits) -> Result` |
| `client.properties().memory.max_page_size` | `limits.max_binding_bytes` (`u64`) |
| `WORKGROUP_SIZE_X` (upstream, = 32) | `crate::gpu::WORKGROUP_32` |

Unchanged, which is what keeps this an import swap plus `?` rather than a rewrite
of 256 call sites: `into_tensor_arg`, `handle`, `shape`, `len`, `read`,
`reshaped_view`, `vram_bytes`. `LINE_SIZE` and `pad_vectors` exist in
`cubecl-utils-rs` with the same semantics.

Scale: 256 `empty`/`from_slice` sites (239 src, 17 benches), 41 `grid_2d`, 10
`checked_cube_count`, 12 src files plus 3 benches carrying an import.

## Threading `GpuLimits`

`let limits = GpuLimits::from_client(client);` locally at the top of each
dispatcher. Do not cache it on structs.

Every non-test `grid_2d` / `checked_cube_count` / plane-viability site already
has `&ComputeClient<R>` in scope. All 18 `launch_*` wrappers in `scenic_gpu.rs`,
all 8 in `harmony_kernels.rs`, and everything in `spmm.rs`, `k_means_gpu.rs`,
`gram.rs`, `cholesky_gpu.rs`, `corr.rs`, `seacells_kernels.rs`, `harmony_gpu.rs`.
`GpuLimits::from_client` is a field borrow plus seven scalar copies, not a device
query, so deriving it per dispatch is free. Caching it on `WaveState` or
`GpuCompressedSparseData` would mean handing those structs a client they do not
otherwise hold, and creates a second source of truth that can go stale.

The two host-side sizing helpers with no client take `&GpuLimits` as a parameter
instead: `pick_wave_size` (`src/gpu/sc_gpu/scenic_gpu.rs:4896`, drop its
`max_binding_bytes: usize` ninth argument) and `viable_max_active_nodes`
(`:4745`, see the bug fix below).

## Cargo and the worktree path

```toml
ann-search-rs = { version = "0.5.0", path = "../ann-search-rs" }
cubecl-utils-rs = { version = "0.1.0", optional = true }

gpu = ["cubecl", "cubecl-utils-rs", "half", "cubek"]   # ann-search-rs/gpu drops out
```

`../ann-search-rs` is correct relative to the main checkout. It is **not** correct
from this worktree, where it resolves to `.claude/worktrees/ann-search-rs`. Fix it
with a symlink rather than an absolute path in the manifest:

```bash
ln -s ~/repos/shared/ann-search-rs ~/repos/shared/bixverse-rs/.claude/worktrees/ann-search-rs
```

One symlink covers every worktree under that directory and leaves the manifest
publishable. Swap the dependency to a plain `"0.5.0"` once `ann-search-rs` is on
crates.io.

Versions unify: `cubecl` 0.10.0 in all three crates, `cubecl-utils-rs` 0.1.0 from
crates.io on both sides of the diamond. `ann-search-rs` 0.5.0 already wraps
`CubeclUtilsErrors` in its own enum (`src/errors.rs:189`); a direct
`From<CubeclUtilsErrors> for BixverseErrors` is a separate impl and `?` picks the
direct one.

## Errors

In the `// -- gpu --` block of `src/errors.rs` (around `:641`):

- **Add** `CubeclUtils(#[from] cubecl_utils_rs::CubeclUtilsErrors)`, gpu-gated.
- **Delete** `GpuCubeCountExceeded` (`:648-661`). Fully superseded by
  `CubeclUtilsErrors::CubeCountExceeded`, same three fields.
- **Keep** `GpuBindingTooLarge`. The two named-buffer pre-check loops in
  `seacells_kernels.rs:1140-1164` and `:1370-1392` walk 14 and 12 named buffers
  respectively; the crate's `fits_binding` carries no buffer name and would turn
  a precise message into a bare byte count.

`### Errors` doc blocks naming `GpuCubeCountExceeded` need updating at
`corr.rs:325,443`, `gram.rs:374`, `cholesky_gpu.rs:377`,
`seacells_kernels.rs:1103,1314`.

## Collapsing the local helpers

Delete `checked_cube_count` from `src/gpu/mod.rs:59-74` once all 10 call sites are
on the crate version. Keep the `WORKGROUP_32..512` constants there: 32 is a
bixverse dispatch convention, not a device fact.

The three plane-viability functions and the shared-memory guard collapse onto
crate primitives. Each mapping is exact:

| local | replacement |
|---|---|
| `plane_compact_viable` (`scenic_gpu.rs:3804`) | `plane_partitions(wg, &limits).is_some_and(\|p\| p <= MAX_PLANES_PER_CUBE)` |
| `plane_argmax_viable` (`scenic_gpu.rs:3831`) | `plane_uniform(wg, &limits)` |
| `plane_reduce_viable` (`seacells_kernels.rs:1268`) | `plane_partitions(wg, &limits).is_some_and(\|p\| p <= wg / MIN_PLANE_WIDTH)` |
| `fused_rf_viable` (`scenic_gpu.rs:1272`) | `fits_shared_memory("fused_rf", fused_rf_smem_bytes(), &limits).is_ok()` |

The `plane_reduce_viable` form is exact because `plane_partitions` guarantees
`p * plane == wg`, so `p <= wg / MIN_PLANE_WIDTH` is `plane >= MIN_PLANE_WIDTH`
including the ragged case. It is `pub` but has exactly one caller
(`seacells_kernels.rs:1398`) and nothing in `tests/` or `benches/`, so inline it
there and move the `MIN_PLANE_WIDTH` rationale into `reduce_scratch_len`'s doc
comment (`:165-184`), which already half states it.

`corr.rs:483`, `scenic_gpu.rs:5655` and the two `seacells_kernels.rs` loops all
hand-read `max_page_size`; they become `limits.max_binding_bytes`.

## The five infallible constructors

Each returns `Result<_, BixverseErrors>` and gains `?` on its allocations. No
caller anywhere needs a signature change: `src/gpu/gpu_r_wrappers.rs` touches no
`GpuTensor`, and every production caller already returns `Result`.

| function | file:line | allocations | callers |
|---|---|---|---|
| `GpuCompressedSparseData::from_parts` | `linalg/sparse_gpu.rs:76` | 3 | 1 production (`:127`, already `Result`), 6 test, 1 bench |
| `WaveState::allocate` | `sc_gpu/scenic_gpu.rs:4630` | 34, one struct literal | `:5719`, `:5782` |
| `upload_dense_y` | `sc_gpu/scenic_gpu.rs:4991` | 1 | `:5700` |
| `SparseYGpu::upload` | `sc_gpu/scenic_gpu.rs:5026` | 5 | `:5695` |
| `GpuFwArgminB::with_verbosity` (+ `new` `:119`) | `sc_gpu/seacells_gpu.rs:137` | 5 | `:569`, 3 test |

The "25 callers" figure in the blast-radius doc is stale; `from_parts` has 8.
Three of the test callers sit behind `large_scale_diagnostics`
(`seacells_gpu.rs:716,833`), which is the bit-rot trap.

## Bug fix 1: unvalidated grid y in SCENIC

Six dispatches put `n_active_nodes` on a grid axis with no check. All are 3D with
every axis occupied, so `grid_2d` cannot help; route them through
`checked_cube_count` instead.

| site | dispatch | wrapper |
|---|---|---|
| `scenic_gpu.rs:3274` | `(n_active_nodes, 1, wave_size)` | `launch_sample_features` `:3261` |
| `:3459` | `(k_feats, n_active_nodes, wave_size)` | `launch_rf_fused` `:3419` |
| `:3626` | same | `launch_build_hist` `:3604` |
| `:3769` | same | `launch_prefix_sum` `:3751` |
| `:4125` | same | `launch_scan_slot_bin_range` `:4108` |
| `:4181` | `(k_feats * n_thresholds, n_active_nodes, wave_size)` | `launch_accumulate_split_stats_et` `:4155` |

`viable_max_active_nodes` (`:4745`) caps at `1 << max_depth.min(20)` = 1_048_576,
sixteen times the wgpu per-dimension limit. Error rather than silently capping
against `limits.max_cube_count.1`: truncating a deep tree without saying so is
worse than a typed failure. Reachable at `n_samples > 131_070` with
`max_depth >= 17`, since `leaf_cap` usually binds first. Real at a million cells.

Fix the stale doc claim at `:277` ("`CUBE_POS_Y` -> node index (max 65535, so
`grid_2d` is not needed here)") and grep `max 65535` for the same sentence on the
other five kernels.

## Bug fix 2: `build_csr_gpu_privatised` is infallible

`src/gpu/ml/k_means_gpu.rs:1148` dispatches five unchecked `CubeCount::Static`
(`:1164`, `:1174`, `:1184`, `:1194`, `:1204`) and cannot report a breach. `:1194`
is the one that actually busts: `cube_count * k / 256` grows in both terms.

Make it `Result<(), BixverseErrors>` with `checked_cube_count` on the four
non-trivial dispatches. The chain above it (`lloyd_step` `:1504`) has to change
for `grid_2d` anyway; `run_kmeans_loop` `:1577` and `harmony_v2_gpu`
(`harmony_gpu.rs:552`) are already `Result`. Two test callers, one of them
`large_scale_diagnostics`-gated (`k_means_gpu.rs:2121`).

## Execution order

There is no green intermediate state that splits the type swap. `GpuTensor` is a
type, and the moment one file imports it from `cubecl_utils_rs` while another
still imports it from `ann_search_rs`, they are distinct types and everything
downstream is a type error. Sequence to keep the error count readable, not to
stay compiling.

**Phase A, green after each step.**

1. `Cargo.toml` as above. Dropping `ann-search-rs/gpu` makes the compiler
   enumerate every remaining `ann_search_rs::gpu` reference, which is the
   migration checklist.
2. `src/errors.rs`: add the `#[from]` variant. Keep `GpuCubeCountExceeded` for
   now, `src/gpu/mod.rs` still builds it.
3. `WORKGROUP_SIZE_X` -> `WORKGROUP_32` (both are 32, so this stays green). 8 src
   sites in `k_means_gpu.rs` and `harmony_kernels.rs`, 6 in
   `benches/gpu_k_means_bench.rs`. Host dispatch geometry and the `#[cube]` body
   reading it must move together: `harmony_kernels.rs:269`/`:779`,
   `k_means_gpu.rs:958`/`:1165`, `:1113`/`:1205`, `:1635`/`:1669`.

**Phase B, the red window, leaf-first.**

4. `linalg/sparse_gpu.rs` (145 lines, defines the shared sparse type)
5. `linalg/spmm.rs` (`launch_dense_column_sum` `:574` and
   `launch_dense_column_weighted_sum` `:601` go `()` -> `Result`)
6. `linalg/gram.rs`, `linalg/corr.rs`, `linalg/cholesky_gpu.rs` (independent
   leaves, all already `Result`, mechanical)
7. `linalg/sparse_rand_svd_gpu.rs`
8. `ml/k_means_gpu.rs` (five functions go `()` -> `Result`, plus bug fix 2)
9. `sc_gpu/kernels/harmony_kernels.rs` (7 of 8 `launch_*` go `()` -> `Result`;
   `launch_objective_partials` `:792` has no `grid_2d`, leave it)
10. `sc_gpu/harmony_gpu.rs`
11. `sc_gpu/kernels/seacells_kernels.rs` (keep the named-buffer loops verbatim)
12. `sc_gpu/seacells_gpu.rs`
13. `sc_gpu/scenic_gpu.rs`, the 6585-line one. 16 of 18 wrappers go `Result`
    (`launch_compute_child_ids` `:4073` and `launch_init_root_stats` `:4287`
    dispatch `(wave_size, 1, 1)` and stay infallible), plus the four constructors,
    `pick_wave_size`, the three viability functions, bug fix 1, and ~21 `?` in
    `run_wave_bfs` (`:5115-5494`)
14. `src/gpu/mod.rs`: delete the local `checked_cube_count`
15. `src/errors.rs`: delete `GpuCubeCountExceeded`

**Phase C, benches.** `gpu_k_means_bench.rs` (imports `:37-38`, 4 `grid_2d`),
`gpu_corr_bench.rs` (`:38`), `seacells_gpu_bench.rs` (`:31`),
`gpu_scenic_micro.rs` (`pick_wave_size` signature at `:443`).
`gpu_scenic_bench.rs` and `gpu_sparse_svd_bench.rs` need nothing.

**Phase D, docs.** `CLAUDE.md:14` and `:25` describe the GPU feature as built on
`ann-search-rs` primitives. `docs/scenic_gpu.md` if it names any of this.

## Verification

```bash
cargo check   --features gpu
cargo check   --features gpu,single-cell
cargo clippy  --features gpu,single-cell --all-targets
cargo clippy  --features gpu,single-cell,large_scale_diagnostics --all-targets
cargo test    --features gpu,single-cell
cargo test    --release --features gpu,single-cell,multi-modal,large_scale_diagnostics
cargo bench   --features gpu --bench gpu_k_means_bench --no-run
cargo bench   --features gpu,single-cell --bench gpu_scenic_micro --no-run
cargo bench   --features gpu,single-cell --bench seacells_gpu_bench --no-run
cargo test    --no-default-features
cargo test    --features single-cell,multi-modal
cargo doc     --features gpu,single-cell,multi-modal --no-deps
```

The `large_scale_diagnostics` clippy line is not optional: six of the call sites
that must change sit inside gated tests (`seacells_gpu.rs:716,833`,
`k_means_gpu.rs:2121`, `cholesky_gpu.rs:860`, plus sites in `gram.rs` and
`seacells_kernels.rs`). No CI job enables that feature. `--all-targets` is what
reaches the benches; `--no-run` compiles them without paying the runtime.

The last two lines matter because dropping `ann-search-rs/gpu` changes feature
unification for the CPU-only builds.

### What can break silently

1. **`grid_2d`'s bound is no longer a hardcoded 65535.** It now comes from
   `client.properties().hardware.max_cube_count` rather than a constant, so on a
   device reporting more (CUDA) the same `total` yields a different `(x, y)`
   split. The x-fast row-major packing is unchanged and is a documented contract,
   so kernels decoding `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` are safe. Grep
   `65535` for host code assuming the old value. Apple Silicon reports 65535, so
   the dev machine will not surface a difference.
2. **`checked_cube_count` gets more permissive** for the same reason. Two tests
   assert a bust errors: `cholesky_gpu.rs:871
   test_cholesky_qr2_above_dispatch_limit` and `gram.rs:612
   test_gram_reduce_grid_within_dispatch_limit`. Read them, they are wgpu-only.
3. **The constructors now pre-check `fits_binding`.** An allocation that
   previously went through and silently produced zeros now returns
   `BindingTooLarge`. That is the point, but a shape that appeared to work will
   start erroring. Candidates: `gram_aat`'s `vec![n_chunks, d, d]` partials
   (`gram.rs:393`) and `WaveState::allocate`'s histograms (`:4682-4684`), where
   `hist_sums_len = hist_counts_len * n_targets` reaches GiB. Run the SCENIC
   benches at their large shapes deliberately.
4. **`grid_2d(0)` no longer panics**, it treats 0 as 1. Every bixverse call site
   already guards with `.max(1)`, so those guards are redundant now. Leave them.
