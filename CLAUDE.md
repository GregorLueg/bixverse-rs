# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Crate overview

`bixverse-rs` is a Rust library for computational biology, statistics, and single-cell analysis. It was extracted from the `bixverse` R package's `src/Rust/` directory. Since 0.8.0 it is a cargo workspace of crates under `crates/`, all on one version (`[workspace.package]`), published together to crates.io:

| Crate | Contents | Depends on |
|---|---|---|
| `bixverse-core` | `core`, `utils`, `ml` (sparse structures, linear algebra, stats, SIMD, k-means, GPs) | nothing in the workspace |
| `bixverse-methods` | `enrichment`, `graph`, `methods`, `ontology` | core |
| `bixverse-sc-data` | the single cell data layer, flattened (`data_io`, `v4_io`, `h5ad_io`, `sc_traits`, ...) | core |
| `bixverse-sc` | the rest of `single_cell`, flattened (`sc_processing`, `sc_analysis`, ...) | methods (with `dge`), sc-data |
| `bixverse-sp` | spatial transcriptomics, the old `spatial` module flattened (`sp_graph`, `sp_processing`, `sp_analysis`, `sp_data`, `sp_image`) | sc (always) |
| `bixverse-gpu` | `linalg`, `methods_gpu` | methods |
| `bixverse-sc-gpu` | the old `gpu::sc_gpu`, flattened | sc, gpu |
| `bixverse-mlx` | `linalg`, `methods`, `sc` | sc (always) |
| `bixverse-rs` | facade: re-exports everything under the pre-0.8 module paths | all, optional |
| `bixverse-r` | every `extendr-api` type in the workspace; re-exports `bixverse-rs` | the facade |

The rule that keeps the graph sane: a crate never depends on a sibling through an optional feature, except the two leaves (`bixverse-rs`, `bixverse-r`). Optional features only gate external dependencies (hdf5, edge-rs, bonsai-rs).

`python/` is its own workspace root (excluded from this one) and depends on `bixverse-sc-data` only. Its version is independent (`python/Cargo.toml`, `python/CHANGELOG.md`) and its changelog names the `bixverse-sc-data` version it requires, so bump and note it whenever the data layer changes in a way the wheel sees.

## Sister crates: ann-search-rs, cubecl-utils-rs and edge-rs

The user also maintains three upstream crates, all with local checkouts:

- [`ann-search-rs`](https://crates.io/crates/ann-search-rs) (`~/repos/shared/ann-search-rs`), a vector-search crate for the same computational-biology use cases. `bixverse-rs` reuses its SIMD primitives, distance metrics (`ann_search_rs::utils::dist::Dist`), kNN search, CPU k-means (`build_csr_layout`, etc.) and, since 0.5.2, the whole GPU k-means (`ann_search_rs::gpu::k_means_gpu`: `k_means_clusters_gpu`, `build_csr_gpu_privatised`, `segmented_update`, `KMeansGpuParams`). There is no local GPU k-means any more.
- [`edge-rs`](https://crates.io/crates/edge-rs) (`~/repos/shared/edge-rs`), the negative binomial differential expression layer: the edgeR numerical stack (`filter_by_expr`, `calc_norm_factors`, `estimate_disp`, `glm_fit`, `glm_ql_fit`, `glm_lrt`, `glm_ql_ftest`, `top_tags`), the limma linear model stack (`squeeze_var`, `voom_lmfit`, `lm_fit`, `MArrayLm`, `contrasts_fit`, `make_contrasts`, `ebayes`, `top_table`, `top_table_f`, lowess and locfit), and NEBULA (`nebula`, `nebula_sparse`, `glm_sc_test`, `shrink_sc_dispersion`). Since 0.0.3 the linear model chain is complete through `topTable`, which is what `methods::dge_bulk::run_limma_dge` wires up. Still missing: `treat` and `topTreat`, limma's `decideTests`, and `block` / `correlation` on `voom_lmfit`.
- [`cubecl-utils-rs`](https://crates.io/crates/cubecl-utils-rs) (`~/repos/shared/cubecl-utils-rs`), the GPU primitives layer: `GpuTensor`, `GpuLimits`, `grid_2d`, `checked_cube_count`, `fits_binding`, `fits_shared_memory`, `plane_uniform` / `plane_partitions`, `resolve_workgroup_size`, `LINE_SIZE`, `pad_vectors`, `CubeclFloat`. Import it as `use cubecl_utils_rs::prelude::*;`.

`ann-search-rs` depends on `cubecl-utils-rs` too, so both sides of the diamond must resolve to one copy of `GpuTensor` or nothing typechecks across the boundary.

Everything in `cubecl-utils-rs` except `GpuLimits::from_client` and the `GpuTensor` constructors is a pure function of `&GpuLimits`, so derive limits once per dispatcher (`let limits = GpuLimits::from_client(client);`) and pass them down. Do not cache them on long-lived structs.

When a task looks like it wants a new SIMD kernel, distance metric, kNN structure or k-means variant, check `ann-search-rs` first; for a new tensor, grid, device-limit or workgroup-sizing helper, check `cubecl-utils-rs`; for anything negative binomial, dispersion-shaped or design-matrix-shaped, check `edge-rs`. The code may already exist and just need exposing. Bug fixes to those primitives belong upstream, not here, and so do ergonomics that would otherwise mean re-encoding upstream internals here (`QlFit::as_glm_fit` and `QlFit::ql_summary` went up for exactly that reason).

## Feature flags

Features live on the crate that owns the dependency; the facade and `bixverse-r` forward them under the old names.

- `bixverse-core`: `accelerate` (Apple Accelerate GEMM via `ann-search-rs/accelerate`)
- `bixverse-methods`: `dge` (pulls in `edge-rs`; `methods::dge_bulk`)
- `bixverse-sc-data`: `h5` (the h5ad and 10x h5 readers; pulls in `hdf5`, `ndarray`), `hdf5-static`, `hdf5-filters`
- `bixverse-sc`: `multi-modal`, `bonsai` (pulls in `bonsai-rs` with `sanity`; Sanity is only ever named through `bonsai_rs::sanity_sc_rs`)
- `bixverse-sc-gpu`: `bonsai` (the GPU Sanity to Bonsai path)
- `bixverse-sp`: `h5` (the `obsm/spatial` and `uns/spatial` reader; forwards `bixverse-sc-data/h5`), `hdf5-static`, `spatial-image` (per-spot histology features; pulls in `image` and `openslide-rs`, needs the OpenSlide C library installed, linked dynamically because OpenSlide is LGPL)
- `bixverse-mlx`: none. Apple Silicon only. `ann-search-rs/mlx` owns `links = "mlxc"` and exports `DEP_MLXC_AVAILABLE`; this crate's `build.rs` turns that into `cfg(mlx_available)`, exposed as `MLX_AVAILABLE`. Without `libmlxc` the crate compiles to nothing but that constant. The FFI bindings stay local because `ann_search_rs::mlx::ffi` is private
- Facade `bixverse-rs`: `accelerate` (default), `dge`, `single-cell` (implies `dge`; no HDF5), `h5` (implies `single-cell`, turns on `bixverse-sc-data/h5`), `hdf5-static` and `hdf5-filters` (imply `h5`), `multi-modal`, `bonsai`, `gpu` (bulk GPU only), `sc-gpu` (gpu + single-cell + `bixverse-sc-gpu`), `spatial` (single-cell + `bixverse-sp`, re-exported as `bixverse_rs::spatial`), `spatial-image`, `mlx`. A consumer that never reads h5ad or 10x h5 (bixverse.gpu) leaves `h5` off and has no HDF5 in its tree
- `bixverse-r`: the facade's features plus nothing else. The h5 parameter impls (`H5adFileTask`, `TenxFileTask`) sit behind its `h5`
- `large-test` and `large_scale_diagnostics` exist on every crate that has gated tests (methods, sc, gpu, sc-gpu, mlx, facade); the facade forwards them. `large-test` is for slow tests that assert; `large_scale_diagnostics` for unasserted sweeps that print tables. No CI job enables either

## Common commands

```bash
# What CI runs (rust-workspace-test.yml in personal-actions)
cargo test -p bixverse-core
cargo test -p bixverse-methods --features dge
cargo test -p bixverse-sc-data --features hdf5-static
cargo test -p bixverse-sc --features multi-modal
cargo test -p bixverse-sp
cargo test -p bixverse-sp --features hdf5-static
cargo test -p bixverse-rs --features single-cell,multi-modal               # no HDF5
cargo test -p bixverse-rs --features single-cell,multi-modal,hdf5-static
cargo test -p bixverse-rs --features spatial,hdf5-static
cargo test -p bixverse-r                                                    # needs R
cargo test -p bixverse-r --features single-cell,multi-modal
cargo test -p bixverse-r --features single-cell,multi-modal,hdf5-static
cargo test -p bixverse-r --features spatial,hdf5-static
cargo test -p bixverse-gpu
cargo test -p bixverse-sc-gpu
cargo test -p bixverse-rs --features sc-gpu,hdf5-static

# Every member on its own: catches features that workspace unification hides.
# bixverse-r is left out because extendr's build script needs R.
cargo check -p <member> --no-default-features --all-targets

# Not in CI: the reusable workflow cannot install OpenSlide.
cargo test -p bixverse-sp --features hdf5-static,spatial-image

# The slow asserting gates. Release, or the CPU reference solves dominate.
cargo test --release -p bixverse-rs --features sc-gpu,multi-modal,large-test

# MLX backend (Apple Silicon, needs mlx-c installed)
cargo test -p bixverse-mlx
cargo bench -p bixverse-mlx --bench mlx_corr_bench

# Single test by name (substring match)
cargo test -p bixverse-sc -- test_name_substring

# Format / lint. Gated tests do not compile under the CI sets, so lint with all of them.
cargo fmt --all
cargo clippy --workspace --exclude bixverse-r --all-targets --features bixverse-rs/sc-gpu,bixverse-rs/multi-modal,bixverse-rs/bonsai,bixverse-rs/hdf5-static,bixverse-rs/mlx,bixverse-rs/large-test,bixverse-rs/large_scale_diagnostics
cargo clippy -p bixverse-r --features single-cell,multi-modal,bonsai,sc-gpu --all-targets

# Benches: the facade's live in crates/bixverse-rs/benches, the MLX ones in crates/bixverse-mlx/benches
cargo bench -p bixverse-rs --features gpu --bench gpu_corr_bench
```

`bixverse-r` needs `R_HOME` / R shared libraries on the linker path because `extendr-api` links against libR. No other crate does. On macOS with R installed this generally works out of the box.

## Architecture

Each crate is organised by domain, not by algorithmic layer. Module trees were kept as they were inside `src/`; `single_cell`, `single_cell::sc_data`, `gpu::sc_gpu` and `mlx` became crate roots, so `crate::single_cell::sc_processing` inside `bixverse-sc` is now `crate::sc_processing`. Cross-crate paths are written out (`bixverse_core::core::math::...`); there is no aliasing at crate roots.

- `bixverse-core`: `core/` (math primitives, `faer` linear algebra, `CompressedSparseData2`, PCA/SVD, correlations, RBF kernels, synthetic data), `utils/` (SIMD via `BixverseSimd`, matrix helpers, traits, heap structures, assertion macros), `ml/` (k-means, Gaussian processes)
- `bixverse-methods`: `enrichment/` (fgsea, GSVA, singscore, mitch, OAE, blitzGSEA, decoupler ULM / MLM in `lm_activity`), `graph/` (`SparseGraph`, community detection, PageRank), `methods/` (NMF, ICA, LDA, diffcor, graph diffusion, SNF, RBH, dgRDL, CoReMo, cis-target, `dge_bulk`), `ontology/`
- `bixverse-sc-data`: binary format (`SC_FILE_VERSION`, currently 4, the previous `SC_FILE_VERSION_V3` still reads; see "Single cell binary format" below), `ParallelSparseReader` (checks the magic string, the version and every offset against the mmap length), in-memory reader, h5ad / 10x h5 / MTX ingestion, `sc_traits`, `CELL_BATCH_SIZE`
- `bixverse-sc`: processing, kNN, batch correction, annotation, analysis (Hotspot, MELD, SEACells, MetaCells2, NEBULA, pseudobulk DE), trajectories, multi-modal (WNN)
- `bixverse-sp`: spatial graph builder (hex, square, kNN, radius), Moran's I, SPARK-X, neighbourhood enrichment, h5ad spatial ingest and orientation, histology image features. `GraphCsr`, the symmetric CSR shared with HotSpot, lives in `bixverse-methods` as `graph::spatial_graph`
- `bixverse-gpu`, `bixverse-sc-gpu`: GPU kernels via `cubecl` / `cubek`. K-means comes from `ann-search-rs`
- `bixverse-mlx`: experimental MLX backend via `mlx-c`, f32 only

Where an sc algorithm has a GPU or MLX twin, the seam is generic over the backend's error: `FwArgminB` has `type Error: From<ScErrors> + From<CoreErrors> + Debug`, and the fastMNN drivers take a search closure returning `Result<_, E>`. `ScKnnResults<E>` and `SingleCellPcaRes<E>` default `E` to `ScErrors`. Use the same pattern for a new seam rather than adding a GPU variant to `ScErrors`.

**Errors.** One `thiserror` enum per crate: `CoreErrors`, `MethodsErrors`, `ScDataErrors`, `ScErrors`, `SpErrors`, `GpuErrors`, `ScGpuErrors`, `MlxErrors`, `RErrors`. Each upper enum wraps the ones below with `#[error(transparent)] Core(#[from] CoreErrors)` and so on, so `?` converts upwards; constructing a lower variant in an upper crate needs `.into()` (`Err(CoreErrors::InvalidArgument(..).into())`), and a test matching on it needs the wrapper (`Err(ScErrors::Core(CoreErrors::X { .. }))`). Add a variant to the lowest crate that raises it, in the matching subsystem section. The facade's `errors` module re-exports all of them.

**Single cell binary format.** Two files per store, cell-major (CSR) and gene-major (CSC), one independently compressed chunk per cell or gene, offsets plus an index map in a tail header, `mmap` reads. Version 4 (the writers' default) stores indices and raw counts delta bit-packed in blocks of 128 behind an uncompressed 32 byte chunk header and a per-chunk codec (`ChunkCodec`: lz4 by default, zstd, none), and drops the normalised layer: it is recomputed on read from the raw counts, library size and target size, bit for bit, and a chunk where it does not recompute keeps it verbatim. A gene file carries a per-cell library-size table in its tail (pointed at by `library_index_offset`) for that recompute, and a table of normalised values for counts 1 to 4 is built when it opens. All of it lives in `v4_io`. Rules that follow from it:

- `CellGeneSparseWriter::new` writes v4, `new_v3` and `write_gene_file_v3` the legacy layout (tests of v3 internals, conversion input). Parallel payload builders encode through `writer.format()` (`ChunkFormat::encode_cell` / `encode_gene`); `to_compressed_bytes` still emits v3 payloads, so never hand it to a v4 writer.
- The reader dispatches on the file version, so v3 stores keep opening. `peek_file_version` reads the version from the 64 byte header, `upgrade_store_to_v4` swaps a v3 store for its conversion (temporary files first, optional `.v3.bak`), `convert_*_to_v4` write it elsewhere. The gene file needs the cell file for its library table, so convert them together.
- The norm recompute goes through `norm_value`, the single definition of the layer. Changing it breaks bit identity with every file written before, as before. It uses the platform `ln_1p`; cross-platform identity of the recompute is untested.
- Reads prefetch their byte range into the page cache (unix only, sampled residency probe first, so warm stores pay almost nothing). Cold-read benchmarks must verify coldness with `mincore`; sizing chunks by reading their mapped bytes warms the cache.
- The archive (`archive_io`) only reads cells through the reader API and restores through the regular writers, so it is format independent and restores to v4.

**Preludes.** Each crate has a `prelude` that re-exports the preludes below it plus its own errors and common types. Prefer `use crate::prelude::*;` in new modules. The facade prelude is the union.

## Performance conventions

Performance is a first-class concern. This crate is the fast core underneath an R package.

- Linear algebra: prefer `faer` (`Mat`, `MatRef`, `MatMut`) over `ndarray`. `ndarray` is only pulled in for HDF5 interop under `single-cell`.
- `faer` is a workspace dependency with default features off (`std`, `rayon`, `sparse-linalg`, `rand` on). Its default `npy` feature drags `npyz`, `py_literal`, `pest_derive` and `stacker` into the tree, and `stacker`'s build script fails on the Windows ARM64 r-universe build. Cargo unifies features, so one crate or upstream dependency on the defaults brings it back for everyone: check with `cargo tree -e all --target all | grep -E "npyz|stacker|pest"`, which should print nothing.
- Parallelism: `rayon` for CPU fan-out. `Par::Rayon(n_threads.try_into().unwrap())` is the standard `faer` parallelism knob.
- SIMD: use `BixverseSimd` in `utils::simd` rather than hand-writing `wide` calls.
- Hash maps/sets: use `FxHashMap` / `FxHashSet` from `rustc_hash`, not `std::collections::HashMap`.
- Release profile is already tuned (`opt-level = 3`, `lto = "thin"`, `codegen-units = 4`), so don't override per-crate.

## R interop pattern

There is no `#[extendr]` anywhere in this workspace; the entry points live downstream in the `bixverse` R package. All extendr code lives in `bixverse-r`: the `*_r_wrapper(s).rs` files and `r_rust_interface.rs` (`r_list_to_map`, `r_list_count`, `faer_to_r_matrix`, `NamedNumericVec`, `IntoExtendrErr`, `FaerRType`, ...).

The params types live in other crates, so the orphan rule rules out inherent `from_r_list` impls. Three traits in `bixverse-r/src/lib.rs`, all in its prelude:

- `FromRList::from_r_list(List) -> extendr_api::Result<Self>` for parameter structs
- `FromRMap::from_r_map(&HashMap<&str, Robj>) -> extendr_api::Result<Self>` for sub-configs read out of an already flattened list
- `ToRList::to_r_list(&self) -> List` for fitted structures that round-trip

Constructors that need extra arguments (a seed, isotype indices) are free functions in the wrapper files instead.

The convention inside a `from_r_list`: `r_list_to_map(r_list)?` then `let defaults = Self::default();`, numerics through `.and_then(|v| v.as_real()).unwrap_or(defaults.field)`, and string enums through a free `parse_*(s: &str) -> Option<T>` that errors on an unrecognised value rather than falling back.

## Testing layout

- Unit tests live inline (`#[cfg(test)] mod tests`) in each module file. A unit test that needs a crate above its own moves up: `crates/bixverse-sc/tests/in_memory_cell_subset.rs` exists because those tests use `sc_processing`
- Integration tests and benches that span crates live in the facade, `crates/bixverse-rs/{tests,benches}`, under the old module paths, each gated by a file-level `#![cfg(...)]` on facade features (`single-cell`, `h5`, `dge`, `bonsai`, `sc-gpu`, `large-test`, ...). The MLX ones live in `crates/bixverse-mlx/{tests,benches}`, because `cfg(mlx_available)` only exists in that crate. `r_ffi_bench` lives in `crates/bixverse-r/benches`
- Tests that write a store through `CellGeneSparseWriter::new` now exercise v4 end to end. A test that needs v3 on purpose (the reader's v3 path, conversion input) writes it with `new_v3` and `write_gene_file_v3`, as `v4_io`'s tests do.
- Parity fixtures are generated, never recalled. `dev/gen_edger_fixtures.R` writes `crates/bixverse-rs/tests/edger_fixtures/mod.rs` against edgeR 4.8.2; `dev/gen_blitzgsea_fixtures.py` does the same for `crates/bixverse-rs/tests/blitzgsea_fixtures/`. Both rebuild their input from a seeded LCG on each side so no float data crosses as text. Run the generators from the repository root
- CI matrix: Ubuntu / macOS / Windows for the CPU and R lanes (Windows uses msvc on the CPU lane and gnu on the R lane); Ubuntu / macOS for GPU (Linux uses Vulkan via `WGPU_BACKEND=vulkan`); plus a per-member `cargo check` on Ubuntu

### Expensive tests

Anything that takes more than a second or two goes behind a feature, placed directly after `#[test]` with a one-line comment giving the problem size. Which feature depends on whether it can fail:

- `large-test` if it asserts. These are real gates and are worth running on a schedule.
- `large_scale_diagnostics` if it only prints for a human to read. Running these in CI buys nothing, since they cannot fail.

If you are adding a sweep that computes a difference against a reference, assert on it and use `large-test`. A sweep that prints a number it never checks is not a test.

No CI job enables either yet, so CI runs toy sizes only.

Two consequences to keep in mind:

- Gated tests do not compile under the CI feature sets, so they can bit-rot. Run the all-features clippy line from Common commands whenever you touch them.
- Gating a test can orphan a helper or an import inside a `#[cfg(test)] mod tests` block. Carry the same `cfg` on the helper rather than reaching for `#[allow(dead_code)]` in `src/`.
- Before doing that, check whether the gate is right at all. `run_columns_a_raw` in `crates/bixverse-sc-gpu/src/seacells_gpu.rs` used to carry the gate purely because its only caller was gated, which left the shared-memory reduction arm of `fw_columns_a_gpu` executing nowhere: `plane_reduce_viable` is true on Apple Silicon, so every default run took the plane path. The fix was a cheap ungated sibling at n = 64, k = 32, not a `cfg` on the helper. A gate that hides a whole code path is a coverage hole, not a cost saving.

When gating, leave at least one cheap test covering each structural property. `test_fw_argmin_b_matches_cpu` and `test_fw_columns_a_capacity_boundary` exist for exactly that reason; do not gate them alongside their heavier siblings.

The Linux GPU runner has no real GPU and falls back to lavapipe software Vulkan. It reports a plane width of 8, so it takes the shared-memory reduction arms that Apple Silicon (which reports 32/32) never touches. That coverage is the reason the Ubuntu GPU job is worth keeping, but treat large-data failures there with suspicion: lavapipe is happy to hand back recycled uninitialised buffers.
