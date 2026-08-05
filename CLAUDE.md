# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Crate overview

`bixverse-rs` is a Rust library for computational biology, statistics, and single-cell analysis. It was extracted from the `bixverse` R package's `src/Rust/` directory and refactored into a standalone crate. It is published on crates.io and consumed both as a pure Rust library and (via `extendr-api`) from R.

## Sister crate: ann-search-rs

The user also maintains [`ann-search-rs`](https://crates.io/crates/ann-search-rs) (local checkout: `~/repos/shared/ann-search-rs`), a vector-search crate built for the same computational-biology use cases. `bixverse-rs` depends on it directly and reuses:

- **CPU side:** SIMD primitives, distance metrics (`ann_search_rs::utils::dist::Dist`), kNN search, and k-means clustering (`build_csr_layout`, etc.)
- **GPU side:** the `gpu` feature here is built on top of GPU primitives from `ann-search-rs`: tensors, 2D grid helpers, work-group conventions, and related dispatch scaffolding. Changes to those primitives originate upstream in `ann-search-rs`, not in a local fork here.

When a task looks like it wants a new SIMD kernel, distance metric, kNN structure, k-means variant, or GPU primitive, check `ann-search-rs` first. The code may already exist there and just need to be exposed. Bug fixes to those primitives usually belong upstream in `ann-search-rs`, not here.

## Feature flags

Feature flags gate large chunks of the crate. Match your `cargo` invocations to what you are touching:

- default (no features): pure Rust bulk / statistics / graph / enrichment code
- `single-cell`: enables the `single_cell` module and pulls in `hdf5`, `ndarray`, `memmap2`, `lz4_flex`, `bincode`, `indexmap`, `half`
- `multi-modal`: enables `single_cell::multi_modal` (implies `single-cell`)
- `gpu`: enables the `gpu` module, `cubecl` (wgpu + cpu backends), `cubek`, `half`, and `ann-search-rs/gpu`
- `large_scale_diagnostics`: development-only. Gates the expensive tests and the unasserted diagnostic sweeps. No CI job enables it

## Common commands

```bash
# Match CI: two independent test passes
cargo test --no-default-features
cargo test --features single-cell,multi-modal

# GPU tests (separate CI job). single-cell is required: tests/scenic_gpu.rs and
# tests/seacells_gpu.rs are both cfg'd on single-cell + gpu, so `--features gpu`
# alone silently skips them entirely.
cargo test --features gpu,single-cell

# The expensive tests. Release, or the CPU reference solves dominate.
cargo test --release --features gpu,single-cell,multi-modal,large_scale_diagnostics

# Run a single test by name (substring match)
cargo test --features single-cell,multi-modal -- test_name_substring

# Format / lint
cargo fmt
cargo clippy --features single-cell,multi-modal --all-targets

# Benches (GPU k-means bench requires the gpu feature)
cargo bench --features gpu --bench gpu_k_means_bench

# Docs: docs.rs builds with single-cell + multi-modal
cargo doc --features single-cell,multi-modal --open
```

Linux and Windows CI need `R_HOME` / R shared libraries on the linker path because `extendr-api` links against libR. On macOS with R installed this generally works out of the box.

## Architecture

The crate is organised by domain, not by algorithmic layer. Each top-level module contains its methods plus a sibling `*_r_wrapper.rs` file that exposes R-callable entry points via `extendr_api`. Keep the pure Rust surface free of R types: do all R conversions in the wrapper file.

Top-level modules:

- `core/`: shared math primitives, linear algebra (`faer`), sparse structures (`CompressedSparseData2`, `CompressedSparseFormat`, `SparseAxis`), PCA/SVD, correlations, RBF kernels, synthetic data
- `enrichment/`: GSEA (fgsea multi-level), GSVA, singscore, mitch, over-representation (OAE)
- `graph/`: `SparseGraph` structure, community detection, label propagation, PageRank, graph metrics
- `methods/`: bulk-omics methods, NMF (dense + HALS sparse), ICA, differential correlation, graph diffusion, SNF, RBH, dgRDL, CoReMo, cis-target
- `ml/clustering/`: general-purpose clustering
- `ontology/`: GO Elim algorithm and semantic similarity
- `utils/`: SIMD wrappers (`wide` via `BixverseSimd`), matrix helpers, traits, R↔Rust conversion (`r_rust_interface.rs`), heap structures, assertion macros
- `single_cell/` (feature): sc/mc data I/O (h5ad, 10x h5, mtx, bixverse binary format), processing, kNN, batch correction (Harmony), annotation (scType), analysis (Hotspot, MELD, SEACells, MetaCells2), multi-modal (WNN)
- `gpu/` (feature): GPU kernels via `cubecl`/`cubek`, sparse randomised SVD, sparse GEMM, correlation, Cholesky, Harmony, PCA, k-means

`prelude.rs` re-exports the most-used types (errors, sparse structures, `SparseGraph`, matrix/vector utils, SIMD trait, assertion macros). Prefer `use crate::prelude::*;` in new modules over deep imports.

`errors.rs` defines `BixverseErrors` (a single `thiserror` enum for the whole crate). Variants are grouped by subsystem, so add new variants in the matching section rather than at the bottom. Many variants are gated by `#[cfg(feature = "single-cell")]` / `"multi-modal"` / `"gpu"`. Match the gating of the code that produces them.

The single-cell binary sparse format is versioned via `SC_FILE_VERSION` (currently 3) and `MC_SPARSE_VERSION` (currently 1) in `prelude.rs`. Bumping either invalidates existing files on disk.

## Performance conventions

Performance is a first-class concern. This crate is the fast core underneath an R package.

- Linear algebra: prefer `faer` (`Mat`, `MatRef`, `MatMut`) over `ndarray`. `ndarray` is only pulled in for HDF5 interop under `single-cell`.
- Parallelism: `rayon` for CPU fan-out. `Par::Rayon(n_threads.try_into().unwrap())` is the standard `faer` parallelism knob.
- SIMD: use `BixverseSimd` in `utils::simd` rather than hand-writing `wide` calls.
- Hash maps/sets: use `FxHashMap` / `FxHashSet` from `rustc_hash`, not `std::collections::HashMap`.
- Release profile is already tuned (`opt-level = 3`, `lto = "thin"`, `codegen-units = 4`), so don't override per-crate.

## R interop pattern

R-facing functions live in `*_r_wrapper.rs` files and use `extendr_api`. The convention: the wrapper deserialises R types into Rust-native inputs, calls the pure Rust implementation, then serialises the result back. `utils/r_rust_interface.rs` has the shared helpers (`r_list_to_hashmap`, `faer_to_r_matrix`, `NamedNumericVec`, etc). `NamedVecError` implements `From` for `extendr_api::Error` so `?` works across the boundary.

## Testing layout

- Unit tests live inline (`#[cfg(test)] mod tests`) in each module file
- Integration tests in `tests/`, each gated by a file-level `#![cfg(...)]`: `meta_cells2.rs` (single-cell), `scenic_gpu.rs` (single-cell + gpu), `seacells_gpu.rs` (single-cell + gpu + large_scale_diagnostics), `gpu_corr.rs` (gpu + large_scale_diagnostics), `large_scale_diagnostics.rs` (single-cell + large_scale_diagnostics)
- CI matrix: Ubuntu / macOS / Windows for CPU tests; Ubuntu / macOS for GPU tests (Linux uses Vulkan via `WGPU_BACKEND=vulkan`)

### Expensive tests

Anything that takes more than a second or two goes behind `#[cfg(feature = "large_scale_diagnostics")]`, placed directly after `#[test]` with a one-line comment giving the problem size. The feature covers both the slow GPU parity tests and the unasserted diagnostic sweeps. No CI job enables it, so CI runs toy sizes only.

Two consequences to keep in mind:

- Gated tests do not compile under the CI feature sets, so they can bit-rot. Run `cargo clippy --features gpu,single-cell,large_scale_diagnostics --all-targets` whenever you touch them.
- Gating a test can orphan a helper or an import inside a `#[cfg(test)] mod tests` block. Carry the same `cfg` on the helper rather than reaching for `#[allow(dead_code)]` in `src/`. `run_columns_a_raw` in `gpu/sc_gpu/seacells_gpu.rs` is the worked example.

When gating, leave at least one cheap test covering each structural property. `test_fw_argmin_b_matches_cpu` and `test_fw_columns_a_capacity_boundary` exist for exactly that reason; do not gate them alongside their heavier siblings.

The Linux GPU runner has no real GPU and falls back to lavapipe software Vulkan. It reports a plane width of 8, so it takes the shared-memory reduction arms that Apple Silicon (which reports 32/32) never touches. That coverage is the reason the Ubuntu GPU job is worth keeping, but treat large-data failures there with suspicion: lavapipe is happy to hand back recycled uninitialised buffers.
