[![CI](https://github.com/GregorLueg/bixverse-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/bixverse-rs/actions/workflows/test.yml)
[![Crates.io Version](https://img.shields.io/crates/v/bixverse-rs.svg)](https://crates.io/crates/bixverse-rs)
[![docs.rs](https://img.shields.io/docsrs/bixverse-rs)](https://docs.rs/bixverse-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# bixverse-rs

## Description

A Rust crate for computational biology: statistics, gene set enrichment, graph
algorithms, matrix factorisation, ontologies and a full single cell suite. It is
the numerical core underneath the
[bixverse](https://github.com/GregorLueg/bixverse) R package, but it has no R
dependency of its own and can be used as a plain Rust library.

The code originally lived in `src/Rust/` of that R package. Pulling it out into
its own crate means the numerics get tested, benchmarked and versioned
independently of the wrapper that happens to call them. Design constraint
throughout: atlas-scale analysis on a laptop, not on a cluster.

## Crates

Since 0.8.0 this is a workspace of smaller crates, so a consumer pulls in the
layer it needs and not the dependency graph of everything else. They share one
version and are released together.

| Crate | What it holds | Builds on |
| --- | --- | --- |
| `bixverse-core` | `core`, `utils`, `ml`: sparse structures, linear algebra, statistics, SIMD, k-means, Gaussian processes | |
| `bixverse-methods` | `enrichment`, `graph`, `methods`, `ontology` | core |
| `bixverse-sc-data` | The single cell data layer: the binary store, its readers and writers, MTX, and h5ad / 10x h5 behind `h5` | core |
| `bixverse-sc` | The rest of the single cell suite | methods, sc-data |
| `bixverse-gpu` | GPU linear algebra and bulk methods via `cubecl` | methods |
| `bixverse-sc-gpu` | GPU single cell methods | sc, gpu |
| `bixverse-mlx` | Experimental MLX backend, Apple Silicon only | sc |
| `bixverse-rs` | Facade: everything above under the module paths the crate always had | all of them, optional |
| `bixverse-r` | The extendr conversions; re-exports `bixverse-rs` | the facade |

Only `bixverse-r` links libR, and only `bixverse-sc-data` with `h5` pulls in
HDF5.

## What's in it

Well over a hundred named methods, plus the shared numerical scaffolding they sit on.

| Domain | Methods |
| --- | --- |
| `core` | Correlations and pairwise similarity, PCA/SVD, randomised SVD, LOESS, RBF kernels, mutual information and entropy, scalar optimisation (R's `optimize`), linear mixed models (REML with Satterthwaite), E-distance and perturbation distances, sparse structures, synthetic data generators |
| `enrichment` | GSEA (fgsea multi-level), blitzGSEA (gamma-approximated null), GSVA, ssGSEA, singscore, mitch, over-representation |
| `graph` | Leiden, Louvain, WalkTrap, spectral clustering, label propagation, personalised and constrained PageRank, connected components, Dijkstra (single and multi-source), Kruskal spanning forests, PAGA-style graph abstraction, graph metrics |
| `methods` | NMF (bulk, dense HALS, sparse HALS, consensus, refit), ICA, LDA via variational Bayes, sparse multiple CCA, differential correlation, graph diffusion, SNF, RBH, CoReMo, dgRDL, cis-target; bulk DGE via edgeR's quasi-likelihood chain and limma-voom (feature `dge`) |
| `ml` | k-means, clustering metrics, landmark Gaussian process regression, Matern kernels |
| `ontology` | GO Elim, Wang and Resnik-style semantic similarity |
| `single_cell` | I/O for h5ad, 10x h5, mtx, R/Seurat counts and a versioned binary sparse format (single and multi-file, mmap-backed, mergeable); QC, CellSweep ambient removal, HVG, scTransform v2 (per-sample fits, covariate regression, analytic Pearson residuals, corrected counts), PCA, kNN, SNN, MAGIC; doublet detection (Scrublet, scDblFinder, cxds); batch correction (Harmony v1 and v2, BBKNN, fastMNN, Seurat CCA/rPCA anchors); integration metrics (kBET, iLISI/cLISI, silhouette, PC regression); annotation (scType, Symphony); DGE (Mann-Whitney with AUROC, pseudobulk edgeR, NEBULA); analysis (SCENIC with regulon binarisation, AUCell, Hotspot, VISION, DIALOGUE, MELD, miloR, NicheNet, NMF, module scoring, fast k-means/Louvain clustering); meta cells (SEACells, MetaCells2, SuperCell, hdWGCNA, plus density and compactness metrics); trajectories (Palantir, PAGA, diffusion maps, Markov chains, gene trends); Bonsai trees from Sanity posteriors (feature `bonsai`); multi-modal (WNN, DSB) |
| `gpu` | Sparse randomised SVD, SpMM and sparse GEMM, skinny GEMM, Gram, CholeskyQR2, correlation, NMF and consensus NMF; with `sc-gpu` also PCA, kNN, Harmony, BBKNN, SCENIC, Scrublet, SEACells, NEBULA, Sanity for Bonsai, fast clustering |

Heavy lifting goes through [`faer`](https://github.com/sarah-quinones/faer-rs) for
dense linear algebra (GEMM through Apple Accelerate on macOS), `rayon` for CPU
fan-out, `wide` for SIMD, and
[`cubecl`](https://github.com/tracel-ai/cubecl) for GPU kernels. Vector search,
distance metrics and k-means come from the sister crate
[`ann-search-rs`](https://crates.io/crates/ann-search-rs); the GPU primitives
layer from [`cubecl-utils-rs`](https://crates.io/crates/cubecl-utils-rs); the
edgeR, limma and NEBULA numerics from [`edge-rs`](https://crates.io/crates/edge-rs).

## Feature flags

The flags below are the facade's (`bixverse-rs`); `bixverse-r` has the same
ones. Each forwards to the crate that owns the dependency, so a consumer of a
single crate sets that crate's flag instead (`h5` on `bixverse-sc-data`,
`dge` on `bixverse-methods`, and so on).

| Flag | What it turns on |
| --- | --- |
| *(always on)* | Bulk statistics, enrichment, graph, ontology, matrix factorisation |
| `accelerate` | On by default. Dense CPU GEMM through Apple Accelerate on macOS, no effect elsewhere. See below |
| `dge` | Negative binomial DGE via `edge-rs`: bulk edgeR and limma-voom. Implied by `single-cell`, usable on its own |
| `single-cell` | The `single_cell` module, mmap I/O and the binary sparse format. No HDF5 |
| `h5` | The h5ad and 10x h5 readers and writers. Pulls in HDF5, implies `single-cell` |
| `hdf5-static` | Builds the bundled HDF5 from source instead of linking an external `libhdf5`, implies `h5` |
| `hdf5-filters` | Compiles the LZF and Blosc filters into HDF5, so h5ad files written with those codecs load, implies `h5` |
| `multi-modal` | `single_cell::multi_modal` (WNN, DSB), implies `single-cell` |
| `bonsai` | Bonsai trees over Sanity posteriors via `bonsai-rs`, implies `single-cell` |
| `gpu` | The `gpu` module via `cubecl` (wgpu and CPU backends) and `cubek`: the bulk GPU methods |
| `sc-gpu` | `gpu::sc_gpu`, the GPU single cell methods, implies `gpu` and `single-cell` |
| `mlx` | Experimental, Apple Silicon only: an MLX backend via [`mlx-c`](https://github.com/ml-explore/mlx-c) (f32) for correlation, SpMM, randomised SVD, NMF (dense and sparse, consensus, k sweeps), single cell PCA, Harmony and SEACells. Independent of `gpu`, implies `single-cell`. Needs `brew install mlx-c` or `MLX_C_PREFIX`; without the library the build warns and skips it |
| `large-test` | Slow but asserting tests: GPU parity gates, large-scale numerical checks |
| `large_scale_diagnostics` | Unasserted diagnostic sweeps that print tables for a human |

### macOS and Apple Accelerate

On macOS the default `accelerate` feature sends every dense CPU GEMM
(correlations, covariance, perturbation distances, NMF HALS, Hotspot, Markov
chains, the landmark GP) through Apple's Accelerate framework. It ships with
every macOS, so there is nothing to install. Turn it off with
`default-features = false`. End to end on an M1 Max:

| function | f32 | f64 |
| --- | --- | --- |
| `column_pairwise_cov`, 50k x 2k | 2.3x | 1.8x |
| `column_pairwise_cor`, 50k x 2k | 1.9x | 1.7x |
| dense NMF HALS, 20k x 2k, k = 10 | 1.3x | 1.6x |
| dense NMF HALS, 20k x 2k, k = 50 | 1.8x | 1.8x |

## Using it from Rust

Everything through the facade:

```toml
[dependencies]
bixverse-rs = { version = "0.8", features = ["single-cell", "h5"] }
```

```rust
use bixverse_rs::prelude::*;
```

Or only the layer you need, e.g. the binary store without HDF5:

```toml
[dependencies]
bixverse-sc-data = "0.8"
```

Every crate has a prelude that re-exports the ones below it: the errors, sparse
structures, `SparseGraph`, matrix and vector helpers, the SIMD trait and the
assertion macros. Each crate has its own error enum (`CoreErrors`,
`MethodsErrors`, `ScDataErrors`, `ScErrors`, ...), and each wraps the ones
below it, so `?` converts upwards.

## How it plugs into R

The R side is a thin wrapper, and deliberately so. `bixverse` depends on this
crate through [extendr](https://extendr.github.io/), and does the things R is
good at: argument checking, S7 classes, documentation, plotting handoff. The
Rust side owns the maths.

The R conversions live in their own crate, `bixverse-r`, which re-exports the
facade, so an R package needs that one dependency. It holds the
`*_r_wrapper.rs` files and three traits: `FromRList` deserialises a parameter
struct from an R list, `FromRMap` does the same from an already flattened one,
and `ToRList` serialises fitted structures back. The `#[extendr]` entry points
themselves live downstream in the R packages, so R types never touch the
numerical surface, and no crate other than `bixverse-r` links libR.

R packages that consume it:

- [bixverse](https://github.com/GregorLueg/bixverse) with `single-cell`, `h5`
  and `multi-modal`. The main package.
- [bixverse.gpu](https://github.com/GregorLueg/bixverse.gpu) with `sc-gpu`,
  adding wgpu-backed versions of the expensive methods. It does not need
  HDF5.
- [manifoldsR](https://github.com/GregorLueg/manifoldsR), on the core layer
  only.

Anything on the Rust side that never crosses into R works standalone, so the
crate is equally usable from a pure Rust binary or behind a PyO3 layer.

## Using it from Python

The [`bixverse`](https://github.com/GregorLueg/bixverse-rs/tree/main/python)
Python package wraps the single cell streaming engine with PyO3 and maturin.
It depends on `bixverse-sc-data` alone, so the wheel needs no R install. It opens a
bixverse binary store and streams shuffled minibatches out of it, with the
decode in Rust and the GIL released, fast enough to keep a GPU training loop
fed. First consumer is scVI in `bixverse.nn`, on torch or MLX, trained straight
off the store.

```bash
uv pip install bixverse            # numpy only
uv pip install "bixverse[torch]"   # scVI on torch
uv pip install "bixverse[mlx]"     # scVI on MLX, Apple Silicon
```

Docs live [here](https://gregorlueg.github.io/bixverse-rs/). The Python package
is versioned on its own line, see its
[changelog](https://github.com/GregorLueg/bixverse-rs/blob/main/python/CHANGELOG.md).

## Roadmap

### Methods

- [x] NMF for dense and sparse data
- [x] NicheNet for single cell, see
  [Browaeys et al.](https://www.nature.com/articles/s41592-019-0667-5).
- [x] Palantir for single-cell trajectories, see
  [Setty, et al.](https://www.nature.com/articles/s41587-019-0068-4).
- [ ] Slingshot for single-cell trajectories, see
  [Street, et al.](https://link.springer.com/article/10.1186/s12864-018-4772-0).

### Differential expression

- [x] Bulk edgeR quasi-likelihood and limma-voom via `edge-rs`
- [x] Pseudobulk and NEBULA for single cell
- [ ] limma `treat`/`topTreat`, `decideTests`, and `block`/`correlation` in voom

### GPU accelerations

- [x] GPU-accelerated sparse, randomised SVD
- [x] GPU-accelerated Harmony
- [x] GPU-accelerated correlations (Metal does not have the absolute greatest
  performance here unfortunately...)
- [x] SEACells with GPU acceleration.
- [x] GPU-accelerated BBKNN version
- [x] GPU-accelerated NEBULA

### Python interface

- [x] Data loader to pull the counts out of the binary files and expose them to
  Python-based deep learning frameworks (PyTorch, MLX).
- [x] scVI trained straight off the binary files, on torch and MLX.

## Updates

Updates on what's happening in this crate can be found
[here](https://github.com/GregorLueg/bixverse-rs/blob/main/CHANGELOG.md)

## Licence

MIT License

Copyright (c) 2026 Gregor Alexander Lueg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
