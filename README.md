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

## What's in it

Roughly 110 named methods, plus the shared numerical scaffolding they sit on.

| Domain | Methods |
| --- | --- |
| `core` | Correlations and pairwise similarity, PCA/SVD, randomised SVD, LOESS, RBF kernels, linear mixed models (REML with Satterthwaite), E-distance and perturbation distances, sparse structures, synthetic data generators |
| `enrichment` | GSEA (fgsea multi-level), GSVA, ssGSEA, singscore, mitch, over-representation |
| `graph` | Louvain, WalkTrap, spectral clustering, label propagation, PageRank, connected components, Dijkstra, Kruskal spanning forests, PAGA-style graph abstraction, graph metrics |
| `methods` | NMF (bulk, dense HALS, sparse HALS, consensus, refit), ICA, LDA via variational Bayes, sparse multiple CCA, differential correlation, graph diffusion, SNF, RBH, CoReMo, dgRDL, cis-target |
| `ml` | k-means, clustering metrics, landmark Gaussian process regression, Matern kernels |
| `ontology` | GO Elim, Wang and Resnik-style semantic similarity |
| `single_cell` | I/O for h5ad, 10x h5, mtx and a versioned binary sparse format (single and multi-file, mmap-backed); QC, HVG, PCA, kNN, SNN, MAGIC; doublet detection (Scrublet, scDblFinder, cxds); batch correction (Harmony, BBKNN, fastMNN, Seurat CCA/rPCA anchors); annotation (scType, Symphony); analysis (SCENIC, AUCell, Hotspot, VISION, DIALOGUE, MELD, miloR, NicheNet, module scoring); meta cells (SEACells, MetaCells2, SuperCell, hdWGCNA); trajectories (Palantir, PAGA, diffusion maps, Markov chains, gene trends); multi-modal (WNN, DSB) |
| `gpu` | Sparse randomised SVD, SpMM and sparse GEMM, skinny GEMM, Gram, CholeskyQR2, correlation, PCA, kNN, NMF and consensus NMF, Harmony, SCENIC, Scrublet, SEACells, fast clustering |

Heavy lifting goes through [`faer`](https://github.com/sarah-quinones/faer-rs) for
dense linear algebra, `rayon` for CPU fan-out, `wide` for SIMD, and
[`cubecl`](https://github.com/tracel-ai/cubecl) for GPU kernels. Vector search,
distance metrics and k-means come from the sister crate
[`ann-search-rs`](https://crates.io/crates/ann-search-rs); the GPU primitives
layer from [`cubecl-utils-rs`](https://crates.io/crates/cubecl-utils-rs).

## Feature flags

Everything past the base statistics and graph code is gated, so a consumer only
pays for what it uses.

| Flag | What it turns on |
| --- | --- |
| *(default)* | Bulk statistics, enrichment, graph, ontology, matrix factorisation |
| `single-cell` | The `single_cell` module, HDF5 and mmap I/O, the binary sparse format |
| `multi-modal` | `single_cell::multi_modal` (WNN, ADT), implies `single-cell` |
| `gpu` | The `gpu` module via `cubecl` (wgpu and CPU backends) and `cubek` |
| `large-test` | Slow but asserting tests: GPU parity gates, large-scale numerical checks |
| `large_scale_diagnostics` | Unasserted diagnostic sweeps that print tables for a human |

## Using it from Rust

```toml
[dependencies]
bixverse-rs = { version = "0.4", features = ["single-cell"] }
```

```rust
use bixverse_rs::prelude::*;
```

The prelude re-exports the errors, sparse structures, `SparseGraph`, matrix and
vector helpers, the SIMD trait and the assertion macros.

## How it plugs into R

The R side is a thin wrapper, and deliberately so. `bixverse` depends on this
crate through [extendr](https://extendr.github.io/), and does the things R is
good at: argument checking, S7 classes, documentation, plotting handoff. The
Rust side owns the maths.

Each top-level module carries a `*_r_wrapper.rs` sibling holding the R-callable
entry points. Those files deserialise R types into Rust-native inputs, call the
pure implementation, then serialise the result back, which keeps R types out of
the numerical surface entirely. Around 90 such entry points exist today, backing
the ~685 functions the R package exports.

Two R packages consume the crate:

- [bixverse](https://github.com/GregorLueg/bixverse) with `single-cell` and
  `multi-modal`. The main package.
- [bixverse.gpu](https://github.com/GregorLueg/bixverse.gpu) with `gpu` and
  `single-cell`, adding wgpu-backed versions of the expensive methods.

Anything on the Rust side that never crosses into R works standalone, so the
crate is equally usable from a pure Rust binary or behind a PyO3 layer.

## Roadmap

### Methods

- [x] NMF for dense and sparse data
- [x] NicheNet for single cell, see
  [Browaeys et al.](https://www.nature.com/articles/s41592-019-0667-5).
- [x] Palantir for single-cell trajectories, see
  [Setty, et al.](https://www.nature.com/articles/s41587-019-0068-4).
- [ ] Slingshot for single-cell trajectories, see
  [Street, et al.](https://link.springer.com/article/10.1186/s12864-018-4772-0).

### GPU accelerations

- [x] GPU-accelerated sparse, randomised SVD
- [x] GPU-accelerated Harmony
- [x] GPU-accelerated correlations (Metal does not have the absolute greatest
  performance here unfortunately...)
- [x] SEACells with GPU acceleration.
- [ ] GPU-accelerated BBKNN version

### Python interface

- [ ] Data loader to pull the counts out of the binary files and expose them to
  Python-based deep learning frameworks (JAX, PyTorch).

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
