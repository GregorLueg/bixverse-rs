# Changelog

Versions here track the Python package, not the `bixverse-rs` crate. A wheel
reports the crate it vendored as `bixverse.__core_version__`.

## 0.3.0

Requires `bixverse-sc-data` 0.10.0.

**Features**

- The single cell suite of the bixverse R package, ported. `SingleCells`
  opens the same experiment directories as R (the two binary stores plus the
  DuckDB with obs and var), so a directory written on one side opens on the
  other. Module functions take the experiment first and write their results
  into it, as the R generics do; obs and var come back as polars frames.
  Parity with R is tested function by function against the R package.
  - `bv.io`: h5ad, 10x h5 and MTX loaders (single and multi-sample, with
    prescans), in-memory CSR, saving and reopening, h5ad and Cell Ranger
    writers, archives, store upgrades, synthetic data and the example data
    downloaders.
  - `bv.qc`: gene set proportions, MAD outlier calls with `CellQc`, fixed
    thresholds, rescue.
  - `bv.pp`: HVGs (VST, batch aware, residual), PCA, Pearson residuals and
    scTransform, MAGIC.
  - `bv.graph`: kNN (all the `ann-search-rs` indices), neighbour graphs,
    Leiden / Louvain, fast clustering, kNN metrics. `bv.embed`: UMAP, t-SNE,
    PHATE, densMAP, den-SNE and ForceAtlas2 through manifolds-rs (`embed`
    extra).
  - `bv.integrate`: fastMNN, Harmony v1 and v2, Seurat CCA and RPCA, BBKNN,
    kBET and the integration metrics.
  - `bv.doublets`: Scrublet, boosted doublet detection, scDblFinder, empty
    droplet calls and CellSweep ambient RNA removal.
  - `bv.de`: markers, pseudobulk with edgeR QL, NEBULA, Milo and MELD.
  - `bv.annotate`: scType and Symphony reference mapping.
  - `bv.metacells`: SEACells, SuperCells, hdWGCNA-style and MetaCells2 meta
    cells as `MetaCells`, with their quality metrics.
  - `bv.traj`: Palantir and PAGA. `bv.trees`: Bonsai cell trees.
  - `bv.pathways` and `bv.nmf`: module scores, AUCell, VISION, Hotspot, SCENIC /
    CisTarget regulons, decoupler-style activities, consensus NMF.
  - `bv.mm`: `SingleCellsMultiModal` with ADT counts, DSB / CLR and WNN.
  - `bv.lr`: CellPhoneDB and NicheNet-style ligand-receptor analysis.
    `bv.dialogue`: DIALOGUE multicellular programmes.
  - `SingleCellsSubset` for sub-clustering, and `bv.pipeline` for per-group
    pipelines (`Pipeline([...])` or chained with `>>`).
  - `bv.pl`: the plot data extractors, and matplotlib plots (`plots` extra)
    for embeddings, features, dot plots, heatmaps, violins, QC, PAGA and Milo.
  - `bv.params`: every parameter set as a frozen dataclass, generated from the
    R package's parameter specs, so names and defaults match R.
- `bixverse-gpu`, a separate wheel (`gpu` extra) with the GPU twins of PCA,
  kNN, UMAP / t-SNE, Harmony, Scrublet, DE, meta cells, NMF, pathways and
  Bonsai, under the same module names (`bixverse_gpu.pp.calculate_pca(sc,
  ...)`).
- `bixverse.nn` on JAX (`bixverse[jax]`, `backend="jax"`): the same scVI as
  the torch and MLX backends, as pure functions over parameter pytrees with an
  optax Adam and one `jax.jit` per training step. Torch weights convert with
  `_jax.torch_state_to_jax` and reproduce the loss to `rtol = 1e-5`. On 5963
  PBMC cells x 2000 HVGs, 400 epochs on an M1 Max: JAX on CPU 106 s against
  202 s for torch on CPU and 260 s for scvi-tools on CPU, with the same
  held-out ELBO and integration metrics. On a GPU MLX (35 s) and torch on MPS
  (82 s) stay ahead.
- Vignettes: the 22 single cell vignettes of the R package as Python
  notebooks, from the PBMC3k walkthrough and the design notes to batch
  correction, meta cells, trajectories, DE and DA. Each is a jupytext
  `docs/vignettes/<name>.py` (the source) with its executed `.ipynb` next to
  it, rendered on the docs site. `dev/vignettes.py execute` re-runs them,
  `dev/vignettes.py check` flags a notebook that no longer matches its source.
- `bixverse.io.get_cell_ranger_params()` builds the mtx parameters for a Cell
  Ranger output directory, as R's `get_cell_ranger_params()`.
- `download_kang_pbmc()` and `download_thymus_ageing()` return h5ad files
  Python can load (R's are qs2 `SingleCellExperiment`s), and
  `download_pbmc_1k_5p()` fetches from the bixverse-data release first.

## 0.2.1

Requires `bixverse-sc-data` 0.9.1.

**Features**

- Smaller wheel: depends on `bixverse-sc-data` without its `analysis` feature,
  which drops `faer`, `ann-search-rs`, `statrs` and `wide` from the build.

## 0.2.0

Requires `bixverse-sc-data` 0.9.0.

**Features**

- Reads stores of file version 4, the new format of the crate: delta
  bit-packed chunks that are about five times smaller on disk. Stores of
  version 3 keep opening, so no existing file needs converting.
- Cold reads prefetch their byte range into the page cache with large
  parallel reads (unix), which roughly doubled cold read throughput on the
  stores measured.

**Notes**

- Version 4 files are not readable by wheels built against `bixverse-sc-data`
  0.8.x.

## 0.1.1

Requires `bixverse-sc-data` 0.8.0.

**Features**

- Leverages the redesign of the crate into smaller subcrates.

## 0.1.0

First release.

- `CellReader` opens a bixverse binary sparse store and reports its shape,
  layout and normalisation target size.
- `CellLoader` iterates a cell-major store in minibatches. One feeder thread
  decompresses ahead of the consumer into a bounded queue, with the GIL
  released while it works. Each pass reshuffles and spawns its own feeder, so
  concurrent passes do not share state.
- Batches are sparse CSR by default. On a 20k x 20k store at 10% density the
  dense layout measured 53% slower per epoch, which is what settled the
  default.
- `Batch.scipy_csr()` rebuilds a batch as a `scipy.sparse.csr_matrix` over the
  same buffers.
- Errors from the crate surface as `BixverseError` and `SparseFileError`, with
  missing and unreadable files mapping to the built-in `OSError` hierarchy.
- `CellLoader(genes=...)` loads a gene subset in the order given, on both the
  sparse and the dense path, and every `Batch` carries `cell_ids`, the store
  index of each row, so per-cell covariates can be joined back on.
- `bixverse.nn`: a barebone scVI trained straight off a store, on torch
  (`bixverse[torch]`) or MLX (`bixverse[mlx]`, Apple Silicon). `train_scvi`,
  `scvi_latent` and `scvi_elbo`, with scvi-tools 1.5.1's defaults (ZINB or NB,
  observed library size, gene-wise dispersion). A trained scvi-tools module
  loads into it and reproduces its loss to `rtol = 1e-5`. On 5257 PBMC cells
  x 2000 HVGs, 400 epochs on an M1 Max: scvi-tools on MPS 170 s, ours on MPS
  74 s, ours on MLX 31 s, with held-out ELBO and integration metrics inside
  scvi-tools' seed-to-seed spread.
