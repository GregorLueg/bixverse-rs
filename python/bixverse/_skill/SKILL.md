---
name: bixverse-python
description: How to USE the bixverse Python package (`import bixverse as bv`), a single cell suite on the bixverse-rs Rust crates. Covers installation and extras, the on-disk SingleCells store, loading MTX / h5ad / anndata Zarr / 10x h5 / in-memory matrices, store format upgrades, QC, doublets and empty droplets, CellSweep ambient RNA removal, HVG, PCA, Pearson residuals, batch correction (Harmony, fastMNN, BBKNN, Seurat CCA/rPCA) and integration metrics, neighbours and Leiden clustering, UMAP/t-SNE/PHATE, markers (Wilcoxon or logistic regression) and differential expression (pseudobulk edgeR, NEBULA), differential abundance (miloR, MELD), ScType, Symphony and Seurat label transfer, ligand-receptor analysis, gene set scoring, HotSpot and SCENIC, NMF, DIALOGUE, trajectories, Bonsai, metacells, CITE-seq/WNN, single cell ATAC (fragments, peak calling, TF-IDF, LSI, rLSI, gene activity, differential accessibility), the minibatch loader and scVI (bixverse.nn), and the bixverse-gpu twins. Use this whenever Python code imports bixverse or bixverse_gpu, calls any bv.io / bv.qc / bv.pp / bv.graph / bv.integrate / bv.de / bv.annotate / bv.atac / bv.params function, or works with SingleCells, SingleCellsSubset, SingleCellsMultiModal, MetaCells, CellLoader or get_counts. Apply even when the package is not named but the user works on single cell data in a Python project that depends on bixverse.
---

# Using bixverse in Python

`bixverse` is a single cell suite for Python whose numerics run in the
bixverse-rs Rust crates. An experiment lives on disk, so a million cells on a
laptop is the design target.

The API has about 420 public names across 23 modules. Do not guess at
signatures. Check `references/api-index.md` for whether something exists, then
`help(bv.<module>.<name>)` or `inspect.signature()` for how to call it.

## The three things to know before writing any code

**1. The experiment is a directory, and `bv.SingleCells` is a handle on it.**
Counts sit in two compressed binary stores (cell-major and gene-major), obs and
var in a DuckDB next to them. Counts never sit in memory as a whole: methods
read the cells or genes they need. A `bv.io.load_*` call builds the directory
and returns the handle; `bv.io.load_existing(path)` reopens it.

```python
import bixverse as bv

sc = bv.io.load_h5ad("experiments/pbmc", "pbmc.h5ad")
sc.obs()  # polars frame, every cell
sc.obs(filtered=True)  # only the cells to keep
```

**2. Module functions mutate the object in place and return `None`.**
`bv.pp.find_hvg(sc, hvg_no=2000)` writes the HVGs into `sc`'s cache. Writing
`sc = bv.pp.find_hvg(sc)` sets `sc` to `None`. Methods that produce a result
table (doublet callers, QC calls, DE, annotation) return a result object
instead and leave the store untouched; read it with the module's `get_data()`.
Tunables come in frozen parameter classes from `bixverse.params`, passed as
`<what>_params=`:

```python
bv.graph.find_neighbours(
    sc, neighbours_params=bv.params.NeighboursParams(k=20, knn_method="hnsw")
)
```

**3. Print the object.** `print(sc)` shows what the cache holds (HVG, PCA,
kNN, graph, embeddings) and what has gone stale after a change to the cells to
keep. That is far cheaper than guessing, and cheaper than recomputing a PCA you
did not need.

## Where to look

| Task | Read |
|---|---|
| Installing, extras, building from source, the GPU wheel, nn backends | `references/install.md` |
| The store, obs / var, params classes, cells to keep, indices, cache and staleness, errors | `references/conventions.md` |
| Single cell: load, write (incl. streaming a gene subset of a large store to h5ad), read counts, QC, doublets, HVG, PCA, residuals, batch correction, neighbours, clustering, embeddings, markers, CellSweep | `references/single-cell.md` |
| Single cell downstream: DE, differential abundance, annotation, ligand-receptor, gene set scores, HotSpot, SCENIC, NMF, DIALOGUE, trajectories, Bonsai, metacells, CITE-seq / WNN, single cell ATAC, pipelines, loader, scVI | `references/single-cell-analysis.md` |
| Does function X exist? What is it called? | `references/api-index.md` |

## Smoke test

Runs on synthetic data, no downloads. Use it to confirm an install works before
debugging anything else.

```python
import tempfile
from pathlib import Path

import bixverse as bv
from bixverse.params import SyntheticDataParams

sim = bv.io.generate_single_cell_test_data(
    SyntheticDataParams(n_cells=3000, n_genes=2000)
)
tmp = tempfile.TemporaryDirectory()
sc = bv.io.load_csr(Path(tmp.name) / "demo", sim.counts, sim.obs, sim.var)

bv.pp.find_hvg(sc, hvg_no=500)
bv.pp.calculate_pca(sc, no_pcs=20)
bv.graph.find_neighbours(sc)
bv.graph.find_clusters(sc, res=1.0, name="clusters")

print(sc)  # the cache should list hvg, pca and knn
```

## Traps that bite everywhere

- **This is not Scanpy.** There is no AnnData in memory and no `sc.pp` /
  `sc.tl` namespace. Where an equivalent exists the name differs and the
  semantics usually differ too.
- **Normalisation and minimum QC happen at load time.** `MinQualityParams`
  (default 100 genes and 250 counts per cell, 10 cells per gene,
  `target_size` 1e4) applies even with `qc=None`, and the dropped cells and
  genes never enter the store. Pass
  `MinQualityParams(min_unique_genes=0, min_lib_size=0, min_cells=0)` to keep
  everything.
- **A missing prerequisite warns and returns.** `calculate_pca` without HVGs,
  `find_neighbours` without the named embedding, and batch correction on a
  single batch (except `fast_mnn`) warn and do nothing. Check `print(sc)` after
  each step rather than assuming a chain ran.
- **Indices are 0-based everywhere in Python** (`cell_idx`, `gene_idx`,
  `cells_to_keep`, HVG indices). The DuckDB stores them 1-based, so a raw
  `duckdb` query on the file sees 1-based keys.
- **`sc.obs()` returns every cell; `add_obs` without `cell_idx` matches by
  position.** Use `obs(filtered=True)` for the kept cells. A frame without
  `cell_idx` must hold one row per kept cell in ascending cell order, or cells
  are silently mislabelled. Join on `cell_idx` when in doubt.
- **Unknown ids warn and drop, they do not raise**, in `get_counts`,
  `gene_set_proportions` and `set_cells_to_keep`. Check the returned
  `Counts.cells` / `Counts.genes`.
- **`import bixverse` does not import `bv.embed`.** Add `import bixverse.embed`
  (needs the `embed` extra). `bv.pl` needs the `plots` extra, `.scipy_csr()`
  the `sparse` extra.
- **Fold-change units differ by method.** Pseudobulk edgeR `log_fc` is log2;
  NEBULA `log_fc` is natural log. With no `coef`, the last design column is
  tested; name `coef` explicitly.
- **Some calls return a new object you must keep.** `bv.qc.flag_cells` and
  `bv.qc.rescue_cells` return a new `CellQc`; Milo's `test_nhoods` and
  `add_nhoods_info` return a new `MiloR`; Symphony mapping writes into the
  query object.
- **`reset_cells_to_keep` raises without `force=True`.**
- **The cache survives a restart only after `bv.io.save(sc)`** (written to
  `memory_py.*`). `bv.io.save()` does not persist the residual model.
- **Stores are format v5.** `load_existing` warns on a v4 store and refuses a
  v3 one; `bv.io.upgrade_sc_store` converts both. A v5 store does not open in
  bixverse 0.3.x or in an R build on bixverse-rs 0.10.x.
- **k-means defaults to e-means** in Harmony, fast clustering (also inside
  Boost and scDblFinder), SCENIC, DSB and NMF consensus. Lloyd's is opt-in:
  `km_type="standard"` or `k_means_algorithm="lloyd"`.
- **GPU twins write their own slots for Harmony and fastMNN**
  (`harmony_gpu`, `mnn_gpu`); pass that name as `embd_to_use` downstream.
