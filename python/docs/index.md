# bixverse

A single cell suite for Python, with the numerics in Rust. Same engine as the
[R package](https://github.com/GregorLueg/bixverse) and the
[Rust crates](https://github.com/GregorLueg/bixverse-rs) underneath it:
`import bixverse as bv` and you get QC, highly variable genes, PCA,
neighbours, clustering, embeddings, differential expression, doublet
detection, batch correction, metacells, annotation, ligand-receptor analysis,
trajectories and multi-modal (WNN) workflows.

## How it fits together

An experiment is a directory, and `bv.SingleCells` is a handle on it.

- **On-disk store.** Two compressed binary files, cell-major and gene-major,
  memory-mapped on read. Counts never sit in memory as a whole. Methods pull
  the cells or genes they need from the right file.
- **DuckDB for obs and var.** The cell and gene tables live in a DuckDB next
  to the store. `sc.obs()` and `sc.var()` hand you polars frames, `sc.add_obs()`
  writes columns back.
- **Cells to keep.** QC doesn't delete anything. `sc.set_cells_to_keep()`
  decides which cells every downstream method sees, and you can reset it.
- **Rust numerics.** Everything heavy runs in the extension. Python wraps it,
  validates the arguments and keeps the results (PCA, kNN, graph, embeddings)
  in a cache on the object that tracks what's stale.

The design choices are written up in the
[design vignette](vignettes/design_single_cell.ipynb).

## Install

```bash
uv pip install bixverse            # the suite: numpy, polars, duckdb
uv pip install "bixverse[embed]"   # UMAP, t-SNE, PHATE (manifolds-rs)
uv pip install "bixverse[plots]"   # bv.pl plots (matplotlib)
uv pip install "bixverse[gpu]"     # the bixverse-gpu wheel
uv pip install "bixverse[sparse]"  # scipy, for scipy_csr on batches
uv pip install "bixverse[torch]"   # torch, for bixverse.nn
uv pip install "bixverse[mlx]"     # MLX for bixverse.nn, Apple Silicon only
uv pip install "bixverse[jax]"     # JAX for bixverse.nn, CPU or CUDA
```

Extras combine: the quickstart needs `"bixverse[embed,plots,sparse]"`.

## Thirty seconds

```python
import bixverse as bv
import bixverse.embed

sc = bv.io.load_existing("my_experiment")

bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=30)
bv.graph.find_neighbours(sc)
bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
bv.embed.umap(sc)
```

Building an experiment from MTX, h5ad, 10x h5 or an in-memory matrix is in the
[quickstart](quickstart.md). The [vignettes](vignettes/pbmc_single_cell.ipynb)
walk through the rest of the suite, the [API reference](api/core.md) has
every function.

## Loader and scVI

The package started as a minibatch loader, and that part is still here.
`bv.CellReader` opens the cell-major store and `bv.CellLoader` yields
shuffled minibatches, sparse CSR by default. A feeder thread decodes ahead of
your training step with the GIL released, so there is no Python-side decode
and no materialised experiment.

`bixverse.nn` builds on it: a barebone scVI that trains straight off a store,
on torch, MLX or JAX. See [scVI](scvi.md).

## GPU wheel

`bixverse-gpu` (`import bixverse_gpu`) is a separate wheel with GPU twins of
the CPU methods on wgpu. Modules mirror the CPU ones, take the same
`SingleCells` and write the same cache slots, so a pipeline swaps one import
for the other:

```python
import bixverse_gpu as bvg

bvg.pp.calculate_pca(sc, 30)
bvg.graph.find_neighbours(sc)
```

Without a GPU adapter the twins raise `bixverse.BixverseError`;
`bvg.gpu_available()` probes the device.
