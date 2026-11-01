# bixverse

The single cell suite of the [bixverse R package](https://github.com/GregorLueg/bixverse)
in Python, with the numerics in Rust ([`bixverse-rs`](https://github.com/GregorLueg/bixverse-rs)).
Same engine, same parameters, same on-disk format: an experiment written from R
opens in Python and the other way round, and the results are tested against R
function by function.

It works on the store, not in memory, so it scales past what fits in RAM. On top
sit a minibatch loader and a barebone scVI that train straight off that store,
and a separate wheel with GPU twins of the expensive methods.

Docs, API reference and 22 vignettes: <https://gregorlueg.github.io/bixverse-rs/>

## Install

```bash
uv pip install bixverse                         # the suite
uv pip install "bixverse[embed,plots]"          # plus UMAP & co and matplotlib plots
uv pip install "bixverse[gpu]"                  # plus the GPU twins
uv pip install "bixverse[torch]"                # scVI on torch; also [mlx], [jax]
uv pip install "bixverse[sparse]"               # scipy, for Batch.scipy_csr
```

## Use

```python
from pathlib import Path

import bixverse as bv
import bixverse.embed

sc = bv.io.load_mtx(
    Path("experiments/pbmc3k"),
    bv.io.get_cell_ranger_params(bv.io.download_pbmc3k()),
)

bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=30)
bv.graph.find_neighbours(sc)
bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
bv.embed.umap(sc)

bv.pl.embedding_plot(sc, "umap", "leiden_clusters", discrete=True)
markers = bv.de.find_all_markers(sc, "leiden_clusters")

bv.io.save(sc)
sc = bv.io.load_existing("experiments/pbmc3k")
```

An experiment is a directory and `SingleCells` a handle on it. Counts sit in two
compressed binary files, cell-major and gene-major, memory-mapped on read, so
each method pulls what it needs from the right one. obs and var live in a DuckDB
next to them and come back as polars frames. QC doesn't delete anything:
`sc.set_cells_to_keep()` decides which cells every later step sees. Functions take
the experiment first and write into it, as the R generics do, and the cache
tracks what has gone stale.

## What's in it

| Module | Contents |
| --- | --- |
| `bv.io` | h5ad, 10x h5, MTX and in-memory loaders, saving, h5ad and Cell Ranger writers, archives, example data |
| `bv.qc` | gene set proportions, MAD outlier calls, fixed thresholds |
| `bv.pp` | HVGs, PCA, Pearson residuals and scTransform, MAGIC |
| `bv.graph` | kNN (the `ann-search-rs` indices), neighbour graphs, Leiden and Louvain, fast clustering |
| `bv.embed` | UMAP, t-SNE, PHATE, densMAP, den-SNE, ForceAtlas2 (`embed` extra) |
| `bv.integrate` | fastMNN, Harmony v1 and v2, Seurat CCA and RPCA, BBKNN, kBET and friends |
| `bv.doublets` | Scrublet, boosted doublet detection, scDblFinder, empty droplets, CellSweep |
| `bv.de` | markers, pseudobulk with edgeR QL, NEBULA, Milo, MELD |
| `bv.annotate` | scType, Symphony reference mapping |
| `bv.metacells` | SEACells, SuperCells, hdWGCNA-style and MetaCells2 meta cells |
| `bv.traj`, `bv.trees` | Palantir, PAGA, gene trends; Bonsai cell trees |
| `bv.pathways`, `bv.nmf` | module scores, AUCell, VISION, Hotspot, SCENIC, decoupler-style activities; consensus NMF |
| `bv.mm` | ADT counts, DSB and CLR, WNN |
| `bv.lr`, `bv.dialogue` | CellPhoneDB, NicheNet-style ligand-receptor; DIALOGUE |
| `bv.pl` | matplotlib plots (`plots` extra) and the `extract_*` frames behind them |
| `bv.pipeline` | chained steps with `>>`, per-group pipelines |
| `bv.params` | every parameter set as a frozen dataclass, generated from the R specs |

`SingleCellsSubset` handles sub-clustering, `MetaCells` and `SingleCellsMultiModal`
go through the same functions.

## GPU

`bixverse-gpu` is its own wheel, on wgpu. Each module mirrors its CPU namesake,
takes the same objects and writes the same cache slots, so a pipeline swaps one
import for the other:

```python
import bixverse_gpu as bvg

bvg.pp.calculate_pca(sc, 30)
bvg.graph.find_neighbours(sc)
```

`bvg.gpu_available()` probes for an adapter. Without one the twins raise
`bixverse.BixverseError`.

## Loader and scVI

`CellLoader` streams shuffled minibatches off the cell-major store. A feeder thread
decodes ahead of your training step with the GIL released, and nothing gets
materialised in memory.

```python
reader = bv.CellReader(str(sc.cells_path))
loader = bv.CellLoader(reader, batch_size=256, layer="norm", seed=42)

for batch in loader:
    x = batch.scipy_csr()  # (256, n_genes) CSR
```

Pass an index array for a train / validation split and `genes=` to subset or
reorder the columns. `cell_ids` on a batch joins per-cell covariates back on after
the shuffle. Batches are sparse by default and you want that: a dense batch pays a
zero-fill and a scatter over `batch_size * n_genes` floats, however few counts it
holds. A few things to watch:

- Cell-major stores only. A gene-major file raises on construction.
- `Ctrl-C` can't interrupt a batch mid-decode, because the GIL is released while
  the feeder works.
- The reader hints random access: right for a shuffled loader, wrong for a
  sequential full scan.

`bixverse.nn` trains scVI off the same store on torch, MLX or JAX, with scvi-tools
1.5.1's defaults, and a trained scvi-tools module loads into it and reproduces its
loss.

```python
import bixverse.nn as bvnn

fit = bvnn.train_scvi(reader, cells, hvg, batch, backend="torch")
latent = bvnn.scvi_latent(fit, reader, cells, batch)
```

The docs have the [validation against scvi-tools](https://gregorlueg.github.io/bixverse-rs/scvi/).

## Vignettes

The single cell vignettes of the R package, ported to notebooks. Start with
"Thinking the bixverse way" and "Analysing PBMCs". Each is a jupytext `.py` in
`docs/vignettes/` with its executed `.ipynb` next to it, and the data sets download
themselves on first use.

```bash
uv sync --group dev --group vignettes
uv run python dev/vignettes.py execute                   # all of them
uv run python dev/vignettes.py execute pbmc_single_cell  # or by name
```

## Development

```bash
uv sync --group dev
uv run maturin develop --release
uv run pytest tests -q
```

The `tests/*_r.py` parity tests run `dev/sc_*_fixtures.R` and skip without Rscript
and bixverse 0.9.0 or later. The GPU wheel builds from `gpu/`.
