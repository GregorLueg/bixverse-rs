# Quickstart

The standard route: build an experiment, QC, HVGs, PCA, neighbours, clusters,
embedding, plots. It needs `bixverse[embed,plots,sparse]`. The blocks run top
to bottom on synthetic data, so you can paste them as they are. The
[PBMC vignette](vignettes/pbmc_single_cell.ipynb) does the same on real data and
carries on to markers and annotation.

## An experiment

Synthetic counts with planted cell types, written into a new experiment
directory:

```python
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
import bixverse.embed
from bixverse.params import SyntheticDataParams

sim = bv.io.generate_single_cell_test_data(
    SyntheticDataParams(n_cells=3000, n_genes=2000)
)

tmp = tempfile.TemporaryDirectory()
sc = bv.io.load_csr(Path(tmp.name) / "demo", sim.counts.scipy_csr(), sim.obs, sim.var)
sc
```

`load_csr` takes a cells x genes CSR matrix plus obs and var as polars frames.
Real data goes through `bv.io.load_mtx`, `load_h5ad` or `load_tenx_h5`
instead, and `bv.io.download_pbmc3k()` fetches PBMC3k to try them on. The
result is the same: a directory holding the cell- and gene-major stores and a
DuckDB with obs and var.

Back at it tomorrow? `bv.io.save(sc)` writes the cache next to the stores and
`bv.io.load_existing(directory)` opens it again, ready to go.

```python
sc.obs().head()
sc.var().head()
```

## Quality control

Gene set proportions land straight in obs. Here we take the first ten genes
as the set; on real data you'd pass the mitochondrial and ribosomal gene ids.

```python
var = sc.var()
gene_sets = {"set_a": var["gene_id"].head(10).to_list()}

bv.qc.gene_set_proportions(sc, gene_sets, streaming=False)
```

`run_cell_qc` does MAD-based outlier calls per metric and combines them:

```python
qc_df = sc.obs(cols=["lib_size", "nnz", "set_a"])

qc = bv.qc.run_cell_qc(
    {
        "log10_lib_size": np.log10(qc_df["lib_size"].to_numpy()),
        "log10_nnz": np.log10(qc_df["nnz"].to_numpy()),
        "set_a": qc_df["set_a"].to_numpy(),
    },
    sc.cells_to_keep,
    directions={
        "log10_lib_size": "twosided",
        "log10_nnz": "twosided",
        "set_a": "above",
    },
    threshold=3.0,
)

print(f"{qc.combined.sum()} of {qc.combined.size} cells flagged")

bv.pl.violin_plot(qc)
plt.show()
bv.pl.joint_plot(qc)
plt.show()
```

Nothing is deleted. Store the flag in obs and tell the object which cells to
keep. Every method from here on only sees those.

```python
sc.add_obs(pl.DataFrame({"cell_idx": qc.cell_idx, "outlier": qc.combined}))
sc.set_cells_to_keep(qc.cell_idx[~qc.combined])
sc
```

## HVGs, PCA, neighbours, clusters

```python
bv.pp.find_hvg(sc, hvg_no=500)
bv.pp.calculate_pca(sc, no_pcs=20, sparse_svd=True)
bv.graph.find_neighbours(sc)
bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
```

HVGs, PCA and the kNN graph sit in the object's cache, and `sc` prints what's
there and what has gone stale. Clusters are an obs column. On a real data set
use `hvg_no=2000` and `no_pcs=30`, and see
[clustering and annotation](vignettes/cell_type_annotation.ipynb) for
`bv.graph.fast_cluster` on big data.

## Embedding and plots

UMAP and t-SNE come from manifolds-rs (the `embed` extra), the plots from
matplotlib (the `plots` extra):

```python
bv.embed.umap(sc)

bv.pl.embedding_plot(
    sc, "umap", "leiden_clusters", label_by="leiden_clusters", discrete=True
)
plt.show()

gene = sc.var()["gene_id"][0]
bv.pl.feature_plot(sc, [gene], "umap")
plt.show()
```

`bv.embed.tsne(sc)` is the t-SNE twin, and `embedding_plot` takes `"tsne"` the
same way.

## Next

Markers with `bv.de.find_all_markers`, scType annotation, doublets, batch
correction, metacells, ligand-receptor: the vignettes cover each. Start with
[PBMCs](vignettes/pbmc_single_cell.ipynb),
[design choices](vignettes/design_single_cell.ipynb),
[I/O](vignettes/single_cell_io.ipynb) and
[visualisation](vignettes/single_cell_visualisation.ipynb). On a GPU,
`import bixverse_gpu as bvg` and swap `bv.pp.calculate_pca` for
`bvg.pp.calculate_pca`, likewise `graph` and `embed`.

## Loader

The cell-major store streams into minibatches for training. Open it with
`bv.CellReader` and iterate:

```python
reader = bv.CellReader(str(sc.cells_path))
loader = bv.CellLoader(reader, batch_size=256, layer="norm", seed=42)

for batch in loader:
    x = batch.scipy_csr()  # (256, n_genes) CSR
    break
```

A `Batch` carries `raw`, `norm`, `indices`, `indptr`, `shape` and `cell_ids`;
the layer you didn't ask for is `None`. `layer` is `"raw"`, `"norm"` or
`"both"`. `norm` is `log1p(count / library_size * target_size)` at `float16`
precision, so expect agreement with your own recomputation to about three
decimal places.

Pass an index array to iterate a subset, and `genes=` (store gene indices, in
the column order you want) to subset or reorder genes:

```python
rng = np.random.default_rng(0)
perm = rng.permutation(reader.total_cells)
n_train = int(0.8 * len(perm))

train = bv.CellLoader(reader, perm[:n_train], batch_size=256)
val = bv.CellLoader(reader, perm[n_train:], batch_size=256, shuffle=False)
```

Batches are sparse by default and you want it that way: a dense batch pays a
zero-fill and a scatter over `batch_size * n_genes` floats however few counts
it holds. Each pass reshuffles and spawns its own feeder thread; the seed is
mixed with the epoch counter. `Ctrl-C` can't interrupt a batch mid-decode,
because the GIL is released while the feeder works. For training scVI off the
store, see [scVI](scvi.md).
