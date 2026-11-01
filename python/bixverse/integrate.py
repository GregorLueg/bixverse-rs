"""Batch correction and integration metrics.

Pick a correction, then check it. `harmony` (and `harmony_v2`) correct the cached
PCA. `fast_mnn`, `seurat_cca` and `seurat_rpca` work from the counts of the
batch-aware HVGs, so run `bv.pp.find_hvg_batch_aware` first and hand them its
`hvg_gene_idx`. `bbknn` skips the embedding and builds a batch-balanced kNN
instead. The `calculate_*` functions score the result, and
`calculate_integration_metrics` runs the lot and returns one row.

**What lands where.** Harmony, fastMNN, CCA and rPCA write a corrected
embedding to `sc.cache.embeddings` under `"harmony"`, `"harmony_v2"`, `"mnn"`,
`"cca"` and `"rpca"`. `bbknn` writes `sc.cache.knn` and
`sc.cache.connectivities`, replacing what is there. All are stamped. The
metrics return records (or a table) and leave the experiment alone. Everything
takes a `SingleCells` or `SingleCellsSubset` except `seurat_cca` and
`seurat_rpca`, which want a `SingleCells`. The batch labels are an obs column
named by `batch_column`; its levels are sorted, an Enum keeps its own order.

```python
bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=30)
bv.integrate.harmony(sc, "batch_index")
bv.graph.find_neighbours(sc, embd_to_use="harmony")
metrics = bv.integrate.calculate_integration_metrics(
    sc, "batch_index", cell_type_column="cell_grp", embd_to_use="harmony"
)
```

**Gotchas.** The kNN metrics (`calculate_kbet`, `calculate_lisi`,
`calculate_graph_connectivity`) read the cached kNN, so rebuild the neighbours
on the corrected embedding before scoring it; on a stale kNN they warn, on one
that covers other cells they raise. The silhouette widths are O(n^2) and
subsample to `max_cells` (5000). `calculate_pcr` compares a corrected embedding
with the PCA, so it needs both. For `bbknn` there is no embedding: pass
`embd_to_use=None` to `calculate_integration_metrics`, and the metrics that need
one come back null. A single batch warns and returns without doing anything,
bar `fast_mnn`, which raises. `seurat_cca` and `seurat_rpca` with
`use_precomputed_pca=True` still need the gene loadings, so they save less than
fastMNN does.
"""

from bixverse._core.integrate import (
    BatchSilhouetteScores,
    CellTypeAswScores,
    GraphConnectivityScores,
    KbetScores,
    LisiScores,
    PcrScores,
    bbknn,
    calculate_batch_asw,
    calculate_cell_type_asw,
    calculate_graph_connectivity,
    calculate_integration_metrics,
    calculate_kbet,
    calculate_lisi,
    calculate_pcr,
    fast_mnn,
    harmony,
    harmony_v2,
    seurat_cca,
    seurat_rpca,
)

__all__ = [
    "BatchSilhouetteScores",
    "CellTypeAswScores",
    "GraphConnectivityScores",
    "KbetScores",
    "LisiScores",
    "PcrScores",
    "bbknn",
    "calculate_batch_asw",
    "calculate_cell_type_asw",
    "calculate_graph_connectivity",
    "calculate_integration_metrics",
    "calculate_kbet",
    "calculate_lisi",
    "calculate_pcr",
    "fast_mnn",
    "harmony",
    "harmony_v2",
    "seurat_cca",
    "seurat_rpca",
]
