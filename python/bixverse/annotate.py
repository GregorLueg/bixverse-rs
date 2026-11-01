"""Cell type annotation with ScType and reference mapping with Symphony.

ScType scores every cell against marker gene sets and calls a type per cell,
optionally per cluster. Symphony builds a Harmony-corrected reference from an
annotated experiment, maps a query into its embedding and transfers the labels
by kNN vote. Start with `prepare_cell_markers`, `calc_sc_type_scores` and
`assign_sc_type` for ScType, `build_symphony_ref`, `map_symphony_query` and
`transfer_labels_symphony` for Symphony.

**What lands where.** The ScType functions leave the experiment alone and return
frozen records (`ScTypeResults`, `ScTypeCellResults`); put a call into obs
yourself. `map_symphony_query` writes the query embeddings to
`query.cache.embeddings` as `"symphony"` (corrected), `"symphony_pca"`
(projected) and `"symphony_r"` (soft cluster assignments).
`transfer_labels_symphony` writes `predicted_<column>` and `confidence_<column>`
to the query's obs and returns them. `build_symphony_ref` and
`add_symphony_labels` return a `SymphonyReference` and touch neither experiment.
A reference is immutable: adding labels gives you a new one. All functions take
a `SingleCells`.

ScType wants markers as a frame of `cell_type` and `gene_id`, in the id type of
the experiment, and scores the kept cells. The neighbour graph smooths the
scores, so run `bv.graph.find_neighbours` first; clusters from
`bv.graph.find_clusters` add a per cluster call.

```python
import polars as pl

genes = sc.gene_ids().tolist()
marker_df = pl.DataFrame(
    {
        "cell_type": ["type_1"] * 10 + ["type_2"] * 10 + ["type_3"] * 10,
        "gene_id": genes[:30],
    }
)
markers = bv.annotate.prepare_cell_markers(sc, marker_df)
scores = bv.annotate.calc_sc_type_scores(sc, markers)
calls = bv.annotate.assign_sc_type(sc, scores, cluster_col="leiden_clustering")
sc.add_obs(pl.DataFrame({"cell_idx": sc.cells_to_keep, "sc_type": calls.hybrid}))
```

Symphony needs the reference's HVGs, so run `bv.pp.find_hvg` on it first and
hand `sc.map.hvg` over. Genes match by id; a reference HVG missing from the
query becomes a zero column:

```python
ref = bv.annotate.build_symphony_ref(
    sc, "sample_id", sc.map.hvg, label_columns=["cell_grp"]
)
bv.annotate.map_symphony_query(ref, query, batch_column="sample_id")
pred = bv.annotate.transfer_labels_symphony(ref, query, "cell_grp")
```

**Gotchas.** Every index is 0-based: marker genes, HVGs and cluster ids.
`cluster_col` of `assign_sc_type` has to be a non-negative integer obs column
with a value for every scored cell, which is what `find_clusters` writes.
Without a neighbour graph `assign_sc_type` warns and scores without smoothing.
`calls.hybrid` is `None` unless you gave a `cluster_col`; `calls.assignments`
is the per cell call, null where the cell is Unknown. Labels are stored in a
reference in `cells_to_keep` order, so `add_symphony_labels` needs the cell set
the reference was built on. `transfer_labels_symphony` raises on a query that
has no current `"symphony"` embedding: map again after the cell filter moves.
`save_symphony_ref` and `load_symphony_ref` use one `.npz` file; a reference
saved by the R package does not load here.
"""

from bixverse._core.annotate import (
    CellMarkers,
    ScTypeCellResults,
    ScTypeResults,
    SymphonyReference,
    add_symphony_labels,
    assign_sc_type,
    build_symphony_ref,
    calc_sc_type_scores,
    get_scores,
    get_symphony_hvg_names,
    get_symphony_labels,
    get_symphony_loadings,
    get_symphony_z_corr,
    load_symphony_ref,
    map_symphony_query,
    prepare_cell_markers,
    save_symphony_ref,
    score_clusters,
    transfer_labels_symphony,
)

__all__ = [
    "CellMarkers",
    "ScTypeCellResults",
    "ScTypeResults",
    "SymphonyReference",
    "add_symphony_labels",
    "assign_sc_type",
    "build_symphony_ref",
    "calc_sc_type_scores",
    "get_scores",
    "get_symphony_hvg_names",
    "get_symphony_labels",
    "get_symphony_loadings",
    "get_symphony_z_corr",
    "load_symphony_ref",
    "map_symphony_query",
    "prepare_cell_markers",
    "save_symphony_ref",
    "score_clusters",
    "transfer_labels_symphony",
]
