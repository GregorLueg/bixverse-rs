"""Multi-modal single cell: ADT counts, their PCA and the WNN graph.

Adds a protein modality to a `SingleCellsMultiModal`, normalises it (CLR or
DSB), runs a PCA over it and fuses it with the RNA through a weighted nearest
neighbour (WNN) graph. The RNA side is a plain `SingleCells`, so every RNA
function takes the object as it is.

Start with `add_adt_counts`, then `calculate_pca_adt`. Run the RNA PCA the
usual way (`bv.pp.find_hvg`, `bv.pp.calculate_pca`) and call
`generate_wnn_graph`. From there `bv.graph.find_clusters` and `bv.embed`
take `modality="wnn"`; `modality="adt"` does the same for the protein side
alone.

**What lands where.** `add_adt_counts` cuts the counts to the kept cells,
normalises them and stores them in `mm.adt`; it also writes the `var_adt` table
to the DuckDB and `adt_nnz` and `adt_lib_size` to obs. The counts stay in
memory, not in the count store, and `bv.io.save` keeps them. `calculate_pca_adt`
writes scores, loadings and singular values to `mm.adt_cache`.
`generate_wnn_graph` puts the fused kNN, its graph and the per-cell modality
weights in `mm.wnn`. `new_adt_counts_clr`, `new_adt_counts_dsb` and the
`get_*` functions return their result and leave the experiment alone.

```python
# adt_counts: dense cells x proteins array; barcodes and proteins: their labels
mm = bv.SingleCellsMultiModal(sc.dir)
bv.mm.add_adt_counts(mm, adt_counts, barcodes, proteins, method="clr")
bv.pp.find_hvg(mm, hvg_no=500)
bv.pp.calculate_pca(mm, no_pcs=20)
bv.mm.calculate_pca_adt(mm, no_pcs=10)
bv.mm.generate_wnn_graph(mm)
bv.graph.find_clusters(mm, modality="wnn")
weights = bv.mm.get_wnn_weights(mm)
```

**Gotchas.** Every kept cell needs a row in the ADT barcodes, or
`add_adt_counts` raises. The matrix is cut to the kept cells when you add it:
dropping cells afterwards is fine, putting dropped cells back is not
(`calculate_pca_adt` then raises). DSB
wants the isotype controls by default (`DsbParams.use_isotype_controls`), so
pass `isotype_names`, or switch the option off. Leave the isotypes out of the
ADT PCA with `features=`. `generate_wnn_graph` needs the PCA of both
modalities, rebuilds the per-modality kNN graphs itself at `knn_range`
neighbours rather than reusing the cached ones, and replaces any earlier WNN
result with its embeddings.
"""

from bixverse._core.mm import (
    add_adt_counts,
    calculate_pca_adt,
    generate_wnn_graph,
    get_adt_counts,
    get_adt_feature_info,
    get_adt_names,
    get_adt_sample_info,
    get_wnn_weights,
    new_adt_counts_clr,
    new_adt_counts_dsb,
)
from bixverse._core.multimodal import AdtCounts, DsbInfo, WnnSlot

__all__ = [
    "AdtCounts",
    "DsbInfo",
    "WnnSlot",
    "add_adt_counts",
    "calculate_pca_adt",
    "generate_wnn_graph",
    "get_adt_counts",
    "get_adt_feature_info",
    "get_adt_names",
    "get_adt_sample_info",
    "get_wnn_weights",
    "new_adt_counts_clr",
    "new_adt_counts_dsb",
]
