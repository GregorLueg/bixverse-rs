"""Multi-modal single cell: ADT counts (CLR or DSB), their PCA and the
weighted nearest neighbour (WNN) graph, the port of the bixverse R
`*_multimodal.R` files.

The methods take a `SingleCellsMultiModal`. Clustering, embeddings, MAGIC,
Harmony and trajectories take its other modalities through their `modality`
argument. See `bixverse._core.mm` and `bixverse._core.multimodal`.
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
