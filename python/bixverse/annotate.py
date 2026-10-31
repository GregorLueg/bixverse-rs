"""Cell type annotation (ScType) and reference mapping (Symphony), the port
of the bixverse R annotation and reference methods.

ScType returns frozen records with R's getters; Symphony builds an immutable
`SymphonyReference`, maps queries into its embedding and transfers labels by
kNN vote. See `bixverse._core.annotate` for the workflow.
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
