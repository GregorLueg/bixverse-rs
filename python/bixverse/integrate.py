"""Batch correction and integration metrics, the port of the bixverse R batch
correction methods and the scIB-style metrics.

Corrections take a `SingleCells` first and write what R writes: Harmony,
fastMNN, CCA and rPCA an embedding to `sc.cache.embeddings`, BBKNN the kNN
and the neighbour graph, all stamped. The `calculate_*` metrics return
records and leave the experiment alone. All but `seurat_cca` and
`seurat_rpca` also take a `SingleCellsSubset`, as in R. See
`bixverse._core.integrate` for the workflow.
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
