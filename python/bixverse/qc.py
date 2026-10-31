"""Cell QC, the port of the bixverse R QC helpers.

Metric passes write obs columns under R's names; outlier calls return a
`CellQc` record that `get_data` turns into an obs frame. See
`bixverse._core.qc` for the workflow.
"""

from bixverse._core.qc import (
    N_CELLS_STREAMING_THRESHOLD,
    CellQc,
    MadOutlier,
    flag_cells,
    gene_set_proportions,
    get_available_features,
    get_data,
    per_cell_qc_outlier,
    rescue_cells,
    run_cell_qc,
    run_cell_qc_fixed,
    top_genes_perc,
)

__all__ = [
    "N_CELLS_STREAMING_THRESHOLD",
    "CellQc",
    "MadOutlier",
    "flag_cells",
    "gene_set_proportions",
    "get_available_features",
    "get_data",
    "per_cell_qc_outlier",
    "rescue_cells",
    "run_cell_qc",
    "run_cell_qc_fixed",
    "top_genes_perc",
]
