"""Processing, the port of the bixverse R HVG, PCA, residual and MAGIC
helpers.

Functions take a `SingleCells` first and write what R writes: HVG statistics
to var, the HVG set to `sc.map.hvg`, the PCA, residual model and MAGIC layer
to `sc.cache`, stamped. `get_*` variants return data and leave the experiment
alone. All but `get_hvg_data` also take a `SingleCellsSubset`, as in R;
on a subset `find_hvg` leaves var alone. See `bixverse._core.pp` for the
workflow.
"""

from bixverse._core.pp import (
    BatchHvg,
    CorrectedCounts,
    ResidualFit,
    calculate_pca,
    find_hvg,
    find_hvg_batch_aware,
    fit_residuals,
    get_hvg_data,
    run_magic,
    sct_corrected_counts,
)

__all__ = [
    "BatchHvg",
    "CorrectedCounts",
    "ResidualFit",
    "calculate_pca",
    "find_hvg",
    "find_hvg_batch_aware",
    "fit_residuals",
    "get_hvg_data",
    "run_magic",
    "sct_corrected_counts",
]
