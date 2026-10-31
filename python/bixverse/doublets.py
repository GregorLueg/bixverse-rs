"""Doublet detection, empty droplets and CellSweep, the port of the bixverse R
doublet callers and denoising.

The doublet callers take a `SingleCells` and return a record that `get_data`
turns into an obs frame; `cellsweep` writes a new experiment holding the
denoised counts. See `bixverse._core.doublets` for the workflow.
"""

from bixverse._core.doublets import (
    BoostRes,
    ScDblFinderRes,
    ScrubletRes,
    call_doublets_manual,
    cellsweep,
    doublet_detection_boost,
    get_data,
    get_feature_mat,
    get_scores,
    infer_empty_droplets,
    scdblfinder,
    scrublet,
)

__all__ = [
    "BoostRes",
    "ScDblFinderRes",
    "ScrubletRes",
    "call_doublets_manual",
    "cellsweep",
    "doublet_detection_boost",
    "get_data",
    "get_feature_mat",
    "get_scores",
    "infer_empty_droplets",
    "scdblfinder",
    "scrublet",
]
