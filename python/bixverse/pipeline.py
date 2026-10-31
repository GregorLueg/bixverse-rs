"""Pipelines, the port of the bixverse R `step_*`, `%>>%` and
`apply_pipeline` helpers.

Steps chain with `>>` into a `Pipeline`, which `apply_pipeline` checks
against the class of the object and then runs in order, each step mutating
the object in place as its module function does. `step_metacells` turns a
`SingleCells` or `SingleCellsSubset` into a `MetaCells` for the steps after
it. `apply_pipeline_per_group` and `meta_cells_per_group` run a pipeline on
one `SingleCellsSubset` per obs level. `Step` wraps any function that takes
the object first, the `bixverse_gpu` twins included.
See `bixverse._core.pipeline` for the workflow.
"""

from bixverse._core.pipeline import (
    Pipeline,
    Step,
    apply_pipeline,
    apply_pipeline_per_group,
    meta_cells_per_group,
    step_bbknn,
    step_clusters,
    step_fast_mnn,
    step_harmony,
    step_harmony_v2,
    step_hvg,
    step_metacells,
    step_neighbours,
    step_pca,
    validate_pipeline,
)

__all__ = [
    "Pipeline",
    "Step",
    "apply_pipeline",
    "apply_pipeline_per_group",
    "meta_cells_per_group",
    "step_bbknn",
    "step_clusters",
    "step_fast_mnn",
    "step_harmony",
    "step_harmony_v2",
    "step_hvg",
    "step_metacells",
    "step_neighbours",
    "step_pca",
    "validate_pipeline",
]
