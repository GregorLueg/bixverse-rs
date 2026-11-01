"""Pipelines: the preprocessing calls you repeat on every subset, written once.

Chain steps with `>>` into a `Pipeline` and apply it. A `step_*` function wraps
one module function (`step_hvg` is `bv.pp.find_hvg`, `step_pca` is
`bv.pp.calculate_pca`, and so on) with its arguments bound; `Step` wraps any
other function that takes the object first. Nothing runs until `apply_pipeline`.
It checks the chain against the class of the object, then calls each step in
order. Start with `apply_pipeline_per_group` when the point is sub-clustering or
per-donor processing.

**What lands where.** A pipeline is inert, so building one touches nothing.
`apply_pipeline` runs the steps on the object you pass and the results land
where each wrapped function puts them: the HVGs and PCA in `sc.map` and
`sc.cache`, the clusters in obs. `apply_pipeline_per_group` cuts one
`SingleCellsSubset` per obs level over the kept cells, runs the pipeline on each
and returns a dict keyed by group; the parent is left alone. `step_metacells` is
the one step that changes the class: it takes a `SingleCells` or
`SingleCellsSubset` and hands a `MetaCells` to the steps after it, so a pipeline
ending on it gives you meta cells that never mix two groups.
`meta_cells_per_group` does that in one call and merges them with
`merge_meta_cells`.

```python
pipe = (
    bv.pipeline.step_hvg(hvg_no=500)
    >> bv.pipeline.step_pca(20)
    >> bv.pipeline.step_neighbours()
    >> bv.pipeline.step_clusters(res=0.5, name="sub_leiden")
)
per_sample = bv.pipeline.apply_pipeline_per_group(pipe, sc, "sample_id")

# End on meta cells: one set per sample, never mixed
params = bv.params.BtMetacellsParams(target_no_metacells=50)
mc_pipe = pipe >> bv.pipeline.step_metacells("bootstrapped", params=params)
per_sample_mc = bv.pipeline.apply_pipeline_per_group(mc_pipe, sc, "sample_id")
```

`Step` also takes the `bixverse_gpu` twins:

```python
import bixverse_gpu as bvg

gpu_pca = bv.pipeline.Step("gpu_pca", bvg.pp.calculate_pca, {"no_pcs": 20})
bv.pipeline.apply_pipeline(bv.pipeline.step_hvg(hvg_no=500) >> gpu_pca, sc)
```

**Gotchas.** Arguments are bound when the step is built, so a misspelt keyword
fails there, and the class check runs before the first step does, so a bad
chain fails before the expensive part. Order is on you otherwise: clusters and
meta cells need `step_neighbours` first, which needs the PCA. The integration
steps (`step_harmony`, `step_harmony_v2`, `step_bbknn`, `step_fast_mnn`) take a
`SingleCells` or a subset, not `MetaCells`, so they go before `step_metacells`.
`step_fast_mnn` wants the batch-aware HVGs from `bv.pp.find_hvg_batch_aware`.
After `step_harmony`, point `step_neighbours` at it with
`embd_to_use="harmony"`. Applied to a `SingleCells`, a pipeline writes into that
object.
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
