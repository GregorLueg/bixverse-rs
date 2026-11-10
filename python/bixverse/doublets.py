"""Doublet callers, empty-droplet calls and CellSweep ambient RNA removal.

Three callers find doublets from the counts: `scrublet`,
`doublet_detection_boost` and `scdblfinder`. Run one, turn the result into an
obs frame with `get_data` and add it. `cellsweep` is a separate job: it removes
ambient and bulk contamination from a raw experiment that still holds its empty
droplets, and uses `infer_empty_droplets` to find them.

**What lands where.** The callers leave the experiment alone and return a frozen
record (`ScrubletRes`, `BoostRes`, `ScDblFinderRes`). `get_data` turns a record
into a frame with a 0-based `cell_idx` for `sc.add_obs`. `call_doublets_manual`
re-calls a Scrublet result at a threshold of your choosing and returns a new
record. `get_feature_mat` and `get_scores` read the classifier output of an
scDblFinder record. `infer_empty_droplets` returns a boolean array.
`cellsweep` writes a new experiment into the directory you give it and returns
it, with `cellsweep_alpha`, `cellsweep_z`, `cellsweep_beta`, `cellsweep_ll` and
`cellsweep_converged` in obs and `cellsweep_ambient` in var. The input stays as
it was.

**Object types.** All functions take a `SingleCells`. The callers run on the
kept cells unless you hand them `cells_to_use`.

```python
res = bv.doublets.scrublet(sc, group_by="sample_id")
sc.add_obs(bv.doublets.get_data(res))
sc.obs()[["doublet", "doublet_score"]].head()
```

With `group_by` every method runs once per group, and per-group scalars such as
`threshold` become `{group: value}` dicts. Doublets form inside a droplet
emulsion, so group by the sample or the lane, never across them. A group below
50 cells is an error, below 500 a warning.

**Gotchas.** `get_data` names its columns after the caller: Scrublet and Boost
both write `doublet` and `doublet_score`, so adding the second replaces the
first. Take one result at a time, or rename the columns before `add_obs`.
`call_doublets_manual` only takes Scrublet results; on a grouped one it
re-calls a single group (`for_sample`, else the first). `cellsweep` runs after
annotation and wants the empty droplets in the object, so ingest with
`MinQualityParams(min_unique_genes=0, min_lib_size=0, min_cells=0)`; with the
default `freeze_ambient_profile` it refuses an object whose smallest library
size is above 100. Barcodes that are neither
empty nor annotated and kept are left out of the fit and the output.
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
