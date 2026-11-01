"""Marker genes, donor-aware differential expression and differential abundance.

Three jobs. Markers: Wilcoxon between two groups (`find_markers`), one-vs-rest
for every group of a column (`find_all_markers`), or markers that hold up
against every rival group separately (`find_specific_markers`). Expression with
donor structure: pseudobulk edgeR (`get_pseudobulked`, `pseudobulk_dge`,
`run_edger_ql`) or the mixed model `nebula`. Abundance: Milo neighbourhoods
(`get_milor_abundances`, `test_nhoods`, `add_nhoods_info`) and `meld` densities.
Start with `find_all_markers` to look at clusters, `pseudobulk_dge` or `nebula`
to compare conditions across donors.

**What lands where.** Nothing here writes to obs, var or the cache. The marker
tests and `pseudobulk_dge` return polars frames; `find_specific_markers`,
`nebula`, `get_milor_abundances` and `meld` return frozen records
(`ScSpecificMarkers`, `ScNebula`, `MiloR`, `MeldRes`) with getters for their
tables. `test_nhoods` and `add_nhoods_info` hand back a new `MiloR` carrying
`nhoods_info`.

**Object types.** The marker functions, `get_pseudobulked`, `pseudobulk_dge` and
`nebula` take a `SingleCells` or a `SingleCellsSubset`, so a per cell type
comparison is a subset away. `get_milor_abundances` and `meld` take a
`SingleCells` only.

```python
markers = bv.de.find_all_markers(sc, "cell_grp")
top = markers.filter(markers["fdr"] < 0.05)
```

Condition across donors, one pseudobulk sample per donor. `cell_list` maps the
sample names to their cells and the rows of the design follow its order:

```python
import polars as pl

obs = sc.obs(filtered=True)
cell_list = {
    s: obs.filter(obs["sample_id"] == s)["cell_id"].to_list()
    for s in obs["sample_id"].unique().sort()
}
design = pl.DataFrame(
    {"(Intercept)": [1.0] * 6, "stim": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}
)
pb_res = bv.de.pseudobulk_dge(sc, cell_list, design, coef="stim")
```

Milo needs a PCA and the kNN over it, and the sample column of obs. The design
frame has one row per sample:

```python
bv.pp.find_hvg(sc, hvg_no=500)
bv.pp.calculate_pca(sc, no_pcs=20)
bv.graph.find_neighbours(sc)

design_df = sc.obs().select("sample_id", "condition").unique().sort("sample_id")
milo = bv.de.get_milor_abundances(sc, "sample_id")
milo = bv.de.test_nhoods(milo, "~ condition", design_df)
da = bv.de.get_differential_abundance_res(milo)
```

**Gotchas.** Designs are additive formulas over a frame, `"~ condition + age"`:
numeric columns as they are, everything else treatment coded against its first
(sorted) level and named `<column><level>`. Interactions and transformations
are not parsed. `coef` takes a 0-based design column or its name, and the last
column is tested if you give neither. Every index is 0-based, Milo's `Nhood`
included. `get_milor_abundances` and `meld` raise on a stale embedding or kNN,
so rerun those after the cell filter moves, and they warn and return `None`
without a kNN. `nebula` costs milliseconds to seconds per gene: hand
`genes_to_use` the HVGs or a candidate set. `run_edger_ql` wants raw counts and
a design with an intercept and at least one more column.
"""

from bixverse._core.de import (
    MeldRes,
    MiloR,
    MiloRParams,
    PseudoBulked,
    ScNebula,
    ScSpecificMarkers,
    add_nhoods_info,
    find_all_markers,
    find_markers,
    find_specific_markers,
    get_differential_abundance_res,
    get_index_cells,
    get_marker_comparisons,
    get_marker_summary,
    get_milor_abundances,
    get_params,
    get_pseudobulked,
    meld,
    nebula,
    pseudobulk_dge,
    run_edger_ql,
    test_nhoods,
)

__all__ = [
    "MeldRes",
    "MiloR",
    "MiloRParams",
    "PseudoBulked",
    "ScNebula",
    "ScSpecificMarkers",
    "add_nhoods_info",
    "find_all_markers",
    "find_markers",
    "find_specific_markers",
    "get_differential_abundance_res",
    "get_index_cells",
    "get_marker_comparisons",
    "get_marker_summary",
    "get_milor_abundances",
    "get_params",
    "get_pseudobulked",
    "meld",
    "nebula",
    "pseudobulk_dge",
    "run_edger_ql",
    "test_nhoods",
]
