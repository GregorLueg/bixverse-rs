"""Meta cell generators, quality metrics, merging and persistence.

Four generators collapse a store into far fewer meta cells held in memory as a
`MetaCells`: bootstrapped neighbourhoods (hdWGCNA), SEACells, SuperCells and
MetaCells2. Start with one of them, then judge the result with the metrics,
and run the usual `bv.pp`, `bv.graph` and `bv.embed` calls on the `MetaCells`
itself. `save_metacells` and `load_metacells` write and read a meta cell set
as a directory.

**What lands where.** The generators take a `SingleCells` or
`SingleCellsSubset`, leave it alone and return a new `MetaCells`.
`calc_meta_cell_purity` writes `mc_purity` (and optional label and entropy
columns) to the meta cells' obs, `calc_diffusion_coordinates` writes
`density_region` to obs and `dcs`, `density_dist` and `regions` to
`mc.other_data`, and `calc_manifold_metrics` writes `compactness` and
`separation` to obs. `get_meta_cell_purity`, `calc_meta_cell_compactness` and
`calc_meta_cell_separation` return a frame and touch nothing. Each meta cell
keeps the store cells it aggregates in obs `original_cell_idx`.
`merge_meta_cells` and `meta_cells_per_group` return a new object.

**Object types.** The generators accept `SingleCells` and `SingleCellsSubset`;
the metrics and `merge_meta_cells` take `MetaCells`.

```python
mc = bv.metacells.generate_supercells(
    sc, params=bv.params.SupercellParams(graining_factor=20)
)
bv.metacells.calc_meta_cell_purity(mc, sc.obs()["cell_grp"].to_list())
bv.metacells.calc_diffusion_coordinates(mc, sc.cache.knn)
bv.metacells.calc_manifold_metrics(mc)
mc.obs(cols=["meta_cell_id", "mc_purity", "compactness", "separation"])
```

From there the meta cells go through the normal chain:

```python
bv.pp.find_hvg(mc, hvg_no=500)
bv.pp.calculate_pca(mc, no_pcs=10)
bv.graph.find_neighbours(mc)
```

**Gotchas.** Bootstrapped, SEACells and SuperCells read the cached PCA and kNN
of the source (or build the kNN again with `regenerate_knn=True`), so run
`bv.pp.find_hvg`, `bv.pp.calculate_pca` and `bv.graph.find_neighbours` on `sc`
first. A stale PCA or kNN raises. Without a kNN, the bootstrapped and
SuperCells generators warn and return `None`, SEACells raises. MetaCells2 works
on the raw counts and needs neither. Bootstrapped meta cells overlap; the
others partition, and MetaCells2 leaves outlier cells unassigned. The labels
for the purity are one per row of the source obs, or one per kept cell. The
diffusion map is built from the source kNN, so pass `sc.cache.knn`, and
`calc_manifold_metrics` needs `calc_diffusion_coordinates` first. A merged
object has several index spaces, so both of them warn and do nothing on it:
run them per source before `merge_meta_cells`. Merging also drops the caches
and the HVGs, so redo the chain afterwards. The residual model is not saved.
"""

from bixverse._core.metacells import (
    MetaCellAssignment,
    MetaCells,
    load_metacells,
    merge_meta_cells,
    save_metacells,
)
from bixverse._core.metacells_generate import (
    generate_bt_meta_cells,
    generate_metacells2,
    generate_seacells,
    generate_supercells,
)
from bixverse._core.metacells_metrics import (
    calc_diffusion_coordinates,
    calc_manifold_metrics,
    calc_meta_cell_compactness,
    calc_meta_cell_purity,
    calc_meta_cell_separation,
    get_meta_cell_purity,
)
from bixverse._core.pipeline import meta_cells_per_group

__all__ = [
    "MetaCellAssignment",
    "MetaCells",
    "calc_diffusion_coordinates",
    "calc_manifold_metrics",
    "calc_meta_cell_compactness",
    "calc_meta_cell_purity",
    "calc_meta_cell_separation",
    "generate_bt_meta_cells",
    "generate_metacells2",
    "generate_seacells",
    "generate_supercells",
    "get_meta_cell_purity",
    "load_metacells",
    "merge_meta_cells",
    "meta_cells_per_group",
    "save_metacells",
]
