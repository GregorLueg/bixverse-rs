"""Meta cells, the port of the bixverse R meta cell generators, merge,
persistence and quality metrics.

Generators take a `SingleCells` or `SingleCellsSubset`, leave it alone and
return a `MetaCells`. HVGs, PCA, residual models, neighbours, clustering and
the embeddings run on a `MetaCells` through `bixverse.pp`, `bixverse.graph`
and `bixverse.embed`, as in R. See `bixverse._core.metacells` for the class.
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
    meta_cells_per_group,
)
from bixverse._core.metacells_metrics import (
    calc_diffusion_coordinates,
    calc_manifold_metrics,
    calc_meta_cell_compactness,
    calc_meta_cell_purity,
    calc_meta_cell_separation,
    get_meta_cell_purity,
)

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
