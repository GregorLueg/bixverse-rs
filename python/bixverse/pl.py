"""Plot data extractors and matplotlib plots for single cell results.

Two layers. The `extract_*` functions pull what one plot needs out of an
experiment into a polars frame and need nothing beyond the core package. The
plots draw from the same data, take the object or the frame its extractor
returns, and need matplotlib, the `plots` extra (`pip install
bixverse[plots]`). matplotlib is imported on first use of a plot, not with this
module. Hand a plot the frame when you want to restyle it, or draw the same data
with another library.

**Where to start.** `embedding_plot` for an obs column on an embedding,
`feature_plot` for genes on an embedding, `dot_plot`, `stacked_violin_plot` and
`heatmap_plot` for markers across groups, `feature_scatter_plot` for two genes
against each other. The QC plots (`violin_plot`, `density_plot`, `joint_plot`)
take the `CellQc` from `bv.qc.run_cell_qc` or a frame. `paga_plot` and
`milo_nhood_plot` take the result of `bv.traj.run_paga` and the tested `MiloR`
from `bv.de`.

**What lands where.** Nothing here writes to the experiment. Every plot reads
the kept cells of a `SingleCells`, `SingleCellsSubset`, `MetaCells` or
`SingleCellsMultiModal` and returns a matplotlib object: the single panel plots
take an optional `ax` and return it, the multi-panel ones return the `Figure`.
The extractors return one row per kept cell (per meta cell for `MetaCells`),
long over genes where the plot needs it. Gene arguments are var ids.

```python
import matplotlib.pyplot as plt

genes = sc.gene_ids()[:4].tolist()
bv.pl.embedding_plot(sc, "umap", "cell_grp", label_by="cell_grp")
bv.pl.feature_plot(sc, genes, "umap", ncol=2)
bv.pl.dot_plot(sc, genes, "cell_grp")

frame = bv.pl.extract_embedding_data(sc, "umap", obs_cols=["cell_grp"])
plt.show()
```

**Gotchas.** The embedding has to exist (`bv.embed.umap` and friends; `"pca"`
works after `bv.pp.calculate_pca`). Numeric obs columns get a colour bar, and
`label_by` refuses them: cluster ids stored as numbers need `discrete=True`.
Point and label sizes are in mm and `point_size=None` picks a size from the
cell count; above 100,000 cells the points are rasterised. `density_plot` needs a
`CellQc` run with `groups`. `violin_plot` and `density_plot` take a plain frame
too, with `grouping_column` and `variable`; the outliers are then recomputed by
MAD per group.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from bixverse._core.plot_data import (
    GraphPlotData,
    extract_dot_plot_data,
    extract_embedding_data,
    extract_feature_pair,
    extract_feature_plot_data,
    extract_gene_expression,
    extract_gene_violin_data,
    extract_milo_plot_data,
    extract_paga_plot_data,
)

if TYPE_CHECKING:
    from bixverse._core.plots import (
        bx_colors,
        density_plot,
        dot_plot,
        embedding_plot,
        feature_plot,
        feature_scatter_plot,
        heatmap_plot,
        joint_plot,
        milo_nhood_plot,
        paga_plot,
        stacked_violin_plot,
        violin_plot,
    )

PLOTS: tuple[str, ...] = (
    "bx_colors",
    "density_plot",
    "dot_plot",
    "embedding_plot",
    "feature_plot",
    "feature_scatter_plot",
    "heatmap_plot",
    "joint_plot",
    "milo_nhood_plot",
    "paga_plot",
    "stacked_violin_plot",
    "violin_plot",
)


def __getattr__(name: str) -> object:
    """The plots, imported with matplotlib on first access."""
    if name in PLOTS:
        return getattr(import_module("bixverse._core.plots"), name)
    raise AttributeError(f"module 'bixverse.pl' has no attribute '{name}'")


__all__ = [
    "GraphPlotData",
    "bx_colors",
    "density_plot",
    "dot_plot",
    "embedding_plot",
    "extract_dot_plot_data",
    "extract_embedding_data",
    "extract_feature_pair",
    "extract_feature_plot_data",
    "extract_gene_expression",
    "extract_gene_violin_data",
    "extract_milo_plot_data",
    "extract_paga_plot_data",
    "feature_plot",
    "feature_scatter_plot",
    "heatmap_plot",
    "joint_plot",
    "milo_nhood_plot",
    "paga_plot",
    "stacked_violin_plot",
    "violin_plot",
]
