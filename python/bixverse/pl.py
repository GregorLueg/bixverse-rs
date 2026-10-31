"""Plotting: the plot data extractors of the bixverse R package and the
matplotlib port of the bixverse.plots single cell plots.

The `extract_*` functions return polars frames and need nothing beyond the
core package. The plots need matplotlib, the `plots` extra
(`pip install bixverse[plots]`), which is imported on first use of a plot,
not with this module. See `bixverse._core.plot_data` and
`bixverse._core.plots`.

    >>> import bixverse as bv
    >>> frame = bv.pl.extract_embedding_data(sc, "umap", obs_cols=["leiden"])
    >>> ax = bv.pl.embedding_plot(sc, "umap", "leiden", label_by="leiden")
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
