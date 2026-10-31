"""Single cell plots, the matplotlib port of the bixverse.plots `*_sc` plots.

Every plot takes the object (or the frame its `extract_*` function returns)
and draws with the bixverse palettes, ordering and facets of the R package.
Single panel plots take an optional `ax` and return it; multi-panel ones
return the `Figure`. ggplot sizes (mm) are converted to points, so the R
defaults keep their meaning.

Needs matplotlib, the `plots` extra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.colors import (
        Colormap,
        LinearSegmentedColormap,
        Normalize,
        to_hex,
    )
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
except ImportError as e:
    raise ImportError(
        "The bixverse.pl plots need matplotlib: `pip install bixverse[plots]`. "
        "The extract_* functions work without it."
    ) from e

from bixverse._core.de import MiloR
from bixverse._core.multimodal import Modality
from bixverse._core.plot_data import (
    ExprModality,
    Layer,
    _as_factor,
    extract_dot_plot_data,
    extract_embedding_data,
    extract_feature_pair,
    extract_feature_plot_data,
    extract_gene_expression,
    extract_gene_violin_data,
    extract_milo_plot_data,
    extract_paga_plot_data,
)
from bixverse._core.qc import CellQc, Direction, get_data, per_cell_qc_outlier
from bixverse._core.subset import ScOrMc
from bixverse._core.traj import PagaRes

###########
# Globals #
###########

Palette = Literal["main", "sequential", "diverging", "viridis", "spectral"]
ContinuousPalette = Literal["sequential", "spectral", "viridis", "diverging"]

# MetBrewer Austria, Benedictus[6:10], Hiroshige and RColorBrewer Spectral
# reversed, dumped from R.
_AUSTRIA = ("#a40000", "#16317d", "#007e2f", "#ffcd12", "#b86092", "#721b3e", "#00b7a7")
# MetBrewer's discrete pick: a colour is used once n reaches its rank
_AUSTRIA_ORDER = (1, 2, 3, 4, 6, 5, 7)
_RAMPS: dict[str, tuple[str, ...]] = {
    "sequential": ("#eaf3ff", "#c5daf6", "#a1c2ed", "#4060c8", "#1a318b"),
    "diverging": (
        "#e76254",
        "#ef8a47",
        "#f7aa58",
        "#ffd06f",
        "#ffe6b7",
        "#aadce0",
        "#72bcd5",
        "#528fad",
        "#376795",
        "#1e466e",
    ),
    "spectral": (
        "#5E4FA2",
        "#3288BD",
        "#66C2A5",
        "#ABDDA4",
        "#E6F598",
        "#FFFFBF",
        "#FEE08B",
        "#FDAE61",
        "#F46D43",
        "#D53E4F",
        "#9E0142",
    ),
}

# R's named greys
GREY = {
    "grey30": "#4d4d4d",
    "grey40": "#666666",
    "grey50": "#7f7f7f",
    "grey60": "#999999",
    "grey80": "#cccccc",
    "grey90": "#e5e5e5",
    "lightgrey": "#d3d3d3",
}
OUTLIER_COLOURS = {False: GREY["lightgrey"], True: "orange"}

# ggplot sizes are mm, matplotlib wants points
_PT_PER_MM = 72.27 / 25.4
# ggplot's `.stroke`: stroke widths are in 1/96 inch
_STROKE_PT = 96 / 25.4
RASTER_CELLS = 100_000


############
# Palettes #
############


@beartype
def _ramp(colours: Sequence[str], n: int) -> list[str]:
    """`n` colours interpolated along `colours`, as `colorRampPalette`."""
    cmap = LinearSegmentedColormap.from_list("ramp", list(colours))
    return [to_hex(cmap(t)) for t in np.linspace(0, 1, n)]


@beartype
def bx_colors(
    palette: Palette = "main", *, n: int = 20, reverse: bool = False
) -> list[str]:
    """The bixverse palettes, as R's `bx_colors`.

    Palettes shorter than `n` are ramped up (in RGB; R ramps the fixed
    length ones in Lab).

    Args:
        palette: `"main"` (MetBrewer Austria), `"sequential"` (Benedictus),
            `"diverging"` (Hiroshige, red to blue), `"viridis"` or
            `"spectral"` (reversed Spectral, indigo to dark red).
        n: Number of colours.
        reverse: Reverse the order.

    Returns:
        Hex colours.
    """
    if n < 1:
        raise ValueError("n must be positive")
    match palette:
        case "main":
            if n <= len(_AUSTRIA):
                pal = [
                    c for c, r in zip(_AUSTRIA, _AUSTRIA_ORDER, strict=True) if r <= n
                ]
            else:
                pal = _ramp(_AUSTRIA, n)
        case "viridis":
            cmap = mpl.colormaps["viridis"]
            pal = [to_hex(cmap(t)) for t in np.linspace(0, 1, n)]
        case "sequential" | "diverging" | "spectral":
            base = _RAMPS[palette]
            pal = list(base) if n <= len(base) else _ramp(base, n)
        case _:
            raise ValueError(f"Unknown palette: '{palette}'")
    return pal[::-1] if reverse else pal


@beartype
def _cmap(palette: Palette, reverse: bool = False) -> Colormap:
    """A continuous colour map over a palette."""
    if palette == "viridis":
        cmap = mpl.colormaps["viridis"]
        return cmap.reversed() if reverse else cmap
    colours = _AUSTRIA if palette == "main" else _RAMPS[palette]
    return LinearSegmentedColormap.from_list(
        f"bx_{palette}", list(colours)[::-1] if reverse else list(colours)
    )


###########
# Helpers #
###########


@beartype
def _area(size_mm: float, stroke: float = 0.5) -> float:
    """A ggplot point size (mm diameter) as a matplotlib marker area.

    ggplot draws a point at `size * .pt + stroke * .stroke / 2`; the stroke
    term (0.5 by default) dominates small points, so it is kept.
    """
    return (size_mm * _PT_PER_MM + stroke * _STROKE_PT / 2) ** 2


@beartype
def _figure(ax: Axes) -> Figure:
    """The figure an axes sits in."""
    fig = ax.get_figure(root=True)
    assert isinstance(fig, Figure)
    return fig


@beartype
def auto_point_size(n_samples: int, raster: bool = False) -> float:
    """Seurat's `AutoPointSize`, as R's `auto_point_size`."""
    return 1.0 if raster else min(1583 / max(n_samples, 1), 1.0)


@beartype
def _new_ax(ax: Axes | None, figsize: tuple[float, float] = (6.0, 5.0)) -> Axes:
    """`ax`, or the axes of a new figure."""
    if ax is not None:
        return ax
    _, new = plt.subplots(figsize=figsize, layout="constrained")
    return new


@beartype
def _is_discrete(s: pl.Series) -> bool:
    """A factor-like column: strings, Enums, categoricals, booleans."""
    return s.dtype in (pl.String, pl.Boolean) or isinstance(
        s.dtype, pl.Enum | pl.Categorical
    )


@beartype
def _factor(s: pl.Series) -> tuple[np.ndarray, list[str]]:
    """Codes and the levels that occur, in level order, as a ggplot scale."""
    codes, levels = _as_factor(s)
    used = sorted(set(codes[codes >= 0].tolist()))
    remap = {c: i for i, c in enumerate(used)}
    return np.asarray([remap.get(c, -1) for c in codes.tolist()]), [
        levels[c] for c in used
    ]


@beartype
def _legend(
    ax: Axes,
    title: str,
    handles: Sequence[Line2D],
    labels: Sequence[str],
    *,
    where: Literal["right", "under"] = "right",
) -> None:
    """A titled legend outside the axes: to the right, or `"under"` a colour
    bar (given its axes). Several can sit on one axes."""
    if not handles:
        return
    loc, anchor = {
        "right": ("upper left", (1.02, 1.0)),
        "under": ("upper left", (0.0, -0.06)),
    }[where]
    leg = Legend(
        ax,
        list(handles),
        list(labels),
        loc=loc,
        bbox_to_anchor=anchor,
        frameon=False,
        fontsize=8,
        title=title,
        title_fontproperties={"weight": "bold", "size": 8},
        alignment="left",
    )
    # add_artist clips to the axes, which hides a legend outside them and
    # keeps it out of the layout's tight bbox
    leg.set_clip_on(False)
    ax.add_artist(leg)


@beartype
def _key_rows(
    ax: Axes, rows: Sequence[tuple[str, Sequence[Line2D], Sequence[str]]]
) -> None:
    """Size and width keys under the axes, one titled row per scale.

    A legend fills column-major, so the keys are interleaved row by row
    behind a column of row titles."""
    rows = [r for r in rows if r[1]]
    if not rows:
        return
    width = max(len(r[1]) for r in rows)
    blank = Line2D([], [], linestyle="none")
    handles: list[Line2D] = [blank] * len(rows)
    labels = [r[0] for r in rows]
    for k in range(width):
        for _, hs, ls in rows:
            handles.append(hs[k] if k < len(hs) else blank)
            labels.append(ls[k] if k < len(ls) else "")
    # handles are sized in font units, and a big node key would spill over
    biggest = max(float(h.get_markersize()) for h in handles) / 8
    leg = Legend(
        ax,
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=width + 1,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        handlelength=max(2.0, biggest),
        handleheight=max(0.7, biggest),
    )
    for text in leg.get_texts()[: len(rows)]:
        text.set_fontweight("bold")
    # as in `_legend`: unclipped, so it shows and the layout makes room
    leg.set_clip_on(False)
    ax.add_artist(leg)


@beartype
def _dot_handles(colours: Sequence[str], size: float = 6.0) -> list[Line2D]:
    """Round legend keys."""
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=size, color=c)
        for c in colours
    ]


@beartype
def _size_handles(
    values: np.ndarray, area: Callable[[np.ndarray], np.ndarray], n: int = 4
) -> tuple[list[Line2D], list[str]]:
    """Legend keys for a size scale at a few round values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [], []
    ticks = np.asarray(
        MaxNLocator(n, min_n_ticks=2).tick_values(finite.min(), finite.max())
    )
    ticks = ticks[(ticks >= finite.min()) & (ticks <= finite.max()) & (ticks > 0)]
    sizes = np.sqrt(area(ticks))
    return (
        [
            Line2D(
                [], [], marker="o", linestyle="none", markersize=s, color=GREY["grey60"]
            )
            for s in sizes
        ],
        [f"{t:g}" for t in ticks],
    )


@beartype
def _label_centroids(
    ax: Axes,
    frame: pl.DataFrame,
    label_by: str,
    *,
    colour: str,
    size: float,
    font: str,
) -> None:
    """Boxed labels at the mean position of each group, R's `label_centroids`."""
    if frame[label_by].dtype.is_numeric():
        raise ValueError("label_centroids(): label_by is continuous.")
    cent = (
        frame.group_by(label_by, maintain_order=True)
        .agg(pl.col("dim_1").mean(), pl.col("dim_2").mean())
        .drop_nulls(label_by)
    )
    for lab, x, y in cent.iter_rows():
        ax.text(
            x,
            y,
            # booleans as the legend spells them
            {True: "TRUE", False: "FALSE"}[lab] if isinstance(lab, bool) else str(lab),
            color=colour,
            fontsize=size * _PT_PER_MM,
            fontweight=font if font in ("bold", "normal") else "normal",
            fontstyle="italic" if font == "italic" else "normal",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.5, "lw": 0},
        )


@beartype
def _scatter(
    ax: Axes,
    frame: pl.DataFrame,
    colour: str,
    *,
    palette: Palette | None,
    size: float,
    alpha: float,
    raster: bool,
    norm: Normalize | None = None,
    highlight: bool = False,
    highlight_quantile: float = 0.25,
) -> PathCollection | None:
    """R's `.plot_embedding` on one panel: points in data order coloured by
    a discrete or continuous column. Returns the colour mappable, if any."""
    values = frame[colour]
    x, y = frame["dim_1"].to_numpy(), frame["dim_2"].to_numpy()
    if _is_discrete(values):
        codes, levels = _factor(values)
        cols = np.array(
            [*bx_colors(palette or "main", n=max(len(levels), 1)), GREY["grey50"]]
        )
        ax.scatter(
            x,
            y,
            c=cols[codes],
            s=_area(size),
            alpha=alpha,
            linewidths=0,
            rasterized=raster,
        )
        _legend(ax, colour, _dot_handles(cols[: len(levels)].tolist()), levels)
        return None

    v = values.cast(pl.Float64).to_numpy()
    cmap = _cmap(palette or "sequential")
    norm = norm or Normalize(np.nanmin(v), np.nanmax(v))
    if highlight:
        thr = np.nanquantile(v, highlight_quantile)
        bg = v <= thr
        ax.scatter(
            x[bg],
            y[bg],
            c=GREY["lightgrey"],
            s=_area(size),
            alpha=alpha,
            linewidths=0,
            rasterized=raster,
        )
        return ax.scatter(
            x[~bg],
            y[~bg],
            c=v[~bg],
            cmap=cmap,
            norm=norm,
            s=_area(size + 1),
            linewidths=0,
            rasterized=raster,
        )
    return ax.scatter(
        x,
        y,
        c=v,
        cmap=cmap,
        norm=norm,
        s=_area(size),
        alpha=alpha,
        linewidths=0,
        rasterized=raster,
    )


@beartype
def _axis_labels(ax: Axes, embedding: str | None) -> None:
    labels = (
        ("dim 1", "dim 2")
        if embedding is None
        else (f"{embedding} 1", f"{embedding} 2")
    )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


@beartype
def _relabel(features: Sequence[str], labels: Mapping[str, str] | None) -> list[str]:
    """Display labels, R's `feature_labels[...]` lookup.

    Raises:
        ValueError: If a feature has no label.
    """
    if labels is None:
        return list(features)
    missing = [f for f in features if f not in labels]
    if missing:
        raise ValueError(f"feature_labels has no entry for {missing}")
    return [labels[f] for f in features]


#############
# Embedding #
#############


@beartype
def embedding_plot(
    x: ScOrMc | pl.DataFrame,
    embedding: str,
    colour_by: str,
    *,
    label_by: str | None = None,
    discrete: bool | None = None,
    embd_modality: Modality = "rna",
    point_size: float | None = None,
    point_alpha: float = 0.5,
    raster: bool | None = None,
    palette: Palette | None = None,
    label_size: float = 3.0,
    label_color: str = "black",
    label_font: str = "bold",
    ax: Axes | None = None,
) -> Axes:
    """An embedding coloured by an obs column, as R's `embedding_plot_sc`.

    Args:
        x: The object, or `extract_embedding_data` output holding
            `colour_by` (and `label_by`).
        embedding: The embedding name, also the axis labels.
        colour_by: Column to colour by. Strings, Enums and booleans get the
            discrete palette in level order, numbers a colour bar.
        label_by: Column to label the group centroids by.
        discrete: Force a discrete scale.
        embd_modality: Where the embedding lives.
        point_size: Point size in mm; Seurat's auto size when `None`.
        point_alpha: Point alpha.
        raster: Rasterise the points (vector output); `None` does above
            100,000 cells.
        palette: `"main"` for discrete, `"sequential"` for continuous when
            `None`.
        label_size: Label size in mm.
        label_color: Label colour.
        label_font: `"bold"`, `"normal"` or `"italic"`.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    cols = [c for c in (colour_by, label_by) if c is not None]
    if isinstance(x, pl.DataFrame):
        frame = x
    else:
        frame = extract_embedding_data(
            x, embedding, obs_cols=cols, modality=embd_modality
        )
    if discrete:
        # numbers as an Enum in numeric order, as R's as.factor; a plain
        # string cast would sort 10 before 2
        frame = frame.with_columns(
            frame[c]
            .cast(pl.String)
            .cast(pl.Enum(frame[c].drop_nulls().unique().sort().cast(pl.String)))
            if frame[c].dtype.is_numeric()
            else frame[c].cast(pl.String)
            for c in dict.fromkeys(cols)
        )
    n_cells = (
        frame["cell_id"].n_unique() if "cell_id" in frame.columns else frame.height
    )
    raster = n_cells > RASTER_CELLS if raster is None else raster
    size = auto_point_size(n_cells, raster) if point_size is None else point_size

    ax = _new_ax(ax)
    mappable = _scatter(
        ax,
        frame,
        colour_by,
        palette=palette,
        size=size,
        alpha=point_alpha,
        raster=raster,
    )
    if mappable is not None:
        _figure(ax).colorbar(mappable, ax=ax, label=colour_by)
    if label_by is not None:
        _label_centroids(
            ax, frame, label_by, colour=label_color, size=label_size, font=label_font
        )
    _axis_labels(ax, embedding)
    return ax


@beartype
def feature_plot(
    x: ScOrMc | pl.DataFrame,
    features: Sequence[str] | None = None,
    embedding: str | None = None,
    *,
    feature_labels: Mapping[str, str] | None = None,
    scale: bool = False,
    clip: float | None = None,
    expr_modality: ExprModality = "rna",
    embd_modality: Modality = "rna",
    layer: Layer = "norm",
    point_size: float | None = None,
    point_alpha: float = 0.5,
    raster: bool | None = None,
    label_by: str | None = None,
    label_size: float = 3.0,
    label_color: str = "black",
    label_font: str = "bold",
    highlight_features: bool = False,
    highlight_quantile: float = 0.25,
    scale_mode: Literal["free", "shared"] = "free",
    palette: ContinuousPalette = "sequential",
    ncol: int | None = None,
) -> Figure:
    """Feature expression over an embedding, one panel per feature, as R's
    `feature_plot_sc`.

    With `scale_mode="free"` every panel gets its own colour bar, so a weak
    gene is not flattened by the loudest one; `"shared"` uses one colour bar
    over the pooled expression. Points are drawn low to high.

    Args:
        x: The object, or `extract_feature_plot_data` output.
        features: Feature ids from `expr_modality`. Not needed for a frame.
        embedding: The embedding name, also the axis labels.
        feature_labels: Feature id to panel title.
        scale: z-score the expression.
        clip: Clip the z-scores, with `scale`.
        expr_modality: `"rna"` or `"adt"`.
        embd_modality: `"rna"`, `"adt"` or `"wnn"`.
        layer: `"norm"` or `"magic"`.
        point_size: Point size in mm; Seurat's auto size when `None`.
        point_alpha: Point alpha.
        raster: Rasterise the points; `None` does above 100,000 cells.
        label_by: obs column to label the group centroids by.
        label_size: Label size in mm.
        label_color: Label colour.
        label_font: `"bold"`, `"normal"` or `"italic"`.
        highlight_features: Push the cells at or below `highlight_quantile`
            (per feature) into a grey background layer and draw the rest
            larger.
        highlight_quantile: The background quantile.
        scale_mode: `"free"` or `"shared"` colour bars.
        palette: The continuous palette.
        ncol: Panel columns; a square-ish grid when `None`.

    Returns:
        The figure.

    Raises:
        ValueError: Without `features` and `embedding` for an object.
    """
    if isinstance(x, pl.DataFrame):
        frame = x
    else:
        if features is None or embedding is None:
            raise ValueError("features and embedding are needed to extract the data")
        frame = extract_feature_plot_data(
            x,
            features,
            embedding,
            scale=scale,
            clip=clip,
            obs_cols=[label_by] if label_by else None,
            expr_modality=expr_modality,
            embd_modality=embd_modality,
            layer=layer,
        )
    frame = frame.sort("expression")
    genes = [g for g in frame["gene"].cast(pl.String).unique(maintain_order=True)]
    if isinstance(frame["gene"].dtype, pl.Enum):
        genes = [g for g in frame["gene"].dtype.categories if g in set(genes)]
    titles = _relabel(genes, feature_labels)
    n_cells = frame["cell_id"].n_unique()
    raster = n_cells > RASTER_CELLS if raster is None else raster
    size = auto_point_size(n_cells, raster) if point_size is None else point_size

    ncol = ncol or int(np.ceil(np.sqrt(len(genes))))
    nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4.2 * ncol, 3.6 * nrow),
        squeeze=False,
        layout="constrained",
    )
    norm = None
    if scale_mode == "shared":
        v = frame["expression"].to_numpy()
        norm = Normalize(np.nanmin(v), np.nanmax(v))
    mappable = None
    for k, (gene, title) in enumerate(zip(genes, titles, strict=True)):
        ax = axes.flat[k]
        sub = frame.filter(pl.col("gene").cast(pl.String) == gene)
        mappable = _scatter(
            ax,
            sub,
            "expression",
            palette=palette,
            size=size,
            alpha=point_alpha,
            raster=raster,
            norm=norm,
            highlight=highlight_features,
            highlight_quantile=highlight_quantile,
        )
        if scale_mode == "free" and mappable is not None:
            fig.colorbar(mappable, ax=ax)
        if label_by is not None:
            _label_centroids(
                ax, sub, label_by, colour=label_color, size=label_size, font=label_font
            )
        ax.set_title(title)
        _axis_labels(ax, embedding)
    for ax in axes.flat[len(genes) :]:
        ax.set_visible(False)
    if scale_mode == "shared" and mappable is not None:
        fig.colorbar(mappable, ax=axes.ravel().tolist(), label="expression")
    return fig


#############
# Dot plots #
#############


@beartype
def _hclust_order(mat: np.ndarray) -> list[int]:
    """Leaf order of a complete-linkage Euclidean clustering, as
    `hclust(dist(mat))$order`: singletons before clusters and earlier merges
    before later ones on each merge, ties to the first pair."""
    n = mat.shape[0]
    if n < 3:
        return list(range(n))
    d = np.sqrt(((mat[:, None, :] - mat[None, :, :]) ** 2).sum(axis=2))
    # slot -> (leaves in order, merge step, 0 for a singleton)
    slots: dict[int, tuple[list[int], int]] = {i: ([i], 0) for i in range(n)}
    dist = d.copy()
    np.fill_diagonal(dist, np.inf)
    for step in range(1, n):
        alive = sorted(slots)
        best, pair = np.inf, (alive[0], alive[1])
        for a_pos, a in enumerate(alive):
            for b in alive[a_pos + 1 :]:
                if dist[a, b] < best:
                    best, pair = dist[a, b], (a, b)
        a, b = pair
        left, right = slots.pop(a), slots.pop(b)
        if (left[1] > 0 and right[1] == 0) or (
            left[1] > 0 and right[1] > 0 and left[1] > right[1]
        ):
            left, right = right, left
        slots[a] = (left[0] + right[0], step)
        merged = np.maximum(dist[a], dist[b])
        dist[a, :], dist[:, a] = merged, merged
        dist[a, a] = np.inf
        dist[b, :], dist[:, b] = np.inf, np.inf
    return next(iter(slots.values()))[0]


@beartype
def _gene_groups(
    genes: Sequence[str], grouping: Mapping[str, str] | None
) -> tuple[list[str], list[str | None]]:
    """Genes ordered by their group's first appearance, input order within a
    group, with the group of each."""
    if grouping is None:
        return list(genes), [None] * len(genes)
    missing = [g for g in genes if g not in grouping]
    if missing:
        raise ValueError(f"feature_grouping has no entry for {missing}")
    order = list(dict.fromkeys(grouping[g] for g in genes))
    ranked = sorted(genes, key=lambda g: order.index(grouping[g]))
    return ranked, [grouping[g] for g in ranked]


@beartype
def _group_separators(ax: Axes, groups: Sequence[str | None]) -> None:
    """Thin lines between consecutive feature groups (bottom up, one per y
    unit), labelled on the right."""
    if groups[0] is None:
        return
    start = 0
    for k in range(1, len(groups) + 1):
        if k == len(groups) or groups[k] != groups[start]:
            ax.text(
                1.01,
                (start + k - 1) / 2,
                str(groups[start]),
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
            )
            if k < len(groups):
                ax.axhline(k - 0.5, color=GREY["grey80"], linewidth=0.8)
            start = k


@beartype
def dot_plot(
    x: ScOrMc | pl.DataFrame,
    features: Sequence[str] | None = None,
    grouping_variable: str | None = None,
    *,
    feature_labels: Mapping[str, str] | None = None,
    feature_grouping: Mapping[str, str] | None = None,
    scale_exp: bool = True,
    modality: ExprModality = "rna",
    cluster_groups: bool = True,
    palette: ContinuousPalette = "sequential",
    ax: Axes | None = None,
) -> Axes:
    """Marker expression across groups, colour the scaled mean and size the
    percent expressing, as R's `dot_plot_sc`.

    The first feature sits at the top. With `feature_grouping` the features
    are blocked by group (groups in order of first appearance) and the
    blocks labelled on the right.

    Args:
        x: The object, or `extract_dot_plot_data` output.
        features: Feature ids. Not needed for a frame.
        grouping_variable: obs column to group by. Not needed for a frame.
        feature_labels: Feature id to display label.
        feature_grouping: Feature id (or display label, with
            `feature_labels`) to group, e.g. a cell type.
        scale_exp: Min-max scale the means per feature.
        modality: `"rna"` or `"adt"`.
        cluster_groups: Order the groups by a complete-linkage clustering of
            their scaled expression profiles.
        palette: The continuous palette.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    if isinstance(x, pl.DataFrame):
        frame = x
    else:
        if features is None or grouping_variable is None:
            raise ValueError(
                "features and grouping_variable are needed to extract the data"
            )
        frame = extract_dot_plot_data(
            x, features, grouping_variable, scale_exp=scale_exp, modality=modality
        )
    genes = (
        list(frame["gene"].dtype.categories)
        if isinstance(frame["gene"].dtype, pl.Enum)
        else list(frame["gene"].unique(maintain_order=True))
    )
    labels = _relabel(genes, feature_labels)
    label_of = dict(zip(genes, labels, strict=True))
    groups = (
        list(frame["group"].dtype.categories)
        if isinstance(frame["group"].dtype, pl.Enum)
        else list(frame["group"].unique(maintain_order=True))
    )
    groups = [g for g in groups if g in set(frame["group"].cast(pl.String).to_list())]
    if cluster_groups and len(groups) > 1:
        wide = np.zeros((len(groups), len(genes)))
        gi = {g: i for i, g in enumerate(groups)}
        ci = {g: i for i, g in enumerate(genes)}
        for g, grp, v in frame.select(
            pl.col("gene").cast(pl.String),
            pl.col("group").cast(pl.String),
            "scaled_exp",
        ).iter_rows():
            wide[gi[grp], ci[g]] = v
        groups = [groups[i] for i in _hclust_order(wide)]

    display, gene_groups = _gene_groups(labels, feature_grouping)
    y_of = {lab: len(display) - 1 - k for k, lab in enumerate(display)}
    x_of = {g: k for k, g in enumerate(groups)}
    f = frame.with_columns(
        pl.col("gene").cast(pl.String), pl.col("group").cast(pl.String)
    )
    xs = np.array([x_of[g] for g in f["group"]])
    ys = np.array([y_of[label_of[g]] for g in f["gene"]])
    pct = f["pct_exp"].to_numpy()
    lo, hi = float(pct.min()), float(pct.max())

    def area(v: np.ndarray) -> np.ndarray:
        # ggplot's area palette over c(0, 3) mm
        r = (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)
        return (3 * np.sqrt(np.clip(r, 0, 1)) * _PT_PER_MM) ** 2

    ax = _new_ax(ax, (2.6 + 0.45 * len(groups), 1.6 + 0.3 * len(genes)))
    points = ax.scatter(
        xs,
        ys,
        s=area(pct),
        c=f["scaled_exp"].to_numpy(),
        cmap=_cmap(palette),
        linewidths=0,
    )
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
    ax.set_yticks([y_of[lab] for lab in display], display)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(-0.6, len(display) - 0.4)
    ax.set_xlabel("Group")
    _group_separators(ax, list(reversed(gene_groups)))
    cbar = _figure(ax).colorbar(
        points,
        ax=ax,
        label="Scaled\nexpression",
        shrink=0.45,
        anchor=(0.0, 1.0),
        panchor=(1.0, 1.0),
        pad=0.15 if feature_grouping else 0.04,
    )
    _legend(cbar.ax, "% expressed", *_size_handles(pct, area), where="under")
    return ax


###########
# Violins #
###########


@beartype
def _bw_nrd0(x: np.ndarray) -> float:
    """R's `bw.nrd0`, the default bandwidth of `density` and ggplot."""
    hi = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    lo = min(hi, float(q75 - q25) / 1.34)
    lo = lo or hi or abs(float(x[0])) or 1.0
    return 0.9 * lo * x.size ** (-0.2)


@beartype
def _kde(
    x: np.ndarray, a: float, b: float, m: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE on `m` points over `[a, b]` with R's bandwidth, by linear
    binning and a discrete convolution (R's `density` does the same with an
    FFT)."""
    grid = np.linspace(a, b, m)
    if b <= a:
        return grid, np.zeros(m)
    bw = _bw_nrd0(x)
    pos = (np.clip(x, a, b) - a) / (b - a) * (m - 1)
    left = np.floor(pos).astype(np.int64).clip(0, m - 2)
    frac = pos - left
    counts = np.bincount(left, 1 - frac, m) + np.bincount(left + 1, frac, m)
    step = (b - a) / (m - 1)
    offsets = np.arange(-(m - 1), m) * step
    kernel = np.exp(-0.5 * (offsets / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    dens = np.convolve(counts, kernel)[m - 1 : 2 * m - 1] / x.size
    return grid, dens


@beartype
def _violin(
    ax: Axes,
    values: np.ndarray,
    pos: int,
    colour: str,
    *,
    alpha: float,
    width: float = 0.45,
) -> None:
    """One violin, trimmed to the data and scaled to `width`, as
    `geom_violin(scale = "width")`. Constant data draws a flat line."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        ax.plot([pos - width, pos + width], [lo, lo], color="black", linewidth=0.5)
        return
    grid, dens = _kde(values, lo, hi)
    half = dens / dens.max() * width
    ax.fill_betweenx(
        grid,
        pos - half,
        pos + half,
        facecolor=colour,
        alpha=alpha,
        edgecolor="black",
        linewidth=0.4,
    )


@beartype
def stacked_violin_plot(
    x: ScOrMc | pl.DataFrame,
    features: Sequence[str] | None = None,
    grouping_variable: str | None = None,
    *,
    feature_labels: Mapping[str, str] | None = None,
    scale: bool = False,
    clip: float | None = None,
    modality: ExprModality = "rna",
    layer: Layer = "norm",
    palette: Palette = "main",
) -> Figure:
    """Expression across groups, one row of violins per feature, as R's
    `stacked_violin_plot_sc`.

    Args:
        x: The object, or `extract_gene_violin_data` output.
        features: Feature ids, one row each. Not needed for a frame.
        grouping_variable: obs column to group by. Not needed for a frame.
        feature_labels: Feature id to row label.
        scale: z-score the expression.
        clip: Clip the z-scores, with `scale`.
        modality: `"rna"` or `"adt"`.
        layer: `"norm"` or `"magic"`.
        palette: The discrete palette for the groups.

    Returns:
        The figure, one axes per feature, the first on top.
    """
    if isinstance(x, pl.DataFrame):
        frame = x
    else:
        if features is None or grouping_variable is None:
            raise ValueError(
                "features and grouping_variable are needed to extract the data"
            )
        frame = extract_gene_violin_data(
            x,
            features,
            grouping_variable,
            scale=scale,
            clip=clip,
            modality=modality,
            layer=layer,
        )
    genes = (
        list(frame["gene"].dtype.categories)
        if isinstance(frame["gene"].dtype, pl.Enum)
        else list(frame["gene"].unique(maintain_order=True))
    )
    labels = _relabel(genes, feature_labels)
    codes, groups = _factor(frame["group"])
    colours = bx_colors(palette, n=max(len(groups), 1))
    frame = frame.with_columns(pl.Series("_code", codes))

    fig, axes = plt.subplots(
        len(genes),
        1,
        sharex=True,
        figsize=(1.5 + 0.5 * len(groups), 0.9 * len(genes) + 1.2),
        squeeze=False,
        layout="constrained",
    )
    for ax, gene, label in zip(axes[:, 0], genes, labels, strict=True):
        sub = frame.filter(pl.col("gene").cast(pl.String) == gene)
        for k in range(len(groups)):
            vals = sub.filter(pl.col("_code") == k)["expression"].to_numpy()
            _violin(ax, vals, k, colours[k], alpha=0.5)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    last = axes[-1, 0]
    last.set_xticks(range(len(groups)), groups, rotation=-45, ha="left")
    last.set_xlim(-0.6, len(groups) - 0.4)
    fig.supylabel("Expression")
    return fig


############
# Heatmaps #
############


@beartype
def heatmap_plot(
    x: ScOrMc | pl.DataFrame,
    features: Sequence[str] | None = None,
    grouping_variable: str | None = None,
    *,
    feature_labels: Mapping[str, str] | None = None,
    feature_grouping: Mapping[str, str] | None = None,
    scale: bool = True,
    clip: float | None = 2.5,
    modality: ExprModality = "rna",
    max_cells_per_group: int | None = None,
    seed: int = 42,
    group_gap: float = 0.002,
    show_feature_labels: bool | None = None,
    palette: ContinuousPalette = "diverging",
    ax: Axes | None = None,
) -> Axes:
    """Genes x cells, cells blocked by group with a colour strip per group,
    as R's `heatmap_plot_sc`. Every block is one image.

    Args:
        x: The object, or `extract_gene_expression` output holding
            `grouping_variable`.
        features: Feature ids, one row each, in this order. Not needed for a
            frame.
        grouping_variable: obs column to block the cells by; Enum levels set
            the order.
        feature_labels: Feature id to row label.
        feature_grouping: Feature id (or label) to group, blocking the rows;
            groups in order of first appearance.
        scale: z-score per gene; the colour range is then centred on zero.
        clip: Clip the z-scores.
        modality: `"rna"` or `"adt"`.
        max_cells_per_group: Subsample every group to at most this many.
        seed: Seed for the subsampling.
        group_gap: Width of each gap as a fraction of the axis, `[0, 0.1]`.
        show_feature_labels: Gene names on the right; `None` shows up to 80.
        palette: The continuous palette; `"diverging"` runs blue to red.
        ax: Axes to draw on.

    Returns:
        The axes.

    Raises:
        ValueError: Without `grouping_variable`, or an out of range gap.
    """
    if grouping_variable is None:
        raise ValueError("grouping_variable is needed")
    if not 0 <= group_gap <= 0.1:
        raise ValueError("group_gap must be in [0, 0.1]")
    if isinstance(x, pl.DataFrame):
        frame = x
    else:
        if features is None:
            raise ValueError("features are needed to extract the data")
        frame = extract_gene_expression(
            x,
            features,
            obs_cols=[grouping_variable],
            scale=scale,
            clip=clip,
            modality=modality,
        )
    feats = [c for c in frame.columns if c not in ("cell_id", grouping_variable)]
    codes, levels = _factor(frame[grouping_variable])
    keep = np.flatnonzero(codes >= 0)
    if max_cells_per_group is not None:
        rng = np.random.default_rng(seed)
        keep = np.concatenate(
            [
                rng.permutation(idx)[:max_cells_per_group]
                for k in range(len(levels))
                if (idx := keep[codes[keep] == k]).size
            ]
        )
    keep = keep[np.argsort(codes[keep], kind="stable")]
    col_grp = codes[keep]

    labels = _relabel(feats, feature_labels)
    rows, row_grp_labels = _gene_groups(labels, feature_grouping)
    row_src = [feats[labels.index(lab)] for lab in rows]
    row_levels = list(dict.fromkeys(row_grp_labels)) if feature_grouping else ["all"]
    row_grp = np.array(
        [row_levels.index(g) if g is not None else 0 for g in row_grp_labels]
    )
    mat = frame.select(row_src).to_numpy()[keep].T

    finite = mat[np.isfinite(mat)]
    if scale:
        lim = float(np.abs(finite).max()) if finite.size else 1.0
        norm = Normalize(-lim, lim)
    else:
        norm = Normalize(float(finite.min()), float(finite.max()))
    cmap = _cmap(palette, reverse=palette == "diverging")

    n_rows, n_cols = mat.shape
    gap_x, gap_y = group_gap * n_cols, group_gap * n_rows
    x0 = np.arange(n_cols) + col_grp * gap_x
    y_top = -(np.arange(n_rows) + row_grp * gap_y)
    ax = _new_ax(ax, (8.0, 0.18 * n_rows + 2.0))
    col_blocks = [
        (
            k,
            x0[col_grp == k].min(),
            x0[col_grp == k].max() + 1,
            np.flatnonzero(col_grp == k),
        )
        for k in np.unique(col_grp)
    ]
    row_blocks = [
        (
            k,
            y_top[row_grp == k].min() - 1,
            y_top[row_grp == k].max(),
            np.flatnonzero(row_grp == k),
        )
        for k in np.unique(row_grp)
    ]
    for _, xmin, xmax, cidx in col_blocks:
        for _, ymin, ymax, ridx in row_blocks:
            ax.imshow(
                mat[np.ix_(ridx, cidx)],
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
                extent=(xmin, xmax, ymin, ymax),
            )
    x_max = max(b[2] for b in col_blocks)
    y_min = min(b[1] for b in row_blocks)
    strip_x, strip_y = 0.015 * x_max, 0.03 * abs(y_min)
    show_col, show_row = len(col_blocks) > 1, len(row_blocks) > 1
    strip_levels = list(
        dict.fromkeys(
            ([levels[b[0]] for b in col_blocks] if show_col else [])
            + ([row_levels[b[0]] for b in row_blocks] if show_row else [])
        )
    )
    strip_col = dict(
        zip(strip_levels, bx_colors("main", n=max(len(strip_levels), 1)), strict=False)
    )
    if show_col:
        for k, xmin, xmax, _ in col_blocks:
            ax.add_patch(
                Rectangle(
                    (xmin, strip_y * 0.5),
                    xmax - xmin,
                    strip_y,
                    color=strip_col[levels[k]],
                    lw=0,
                )
            )
            ax.text(
                (xmin + xmax) / 2,
                strip_y * 1.7,
                levels[k],
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
            )
    if show_row:
        for k, ymin, ymax, _ in row_blocks:
            ax.add_patch(
                Rectangle(
                    (-strip_x * 1.5, ymin),
                    strip_x,
                    ymax - ymin,
                    color=strip_col[row_levels[k]],
                    lw=0,
                )
            )
            ax.text(
                -strip_x * 1.7,
                (ymin + ymax) / 2,
                row_levels[k],
                ha="right",
                va="center",
                fontsize=7,
            )
    show_feature_labels = (
        n_rows <= 80 if show_feature_labels is None else show_feature_labels
    )
    if show_feature_labels:
        for lab, y in zip(rows, y_top, strict=True):
            ax.text(
                x_max + strip_x * 0.5, y - 0.5, lab, ha="left", va="center", fontsize=6
            )
    ax.set_xlim(-strip_x * 1.5 if show_row else 0, x_max)
    ax.set_ylim(y_min, strip_y * 1.5 if show_col else 0)
    ax.set_axis_off()
    _figure(ax).colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label="Scaled\nexpression" if scale else "Expression",
        fraction=0.04,
        pad=0.12 if show_feature_labels else 0.03,
    )
    return ax


#################
# Feature pairs #
#################


@beartype
def _point_density(x: np.ndarray, y: np.ndarray, smoothness: float) -> np.ndarray:
    """2D KDE at every point, R's `.compute_point_density`: `MASS::kde2d`
    (bandwidths `sd * smoothness`, 100 x 100 grid) read off bilinearly, as
    `fields::interp.surface`."""
    n = 100
    gx, gy = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hx = (np.std(x, ddof=1) * smoothness / 4) or 1.0
    hy = (np.std(y, ddof=1) * smoothness / 4) or 1.0
    z = np.zeros((n, n))
    for start in range(0, x.size, 20_000):
        xs, ys = x[start : start + 20_000], y[start : start + 20_000]
        dx = np.exp(-0.5 * ((gx[:, None] - xs[None, :]) / hx) ** 2)
        dy = np.exp(-0.5 * ((gy[:, None] - ys[None, :]) / hy) ** 2)
        z += dx @ dy.T
    z /= 2 * np.pi * x.size * hx * hy

    def cell(v: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        step = g[1] - g[0] if g[-1] > g[0] else 1.0
        pos = np.clip((v - g[0]) / step, 0, n - 1)
        i = np.minimum(pos.astype(np.int64), n - 2)
        return i, pos - i

    i, fx = cell(x, gx)
    j, fy = cell(y, gy)
    return (
        z[i, j] * (1 - fx) * (1 - fy)
        + z[i + 1, j] * fx * (1 - fy)
        + z[i, j + 1] * (1 - fx) * fy
        + z[i + 1, j + 1] * fx * fy
    )


@beartype
def feature_scatter_plot(
    x: ScOrMc | pl.DataFrame,
    feature_1: str | None = None,
    feature_2: str | None = None,
    *,
    geom: Literal["density", "hex"] = "density",
    remove_zeros: bool = True,
    smoothness: float = 10.0,
    bins: int = 60,
    modality: ExprModality = "rna",
    point_size: float = 2.5,
    point_alpha: float = 0.5,
    raster: bool | None = None,
    palette: ContinuousPalette = "viridis",
    ax: Axes | None = None,
) -> Axes:
    """Two features against each other, coloured by point density or binned
    into hexagons, as R's `feature_scatter_plot_sc`.

    Args:
        x: The object, or `extract_feature_pair` output.
        feature_1: x feature, optionally `_rna` / `_adt` suffixed. The axis
            label for a frame.
        feature_2: y feature, as `feature_1`.
        geom: `"density"` (per-point 2D KDE) or `"hex"`.
        remove_zeros: Drop cells where both are zero.
        smoothness: KDE bandwidth in standard deviations.
        bins: Hexagons across x.
        modality: Fallback modality for unsuffixed features.
        point_size: Point size in mm.
        point_alpha: Point alpha.
        raster: Rasterise the points; `None` does above 100,000 cells.
        palette: The continuous palette.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    if isinstance(x, pl.DataFrame):
        frame = x
    else:
        if feature_1 is None or feature_2 is None:
            raise ValueError("feature_1 and feature_2 are needed to extract the data")
        frame = extract_feature_pair(x, feature_1, feature_2, modality=modality)
    if remove_zeros:
        frame = frame.filter((pl.col("feature_1") > 0) | (pl.col("feature_2") > 0))
    raster = frame.height > RASTER_CELLS if raster is None else raster
    xv, yv = frame["feature_1"].to_numpy(), frame["feature_2"].to_numpy()
    ax = _new_ax(ax)
    cmap = _cmap(palette)
    if geom == "density":
        dens = _point_density(xv, yv, smoothness) if xv.size else np.zeros(0)
        mappable = ax.scatter(
            xv,
            yv,
            c=dens,
            cmap=cmap,
            # R draws these with `stroke = 0`
            s=_area(point_size, stroke=0.0),
            alpha=point_alpha,
            linewidths=0,
            rasterized=raster,
        )
        _figure(ax).colorbar(mappable, ax=ax, label="Density")
    else:
        mappable = ax.hexbin(xv, yv, gridsize=bins, cmap=cmap, mincnt=1)
        _figure(ax).colorbar(mappable, ax=ax, label="Count")
    ax.set_xlabel(feature_1 or "feature_1")
    ax.set_ylabel(feature_2 or "feature_2")
    return ax


############
# QC plots #
############


@beartype
def _qc_frame(x: CellQc | pl.DataFrame) -> pl.DataFrame:
    return get_data(x) if isinstance(x, CellQc) else x


@beartype
def _violin_panel(
    ax: Axes,
    df: pl.DataFrame,
    grouping_column: str,
    variable: str,
    *,
    outlier_column: str | None,
    group_name: str | None,
    var_name: str | None,
    log_scale: bool,
    raster: bool,
    palette: Palette,
    seed: int = 42,
) -> None:
    """R's `.plot_violin`: violins per group with the cells jittered on top,
    outliers in orange."""
    codes, groups = _factor(df[grouping_column])
    v = df[variable].cast(pl.Float64).to_numpy()
    shown = np.log10(v) if log_scale else v
    for k in range(len(groups)):
        _violin(ax, shown[codes == k], k, "white", alpha=0.6)
    rng = np.random.default_rng(seed)
    jitter = codes + rng.uniform(-0.05, 0.05, codes.size)
    if outlier_column is not None:
        out = df[outlier_column].fill_null(False).to_numpy().astype(bool)
        colours = np.where(out, OUTLIER_COLOURS[True], OUTLIER_COLOURS[False])
    else:
        colours = np.array(bx_colors(palette, n=max(len(groups), 1)))[codes]
    ok = codes >= 0
    ax.scatter(
        jitter[ok],
        shown[ok],
        c=colours[ok],
        s=_area(0.4),
        alpha=0.5,
        linewidths=0,
        rasterized=raster,
    )
    var_name = var_name or variable
    ax.set_xticks(range(len(groups)), groups, rotation=-45, ha="left")
    ax.set_xlabel(group_name or grouping_column)
    ax.set_ylabel(f"{var_name} (log10)" if log_scale else var_name)
    ax.set_title(var_name, loc="left", fontweight="bold")


@beartype
def violin_plot(
    x: CellQc | pl.DataFrame,
    grouping_column: str | None = None,
    variable: str | None = None,
    *,
    direction: Direction = "twosided",
    threshold: float = 3.0,
    group_name: str | None = None,
    var_name: str | None = None,
    log_scale: bool | None = None,
    show_outlier: bool = True,
    raster: bool | None = None,
    palette: Palette = "main",
    ax: Axes | None = None,
) -> Figure:
    """QC violins, as R's `violin_plot_sc`.

    For a `CellQc` one panel per metric, cells coloured by its final outlier
    call, grouped by the QC groups. For a frame one panel of `variable`
    across `grouping_column`, outliers recomputed by MAD within each group.

    Args:
        x: A `CellQc`, or a frame with the metric.
        grouping_column: Frame only: the x-axis groups.
        variable: Frame only: the metric.
        direction: Frame only: which tail(s) count as outliers.
        threshold: Frame only: MADs for an outlier.
        group_name: x-axis label.
        var_name: y-axis label.
        log_scale: log10 y-axis; `False` for a `CellQc`, `True` for a frame
            when `None`, as in R.
        show_outlier: Colour the cells by outlier call; otherwise by group.
        raster: Rasterise the cells; `None` does above 100,000.
        palette: Discrete palette for the groups without `show_outlier`.
        ax: Frame only: axes to draw on.

    Returns:
        The figure.

    Raises:
        ValueError: For a frame without `grouping_column` and `variable`, or
            `ax` with a `CellQc`.
    """
    df = _qc_frame(x)
    raster = df.height > RASTER_CELLS if raster is None else raster
    if isinstance(x, CellQc):
        if ax is not None:
            raise ValueError("a CellQc draws one panel per metric; ax is for a frame")
        metrics = list(x.metrics)
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(4.5 * len(metrics), 4.0),
            squeeze=False,
            layout="constrained",
        )
        for a, m in zip(axes[0], metrics, strict=True):
            _violin_panel(
                a,
                df,
                "grp",
                m,
                outlier_column=f"{m}_is_outlier" if show_outlier else None,
                group_name=group_name,
                var_name=m,
                log_scale=bool(log_scale),
                raster=raster,
                palette=palette,
            )
        return fig
    if grouping_column is None or variable is None:
        raise ValueError("grouping_column and variable are needed for a frame")
    outlier_column = None
    if show_outlier:
        flags = np.zeros(df.height, dtype=bool)
        groups = df[grouping_column].to_numpy()
        values = df[variable].cast(pl.Float64).to_numpy()
        for g in dict.fromkeys(groups.tolist()):
            idx = groups == g
            flags[idx] = per_cell_qc_outlier(
                values[idx], threshold=threshold, direction=direction
            ).outlier
        df = df.with_columns(pl.Series(".qc_outlier", flags))
        outlier_column = ".qc_outlier"
    ax = _new_ax(ax, (5.0, 4.0))
    _violin_panel(
        ax,
        df,
        grouping_column,
        variable,
        outlier_column=outlier_column,
        group_name=group_name,
        var_name=var_name,
        log_scale=True if log_scale is None else log_scale,
        raster=raster,
        palette=palette,
    )
    return _figure(ax)


@beartype
def _density_panel(
    ax: Axes,
    df: pl.DataFrame,
    grouping_column: str,
    variable: str,
    outliers: pl.DataFrame,
    *,
    var_name: str | None,
    log_scale: bool,
    adjust_position_label: float,
    palette: Palette,
) -> None:
    """R's `.plot_density`: one density per group, outlier groups named at
    their median."""
    codes, groups = _factor(df[grouping_column])
    colours = bx_colors(palette, n=max(len(groups), 1))
    v = df[variable].cast(pl.Float64).to_numpy()
    shown = np.log10(v) if log_scale else v
    ok = np.isfinite(shown)
    a, b = float(shown[ok].min()), float(shown[ok].max())
    top = 0.0
    for k in range(len(groups)):
        vals = shown[(codes == k) & ok]
        if vals.size < 2:
            continue
        grid, dens = _kde(vals, a, b)
        xs = 10**grid if log_scale else grid
        ax.fill_between(xs, dens, color=colours[k], alpha=0.2, linewidth=0)
        ax.plot(xs, dens, color="black", linewidth=0.5)
        top = max(top, float(dens.max()))
    if log_scale:
        ax.set_xscale("log")
    colour_of = dict(zip(groups, colours, strict=True))
    for gid, med in outliers.select("group_id", "group_median").iter_rows():
        ax.text(
            med + adjust_position_label,
            top,
            str(gid),
            color=colour_of.get(str(gid), "black"),
            alpha=0.4,
            fontsize=3 * _PT_PER_MM,
            ha="center",
            va="top",
        )
    var_name = var_name or variable
    ax.set_xlabel(f"{var_name} (log10)" if log_scale else var_name)
    ax.set_ylabel("Density")
    ax.set_title(var_name, loc="left", fontweight="bold")


@beartype
def density_plot(
    x: CellQc | pl.DataFrame,
    grouping_column: str | None = None,
    variable: str | None = None,
    *,
    direction: Direction = "twosided",
    threshold: float = 3.0,
    var_name: str | None = None,
    log_scale: bool | None = None,
    adjust_position_label: float = 0.0,
    palette: Palette = "main",
    ax: Axes | None = None,
) -> Figure:
    """QC densities per group with the outlier groups named, as R's
    `density_plot_sc`.

    For a `CellQc` (grouped, so `per_group_stats` exists) one panel per
    metric with the group calls it holds. For a frame one panel, outlier
    groups recomputed by MAD over the group medians.

    Args:
        x: A grouped `CellQc`, or a frame with the metric.
        grouping_column: Frame only: the groups.
        variable: Frame only: the metric.
        direction: Frame only: which tail(s) count as outliers.
        threshold: Frame only: MADs for an outlier.
        var_name: x-axis label.
        log_scale: log10 x-axis (densities on the log scale); `False` for a
            `CellQc`, `True` for a frame when `None`, as in R.
        adjust_position_label: x offset of the labels.
        palette: Discrete palette for the groups and labels.
        ax: Frame only: axes to draw on.

    Returns:
        The figure.

    Raises:
        ValueError: On an ungrouped `CellQc`, or a frame without
            `grouping_column` and `variable`.
    """
    df = _qc_frame(x)
    if isinstance(x, CellQc):
        if x.per_group_stats is None:
            raise ValueError(
                "Density QC requires grouped data; `per_group_stats` is None."
            )
        if ax is not None:
            raise ValueError("a CellQc draws one panel per metric; ax is for a frame")
        metrics = list(x.metrics)
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(4.5 * len(metrics), 3.6),
            squeeze=False,
            layout="constrained",
        )
        stats = x.per_group_stats
        for a, m in zip(axes[0], metrics, strict=True):
            out = stats.filter((pl.col("metric") == m) & pl.col("is_outlier")).select(
                group_id=pl.col("group"), group_median=pl.col("group_median")
            )
            _density_panel(
                a,
                df,
                "grp",
                m,
                out,
                var_name=m,
                log_scale=bool(log_scale),
                adjust_position_label=adjust_position_label,
                palette=palette,
            )
        return fig
    if grouping_column is None or variable is None:
        raise ValueError("grouping_column and variable are needed for a frame")
    medians = df.group_by(grouping_column, maintain_order=True).agg(
        group_median=pl.col(variable).median()
    )
    res = per_cell_qc_outlier(
        medians["group_median"].to_numpy(), threshold=threshold, direction=direction
    )
    out = medians.filter(pl.Series(res.outlier)).select(
        group_id=pl.col(grouping_column).cast(pl.String), group_median="group_median"
    )
    log = True if log_scale is None else log_scale
    ax = _new_ax(ax, (5.0, 3.6))
    _density_panel(
        ax,
        df,
        grouping_column,
        variable,
        out,
        var_name=var_name,
        log_scale=log,
        adjust_position_label=adjust_position_label,
        palette=palette,
    )
    return _figure(ax)


@beartype
def joint_plot(
    x: CellQc | pl.DataFrame,
    library_size: str | None = None,
    nb_features: str | None = None,
    *,
    log_scale: bool = False,
    palette: ContinuousPalette = "sequential",
) -> Figure:
    """Genes per cell against UMIs per cell, hexbinned with marginal
    histograms, as R's `joint_plot_sc`.

    Args:
        x: A `CellQc`, or a frame with both columns.
        library_size: Library size column; `"log10_lib_size"` for a
            `CellQc` and `"lib_size"` for a frame when `None`.
        nb_features: Genes per cell column; `"log10_nnz"` / `"nnz"`.
        log_scale: log10 both axes first.
        palette: The continuous palette; the histograms take its middle.

    Returns:
        The figure: the hexbin and the two marginal histograms.
    """
    df = _qc_frame(x)
    is_qc = isinstance(x, CellQc)
    library_size = library_size or ("log10_lib_size" if is_qc else "lib_size")
    nb_features = nb_features or ("log10_nnz" if is_qc else "nnz")
    missing = [c for c in (library_size, nb_features) if c not in df.columns]
    if missing:
        raise ValueError(f"column(s) not found: {missing}")
    xv = df[nb_features].cast(pl.Float64).to_numpy()
    yv = df[library_size].cast(pl.Float64).to_numpy()
    if log_scale:
        xv, yv = np.log10(xv), np.log10(yv)

    fig = plt.figure(figsize=(5.5, 5.5))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
    )
    main = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=main)
    side = fig.add_subplot(grid[1, 1], sharey=main)
    main.hexbin(xv, yv, gridsize=80, cmap=_cmap(palette), mincnt=1)
    fill = bx_colors(palette, n=5)[2]
    top.hist(xv, bins=50, color=fill, edgecolor="black", linewidth=0.3)
    side.hist(
        yv,
        bins=50,
        color=fill,
        edgecolor="black",
        linewidth=0.3,
        orientation="horizontal",
    )
    top.set_axis_off()
    side.set_axis_off()

    def logged(col: str) -> bool:
        return log_scale or col.startswith("log10")

    main.set_xlabel("Genes per cell" + (" (log10)" if logged(nb_features) else ""))
    main.set_ylabel("UMIs per cell" + (" (log10)" if logged(library_size) else ""))
    return fig


###############
# Graph plots #
###############


@beartype
def _cells_layer(
    ax: Axes,
    sc: ScOrMc,
    embedding: str,
    modality: Modality,
    *,
    colour: str,
    size: float | None,
    alpha: float,
    raster: bool | None,
) -> None:
    """The cells as a uniform grey layer underneath a graph."""
    cells = extract_embedding_data(sc, embedding, modality=modality)
    n = cells.height
    raster = n > RASTER_CELLS if raster is None else raster
    size = auto_point_size(n, raster) if size is None else size
    ax.scatter(
        cells["dim_1"].to_numpy(),
        cells["dim_2"].to_numpy(),
        c=colour,
        s=_area(size),
        alpha=alpha,
        linewidths=0,
        rasterized=raster,
        zorder=1,
    )


@beartype
def _segments(
    ax: Axes,
    edges: pl.DataFrame,
    colour: str,
    widths: tuple[float, float],
    alpha: float,
) -> tuple[list[Line2D], list[str]]:
    """Edges with linewidth scaled linearly from weight, as
    `scale_linewidth(range = widths)`, with legend keys."""
    if edges.height == 0:
        return [], []
    w = edges["weight"].to_numpy()
    lo, hi = float(w.min()), float(w.max())

    def width(v: np.ndarray) -> np.ndarray:
        r = (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5)
        return (widths[0] + r * (widths[1] - widths[0])) * _PT_PER_MM

    segs = np.stack(
        [edges.select("x", "y").to_numpy(), edges.select("xend", "yend").to_numpy()],
        axis=1,
    )
    ax.add_collection(
        LineCollection(
            list(segs),
            linewidths=width(w),
            colors=colour,
            alpha=alpha,
            capstyle="round",
            zorder=2,
        )
    )
    ticks = np.asarray(MaxNLocator(4).tick_values(lo, hi))
    ticks = ticks[(ticks >= lo) & (ticks <= hi)]
    return (
        [Line2D([], [], color=colour, linewidth=float(lw)) for lw in width(ticks)],
        [f"{t:g}" for t in ticks],
    )


@beartype
def paga_plot(
    sc: ScOrMc,
    paga_res: PagaRes,
    *,
    embedding: str = "umap",
    threshold: float = 0.01,
    tree_only: bool = False,
    node_colour_by: str | None = None,
    show_cells: bool = True,
    label: bool = True,
    embd_modality: Modality = "rna",
    centroid: Literal["median", "mean"] = "median",
    point_size: float | None = None,
    point_alpha: float = 0.4,
    raster: bool | None = None,
    cell_colour: str = GREY["grey80"],
    edge_colour: str = GREY["grey40"],
    edge_width: tuple[float, float] = (0.2, 3.0),
    max_node_size: float = 12.0,
    palette: Palette | None = None,
    label_size: float = 3.0,
    label_color: str = "black",
    label_font: str = "bold",
    ax: Axes | None = None,
) -> Axes:
    """The PAGA graph over an embedding, every cluster at the centroid of
    its cells, as R's `paga_plot_sc`.

    Node area tracks the cell count, edge width the connectivity.

    Args:
        sc: The object PAGA ran on.
        paga_res: The `run_paga` result.
        embedding: The embedding to place the nodes in.
        threshold: Edges below this connectivity are dropped.
        tree_only: Draw the maximum spanning forest.
        node_colour_by: A numeric obs column summarised per cluster, e.g.
            the pseudotime; the nodes are coloured by cluster otherwise.
        show_cells: Draw the cells underneath.
        label: Label the nodes.
        embd_modality: Where the embedding lives.
        centroid: How a cluster's position is summarised.
        point_size: Cell size in mm; Seurat's auto size when `None`.
        point_alpha: Cell alpha.
        raster: Rasterise the cells; `None` does above 100,000.
        cell_colour: Colour of the cells.
        edge_colour: Colour of the edges.
        edge_width: Edge width range in mm.
        max_node_size: Diameter of the largest node in mm.
        palette: `"main"` for clusters, `"sequential"` for `node_colour_by`
            when `None`.
        label_size: Label size in mm.
        label_color: Label colour.
        label_font: `"bold"`, `"normal"` or `"italic"`.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    data = extract_paga_plot_data(
        sc,
        paga_res,
        embedding=embedding,
        threshold=threshold,
        tree_only=tree_only,
        centroid=centroid,
        node_stat_col=node_colour_by,
        modality=embd_modality,
    )
    nodes = data.nodes
    ax = _new_ax(ax)
    if show_cells:
        _cells_layer(
            ax,
            sc,
            embedding,
            embd_modality,
            colour=cell_colour,
            size=point_size,
            alpha=point_alpha,
            raster=raster,
        )
    edge_h, edge_l = _segments(ax, data.edges, edge_colour, edge_width, 0.8)

    n_max = float(np.max(nodes["n_cells"].to_numpy()))

    def area(v: np.ndarray) -> np.ndarray:
        return (max_node_size * np.sqrt(v / n_max) * _PT_PER_MM) ** 2

    n_cells = nodes["n_cells"].to_numpy().astype(np.float64)
    xy = nodes.select("dim_1", "dim_2").to_numpy()
    if node_colour_by is None:
        dtype = nodes["cluster"].dtype
        assert isinstance(dtype, pl.Enum)
        levels = dtype.categories.to_list()
        all_cols = bx_colors(palette or "main", n=max(len(levels), 1))
        idx = nodes["cluster"].to_physical().to_numpy()
        fill = [all_cols[i] for i in idx]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=area(n_cells),
            c=fill,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        present = nodes["cluster"].cast(pl.String).to_list()
        _legend(ax, "Cluster", _dot_handles(fill), present)
    else:
        mappable = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=area(n_cells),
            c=nodes["stat"].to_numpy(),
            cmap=_cmap(palette or "sequential"),
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        _figure(ax).colorbar(mappable, ax=ax, label=node_colour_by)
    size_h, size_l = _size_handles(n_cells, area)
    _key_rows(ax, [("Cells", size_h, size_l), ("Connectivity", edge_h, edge_l)])
    if label:
        for name, x0, y0 in zip(
            nodes["cluster"].cast(pl.String), xy[:, 0], xy[:, 1], strict=True
        ):
            ax.annotate(
                name,
                (x0, y0),
                xytext=(6, 6),
                textcoords="offset points",
                color=label_color,
                fontsize=label_size * _PT_PER_MM,
                fontweight="bold" if label_font == "bold" else "normal",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "alpha": 0.8,
                    "lw": 0,
                },
                zorder=4,
            )
    _axis_labels(ax, embedding)
    ax.autoscale_view()
    return ax


@beartype
def milo_nhood_plot(
    sc: ScOrMc,
    milo_res: MiloR,
    *,
    embedding: str = "umap",
    alpha: float = 0.1,
    overlap: int = 1,
    colour_by: Literal["logFC", "majority_celltype"] = "logFC",
    show_cells: bool = True,
    embd_modality: Modality = "rna",
    point_size: float | None = None,
    point_alpha: float = 0.4,
    raster: bool | None = None,
    cell_colour: str = GREY["grey90"],
    edge_colour: str = GREY["grey50"],
    edge_width: tuple[float, float] = (0.1, 1.5),
    size_range: tuple[float, float] = (0.5, 4.0),
    palette: Palette | None = None,
    ax: Axes | None = None,
) -> Axes:
    """The Milo neighbourhood graph over an embedding, as R's
    `milo_nhood_plot_sc` and Milo's `plotNhoodGraphDA`.

    With `colour_by="logFC"` the significant neighbourhoods are filled by
    logFC on a range centred on zero, drawn on top with the strongest last;
    the rest stay white underneath.

    Args:
        sc: The object the Milo result was built on.
        milo_res: A tested `MiloR`.
        embedding: The embedding to place the nodes in.
        alpha: SpatialFDR threshold for significance.
        overlap: Minimum shared cells for an edge.
        colour_by: `"logFC"`, or `"majority_celltype"` after
            `add_nhoods_info`.
        show_cells: Draw the cells underneath.
        embd_modality: Where the embedding lives.
        point_size: Cell size in mm; Seurat's auto size when `None`.
        point_alpha: Cell alpha.
        raster: Rasterise the cells; `None` does above 100,000.
        cell_colour: Colour of the cells.
        edge_colour: Colour of the edges.
        edge_width: Edge width range in mm.
        size_range: Node diameter range in mm.
        palette: `"diverging"` (red enriched) for logFC, `"main"` for cell
            types when `None`.
        ax: Axes to draw on.

    Returns:
        The axes.

    Raises:
        ValueError: On `"majority_celltype"` before `add_nhoods_info`.
    """
    data = extract_milo_plot_data(
        sc,
        milo_res,
        embedding=embedding,
        alpha=alpha,
        overlap=overlap,
        modality=embd_modality,
    )
    nodes = data.nodes
    if colour_by not in nodes.columns:
        raise ValueError("No `majority_celltype` found. Run add_nhoods_info() first.")
    ax = _new_ax(ax)
    if show_cells:
        _cells_layer(
            ax,
            sc,
            embedding,
            embd_modality,
            colour=cell_colour,
            size=point_size,
            alpha=point_alpha,
            raster=raster,
        )
    edge_h, edge_l = _segments(ax, data.edges, edge_colour, edge_width, 0.2)

    size = nodes["size"].to_numpy().astype(np.float64)
    lo, hi = float(np.min(size)), float(np.max(size))

    def area(v: np.ndarray) -> np.ndarray:
        # ggplot's scale_size: area palette over `size_range`
        r = (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)
        d = size_range[0] + np.sqrt(np.clip(r, 0, 1)) * (size_range[1] - size_range[0])
        return (d * _PT_PER_MM) ** 2

    edge, stroke = GREY["grey30"], 0.2 * _PT_PER_MM
    if colour_by == "logFC":
        order = np.lexsort(
            (nodes["logFC"].abs().to_numpy(), nodes["is_sig"].to_numpy())
        )
        nodes = nodes[order]
        sig = nodes["is_sig"].to_numpy()
        lfc = nodes["logFC"].to_numpy()
        xy = nodes.select("dim_1", "dim_2").to_numpy()
        s = area(nodes["size"].to_numpy().astype(np.float64))
        ax.scatter(
            xy[~sig, 0],
            xy[~sig, 1],
            s=s[~sig],
            c="white",
            edgecolors=edge,
            linewidths=stroke,
            zorder=3,
        )
        lim = float(np.nanmax(np.abs(lfc[sig]))) if sig.any() else 0.0
        lim = lim if lim > 0 else 1.0
        pal = palette or "diverging"
        mappable = ax.scatter(
            xy[sig, 0],
            xy[sig, 1],
            s=s[sig],
            c=lfc[sig],
            cmap=_cmap(pal, reverse=pal == "diverging"),
            norm=Normalize(-lim, lim),
            edgecolors=edge,
            linewidths=stroke,
            zorder=3,
        )
        _figure(ax).colorbar(mappable, ax=ax, label="logFC")
    else:
        codes, levels = _factor(nodes["majority_celltype"])
        cols = np.array([*bx_colors(palette or "main", n=max(len(levels), 1)), "white"])
        xy = nodes.select("dim_1", "dim_2").to_numpy()
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=area(size),
            c=cols[codes],
            edgecolors=edge,
            linewidths=stroke,
            zorder=3,
        )
        _legend(
            ax, "Majority cell type", _dot_handles(cols[: len(levels)].tolist()), levels
        )
    size_h, size_l = _size_handles(size, area)
    _key_rows(ax, [("Nhood size", size_h, size_l), ("Shared cells", edge_h, edge_l)])
    _axis_labels(ax, embedding)
    ax.autoscale_view()
    return ax
