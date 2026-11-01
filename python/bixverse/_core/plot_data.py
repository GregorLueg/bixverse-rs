"""Plot data extractors, the port of the bixverse R `extract_*` helpers.

Each pulls what one plot needs into a polars frame (or two, for the graph
plots) and does nothing else, so a plot can be redrawn, restyled or drawn
with another library from the same frame. None of this needs matplotlib.

Rows follow the cells results align to: the kept cells in `cells_to_keep`
order, or the meta cells in row order. Factor-like columns (`gene`, `group`,
`cluster`) are polars Enums whose categories carry R's level order.

`scale=True` z-scores by the sample standard deviation on every path: in Rust
on the store (the RNA `"norm"` layer of `SingleCells` / `SingleCellsSubset`),
here on in-memory matrices (ADT, MAGIC, meta cells).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.de import MiloR, get_index_cells
from bixverse._core.metacells import MetaCells
from bixverse._core.multimodal import (
    Modality,
    SingleCellsMultiModal,
    adt_rows,
    check_modality,
    modality_cache,
    require_adt,
)
from bixverse._core.state import CsrMatrix
from bixverse._core.subset import ScOrMc
from bixverse._core.traj import PagaRes

###########
# Globals #
###########

ExprModality = Literal["rna", "adt"]
Layer = Literal["norm", "magic"]


@beartype
@dataclass(frozen=True)
class GraphPlotData:
    """A graph laid out on an embedding: the PAGA or Milo node-link plot.

    Attributes:
        nodes: One row per node with its position `dim_1`, `dim_2`.
        edges: `from`, `to`, `weight` and the segment ends `x`, `y`, `xend`,
            `yend`, each edge once.
        embedding: The embedding the nodes sit in, for the axis labels.
    """

    nodes: pl.DataFrame
    edges: pl.DataFrame
    embedding: str


###########
# Helpers #
###########


@beartype
def _row_ids(sc: ScOrMc) -> np.ndarray:
    """Ids of the rows every extract aligns to."""
    if isinstance(sc, MetaCells):
        return sc.cell_ids()
    return sc.cell_ids(filtered=True)


@beartype
def _obs_rows(sc: ScOrMc, cols: Sequence[str]) -> pl.DataFrame:
    """obs columns in row order: kept cells in `cells_to_keep` order.

    R attaches them positionally in obs order, which only agrees while the
    kept cells are sorted.
    """
    cols = list(dict.fromkeys(cols))
    if isinstance(sc, MetaCells):
        return sc.obs(cols=cols)
    obs = sc.obs(cols=list(dict.fromkeys(["cell_idx", *cols])), filtered=True)
    order = pl.DataFrame({"cell_idx": sc.cells_to_keep}).cast(obs.schema["cell_idx"])
    return order.join(obs, on="cell_idx", how="left", maintain_order="left").select(
        cols
    )


@beartype
def _with_obs(
    frame: pl.DataFrame, sc: ScOrMc, cols: Sequence[str] | None
) -> pl.DataFrame:
    """`frame` with obs columns joined on `cell_id`, its row order kept."""
    if not cols:
        return frame
    cols = [c for c in dict.fromkeys(cols) if c != "cell_id"]
    obs = pl.concat(
        [pl.DataFrame({"cell_id": _row_ids(sc)}), _obs_rows(sc, cols)],
        how="horizontal",
    )
    return frame.drop([c for c in cols if c in frame.columns]).join(
        obs, on="cell_id", how="left", maintain_order="left"
    )


@beartype
def _r_labels(values: pl.Series) -> list[str | None]:
    """R's `as.character`: `TRUE` / `FALSE`, floats to 15 significant digits."""
    dtype = values.dtype
    raw = values.to_list()
    if dtype == pl.Boolean:
        return [None if v is None else ("TRUE" if v else "FALSE") for v in raw]
    if dtype.is_float():
        return [None if v is None else f"{v:.15g}" for v in raw]
    return [None if v is None else str(v) for v in raw]


@beartype
def _as_factor(values: pl.Series) -> tuple[np.ndarray, list[str]]:
    """R's `as.factor`: 0-based codes (-1 for missing) and the levels.

    An Enum keeps its declared level order, unused levels included; anything
    else is sorted (numerically for numbers, by code point for strings, where
    R uses the locale's collation).
    """
    labels = _r_labels(values)
    if isinstance(values.dtype, pl.Enum):
        levels = [str(c) for c in values.dtype.categories.to_list()]
    elif values.dtype.is_numeric() or values.dtype == pl.Boolean:
        uniq = _r_labels(values.drop_nulls().unique().sort())
        levels = [v for v in uniq if v is not None]
    else:
        levels = sorted({v for v in labels if v is not None})
    lookup = {v: i for i, v in enumerate(levels)}
    codes = np.asarray([-1 if v is None else lookup[v] for v in labels], dtype=np.int64)
    return codes, levels


@beartype
def _drop_levels(codes: np.ndarray, levels: list[str]) -> tuple[np.ndarray, list[str]]:
    """R's `droplevels`: recode to the levels that occur, in level order."""
    used = np.unique(codes[codes >= 0])
    remap = np.full(len(levels), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return np.where(codes >= 0, remap[np.maximum(codes, 0)], -1), [
        levels[i] for i in used
    ]


@beartype
def _enum(values: Sequence[str | None], levels: Sequence[str], name: str) -> pl.Series:
    """A factor as a polars Enum."""
    return pl.Series(name, list(values), dtype=pl.Enum(list(levels)))


@beartype
def _get_embedding(sc: ScOrMc, name: str, modality: Modality) -> np.ndarray:
    """A cached embedding by name, `"pca"` for the PCA scores, as R's
    `get_embedding`: a stale one warns.

    Raises:
        ValueError: If it is missing, or on a non-RNA modality of a single
            modality object.
    """
    cache = modality_cache(sc, modality)
    embd = cache.pca_factors if name == "pca" else cache.embeddings.get(name)
    if embd is None:
        raise ValueError(
            f"The embedding '{name}' ({modality}) was not found. Please verify "
            "which functions you have run."
        )
    if isinstance(sc, SingleCellsMultiModal):
        state = sc.check_state([name], modality=modality)
    else:
        state = sc.check_state([name])
    if isinstance(state, str):
        warnings.warn(state, stacklevel=3)
    return embd


@beartype
def _match_features(features: Sequence[str], available: Sequence[str]) -> list[str]:
    """R's `.match_features`: unknown ids warn and drop, input order kept.

    Raises:
        ValueError: If nothing matched.
    """
    present = set(available)
    hits = [f for f in features if f in present]
    if len(hits) < len(features):
        warnings.warn(
            f"{len(features) - len(hits)} features could not be matched.",
            stacklevel=4,
        )
    if not hits:
        raise ValueError(
            "No features matched. Please double check provided parameters!"
        )
    return hits


@beartype
def _scale_in_memory(v: np.ndarray, clip: float | None) -> np.ndarray:
    """R's `.scale_and_clip_expr`: z-score by the sample sd, then clip."""
    s = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    if s > 1e-8:
        v = (v - v.mean()) / s
    if clip is not None:
        v = np.clip(v, -clip, clip)
    return v


@beartype
def _dense_cols(m: CsrMatrix) -> np.ndarray:
    """A CSR matrix as a dense `float64` array."""
    dense = np.zeros(m.shape, dtype=np.float64)
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr.astype(np.int64)))
    dense[rows, m.indices.astype(np.int64)] = m.data
    return dense


@beartype
def _in_memory(
    sc: ScOrMc, features: Sequence[str], modality: ExprModality, layer: Layer
) -> tuple[np.ndarray, list[str]] | None:
    """The dense `float64` rows x features matrix of the in-memory sources
    (ADT counts, the MAGIC layer, meta cells), or `None` for the store.

    Raises:
        ValueError: On MAGIC for meta cells or ADT, or a missing layer.
    """
    if modality == "adt":
        assert isinstance(sc, SingleCellsMultiModal)
        if layer == "magic":
            raise ValueError(
                "The imputed layer holds RNA counts. Use modality = 'rna' or "
                "layer = 'norm'."
            )
        adt = require_adt(sc)
        feats = _match_features(features, adt.feature_ids.tolist())
        pos = {f: i for i, f in enumerate(adt.feature_ids.tolist())}
        rows = adt_rows(sc, None, True)
        return adt.norm_counts[np.ix_(rows, [pos[f] for f in feats])], feats
    if isinstance(sc, MetaCells):
        if layer == "magic":
            raise ValueError(
                "MAGIC is not available for MetaCells, which are already "
                "aggregated. Use layer = 'norm'."
            )
        feats = _match_features(features, sc.gene_ids().tolist())
        return _dense_cols(sc.counts(genes=feats, layer="norm")), feats
    if layer == "magic":
        magic = sc.cache.magic
        if magic is None:
            raise ValueError(
                "No imputed layer found. Run run_magic() first, or use layer = 'norm'."
            )
        state = sc.check_state(["magic"])
        if isinstance(state, str):
            warnings.warn(state, stacklevel=4)
        held = magic.features.tolist()
        feats = _match_features(features, held)
        pos = {f: i for i, f in enumerate(held)}
        return magic.data[:, [pos[f] for f in feats]].astype(np.float64), feats
    return None


@beartype
def _store_genes(sc: ScOrMc, features: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """0-based gene indices and the surviving ids, R's `.resolve_gene_features`.

    Raises:
        ValueError: If nothing matched.
    """
    idx = sc.gene_indices(list(features))
    return idx, sc.gene_ids(idx).tolist()


#############
# Embedding #
#############


@beartype
def extract_embedding_data(
    sc: ScOrMc,
    embedding: str,
    *,
    obs_cols: Sequence[str] | None = None,
    modality: Modality = "rna",
) -> pl.DataFrame:
    """Embedding coordinates for plotting, as R's `extract_embedding_data`.

    Args:
        sc: The experiment, a subset or meta cells.
        embedding: The embedding name, `"pca"` for the PCA scores.
        obs_cols: obs columns to attach.
        modality: Where the embedding lives: `"rna"`, or `"adt"` / `"wnn"`
            on a `SingleCellsMultiModal`.

    Returns:
        `cell_id`, `dim_1` ... `dim_k` (`float64`) and the obs columns, one
        row per kept cell (or meta cell).

    Raises:
        ValueError: If the embedding is missing.
    """
    embd = _get_embedding(sc, embedding, modality)
    frame = pl.DataFrame(
        {"cell_id": _row_ids(sc)}
        | {f"dim_{i + 1}": embd[:, i].astype(np.float64) for i in range(embd.shape[1])}
    )
    return _with_obs(frame, sc, obs_cols)


##############
# Expression #
##############


@beartype
def extract_gene_expression(
    sc: ScOrMc,
    features: Sequence[str],
    *,
    obs_cols: Sequence[str] | None = None,
    scale: bool = False,
    clip: float | None = None,
    modality: ExprModality = "rna",
    layer: Layer = "norm",
) -> pl.DataFrame:
    """Per-cell expression, one column per feature, as R's
    `extract_gene_expression`.

    RNA from the store reads the gene-major file in Rust; ADT counts, the
    MAGIC layer and meta cells come from memory. Unknown features warn and
    drop.

    Args:
        sc: The experiment, a subset or meta cells.
        features: Gene ids, or ADT feature ids for `"adt"`.
        obs_cols: obs columns to attach.
        scale: z-score each feature across the rows.
        clip: Clip the z-scores to `[-clip, clip]`, with `scale`.
        modality: `"rna"`, or `"adt"` on a `SingleCellsMultiModal`.
        layer: `"norm"`, or `"magic"` for the imputed layer of `run_magic`.

    Returns:
        `cell_id`, one `float64` column per matched feature, then the obs
        columns.

    Raises:
        ValueError: If nothing matched, on ADT without a multi-modal object
            or ADT counts, or a missing MAGIC layer.
    """
    if clip is not None and clip <= 0:
        raise ValueError("clip must be positive")
    check_modality(sc, modality)
    held = _in_memory(sc, features, modality, layer)
    if held is None:
        assert not isinstance(sc, MetaCells)
        genes, feats = _store_genes(sc, features)
        values = np.asarray(
            _bixverse.extract_genes_dense(
                str(sc.genes_path),
                np.ascontiguousarray(sc.cells_to_keep, dtype=np.uint32),
                np.ascontiguousarray(genes, dtype=np.uint32),
                scale,
                clip,
            ),
            dtype=np.float64,
        )
        columns = {f: values[i] for i, f in enumerate(feats)}
    else:
        mat, feats = held
        columns = {
            f: _scale_in_memory(mat[:, i], clip) if scale else mat[:, i]
            for i, f in enumerate(feats)
        }
    frame = pl.DataFrame({"cell_id": _expr_ids(sc, modality)} | columns)
    return _with_obs(frame, sc, obs_cols)


@beartype
def _expr_ids(sc: ScOrMc, modality: ExprModality) -> np.ndarray:
    """Ids of the expression rows: the ADT matrix can miss kept cells."""
    if modality == "adt":
        assert isinstance(sc, SingleCellsMultiModal)
        return require_adt(sc).cell_ids[adt_rows(sc, None, True)]
    return _row_ids(sc)


@beartype
def _groups_for(
    sc: ScOrMc, ids: np.ndarray, grouping_variable: str
) -> tuple[np.ndarray, list[str]]:
    """Group codes for the rows `ids` (-1 when missing) and the levels.

    Raises:
        ValueError: If the column is entirely missing.
    """
    col = _with_obs(pl.DataFrame({"cell_id": ids}), sc, [grouping_variable])
    codes, levels = _as_factor(col[grouping_variable])
    if (codes < 0).all():
        raise ValueError(f"Grouping variable `{grouping_variable}` is entirely NA.")
    return codes, levels


@beartype
def extract_dot_plot_data(
    sc: ScOrMc,
    features: Sequence[str],
    grouping_variable: str,
    *,
    scale_exp: bool = True,
    modality: ExprModality = "rna",
) -> pl.DataFrame:
    """Per-group mean expression and percent expressing, as R's
    `extract_dot_plot_data`.

    Means run over every cell of the group, zeros included. Cells with a
    missing group are left out and only groups that occur are kept, on every
    source (R keeps empty Enum levels on the in-memory ones, which turns their
    scaled values into NA).

    Args:
        sc: The experiment, a subset or meta cells.
        features: Gene ids, or ADT feature ids for `"adt"`.
        grouping_variable: obs column to group by.
        scale_exp: Min-max scale the means per feature into `scaled_exp`.
        modality: `"rna"`, or `"adt"` on a `SingleCellsMultiModal`.

    Returns:
        `gene` and `group` (Enums, in feature and level order), `mean_exp`,
        `pct_exp` (0 to 100) and `scaled_exp`, gene-major.

    Raises:
        ValueError: If nothing matched or the grouping is entirely missing.
    """
    check_modality(sc, modality)
    codes, levels = _groups_for(sc, _expr_ids(sc, modality), grouping_variable)
    keep = codes >= 0
    codes, levels = _drop_levels(codes[keep], levels)
    n_groups = len(levels)
    held = _in_memory(sc, features, modality, "norm")
    if held is None:
        assert not isinstance(sc, MetaCells)
        genes, feats = _store_genes(sc, features)
        mean, frac = _bixverse.grouped_gene_stats(
            str(sc.genes_path),
            np.ascontiguousarray(sc.cells_to_keep[keep], dtype=np.uint32),
            np.ascontiguousarray(genes, dtype=np.uint32),
            np.ascontiguousarray(codes, dtype=np.uint32),
            n_groups,
        )
        mean = np.asarray(mean, dtype=np.float64)
        pct = np.asarray(frac, dtype=np.float64) * 100
    else:
        mat, feats = held
        mat = mat[keep]
        sizes = np.bincount(codes, minlength=n_groups)
        mean = np.stack(
            [np.bincount(codes, mat[:, j], n_groups) / sizes for j in range(len(feats))]
        )
        pct = 100 * np.stack(
            [
                np.bincount(codes, mat[:, j] > 0, n_groups) / sizes
                for j in range(len(feats))
            ]
        )
    if scale_exp:
        lo = mean.min(axis=1, keepdims=True)
        span = mean.max(axis=1, keepdims=True) - lo
        scaled = np.where(span == 0, 0.0, (mean - lo) / np.where(span == 0, 1, span))
    else:
        scaled = mean
    return pl.DataFrame(
        [
            _enum(np.repeat(feats, n_groups).tolist(), feats, "gene"),
            _enum(levels * len(feats), levels, "group"),
            pl.Series("mean_exp", mean.ravel()),
            pl.Series("pct_exp", pct.ravel()),
            pl.Series("scaled_exp", scaled.ravel()),
        ]
    )


@beartype
def _melt(
    frame: pl.DataFrame, id_cols: Sequence[str], features: Sequence[str]
) -> pl.DataFrame:
    """Wide to long, gene-major, `gene` an Enum in feature order."""
    long = frame.unpivot(
        on=list(features),
        index=list(id_cols),
        variable_name="gene",
        value_name="expression",
    )
    return long.with_columns(pl.col("gene").cast(pl.Enum(list(features))))


@beartype
def extract_feature_plot_data(
    sc: ScOrMc,
    features: Sequence[str],
    embedding: str,
    *,
    scale: bool = False,
    clip: float | None = None,
    obs_cols: Sequence[str] | None = None,
    expr_modality: ExprModality = "rna",
    embd_modality: Modality = "rna",
    layer: Layer = "norm",
) -> pl.DataFrame:
    """Per-cell expression on an embedding, long, as R's
    `extract_feature_plot_data`.

    The expression and the embedding modality are chosen independently, so
    RNA can colour an ADT embedding and either can go on a WNN one. Both key
    on the kept cell ids. Rows stay in kept cell order within each gene; R's
    merge sorts them by cell id.

    Args:
        sc: The experiment, a subset or meta cells.
        features: Feature ids from `expr_modality`.
        embedding: The embedding name.
        scale: z-score each feature.
        clip: Clip the z-scores, with `scale`.
        obs_cols: obs columns to attach.
        expr_modality: `"rna"` or `"adt"`.
        embd_modality: `"rna"`, `"adt"` or `"wnn"`.
        layer: `"norm"` or `"magic"`.

    Returns:
        `cell_id`, `dim_*`, the obs columns, `gene` (Enum) and `expression`.
    """
    expr = extract_gene_expression(
        sc, features, scale=scale, clip=clip, modality=expr_modality, layer=layer
    )
    embd = extract_embedding_data(
        sc, embedding, obs_cols=obs_cols, modality=embd_modality
    )
    feats = [c for c in expr.columns if c != "cell_id"]
    merged = embd.join(expr, on="cell_id", how="inner", maintain_order="left")
    return _melt(merged, embd.columns, feats)


@beartype
def extract_gene_violin_data(
    sc: ScOrMc,
    features: Sequence[str],
    grouping_variable: str,
    *,
    scale: bool = False,
    clip: float | None = None,
    modality: ExprModality = "rna",
    layer: Layer = "norm",
) -> pl.DataFrame:
    """Per-cell expression with its group, long, as R's
    `extract_gene_violin_data`.

    Args:
        sc: The experiment, a subset or meta cells.
        features: Gene ids, or ADT feature ids for `"adt"`.
        grouping_variable: obs column to group by.
        scale: z-score each feature across the cells.
        clip: Clip the z-scores to `[-clip, clip]`, with `scale`.
        modality: `"rna"`, or `"adt"` on a `SingleCellsMultiModal`.
        layer: `"norm"`, or `"magic"` for the imputed layer of `run_magic`.

    Returns:
        `cell_id`, `group` (Enum in level order), `gene` (Enum in feature
        order) and `expression`, gene-major.
    """
    expr = extract_gene_expression(
        sc, features, scale=scale, clip=clip, modality=modality, layer=layer
    )
    feats = [c for c in expr.columns if c != "cell_id"]
    codes, levels = _groups_for(sc, expr["cell_id"].to_numpy(), grouping_variable)
    expr = expr.with_columns(
        _enum([levels[c] if c >= 0 else None for c in codes], levels, "group")
    )
    return _melt(expr, ["cell_id", "group"], feats)


@beartype
def _parse_feature_modality(
    feature: str, default: ExprModality
) -> tuple[str, ExprModality]:
    """A `_rna` / `_adt` suffix picks the modality, as R's
    `.parse_feature_modality`."""
    if feature.endswith("_rna"):
        return feature.removesuffix("_rna"), "rna"
    if feature.endswith("_adt"):
        return feature.removesuffix("_adt"), "adt"
    return feature, default


@beartype
def extract_feature_pair(
    sc: ScOrMc,
    feature_1: str,
    feature_2: str,
    *,
    obs_cols: Sequence[str] | None = None,
    scale: bool = False,
    clip: float | None = None,
    modality: ExprModality = "rna",
    layer: Layer = "norm",
) -> pl.DataFrame:
    """Two features side by side for a scatter, as R's `extract_feature_pair`.

    A `_rna` or `_adt` suffix on either feature picks its modality; without
    one, `modality` applies. Rows stay in kept cell order; R's merge sorts
    them by cell id.

    Args:
        sc: The experiment, a subset or meta cells.
        feature_1: First feature, with an optional `_rna` / `_adt` suffix.
        feature_2: Second feature, same convention.
        obs_cols: obs columns to attach.
        scale: z-score each feature across the cells.
        clip: Clip the z-scores to `[-clip, clip]`, with `scale`.
        modality: Default modality for a feature without a suffix.
        layer: `"norm"`, or `"magic"` for the imputed layer of `run_magic`.

    Returns:
        `cell_id`, `feature_1`, `feature_2` and the obs columns, over the
        cells both features cover.
    """
    id_1, mod_1 = _parse_feature_modality(feature_1, modality)
    id_2, mod_2 = _parse_feature_modality(feature_2, modality)
    e1 = extract_gene_expression(
        sc, [id_1], scale=scale, clip=clip, modality=mod_1, layer=layer
    ).rename({id_1: "feature_1"})
    e2 = extract_gene_expression(
        sc, [id_2], scale=scale, clip=clip, modality=mod_2, layer=layer
    ).rename({id_2: "feature_2"})
    pair = e1.join(e2, on="cell_id", how="inner", maintain_order="left")
    return _with_obs(pair, sc, obs_cols)


#########################
# Graphs on embeddings #
#########################


@beartype
def _upper_edges(
    i: np.ndarray, j: np.ndarray, w: np.ndarray, threshold: float, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edges of a symmetric graph once, R's `.paga_edges`: `i < j`, weight at
    least `threshold`, both ends among `keep`. Sorted by `(i, j)`."""
    mask = (i < j) & (w >= threshold) & np.isin(i, keep) & np.isin(j, keep)
    order = np.lexsort((j[mask], i[mask]))
    return i[mask][order], j[mask][order], w[mask][order]


@beartype
def _edge_frame(
    i: np.ndarray,
    j: np.ndarray,
    w: np.ndarray,
    node_ids: np.ndarray,
    xy: np.ndarray,
    labels: pl.Series,
) -> pl.DataFrame:
    """The edge table, both ends looked up by node: `node_ids[k]` sits at
    `xy[k]` and is called `labels[k]`."""
    pos = {v: k for k, v in enumerate(node_ids.tolist())}
    a = np.asarray([pos[v] for v in i.tolist()], dtype=np.int64)
    b = np.asarray([pos[v] for v in j.tolist()], dtype=np.int64)
    return pl.DataFrame(
        {
            "from": labels.gather(a),
            "to": labels.gather(b),
            "weight": w.astype(np.float64),
            "x": xy[a, 0] if a.size else np.zeros(0),
            "y": xy[a, 1] if a.size else np.zeros(0),
            "xend": xy[b, 0] if b.size else np.zeros(0),
            "yend": xy[b, 1] if b.size else np.zeros(0),
        }
    )


@beartype
def extract_paga_plot_data(
    sc: ScOrMc,
    paga_res: PagaRes,
    *,
    embedding: str = "umap",
    cluster_col: str | None = None,
    threshold: float = 0.01,
    tree_only: bool = False,
    centroid: Literal["median", "mean"] = "median",
    node_stat_col: str | None = None,
    modality: Modality = "rna",
) -> GraphPlotData:
    """The PAGA graph with every cluster at the centroid of its cells, as
    R's `extract_paga_plot_data`.

    Args:
        sc: The object PAGA ran on.
        paga_res: The `run_paga` result.
        embedding: The embedding to place the nodes in.
        cluster_col: The clustering. Defaults to PAGA's, and anything else
            errors: one clustering's nodes with another's edges look right and
            are wrong.
        threshold: Edges below this connectivity are dropped.
        tree_only: The maximum spanning forest instead of the full graph.
        centroid: How a cluster's position is summarised.
        node_stat_col: A numeric obs column summarised per cluster the same
            way, into `stat`.
        modality: Where the embedding lives.

    Returns:
        Nodes `cluster` (Enum in graph order), `dim_1`, `dim_2`, `n_cells`
        and optionally `stat`, without empty clusters; edges with `from` /
        `to` as cluster labels.

    Raises:
        ValueError: On a different `cluster_col`, cells whose cluster PAGA
            does not know, or a non-numeric `node_stat_col`.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    paga_col = paga_res.params["cluster_col"]
    if cluster_col is not None and cluster_col != paga_col:
        raise ValueError(
            f"PAGA was run on `{paga_col}` but `{cluster_col}` was requested. "
            "Drawing one clustering's nodes with another's edges gives a graph "
            "that looks fine and is not."
        )
    conn = paga_res.connectivities_tree if tree_only else paga_res.connectivities
    levels = list(paga_res.clusters)
    obs_cols = list(
        dict.fromkeys([paga_col] + ([node_stat_col] if node_stat_col else []))
    )
    embd = extract_embedding_data(sc, embedding, obs_cols=obs_cols, modality=modality)

    lookup = {v: k for k, v in enumerate(levels)}
    labels = _r_labels(embd[paga_col])
    grp = np.asarray([lookup.get(v, -1) if v is not None else -1 for v in labels])
    n_bad = int((grp < 0).sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} cell(s) carry a `{paga_col}` value the PAGA result does "
            "not know about. Re-run run_paga() on the current clustering."
        )
    summarise = np.nanmedian if centroid == "median" else np.nanmean
    dims = embd.select("dim_1", "dim_2").to_numpy()
    sizes = dict(
        zip(
            paga_res.sizes["cluster"].to_list(),
            paga_res.sizes["n_cells"].to_list(),
            strict=True,
        )
    )
    present = [k for k in range(len(levels)) if (grp == k).any()]
    nodes = {
        "cluster": _enum([levels[k] for k in present], levels, "cluster"),
        "dim_1": [float(summarise(dims[grp == k, 0])) for k in present],
        "dim_2": [float(summarise(dims[grp == k, 1])) for k in present],
        "n_cells": [int(sizes[levels[k]]) for k in present],
    }
    if node_stat_col is not None:
        stat = embd[node_stat_col]
        if not stat.dtype.is_numeric():
            raise ValueError(f"`{node_stat_col}` must be numeric.")
        sv = stat.cast(pl.Float64).fill_null(np.nan).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nodes["stat"] = [float(summarise(sv[grp == k])) for k in present]
    node_frame = pl.DataFrame(nodes).filter(pl.col("n_cells") > 0)

    keep_idx = np.asarray([lookup[c] for c in node_frame["cluster"].to_list()])
    i, j = np.nonzero(conn)
    i, j, w = _upper_edges(i, j, conn[i, j], threshold, keep_idx)
    xy = node_frame.select("dim_1", "dim_2").to_numpy()
    edges = _edge_frame(i, j, w, keep_idx, xy, node_frame["cluster"].cast(pl.String))
    return GraphPlotData(nodes=node_frame, edges=edges, embedding=embedding)


@beartype
def _shared_cells(nhoods: CsrMatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper triangle of `crossprod(nhoods)`: the cells two neighbourhoods
    share, as `(i, j, shared)` with `i < j`."""
    indptr = nhoods.indptr.astype(np.int64)
    lens = np.diff(indptr)
    row = np.repeat(np.arange(lens.size), lens)
    # every entry pairs with the later entries of its row
    later = indptr[row + 1] - np.arange(indptr[-1]) - 1
    total = int(later.sum())
    a = np.repeat(np.arange(indptr[-1]), later)
    starts = np.repeat(np.cumsum(later) - later, later)
    b = a + 1 + (np.arange(total) - starts)
    ca = nhoods.indices[a].astype(np.int64)
    cb = nhoods.indices[b].astype(np.int64)
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    n = nhoods.shape[1]
    keys, inv = np.unique(lo * n + hi, return_inverse=True)
    shared = np.bincount(
        inv, weights=nhoods.data[a].astype(np.float64) * nhoods.data[b]
    )
    return keys // n, keys % n, shared


@beartype
def extract_milo_plot_data(
    sc: ScOrMc,
    milo_res: MiloR,
    *,
    embedding: str = "umap",
    alpha: float = 0.1,
    overlap: int = 1,
    modality: Modality = "rna",
) -> GraphPlotData:
    """The Milo neighbourhood graph with every neighbourhood at its index
    cell, as R's `extract_milo_plot_data` and Milo's `plotNhoodGraphDA`.

    Args:
        sc: The object the Milo result was built on.
        milo_res: A tested `MiloR`.
        embedding: The embedding to place the nodes in.
        alpha: SpatialFDR threshold for `is_sig`.
        overlap: Minimum shared cells for an edge.
        modality: Where the embedding lives.

    Returns:
        Nodes `Nhood` (0-based), `dim_1`, `dim_2`, `size`, every column of the
        test and `is_sig`; edges with `weight` the shared cells.

    Raises:
        ValueError: If untested, or the cell count moved since.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if overlap < 1:
        raise ValueError("overlap must be at least 1")
    da = milo_res.nhoods_info
    if da is None:
        raise ValueError(
            "No differential abundance results found. Did you run test_nhoods()?"
        )
    embd = extract_embedding_data(sc, embedding, modality=modality)
    nhoods = milo_res.nhoods
    if embd.height != nhoods.shape[0]:
        raise ValueError(
            f"The embedding has {embd.height} cells, the miloR neighbourhoods "
            f"{nhoods.shape[0]}. Was the object subset or refiltered after "
            "get_milor_abundances()?"
        )
    nhood = da["Nhood"].to_numpy().astype(np.int64)
    index_cell = get_index_cells(milo_res).astype(np.int64)[nhood]
    size = np.bincount(
        nhoods.indices.astype(np.int64),
        weights=nhoods.data.astype(np.float64),
        minlength=nhoods.shape[1],
    )
    dims = embd.select("dim_1", "dim_2").to_numpy()
    nodes = (
        pl.concat(
            [
                pl.DataFrame(
                    {
                        "Nhood": da["Nhood"],
                        "dim_1": dims[index_cell, 0],
                        "dim_2": dims[index_cell, 1],
                        "size": size[nhood],
                    }
                ),
                da.drop("Nhood"),
            ],
            how="horizontal",
        )
        .with_columns(
            is_sig=pl.col("SpatialFDR").is_not_null()
            & pl.col("SpatialFDR").is_not_nan()
            & (pl.col("SpatialFDR") <= alpha)
        )
        .with_columns(pl.col("is_sig").fill_null(False))
    )

    i, j, w = _shared_cells(nhoods)
    i, j, w = _upper_edges(i, j, w, float(overlap), nhood)
    edges = _edge_frame(i, j, w, nhood, dims[index_cell], da["Nhood"])
    return GraphPlotData(nodes=nodes, edges=edges, embedding=embedding)
