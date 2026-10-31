"""Cell QC: per-cell metrics, MAD and hard-threshold outlier calls.

The port of R's QC helpers. The metric passes (`gene_set_proportions`,
`top_genes_perc`) read the cell store in Rust and write their columns to obs
under R's names, so an R session opening the directory sees the same table.
The outlier calls (`run_cell_qc`, `flag_cells`, `rescue_cells`) work on plain
vectors, as in R: they return a `CellQc` record, `get_data` turns it into an
obs frame for `SingleCells.add_obs`, and the cell filter stays with the
caller.

    >>> obs = sc.obs(filtered=True)
    >>> qc = run_cell_qc(
    ...     {"lib_size": obs["lib_size"].to_numpy()},
    ...     obs["cell_idx"].to_numpy(),
    ...     directions={"lib_size": "below"},
    ... )
    >>> sc.add_obs(get_data(qc))
    >>> sc.set_cells_to_keep(qc.cell_idx[~qc.combined])
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.single_cells import SingleCells

###########
# Globals #
###########

# Above this many kept cells the metric passes stream by default; R's
# `.N_CELLS_STREAMING_THRESHOLD`.
N_CELLS_STREAMING_THRESHOLD: int = 100_000

Direction = Literal["twosided", "below", "above"]
Bounds = Mapping[str, float | int]
Vector = np.ndarray | pl.Series | Sequence[float] | Sequence[int]


###########
# Helpers #
###########


@beartype
def _auto_streaming(n_cells: int, streaming: bool | None) -> bool:
    """R's `auto_streaming`: the given flag, else stream above the threshold."""
    return n_cells > N_CELLS_STREAMING_THRESHOLD if streaming is None else streaming


@beartype
def _vector(x: Vector, name: str) -> np.ndarray:
    """A 1-D numeric array, dtype kept.

    Raises:
        ValueError: If it is empty, not 1-D or not numeric.
    """
    arr = x.to_numpy() if isinstance(x, pl.Series) else np.asarray(x)
    if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a non-empty 1-D numeric vector")
    return arr


@beartype
def _check_bounds(bounds: Bounds, name: str) -> None:
    """Named bounds hold `lower` and / or `upper` and nothing else.

    Raises:
        ValueError: Otherwise.
    """
    if not bounds or not set(bounds) <= {"lower", "upper"}:
        raise ValueError(
            f"`{name}`: bounds need `lower` and / or `upper`, got {sorted(bounds)}"
        )


@beartype
def _within_bounds(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    """`True` where `x` lies within the bounds, inclusive on both ends."""
    ok = np.ones(x.shape[0], dtype=bool)
    if "lower" in bounds:
        ok &= x >= bounds["lower"]
    if "upper" in bounds:
        ok &= x <= bounds["upper"]
    return ok


@beartype
def _check_bounds_direction(bounds: Bounds, direction: Direction, name: str) -> None:
    """Hard bounds must not contradict the metric's MAD direction.

    Raises:
        ValueError: On a `lower` bound for `"above"` or an `upper` for `"below"`.
    """
    if direction == "above" and "lower" in bounds:
        raise ValueError(f"`{name}`: direction is 'above' but `lower` bound supplied.")
    if direction == "below" and "upper" in bounds:
        raise ValueError(f"`{name}`: direction is 'below' but `upper` bound supplied.")


@beartype
def _first_seen(groups: np.ndarray) -> list[str]:
    """Distinct group labels in order of first appearance, as R's `unique`."""
    return list(dict.fromkeys(groups.tolist()))


#######
# MAD #
#######


@beartype
@dataclass(frozen=True)
class MadOutlier:
    """MAD outlier calls on one metric.

    Attributes:
        outlier: One flag per value.
        median: The metric's median.
        upper_threshold: `median + margin`, `None` for direction `"below"`.
        lower_threshold: `median - margin`, `None` for direction `"above"`.
    """

    outlier: np.ndarray
    median: float
    upper_threshold: float | None
    lower_threshold: float | None


@beartype
def per_cell_qc_outlier(
    metric: Vector,
    *,
    threshold: float = 3.0,
    direction: Direction = "twosided",
    r_standard: bool = True,
) -> MadOutlier:
    """MAD outlier detection on a per-cell QC metric, as R's version.

    Args:
        metric: The metric.
        threshold: MADs from the median beyond which a value is an outlier.
        direction: Which tail(s) to flag.
        r_standard: Scale the MAD by 1.4826 like R's `mad()`, so `threshold`
            counts standard deviations under normality. `False` uses the raw
            MAD.

    Returns:
        The flags and the applied thresholds.
    """
    x = np.ascontiguousarray(_vector(metric, "metric"), dtype=np.float64)
    outlier, margin = _bixverse.mad_outlier(x, threshold, direction, r_standard)
    med = float(np.median(x))
    return MadOutlier(
        outlier=outlier,
        median=med,
        upper_threshold=med + margin if direction != "below" else None,
        lower_threshold=med - margin if direction != "above" else None,
    )


@beartype
def _per_group_qc_outlier(
    metrics: dict[str, np.ndarray],
    groups: np.ndarray,
    directions: dict[str, Direction],
    threshold: float,
    r_standard: bool,
) -> pl.DataFrame:
    """MAD calls on per-group medians, flagging whole groups, as R's version.

    Returns:
        One row per metric and group: `metric`, `group`, `group_median`,
        `lower_threshold`, `upper_threshold`, `is_outlier`.
    """
    levels = _first_seen(groups)
    frames = []
    for name, x in metrics.items():
        medians = np.array([np.median(x[groups == g]) for g in levels])
        res = per_cell_qc_outlier(
            medians,
            threshold=threshold,
            direction=directions[name],
            r_standard=r_standard,
        )
        frames.append(
            pl.DataFrame(
                {
                    "metric": [name] * len(levels),
                    "group": levels,
                    "group_median": medians,
                    "lower_threshold": [res.lower_threshold] * len(levels),
                    "upper_threshold": [res.upper_threshold] * len(levels),
                    "is_outlier": res.outlier,
                },
                schema_overrides={
                    "lower_threshold": pl.Float64,
                    "upper_threshold": pl.Float64,
                },
            )
        )
    return pl.concat(frames)


##########
# CellQc #
##########


@beartype
@dataclass(frozen=True)
class CellQc:
    """Outlier calls over several per-cell metrics, R's `CellQc`.

    The per-metric flag dicts all share the metric order of `metrics`. A cell
    is an outlier on a metric if MAD flagged it and it was not rescued, or a
    hard threshold flagged it; `combined` is that over any metric.

    Attributes:
        cell_idx: 0-based store index of each cell.
        metrics: Metric name to its values, dtype as given.
        groups: Group label per cell (`"all"` without groups).
        thresholds: The MAD thresholds per metric and group (`metric`,
            `group`, `median`, `upper_threshold`, `lower_threshold`), `None`
            when MAD was skipped. R's `per_metric`.
        outlier_mat: MAD flags per metric.
        hard_mat: Hard-threshold flags per metric.
        rescued_mat: Rescued MAD flags per metric.
        hard_thresholds: The hard bounds applied, per metric.
        combined: The final call per cell.
        per_group_stats: Group-level MAD calls when there is more than one
            group, else `None`.
    """

    cell_idx: np.ndarray
    metrics: dict[str, np.ndarray]
    groups: np.ndarray
    thresholds: pl.DataFrame | None
    outlier_mat: dict[str, np.ndarray]
    hard_mat: dict[str, np.ndarray]
    rescued_mat: dict[str, np.ndarray]
    hard_thresholds: dict[str, dict[str, float]] | None
    combined: np.ndarray
    per_group_stats: pl.DataFrame | None

    @beartype
    def final(self, metric: str) -> np.ndarray:
        """The final per-metric call: MAD and not rescued, or hard."""
        return (self.outlier_mat[metric] & ~self.rescued_mat[metric]) | self.hard_mat[
            metric
        ]


@beartype
def _combine(
    outlier_mat: dict[str, np.ndarray],
    hard_mat: dict[str, np.ndarray],
    rescued_mat: dict[str, np.ndarray],
) -> np.ndarray:
    """A cell is out if any metric's final call flags it."""
    return np.any(
        [(outlier_mat[m] & ~rescued_mat[m]) | hard_mat[m] for m in outlier_mat], axis=0
    )


@beartype
def run_cell_qc(
    metrics: Mapping[str, Vector],
    cells_to_keep: np.ndarray | Sequence[int],
    *,
    directions: Mapping[str, Direction] | None = None,
    threshold: float = 3.0,
    groups: Sequence[object] | np.ndarray | pl.Series | None = None,
    hard_thresholds: Mapping[str, Bounds] | None = None,
    mad: bool = True,
    r_standard: bool = True,
) -> CellQc:
    """Outlier detection over several per-cell QC metrics, as R's version.

    MAD runs per group when `groups` is given; hard thresholds ignore groups.

    Args:
        metrics: Metric name to its values, one per cell.
        cells_to_keep: 0-based store index of each cell, aligned to the
            metrics.
        directions: Metric name to direction. Missing means `"twosided"` for
            every metric; given, it must name every metric.
        threshold: MADs for the outlier calls.
        groups: Optional group label per cell (donor, sample).
        hard_thresholds: Optional metric name to `{"lower": .., "upper": ..}`
            (either or both), applied regardless of groups.
        mad: Run MAD. `False` needs `hard_thresholds`.
        r_standard: Scale the MAD by 1.4826, see `per_cell_qc_outlier`.

    Returns:
        The calls.

    Raises:
        ValueError: On mismatched lengths, unknown metric names, malformed
            bounds, bounds that contradict a direction, or nothing to do.
    """
    if not mad and hard_thresholds is None:
        raise ValueError(
            "Nothing to do: `mad = False` and no `hard_thresholds` supplied."
        )
    if not metrics:
        raise ValueError("metrics must not be empty")
    vals = {name: _vector(v, name) for name, v in metrics.items()}
    names = list(vals)
    n = next(iter(vals.values())).shape[0]
    if any(v.shape[0] != n for v in vals.values()):
        raise ValueError("every metric must have one value per cell")

    cell_idx = np.asarray(cells_to_keep)
    if cell_idx.ndim != 1 or cell_idx.dtype.kind not in "iu" or cell_idx.size != n:
        raise ValueError(f"cells_to_keep must hold {n} integer indices")
    if cell_idx.min() < 0:
        raise ValueError("cells_to_keep must be 0-based and non-negative")
    cell_idx = cell_idx.astype(np.uint32)

    grp = (
        np.full(n, "all", dtype=object)
        if groups is None
        else np.asarray(
            groups.to_list() if isinstance(groups, pl.Series) else groups, dtype=object
        )
    )
    if grp.shape != (n,) or any(g is None for g in grp):
        raise ValueError(f"groups must hold {n} non-missing labels")
    grp = np.array([str(g) for g in grp], dtype=object)

    if directions is None:
        dirs: dict[str, Direction] = dict.fromkeys(names, "twosided")
    else:
        missing = [m for m in names if m not in directions]
        if missing:
            raise ValueError(f"directions is missing metric(s): {missing}")
        dirs = {m: directions[m] for m in names}

    hard = None
    if hard_thresholds is not None:
        unknown = [m for m in hard_thresholds if m not in vals]
        if unknown:
            raise ValueError(f"hard_thresholds names unknown metric(s): {unknown}")
        for m, bounds in hard_thresholds.items():
            _check_bounds(bounds, m)
            if mad:
                _check_bounds_direction(bounds, dirs[m], m)
        hard = {
            m: {k: float(v) for k, v in b.items()} for m, b in hard_thresholds.items()
        }

    no_flags = {m: np.zeros(n, dtype=bool) for m in names}
    thresholds = None
    per_group_stats = None
    if mad:
        levels = _first_seen(grp)
        outlier_mat = {}
        rows = []
        for m in names:
            flags = np.zeros(n, dtype=bool)
            for g in levels:
                in_g = grp == g
                res = per_cell_qc_outlier(
                    vals[m][in_g],
                    threshold=threshold,
                    direction=dirs[m],
                    r_standard=r_standard,
                )
                flags[in_g] = res.outlier
                rows.append(
                    (m, g, res.median, res.upper_threshold, res.lower_threshold)
                )
            outlier_mat[m] = flags
        thresholds = pl.DataFrame(
            rows,
            schema={
                "metric": pl.String,
                "group": pl.String,
                "median": pl.Float64,
                "upper_threshold": pl.Float64,
                "lower_threshold": pl.Float64,
            },
            orient="row",
        )
        if len(levels) > 1:
            per_group_stats = _per_group_qc_outlier(
                vals, grp, dirs, threshold, r_standard
            )
    else:
        outlier_mat = dict(no_flags)

    hard_mat = {
        m: ~_within_bounds(vals[m], hard[m])
        if hard is not None and m in hard
        else np.zeros(n, dtype=bool)
        for m in names
    }
    rescued_mat = {m: np.zeros(n, dtype=bool) for m in names}

    return CellQc(
        cell_idx=cell_idx,
        metrics=vals,
        groups=grp,
        thresholds=thresholds,
        outlier_mat=outlier_mat,
        hard_mat=hard_mat,
        rescued_mat=rescued_mat,
        hard_thresholds=hard,
        combined=_combine(outlier_mat, hard_mat, rescued_mat),
        per_group_stats=per_group_stats,
    )


@beartype
def run_cell_qc_fixed(
    metrics: Mapping[str, Vector],
    cells_to_keep: np.ndarray | Sequence[int],
    hard_thresholds: Mapping[str, Bounds],
    *,
    groups: Sequence[object] | np.ndarray | pl.Series | None = None,
) -> CellQc:
    """Fixed-threshold cell QC: `run_cell_qc` with MAD off."""
    return run_cell_qc(
        metrics,
        cells_to_keep,
        groups=groups,
        hard_thresholds=hard_thresholds,
        mad=False,
    )


@beartype
def _check_metric_names(qc: CellQc, names: Mapping[str, Bounds], arg: str) -> None:
    """Every name is a metric of `qc` and every bounds entry is well formed.

    Raises:
        ValueError: Otherwise.
    """
    unknown = [m for m in names if m not in qc.metrics]
    if unknown:
        raise ValueError(f"{arg} names unknown metric(s): {unknown}")
    for m, bounds in names.items():
        _check_bounds(bounds, m)


@beartype
def rescue_cells(qc: CellQc, rescue_thresholds: Mapping[str, Bounds]) -> CellQc:
    """Un-flag MAD outliers that fall within safe bounds, as R's version.

    For each metric named, MAD-flagged cells within `[lower, upper]` are
    rescued; this replaces any earlier rescue of that metric. Hard flags are
    never cleared, and a warning counts the rescued cells they still catch.

    Args:
        qc: The calls.
        rescue_thresholds: Metric name to `{"lower": .., "upper": ..}`.

    Returns:
        A new `CellQc`.

    Raises:
        ValueError: On unknown metrics or malformed bounds, or if MAD was
            skipped.
    """
    _check_metric_names(qc, rescue_thresholds, "rescue_thresholds")
    if qc.thresholds is None:
        raise ValueError("No MAD results to rescue from.")

    rescued = dict(qc.rescued_mat)
    for m, bounds in rescue_thresholds.items():
        rescued[m] = qc.outlier_mat[m] & _within_bounds(qc.metrics[m], bounds)
        overlap = int((rescued[m] & qc.hard_mat[m]).sum())
        if overlap:
            warnings.warn(
                f"`{m}`: {overlap} cell(s) match rescue bounds but remain flagged "
                "by hard threshold.",
                stacklevel=2,
            )
    return dataclasses.replace(
        qc,
        rescued_mat=rescued,
        combined=_combine(qc.outlier_mat, qc.hard_mat, rescued),
    )


@beartype
def flag_cells(
    qc: CellQc, hard_thresholds: Mapping[str, Bounds], *, reset: bool = False
) -> CellQc:
    """Add hard-threshold flags, as R's version.

    New flags are unioned with the existing ones per metric, and the bounds
    replace any earlier bounds of the same metric.

    Args:
        qc: The calls.
        hard_thresholds: Metric name to `{"lower": .., "upper": ..}`.
        reset: Clear every existing hard flag and bound first.

    Returns:
        A new `CellQc`.

    Raises:
        ValueError: On unknown metrics or malformed bounds.
    """
    _check_metric_names(qc, hard_thresholds, "hard_thresholds")
    n = qc.cell_idx.shape[0]
    hard_mat = (
        {m: np.zeros(n, dtype=bool) for m in qc.hard_mat}
        if reset
        else dict(qc.hard_mat)
    )
    existing = {} if reset or qc.hard_thresholds is None else qc.hard_thresholds
    for m, bounds in hard_thresholds.items():
        hard_mat[m] = hard_mat[m] | ~_within_bounds(qc.metrics[m], bounds)
    merged = {
        **existing,
        **{m: {k: float(v) for k, v in b.items()} for m, b in hard_thresholds.items()},
    }
    return dataclasses.replace(
        qc,
        hard_mat=hard_mat,
        hard_thresholds=merged,
        combined=_combine(qc.outlier_mat, hard_mat, qc.rescued_mat),
    )


@beartype
def get_data(qc: CellQc) -> pl.DataFrame:
    """The calls as an obs frame for `SingleCells.add_obs`, R's `get_data`.

    Columns are R's: the metrics, `cell_idx` (0-based here), `grp`,
    `global_outlier`, then per metric `<m>_is_outlier`, `<m>_mad_outlier`,
    `<m>_hard_outlier` and `<m>_rescued`.
    """
    cols: dict[str, object] = dict(qc.metrics)
    cols["cell_idx"] = qc.cell_idx
    cols["grp"] = pl.Series(qc.groups.tolist(), dtype=pl.String)
    cols["global_outlier"] = qc.combined
    for m in qc.metrics:
        cols[f"{m}_is_outlier"] = qc.final(m)
        cols[f"{m}_mad_outlier"] = qc.outlier_mat[m]
        cols[f"{m}_hard_outlier"] = qc.hard_mat[m]
        cols[f"{m}_rescued"] = qc.rescued_mat[m]
    return pl.DataFrame(cols)


##################
# Metric passes #
##################


@beartype
def _write_metrics(sc: SingleCells, names: list[str], values: list[np.ndarray]) -> None:
    """Join per kept cell metrics onto obs; cells not kept get NULL."""
    frame = pl.DataFrame(
        {"cell_idx": sc.cells_to_keep, **dict(zip(names, values, strict=True))}
    )
    sc.add_obs(frame)


@beartype
def gene_set_proportions(
    sc: SingleCells,
    gene_set_list: Mapping[str, Sequence[str]],
    *,
    streaming: bool | None = None,
    verbose: bool = False,
) -> None:
    """Share of each kept cell's reads in each gene set, as R's
    `gene_set_proportions_sc`.

    The mitochondrial percentage pattern. One obs column per set, named by
    its key, as a fraction in `[0, 1]`.

    Args:
        sc: The experiment.
        gene_set_list: Set name to gene ids, as in `sc.gene_ids()`. Unknown
            ids warn and drop.
        streaming: Read the cells in batches. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` kept cells.
        verbose: Controls verbosity.

    Raises:
        ValueError: If no id of a set matches.
    """
    if not gene_set_list:
        raise ValueError("gene_set_list must not be empty")
    names = list(gene_set_list)
    sets = [sc.gene_indices(list(gene_set_list[n])) for n in names]
    res = _bixverse.gene_set_perc(
        str(sc.cells_path),
        sets,
        sc.cells_to_keep,
        _auto_streaming(sc.n_cells, streaming),
        verbose,
    )
    _write_metrics(sc, names, res)


@beartype
def top_genes_perc(
    sc: SingleCells,
    *,
    top_n_vals: Sequence[int] = (25, 50, 100),
    streaming: bool | None = None,
    verbose: bool = False,
) -> None:
    """Share of each kept cell's reads in its top N genes, as R's
    `top_genes_perc_sc`.

    High values flag low-complexity cells. One obs column per N, named
    `top_<N>_genes_percentage`, as a fraction in `[0, 1]`.

    Args:
        sc: The experiment.
        top_n_vals: The N values.
        streaming: Read the cells in batches. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` kept cells.
        verbose: Controls verbosity.

    Raises:
        ValueError: If `top_n_vals` is empty or holds a value below 1.
    """
    if not top_n_vals or min(top_n_vals) < 1:
        raise ValueError("top_n_vals must hold positive integers")
    res = _bixverse.top_genes_perc(
        str(sc.cells_path),
        list(top_n_vals),
        sc.cells_to_keep,
        _auto_streaming(sc.n_cells, streaming),
        verbose,
    )
    _write_metrics(sc, [f"top_{n}_genes_percentage" for n in top_n_vals], res)


############
# Features #
############


@beartype
def get_available_features(sc: SingleCells) -> pl.DataFrame:
    """Every obs column and gene id, as R's `get_sc_available_features`.

    Returns:
        `feature_name` and `origin` (`"obs"` or `"counts"`).
    """
    with db.connect(sc.db_path, read_only=True) as con:
        obs_cols = db.columns(con, "obs")
    genes = [str(g) for g in sc.gene_ids()]
    return pl.DataFrame(
        {
            "feature_name": obs_cols + genes,
            "origin": ["obs"] * len(obs_cols) + ["counts"] * len(genes),
        }
    )
