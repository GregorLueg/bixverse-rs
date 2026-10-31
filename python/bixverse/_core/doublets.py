"""Doublet detection, empty droplets and CellSweep.

The port of R's `scrublet_sc`, `doublet_detection_boost_sc`,
`scdblfinder_sc`, `call_doublets_manual`, `cellsweep_sc` and the empty
droplet inference behind it. The work runs in Rust.

The doublet callers leave the experiment alone and return a record, as R's
S3 results; `get_data` turns one into an obs frame for `SingleCells.add_obs`:

    >>> res = scrublet(sc, group_by="sample_id")
    >>> sc.add_obs(get_data(res))

`cellsweep` writes a new experiment holding the denoised counts, as R's.

With `group_by`, every method runs once per group (labels sorted) on the
group's kept cells, and the per-cell vectors come back in ascending cell
order; per-group scalars become `{group: value}` dicts, as R's named vectors.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.ingest import _prepare
from bixverse._core.qc import _auto_streaming
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN, Selection, SingleCells
from bixverse.params import (
    BoostParams,
    CellsweepParams,
    EmptyDropletsParams,
    MinQualityParams,
    ScdblfinderParams,
    ScrubletParams,
    to_rust,
)

###########
# Globals #
###########

# R's `.validate_group_sizes`: below the first a group errors, below the
# second it warns.
GROUP_MIN_CELLS: int = 50
GROUP_WARN_CELLS: int = 500

# R's fast clustering default: `as.integer(sqrt(n_cells) * 4)` centroids.
FAST_CLUSTER_CENTROIDS_PER_ROOT: float = 4.0

# R's `CELLSWEEP_MAX_MIN_LIB_SIZE`: a smallest library size above this means
# the empty droplets were filtered out at ingest.
CELLSWEEP_MAX_MIN_LIB_SIZE: int = 100

# Below this share of simulated doublets above the threshold, the overall
# doublet rate is reported as 0, as R's `.update_scrublet_threshold`.
MIN_DETECTABLE_FRACTION: float = 0.01

ScoreType = Literal["weighted", "cxds_scores"]
GroupScalar = float | dict[str, float]

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
def _rate_line(n_cells: int, n_doublets: int) -> str:
    """`N cells, M doublets (P%)`, as R's print methods."""
    return f"{n_cells} cells, {n_doublets} doublets ({100 * n_doublets / n_cells:.1f}%)"


@beartype
def _scalar(x: GroupScalar) -> float:
    """An ungrouped result's scalar."""
    assert isinstance(x, float)
    return x


@beartype
def _group_header(name: str, group_by: str, n_groups: int, rest: str) -> str:
    """The first line of a grouped result's repr."""
    return f"{name} (grouped by '{group_by}', {n_groups} groups): {rest}"


@beartype
@dataclass(frozen=True, repr=False)
class ScrubletRes:
    """Scrublet result, R's `ScrubletRes`.

    Grouped results hold the per-group scalars and simulated scores as
    `{group: value}` dicts.

    Attributes:
        predicted_doublets: Per cell, whether it is called a doublet.
        doublet_scores_obs: Per-cell doublet score.
        doublet_errors_obs: Per-cell standard error of the score.
        z_scores: `(score - threshold) / error` per cell.
        threshold: The score threshold.
        detected_doublet_rate: Share of cells called doublets.
        detectable_doublet_fraction: Share of simulated doublets above the
            threshold.
        overall_doublet_rate: Estimated overall doublet rate.
        doublet_scores_sim: Scores of the simulated doublets.
        cell_indices: 0-based store indices the per-cell vectors align to.
        pca: PCA of observed then simulated cells, if asked for.
        pair_1: First parent (0-based, within the run's cells) per simulated
            doublet, if asked for.
        pair_2: Second parent per simulated doublet, if asked for.
        cell_groups: Group label per cell when grouped, else `None`.
        group_by: The grouping column when grouped, else `None`.
    """

    predicted_doublets: np.ndarray
    doublet_scores_obs: np.ndarray
    doublet_errors_obs: np.ndarray
    z_scores: np.ndarray
    threshold: GroupScalar
    detected_doublet_rate: GroupScalar
    detectable_doublet_fraction: GroupScalar
    overall_doublet_rate: GroupScalar
    doublet_scores_sim: np.ndarray | dict[str, np.ndarray]
    cell_indices: np.ndarray
    pca: np.ndarray | dict[str, np.ndarray] | None = None
    pair_1: np.ndarray | dict[str, np.ndarray] | None = None
    pair_2: np.ndarray | dict[str, np.ndarray] | None = None
    cell_groups: np.ndarray | None = None
    group_by: str | None = None

    def __repr__(self) -> str:
        """R's `print.ScrubletRes`."""
        n = len(self.predicted_doublets)
        d = int(self.predicted_doublets.sum())
        if self.group_by is not None:
            assert isinstance(self.threshold, dict) and self.cell_groups is not None
            lines = [
                _group_header(
                    "ScrubletRes",
                    self.group_by,
                    len(self.threshold),
                    _rate_line(n, d),
                ),
                "Per-group:",
            ]
            for g, thr in self.threshold.items():
                mask = self.cell_groups == g
                calls = self.predicted_doublets[mask]
                lines.append(
                    f"  - {g}: {_rate_line(int(mask.sum()), int(calls.sum()))}, "
                    f"threshold = {thr:.4f}"
                )
        else:
            assert isinstance(self.doublet_scores_sim, np.ndarray)
            lines = [
                f"ScrubletRes: {_rate_line(n, d)}",
                f"  Threshold:              {_scalar(self.threshold):.4f}",
                "  Detected doublet rate:  "
                f"{100 * _scalar(self.detected_doublet_rate):.1f}%",
                "  Detectable fraction:    "
                f"{100 * _scalar(self.detectable_doublet_fraction):.1f}%",
                "  Overall doublet rate:   "
                f"{100 * _scalar(self.overall_doublet_rate):.1f}%",
                f"  Simulated doublets:     {len(self.doublet_scores_sim)}",
            ]
        extras = [
            name
            for name, value in (("PCA", self.pca), ("doublet pairs", self.pair_1))
            if value is not None
        ]
        if extras:
            lines.append(f"  Optional data:          {', '.join(extras)}")
        return "\n".join(lines)


@beartype
@dataclass(frozen=True, repr=False)
class BoostRes:
    """Boosted doublet classification result, R's `BoostRes`.

    Attributes:
        doublet: Per cell, whether it is called a doublet.
        doublet_score: Per-cell doublet score, averaged over the iterations.
        voting_avg: Per-cell share of iterations that voted doublet.
        cell_indices: 0-based store indices the per-cell vectors align to.
        cell_groups: Group label per cell when grouped, else `None`.
        group_by: The grouping column when grouped, else `None`.
    """

    doublet: np.ndarray
    doublet_score: np.ndarray
    voting_avg: np.ndarray
    cell_indices: np.ndarray
    cell_groups: np.ndarray | None = None
    group_by: str | None = None

    def __repr__(self) -> str:
        """R's `print.BoostRes`."""
        n, d = len(self.doublet), int(self.doublet.sum())
        if self.group_by is not None:
            assert self.cell_groups is not None
            groups = list(dict.fromkeys(self.cell_groups.tolist()))
            lines = [
                _group_header("BoostRes", self.group_by, len(groups), _rate_line(n, d)),
                "Per-group:",
            ]
            for g in groups:
                mask = self.cell_groups == g
                lines.append(
                    f"  - {g}: "
                    f"{_rate_line(int(mask.sum()), int(self.doublet[mask].sum()))}"
                )
            return "\n".join(lines)
        return "\n".join(
            [
                f"BoostRes: {_rate_line(n, d)}",
                f"  Score range: [{self.doublet_score.min():.4f}, "
                f"{self.doublet_score.max():.4f}]",
            ]
        )


@beartype
@dataclass(frozen=True, repr=False)
class ScDblFinderRes:
    """scDblFinder result, R's `ScDblFinderRes`.

    Attributes:
        predicted_doublets: Per cell, whether it is called a doublet.
        doublet_score: Per-cell classifier probability.
        cxds_scores: Per-cell cxds score.
        weighted: Per-cell weighted score.
        threshold: The score threshold.
        cluster_labels: Per-cell cluster of the last iteration, 0-based;
            grouped, `<group>_<label>` strings.
        detected_doublet_rate: Share of cells called doublets.
        features: The classifier features of the observed cells, `cell_id`
            first, if asked for.
        cell_indices: 0-based store indices the per-cell vectors align to.
        cell_groups: Group label per cell when grouped, else `None`.
        group_by: The grouping column when grouped, else `None`.
    """

    predicted_doublets: np.ndarray
    doublet_score: np.ndarray
    cxds_scores: np.ndarray
    weighted: np.ndarray
    threshold: GroupScalar
    cluster_labels: np.ndarray
    detected_doublet_rate: GroupScalar
    cell_indices: np.ndarray
    features: pl.DataFrame | None = None
    cell_groups: np.ndarray | None = None
    group_by: str | None = None

    def __repr__(self) -> str:
        """R's `print.ScDblFinderRes`."""
        n = len(self.predicted_doublets)
        d = int(self.predicted_doublets.sum())
        n_clusters = len(np.unique(self.cluster_labels))
        has_features = "TRUE" if self.features is not None else "FALSE"
        if self.group_by is not None:
            assert isinstance(self.threshold, dict) and self.cell_groups is not None
            lines = [
                _group_header(
                    "ScDblFinderRes",
                    self.group_by,
                    len(self.threshold),
                    _rate_line(n, d),
                ),
                "Per-group:",
            ]
            for g, thr in self.threshold.items():
                mask = self.cell_groups == g
                calls = self.predicted_doublets[mask]
                lines.append(
                    f"  - {g}: {_rate_line(int(mask.sum()), int(calls.sum()))}, "
                    f"threshold = {thr:.4f}"
                )
            lines += [
                f"  Total clusters:    {n_clusters} (prefixed)",
                f"  Features available: {has_features}",
            ]
            return "\n".join(lines)
        return "\n".join(
            [
                f"ScDblFinderRes: {_rate_line(n, d)}",
                f"  Threshold:        {_scalar(self.threshold):.4f}",
                f"  Score range:      [{self.doublet_score.min():.4f}, "
                f"{self.doublet_score.max():.4f}]",
                f"  Final clusters:   {n_clusters}",
                f"  Features available: {has_features}",
            ]
        )


###########
# Helpers #
###########


@beartype
def _as_character(values: pl.Series) -> list[str | None]:
    """R's `as.character`: floats at 15 significant digits, nulls kept."""
    if values.dtype.is_float():
        return [None if v is None else f"{v:.15g}" for v in values.to_list()]
    return [None if v is None else str(v) for v in values.cast(pl.String).to_list()]


@beartype
def _cells(sc: SingleCells, cells_to_use: Selection | None) -> np.ndarray:
    """0-based cells to run on: the given ones, else the kept ones."""
    if cells_to_use is None:
        return sc.cells_to_keep
    return sc._resolve_cells(cells_to_use)


@beartype
def _split_by_group(
    sc: SingleCells, group_by: str, cells: np.ndarray
) -> dict[str, np.ndarray]:
    """R's `.split_cells_by_group`: the kept cells among `cells`, split by
    the group label, groups sorted, cells ascending.

    Raises:
        ValueError: If the column is missing or holds missing values.
    """
    with db.connect(sc.db_path, read_only=True) as con:
        obs_cols = db.columns(con, "obs")
    if group_by not in obs_cols:
        raise ValueError(
            f"Column '{group_by}' not found in the obs table. Available: "
            f"{', '.join(obs_cols)}"
        )
    obs = sc.obs(cols=["cell_idx", group_by], filtered=True)
    obs = obs.filter(pl.Series(np.isin(obs["cell_idx"].to_numpy(), cells)))
    labels = _as_character(obs[group_by])
    n_missing = sum(v is None for v in labels)
    if n_missing:
        raise ValueError(
            f"Column '{group_by}' contains {n_missing} NA value(s); cannot group on it."
        )
    idx = obs["cell_idx"].to_numpy()
    lab = np.asarray(labels)
    return {
        g: idx[lab == g].astype(np.uint32)
        for g in sorted({v for v in labels if v is not None})
    }


@beartype
def _validate_group_sizes(groups: dict[str, np.ndarray]) -> None:
    """R's `.validate_group_sizes`.

    Raises:
        ValueError: If a group has fewer than `GROUP_MIN_CELLS` cells.
    """
    sizes = {g: len(c) for g, c in groups.items()}
    small = [f"'{g}' (n={n})" for g, n in sizes.items() if n < GROUP_MIN_CELLS]
    if small:
        raise ValueError(
            f"Group(s) with fewer than {GROUP_MIN_CELLS} cells (hard minimum): "
            f"{', '.join(small)}"
        )
    low = [f"'{g}' (n={n})" for g, n in sizes.items() if n < GROUP_WARN_CELLS]
    if low:
        warnings.warn(
            f"Group(s) with fewer than {GROUP_WARN_CELLS} cells (results may be "
            f"unreliable): {', '.join(low)}",
            stacklevel=3,
        )


@beartype
def _streaming(n_cells: int, streaming: bool | None, verbose: bool) -> bool:
    """R's `auto_streaming`, with its message."""
    res = _auto_streaming(n_cells, streaming)
    if res and verbose:
        logger.info("Setting streaming for large data to TRUE.")
    return res


@beartype
def _with_centroids(
    params: BoostParams | ScdblfinderParams, n_cells: int
) -> dict[str, object]:
    """`to_rust(params)`, with the fast clustering centroids R fills in."""
    d = to_rust(params)
    if params.fast_cluster and params.n_centroids is None:
        d["n_centroids"] = int(math.sqrt(n_cells) * FAST_CLUSTER_CENTROIDS_PER_ROOT)
        logger.info(
            "Fast clustering activated without any n_centroids set. Setting "
            "n_centroids to %i.",
            d["n_centroids"],
        )
    return d


@beartype
def _run_grouped(
    sc: SingleCells,
    group_by: str,
    cells: np.ndarray,
    run: Callable[[np.ndarray], dict[str, Any]],
    verbose: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Split, validate and run per group, as R's `.run_per_group`.

    Returns:
        Per group the Rust result and the cells, the order that puts the
        concatenated cells in ascending order, and the group label per
        concatenated cell.
    """
    groups = _split_by_group(sc, group_by, cells)
    _validate_group_sizes(groups)
    results = {}
    for g, group_cells in groups.items():
        if verbose:
            logger.info("Running group '%s'.", g)
        results[g] = run(group_cells)
    all_cells = np.concatenate(list(groups.values()))
    labels = np.concatenate(
        [np.full(len(c), g, dtype=object) for g, c in groups.items()]
    )
    return results, groups, np.argsort(all_cells, kind="stable"), labels


@beartype
def _concat(
    results: dict[str, dict[str, Any]], key: str, order: np.ndarray
) -> np.ndarray:
    """One per-cell field over the groups, in ascending cell order."""
    return np.concatenate([r[key] for r in results.values()])[order]


@beartype
def _per_group(results: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    """One per-group field as a `{group: value}` dict."""
    return {g: r[key] for g, r in results.items()}


############
# Scrublet #
############


@beartype
def scrublet(
    sc: SingleCells,
    *,
    scrublet_params: ScrubletParams | None = None,
    seed: int = 42,
    streaming: bool | None = None,
    cells_to_use: Selection | None = None,
    group_by: str | None = None,
    return_combined_pca: bool = False,
    return_pairs: bool = False,
    verbose: bool = False,
) -> ScrubletRes:
    """Scrublet doublet detection, as R's `scrublet_sc`.

    Simulates doublets from random pairs of observed cells, embeds both in
    one PCA on the HVGs and scores each observed cell by the share of
    simulated doublets among its neighbours. The threshold comes from Otsu's
    method on the observed scores unless set by hand.

    Args:
        sc: The experiment.
        scrublet_params: The parameters. `None` uses the defaults.
        seed: Seed.
        streaming: Stream the HVG pass. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` cells.
        cells_to_use: 0-based cell indices or ids to run on. `None` uses the
            kept cells.
        group_by: obs column to run per group on, e.g. the sample.
        return_combined_pca: Also return the PCA of observed and simulated
            cells.
        return_pairs: Also return the parents of the simulated doublets.
        verbose: Controls verbosity.

    Returns:
        The `ScrubletRes`.

    Raises:
        ValueError: If `group_by` is missing or incomplete, or a group has
            fewer than `GROUP_MIN_CELLS` cells.

    References:
        Wolock, et al., Cell Syst, 2019.
    """
    params = to_rust(scrublet_params or ScrubletParams())
    cells = _cells(sc, cells_to_use)

    def run(c: np.ndarray) -> dict[str, Any]:
        return _bixverse.scrublet(
            str(sc.genes_path),
            str(sc.cells_path),
            c,
            params,
            seed,
            _streaming(len(c), streaming, verbose),
            return_combined_pca,
            return_pairs,
            verbose,
        )

    if group_by is None:
        res = run(cells)
        return ScrubletRes(
            predicted_doublets=res["predicted_doublets"],
            doublet_scores_obs=res["doublet_scores_obs"],
            doublet_errors_obs=res["doublet_errors_obs"],
            z_scores=res["z_scores"],
            threshold=res["threshold"],
            detected_doublet_rate=res["detected_doublet_rate"],
            detectable_doublet_fraction=res["detectable_doublet_fraction"],
            overall_doublet_rate=res["overall_doublet_rate"],
            doublet_scores_sim=res["doublet_scores_sim"],
            cell_indices=cells,
            pca=res["pca"],
            pair_1=res["pair_1"],
            pair_2=res["pair_2"],
        )

    results, groups, order, labels = _run_grouped(sc, group_by, cells, run, verbose)
    return ScrubletRes(
        predicted_doublets=_concat(results, "predicted_doublets", order),
        doublet_scores_obs=_concat(results, "doublet_scores_obs", order),
        doublet_errors_obs=_concat(results, "doublet_errors_obs", order),
        z_scores=_concat(results, "z_scores", order),
        threshold=_per_group(results, "threshold"),
        detected_doublet_rate=_per_group(results, "detected_doublet_rate"),
        detectable_doublet_fraction=_per_group(results, "detectable_doublet_fraction"),
        overall_doublet_rate=_per_group(results, "overall_doublet_rate"),
        doublet_scores_sim=_per_group(results, "doublet_scores_sim"),
        cell_indices=np.sort(np.concatenate(list(groups.values()))),
        pca=_per_group(results, "pca") if return_combined_pca else None,
        pair_1=_per_group(results, "pair_1") if return_pairs else None,
        pair_2=_per_group(results, "pair_2") if return_pairs else None,
        cell_groups=labels[order].astype(str),
        group_by=group_by,
    )


@beartype
def call_doublets_manual(
    scrublet_res: ScrubletRes,
    threshold: float,
    *,
    for_sample: str | None = None,
    verbose: bool = False,
) -> ScrubletRes:
    """Re-call Scrublet doublets at a hand-picked threshold, as R's
    `call_doublets_manual`.

    Recomputes the calls, the z-scores and the three rates, for the whole
    result or, when grouped, for one group.

    Args:
        scrublet_res: The result.
        threshold: The new threshold, in `[0, 1]`.
        for_sample: The group to update when grouped; `None` takes the first.
            Ignored when not grouped.
        verbose: Log the new rates.

    Returns:
        A new `ScrubletRes`.

    Raises:
        ValueError: If `threshold` is outside `[0, 1]` or `for_sample` is not
            a group.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    res = scrublet_res
    if res.group_by is None:
        mask = np.ones(len(res.doublet_scores_obs), dtype=bool)
        sim = res.doublet_scores_sim
        name = None
    else:
        assert res.cell_groups is not None and isinstance(res.doublet_scores_sim, dict)
        groups = list(dict.fromkeys(res.cell_groups.tolist()))
        name = groups[0] if for_sample is None else for_sample
        if name not in groups:
            raise ValueError(
                f"Sample '{name}' not found. Available: {', '.join(groups)}"
            )
        mask = res.cell_groups == name
        sim = res.doublet_scores_sim[name]
    assert isinstance(sim, np.ndarray)

    obs = res.doublet_scores_obs[mask]
    calls = obs > threshold
    predicted = res.predicted_doublets.copy()
    z = res.z_scores.copy()
    predicted[mask] = calls
    z[mask] = (obs - threshold) / res.doublet_errors_obs[mask]

    detected = float(calls.sum() / len(calls))
    detectable = float((sim > threshold).sum() / len(sim))
    overall = detected / detectable if detectable > MIN_DETECTABLE_FRACTION else 0.0

    if verbose:
        prefix = "" if name is None else f"[{name}] "
        logger.info("%sDetected doublet rate = %.1f%%", prefix, 100 * detected)
        logger.info(
            "%sEstimated detectable doublet fraction = %.1f%%", prefix, 100 * detectable
        )
        logger.info("%sOverall doublet rate: estimated = %.1f%%", prefix, 100 * overall)

    def update(old: GroupScalar, new: float) -> GroupScalar:
        if name is None:
            return new
        assert isinstance(old, dict)
        return {**old, name: new}

    return dataclasses.replace(
        res,
        predicted_doublets=predicted,
        z_scores=z,
        threshold=update(res.threshold, threshold),
        detected_doublet_rate=update(res.detected_doublet_rate, detected),
        detectable_doublet_fraction=update(res.detectable_doublet_fraction, detectable),
        overall_doublet_rate=update(res.overall_doublet_rate, overall),
    )


#########
# Boost #
#########


@beartype
def doublet_detection_boost(
    sc: SingleCells,
    *,
    boost_params: BoostParams | None = None,
    cells_to_use: Selection | None = None,
    group_by: str | None = None,
    seed: int = 42,
    streaming: bool | None = None,
    verbose: bool = False,
) -> BoostRes:
    """Boosted doublet classification, as R's `doublet_detection_boost_sc`.

    Each iteration simulates doublets, clusters observed and simulated cells
    together with Louvain and tests every cluster for doublet enrichment; a
    cell is a doublet if enough iterations vote for it.

    Args:
        sc: The experiment.
        boost_params: The parameters. `None` uses the defaults. With
            `fast_cluster` and no `n_centroids`, `int(sqrt(n_cells) * 4)`
            centroids.
        cells_to_use: 0-based cell indices or ids to run on. `None` uses the
            kept cells.
        group_by: obs column to run per group on, e.g. the sample.
        seed: Seed.
        streaming: Stream the HVG pass. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` cells.
        verbose: Controls verbosity.

    Returns:
        The `BoostRes`.

    Raises:
        ValueError: If `group_by` is missing or incomplete, or a group has
            fewer than `GROUP_MIN_CELLS` cells.
    """
    params = boost_params or BoostParams()
    cells = _cells(sc, cells_to_use)

    def run(c: np.ndarray) -> dict[str, Any]:
        return _bixverse.boost(
            str(sc.genes_path),
            str(sc.cells_path),
            c,
            _with_centroids(params, len(c)),
            seed,
            _streaming(len(c), streaming, verbose),
            verbose,
        )

    if group_by is None:
        res = run(cells)
        return BoostRes(
            doublet=res["doublet"],
            doublet_score=res["doublet_score"],
            voting_avg=res["voting_avg"],
            cell_indices=cells,
        )

    results, groups, order, labels = _run_grouped(sc, group_by, cells, run, verbose)
    return BoostRes(
        doublet=_concat(results, "doublet", order),
        doublet_score=_concat(results, "doublet_score", order),
        voting_avg=_concat(results, "voting_avg", order),
        cell_indices=np.sort(np.concatenate(list(groups.values()))),
        cell_groups=labels[order].astype(str),
        group_by=group_by,
    )


###############
# scDblFinder #
###############


@beartype
def _feature_frame(
    sc: SingleCells, cells: np.ndarray, res: dict[str, Any]
) -> pl.DataFrame | None:
    """The classifier features as a frame, `cell_id` first."""
    if res["feature_mat"] is None:
        return None
    mat = res["feature_mat"]
    return pl.DataFrame(
        {
            "cell_id": sc.cell_ids(cells),
            **{name: mat[:, j] for j, name in enumerate(res["feature_names"])},
        }
    )


@beartype
def scdblfinder(
    sc: SingleCells,
    *,
    scdblfinder_params: ScdblfinderParams | None = None,
    return_features: bool = False,
    cells_to_use: Selection | None = None,
    group_by: str | None = None,
    streaming: bool | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> ScDblFinderRes:
    """Cluster-aware doublet detection, as R's `scdblfinder_sc`.

    Simulates doublets with a bias towards heterotypic pairs, engineers kNN
    and cxds features and trains a gradient-boosted classifier, iterating to
    drop likely doublets from the training labels.

    Args:
        sc: The experiment.
        scdblfinder_params: The parameters. `None` uses the defaults. With
            `fast_cluster` and no `n_centroids`, `int(sqrt(n_cells) * 4)`
            centroids.
        return_features: Also return the classifier features.
        cells_to_use: 0-based cell indices or ids to run on. `None` uses the
            kept cells.
        group_by: obs column to run per group on, e.g. the sample.
        streaming: Stream the gene selection pass. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` cells.
        seed: Seed.
        verbose: Controls verbosity.

    Returns:
        The `ScDblFinderRes`.

    Raises:
        ValueError: If `group_by` is missing or incomplete, or a group has
            fewer than `GROUP_MIN_CELLS` cells.

    References:
        Germain, et al., F1000Research, 2022.
    """
    params = scdblfinder_params or ScdblfinderParams()
    cells = _cells(sc, cells_to_use)

    def run(c: np.ndarray) -> dict[str, Any]:
        res = _bixverse.scdblfinder(
            str(sc.genes_path),
            str(sc.cells_path),
            c,
            _with_centroids(params, len(c)),
            return_features,
            seed,
            _streaming(len(c), streaming, verbose),
            verbose,
        )
        res["features"] = _feature_frame(sc, c, res)
        return res

    if group_by is None:
        res = run(cells)
        return ScDblFinderRes(
            predicted_doublets=res["predicted_doublets"],
            doublet_score=res["doublet_scores"],
            cxds_scores=res["cxds_scores"],
            weighted=res["weighted"],
            threshold=res["threshold"],
            cluster_labels=res["cluster_labels"],
            detected_doublet_rate=res["detected_doublet_rate"],
            cell_indices=cells,
            features=res["features"],
        )

    results, groups, order, labels = _run_grouped(sc, group_by, cells, run, verbose)
    clusters = np.concatenate(
        [
            np.asarray([f"{g}_{c}" for c in r["cluster_labels"].tolist()])
            for g, r in results.items()
        ]
    )
    features = (
        pl.concat([r["features"] for r in results.values()])[order]
        if return_features
        else None
    )
    return ScDblFinderRes(
        predicted_doublets=_concat(results, "predicted_doublets", order),
        doublet_score=_concat(results, "doublet_scores", order),
        cxds_scores=_concat(results, "cxds_scores", order),
        weighted=_concat(results, "weighted", order),
        threshold=_per_group(results, "threshold"),
        cluster_labels=clusters[order],
        detected_doublet_rate=_per_group(results, "detected_doublet_rate"),
        cell_indices=np.sort(np.concatenate(list(groups.values()))),
        features=features,
        cell_groups=labels[order].astype(str),
        group_by=group_by,
    )


###########
# Getters #
###########


@beartype
def get_data(res: ScrubletRes | BoostRes | ScDblFinderRes) -> pl.DataFrame:
    """The per-cell calls as an obs frame for `SingleCells.add_obs`, R's
    `get_data`.

    Columns are R's, with `cell_idx` 0-based: `doublet` and `doublet_score`
    for Scrublet; every per-cell field for Boost (plus `cell_groups` when
    grouped); `predicted_doublets`, `doublet_score`, `cxds_scores`,
    `weighted` and `cluster_labels` for scDblFinder.
    """
    match res:
        case ScrubletRes():
            cols: dict[str, object] = {
                "doublet": res.predicted_doublets,
                "doublet_score": res.doublet_scores_obs,
            }
        case BoostRes():
            cols = {
                "doublet": res.doublet,
                "doublet_score": res.doublet_score,
                "voting_avg": res.voting_avg,
            }
            if res.cell_groups is not None:
                cols["cell_groups"] = res.cell_groups
        case ScDblFinderRes():
            cols = {
                "predicted_doublets": res.predicted_doublets,
                "doublet_score": res.doublet_score,
                "cxds_scores": res.cxds_scores,
                "weighted": res.weighted,
                "cluster_labels": res.cluster_labels,
            }
        case _:
            raise TypeError(f"No get_data for {type(res).__name__}")
    cols["cell_idx"] = res.cell_indices.astype(np.uint32)
    return pl.DataFrame(cols)


@beartype
def get_feature_mat(res: ScDblFinderRes) -> pl.DataFrame | None:
    """The scDblFinder classifier features, R's `get_feature_mat`.

    Warns and returns `None` if they were not asked for.
    """
    if res.features is None:
        warnings.warn(
            "You did not extract the features during run of the function. "
            "Returning None.",
            stacklevel=2,
        )
    return res.features


@beartype
def get_scores(res: ScDblFinderRes, score_type: ScoreType = "weighted") -> np.ndarray:
    """The weighted or cxds scores, R's `get_scores` on a `ScDblFinderRes`."""
    return res.weighted if score_type == "weighted" else res.cxds_scores


##################
# Empty droplets #
##################


@beartype
def infer_empty_droplets(
    lib_size: np.ndarray | pl.Series | Sequence[int] | Sequence[float],
    empty_params: EmptyDropletsParams,
) -> np.ndarray:
    """Call empty droplets from the library sizes, R's
    `rs_sc_infer_empty_droplets`.

    Library sizes are truncated to integers as R's `as.integer`; missing and
    negative ones count as zero.

    Args:
        lib_size: Library size per barcode.
        empty_params: How to call them; `"umi_cutoff"`, `"expected_cells"`
            or `"knee"`.

    Returns:
        A `bool` array, `True` where the barcode is empty.

    Raises:
        ValueError: With `method = "supplied"`, which has nothing to infer.
    """
    if empty_params.method == "supplied":
        raise ValueError(
            "method = 'supplied' reads the mask from obs; nothing to infer."
        )
    sizes = pl.Series(lib_size, dtype=pl.Float64).fill_null(0).fill_nan(0).to_numpy()
    sizes = np.clip(np.trunc(sizes), 0, np.iinfo(np.uint32).max).astype(np.uint32)
    return _bixverse.empty_droplets(sizes, to_rust(empty_params))


@beartype
def _resolve_empty_droplets(
    obs: pl.DataFrame, empty_params: EmptyDropletsParams, verbose: bool
) -> np.ndarray:
    """R's `.resolve_empty_droplets`: the mask from obs or inferred.

    Raises:
        ValueError: If the supplied column is missing, not boolean or holds
            nulls.
    """
    if empty_params.method == "supplied":
        col = empty_params.is_empty_column
        assert col is not None
        if col not in obs.columns:
            raise ValueError(f"'{col}' is not a column in the obs table.")
        if obs[col].dtype != pl.Boolean:
            raise ValueError(f"'{col}' must be a logical column.")
        if obs[col].null_count():
            raise ValueError(f"'{col}' contains NAs.")
        return obs[col].to_numpy()
    is_empty = infer_empty_droplets(obs["lib_size"], empty_params)
    if verbose:
        logger.info(
            "Empty droplets (method '%s'): %i of %i barcodes called empty.",
            empty_params.method,
            int(is_empty.sum()),
            obs.height,
        )
    return is_empty


#############
# CellSweep #
#############


@beartype
def _celltype_codes(values: pl.Series) -> tuple[np.ndarray, list[str]]:
    """R's `as.factor` with NA: 0-based codes (-1 for null) and the levels.

    An Enum keeps all its levels, unused ones too; anything else takes the
    sorted distinct values.
    """
    if isinstance(values.dtype, pl.Enum):
        levels = values.dtype.categories.to_list()
        strings = values.cast(pl.String).to_list()
    else:
        strings = _as_character(values)
        levels = sorted({v for v in strings if v is not None})
    lookup = {v: i for i, v in enumerate(levels)}
    codes = np.asarray(
        [-1 if v is None else lookup[v] for v in strings], dtype=np.int64
    )
    return codes, levels


@beartype
def cellsweep(
    directory: Path | str,
    sc: SingleCells,
    celltype_column: str,
    sample_column: str,
    empty_params: EmptyDropletsParams,
    *,
    cellsweep_params: CellsweepParams | None = None,
    qc: MinQualityParams | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Remove ambient and bulk contamination with CellSweep, as R's
    `cellsweep_sc`.

    One EM fit per sample splits every count three ways (ambient, bulk,
    true expression); the denoised counts go to a new experiment. It runs
    after annotation, and the experiment needs its empty droplets, so ingest
    with a permissive `MinQualityParams(min_unique_genes=0, min_lib_size=0,
    min_cells=0)`.

    Barcodes split three ways: empty droplets, annotated barcodes that passed
    QC, and the rest, which is left out of the fit and the output. The raw
    layer takes the stochastically rounded counts, the normalised layer the
    float magnitudes. As in R, obs is the input's (including its now stale
    `nnz` and `lib_size`) plus `cellsweep_alpha`, `cellsweep_z`,
    `cellsweep_beta`, `cellsweep_ll` and `cellsweep_converged`; var gains a
    fresh `no_cells_exp` and `cellsweep_ambient`, the mean ambient profile
    over the samples.

    Args:
        directory: The target directory, created if missing.
        sc: The raw experiment, empty droplets included.
        celltype_column: obs column with the cell-type labels.
        sample_column: obs column with the emulsion; one fit per value.
        empty_params: How to find the empty droplets.
        cellsweep_params: The model parameters. `None` uses the defaults.
        qc: Only `target_size` is used, to scale the normalised layer.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
            `None` converts in one pass.
        verbose: Controls verbosity.

    Returns:
        The denoised experiment.

    Raises:
        ValueError: If the target is the input directory, a column is
            missing, the empty droplets were filtered at ingest, or no
            barcode is both annotated and kept.

    References:
        Sullivan, et al., CellSweep, 2025.
    """
    params = cellsweep_params or CellsweepParams()
    qc = qc or MinQualityParams()
    target = Path(directory).expanduser()
    if target.resolve() == sc.dir.resolve():
        raise ValueError("The target and the input must be different directories.")

    with db.connect(sc.db_path, read_only=True) as con:
        obs_cols = db.columns(con, "obs")
    for col in (celltype_column, sample_column):
        if col not in obs_cols:
            raise ValueError(f"'{col}' is not a column in the obs table.")

    obs = sc.obs(filtered=False)
    if "lib_size" not in obs.columns:
        raise ValueError(
            "The obs table has no `lib_size` column; re-ingest the object."
        )
    lib = obs["lib_size"].drop_nulls().to_numpy()
    if params.freeze_ambient_profile and lib.size:
        min_lib = int(lib.min())
        if min_lib > CELLSWEEP_MAX_MIN_LIB_SIZE:
            raise ValueError(
                f"The smallest library size in the object is {min_lib}, so "
                "the empty droplets were filtered out at ingest and the ambient "
                "profile cannot be estimated. Re-ingest with "
                "`MinQualityParams(min_unique_genes=0, min_lib_size=0, "
                "min_cells=0)`, or set `freeze_ambient_profile=False`."
            )

    is_empty = _resolve_empty_droplets(obs, empty_params, verbose)
    keep = np.zeros(obs.height, dtype=bool)
    keep[sc.cells_to_keep] = True
    codes, levels = _celltype_codes(obs[celltype_column])
    real = keep & (codes >= 0) & ~is_empty
    dropped = int((~real & ~is_empty).sum())
    if verbose and dropped:
        logger.info(
            "%i barcodes are neither empty nor annotated-and-passing-QC. They are "
            "excluded from the fit and from the output.",
            dropped,
        )
    if not real.any():
        raise ValueError("No barcode is both annotated and passing QC.")

    sample_id = np.asarray(_as_character(obs[sample_column]), dtype=object)
    samples = []
    for sid in sorted({s for s in sample_id[real] if s is not None}):
        in_sample = sample_id == sid
        real_idx = np.flatnonzero(in_sample & real)
        samples.append(
            {
                "sample_id": sid,
                "real_cells": real_idx.tolist(),
                "empty_cells": np.flatnonzero(in_sample & is_empty).tolist(),
                "celltype_idx": codes[real_idx].tolist(),
                "n_celltypes": len(levels),
            }
        )
    if verbose:
        logger.info(
            "Running CellSweep over %i samples, %i barcodes, %i empty droplets.",
            len(samples),
            int(real.sum()),
            int(is_empty.sum()),
        )

    path = _prepare(target)
    res = _bixverse.cellsweep(
        str(sc.cells_path),
        str(path / CELLS_BIN),
        samples,
        to_rust(params),
        qc.target_size,
        verbose,
    )
    if verbose:
        logger.info("Generating gene-based binary.")
    _bixverse.generate_gene_based_data(
        path / CELLS_BIN, path / GENES_BIN, csc_mem_gb, verbose
    )

    fits = res["fits"]
    diagnostics = pl.concat(
        [
            pl.DataFrame(
                {
                    "cellsweep_alpha": fit["alpha"],
                    "cellsweep_z": [levels[k] for k in fit["z_hat"].tolist()],
                    "cellsweep_beta": np.full(len(fit["alpha"]), fit["beta"]),
                    "cellsweep_ll": np.full(len(fit["alpha"]), fit["log_likelihood"]),
                    "cellsweep_converged": np.full(len(fit["alpha"]), fit["converged"]),
                },
                schema_overrides={"cellsweep_z": pl.String},
            )
            for fit in fits
        ]
    )
    ambient = np.mean(np.stack([fit["ambient"] for fit in fits]), axis=0)
    with db.connect(db.db_path(path)) as con:
        db.obs_from_duckdb_subset(con, sc.db_path, res["cell_order"])
        db.var_from_duckdb_reordered(con, sc.db_path, sc.gene_ids().tolist())
        db.add_data(con, "obs", diagnostics)
        db.add_data(
            con,
            "var",
            pl.DataFrame(
                {
                    "no_cells_exp": _bixverse.gene_nnz(path / GENES_BIN).astype(
                        np.int32
                    ),
                    "cellsweep_ambient": ambient,
                }
            ),
        )
        db.set_to_keep_column(con)
    return SingleCells(path)
