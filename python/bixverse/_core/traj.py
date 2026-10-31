"""Trajectory inference: Palantir, PAGA and gene trends.

The port of R's `run_palantir_sc`, `run_paga_sc` and `run_gene_trends_sc`
(`R/methods_sc_trajectory.R`). Palantir and PAGA only read the cached kNN
graph, so all three take a `SingleCells`, a `SingleCellsSubset` or a
`MetaCells`, as in R. The work runs in Rust.

Each returns a frozen record, R's `PalantirRes`, `PagaRes` and
`GeneTrendsRes`, keyed by cell id. Nothing is written to obs, as in R; to
colour by pseudotime, join `res.pseudotime` onto obs on `cell_id`.

    >>> res = run_palantir(sc, early_cell="cell_0001")
    >>> paga = run_paga(sc, "leiden")
    >>> trends = run_gene_trends(sc, res, features=["Gata1", "Mpo"])

Cell ids in and out are the ones the kNN graph was built over: the kept
cells in `cells_to_keep` order, or the meta cells in row order. The PAGA
graphs are dense `float64` cluster by cluster arrays rather than R's sparse
matrices; they are small and the values are the same.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells
from bixverse._core.multimodal import (
    Modality,
    check_modality,
    modality_cache,
    require_fresh,
)
from bixverse._core.pp import _factor_codes
from bixverse._core.single_cells import get_counts
from bixverse._core.state import KnnGraph
from bixverse._core.subset import ScOrMc
from bixverse.params import (
    BranchSelectionParams,
    GeneTrendsParams,
    PalantirParams,
    to_rust,
)

###########
# Globals #
###########


# kNN metrics the diffusion kernel treats as a metric without a warning
METRIC_DISTANCES: tuple[str, ...] = ("euclidean", "cosine")

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class PalantirRes:
    """A Palantir run, R's `PalantirRes`.

    Attributes:
        pseudotime: `cell_id`, `pseudotime` (min-max scaled to `[0, 1]`; the
            start cell is not pinned to 0) and `entropy` per cell.
        branch_probs: `cell_id` and one column per terminal state with the
            fate probabilities. Rows need not sum to one, as sub-threshold
            values are zeroed without renormalisation.
        terminal_states: Branch name to terminal state cell id, in the column
            order of `branch_probs`. The name is the caller's label, or the
            cell id itself for unlabelled or detected states.
        waypoints: The waypoint cell ids; the first is the start cell.
        start_cell: The start cell actually used.
        multiscale: `cell_id` and the multiscale diffusion components
            `MS_1`, `MS_2`, ...
        run_info: `iterations`, `converged`, `eigen_converged`,
            `eigen_residual`, `repair_edges`, `stranded_waypoints`,
            `n_waypoints` and `modality`. An `eigen_converged` of `False`
            means the diffusion eigensolve ran out of restarts.
    """

    pseudotime: pl.DataFrame
    branch_probs: pl.DataFrame
    terminal_states: dict[str, str]
    waypoints: tuple[str, ...]
    start_cell: str
    multiscale: pl.DataFrame
    run_info: dict[str, object]

    @property
    def branches(self) -> list[str]:
        """The branch names, the fate columns of `branch_probs`."""
        return list(self.terminal_states)

    def __repr__(self) -> str:
        """R's `print.PalantirRes`."""
        info = self.run_info
        n_states = len(self.terminal_states)
        lines = [
            f"PalantirRes: {self.pseudotime.height} cells, {n_states} terminal "
            f"{'state' if n_states == 1 else 'states'} ({info['modality']} modality)",
            f"  Start cell: {self.start_cell} | waypoints: {info['n_waypoints']}",
            f"  Converged: {info['converged']} ({info['iterations']} iterations)",
            f"  Repair edges: {info['repair_edges']}, stranded waypoints: "
            f"{info['stranded_waypoints']}",
        ]
        if info["repair_edges"]:
            lines.append(
                "  Note: the kNN graph was disconnected. Consider a larger knn."
            )
        if not info["eigen_converged"]:
            lines.append(
                "  Note: the diffusion eigensolve did not converge (residual "
                f"{info['eigen_residual']:.2e}).\n         The embedding is "
                "under-resolved, raise lanczos_max_restarts."
            )
        return "\n".join(lines)


@beartype
@dataclass(frozen=True, repr=False)
class PagaRes:
    """A PAGA graph abstraction, R's `PagaRes`.

    Attributes:
        connectivities: `(n_clusters, n_clusters)` `float64`, symmetric with a
            zero diagonal, values in `(0, 1]` on the edges.
        connectivities_tree: The maximum spanning forest of
            `connectivities`, carrying its values on the retained edges.
        clusters: The cluster levels, the rows and columns of both graphs.
        sizes: `cluster` and `n_cells`.
        params: `cluster_col` and `modality`.
    """

    connectivities: np.ndarray
    connectivities_tree: np.ndarray
    clusters: tuple[str, ...]
    sizes: pl.DataFrame
    params: dict[str, str]

    def __repr__(self) -> str:
        """R's `print.PagaRes`; both graphs are symmetric, hence the halving."""
        return "\n".join(
            [
                f"PagaRes: {self.sizes.height} clusters, "
                f"{int(self.sizes['n_cells'].sum())} cells "
                f"({self.params['modality']} modality)",
                f"  Clustering: {self.params['cluster_col']}",
                f"  Abstracted edges: {np.count_nonzero(self.connectivities) / 2:.0f}",
                "  Spanning forest edges: "
                f"{np.count_nonzero(self.connectivities_tree) / 2:.0f}",
            ]
        )


@beartype
@dataclass(frozen=True, repr=False)
class GeneTrendsRes:
    """Gene trends along the Palantir branches, R's `GeneTrendsRes`.

    Attributes:
        trends: Long frame with `branch` and `gene` (Enums in branch and
            feature order), `pseudotime` and `expression`, one row per grid
            point, gene and branch.
        branch_cells: Branch name to the cell ids selected for it.
        branches: The branch names, the fate columns of the Palantir result.
        params: `use_magic`, `branch_params` and `gene_trend_params`.
        run_info: `n_cells` and `jitter_used` per branch, and the grid
            `resolution`.
    """

    trends: pl.DataFrame
    branch_cells: dict[str, tuple[str, ...]]
    branches: tuple[str, ...]
    params: dict[str, object]
    run_info: dict[str, object]

    def __repr__(self) -> str:
        """R's `print.GeneTrendsRes`."""
        info = self.run_info
        n_cells = info["n_cells"]
        jitter = info["jitter_used"]
        assert isinstance(n_cells, dict) and isinstance(jitter, dict)
        gp = self.params["gene_trend_params"]
        assert isinstance(gp, GeneTrendsParams)
        n_b = len(self.branches)
        source = (
            "MAGIC imputed counts" if self.params["use_magic"] else "normalised counts"
        )
        lines = [
            f"GeneTrendsRes: {n_b} {'branch' if n_b == 1 else 'branches'}, "
            f"{self.trends['gene'].n_unique()} genes, {info['resolution']} grid "
            "points",
            "  Cells per branch: "
            + ", ".join(f"{b} ({n_cells[b]})" for b in self.branches),
            f"  Source: {source}",
            f"  Length scale: {gp.length_scale:g} | sigma: {gp.sigma:g}",
        ]
        if any(j > gp.jitter for j in jitter.values()):
            lines.append(
                "  Note: the Cholesky needed extra jitter on at least one branch."
            )
        return "\n".join(lines)


###########
# Helpers #
###########


@beartype
def _cell_names(sc: ScOrMc) -> np.ndarray:
    """The cell ids in kNN row order: kept cells or meta cells."""
    if isinstance(sc, MetaCells):
        return sc.cell_ids()
    return sc.cell_ids(filtered=True)


@beartype
def _knn_or_warn(sc: ScOrMc, n_cells: int, modality: Modality) -> KnnGraph | None:
    """The cached kNN of a modality, R's hard state tier first.

    Raises:
        ValueError: If it is stale or does not cover the cells, or on a
            non-RNA modality of a single modality object.
    """
    check_modality(sc, modality)
    require_fresh(sc, ["knn"], modality)
    knn = modality_cache(sc, modality).knn
    if knn is None:
        warnings.warn(
            "No kNN data found. Did you run find_neighbours(). Returning None.",
            stacklevel=3,
        )
        return None
    if knn.indices.shape[0] != n_cells:
        raise ValueError(
            f"The kNN graph covers {knn.indices.shape[0]} cells but there are "
            f"{n_cells}. Regenerate the neighbours so the two agree."
        )
    return knn


@beartype
def _cells_to_knn_idx(
    cell_ids: Sequence[str], used_cells: np.ndarray, arg_name: str
) -> np.ndarray:
    """0-based kNN rows of cell ids, R's `.cells_to_knn_idx`.

    Raises:
        ValueError: If a cell is not in the kNN graph.
    """
    pos = {c: i for i, c in enumerate(used_cells.tolist())}
    missing = [c for c in cell_ids if c not in pos]
    if missing:
        raise ValueError(
            f"`{arg_name}`: {len(missing)} cell(s) not found in the kNN graph, "
            f"e.g. '{missing[0]}'."
        )
    return np.asarray([pos[c] for c in cell_ids], dtype=np.uint32)


@beartype
def _as_factor(values: pl.Series, name: str) -> tuple[np.ndarray, list[str]]:
    """0-based codes and levels of R's `as.factor`: an Enum keeps every
    level, empty ones included; anything else is `factor()`.

    Raises:
        ValueError: On a missing value.
    """
    if isinstance(values.dtype, pl.Enum):
        if values.null_count():
            raise ValueError(f"Column '{name}' holds missing values.")
        return (
            values.to_physical().to_numpy().astype(np.uint32),
            values.dtype.categories.to_list(),
        )
    return _factor_codes(values, name)


@beartype
def _cluster_column(sc: ScOrMc, cluster_col: str) -> pl.Series:
    """An obs column in kNN row order.

    Raises:
        ValueError: If the column is missing.
    """
    if cluster_col not in sc.obs_columns():
        raise ValueError(f"Column '{cluster_col}' not found in the obs table.")
    if isinstance(sc, MetaCells):
        return sc.obs(cols=[cluster_col])[cluster_col]
    obs = sc.obs(cols=["cell_idx", cluster_col], filtered=True)
    order = pl.DataFrame({"cell_idx": sc.cells_to_keep})
    return order.join(obs, on="cell_idx", how="left", maintain_order="left")[
        cluster_col
    ]


############
# Palantir #
############


@beartype
def run_palantir(
    sc: ScOrMc,
    early_cell: str,
    *,
    terminal_states: Sequence[str] | dict[str, str] | None = None,
    modality: Modality = "rna",
    palantir_params: PalantirParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> PalantirRes | None:
    """Palantir trajectory inference over the cached kNN graph, as R's
    `run_palantir_sc`.

    Models differentiation as a Markov chain over a diffusion map of the
    cells: a pseudotime per cell, the probability of each cell reaching each
    terminal state, and the differentiation entropy of those probabilities.
    The kNN graph feeds the diffusion kernel; the geodesics are measured over
    a second kNN graph built on the multiscale space, sized by
    `palantir_params.knn`.

    Args:
        sc: The experiment, a subset or meta cells, with the kNN.
        early_cell: Id of the start cell, one the kNN was built over.
        terminal_states: Terminal state cell ids, or label to cell id, the
            labels then naming the `branch_probs` columns. `None` detects
            them from the waypoint Markov chain.
        modality: The kNN graph: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
        palantir_params: The parameters. `None` uses the defaults.
        seed: Seed for the eigensolver, the waypoints and the multiscale kNN.
        verbose: Controls verbosity.

    Returns:
        The `PalantirRes`, or `None` with a warning if there is no kNN.

    Raises:
        ValueError: On a stale kNN, a non-RNA modality without a multi-modal
            object, or a cell not in the kNN graph.

    References:
        Setty, et al., Nat. Biotechnol., 2019.
    """
    params = palantir_params or PalantirParams()
    used = _cell_names(sc)
    knn = _knn_or_warn(sc, len(used), modality)
    if knn is None:
        return None
    if knn.dist_metric not in METRIC_DISTANCES:
        warnings.warn(
            f"The kNN graph uses '{knn.dist_metric}', which is not a metric. The "
            "diffusion map is built on it regardless, but interpret the "
            "pseudotime with care.",
            stacklevel=2,
        )
    early = _cells_to_knn_idx([early_cell], used, "early_cell")
    labels: dict[str, str] = {}
    terminal = None
    if terminal_states is not None:
        if isinstance(terminal_states, dict):
            labels = {cell: label for label, cell in terminal_states.items()}
            cells = list(terminal_states.values())
        else:
            cells = list(terminal_states)
        terminal = _cells_to_knn_idx(cells, used, "terminal_states")
    if verbose:
        logger.info(
            "Running Palantir over the %s kNN graph (%i cells).", modality, len(used)
        )

    res = _bixverse.palantir(
        np.ascontiguousarray(knn.indices),
        np.ascontiguousarray(knn.distances),
        int(early[0]),
        terminal,
        to_rust(params),
        seed,
        verbose,
    )

    states = used[res["terminal_states"].astype(np.int64)].tolist()
    names = [labels.get(c, c) for c in states]
    probs = res["branch_probs"]
    ms = res["multiscale"]
    waypoints = tuple(used[res["waypoints"].astype(np.int64)].tolist())
    return PalantirRes(
        pseudotime=pl.DataFrame(
            {
                "cell_id": used,
                "pseudotime": res["pseudotime"],
                "entropy": res["entropy"],
            }
        ),
        branch_probs=pl.DataFrame(
            {"cell_id": used, **{n: probs[:, j] for j, n in enumerate(names)}}
        ),
        terminal_states=dict(zip(names, states, strict=True)),
        waypoints=waypoints,
        start_cell=str(used[res["start_cell"]]),
        multiscale=pl.DataFrame(
            {"cell_id": used, **{f"MS_{j + 1}": ms[:, j] for j in range(ms.shape[1])}}
        ),
        run_info={
            "iterations": res["iterations"],
            "converged": res["converged"],
            "eigen_converged": res["eigen_converged"],
            "eigen_residual": res["eigen_residual"],
            "repair_edges": res["repair_edges"],
            "stranded_waypoints": res["stranded_waypoints"],
            "n_waypoints": len(waypoints),
            "modality": modality,
        },
    )


########
# PAGA #
########


@beartype
def run_paga(
    sc: ScOrMc,
    cluster_col: str,
    *,
    modality: Modality = "rna",
    verbose: bool = False,
) -> PagaRes | None:
    """PAGA graph abstraction of the cached kNN graph, as R's `run_paga_sc`.

    Abstracts the kNN graph into a cluster graph whose edge weights measure
    how much more two clusters connect than a random null model expects,
    plus its maximum spanning forest. Only the kNN indices are read.

    Args:
        sc: The experiment, a subset or meta cells, with the kNN and a
            clustering.
        cluster_col: obs column with the clusters. Empty levels of an Enum
            are kept; other columns are sorted into levels.
        modality: The kNN graph: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
        verbose: Controls verbosity.

    Returns:
        The `PagaRes`, or `None` with a warning if there is no kNN.

    Raises:
        ValueError: On a stale kNN, a non-RNA modality without a multi-modal
            object, a missing column or missing cluster labels.

    References:
        Wolf, et al., Genome Biol., 2019.
    """
    n = len(_cell_names(sc))
    knn = _knn_or_warn(sc, n, modality)
    if knn is None:
        return None
    codes, levels = _as_factor(_cluster_column(sc, cluster_col), cluster_col)
    if verbose:
        logger.info(
            "Running PAGA over the %s kNN graph (%i cells, %i clusters).",
            modality,
            n,
            len(levels),
        )
    res = _bixverse.paga(np.ascontiguousarray(knn.indices), codes, len(levels))
    return PagaRes(
        connectivities=res["connectivities"],
        connectivities_tree=res["connectivities_tree"],
        clusters=tuple(levels),
        sizes=pl.DataFrame(
            {
                "cluster": pl.Series(levels, dtype=pl.String),
                "n_cells": pl.Series(res["sizes"], dtype=pl.UInt32),
            }
        ),
        params={"cluster_col": cluster_col, "modality": modality},
    )


###############
# Gene trends #
###############


@beartype
def _trend_expression(
    sc: ScOrMc,
    features: Sequence[str] | None,
    use_magic: bool,
    obs_cells: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """The cells x genes `float64` matrix a trend fit runs over, rows in
    `obs_cells` order, R's `.gene_trends_expression`.

    Raises:
        ValueError: On MAGIC for meta cells, a missing or stale imputed layer,
            or genes it does not hold.
    """
    if not use_magic:
        assert features is not None
        genes = sc.gene_indices(list(features))
        # rows come back in kNN row order, which is `obs_cells`
        if isinstance(sc, MetaCells):
            m = sc.counts(genes=genes, layer="norm")
        else:
            m = get_counts(sc, genes=genes, layer="norm").matrix
        dense = np.zeros(m.shape, dtype=np.float64)
        rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr.astype(np.int64)))
        dense[rows, m.indices.astype(np.int64)] = m.data
        return dense, sc.gene_ids(genes).tolist()

    if isinstance(sc, MetaCells):
        raise ValueError(
            "MAGIC is only available for SingleCells and SingleCellsSubset. Set "
            "use_magic=False."
        )
    state = sc.check_state(["magic"])
    if state is not True:
        raise ValueError(state)
    magic = sc.cache.magic
    if magic is None:
        raise ValueError(
            "No imputed layer found. Run run_magic() first, or set use_magic=False."
        )
    held = magic.features.tolist()
    features = list(features) if features is not None else held
    missing = [g for g in features if g not in held]
    if missing:
        raise ValueError(
            f"{len(missing)} gene(s) are not in the imputed layer, e.g. "
            f"'{missing[0]}'. Re-run run_magic() with the full set you want to fit."
        )
    if magic.data.shape[0] != len(obs_cells):
        raise ValueError(
            f"The imputed layer covers {magic.data.shape[0]} cells but "
            f"{len(obs_cells)} are kept. Re-run run_magic()."
        )
    col = {g: j for j, g in enumerate(held)}
    return (
        magic.data[:, [col[g] for g in features]].astype(np.float64),
        features,
    )


@beartype
def run_gene_trends(
    sc: ScOrMc,
    palantir_res: PalantirRes,
    *,
    features: Sequence[str] | None = None,
    use_magic: bool = False,
    branch_params: BranchSelectionParams | None = None,
    gene_trend_params: GeneTrendsParams | None = None,
    verbose: bool = False,
) -> GeneTrendsRes:
    """Gene trends over the Palantir pseudotime, as R's `run_gene_trends_sc`.

    Assigns cells to branches from their fate probabilities, then fits a
    landmark Gaussian process with a Matern-5/2 kernel per branch, the
    Mellon-based estimator of the reference. One smooth curve per gene and
    branch on a common grid.

    The defaults are prior-dominated: the pseudotime spans `[0, 1]`, so a
    `length_scale` of 1 covers the whole domain. Shorten it before believing
    a bump.

    Args:
        sc: The experiment, a subset or meta cells, the one Palantir ran on.
        palantir_res: The `run_palantir` result over the same cells.
        features: Gene ids to fit. `None` takes every gene of the imputed
            layer, so needs `use_magic`. Unknown ids warn and drop.
        use_magic: Fit the MAGIC imputed layer instead of the normalised
            counts; `SingleCells` and `SingleCellsSubset` only.
        branch_params: Branch cell selection. `None` uses the defaults.
        gene_trend_params: The Gaussian process. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The `GeneTrendsRes`.

    Raises:
        ValueError: If `features` is missing without MAGIC, the Palantir result
            does not cover the cells, or the imputed layer is unusable.

    References:
        Setty, et al., Nat. Biotechnol., 2019.
    """
    branch_params = branch_params or BranchSelectionParams()
    gene_trend_params = gene_trend_params or GeneTrendsParams()
    if features is None and not use_magic:
        raise ValueError("`features` is only optional when `use_magic=True`.")

    # the Palantir result is keyed by cell id, so align by id, not by order
    obs_cells = _cell_names(sc)
    pos = {c: i for i, c in enumerate(palantir_res.pseudotime["cell_id"].to_list())}
    missing = [c for c in obs_cells.tolist() if c not in pos]
    if missing:
        raise ValueError(
            f"{len(missing)} of the object's cells are missing from the Palantir "
            f"result, e.g. '{missing[0]}'. The two disagree on their cell set; "
            "re-run run_palantir() on this object."
        )
    idx = np.asarray([pos[c] for c in obs_cells.tolist()], dtype=np.int64)

    expr, genes = _trend_expression(sc, features, use_magic, obs_cells)
    branches = palantir_res.branches
    probs = np.ascontiguousarray(
        palantir_res.branch_probs.select(branches).to_numpy()[idx], dtype=np.float32
    )
    pt = np.ascontiguousarray(
        palantir_res.pseudotime["pseudotime"].to_numpy()[idx], dtype=np.float32
    )
    if verbose:
        logger.info(
            "Fitting gene trends over %i %s (%i cells, %i genes, %s).",
            len(branches),
            "branch" if len(branches) == 1 else "branches",
            expr.shape[0],
            expr.shape[1],
            "imputed counts" if use_magic else "normalised counts",
        )

    res = _bixverse.gene_trends(
        np.ascontiguousarray(expr),
        pt,
        probs,
        to_rust(branch_params),
        to_rust(gene_trend_params),
    )

    # each trend is grid x genes; R unrolls it column major, gene by gene
    branch_enum = pl.Enum(branches)
    gene_enum = pl.Enum(genes)
    frames = []
    for b, mat, grid in zip(branches, res["trends"], res["grids"], strict=True):
        n_grid = mat.shape[0]
        frames.append(
            pl.DataFrame(
                {
                    "branch": pl.Series([b] * (n_grid * len(genes)), dtype=branch_enum),
                    "pseudotime": np.tile(grid, len(genes)),
                    "gene": pl.Series(np.repeat(genes, n_grid), dtype=gene_enum),
                    "expression": mat.T.ravel(),
                }
            )
        )
    return GeneTrendsRes(
        trends=pl.concat(frames),
        branch_cells={
            b: tuple(obs_cells[c.astype(np.int64)].tolist())
            for b, c in zip(branches, res["branch_cells"], strict=True)
        },
        branches=tuple(branches),
        params={
            "use_magic": use_magic,
            "branch_params": branch_params,
            "gene_trend_params": gene_trend_params,
        },
        run_info={
            "n_cells": dict(zip(branches, res["n_cells"], strict=True)),
            "jitter_used": dict(zip(branches, res["jitter_used"], strict=True)),
            "resolution": res["trends"][0].shape[0],
        },
    )
