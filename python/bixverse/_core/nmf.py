"""HALS NMF over single cells and meta cells: one run, stabilised restarts,
consensus NMF and the consensus k sweep.

The work runs in Rust; the result classes are `NmfResult`,
`StabilisedNmfResult`, `ConsensusNmfResult` and `NmfKSweepResult`.

    >>> sweep = nmf_k_sweep(sc, k_range=range(3, 9))
    >>> res = consensus_nmf(sc, k=5)
    >>> sc.add_obs(get_data(res))

The factorisation is genes x cells: `w` holds the gene loadings (genes x k),
`h` the cell usages (k x cells). Genes default to the HVGs. Every
index is 0-based, the run and component numbers of the consensus clusters
included; the `component_id` strings are 1-based (`run_01.comp_01`).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells, csr_args
from bixverse._core.repr import compact_repr
from bixverse._core.subset import ScOrMc
from bixverse._validate import Verbose
from bixverse.params import NmfConsensusParams, NmfHalsParams, to_rust

###########
# Globals #
###########

Preprocessing = Literal["none", "sd", "sqrt_sd"]

# above this many cells `consensus_target = "w"` clusters a dense
# `(k * n_runs) x n_cells` matrix, so it warns
CELL_SPACE_CUTOFF: int = 20000

CONSENSUS_ADVICE: str = (
    "\nOptions: raise `density_threshold` in NmfConsensusParams (2 switches the "
    "filter off entirely), increase `n_runs`, or run the k sweep first to find a "
    "k the restarts actually agree on."
)

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class NmfResult:
    """A single HALS NMF run.

    Attributes:
        w: `(n_genes, k)` gene loadings.
        h: `(k, n_cells)` cell usages.
        gene_ids: The genes, the rows of `w`.
        cell_ids: The cells (meta cells), the columns of `h`.
        cell_indices: 0-based store index (meta cell row) of each cell.
        final_loss: The final reconstruction loss.
        n_iter: Iterations run; `None` for a run taken from a stabilised fit.
        converged: Whether the tolerance was met.
        source_class: `"SingleCells"` or `"MetaCells"`.
        params: The parameters of the run.
    """

    w: np.ndarray
    h: np.ndarray
    gene_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    cell_indices: np.ndarray
    final_loss: float
    n_iter: int | None
    converged: bool
    source_class: str
    params: dict[str, object]

    @property
    def shape(self) -> tuple[int, int, int]:
        """`(n_genes, n_cells, k)`."""
        return (self.w.shape[0], self.h.shape[1], self.w.shape[1])

    def __repr__(self) -> str:
        """A summary of the run."""
        return compact_repr(
            "NmfResult",
            source_class=self.source_class,
            n_genes=self.w.shape[0],
            n_cells=self.h.shape[1],
            k=self.w.shape[1],
            final_loss=round(self.final_loss, 4),
            n_iter=self.n_iter,
            converged=self.converged,
            preprocessing=self.params.get("preprocessing"),
        )


@beartype
@dataclass(frozen=True, repr=False)
class StabilisedNmfResult:
    """Randomly initialised HALS NMF restarts.

    Attributes:
        w_all: `(n_genes, k * n_runs)` gene loadings, run blocks side by side,
            columns named in `w_all_columns`.
        w_all_columns: `run_01.comp_01`, ...
        h_per_run: One `(k, n_cells)` usage matrix per run.
        gene_ids: The genes.
        cell_ids: The cells (meta cells).
        cell_indices: 0-based store index (meta cell row) of each cell.
        losses: Final loss per run.
        converged: Convergence per run.
        best_idx: 0-based run with the lowest loss.
        source_class: `"SingleCells"` or `"MetaCells"`.
        params: The parameters of the run.
    """

    w_all: np.ndarray
    w_all_columns: tuple[str, ...]
    h_per_run: tuple[np.ndarray, ...]
    gene_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    cell_indices: np.ndarray
    losses: np.ndarray
    converged: tuple[bool, ...]
    best_idx: int
    source_class: str
    params: dict[str, object]

    def __repr__(self) -> str:
        """A summary of the restarts, the best run 0-based."""
        has_runs = len(self.losses) > 0
        return compact_repr(
            "StabilisedNmfResult",
            source_class=self.source_class,
            n_genes=self.w_all.shape[0],
            n_cells=self.h_per_run[0].shape[1] if self.h_per_run else 0,
            k=self.params.get("k"),
            n_runs=len(self.h_per_run),
            n_converged=sum(self.converged),
            best_run=self.best_idx,
            best_loss=round(float(self.losses[self.best_idx]), 4) if has_runs else None,
            preprocessing=self.params.get("preprocessing"),
        )


@beartype
@dataclass(frozen=True, repr=False)
class ConsensusNmfResult:
    """Consensus HALS NMF.

    Attributes:
        w: `(n_genes, k)` consensus gene loadings.
        h: `(k, n_cells)` cell usages refit against them.
        gene_ids: The genes.
        cell_ids: The cells (meta cells).
        cell_indices: 0-based store index (meta cell row) of each cell.
        stability: Mean silhouette of the consensus clustering.
        rel_error: Relative reconstruction error of the consensus fit.
        rel_run_errors: Relative error per restart.
        clusters: One row per pooled component: `component_id`, `run`,
            `component`, `pooled_idx`, `cluster` (null when dropped),
            `local_density`, `silhouette` (null when dropped) and `kept`.
        cluster_sizes: `cluster` and `n`.
        n_dropped: Components the density filter dropped.
        n_empty_clusters: Clusters left empty.
        source_class: `"SingleCells"` or `"MetaCells"`.
        params: The parameters of the run.
    """

    w: np.ndarray
    h: np.ndarray
    gene_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    cell_indices: np.ndarray
    stability: float
    rel_error: float
    rel_run_errors: np.ndarray
    clusters: pl.DataFrame
    cluster_sizes: pl.DataFrame
    n_dropped: int
    n_empty_clusters: int
    source_class: str
    params: dict[str, object]

    @property
    def shape(self) -> tuple[int, int, int]:
        """`(n_genes, n_cells, k)`."""
        return (self.w.shape[0], self.h.shape[1], self.w.shape[1])

    def __repr__(self) -> str:
        """A summary of the consensus fit."""
        return compact_repr(
            "ConsensusNmfResult",
            source_class=self.source_class,
            n_genes=self.w.shape[0],
            n_cells=self.h.shape[1],
            k=self.w.shape[1],
            n_runs=len(self.rel_run_errors),
            stability=round(self.stability, 4),
            rel_error=round(self.rel_error, 4),
            n_dropped=self.n_dropped,
            n_components=self.clusters.height,
            preprocessing=self.params.get("preprocessing"),
        )


@beartype
@dataclass(frozen=True, repr=False)
class NmfKSweepResult:
    """A consensus NMF k sweep.

    Attributes:
        frame: One row per k: `k`, `stability` (null where the consensus step
            failed), `best_error`, `median_error`, `consensus_failed`,
            `n_dropped`, `n_empty_clusters` and `n_converged`.
        source_class: `"SingleCells"` or `"MetaCells"`.
        params: The parameters of the sweep.
    """

    frame: pl.DataFrame
    source_class: str
    params: dict[str, object]

    def __repr__(self) -> str:
        """A summary of the sweep."""
        f = self.frame
        usable = f.filter(pl.col("stability").is_not_null())
        best = (
            usable.sort("stability", descending=True, maintain_order=True).row(
                0, named=True
            )
            if usable.height
            else None
        )
        return compact_repr(
            "NmfKSweepResult",
            source_class=self.source_class,
            k_range=[f["k"].min(), f["k"].max()],
            n_runs=self.params.get("n_runs"),
            best_k=None if best is None else best["k"],
            best_stability=None if best is None else round(best["stability"], 4),
        )


###########
# Helpers #
###########


@beartype
def _selection(
    sc: ScOrMc, cell_ids: Sequence[str] | None, gene_ids: Sequence[str] | None
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """The cells (kept ones by default) and the genes (the HVGs by default).

    Returns:
        `(cell_indices, gene_indices, cell_ids, gene_ids)`, 0-based store
        indices (meta cell rows).

    Raises:
        ValueError: Without genes and without HVGs.
    """
    if isinstance(sc, MetaCells):
        cells = (
            np.arange(sc.n_cells, dtype=np.uint32)
            if cell_ids is None
            else sc.cell_indices(list(cell_ids))
        )
        hvg = sc.hvg
        all_cells = sc.cell_ids()
    else:
        cells = (
            np.asarray(sc.cells_to_keep, dtype=np.uint32)
            if cell_ids is None
            else sc.cell_indices(list(cell_ids))
        )
        hvg = sc.map.hvg
        all_cells = sc.cell_ids()
    if gene_ids is not None:
        genes = sc.gene_indices(list(gene_ids))
    elif hvg is not None:
        genes = np.asarray(hvg, dtype=np.uint32)
    else:
        raise ValueError("No HVGs found. Run find_hvg() first or pass gene_ids.")
    cells = np.asarray(cells, dtype=np.uint32)
    genes = np.asarray(genes, dtype=np.uint32)
    return (
        cells,
        genes,
        tuple(all_cells[cells.astype(np.int64)].tolist()),
        tuple(sc.gene_ids(genes).tolist()),
    )


@beartype
def _mc_matrix(
    sc: MetaCells, cells: np.ndarray, genes: np.ndarray, use_second_layer: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """The selected meta cell counts as CSR parts: the normalised layer with
    `use_second_layer`, else the raw counts."""
    layer = "norm" if use_second_layer else "raw"
    return csr_args(sc.counts(cells=cells, genes=genes, layer=layer))


@beartype
def _source(sc: ScOrMc) -> str:
    """The class name of `sc`, as stored on the result."""
    return "MetaCells" if isinstance(sc, MetaCells) else "SingleCells"


@beartype
def _check_k(k: int, lo: int) -> None:
    """Raises:
    ValueError: If `k` is below `lo`."""
    if k < lo:
        raise ValueError(f"k must be at least {lo}.")


@beartype
def _warn_target_w(
    params: NmfConsensusParams, n_samples: int, k: int, n_runs: int
) -> None:
    """Warn when `consensus_target = "w"` runs over many cells."""
    if params.consensus_target == "w" and n_samples > CELL_SPACE_CUTOFF:
        warnings.warn(
            f"consensus_target = 'w' clusters in cell space over {n_samples} cells, "
            f"which means a dense {k * n_runs} x {n_samples} matrix and an "
            "exhaustive cosine search across it. Use consensus_target = 'h' unless "
            "you specifically want the cell-side consensus.",
            stacklevel=3,
        )


@beartype
def _base_params(
    hals: NmfHalsParams,
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    seed: int,
) -> dict[str, object]:
    """The `params` stored on a result: the HALS keys plus the call arguments."""
    return {
        **to_rust(hals),
        "k": k,
        "preprocessing": preprocessing,
        "use_second_layer": use_second_layer,
        "seed": seed,
    }


#######
# NMF #
#######


@beartype
def nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> NmfResult:
    """One HALS NMF run.

    Args:
        sc: The experiment, a subset or meta cells.
        k: Number of components.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver. `None` uses the defaults (NNDSVD).
        seed: Seed for a random initialisation.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The `NmfResult`.

    Raises:
        ValueError: On `k` below 1 or without genes.

    Examples:
        >>> res = bv.nmf.nmf(sc, k=5)
        >>> res = bv.nmf.nmf(sc, k=5, preprocessing="sd")
        >>> sc.add_obs(bv.nmf.get_data(res))
    """
    return _nmf(
        sc,
        k,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        seed=seed,
        verbose=verbose,
        ext=_bixverse,
    )


@beartype
def _nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
    ext: ModuleType,
) -> NmfResult:
    """The body of `nmf`, with the extension module `ext` supplied, so
    the GPU twin shares it."""
    _check_k(k, 1)
    hals = nmf_hals_params or NmfHalsParams()
    cells, genes, c_ids, g_ids = _selection(sc, cell_ids, gene_ids)
    if isinstance(sc, MetaCells):
        res = ext.nmf_single_mc(
            *_mc_matrix(sc, cells, genes, use_second_layer),
            k,
            preprocessing,
            use_second_layer,
            to_rust(hals),
            seed,
            verbose,
        )
    else:
        res = ext.nmf_single_sc(
            str(sc.genes_path),
            genes,
            cells,
            k,
            preprocessing,
            use_second_layer,
            to_rust(hals),
            seed,
            verbose,
        )
    return NmfResult(
        w=np.ascontiguousarray(res["h"].T),
        h=np.ascontiguousarray(res["w"].T),
        gene_ids=g_ids,
        cell_ids=c_ids,
        cell_indices=cells,
        final_loss=res["final_loss"],
        n_iter=res["n_iter"],
        converged=res["converged"],
        source_class=_source(sc),
        params=_base_params(hals, k, preprocessing, use_second_layer, seed),
    )


@beartype
def stabilised_nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: Verbose = True,
) -> StabilisedNmfResult:
    """Randomly initialised HALS NMF restarts.

    Args:
        sc: The experiment, a subset or meta cells.
        k: Number of components.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver; `nmf_init` is ignored, the restarts are
            random.
        n_runs: Number of restarts.
        seed: Base seed of the restarts.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The `StabilisedNmfResult`.

    Raises:
        ValueError: On `k` or `n_runs` below 1, or without genes.

    Examples:
        >>> stab = bv.nmf.stabilised_nmf(sc, k=5, n_runs=10)
        >>> best = bv.nmf.get_best_run(stab)
    """
    return _stabilised_nmf(
        sc,
        k,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        n_runs=n_runs,
        seed=seed,
        verbose=verbose,
        ext=_bixverse,
    )


@beartype
def _stabilised_nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: Verbose = True,
    ext: ModuleType,
) -> StabilisedNmfResult:
    """The body of `stabilised_nmf`, with the extension module `ext` supplied, so
    the GPU twin shares it."""
    _check_k(k, 1)
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    hals = nmf_hals_params or NmfHalsParams()
    cells, genes, c_ids, g_ids = _selection(sc, cell_ids, gene_ids)
    if isinstance(sc, MetaCells):
        res = ext.nmf_multi_mc(
            *_mc_matrix(sc, cells, genes, use_second_layer),
            k,
            preprocessing,
            use_second_layer,
            to_rust(hals),
            n_runs,
            seed,
            verbose,
        )
    else:
        res = ext.nmf_multi_sc(
            str(sc.genes_path),
            genes,
            cells,
            k,
            preprocessing,
            use_second_layer,
            to_rust(hals),
            n_runs,
            seed,
            verbose,
        )
    runs = len(res["h_per_run"])
    w_all = np.hstack([h.T for h in res["h_per_run"]])
    h_per_run = tuple(
        np.ascontiguousarray(res["w_all"][:, i * k : (i + 1) * k].T)
        for i in range(runs)
    )
    params = _base_params(hals, k, preprocessing, use_second_layer, seed)
    params["n_runs"] = n_runs
    return StabilisedNmfResult(
        w_all=w_all,
        w_all_columns=tuple(
            f"run_{r + 1:02d}.comp_{c + 1:02d}" for r in range(runs) for c in range(k)
        ),
        h_per_run=h_per_run,
        gene_ids=g_ids,
        cell_ids=c_ids,
        cell_indices=cells,
        losses=res["losses"],
        converged=tuple(res["converged"]),
        best_idx=res["best_idx"],
        source_class=_source(sc),
        params=params,
    )


@beartype
def _clusters_frame(res: dict, k: int, n_runs: int) -> pl.DataFrame:
    """The pooled components with their cluster labels, 0-based."""
    n = k * n_runs
    run = np.repeat(np.arange(n_runs), k)
    comp = np.tile(np.arange(k), n_runs)
    kept = np.zeros(n, dtype=bool)
    kept[res["kept"].astype(np.int64)] = True
    sil = np.full(n, np.nan)
    sil[res["kept"].astype(np.int64)] = res["silhouette"]
    labels = res["labels"]
    return pl.DataFrame(
        {
            "component_id": [
                f"run_{r + 1:02d}.comp_{c + 1:02d}"
                for r, c in zip(run, comp, strict=True)
            ],
            "run": run.astype(np.int64),
            "component": comp.astype(np.int64),
            "pooled_idx": np.arange(n, dtype=np.int64),
            "cluster": pl.Series(labels).replace(-1, None),
            "local_density": res["local_density"],
            "silhouette": pl.Series(sil).fill_nan(None),
            "kept": kept,
        }
    )


@beartype
def consensus_nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    nmf_consensus_params: NmfConsensusParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: Verbose = True,
) -> ConsensusNmfResult:
    """Consensus HALS NMF (cNMF).

    Pools the components of `n_runs` random restarts, drops the ones in
    sparse neighbourhoods, clusters the rest into `k` programmes and takes
    each cluster's median; the usages are refit against it. The k-means of
    the consensus step is seeded from `seed`.

    Args:
        sc: The experiment, a subset or meta cells.
        k: Number of components, at least 2.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver.
        nmf_consensus_params: The consensus step. `None` uses the defaults.
        n_runs: Number of restarts, at least 2.
        seed: Base seed of the restarts and the consensus k-means.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The `ConsensusNmfResult`.

    Raises:
        ValueError: On `k` or `n_runs` below 2, without genes, or when the
            consensus step fails (the message says what to try).

    References:
        Kotliar, et al., eLife, 2019.

    Examples:
        >>> res = bv.nmf.consensus_nmf(sc, k=5, n_runs=10)
        >>> sc.add_obs(bv.nmf.get_data(res))
        >>> stability = bv.nmf.get_stability(res)
    """
    return _consensus_nmf(
        sc,
        k,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        nmf_consensus_params=nmf_consensus_params,
        n_runs=n_runs,
        seed=seed,
        verbose=verbose,
        ext=_bixverse,
    )


@beartype
def _consensus_nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    nmf_consensus_params: NmfConsensusParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: Verbose = True,
    ext: ModuleType,
) -> ConsensusNmfResult:
    """The body of `consensus_nmf`, with the extension module `ext` supplied, so
    the GPU twin shares it."""
    _check_k(k, 2)
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2.")
    hals = nmf_hals_params or NmfHalsParams()
    cons = nmf_consensus_params or NmfConsensusParams()
    cells, genes, c_ids, g_ids = _selection(sc, cell_ids, gene_ids)
    _warn_target_w(cons, cells.size, k, n_runs)
    try:
        if isinstance(sc, MetaCells):
            res = ext.nmf_consensus_mc(
                *_mc_matrix(sc, cells, genes, use_second_layer),
                k,
                preprocessing,
                use_second_layer,
                to_rust(hals),
                to_rust(cons),
                n_runs,
                seed,
                verbose,
            )
        else:
            res = ext.nmf_consensus_sc(
                str(sc.genes_path),
                genes,
                cells,
                k,
                preprocessing,
                use_second_layer,
                to_rust(hals),
                to_rust(cons),
                n_runs,
                seed,
                verbose,
            )
    except _bixverse.BixverseError as e:
        raise ValueError(f"{e}{CONSENSUS_ADVICE}") from e
    params = _base_params(hals, k, preprocessing, use_second_layer, seed)
    params.update({"nmf_consensus_params": cons, "n_runs": n_runs})
    return ConsensusNmfResult(
        w=np.ascontiguousarray(res["h"].T),
        h=np.ascontiguousarray(res["w"].T),
        gene_ids=g_ids,
        cell_ids=c_ids,
        cell_indices=cells,
        stability=res["stability"],
        rel_error=res["rel_error"],
        rel_run_errors=res["rel_run_errors"],
        clusters=_clusters_frame(res, k, len(res["rel_run_errors"])),
        cluster_sizes=pl.DataFrame(
            {
                "cluster": np.arange(k, dtype=np.int64),
                "n": res["cluster_sizes"].astype(np.int64),
            }
        ),
        n_dropped=res["n_dropped"],
        n_empty_clusters=res["n_empty_clusters"],
        source_class=_source(sc),
        params=params,
    )


@beartype
def nmf_k_sweep(
    sc: ScOrMc,
    k_range: Sequence[int] | range,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    nmf_consensus_params: NmfConsensusParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: Verbose = True,
) -> NmfKSweepResult:
    """Consensus NMF over a range of k.

    High stability while the error is still falling is the sweet spot. A k
    whose consensus step fails gets a null stability and a warning.

    Args:
        sc: The experiment, a subset or meta cells.
        k_range: The ranks, unique and at least 2.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver.
        nmf_consensus_params: The consensus step. `None` uses the defaults.
        n_runs: Restarts per k, at least 2.
        seed: Base seed of the restarts and the consensus k-means.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The `NmfKSweepResult`.

    Raises:
        ValueError: On an invalid `k_range` or `n_runs`, or without genes.

    References:
        Kotliar, et al., eLife, 2019.

    Examples:
        >>> sweep = bv.nmf.nmf_k_sweep(sc, k_range=range(3, 8), n_runs=10)
        >>> sweep.frame
    """
    return _nmf_k_sweep(
        sc,
        k_range,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        nmf_consensus_params=nmf_consensus_params,
        n_runs=n_runs,
        seed=seed,
        verbose=verbose,
        ext=_bixverse,
    )


@beartype
def _nmf_k_sweep(
    sc: ScOrMc,
    k_range: Sequence[int] | range,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    nmf_consensus_params: NmfConsensusParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: Verbose = True,
    ext: ModuleType,
) -> NmfKSweepResult:
    """The body of `nmf_k_sweep`, with the extension module `ext` supplied, so
    the GPU twin shares it."""
    ks = [int(k) for k in k_range]
    if not ks or len(set(ks)) != len(ks) or min(ks) < 2:
        raise ValueError("k_range must hold unique integers of at least 2.")
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2.")
    hals = nmf_hals_params or NmfHalsParams()
    cons = nmf_consensus_params or NmfConsensusParams()
    cells, genes, _, _ = _selection(sc, cell_ids, gene_ids)
    _warn_target_w(cons, cells.size, max(ks), n_runs)
    if isinstance(sc, MetaCells):
        res = ext.nmf_k_sweep_mc(
            *_mc_matrix(sc, cells, genes, use_second_layer),
            ks,
            preprocessing,
            use_second_layer,
            to_rust(hals),
            to_rust(cons),
            n_runs,
            seed,
            verbose,
        )
    else:
        res = ext.nmf_k_sweep_sc(
            str(sc.genes_path),
            genes,
            cells,
            ks,
            preprocessing,
            use_second_layer,
            to_rust(hals),
            to_rust(cons),
            n_runs,
            seed,
            verbose,
        )
    failed = np.asarray(res["consensus_failed"], dtype=bool)
    frame = pl.DataFrame(
        {
            "k": res["k"].astype(np.int64),
            "stability": pl.Series(np.where(failed, np.nan, res["stability"])).fill_nan(
                None
            ),
            "best_error": res["best_error"],
            "median_error": res["median_error"],
            "consensus_failed": failed,
            "n_dropped": res["n_dropped"].astype(np.int64),
            "n_empty_clusters": res["n_empty_clusters"].astype(np.int64),
            "n_converged": res["n_converged"].astype(np.int64),
        }
    )
    if failed.any():
        warnings.warn(
            f"The consensus step failed for {int(failed.sum())} of {len(ks)} values "
            "of k (too few components survived the density filter). Their stability "
            "is null. Raise `density_threshold` in NmfConsensusParams or add "
            "restarts.",
            stacklevel=2,
        )
    n_empty = int((frame["n_empty_clusters"] > 0).sum())
    if n_empty:
        warnings.warn(
            f"{n_empty} of {len(ks)} values of k left at least one cluster empty. "
            "Stability is reported as 0 when fewer than two clusters are "
            "populated, so treat those rows as uninformative rather than as "
            "genuinely unstable.",
            stacklevel=2,
        )
    params: dict[str, object] = {
        **to_rust(hals),
        "k_range": ks,
        "preprocessing": preprocessing,
        "use_second_layer": use_second_layer,
        "nmf_consensus_params": cons,
        "n_runs": n_runs,
        "seed": seed,
    }
    return NmfKSweepResult(frame=frame, source_class=_source(sc), params=params)


###########
# Getters #
###########


@beartype
def get_w(x: NmfResult | StabilisedNmfResult | ConsensusNmfResult) -> pl.DataFrame:
    """The gene loadings: `gene_id` and one column per
    component (`comp_01`, ..., or `run_01.comp_01`, ... for restarts).

    Args:
        x: A single, stabilised or consensus NMF result.
    """
    if isinstance(x, StabilisedNmfResult):
        w, cols = x.w_all, list(x.w_all_columns)
    else:
        w, cols = x.w, [f"comp_{j + 1:02d}" for j in range(x.w.shape[1])]
    return pl.DataFrame(
        {"gene_id": list(x.gene_ids), **dict(zip(cols, w.T, strict=True))}
    )


@beartype
def get_h(
    x: NmfResult | StabilisedNmfResult | ConsensusNmfResult,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """The cell usages: `(k, n_cells)`, one per run for
    restarts.

    Args:
        x: A single, stabilised or consensus NMF result.
    """
    if isinstance(x, StabilisedNmfResult):
        return x.h_per_run
    return x.h


@beartype
def get_best_run(x: StabilisedNmfResult) -> NmfResult:
    """The lowest loss run as an `NmfResult`.

    Args:
        x: The stabilised NMF result.
    """
    k = x.params["k"]
    assert isinstance(k, int)
    b = x.best_idx
    return NmfResult(
        w=np.ascontiguousarray(x.w_all[:, b * k : (b + 1) * k]),
        h=x.h_per_run[b],
        gene_ids=x.gene_ids,
        cell_ids=x.cell_ids,
        cell_indices=x.cell_indices,
        final_loss=float(x.losses[b]),
        n_iter=None,
        converged=x.converged[b],
        source_class=x.source_class,
        params=x.params,
    )


@beartype
def get_stability(x: ConsensusNmfResult) -> dict[str, object]:
    """The consensus diagnostics.

    Args:
        x: The consensus NMF result.
    """
    return {
        "stability": x.stability,
        "rel_error": x.rel_error,
        "rel_run_errors": x.rel_run_errors,
        "clusters": x.clusters,
        "cluster_sizes": x.cluster_sizes,
        "n_dropped": x.n_dropped,
        "n_empty_clusters": x.n_empty_clusters,
    }


@beartype
def get_params(
    x: NmfResult | StabilisedNmfResult | ConsensusNmfResult | NmfKSweepResult,
) -> dict[str, object]:
    """The parameters of the run.

    Args:
        x: Any NMF result, including a k sweep.
    """
    return dict(x.params)


@beartype
def get_data(x: NmfResult | ConsensusNmfResult) -> pl.DataFrame:
    """The cell usages as an obs frame for `add_obs`: one
    `comp_XX` column per component and the 0-based `cell_idx`
    (`meta_cell_idx` for meta cells).

    Args:
        x: A single or consensus NMF result.
    """
    idx_col = "meta_cell_idx" if x.source_class == "MetaCells" else "cell_idx"
    return pl.DataFrame(
        {
            **{f"comp_{j + 1:02d}": x.h[j] for j in range(x.h.shape[0])},
            idx_col: x.cell_indices.astype(np.uint32),
        }
    )
