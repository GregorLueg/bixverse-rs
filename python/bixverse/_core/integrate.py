"""Batch correction and integration metrics.

The work runs in Rust; this layer reads the batch labels from obs by name
and writes the corrected embeddings to `sc.cache.embeddings` (`"harmony"`,
`"harmony_v2"`, `"mnn"`, `"cca"`, `"rpca"`), BBKNN's kNN and neighbour graph
to `sc.cache.knn` and `sc.cache.connectivities`, each stamped with its
parents. The metrics return records and leave the experiment alone.

    >>> harmony(sc, "batch")
    >>> find_neighbours(sc, embd_to_use="harmony")
    >>> calculate_integration_metrics(sc, "batch", embd_to_use="harmony")

Labels are coded by sorted level (an Enum keeps its own order).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.graph import _embedding, _first_dims
from bixverse._core.multimodal import (
    CacheModality,
    modality_cache,
    require_fresh,
    stamp,
)
from bixverse._core.pp import _factor_codes, _obs_aligned
from bixverse._core.single_cells import SingleCells
from bixverse._core.state import CsrMatrix, KnnGraph
from bixverse._core.subset import ScOrScSubset
from bixverse.params import (
    BbknnParams,
    FastmnnParams,
    HarmonyParams,
    HarmonyV2Params,
    SeuratCcaParams,
    SeuratRpcaParams,
    to_rust,
)

###########
# Globals #
###########

# Automatic Harmony cluster number: min(round(n / 30), 100).
HARMONY_CELLS_PER_CLUSTER: int = 30
HARMONY_MAX_CLUSTERS: int = 100

LisiType = Literal["batch", "cell_type"]

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True)
class KbetScores:
    """kBET result.

    Attributes:
        kbet_score: Share of cells whose test is significant; 0 is perfect
            mixing, 1 none.
        significant_tests: Per cell, whether the p-value is at or below
            `threshold`.
        p_values: Per-cell chi-square p-values.
        chi_square_stats: Per-cell chi-square statistics.
        mean_chi_square: Mean chi-square statistic.
        median_chi_square: Median chi-square statistic.
        threshold: The significance threshold.
        n_batches: Number of batches.
    """

    kbet_score: float
    significant_tests: np.ndarray
    p_values: np.ndarray
    chi_square_stats: np.ndarray
    mean_chi_square: float
    median_chi_square: float
    threshold: float
    n_batches: int


@beartype
@dataclass(frozen=True)
class BatchSilhouetteScores:
    """Batch silhouette width.

    Attributes:
        per_cell: Per-cell silhouettes in `[-1, 1]`; near 0 is mixed, near 1
            separated, negative overcorrected.
        mean_asw: Mean silhouette width.
        median_asw: Median silhouette width.
        n_batches: Number of batches.
        embedding_used: The embedding.
    """

    per_cell: np.ndarray
    mean_asw: float
    median_asw: float
    n_batches: int
    embedding_used: str


@beartype
@dataclass(frozen=True)
class LisiScores:
    """iLISI or cLISI.

    Attributes:
        per_cell: Per-cell LISI in `[1, n_labels]`.
        mean_lisi: Mean LISI.
        median_lisi: Median LISI.
        lisi_norm: The scIB rescaling onto `[0, 1]`, higher is better.
        n_labels: Number of distinct labels.
        type: `"batch"` (iLISI) or `"cell_type"` (cLISI).
    """

    per_cell: np.ndarray
    mean_lisi: float
    median_lisi: float
    lisi_norm: float
    n_labels: int
    type: LisiType


@beartype
@dataclass(frozen=True)
class PcrScores:
    """Principal component regression on batch.

    Attributes:
        pcr: Variance-weighted R-squared of batch on the embedding.
        pcr_pca: The same on the uncorrected PCA.
        pcr_comparison: `(pcr_pca - pcr) / pcr_pca`; `None` on the PCA.
        var_explained: Variance per embedding dimension.
        r_squared: Batch R-squared per embedding dimension.
        embedding_used: The embedding.
    """

    pcr: float
    pcr_pca: float
    pcr_comparison: float | None
    var_explained: np.ndarray
    r_squared: np.ndarray
    embedding_used: str


@beartype
@dataclass(frozen=True)
class CellTypeAswScores:
    """Cell type silhouette width rescaled to `[0, 1]`.

    Attributes:
        per_cell: Per-cell rescaled silhouettes.
        mean_asw: Mean rescaled silhouette width.
        median_asw: Median rescaled silhouette width.
        n_cell_types: Number of cell types.
        embedding_used: The embedding.
    """

    per_cell: np.ndarray
    mean_asw: float
    median_asw: float
    n_cell_types: int
    embedding_used: str


@beartype
@dataclass(frozen=True)
class GraphConnectivityScores:
    """Graph connectivity per cell type.

    Attributes:
        per_cell_type: `cell_type` and `connectivity`, the share of the cell
            type's cells in its largest connected component; in order of
            first appearance.
        mean_connectivity: Mean over cell types.
        median_connectivity: Median over cell types.
    """

    per_cell_type: pl.DataFrame
    mean_connectivity: float
    median_connectivity: float


###########
# Helpers #
###########


@beartype
def _labels(sc: ScOrScSubset, column: str) -> tuple[np.ndarray, list[str]]:
    """0-based `factor()` codes of an obs column over the kept cells.

    Raises:
        ValueError: If the column is missing or holds missing values.
    """
    obs = _obs_aligned(sc, [column])
    if column not in obs.columns:
        raise ValueError(f"Column '{column}' not found in the obs table.")
    return _factor_codes(obs[column], column)


@beartype
def _hard_state(sc: ScOrScSubset, labels: Sequence[str]) -> None:
    """Refuse stale inputs.

    Raises:
        ValueError: If one of the artefacts is stale.
    """
    state = sc.check_state(labels)
    if state is not True:
        raise ValueError(state)


@beartype
def _soft_state(sc: ScOrScSubset, labels: Sequence[str]) -> None:
    """Warn on stale inputs."""
    state = sc.check_state(labels)
    if isinstance(state, str):
        warnings.warn(state, stacklevel=3)


@beartype
def _knn_for_metric(sc: ScOrScSubset, n: int, what: str) -> KnnGraph | None:
    """The cached kNN, warning when absent or stale.

    Raises:
        ValueError: If it does not cover the kept cells.
    """
    knn = sc.cache.knn
    if knn is None:
        warnings.warn(
            f"No kNN matrix found to calculate {what}. Returning None", stacklevel=3
        )
        return None
    _soft_state(sc, ["knn"])
    if knn.indices.shape[0] != n:
        raise ValueError(
            f"The kNN covers {knn.indices.shape[0]} cells but {n} are kept. "
            "Regenerate the neighbours so the two agree."
        )
    return knn


@beartype
def _embedding_for_metric(sc: ScOrScSubset, name: str) -> np.ndarray | None:
    """A cached embedding by name, warning when absent or stale."""
    embd = _embedding(sc, name)
    if embd is None:
        warnings.warn(
            "The desired embedding was not found. Returning None", stacklevel=3
        )
        return None
    _soft_state(sc, [name])
    return embd


###########
# Harmony #
###########


@beartype
def _harmony(
    sc: ScOrScSubset,
    batch_column: str,
    additional_batch_columns: Sequence[str] | None,
    params: object,
    name: str,
    run: Callable[..., np.ndarray],
    modality: CacheModality,
    seed: int,
    verbose: bool,
) -> None:
    """Shared body of Harmony v1, v2 and the GPU twin. `params` is any Harmony
    bundle; `run` the extension call."""
    cache = modality_cache(sc, modality)
    require_fresh(sc, ["pca"], modality)
    pca = cache.pca_factors
    if pca is None:
        warnings.warn(
            "No PCA embeddings found in the object. Returning class as is",
            stacklevel=3,
        )
        return
    columns = [batch_column, *(additional_batch_columns or [])]
    labels = [_labels(sc, c)[0] for c in columns]
    if any(len(lab) != pca.shape[0] for lab in labels):
        raise ValueError("The batch columns do not cover the rows of the PCA.")

    d = to_rust(params)
    if d["k"] is None:
        d["k"] = min(
            round(pca.shape[0] / HARMONY_CELLS_PER_CLUSTER), HARMONY_MAX_CLUSTERS
        )
        if verbose:
            logger.info("Auto-determined number of Harmony clusters: %i", d["k"])
    cache.embeddings[name] = run(
        np.ascontiguousarray(pca, dtype=np.float32), d, labels, seed, verbose
    )
    stamp(sc, name, parents=("pca",), modality=modality)


@beartype
def harmony(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    additional_batch_columns: Sequence[str] | None = None,
    modality: CacheModality = "rna",
    harmony_params: HarmonyParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Harmony on the PCA.

    The corrected embedding goes to the modality's `embeddings["harmony"]`,
    stamped as derived from `"pca"`.

    Args:
        sc: The experiment, with a PCA.
        batch_column: obs column with the primary batch.
        additional_batch_columns: Further obs columns to regress out.
        modality: The PCA to correct and the cache to write: `"rna"`, or
            `"adt"` on a `SingleCellsMultiModal`.
        harmony_params: The parameters. `None` uses the defaults; a `None`
            `k` becomes `min(round(n_cells / 30), 100)`.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the PCA is stale, or a batch column is missing or
            incomplete.

    References:
        Korsunsky, et al., Nat Methods, 2019.

    Examples:
        >>> bv.integrate.harmony(sc, "batch_index")
        >>> bv.integrate.harmony(
        ...     sc,
        ...     "batch_index",
        ...     harmony_params=bv.params.HarmonyParams(theta=(4.0,)),
        ... )
    """
    _harmony(
        sc,
        batch_column,
        additional_batch_columns,
        harmony_params or HarmonyParams(),
        "harmony",
        _bixverse.harmony,
        modality,
        seed,
        verbose,
    )


@beartype
def harmony_v2(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    additional_batch_columns: Sequence[str] | None = None,
    modality: CacheModality = "rna",
    harmony_params: HarmonyV2Params | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Harmony v2 on the PCA.

    The corrected embedding goes to the modality's `embeddings["harmony_v2"]`,
    stamped as derived from `"pca"`.

    Args:
        sc: The experiment, with a PCA.
        batch_column: obs column with the primary batch.
        additional_batch_columns: Further obs columns to regress out.
        modality: The PCA to correct and the cache to write: `"rna"`, or
            `"adt"` on a `SingleCellsMultiModal`.
        harmony_params: The parameters. `None` uses the defaults; a `None`
            `k` becomes `min(round(n_cells / 30), 100)`.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the PCA is stale, or a batch column is missing or
            incomplete.

    References:
        Patikas, et al., bioRxiv, 2026.
    """
    _harmony(
        sc,
        batch_column,
        additional_batch_columns,
        harmony_params or HarmonyV2Params(),
        "harmony_v2",
        _bixverse.harmony_v2,
        modality,
        seed,
        verbose,
    )


#########
# BBKNN #
#########


@beartype
def bbknn(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    no_neighbours_to_keep: int = 5,
    embd_to_use: Literal["pca"] = "pca",
    no_embd_to_use: int | None = None,
    bbknn_params: BbknnParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Batch-balanced kNN.

    The kNN goes to `sc.cache.knn`, stamped as derived from the embedding,
    and the max-symmetrised BBKNN connectivities to
    `sc.cache.connectivities`, stamped as derived from `"knn"`. The kNN keeps
    the `no_neighbours_to_keep` nearest of the per-batch neighbours of each
    cell, by distance.

    Args:
        sc: The experiment, with a PCA.
        batch_column: obs column with the batch.
        no_neighbours_to_keep: Neighbours per cell in the stored kNN; capped
            at `n_batches * neighbours_within_batch`.
        embd_to_use: The embedding; only `"pca"`.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        bbknn_params: The parameters. `None` uses the defaults.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the embedding is stale, or the batch column is
            missing or incomplete.

    References:
        Polański, et al., Bioinformatics, 2020.

    Examples:
        >>> bv.integrate.bbknn(
        ...     sc,
        ...     "batch_index",
        ...     bbknn_params=bv.params.BbknnParams(neighbours_within_batch=5),
        ... )
    """
    _bbknn(
        sc,
        batch_column,
        no_neighbours_to_keep=no_neighbours_to_keep,
        embd_to_use=embd_to_use,
        no_embd_to_use=no_embd_to_use,
        params=to_rust(bbknn_params or BbknnParams()),
        seed=seed,
        verbose=verbose,
        run=_bixverse.bbknn_graph,
    )


@beartype
def _bbknn(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    no_neighbours_to_keep: int,
    embd_to_use: str,
    no_embd_to_use: int | None,
    params: dict[str, object],
    seed: int,
    verbose: bool,
    run: Callable[..., tuple],
) -> None:
    """The body of `bbknn` over `to_rust(params)`, with the extension call
    `run` supplied, so the GPU twin shares it."""
    if sc.cache.knn is not None:
        warnings.warn("Prior kNN matrix found. Will be overwritten.", stacklevel=3)
    _hard_state(sc, [embd_to_use])
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        warnings.warn(
            "The desired embedding was not found. Please check the parameters. "
            "Returning None.",
            stacklevel=3,
        )
        return
    codes, levels = _labels(sc, batch_column)
    if len(levels) < 2:
        warnings.warn(
            "The batch column only has one batch. Returning object as is.",
            stacklevel=3,
        )
        return

    n_within = params["neighbours_within_batch"]
    assert isinstance(n_within, int)
    n_generated = len(levels) * n_within
    if no_neighbours_to_keep > n_generated:
        warnings.warn(
            "The number of desired neighbours cannot be generated with these BBKNN "
            "parameters (too few generated neighbours). Please adopt "
            "neighbours_within_batch accordingly. Returning all neighbours from "
            "BBKNN.",
            stacklevel=3,
        )
    if verbose:
        logger.info("Running BBKNN algorithm.")
    indices, distances, data, cols, indptr = run(
        _first_dims(embd, no_embd_to_use),
        codes,
        params,
        min(no_neighbours_to_keep, n_generated),
        seed,
        verbose,
    )
    sc.cache.knn = KnnGraph(
        indices=indices, distances=distances, dist_metric=str(params["ann_dist"])
    )
    sc.stamp("knn", parents=(embd_to_use,))
    n = indices.shape[0]
    sc.cache.connectivities = CsrMatrix(data, cols, indptr, (n, n))
    sc.stamp("connectivities", parents=("knn",))


###########
# Anchors #
###########


@beartype
def _anchor_correction(
    sc: ScOrScSubset,
    batch_column: str,
    batch_hvg_genes: np.ndarray | Sequence[int] | Sequence[str],
    params: dict[str, object],
    use_precomputed_pca: bool,
    name: str,
    run: Callable[..., np.ndarray],
    seed: int,
    verbose: bool,
    *,
    single_batch_returns: bool,
) -> None:
    """Shared body of fastMNN, CCA and rPCA: they read the store, take an
    optional PCA and write one embedding."""
    codes, levels = _labels(sc, batch_column)
    if single_batch_returns and len(levels) < 2:
        warnings.warn(
            "The batch column only has one batch. Returning object as is.",
            stacklevel=3,
        )
        return
    if use_precomputed_pca:
        _hard_state(sc, ["pca"])
    pca = sc.cache.pca_factors if use_precomputed_pca else None
    if pca is not None and verbose:
        logger.info("Using pre-computed PCA found in the object")

    sc.cache.embeddings[name] = run(
        str(sc.genes_path),
        str(sc.cells_path),
        sc.cells_to_keep,
        sc._resolve_genes(batch_hvg_genes),
        codes,
        None if pca is None else np.ascontiguousarray(pca, dtype=np.float32),
        params,
        seed,
        verbose,
    )
    sc.stamp(name, parents=() if pca is None else ("pca",))


@beartype
def fast_mnn(
    sc: ScOrScSubset,
    batch_column: str,
    batch_hvg_genes: np.ndarray | Sequence[int] | Sequence[str],
    *,
    fastmnn_params: FastmnnParams | None = None,
    use_precomputed_pca: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """fastMNN on a PCA of the batch-aware HVGs.

    The corrected embedding goes to `sc.cache.embeddings["mnn"]`, stamped as
    derived from `"pca"` when the cached PCA was used, else unparented.

    Args:
        sc: The experiment.
        batch_column: obs column with the batch.
        batch_hvg_genes: 0-based gene indices or gene ids, ideally from
            `find_hvg_batch_aware`.
        fastmnn_params: The parameters. `None` uses the defaults.
        use_precomputed_pca: Use the cached PCA instead of computing one on
            `batch_hvg_genes`.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the batch column is missing or incomplete, or the
            cached PCA is stale with `use_precomputed_pca`.
        BixverseError: With fewer than two batches.

    References:
        Haghverdi, et al., Nat Biotechnol, 2018.

    Examples:
        >>> batch_hvg = bv.pp.find_hvg_batch_aware(sc, "batch_index", hvg_no=500)
        >>> bv.integrate.fast_mnn(sc, "batch_index", batch_hvg.hvg_gene_idx)
    """
    _anchor_correction(
        sc,
        batch_column,
        batch_hvg_genes,
        to_rust(fastmnn_params or FastmnnParams()),
        use_precomputed_pca,
        "mnn",
        _bixverse.fast_mnn,
        seed,
        verbose,
        single_batch_returns=False,
    )


@beartype
def seurat_cca(
    sc: SingleCells,
    batch_column: str,
    batch_hvg_genes: np.ndarray | Sequence[int] | Sequence[str],
    *,
    cca_params: SeuratCcaParams | None = None,
    use_precomputed_pca: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Seurat CCA anchor integration.

    The corrected embedding goes to `sc.cache.embeddings["cca"]`, stamped as
    derived from `"pca"` when the cached PCA was used, else unparented. One
    batch warns and leaves the experiment alone.

    Args:
        sc: The experiment.
        batch_column: obs column with the batch.
        batch_hvg_genes: 0-based gene indices or gene ids, ideally from
            `find_hvg_batch_aware`.
        cca_params: The parameters. `None` uses the defaults.
        use_precomputed_pca: Correct the cached PCA instead of computing one.
            The gene-space filter still needs the loadings, so this saves
            less than it does for fastMNN.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the batch column is missing or incomplete, or the
            cached PCA is stale with `use_precomputed_pca`.

    References:
        Stuart, et al., Cell, 2019.

    Examples:
        >>> batch_hvg = bv.pp.find_hvg_batch_aware(sc, "batch_index", hvg_no=500)
        >>> bv.integrate.seurat_cca(sc, "batch_index", batch_hvg.hvg_gene_idx)
    """
    _anchor_correction(
        sc,
        batch_column,
        batch_hvg_genes,
        to_rust(cca_params or SeuratCcaParams()),
        use_precomputed_pca,
        "cca",
        _bixverse.seurat_cca,
        seed,
        verbose,
        single_batch_returns=True,
    )


@beartype
def seurat_rpca(
    sc: SingleCells,
    batch_column: str,
    batch_hvg_genes: np.ndarray | Sequence[int] | Sequence[str],
    *,
    rpca_params: SeuratRpcaParams | None = None,
    use_precomputed_pca: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Seurat reciprocal PCA anchor integration.

    The corrected embedding goes to `sc.cache.embeddings["rpca"]`, stamped
    as derived from `"pca"` when the cached PCA was used, else unparented.
    One batch warns and leaves the experiment alone.

    Args:
        sc: The experiment.
        batch_column: obs column with the batch.
        batch_hvg_genes: 0-based gene indices or gene ids, ideally from
            `find_hvg_batch_aware`.
        rpca_params: The parameters. `None` uses the defaults.
        use_precomputed_pca: Correct the cached PCA instead of computing the
            union PCA; the per-batch PCAs are always computed.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the batch column is missing or incomplete, or the
            cached PCA is stale with `use_precomputed_pca`.

    References:
        Stuart, et al., Cell, 2019.

    Examples:
        >>> batch_hvg = bv.pp.find_hvg_batch_aware(sc, "batch_index", hvg_no=500)
        >>> bv.integrate.seurat_rpca(sc, "batch_index", batch_hvg.hvg_gene_idx)
    """
    _anchor_correction(
        sc,
        batch_column,
        batch_hvg_genes,
        to_rust(rpca_params or SeuratRpcaParams()),
        use_precomputed_pca,
        "rpca",
        _bixverse.seurat_rpca,
        seed,
        verbose,
        single_batch_returns=True,
    )


###########
# Metrics #
###########


@beartype
def calculate_kbet(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    threshold: float = 0.05,
    verbose: bool = False,
) -> KbetScores | None:
    """kBET on the cached kNN.

    Args:
        sc: The experiment, with a kNN.
        batch_column: obs column with the batch.
        threshold: p-values at or below it count as significant.
        verbose: Controls verbosity.

    Returns:
        The scores; `None` (with a warning) for one batch or no kNN.

    Raises:
        ValueError: On a `threshold` outside `[0, 1]`, a missing batch column
            or a kNN over other cells.

    References:
        Büttner, et al., Nat Methods, 2019.

    Examples:
        >>> bv.graph.find_neighbours(sc, embd_to_use="mnn")
        >>> kbet = bv.integrate.calculate_kbet(sc, "batch_index")
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    codes, levels = _labels(sc, batch_column)
    if len(levels) < 2:
        warnings.warn(
            "The batch column only has one batch. Returning None", stacklevel=2
        )
        return None
    knn = _knn_for_metric(sc, len(codes), "kBET")
    if knn is None:
        return None
    res = _bixverse.kbet(knn.indices, codes + 1, verbose)
    significant = res["p_values"] <= threshold
    return KbetScores(
        kbet_score=float(significant.mean()),
        significant_tests=significant,
        p_values=res["p_values"],
        chi_square_stats=res["chi_square_stats"],
        mean_chi_square=res["mean_chi_square"],
        median_chi_square=res["median_chi_square"],
        threshold=threshold,
        n_batches=len(levels),
    )


@beartype
def calculate_lisi(
    sc: ScOrScSubset,
    label_column: str,
    *,
    type: LisiType = "batch",
    weighted: bool = False,
    perplexity: float = 30.0,
    verbose: bool = False,
) -> LisiScores | None:
    """LISI on the cached kNN.

    iLISI on batch labels, cLISI on cell type labels; `lisi_norm` is the
    scIB rescaling for the chosen `type`.

    Args:
        sc: The experiment, with a kNN.
        label_column: obs column with the batch or cell type labels.
        type: `"batch"` (iLISI) or `"cell_type"` (cLISI).
        weighted: Weight the neighbours with a perplexity-calibrated
            Gaussian kernel on the kNN distances; else uniformly.
        perplexity: Perplexity of the weighted version; clamped to k.
        verbose: Controls verbosity.

    Returns:
        The scores; `None` (with a warning) for one label or no kNN.

    Raises:
        ValueError: On a non-positive `perplexity`, a missing label column
            or a kNN over other cells.

    References:
        Korsunsky, et al., Nat Methods, 2019; Luecken, et al., Nat Methods,
        2022.

    Examples:
        >>> ilisi = bv.integrate.calculate_lisi(sc, "batch_index")
        >>> clisi = bv.integrate.calculate_lisi(sc, "cell_grp", type="cell_type")
    """
    if perplexity <= 0:
        raise ValueError("perplexity must be positive")
    codes, levels = _labels(sc, label_column)
    if len(levels) < 2:
        warnings.warn(
            "The label column only has one label. Returning None", stacklevel=2
        )
        return None
    knn = _knn_for_metric(sc, len(codes), "LISI")
    if knn is None:
        return None
    res = _bixverse.lisi(
        knn.indices,
        knn.distances if weighted else None,
        codes + 1,
        perplexity,
        verbose,
    )
    return LisiScores(
        per_cell=res["per_cell"],
        mean_lisi=res["mean_lisi"],
        median_lisi=res["median_lisi"],
        lisi_norm=res["ilisi_norm"] if type == "batch" else res["clisi_norm"],
        n_labels=res["n_labels"],
        type=type,
    )


@beartype
def calculate_batch_asw(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    embd_to_use: str = "pca",
    max_cells: int | None = 5000,
    seed: int = 42,
    verbose: bool = False,
) -> BatchSilhouetteScores | None:
    """Batch average silhouette width on an embedding.

    Args:
        sc: The experiment.
        batch_column: obs column with the batch.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        max_cells: Subsample to this many cells, the pass is O(n^2). `None`
            uses all.
        seed: Seed for the subsampling.
        verbose: Controls verbosity.

    Returns:
        The scores; `None` (with a warning) for one batch or a missing
        embedding.

    Raises:
        ValueError: If the batch column is missing or incomplete.

    Examples:
        >>> asw = bv.integrate.calculate_batch_asw(sc, "batch_index", embd_to_use="mnn")
    """
    codes, levels = _labels(sc, batch_column)
    if len(levels) < 2:
        warnings.warn(
            "The batch column only has one batch. Returning None", stacklevel=2
        )
        return None
    embd = _embedding_for_metric(sc, embd_to_use)
    if embd is None:
        return None
    res = _bixverse.batch_asw(
        np.ascontiguousarray(embd, dtype=np.float32),
        codes + 1,
        max_cells,
        seed,
        verbose,
    )
    return BatchSilhouetteScores(
        per_cell=res["per_cell"],
        mean_asw=res["mean_asw"],
        median_asw=res["median_asw"],
        n_batches=len(levels),
        embedding_used=embd_to_use,
    )


@beartype
def calculate_pcr(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    embd_to_use: str = "pca",
) -> PcrScores | None:
    """Principal component regression on batch.

    On a corrected embedding it also runs on the PCA and reports the scIB
    comparison `(pre - post) / pre`: 1 means the batch variance is gone.

    Args:
        sc: The experiment, with a PCA.
        batch_column: obs column with the batch.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.

    Returns:
        The scores; `None` (with a warning) for one batch, or without the
        PCA or the embedding.

    Raises:
        ValueError: If the batch column is missing or incomplete.

    References:
        Büttner, et al., Nat Methods, 2019; Luecken, et al., Nat Methods,
        2022.

    Examples:
        >>> pcr = bv.integrate.calculate_pcr(sc, "batch_index", embd_to_use="mnn")
    """
    codes, levels = _labels(sc, batch_column)
    if len(levels) < 2:
        warnings.warn(
            "The batch column only has one batch. Returning None", stacklevel=2
        )
        return None
    pca, embd = _embedding(sc, "pca"), _embedding(sc, embd_to_use)
    if pca is None or embd is None:
        warnings.warn(
            "The PCA or the desired embedding was not found. Returning None",
            stacklevel=2,
        )
        return None
    _soft_state(sc, ["pca", embd_to_use])

    # The extension takes float64
    pca_res = _bixverse.pcr(np.ascontiguousarray(pca, dtype=np.float64), codes + 1)
    res = pca_res
    comparison = None
    if embd_to_use != "pca":
        res = _bixverse.pcr(np.ascontiguousarray(embd, dtype=np.float64), codes + 1)
        comparison = (pca_res["pcr"] - res["pcr"]) / pca_res["pcr"]
    return PcrScores(
        pcr=res["pcr"],
        pcr_pca=pca_res["pcr"],
        pcr_comparison=comparison,
        var_explained=res["var_explained"],
        r_squared=res["r_squared"],
        embedding_used=embd_to_use,
    )


@beartype
def calculate_cell_type_asw(
    sc: ScOrScSubset,
    cell_type_column: str,
    *,
    embd_to_use: str = "pca",
    max_cells: int | None = 5000,
    seed: int = 42,
    verbose: bool = False,
) -> CellTypeAswScores | None:
    """Cell type silhouette width on an embedding, rescaled to `[0, 1]`.

    Args:
        sc: The experiment.
        cell_type_column: obs column with the cell types.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        max_cells: Subsample to this many cells. `None` uses all.
        seed: Seed for the subsampling.
        verbose: Controls verbosity.

    Returns:
        The scores; `None` (with a warning) for one cell type or a missing
        embedding.

    Raises:
        ValueError: If the cell type column is missing or incomplete.

    References:
        Luecken, et al., Nat Methods, 2022.

    Examples:
        >>> ct_asw = bv.integrate.calculate_cell_type_asw(
        ...     sc, "cell_grp", embd_to_use="mnn"
        ... )
    """
    codes, levels = _labels(sc, cell_type_column)
    if len(levels) < 2:
        warnings.warn(
            "The cell type column only has one cell type. Returning None",
            stacklevel=2,
        )
        return None
    embd = _embedding_for_metric(sc, embd_to_use)
    if embd is None:
        return None
    res = _bixverse.cell_type_asw(
        np.ascontiguousarray(embd, dtype=np.float32),
        codes + 1,
        max_cells,
        seed,
        verbose,
    )
    return CellTypeAswScores(
        per_cell=res["per_cell"],
        mean_asw=res["mean_asw"],
        median_asw=res["median_asw"],
        n_cell_types=len(levels),
        embedding_used=embd_to_use,
    )


@beartype
def calculate_graph_connectivity(
    sc: ScOrScSubset, cell_type_column: str
) -> GraphConnectivityScores | None:
    """Graph connectivity per cell type on the cached kNN.

    Args:
        sc: The experiment, with a kNN.
        cell_type_column: obs column with the cell types.

    Returns:
        The scores; `None` (with a warning) without a kNN.

    Raises:
        ValueError: If the cell type column is missing or incomplete, or the
            kNN covers other cells.

    References:
        Luecken, et al., Nat Methods, 2022.

    Examples:
        >>> conn = bv.integrate.calculate_graph_connectivity(sc, "cell_grp")
    """
    codes, levels = _labels(sc, cell_type_column)
    knn = _knn_for_metric(sc, len(codes), "connectivity")
    if knn is None:
        return None
    res = _bixverse.graph_connectivity(knn.indices, codes + 1)
    # Rust reports the labels in first-appearance order
    _, first = np.unique(codes, return_index=True)
    order = codes[np.sort(first)]
    return GraphConnectivityScores(
        per_cell_type=pl.DataFrame(
            {
                "cell_type": [levels[c] for c in order],
                "connectivity": pl.Series(res["per_label"], dtype=pl.Float64),
            }
        ),
        mean_connectivity=res["mean"],
        median_connectivity=res["median"],
    )


@beartype
def calculate_integration_metrics(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    cell_type_column: str | None = None,
    embd_to_use: str | None = "pca",
    max_cells: int | None = 5000,
    seed: int = 42,
) -> pl.DataFrame:
    """The batch mixing and biological conservation metrics in one row.

    Every column is on `[0, 1]` (the PCR comparison can go negative), higher
    is better: `kbet_accept` (1 - rejection rate), `batch_asw`
    (`mean(1 - |s|)`), `ilisi`, `pcr_comparison`, `clisi`, `cell_type_asw`
    and `graph_connectivity`. The kNN metrics read the cached kNN, so
    rebuild the neighbours on the corrected embedding first. A metric that
    cannot run (no kNN, no embedding, `embd_to_use=None` as for BBKNN, no
    `cell_type_column`) is null.

    Args:
        sc: The experiment.
        batch_column: obs column with the batch.
        cell_type_column: obs column with the cell types; `None` skips the
            conservation metrics.
        embd_to_use: The embedding for the silhouettes and the PCR; `None`
            skips them.
        max_cells: Subsampling for the silhouette widths.
        seed: Seed for the subsampling.

    Returns:
        One row: `embedding`, `kbet_accept`, `batch_asw`, `ilisi`,
        `pcr_comparison`, `clisi`, `cell_type_asw`, `graph_connectivity`.

    References:
        Luecken, et al., Nat Methods, 2022.

    Examples:
        >>> metrics = bv.integrate.calculate_integration_metrics(
        ...     sc, "batch_index", cell_type_column="cell_grp", embd_to_use="mnn"
        ... )
    """
    kbet = calculate_kbet(sc, batch_column)
    ilisi = calculate_lisi(sc, batch_column, type="batch")
    batch_asw = pcr = clisi = ct_asw = conn = None
    if embd_to_use is not None:
        batch_asw = calculate_batch_asw(
            sc, batch_column, embd_to_use=embd_to_use, max_cells=max_cells, seed=seed
        )
        pcr = calculate_pcr(sc, batch_column, embd_to_use=embd_to_use)
    if cell_type_column is not None:
        clisi = calculate_lisi(sc, cell_type_column, type="cell_type")
        conn = calculate_graph_connectivity(sc, cell_type_column)
        if embd_to_use is not None:
            ct_asw = calculate_cell_type_asw(
                sc,
                cell_type_column,
                embd_to_use=embd_to_use,
                max_cells=max_cells,
                seed=seed,
            )

    row = {
        "kbet_accept": None if kbet is None else 1 - kbet.kbet_score,
        "batch_asw": None
        if batch_asw is None
        else float(np.mean(1 - np.abs(batch_asw.per_cell.astype(np.float64)))),
        "ilisi": None if ilisi is None else ilisi.lisi_norm,
        "pcr_comparison": None if pcr is None else pcr.pcr_comparison,
        "clisi": None if clisi is None else clisi.lisi_norm,
        "cell_type_asw": None if ct_asw is None else ct_asw.mean_asw,
        "graph_connectivity": None if conn is None else conn.mean_connectivity,
    }
    return pl.DataFrame(
        {"embedding": [embd_to_use], **{k: [v] for k, v in row.items()}},
        schema={"embedding": pl.String, **dict.fromkeys(row, pl.Float64)},
    )
