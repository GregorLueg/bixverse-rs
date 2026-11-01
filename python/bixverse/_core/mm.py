"""Multi-modal methods: ADT counts, their PCA, and the WNN graph.

The port of R's `new_adt_counts_clr`, `new_adt_counts_dsb`,
`add_adt_counts_sc`, the `ADTCounts` getters, `calculate_pca_adt_sc` and
`generate_wnn_graph_sc` (`R/classes_single_cell_multimodal.R`,
`R/methods_sc_multimodal.R`). The normalisations, the PCA and WNN run in
Rust; this layer aligns the ADT rows to the cells, writes what R writes and
returns the rest.

    >>> add_adt_counts(mm, adt.counts, adt.obs["cell_id"], proteins)
    >>> calculate_pca_adt(mm, no_pcs=10)
    >>> generate_wnn_graph(mm)
    >>> bv.graph.find_clusters(mm, modality="wnn")

Every index is 0-based.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.graph import _first_dims, _knn_params
from bixverse._core.multimodal import (
    AdtCounts,
    CacheModality,
    DsbInfo,
    SingleCellsMultiModal,
    WnnSlot,
    adt_cols,
    adt_rows,
    require_adt,
)
from bixverse._core.state import CsrMatrix, KnnGraph, ScCache
from bixverse.params import DsbParams, KnnGraphDefaults, WnnParams, to_rust

###########
# Globals #
###########

WNN_DIST_METRIC: str = "kernelised pseudo-distance"

logger = logging.getLogger(__name__)


###########
# Helpers #
###########


@beartype
def _cell_info(
    counts: np.ndarray,
    barcodes: Sequence[str] | np.ndarray | pl.Series,
    feature_ids: Sequence[str] | np.ndarray | pl.Series,
    cell_info: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The raw counts reordered to `cell_info`, as R's
    `raw_counts[names(cell_info), ]`.

    Returns:
        `(raw_counts, cell_ids, cell_idx, feature_ids)`.

    Raises:
        ValueError: On a shape mismatch, duplicated barcodes, or a cell of
            `cell_info` missing from the barcodes.
    """
    barcodes = np.asarray(barcodes, dtype=str)
    features = np.asarray(feature_ids, dtype=str)
    if counts.ndim != 2 or counts.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"counts must be (barcodes, feature_ids) = "
            f"({len(barcodes)}, {len(features)}), got {counts.shape}"
        )
    if len(np.unique(barcodes)) != len(barcodes):
        raise ValueError("barcodes must be unique")
    if not {"cell_id", "cell_idx"} <= set(cell_info.columns):
        raise ValueError("cell_info needs the columns cell_id and cell_idx")
    pos = {b: i for i, b in enumerate(barcodes)}
    cells = cell_info["cell_id"].cast(pl.String).to_list()
    missing = [c for c in cells if c not in pos]
    if missing:
        raise ValueError(
            f"{len(missing)} cells of cell_info are not in the barcodes, e.g. "
            f"{missing[0]!r}"
        )
    rows = np.asarray([pos[c] for c in cells], dtype=np.int64)
    raw = np.ascontiguousarray(counts[rows], dtype=np.float64)
    return (
        raw,
        np.asarray(cells, dtype=str),
        cell_info["cell_idx"].to_numpy().astype(np.uint32),
        features,
    )


#######
# ADT #
#######


@beartype
def new_adt_counts_clr(
    raw_counts: np.ndarray,
    barcodes: Sequence[str] | np.ndarray | pl.Series,
    feature_ids: Sequence[str] | np.ndarray | pl.Series,
    cell_info: pl.DataFrame,
    *,
    seurat_clr: bool = False,
    clean_clr_counts: bool = True,
    percentile: float = 0.01,
) -> AdtCounts:
    """CLR-normalised ADT counts, as R's `new_adt_counts_clr`.

    The CLR runs per cell: `log1p(x) - mean(log1p(x))`, or Seurat's
    non-negative `log1p(x / g)`. With `clean_clr_counts` each protein's
    `percentile` quantile is subtracted afterwards to take out background.

    Args:
        raw_counts: `(n_barcodes, n_features)` raw counts.
        barcodes: Barcode of each row.
        feature_ids: Id of each column.
        cell_info: The cells to keep, in order: `cell_id` and the 0-based
            store index `cell_idx`, as R's `get_cell_info()`.
        seurat_clr: Seurat-style CLR.
        clean_clr_counts: Subtract the per-protein `percentile` quantile.
        percentile: The quantile, in `(0, 1)`.

    Raises:
        ValueError: On a percentile outside `(0, 1)`, or the inputs not lining
            up.
    """
    if not 0 < percentile < 1:
        raise ValueError("percentile must be in (0, 1)")
    raw, cells, idx, features = _cell_info(raw_counts, barcodes, feature_ids, cell_info)
    norm = _bixverse.adt_clr(raw, seurat_clr)
    if clean_clr_counts:
        # R's quantile() default is type 7, numpy's "linear"
        norm = norm - np.quantile(norm, percentile, axis=0)
    return AdtCounts(
        raw_counts=raw,
        norm_counts=np.ascontiguousarray(norm),
        cell_ids=cells,
        cell_idx=idx,
        feature_ids=features,
        type="CLR",
    )


@beartype
def new_adt_counts_dsb(
    raw_counts: np.ndarray,
    barcodes: Sequence[str] | np.ndarray | pl.Series,
    feature_ids: Sequence[str] | np.ndarray | pl.Series,
    cell_info: pl.DataFrame,
    *,
    empty_drops: np.ndarray | None = None,
    isotype_names: Sequence[str] | None = None,
    dsb_params: DsbParams | None = None,
    scale_factor: Literal["standardise", "mean_subtract"] = "standardise",
    seed: int = 42,
    verbose: bool = False,
) -> AdtCounts:
    """DSB-normalised ADT counts, as R's `new_adt_counts_dsb`.

    With `empty_drops` the per-protein background comes from the empty
    droplets; without, from a two-component k-means on the log counts
    ("ModelNegativeADTnorm"). Step II regresses out the technical component
    built from the isotypes and the per-cell background.

    Args:
        raw_counts: `(n_barcodes, n_features)` raw counts.
        barcodes: Barcode of each row.
        feature_ids: Id of each column.
        cell_info: The cells to keep, in order: `cell_id`, `cell_idx`.
        empty_drops: Optional `(n_empty, n_features)` empty droplet counts,
            columns as `feature_ids`.
        isotype_names: Feature ids of the isotype controls. Needed when
            `dsb_params.use_isotype_controls`.
        dsb_params: The parameters. `None` uses the defaults.
        scale_factor: `"standardise"` or `"mean_subtract"`; only used with
            `empty_drops`.
        seed: Seed for the k-means initialisations.
        verbose: Controls verbosity.

    Raises:
        ValueError: On isotypes needed but absent or unknown, or the inputs
            not lining up.

    References:
        Mulè, et al., Nat Commun, 2022.
    """
    params = dsb_params or DsbParams()
    raw, cells, idx, features = _cell_info(raw_counts, barcodes, feature_ids, cell_info)
    if empty_drops is not None and (
        empty_drops.ndim != 2 or empty_drops.shape[1] != len(features)
    ):
        raise ValueError("empty_drops must have one column per feature")
    if params.use_isotype_controls and isotype_names is None:
        raise ValueError(
            "dsb_params.use_isotype_controls = True but isotype_names is None."
        )
    isotypes = tuple(isotype_names or ())
    pos = {f: i for i, f in enumerate(features)}
    unknown = [n for n in isotypes if n not in pos]
    if unknown:
        raise ValueError(f"isotype_names not in feature_ids: {unknown}")

    res = _bixverse.dsb(
        raw,
        None
        if empty_drops is None
        else np.ascontiguousarray(empty_drops, dtype=np.float64),
        [pos[n] for n in isotypes],
        to_rust(params),
        scale_factor,
        seed,
        int(verbose),
    )
    return AdtCounts(
        raw_counts=raw,
        norm_counts=res["norm_counts"],
        cell_ids=cells,
        cell_idx=idx,
        feature_ids=features,
        type="DSB",
        dsb=DsbInfo(
            protein_background_mean=np.asarray(res["protein_background_mean"]),
            protein_background_sd=np.asarray(
                res["protein_background_sd"], dtype=np.float64
            ),
            technical_component=np.asarray(
                res["technical_component"], dtype=np.float64
            ),
            cellwise_background_mean=np.asarray(
                res["cellwise_background_mean"], dtype=np.float64
            ),
            isotype_names=isotypes,
            dsb_params=params,
            scale_factor=scale_factor,
        ),
    )


@beartype
def get_adt_feature_info(adt: AdtCounts | SingleCellsMultiModal) -> pl.DataFrame:
    """Cells expressing each protein, as R's `get_adt_feature_info`.

    Returns:
        `feature_idx` (0-based), `feature_id` and `nnz`. `nnz` is `Float64`
        because R writes it as a double to `var_adt`.
    """
    a = adt if isinstance(adt, AdtCounts) else require_adt(adt)
    return pl.DataFrame(
        {
            "feature_idx": pl.Series(np.arange(len(a.feature_ids)), dtype=pl.UInt32),
            "feature_id": pl.Series(a.feature_ids, dtype=pl.String),
            "nnz": (a.raw_counts > 0).sum(axis=0).astype(np.float64),
        }
    )


@beartype
def get_adt_sample_info(adt: AdtCounts | SingleCellsMultiModal) -> pl.DataFrame:
    """ADT capture per cell, as R's `get_adt_sample_info`.

    Returns:
        `cell_idx` (0-based), `adt_nnz` and `adt_lib_size`, both `Float64` as
        R writes them to obs.
    """
    a = adt if isinstance(adt, AdtCounts) else require_adt(adt)
    return pl.DataFrame(
        {
            "cell_idx": pl.Series(a.cell_idx, dtype=pl.UInt32),
            "adt_nnz": (a.raw_counts != 0).sum(axis=1).astype(np.float64),
            "adt_lib_size": a.raw_counts.sum(axis=1),
        }
    )


@beartype
def get_adt_names(adt: AdtCounts | SingleCellsMultiModal) -> list[str] | None:
    """The ADT feature ids, as R's `get_adt_names`.

    Returns:
        The ids, or `None` (with a warning) on an object without ADT counts.
    """
    if isinstance(adt, SingleCellsMultiModal):
        if adt.adt is None:
            warnings.warn("No ADT counts found. Returning None.", stacklevel=2)
            return None
        adt = adt.adt
    return adt.feature_ids.tolist()


@beartype
def add_adt_counts(
    mm: SingleCellsMultiModal,
    adt_counts: np.ndarray,
    barcodes: Sequence[str] | np.ndarray | pl.Series,
    feature_ids: Sequence[str] | np.ndarray | pl.Series,
    *,
    method: Literal["clr", "dsb"] = "clr",
    seurat_clr: bool = False,
    clean_clr_counts: bool = True,
    percentile: float = 0.01,
    empty_drops: np.ndarray | None = None,
    isotype_names: Sequence[str] | None = None,
    dsb_params: DsbParams | None = None,
    scale_factor: Literal["standardise", "mean_subtract"] = "standardise",
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Add ADT counts for the kept cells, as R's `add_adt_counts_sc`.

    The counts are cut to the kept cells (in their order) and normalised,
    `var_adt` is (re)written to the DuckDB and obs gains `adt_nnz` and
    `adt_lib_size`, as in R. The counts land in `mm.adt`.

    Args:
        mm: The experiment, its RNA already loaded.
        adt_counts: `(n_barcodes, n_features)` raw counts.
        barcodes: Barcode of each row; every kept cell must be one.
        feature_ids: Id of each column.
        method: `"clr"` or `"dsb"`.
        seurat_clr: CLR only, see `new_adt_counts_clr`.
        clean_clr_counts: CLR only.
        percentile: CLR only.
        empty_drops: DSB only, see `new_adt_counts_dsb`.
        isotype_names: DSB only.
        dsb_params: DSB only.
        scale_factor: DSB only.
        seed: DSB only.
        verbose: DSB only.

    Raises:
        ValueError: If a kept cell is not among the barcodes.
    """
    info = pl.DataFrame(
        {"cell_id": mm.cell_ids(filtered=True), "cell_idx": mm.cells_to_keep}
    )
    if method == "clr":
        adt = new_adt_counts_clr(
            adt_counts,
            barcodes,
            feature_ids,
            info,
            seurat_clr=seurat_clr,
            clean_clr_counts=clean_clr_counts,
            percentile=percentile,
        )
    else:
        adt = new_adt_counts_dsb(
            adt_counts,
            barcodes,
            feature_ids,
            info,
            empty_drops=empty_drops,
            isotype_names=isotype_names,
            dsb_params=dsb_params,
            scale_factor=scale_factor,
            seed=seed,
            verbose=verbose,
        )
    with db.connect(mm.db_path) as con:
        db.populate_from_frame(con, "var_adt", get_adt_feature_info(adt))
    mm.add_obs(get_adt_sample_info(adt))
    mm.adt = adt


@beartype
def get_adt_counts(
    mm: SingleCellsMultiModal,
    *,
    layer: Literal["raw", "norm"] = "raw",
    cells: np.ndarray | Sequence[int] | Sequence[str] | None = None,
    features: np.ndarray | Sequence[int] | Sequence[str] | None = None,
    use_cells_to_keep: bool = True,
) -> np.ndarray:
    """The dense ADT counts, as R's `get_sc_counts(modality = "adt")`.

    The ADT matrix is keyed by barcode and frozen at add time; requested
    cells it does not hold warn and drop, so check `len()` against what you
    asked for.

    Args:
        mm: The experiment, with ADT counts.
        layer: `"raw"` or `"norm"`.
        cells: 0-based store indices or ids, rows in that order. `None` for
            the kept cells (every ADT row without `use_cells_to_keep`).
        features: 0-based ADT feature indices or ids. `None` for all.
        use_cells_to_keep: Restrict to the kept cells.

    Returns:
        `(n_cells, n_features)` `float64`.

    Raises:
        ValueError: Without ADT counts, or if no cell is left.
    """
    rows = adt_rows(mm, cells, use_cells_to_keep)
    adt = require_adt(mm)
    mat = adt.raw_counts if layer == "raw" else adt.norm_counts
    mat = mat[rows]
    if features is not None:
        mat = mat[:, adt_cols(adt, features)]
    return mat


#######
# PCA #
#######


@beartype
def calculate_pca_adt(
    mm: SingleCellsMultiModal,
    no_pcs: int,
    *,
    features: Sequence[str] | None = None,
    randomised_svd: bool = False,
    seed: int = 42,
) -> None:
    """PCA over the scaled, normalised ADT counts, as R's
    `calculate_pca_adt_sc`.

    Scores, loadings and singular values go to `mm.adt_cache`, stamped
    `"adt:pca"`.

    Args:
        mm: The experiment, with ADT counts.
        no_pcs: Number of components, at most the number of features used.
        features: ADT feature ids to restrict to, e.g. without the isotypes.
            The loadings rows follow them. `None` for all.
        randomised_svd: Randomised SVD, faster and less precise.
        seed: Seed for the randomised SVD.

    Raises:
        ValueError: If a kept cell has no ADT counts, or `no_pcs` is out of
            range.
    """
    adt = require_adt(mm)
    rows = adt_rows(mm, None, True)
    if len(rows) != mm.n_cells:
        raise ValueError(
            f"{mm.n_cells - len(rows)} kept cells have no ADT counts; the PCA "
            "would not align with the cells. Restrict the cells first."
        )
    norm = adt.norm_counts[rows]
    if features is not None:
        norm = norm[:, adt_cols(adt, features)]
    scores, loadings, sv = _bixverse.adt_pca(
        np.ascontiguousarray(norm), no_pcs, randomised_svd, seed
    )
    # the full SVD binding returns component SDs, the randomised one singular
    # values; store singular values on both paths
    if not randomised_svd:
        sv = np.asarray(sv) * np.sqrt(len(rows) - 1)
    cache = mm.adt_cache
    cache.pca_factors = scores.astype(np.float32)
    cache.pca_loadings = loadings.astype(np.float32)
    cache.pca_singular_vals = np.asarray(sv, dtype=np.float32)
    mm.stamp("pca", modality="adt")


#######
# WNN #
#######


@beartype
def _wnn_embedding(
    mm: SingleCellsMultiModal,
    modality: CacheModality,
    name: str,
    no_dims: int | None,
) -> np.ndarray:
    """An embedding of one modality, as R's `get_embedding` with its soft
    state check.

    Raises:
        ValueError: If it is absent.
    """
    cache = mm.modality_cache(modality)
    embd = cache.pca_factors if name == "pca" else cache.embeddings.get(name)
    if embd is None:
        raise ValueError(f"Embedding '{name}' was not found for modality '{modality}'.")
    state = mm.check_state([name], modality=modality)
    if isinstance(state, str):
        warnings.warn(state, stacklevel=3)
    return _first_dims(embd, no_dims)


@beartype
def generate_wnn_graph(
    mm: SingleCellsMultiModal,
    *,
    modality_1: CacheModality = "rna",
    modality_2: CacheModality = "adt",
    embd_to_use_1: str = "pca",
    embd_to_use_2: str = "pca",
    no_embd_to_use_1: int | None = None,
    no_embd_to_use_2: int | None = None,
    wnn_params: WnnParams | None = None,
    graph_params: KnnGraphDefaults | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Weighted nearest neighbour graph over two modalities, as R's
    `generate_wnn_graph_sc`.

    Per-cell modality weights come from comparing within- and cross-modality
    neighbourhood distances; they fuse the two into one kNN. The
    per-modality kNN graphs are always rebuilt from the embeddings at
    `knn_range` neighbours, the cached ones are not reused. The fused kNN
    (pseudo-distances), its neighbour graph and the weights replace
    `mm.wnn`, and any WNN embeddings with them; use `modality="wnn"` to
    cluster or lay it out.

    Args:
        mm: The experiment.
        modality_1: First modality.
        modality_2: Second modality.
        embd_to_use_1: Embedding of the first, `"pca"` or a name in its cache.
        embd_to_use_2: Embedding of the second.
        no_embd_to_use_1: Leading dimensions of the first. `None` for all.
        no_embd_to_use_2: Leading dimensions of the second. `None` for all.
        wnn_params: The WNN parameters. `None` uses the defaults. Their `k`
            is ignored: the searches return `knn_range` neighbours.
        graph_params: How the fused kNN becomes the neighbour graph. `None`
            uses the defaults.
        seed: Seed for the kNN index builds.
        verbose: Controls verbosity.

    Raises:
        ValueError: If an embedding is absent.

    References:
        Hao, et al., Cell, 2021.
    """
    params = wnn_params or WnnParams()
    gparams = graph_params or KnnGraphDefaults()
    e1 = _wnn_embedding(mm, modality_1, embd_to_use_1, no_embd_to_use_1)
    e2 = _wnn_embedding(mm, modality_2, embd_to_use_2, no_embd_to_use_2)
    if verbose:
        logger.info("Running WNN graph generation.")
    idx, dist, w1, w2 = _bixverse.wnn(e1, e2, _knn_params(params), seed, int(verbose))
    knn = KnnGraph(indices=idx, distances=dist, dist_metric=WNN_DIST_METRIC)

    if verbose:
        logger.info("Generating the %s graph.", gparams.knn_sym)
    data, indices, indptr = _bixverse.knn_graph(
        idx, dist if gparams.knn_sym == "umap" else None, to_rust(gparams), verbose
    )
    n = idx.shape[0]
    mm.wnn = WnnSlot(
        cache=ScCache(knn=knn, connectivities=CsrMatrix(data, indices, indptr, (n, n))),
        weights=pl.DataFrame(
            {
                "cell_id": pl.Series(mm.cell_ids(filtered=True), dtype=pl.String),
                "weight_modality_1": w1,
                "weight_modality_2": w2,
            }
        ),
        modality_1=modality_1,
        modality_2=modality_2,
    )
    mm.stamp(
        "knn",
        modality="wnn",
        parents=(f"{modality_1}:{embd_to_use_1}", f"{modality_2}:{embd_to_use_2}"),
    )
    mm.stamp("connectivities", modality="wnn", parents=("knn",))


@beartype
def get_wnn_weights(mm: SingleCellsMultiModal) -> pl.DataFrame:
    """The per-cell modality weights of the WNN integration.

    Raises:
        ValueError: Before `generate_wnn_graph`.
    """
    if mm.wnn is None:
        raise ValueError("No 'wnn' integration found. Run generate_wnn_graph().")
    return mm.wnn.weights
