"""Cell type annotation (ScType) and reference mapping (Symphony).

ScType: `prepare_cell_markers`, `calc_sc_type_scores`,
`score_clusters`, `assign_sc_type`, `build_symphony_ref`,
`add_symphony_labels`, `map_symphony_query` and `transfer_labels_symphony`.
The work runs in Rust.

    >>> markers = prepare_cell_markers(sc, marker_df)
    >>> scores = calc_sc_type_scores(sc, markers)
    >>> calls = assign_sc_type(sc, scores, cluster_col="leiden_clustering")
    >>> ref = build_symphony_ref(ref_sc, "batch", ref_sc.map.hvg)
    >>> map_symphony_query(ref, query)
    >>> transfer_labels_symphony(ref, query, "cell_type")

The ScType results are frozen records and nothing is written to obs. The
Symphony query embeddings go to `query.cache.embeddings` under the names
`"symphony"`, `"symphony_pca"` and `"symphony_r"`; transferred labels go to
the query's obs as `predicted_<column>` and `confidence_<column>`.

A `SymphonyReference` is immutable: `add_symphony_labels` hands back a new
one. `save_symphony_ref` / `load_symphony_ref` persist it to a single `.npz`
without pickling.

Every index is 0-based: marker genes, HVGs and cluster ids.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.integrate import (
    HARMONY_CELLS_PER_CLUSTER,
    HARMONY_MAX_CLUSTERS,
    _labels,
    _soft_state,
)
from bixverse._core.pp import _factor_codes, _obs_aligned
from bixverse._core.single_cells import SingleCells
from bixverse.params import (
    HarmonyParams,
    HarmonyV2Params,
    KnnParams,
    PcaParams,
    SctypeCellsParams,
    SymphonyMapParams,
    to_rust,
)
from bixverse.params._checks import check_range

###########
# Globals #
###########

# Bumped whenever the layout of a saved reference changes.
SYMPHONY_FORMAT: int = 1

# The arrays of a reference, in save order; `z_orig` and `r` are absent in a
# slim one.
_REF_ARRAYS: tuple[str, ...] = (
    "gene_means",
    "gene_sds",
    "loadings",
    "z_orig",
    "z_corr",
    "r",
    "centroids",
    "nr",
    "c_cache",
)

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True)
class CellMarkers:
    """The marker genes of one cell type, one entry of the
    `prepare_cell_markers()` dict.

    Attributes:
        positive: 0-based indices of the positive markers.
        negative: 0-based indices of the negative markers.
    """

    positive: np.ndarray
    negative: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.uint32)
    )


@beartype
@dataclass(frozen=True, repr=False)
class ScTypeResults:
    """Per-cell ScType scores.

    Attributes:
        cell_types: The cell types, in score column order.
        scores: `(n_cells, n_cell_types)` `float64` scores over the kept
            cells, in `cells_to_keep` order.
    """

    cell_types: list[str]
    scores: np.ndarray

    @property
    def n_cells(self) -> int:
        """Number of scored cells."""
        return self.scores.shape[0]

    @property
    def n_cell_types(self) -> int:
        """Number of cell types."""
        return self.scores.shape[1]

    def __repr__(self) -> str:
        """Summary of the scored cells and cell types."""
        return "\n".join(
            [
                f"ScTypeResults: {self.n_cells} cells, {self.n_cell_types} cell types",
                f"  Cell types: {', '.join(self.cell_types)}",
            ]
        )


@beartype
@dataclass(frozen=True, repr=False)
class ScTypeCellResults:
    """Per-cell ScType calls.

    Attributes:
        cell_types: The scored cell types.
        assignments: Per-cell call, null for Unknown.
        scores: Winning score per cell, after calibration and smoothing.
        margins: Best minus second best score per cell; a small margin flags
            an ambiguous call.
        agreement: Share of graph neighbours sharing the call; `None`
            without smoothing.
        hybrid: The hybrid call, `None` without a cluster column.
        composition: Per cluster `cluster_id`, `n_cells`, `n_unknown`,
            `dominant`, `purity`, `second`, `second_fraction`, `entropy` and
            `cluster_mixed`, or `None`.
        counts: `cluster_id` and one count column per cell type, or `None`.
    """

    cell_types: list[str]
    assignments: pl.Series
    scores: np.ndarray
    margins: np.ndarray
    agreement: np.ndarray | None
    hybrid: pl.Series | None
    composition: pl.DataFrame | None
    counts: pl.DataFrame | None

    def __repr__(self) -> str:
        """Summary of the calls and the Unknown cells."""
        lines = [
            f"ScTypeCellResults: {self.assignments.len()} cells, "
            f"{len(self.cell_types)} cell types",
            f"  Unknown cells:  {self.assignments.null_count()}",
            f"  Smoothing used: {'TRUE' if self.agreement is not None else 'FALSE'}",
        ]
        if self.composition is None:
            lines.append("  Cluster composition: not calculated")
        else:
            lines.append(
                f"  Mixed clusters: {int(self.composition['cluster_mixed'].sum())} out "
                f"of {self.composition.height}"
            )
        return "\n".join(lines)


@beartype
@dataclass(frozen=True, repr=False)
class SymphonyReference:
    """A Symphony reference.

    Holds the PCA loadings, the per-HVG scaling statistics, the soft cluster
    centroids and the cached compression terms (`nr`, `c_cache`) that map a
    query without the reference cells, plus the Harmony-corrected embedding
    for label transfer. Everything numeric is `float32`, as computed.

    Attributes:
        hvg_gene_names: HVG gene ids in loading row order.
        gene_means: Per-HVG mean of the normalised reference data.
        gene_sds: Per-HVG standard deviation of the normalised reference data.
        loadings: PCA gene loadings, `(n_hvgs, d)`.
        z_corr: Harmony-corrected embedding, `(N, d)`.
        centroids: Cosine-normalised centroids, `(K, d)`.
        nr: Cluster sizes, the row sums of `r`, `(K,)`.
        c_cache: The compression term `r @ z_corr`, `(K, d)`.
        no_pcs: Number of principal components.
        harmony_backend: `"v1"` or `"v2"`.
        batch_vars: The batch columns of the build.
        slim: Whether `z_orig` and `r` were dropped.
        z_orig: Pre-Harmony PCA scores, `(N, d)`; `None` when slim.
        r: Soft cluster assignments, `(K, N)`; `None` when slim.
        labels: Reference cell labels in `z_corr` row order, or `None`.

    References:
        Kang, et al., Nat Commun, 2021.
    """

    hvg_gene_names: list[str]
    gene_means: np.ndarray
    gene_sds: np.ndarray
    loadings: np.ndarray
    z_corr: np.ndarray
    centroids: np.ndarray
    nr: np.ndarray
    c_cache: np.ndarray
    no_pcs: int
    harmony_backend: str
    batch_vars: list[str]
    slim: bool = False
    z_orig: np.ndarray | None = None
    r: np.ndarray | None = None
    labels: pl.DataFrame | None = None

    def __repr__(self) -> str:
        """Summary of the reference."""
        labels = "none" if self.labels is None else ", ".join(self.labels.columns)
        return "\n".join(
            [
                "Symphony reference",
                f"  Harmony backend: {self.harmony_backend}",
                f"  No HVGs: {self.loadings.shape[0]}",
                f"  No PCs: {self.loadings.shape[1]}",
                f"  No clusters: {self.centroids.shape[0]}",
                f"  Batch variables: {', '.join(self.batch_vars)}",
                f"  Slim: {'TRUE' if self.slim else 'FALSE'}",
                f"  Labels: {labels}",
            ]
        )


###########
# Helpers #
###########


@beartype
def _sc_type_labels(idx: np.ndarray, cell_types: Sequence[str]) -> pl.Series:
    """0-based codes to names, `-1` to null."""
    return pl.Series(
        [cell_types[i] if i >= 0 else None for i in idx.tolist()], dtype=pl.String
    )


@beartype
def _label_frame(sc: SingleCells, columns: Sequence[str]) -> pl.DataFrame:
    """The obs columns in `cells_to_keep` order.

    Raises:
        ValueError: If a column is missing.
    """
    obs = _obs_aligned(sc, columns)
    missing = [c for c in columns if c not in obs.columns]
    if missing:
        raise ValueError(f"Column(s) not found in the obs table: {', '.join(missing)}")
    return obs.select(columns)


@beartype
def _label_codes(values: pl.Series, name: str) -> tuple[np.ndarray, list[str | None]]:
    """0-based factor codes, with null as a level of its own.

    The levels follow `_factor_codes`; a missing value is its own level,
    last.
    """
    present = values.is_not_null().to_numpy()
    codes_present, levels = _factor_codes(values.drop_nulls(), name)
    codes = np.full(values.len(), len(levels), dtype=np.uint32)
    codes[present] = codes_present
    if present.all():
        return codes, list(levels)
    return codes, [*levels, None]


###########
# ScType #
###########


@beartype
def prepare_cell_markers(
    sc: SingleCells, marker_df: pl.DataFrame
) -> dict[str, CellMarkers]:
    """Marker genes per cell type.

    Genes the experiment does not hold are dropped. Cell types keep their
    order of first appearance.

    Args:
        sc: The experiment.
        marker_df: Columns `cell_type` and `gene_id`; the gene ids have to be
            of the experiment's id type.

    Returns:
        Cell type to its `CellMarkers`, positive markers only. A cell type
        none of whose genes matched is left out.

    Raises:
        ValueError: If a column is missing.

    Examples:
        >>> markers = bv.annotate.prepare_cell_markers(sc, marker_df)
    """
    missing = {"cell_type", "gene_id"} - set(marker_df.columns)
    if missing:
        raise ValueError(f"marker_df lacks the column(s): {', '.join(sorted(missing))}")
    lookup = {g: i for i, g in enumerate(sc.gene_ids().tolist())}
    out: dict[str, list[int]] = {}
    for cell_type, gene in marker_df.select("cell_type", "gene_id").iter_rows():
        idx = lookup.get(gene)
        if idx is not None:
            out.setdefault(cell_type, []).append(idx)
    return {
        ct: CellMarkers(positive=np.asarray(idx, dtype=np.uint32))
        for ct, idx in out.items()
    }


@beartype
def calc_sc_type_scores(
    sc: SingleCells,
    cell_marker_list: Mapping[str, CellMarkers],
    *,
    sensitivity: bool = True,
    weight_floor: float | None = None,
    verbose: bool = False,
) -> ScTypeResults:
    """ScType scores per cell.

    Args:
        sc: The experiment.
        cell_marker_list: Cell type to markers, see `prepare_cell_markers`.
        sensitivity: Downweight markers shared between cell types.
        weight_floor: In `[0, 1]`, the floor of those weights; `None` is 0.1.
        verbose: Controls verbosity.

    Returns:
        The scores over the kept cells.

    Raises:
        ValueError: If there are no cell types, or a marker index is out of
            range.

    References:
        Ianevski, et al., Nat Commun, 2022.

    Examples:
        >>> scores = bv.annotate.calc_sc_type_scores(sc, markers, sensitivity=False)
    """
    check_range("weight_floor", weight_floor, 0, 1, False, False)
    if not cell_marker_list:
        raise ValueError("cell_marker_list holds no cell types.")
    n_genes = sc.n_genes
    for ct, m in cell_marker_list.items():
        for idx in (m.positive, m.negative):
            if idx.size and (idx.min() < 0 or idx.max() >= n_genes):
                raise ValueError(f"Cell type '{ct}' has a marker index out of range.")
    markers = [
        {
            "cell_type": ct,
            "positive_indices": [int(i) for i in m.positive],
            "negative_indices": [int(i) for i in m.negative],
        }
        for ct, m in cell_marker_list.items()
    ]
    res = _bixverse.sc_type(
        str(sc.genes_path),
        sc.cells_to_keep,
        markers,
        sensitivity,
        weight_floor,
        verbose,
    )
    return ScTypeResults(cell_types=list(res["cell_types"]), scores=res["scores"])


@beartype
def get_scores(x: ScTypeResults | ScTypeCellResults) -> pl.DataFrame | np.ndarray:
    """The scores of ScType results.

    Args:
        x: ScType results.

    Returns:
        For `ScTypeResults` one column per cell type over the cells; for
        `ScTypeCellResults` the winning score per cell.
    """
    if isinstance(x, ScTypeCellResults):
        return x.scores
    return pl.DataFrame({ct: x.scores[:, j] for j, ct in enumerate(x.cell_types)})


@beartype
def score_clusters(
    x: ScTypeResults, cluster_labels: np.ndarray | Sequence[int]
) -> pl.DataFrame:
    """One ScType call per cluster.

    A cluster whose best summed score falls below a quarter of its size is
    `"Unknown"`.

    Args:
        x: The scores.
        cluster_labels: 0-based cluster per scored cell.

    Returns:
        `cluster_id`, `cell_type`, `scores` and `n_cells` per non-empty
        cluster.

    Raises:
        ValueError: If the labels do not cover the scored cells or are
            negative.

    Examples:
        >>> labels = sc.obs(filtered=True)["leiden_clustering"].to_numpy()
        >>> per_cluster = bv.annotate.score_clusters(scores, labels)
    """
    labels = np.asarray(cluster_labels)
    if labels.shape != (x.n_cells,):
        raise ValueError(
            f"cluster_labels has {labels.size} entries for {x.n_cells} scored cells."
        )
    if labels.dtype.kind not in "iu" or (labels.size and labels.min() < 0):
        raise ValueError("cluster_labels must be non-negative integers.")
    res = _bixverse.sc_type_cluster_assignment(
        x.cell_types,
        np.ascontiguousarray(x.scores, dtype=np.float64),
        np.ascontiguousarray(labels, dtype=np.uint32),
    )
    return pl.DataFrame(
        {
            "cluster_id": res["cluster_id"].astype(np.int32),
            "cell_type": res["cell_type"],
            "scores": res["scores"],
            "n_cells": res["n_cells"].astype(np.int32),
        }
    )


@beartype
def assign_sc_type(
    sc: SingleCells,
    sc_type_res: ScTypeResults,
    *,
    cluster_col: str | None = None,
    sctype_cell_params: SctypeCellsParams | None = None,
    verbose: bool = False,
) -> ScTypeCellResults:
    """Per-cell ScType calls.

    The score matrix is smoothed over the neighbour graph by label spreading
    (Zhou et al.), then each cell takes its argmax; cells whose best score
    falls below `score_floor` are Unknown. With `cluster_col` the per-cluster
    composition and the hybrid call come on top: clusters at or above
    `purity_threshold` keep the cluster-level call, mixed ones the per-cell
    calls.

    Args:
        sc: The experiment, ideally with a neighbour graph over the scored
            cells.
        sc_type_res: The scores, see `calc_sc_type_scores`.
        cluster_col: Optional obs column with 0-based integer clusters.
        sctype_cell_params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The per-cell calls. Without a neighbour graph it warns and scores
        without smoothing.

    Raises:
        ValueError: If the graph or the clusters do not cover the scored
            cells.

    References:
        Ianevski, et al., Nat Commun, 2022; Zhou, et al., NIPS, 2004.

    Examples:
        >>> col = "leiden_clustering"
        >>> calls = bv.annotate.assign_sc_type(sc, scores, cluster_col=col)
        >>> composition = calls.composition
    """
    params = sctype_cell_params or SctypeCellsParams()
    graph = sc.cache.connectivities
    if graph is None:
        warnings.warn(
            "No neighbour graph found. Did you run find_neighbours()? Scoring "
            "without smoothing.",
            stacklevel=2,
        )
    else:
        _soft_state(sc, ["connectivities"])
        if graph.shape[0] != sc_type_res.n_cells:
            raise ValueError(
                f"The neighbour graph has {graph.shape[0]} nodes, but the ScType "
                f"results {sc_type_res.n_cells} cells."
            )

    clusters = None
    if cluster_col is not None:
        col = _label_frame(sc, [cluster_col])[cluster_col]
        if (
            not col.dtype.is_integer()
            or col.null_count()
            or col.len() != sc_type_res.n_cells
            or (col.len() and col.min() < 0)  # ty: ignore[unsupported-operator]
        ):
            raise ValueError(
                f"Column '{cluster_col}' must hold a non-negative integer cluster "
                "for every scored cell."
            )
        clusters = col.to_numpy().astype(np.uint32)

    if verbose:
        logger.info("Assigning cell types per cell (smoothing: %s).", graph is not None)
    res = _bixverse.sc_type_assign_cells(
        sc_type_res.cell_types,
        np.ascontiguousarray(sc_type_res.scores, dtype=np.float64),
        to_rust(params),
        None if graph is None else (graph.data, graph.indices, graph.indptr),
        clusters,
    )

    ct = sc_type_res.cell_types
    composition = counts = hybrid = None
    comp = res["composition"]
    if comp is not None:
        hybrid = _sc_type_labels(res["hybrid_assignments"], ct)
        cluster_id = comp["cluster_id"].astype(np.int32)
        composition = pl.DataFrame(
            {
                "cluster_id": cluster_id,
                "n_cells": comp["n_cells"].astype(np.int32),
                "n_unknown": comp["n_unknown"].astype(np.int32),
                "dominant": _sc_type_labels(comp["dominant"], ct),
                "purity": comp["purity"],
                "second": _sc_type_labels(comp["second"], ct),
                "second_fraction": comp["second_fraction"],
                "entropy": comp["entropy"],
                "cluster_mixed": comp["cluster_mixed"],
            }
        )
        counts = pl.DataFrame(
            {
                "cluster_id": cluster_id,
                **{c: comp["counts"][:, j].astype(np.int32) for j, c in enumerate(ct)},
            }
        )
    return ScTypeCellResults(
        cell_types=list(ct),
        assignments=_sc_type_labels(res["assignments"], ct),
        scores=res["scores"],
        margins=res["margins"],
        agreement=res["agreement"],
        hybrid=hybrid,
        composition=composition,
        counts=counts,
    )


############
# Symphony #
############


@beartype
def build_symphony_ref(
    sc: SingleCells,
    batch_column: str,
    hvg: np.ndarray | Sequence[int],
    *,
    additional_batch_columns: Sequence[str] | None = None,
    harmony_params: HarmonyParams | HarmonyV2Params | None = None,
    pca_params: PcaParams | None = None,
    no_pcs: int = 30,
    slim: bool = False,
    label_columns: Sequence[str] | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> SymphonyReference:
    """Build a Symphony reference.

    Runs sparse PCA over the HVGs and Harmony over the batches, and
    compresses the result into the terms a query is mapped with. The Harmony
    version follows the type of `harmony_params`. Optionally snapshots obs
    columns (cell type labels, say) for label transfer.

    Args:
        sc: The reference experiment.
        batch_column: obs column with the primary batch.
        hvg: 0-based HVG indices into the experiment's genes, e.g.
            `sc.map.hvg`.
        additional_batch_columns: Further obs columns to correct for.
        harmony_params: `HarmonyParams` (v1) or `HarmonyV2Params` (v2).
            `None` is v1 with the defaults; a `None` `k` becomes
            `min(round(n_cells / 30), 100)`.
        pca_params: The PCA parameters. `None` uses the defaults.
        no_pcs: Number of principal components.
        slim: Drop `z_orig` and `r`. `z_corr` is always kept.
        label_columns: obs columns to snapshot, in `cells_to_keep` order.
        seed: Seed.
        verbose: Controls verbosity.

    Returns:
        The reference.

    Raises:
        ValueError: If an HVG index is out of range, or a batch or label
            column is missing or incomplete.

    Examples:
        >>> bv.pp.find_hvg(sc, hvg_no=2000)
        >>> ref = bv.annotate.build_symphony_ref(
        ...     sc, "sample_id", sc.map.hvg, no_pcs=20, label_columns=["cell_grp"]
        ... )

    References:
        Kang, et al., Nat Commun, 2021.
    """
    check_range("no_pcs", no_pcs, 1, None, False, True)
    harmony_params = harmony_params or HarmonyParams()
    version = "v2" if isinstance(harmony_params, HarmonyV2Params) else "v1"
    hvg_idx = np.asarray(hvg)
    if (
        hvg_idx.ndim != 1
        or hvg_idx.size == 0
        or hvg_idx.dtype.kind not in "iu"
        or hvg_idx.min() < 0
        or hvg_idx.max() >= sc.n_genes
    ):
        raise ValueError("hvg contains out-of-range indices.")
    hvg_idx = np.ascontiguousarray(hvg_idx, dtype=np.uint32)

    columns = [batch_column, *(additional_batch_columns or [])]
    batch = [_labels(sc, c)[0] for c in columns]
    labels = None if label_columns is None else _label_frame(sc, label_columns)

    d = to_rust(harmony_params)
    if harmony_params.k is None:
        d["k"] = min(
            round(sc.n_cells / HARMONY_CELLS_PER_CLUSTER), HARMONY_MAX_CLUSTERS
        )
        if verbose:
            logger.info("Auto-determined number of Harmony clusters: %i", d["k"])

    res = _bixverse.build_symphony_ref(
        str(sc.genes_path),
        str(sc.cells_path),
        sc.cells_to_keep,
        hvg_idx,
        batch,
        to_rust(pca_params or PcaParams()),
        no_pcs,
        d,
        version,
        seed,
        verbose,
    )
    return SymphonyReference(
        hvg_gene_names=[str(g) for g in sc.gene_ids(hvg_idx)],
        gene_means=res["gene_means"],
        gene_sds=res["gene_sds"],
        loadings=res["loadings"],
        z_corr=res["z_corr"],
        centroids=res["centroids"],
        nr=res["nr"],
        c_cache=res["c"],
        no_pcs=no_pcs,
        harmony_backend=version,
        batch_vars=columns,
        slim=slim,
        z_orig=None if slim else res["z_orig"],
        r=None if slim else res["r"],
        labels=labels,
    )


@beartype
def add_symphony_labels(
    reference: SymphonyReference,
    sc: SingleCells,
    columns: Sequence[str],
    *,
    overwrite: bool = False,
) -> SymphonyReference:
    """Attach obs columns to a reference.

    `sc` has to keep the cells the reference was built on, in the same
    order; only the count is checked.

    Args:
        reference: The reference.
        sc: Typically the experiment the reference was built from.
        columns: obs columns to store.
        overwrite: Replace stored columns of the same name instead of
            raising.

    Returns:
        A new reference with the labels; the input is left alone.

    Raises:
        ValueError: If the cell count disagrees with `z_corr`, a column is
            missing, or a name collides without `overwrite`.

    Examples:
        >>> ref = bv.annotate.add_symphony_labels(ref, sc, ["cell_grp"], overwrite=True)
    """
    new = _label_frame(sc, columns)
    n_ref = reference.z_corr.shape[0]
    if new.height != n_ref:
        raise ValueError(
            f"z_corr has {n_ref} rows but sc provides {new.height} labels; "
            "cells_to_keep on sc may have changed since the reference was built."
        )
    existing = reference.labels
    if existing is not None:
        overlap = [c for c in columns if c in existing.columns]
        if overlap and not overwrite:
            raise ValueError(
                f"Label column(s) already present: {', '.join(overlap)}. Use "
                "overwrite=True to replace."
            )
        new = pl.concat([existing.drop(overlap), new], how="horizontal")
    return dataclasses.replace(reference, labels=new)


@beartype
def get_symphony_loadings(reference: SymphonyReference) -> np.ndarray:
    """The PCA gene loadings, `(n_hvgs, d)`.

    Args:
        reference: The Symphony reference.

    Returns:
        The loadings.
    """
    return reference.loadings


@beartype
def get_symphony_z_corr(reference: SymphonyReference) -> np.ndarray:
    """The corrected reference embedding, `(N, d)`.

    Args:
        reference: The Symphony reference.

    Returns:
        The corrected embedding.
    """
    return reference.z_corr


@beartype
def get_symphony_hvg_names(reference: SymphonyReference) -> list[str]:
    """The HVG gene ids in loading order.

    Args:
        reference: The Symphony reference.

    Returns:
        The HVG gene ids.
    """
    return reference.hvg_gene_names


@beartype
def get_symphony_labels(reference: SymphonyReference) -> pl.DataFrame | None:
    """The stored labels in `z_corr` row order.

    Args:
        reference: The Symphony reference.

    Returns:
        The labels, or `None` if the reference has none.
    """
    return reference.labels


@beartype
def map_symphony_query(
    reference: SymphonyReference,
    query: SingleCells,
    *,
    batch_column: str | None = None,
    additional_batch_columns: Sequence[str] | None = None,
    params: SymphonyMapParams | None = None,
    verbose: bool = False,
) -> None:
    """Map a query onto a reference.

    Projects the query through the reference loadings and applies the cached
    mixture-of-experts correction. Genes match by id against the reference
    HVGs; HVGs the query lacks are zero columns. Writes
    `query.cache.embeddings["symphony"]` (corrected), `["symphony_pca"]`
    (projected) and `["symphony_r"]` (soft assignments, `(N_q, K)`).

    Args:
        reference: The reference.
        query: The query experiment.
        batch_column: obs column with the query batch. `None` skips the
            correction, so `"symphony"` equals `"symphony_pca"`.
        additional_batch_columns: Further obs columns to correct for.
        params: The mapping parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Raises:
        ValueError: If a batch column is missing or incomplete.

    References:
        Kang, et al., Nat Commun, 2021.

    Examples:
        >>> bv.annotate.map_symphony_query(ref, query, batch_column="sample_id")
        >>> embedding = query.cache.embeddings["symphony"]
    """
    lookup: dict[str, int] = {}
    for i, g in enumerate(query.gene_ids().tolist()):
        lookup.setdefault(g, i)
    gene_map = np.asarray(
        [lookup.get(g, -1) for g in reference.hvg_gene_names], dtype=np.int64
    )
    n_missing = int((gene_map < 0).sum())
    if n_missing and verbose:
        logger.info(
            "%i / %i reference HVGs not found in query; treated as zero columns.",
            n_missing,
            gene_map.size,
        )
    batch = (
        []
        if batch_column is None
        else [
            _labels(query, c)[0]
            for c in [batch_column, *(additional_batch_columns or [])]
        ]
    )
    res = _bixverse.symphony_map_query(
        str(query.genes_path),
        query.cells_to_keep,
        reference.gene_means,
        reference.gene_sds,
        reference.loadings,
        reference.centroids,
        reference.nr,
        reference.c_cache,
        gene_map,
        batch,
        to_rust(params or SymphonyMapParams()),
        verbose,
    )
    query.cache.embeddings["symphony"] = res["z_corr"]
    query.cache.embeddings["symphony_pca"] = res["z_pca"]
    query.cache.embeddings["symphony_r"] = np.ascontiguousarray(res["r"].T)
    for name in ("symphony", "symphony_pca", "symphony_r"):
        query.stamp(name)


@beartype
def transfer_labels_symphony(
    reference: SymphonyReference,
    query: SingleCells,
    label_column: str,
    *,
    knn_params: KnnParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> pl.DataFrame:
    """kNN majority vote of reference labels onto a mapped query.

    The reference `z_corr` is indexed, the query's `"symphony"` embedding
    searched. A missing reference label is its own class, so a query cell
    can come back null. Ties go to the label sorting last (the crate's
    `max_by_key`). The two columns are also written to the
    query's obs.

    Args:
        reference: The reference, with stored labels.
        query: The query, mapped with `map_symphony_query`.
        label_column: A stored label column.
        knn_params: The kNN parameters. `None` uses the defaults; distances
            on `z_corr` are typically Euclidean.
        seed: Seed for the index build.
        verbose: Controls verbosity.

    Returns:
        `predicted_<label_column>` and `confidence_<label_column>` (the
        winning vote share) per kept query cell, in `cells_to_keep` order.

    Raises:
        ValueError: If the reference has no such labels, or the query has no
            current `"symphony"` embedding.

    Examples:
        >>> pred = bv.annotate.transfer_labels_symphony(ref, query, "cell_grp")
    """
    labels = reference.labels
    if labels is None:
        raise ValueError(
            "SymphonyReference has no stored labels. Use `label_columns` in "
            "build_symphony_ref() or attach them via add_symphony_labels()."
        )
    if label_column not in labels.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in reference. Available: "
            f"{', '.join(labels.columns)}"
        )
    embd = query.cache.embeddings.get("symphony")
    if embd is None:
        raise ValueError(
            "Query has no 'symphony' embedding. Run map_symphony_query first."
        )
    _soft_state(query, ["symphony"])
    if embd.shape[0] != query.n_cells:
        raise ValueError(
            f"The 'symphony' embedding covers {embd.shape[0]} cells but "
            f"{query.n_cells} are kept. Map the query again."
        )

    codes, levels = _label_codes(labels[label_column], label_column)
    pred, conf = _bixverse.transfer_labels_symphony(
        np.ascontiguousarray(reference.z_corr, dtype=np.float32),
        np.ascontiguousarray(embd, dtype=np.float32),
        codes,
        len(levels),
        to_rust(knn_params or KnnParams()),
        seed,
        verbose,
    )
    out = pl.DataFrame(
        {
            f"predicted_{label_column}": pl.Series(
                [levels[i] for i in pred.tolist()], dtype=pl.String
            ),
            f"confidence_{label_column}": conf,
        }
    )
    query.add_obs(out.with_columns(pl.Series("cell_idx", query.cells_to_keep)))
    return out


###############
# Persistence #
###############


@beartype
def save_symphony_ref(reference: SymphonyReference, path: Path | str) -> None:
    """Write a reference to one `.npz`, readable without pickling.

    The arrays go in as they are, the labels as Parquet bytes, everything
    else as JSON.

    Args:
        reference: The reference.
        path: The target file.

    Examples:
        >>> bv.annotate.save_symphony_ref(ref, "reference.npz")
    """
    arrays = {
        name: arr
        for name in _REF_ARRAYS
        if (arr := getattr(reference, name)) is not None
    }
    arrays["hvg_gene_names"] = np.asarray(reference.hvg_gene_names, dtype=str)
    if reference.labels is not None:
        buf = io.BytesIO()
        reference.labels.write_parquet(buf)
        arrays["labels"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    meta = {
        "format": SYMPHONY_FORMAT,
        "no_pcs": reference.no_pcs,
        "harmony_backend": reference.harmony_backend,
        "batch_vars": reference.batch_vars,
        "slim": reference.slim,
    }
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    with Path(path).open("wb") as fh:
        np.savez(fh, **arrays)


@beartype
def load_symphony_ref(path: Path | str) -> SymphonyReference:
    """Read a reference written by `save_symphony_ref`.

    Args:
        path: The file.

    Returns:
        The reference.

    Raises:
        ValueError: If the file is not a saved reference of this format.

    Examples:
        >>> ref = bv.annotate.load_symphony_ref("reference.npz")
    """
    with np.load(Path(path), allow_pickle=False) as fh:
        if "meta" not in fh:
            raise ValueError(f"{path} is not a saved Symphony reference.")
        meta = json.loads(fh["meta"].tobytes())
        if meta.get("format") != SYMPHONY_FORMAT:
            raise ValueError(
                f"{path} has reference format {meta.get('format')}, expected "
                f"{SYMPHONY_FORMAT}."
            )
        arrays = {name: fh[name] for name in _REF_ARRAYS if name in fh}
        hvg = [str(g) for g in fh["hvg_gene_names"]]
        labels = (
            pl.read_parquet(io.BytesIO(fh["labels"].tobytes()))
            if "labels" in fh
            else None
        )
    return SymphonyReference(
        hvg_gene_names=hvg,
        no_pcs=int(meta["no_pcs"]),
        harmony_backend=str(meta["harmony_backend"]),
        batch_vars=[str(b) for b in meta["batch_vars"]],
        slim=bool(meta["slim"]),
        labels=labels,
        **arrays,
    )
