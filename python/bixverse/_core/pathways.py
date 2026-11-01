"""Pathway activity and HotSpot: module scores, AUCell, VISION, ULM / MLM
and HotSpot gene autocorrelation and modules.

The work runs in Rust; results come back as `ScMatrixRes`, `LmActivityRes`,
`VisionAutocorRes` and `Hotspot` records.

Every function takes a `SingleCells`, a `SingleCellsSubset` or a `MetaCells`;
module scores are `SingleCells` only. Nothing is written to obs;
`get_data` turns a score matrix into an obs frame:

    >>> res = aucell(sc, {"t_cells": ["CD3E", "CD3D"]})
    >>> sc.add_obs(get_data(res))
    >>> hs = hotspot_gene_cor(sc, genes_to_take=genes)
    >>> hs = generate_hotspot_membership(hs)

Meta cells read their counts from memory: AUCell and HotSpot the raw counts,
VISION and ULM / MLM the normalised ones. Every index is
0-based.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from beartype import BeartypeConf, beartype

from bixverse import _bixverse
from bixverse._core.graph import _embedding, _first_dims
from bixverse._core.metacells import MetaCells, csr_args
from bixverse._core.qc import _auto_streaming
from bixverse._core.subset import ScOrMc, ScOrScSubset
from bixverse.params import (
    AucellParams,
    HotspotParams,
    LmActivityParams,
    VisionParams,
    to_rust,
)

###########
# Globals #
###########

GeneSets = Mapping[str, Sequence[str]]
Signatures = Mapping[str, Mapping[str, Sequence[str]]]

_CONF = BeartypeConf(is_pep484_tower=True)

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class ScMatrixRes:
    """Cells x gene sets scores, for cells and meta cells alike.

    The row index names which obs it joins onto.

    Attributes:
        values: `(n_cells, n_sets)` `float64` scores.
        columns: The gene set names.
        cell_ids: The cell (or meta cell) ids, row order.
        cell_indices: 0-based store index (or meta cell row) of each row.
        index_col: `"cell_idx"` for cells, `"meta_cell_idx"` for meta cells.
    """

    values: np.ndarray
    columns: tuple[str, ...]
    cell_ids: np.ndarray
    cell_indices: np.ndarray
    index_col: str

    @property
    def shape(self) -> tuple[int, int]:
        """`(n_cells, n_sets)`."""
        return (self.values.shape[0], self.values.shape[1])

    def frame(self) -> pl.DataFrame:
        """`cell_id` and one column per gene set."""
        return pl.DataFrame(
            {
                "cell_id": self.cell_ids,
                **dict(zip(self.columns, self.values.T, strict=True)),
            }
        )

    def __repr__(self) -> str:
        n, m = self.shape
        return f"ScMatrixRes: {n} cells x {m} gene sets ({', '.join(self.columns)})"


@beartype
@dataclass(frozen=True, repr=False)
class LmActivityRes:
    """ULM / MLM scores with p-values.

    Attributes:
        es: The activity scores.
        pvals: Two-sided p-values.
        fdr: Benjamini-Hochberg adjusted per cell across the sources, as
            decoupler does.
    """

    es: ScMatrixRes
    pvals: ScMatrixRes
    fdr: ScMatrixRes

    def __repr__(self) -> str:
        n, m = self.es.shape
        return f"LmActivityRes: {n} cells x {m} sources (es, pvals, fdr)"


@beartype
@dataclass(frozen=True, repr=False)
class VisionAutocorRes:
    """VISION scores with their autocorrelation.

    Attributes:
        vision_matrix: The scores.
        auto_cor: `gene_set_name`, `auto_cor` (`1 - Geary's C`), `p_val`
            (against the permuted sets of the same size and balance cluster)
            and `fdr`.
    """

    vision_matrix: ScMatrixRes
    auto_cor: pl.DataFrame

    def __repr__(self) -> str:
        n, m = self.vision_matrix.shape
        return f"VisionAutocorRes: {n} cells x {m} gene sets"


@beartype
@dataclass(frozen=True, repr=False)
class Hotspot:
    """HotSpot gene-gene local correlations.

    Attributes:
        z: `(n_genes, n_genes)` `float64` pair Z-scores, rows and columns in
            `genes` order.
        cor: The local correlation coefficients, same layout.
        genes: The gene ids.
        params: `used_cells`, the cell ids the graph was built over.
        module_membership: `gene_id` and `cluster_member` (0-based module,
            null if unassigned), once `generate_hotspot_membership` ran.
    """

    z: np.ndarray
    cor: np.ndarray
    genes: tuple[str, ...]
    params: dict[str, object]
    module_membership: pl.DataFrame | None = None

    def __repr__(self) -> str:
        """A summary of the genes, cells and modules."""
        n_genes = len(self.genes)
        used = self.params["used_cells"]
        assert isinstance(used, tuple)
        lines = [
            "Hotspot gene-gene local correlation results",
            f"  Genes: {n_genes}",
            f"  Cells: {len(used)}",
        ]
        m = self.module_membership
        if m is None:
            lines.append(
                "  Modules: not yet computed (see generate_hotspot_membership)"
            )
        else:
            assigned = m["cluster_member"].drop_nulls()
            lines.append(
                f"  Modules: {assigned.n_unique()} ({assigned.len()} genes assigned, "
                f"{n_genes - assigned.len()} unassigned)"
            )
        return "\n".join(lines)


###########
# Helpers #
###########


@beartype
def _gene_idx(sc: ScOrMc, ids: Sequence[str]) -> np.ndarray:
    """0-based gene indices, unknown ids dropped with a warning.

    Returns an empty vector rather than failing when nothing matches."""
    pos = {g: i for i, g in enumerate(sc.gene_ids().tolist())}
    hits = [pos[g] for g in ids if g in pos]
    if len(hits) < len(ids):
        warnings.warn(
            f"With the provided gene_ids a total of {len(ids) - len(hits)} could "
            "not be matched.",
            stacklevel=3,
        )
    return np.asarray(hits, dtype=np.uint32)


@beartype
def _rows(sc: ScOrMc) -> tuple[np.ndarray, np.ndarray, str]:
    """Ids, 0-based indices and index column of the rows results align to:
    the kept cells in `cells_to_keep` order, or the meta cells."""
    if isinstance(sc, MetaCells):
        return (
            sc.cell_ids(),
            np.arange(sc.n_cells, dtype=np.uint32),
            "meta_cell_idx",
        )
    keep = np.asarray(sc.cells_to_keep, dtype=np.uint32)
    return sc.cell_ids(keep), keep, "cell_idx"


@beartype
def _matrix_res(sc: ScOrMc, values: np.ndarray, columns: Sequence[str]) -> ScMatrixRes:
    """Wrap a score matrix over the result rows of `sc`."""
    ids, idx, col = _rows(sc)
    return ScMatrixRes(
        values=values,
        columns=tuple(columns),
        cell_ids=ids,
        cell_indices=idx,
        index_col=col,
    )


@beartype
def _streaming(n_cells: int, streaming: bool | None, verbose: bool) -> bool:
    """Resolve `streaming` by size, logging when it switches on."""
    res = _auto_streaming(n_cells, streaming)
    if res and verbose:
        logger.info("Setting streaming for large data to TRUE.")
    return res


@beartype
def _signatures(sc: ScOrMc, gs_list: Signatures) -> list[tuple[np.ndarray, np.ndarray]]:
    """`(pos, neg)` gene indices per VISION signature."""
    return [
        (_gene_idx(sc, list(gs.get("pos", []))), _gene_idx(sc, list(gs.get("neg", []))))
        for gs in gs_list.values()
    ]


@beartype
def _check_named(gs_list: Mapping[str, object]) -> None:
    """Check for a non-empty mapping of gene sets.

    Raises:
        ValueError: If it is empty.
    """
    if not gs_list:
        raise ValueError("gs_list must hold at least one named gene set.")


@beartype
def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment of a vector of p-values."""
    n = p.size
    if n == 0:
        return p.astype(np.float64)
    order = np.argsort(-p, kind="stable")
    ranked = p[order] * n / np.arange(n, 0, -1)
    adj = np.minimum(1.0, np.minimum.accumulate(ranked))
    out = np.empty(n, dtype=np.float64)
    out[order] = adj
    return out


@beartype
def lm_padj_per_obs(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg per row across the sources, decoupler's scheme."""
    return (
        np.vstack([bh_adjust(row) for row in p_values]) if p_values.size else p_values
    )


#################
# Module scores #
#################


@beartype
def module_scores(
    sc: ScOrScSubset,
    gs_list: GeneSets,
    *,
    n_bins: int = 24,
    n_ctrl: int = 100,
    seed: int = 42,
    streaming: bool | None = None,
    verbose: bool = False,
) -> ScMatrixRes:
    """Module activity scores.

    Seurat's `AddModuleScore`: per gene set the mean expression of its genes
    minus that of control genes drawn from the same expression bins.

    Args:
        sc: The experiment or a subset.
        gs_list: Gene set name to gene ids. Unknown ids warn and drop.
        n_bins: Expression bins.
        n_ctrl: Control genes per gene set gene.
        seed: Seed for the control draws.
        streaming: Read the cells in chunks. `None` decides by size.
        verbose: Controls verbosity.

    Returns:
        The `ScMatrixRes` over the kept cells.

    References:
        Tirosh, et al., Science, 2016.

    Examples:
        >>> res = bv.pathways.module_scores(sc, gene_sets, n_ctrl=50)
    """
    _check_named(gs_list)
    streaming = _streaming(sc.n_cells, streaming, verbose)
    sets = [_gene_idx(sc, list(g)) for g in gs_list.values()]
    res = _bixverse.module_scores(
        str(sc.cells_path),
        str(sc.genes_path),
        sets,
        np.asarray(sc.cells_to_keep, dtype=np.uint32),
        n_bins,
        n_ctrl,
        seed,
        streaming,
        verbose,
    )
    return _matrix_res(sc, res, list(gs_list))


##########
# AUCell #
##########


@beartype
def aucell(
    sc: ScOrMc,
    gs_list: GeneSets,
    *,
    aucell_params: AucellParams | None = None,
    streaming: bool | None = None,
    verbose: bool = False,
) -> ScMatrixRes:
    """AUCell scores.

    Ranks the genes within each cell and scores how high a gene set sits;
    see `AucellParams` for the three statistics. Meta cells rank their raw
    counts.

    Args:
        sc: The experiment, a subset or meta cells.
        gs_list: Gene set name to gene ids. Unknown ids warn and drop.
        aucell_params: The parameters. `None` uses the defaults.
        streaming: Read the cells in chunks. `None` decides by size; unused
            for meta cells.
        verbose: Controls verbosity.

    Returns:
        The `ScMatrixRes`.

    References:
        Aibar, et al., Nat. Methods, 2017.

    Examples:
        >>> res = bv.pathways.aucell(sc, gene_sets)
        >>> sc.add_obs(bv.pathways.get_data(res))
    """
    _check_named(gs_list)
    params = to_rust(aucell_params or AucellParams())
    sets = [_gene_idx(sc, list(g)) for g in gs_list.values()]
    if isinstance(sc, MetaCells):
        res = _bixverse.mc_aucell(*csr_args(sc.raw), sets, params, verbose)
    else:
        res = _bixverse.aucell(
            str(sc.cells_path),
            sets,
            np.asarray(sc.cells_to_keep, dtype=np.uint32),
            params,
            _streaming(sc.n_cells, streaming, verbose),
            verbose,
        )
    return _matrix_res(sc, res, list(gs_list))


@beartype
def get_data(res: ScMatrixRes, *, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """The scores as an obs frame for `add_obs`.

    One column per gene set plus the 0-based `cell_idx` (`meta_cell_idx` for
    meta cells).

    Args:
        res: The scores.
        columns: Gene sets to keep. `None` keeps all.

    Raises:
        ValueError: On an unknown gene set.
    """
    cols = list(res.columns) if columns is None else list(columns)
    missing = [c for c in cols if c not in res.columns]
    if missing:
        raise ValueError(f"Unknown gene set(s): {missing}")
    pos = {c: j for j, c in enumerate(res.columns)}
    return pl.DataFrame(
        {
            **{c: res.values[:, pos[c]] for c in cols},
            res.index_col: res.cell_indices.astype(np.uint32),
        }
    )


##########
# VISION #
##########


@beartype
def vision(
    sc: ScOrMc,
    gs_list: Signatures,
    *,
    streaming: bool | None = None,
    verbose: bool = False,
) -> ScMatrixRes:
    """VISION signature scores.

    Args:
        sc: The experiment, a subset or meta cells.
        gs_list: Signature name to `{"pos": [...], "neg": [...]}` gene ids;
            `"neg"` may be left out. Unknown ids warn and drop.
        streaming: Read the cells in chunks. `None` decides by size; unused
            for meta cells.
        verbose: Controls verbosity.

    Returns:
        The `ScMatrixRes`.

    References:
        DeTomaso, et al., Nat. Commun., 2019.

    Examples:
        >>> res = bv.pathways.vision(sc, signatures)
    """
    _check_named(gs_list)
    sigs = _signatures(sc, gs_list)
    if isinstance(sc, MetaCells):
        res = _bixverse.mc_vision(*csr_args(sc.norm), sigs, verbose)
    else:
        res = _bixverse.vision(
            str(sc.cells_path),
            sigs,
            np.asarray(sc.cells_to_keep, dtype=np.uint32),
            _streaming(sc.n_cells, streaming, verbose),
            verbose,
        )
    return _matrix_res(sc, res, list(gs_list))


@beartype
def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Lloyd's k-means with ten random starts, the lowest inertia kept.

    The clusters only group signatures of similar size and balance, so the
    exact algorithm matters little.

    Returns:
        0-based cluster per row, every cluster populated.
    """
    best, best_inertia = np.zeros(len(x), dtype=np.int64), np.inf
    for _ in range(10):
        centres = x[rng.choice(len(x), k, replace=False)]
        labels = np.full(len(x), -1, dtype=np.int64)
        for _ in range(100):
            d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                if (labels == j).any():
                    centres[j] = x[labels == j].mean(0)
        inertia = ((x - centres[labels]) ** 2).sum()
        if len(np.unique(labels)) == k and inertia < best_inertia:
            best, best_inertia = labels, inertia
    return best


@beartype
def generate_null_perm_gs(
    gs_list: Signatures,
    expr_genes: Sequence[str],
    *,
    n_perm: int = 500,
    n_comp: int = 5,
    seed: int = 42,
) -> tuple[list[list[dict[str, list[str]]]], np.ndarray]:
    """Random signatures for the VISION autocorrelation null.

    Signatures are clustered by `log10` size and share of positive genes;
    per cluster centre `n_perm` random signatures of that size and balance
    are drawn from `expr_genes`. With no more signatures than `n_comp`, each
    is its own cluster.

    Args:
        gs_list: The signatures, as for `vision`.
        expr_genes: The gene ids to draw from.
        n_perm: Random signatures per cluster.
        n_comp: Number of clusters.
        seed: Seed for the clustering and the draws.

    Returns:
        `(random_signatures, clusters)`: per cluster a list of
        `{"pos": [...], "neg": [...]}`, and the 0-based cluster of each
        signature.

    Examples:
        >>> random_sigs, clusters = bv.pathways.generate_null_perm_gs(
        ...     signatures, genes, n_perm=100
        ... )
    """
    sizes, balance = [], []
    for gs in gs_list.values():
        n_pos, n_neg = len(gs.get("pos", [])), len(gs.get("neg", []))
        sizes.append(np.log10(n_pos + n_neg))
        balance.append(n_pos / (n_pos + n_neg))
    x = np.column_stack([sizes, balance])
    rng = np.random.default_rng(seed)
    if len(x) <= n_comp:
        centres = x.copy()
        clusters = np.arange(len(x))
    else:
        n_comp = min(n_comp, len(np.unique(x, axis=0)))
        clusters = _kmeans(x, n_comp, rng)
        centres = np.vstack([x[clusters == j].mean(0) for j in range(n_comp)])

    genes = np.asarray(expr_genes)
    random_sigs = []
    for size_log, bal in centres:
        size = int(np.round(10**size_log))
        n_pos = int(np.ceil(bal * size))
        sigs = []
        for _ in range(n_perm):
            draw = rng.choice(genes, size, replace=False).tolist()
            sigs.append({"pos": draw[:n_pos], "neg": draw[n_pos:]})
        random_sigs.append(sigs)
    return random_sigs, clusters.astype(np.uint32)


@beartype
def _knn_for(
    sc: ScOrMc, use_knn: bool, n_rows: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The cached kNN when asked for and present, else `None` so Rust
    searches the embedding.

    Raises:
        ValueError: If the cached kNN is stale or does not cover the rows.
    """
    knn = sc.cache.knn if use_knn else None
    if knn is None:
        return None, None
    state = sc.check_state(["knn"])
    if state is not True:
        raise ValueError(state)
    if knn.indices.shape[0] != n_rows:
        raise ValueError(
            f"The kNN graph covers {knn.indices.shape[0]} cells but the embedding "
            f"{n_rows}. Regenerate the neighbours so the two agree."
        )
    return np.ascontiguousarray(knn.indices), np.ascontiguousarray(knn.distances)


@beartype
def _embedding_or_raise(
    sc: ScOrMc, embd_to_use: str, no_embd_to_use: int | None
) -> np.ndarray:
    """The embedding, its first `no_embd_to_use` columns, as `float32`.

    Raises:
        ValueError: If it is missing or stale.
    """
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        raise ValueError(f"No embedding '{embd_to_use}' found.")
    state = sc.check_state([embd_to_use])
    if state is not True:
        raise ValueError(state)
    return _first_dims(embd, no_embd_to_use)


@beartype
def _vision_autocor(
    sc: ScOrMc,
    gs_list: Signatures,
    embd: np.ndarray,
    random_gs: Sequence[Sequence[Mapping[str, Sequence[str]]]],
    clusters: np.ndarray,
    *,
    use_knn: bool,
    vision_params: VisionParams,
    streaming: bool | None,
    seed: int,
    verbose: bool,
) -> VisionAutocorRes:
    """`vision_w_autocor` with the null signatures given."""
    sigs = _signatures(sc, gs_list)
    # thousands of null sets, so one lookup table rather than one per set
    pos = {g: i for i, g in enumerate(sc.gene_ids().tolist())}

    def idx(ids: Sequence[str]) -> np.ndarray:
        return np.asarray([pos[g] for g in ids if g in pos], dtype=np.uint32)

    random = [
        [(idx(s.get("pos", [])), idx(s.get("neg", []))) for s in cl] for cl in random_gs
    ]
    knn_idx, knn_dist = _knn_for(sc, use_knn, embd.shape[0])
    membership = np.asarray(clusters, dtype=np.uint32)
    params = to_rust(vision_params)
    if isinstance(sc, MetaCells):
        res = _bixverse.mc_vision_autocor(
            *csr_args(sc.norm),
            sigs,
            random,
            membership,
            embd,
            knn_idx,
            knn_dist,
            params,
            seed,
            verbose,
        )
    else:
        res = _bixverse.vision_autocor(
            str(sc.cells_path),
            sigs,
            random,
            membership,
            np.asarray(sc.cells_to_keep, dtype=np.uint32),
            embd,
            knn_idx,
            knn_dist,
            params,
            _streaming(sc.n_cells, streaming, verbose),
            seed,
            verbose,
        )
    return VisionAutocorRes(
        vision_matrix=_matrix_res(sc, res["scores"], list(gs_list)),
        auto_cor=pl.DataFrame(
            {
                "gene_set_name": list(gs_list),
                "auto_cor": res["auto_cor"],
                "p_val": res["p_val"],
                "fdr": res["fdr"],
            }
        ),
    )


@beartype
def vision_w_autocor(
    sc: ScOrMc,
    gs_list: Signatures,
    embd_to_use: str,
    *,
    no_embd_to_use: int | None = None,
    use_knn: bool = True,
    vision_params: VisionParams | None = None,
    streaming: bool | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> VisionAutocorRes:
    """VISION scores with their local autocorrelation.

    Scores the signatures, then tests each for spatial structure: Geary's C
    over the kNN graph, against random signatures of similar size and
    balance (`generate_null_perm_gs`).

    The random signatures come from numpy's generator.

    Args:
        sc: The experiment, a subset or meta cells, with the embedding.
        gs_list: Signature name to `{"pos": [...], "neg": [...]}` gene ids.
        embd_to_use: The embedding for the kNN search, e.g. `"pca"`.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        use_knn: Take the cached kNN rather than searching the embedding.
        vision_params: The null and kNN parameters. `None` uses the defaults.
        streaming: Read the cells in chunks. `None` decides by size; unused
            for meta cells.
        seed: Seed for the null signatures and the kNN search.
        verbose: Controls verbosity.

    Returns:
        The `VisionAutocorRes`.

    Raises:
        ValueError: If the embedding or the cached kNN is missing or stale.

    References:
        DeTomaso, et al., Nat. Commun., 2019.

    Examples:
        >>> res = bv.pathways.vision_w_autocor(
        ...     sc, signatures, "pca", no_embd_to_use=15
        ... )
        >>> res.auto_cor
    """
    _check_named(gs_list)
    params = vision_params or VisionParams()
    embd = _embedding_or_raise(sc, embd_to_use, no_embd_to_use)
    if verbose:
        logger.info(
            "Generating %i random gene set clusters with a total of %i permutations.",
            params.n_cluster,
            params.n_perm,
        )
    random_gs, clusters = generate_null_perm_gs(
        gs_list,
        sc.gene_ids().tolist(),
        n_perm=params.n_perm,
        n_comp=params.n_cluster,
        seed=seed,
    )
    return _vision_autocor(
        sc,
        gs_list,
        embd,
        random_gs,
        clusters,
        use_knn=use_knn,
        vision_params=params,
        streaming=streaming,
        seed=seed,
        verbose=verbose,
    )


###############
# LM activity #
###############


@beartype
def prep_lm_network(
    net: pl.DataFrame, features: Sequence[str], tmin: int
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """A source-target network as weighted gene sets.

    Targets outside `features` are dropped, then sources with fewer than
    `tmin` targets; duplicated source-target pairs keep the first with a
    warning. A missing `weight` column means weight 1.

    Args:
        net: `source`, `target` and optionally `weight`.
        features: The gene ids, in index order.
        tmin: Minimum targets per source.

    Returns:
        Source (sorted) to `(0-based indices, weights)`, or `None` with a
        warning if no source survives.

    Raises:
        ValueError: On missing columns or non-finite weights.

    Examples:
        >>> sets = bv.pathways.prep_lm_network(net, genes, tmin=5)
    """
    if net.height == 0 or not {"source", "target"} <= set(net.columns):
        raise ValueError("net needs rows and the columns 'source' and 'target'.")
    if "weight" not in net.columns:
        net = net.with_columns(weight=pl.lit(1.0))
    net = net.select(
        pl.col("source").cast(pl.String),
        pl.col("target").cast(pl.String),
        pl.col("weight").cast(pl.Float64),
    )
    w = net["weight"]
    if w.null_count() or not np.isfinite(w.to_numpy()).all():
        raise ValueError("net$weight must be finite and not missing.")
    dups = ~net.select(pl.struct("source", "target").is_first_distinct()).to_series()
    if dups.any():
        warnings.warn(
            f"{int(dups.sum())} duplicated source-target pairs found. Keeping the "
            "first of each.",
            stacklevel=3,
        )
        net = net.filter(~dups)
    pos = {g: i for i, g in enumerate(features)}
    net = net.with_columns(
        idx=pl.col("target").replace_strict(pos, default=None, return_dtype=pl.Int64)
    ).drop_nulls("idx")
    counts = net.group_by("source").len()
    keep = counts.filter(pl.col("len") >= tmin)["source"]
    net = net.filter(pl.col("source").is_in(keep.implode()))
    if net.height == 0:
        warnings.warn(
            f"No source has at least tmin = {tmin} targets in the data. Returning "
            "None.",
            stacklevel=3,
        )
        return None
    out = {}
    for src in sorted(net["source"].unique().to_list()):
        sub = net.filter(pl.col("source") == src)
        out[src] = (
            sub["idx"].to_numpy().astype(np.uint32),
            np.ascontiguousarray(sub["weight"].to_numpy(), dtype=np.float64),
        )
    return out


@beartype
def lm_activity(
    sc: ScOrMc,
    net: pl.DataFrame,
    *,
    lm_params: LmActivityParams | None = None,
    verbose: bool = False,
) -> ScMatrixRes | LmActivityRes | None:
    """ULM / MLM activity scores (decoupler style).

    Per cell the normalised expression of every gene is regressed on the
    weights of each source's targets (ULM: one model per source, MLM: all
    sources jointly); the t value (or the slope) is the activity.

    Args:
        sc: The experiment, a subset or meta cells.
        net: `source`, `target` and optionally `weight`, see
            `prep_lm_network`.
        lm_params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The scores as a `ScMatrixRes`, or with `p_values` an `LmActivityRes`;
        `None` with a warning if no source keeps `tmin` targets. Sources
        come in sorted order.

    References:
        Badia-i-Mompel, et al., Bioinform. Adv., 2022.

    Examples:
        >>> res = bv.pathways.lm_activity(sc, net)
        >>> sc.add_obs(bv.pathways.get_data(res))
    """
    params = lm_params or LmActivityParams()
    gs = prep_lm_network(net, sc.gene_ids().tolist(), params.tmin)
    if gs is None:
        return None
    sets = list(gs.values())
    if isinstance(sc, MetaCells):
        res = _bixverse.mc_lm_activity(
            *csr_args(sc.norm), sets, to_rust(params), verbose
        )
    else:
        res = _bixverse.lm_activity(
            str(sc.cells_path),
            sets,
            np.asarray(sc.cells_to_keep, dtype=np.uint32),
            to_rust(params),
            verbose,
        )
    names = list(gs)
    es = _matrix_res(sc, res["es"], names)
    if res["p_values"] is None:
        return es
    return LmActivityRes(
        es=es,
        pvals=_matrix_res(sc, res["p_values"], names),
        fdr=_matrix_res(sc, lm_padj_per_obs(res["p_values"]), names),
    )


###########
# HotSpot #
###########


@beartype
def _hotspot_inputs(
    sc: ScOrMc,
    embd_to_use: str,
    use_knn: bool,
    no_embd_to_use: int | None,
    cells_to_take: Sequence[str] | None,
    genes_to_take: Sequence[str] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    list[str],
    np.ndarray | None,
    np.ndarray | None,
]:
    """The HotSpot inputs: the embedding rows, the cells, the genes and the kNN.

    Choosing cells drops the cached kNN, since its rows no longer match.

    Returns:
        `(embd, cell_indices, cell_ids, gene_indices, gene_ids, knn_indices,
        knn_dist)`; the cell indices are store indices (meta cell rows).

    Raises:
        ValueError: If the embedding is missing or stale, or no chosen cell is
            in it.
    """
    embd = _embedding_or_raise(sc, embd_to_use, no_embd_to_use)
    row_ids, row_idx, _ = _rows(sc)
    if cells_to_take is None:
        rows = np.arange(len(row_ids))
    else:
        use_knn = False
        pos = {c: i for i, c in enumerate(row_ids.tolist())}
        rows = np.asarray([pos[c] for c in cells_to_take if c in pos], dtype=np.int64)
        if rows.size == 0:
            raise ValueError(
                "There is no intersection in the provided cell names and embedding "
                "space. Please double check your parameters!"
            )
        if rows.size < len(cells_to_take):
            warnings.warn(
                f"{len(cells_to_take) - rows.size} of cells_to_take are not in the "
                "embedding and were dropped.",
                stacklevel=3,
            )
    genes = sc.gene_ids().tolist() if genes_to_take is None else list(genes_to_take)
    gene_idx = _gene_idx(sc, genes)
    knn_idx, knn_dist = _knn_for(sc, use_knn, len(row_ids))
    return (
        np.ascontiguousarray(embd[rows]),
        row_idx[rows].astype(np.uint32),
        row_ids[rows].tolist(),
        gene_idx,
        sc.gene_ids(gene_idx).tolist(),
        knn_idx,
        knn_dist,
    )


@beartype
def hotspot_autocor(
    sc: ScOrMc,
    *,
    embd_to_use: str = "pca",
    use_knn: bool = True,
    hotspot_params: HotspotParams | None = None,
    no_embd_to_use: int | None = None,
    cells_to_take: Sequence[str] | None = None,
    genes_to_take: Sequence[str] | None = None,
    streaming: bool | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> pl.DataFrame:
    """HotSpot gene autocorrelation.

    How much each gene's expression varies smoothly over the kNN graph of
    the embedding. Meta cells model their raw counts.

    Args:
        sc: The experiment, a subset or meta cells, with the embedding.
        embd_to_use: The embedding for the kNN, e.g. `"pca"`.
        use_knn: Take the cached kNN rather than searching the embedding.
            Ignored when `cells_to_take` is given.
        hotspot_params: The parameters. `None` uses the defaults.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        cells_to_take: Cell (meta cell) ids to use. `None` for all.
        genes_to_take: Gene ids to test. `None` for all.
        streaming: Stream the genes. `None` decides by size; unused for meta
            cells.
        seed: Seed for the kNN search.
        verbose: Controls verbosity.

    Returns:
        `gene_id`, `gearys_c`, `z_score`, `pval` and `fdr`.

    Raises:
        ValueError: If the embedding or the cached kNN is missing or stale.

    References:
        DeTomaso, et al., Cell Syst., 2021.

    Examples:
        >>> autocor = bv.pathways.hotspot_autocor(sc)
        >>> autocor = bv.pathways.hotspot_autocor(sc, genes_to_take=genes[:200])
    """
    params = to_rust(hotspot_params or HotspotParams())
    embd, cells, _, genes, _, knn_idx, knn_dist = _hotspot_inputs(
        sc, embd_to_use, use_knn, no_embd_to_use, cells_to_take, genes_to_take
    )
    if isinstance(sc, MetaCells):
        res = _bixverse.mc_hotspot_autocor(
            *csr_args(sc.raw),
            embd,
            knn_idx,
            knn_dist,
            params,
            cells,
            genes,
            seed,
            verbose,
        )
    else:
        res = _bixverse.hotspot_autocor(
            str(sc.genes_path),
            str(sc.cells_path),
            embd,
            knn_idx,
            knn_dist,
            params,
            cells,
            genes,
            _streaming(len(cells), streaming, verbose),
            seed,
            verbose,
        )
    return pl.DataFrame(
        {
            "gene_id": sc.gene_ids(res["gene_idx"]),
            "gearys_c": res["gearys_c"],
            "z_score": res["z_score"],
            "pval": res["pval"],
            "fdr": res["fdr"],
        }
    )


@beartype(conf=_CONF)
def hotspot_gene_cor(
    sc: ScOrMc,
    *,
    embd_to_use: str = "pca",
    use_knn: bool = True,
    hotspot_params: HotspotParams | None = None,
    no_embd_to_use: int | None = None,
    cells_to_take: Sequence[str] | None = None,
    genes_to_take: Sequence[str] | None = None,
    streaming: bool | None = None,
    working_mem_gb: float = 4.0,
    seed: int = 42,
    verbose: bool = False,
) -> Hotspot:
    """HotSpot gene-gene local correlations.

    Args:
        sc: The experiment, a subset or meta cells, with the embedding.
        embd_to_use: The embedding for the kNN, e.g. `"pca"`.
        use_knn: Take the cached kNN rather than searching the embedding.
            Ignored when `cells_to_take` is given.
        hotspot_params: The parameters. `None` uses the defaults.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        cells_to_take: Cell (meta cell) ids to use. `None` for all.
        genes_to_take: Gene ids to correlate. `None` for all; the output is
            genes by genes, so pick the autocorrelated ones first.
        streaming: Stream the genes. `None` decides by size; unused for meta
            cells.
        working_mem_gb: Memory for the resident gene panels of the streaming
            path, excluding the two output matrices.
        seed: Seed for the kNN search.
        verbose: Controls verbosity.

    Returns:
        The `Hotspot` result.

    Raises:
        ValueError: If the embedding or the cached kNN is missing or stale.

    References:
        DeTomaso, et al., Cell Syst., 2021.

    Examples:
        >>> hs = bv.pathways.hotspot_gene_cor(sc, genes_to_take=top)
    """
    params = to_rust(hotspot_params or HotspotParams())
    embd, cells, cell_ids, genes, gene_ids, knn_idx, knn_dist = _hotspot_inputs(
        sc, embd_to_use, use_knn, no_embd_to_use, cells_to_take, genes_to_take
    )
    if isinstance(sc, MetaCells):
        res = _bixverse.mc_hotspot_gene_cor(
            *csr_args(sc.raw),
            embd,
            knn_idx,
            knn_dist,
            params,
            cells,
            genes,
            seed,
            verbose,
        )
    else:
        res = _bixverse.hotspot_gene_cor(
            str(sc.genes_path),
            str(sc.cells_path),
            embd,
            knn_idx,
            knn_dist,
            params,
            cells,
            genes,
            float(working_mem_gb),
            _streaming(len(cells), streaming, verbose),
            seed,
            verbose,
        )
    return Hotspot(
        z=res["z"],
        cor=res["cor"],
        genes=tuple(gene_ids),
        params={"used_cells": tuple(cell_ids)},
    )


@beartype(conf=_CONF)
def generate_hotspot_membership(
    x: Hotspot, *, fdr_threshold: float = 0.05, min_size: int = 10
) -> Hotspot:
    """Gene modules from the pair Z-scores.

    Average-linkage clustering of the Z-scores, cut as HotSpot's
    `compute_modules` does.

    Args:
        x: The `hotspot_gene_cor` result.
        fdr_threshold: BH level at which a pair counts as significant.
        min_size: Minimum genes per module.

    Returns:
        A copy with `module_membership` set.

    Raises:
        ValueError: If `fdr_threshold` is outside `[0, 1]`.

    Examples:
        >>> hs = bv.pathways.generate_hotspot_membership(hs, min_size=5)
    """
    if not 0 <= fdr_threshold <= 1:
        raise ValueError("fdr_threshold must be in [0, 1].")
    labels = _bixverse.hotspot_cluster_genes(
        np.ascontiguousarray(x.z, dtype=np.float64), float(fdr_threshold), min_size
    )
    member = pl.Series("cluster_member", labels).fill_nan(None).cast(pl.Int64)
    return dataclasses.replace(
        x,
        module_membership=pl.DataFrame(
            {"gene_id": list(x.genes), "cluster_member": member}
        ),
    )


@beartype
def get_hotspot_membership(x: Hotspot) -> pl.DataFrame | None:
    """The module membership.

    Args:
        x: The HotSpot result.

    Returns:
        `gene_id` and `cluster_member`, or `None` before
        `generate_hotspot_membership`.
    """
    return x.module_membership
