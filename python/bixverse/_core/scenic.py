"""SCENIC: gene filter, GRN inference, TF to gene links, CisTarget motif
enrichment, regulons and their binarisation.

The work runs in Rust.

    >>> grn = scenic_grn(sc, tf_ids=tfs)
    >>> grn = identify_tf_to_genes(grn)
    >>> grn = tf_to_genes_correlations(grn, sc)
    >>> grn = tf_to_genes_motif_enrichment(grn, ranks, genes, motifs, annot)
    >>> regulons = build_regulons(grn)
    >>> auc = aucell(sc, regulons)
    >>> on_off = binarise_regulon_activity(auc)

`ScenicGrn` is frozen; each step returns an updated copy. The motif ranking
database is the caller's: nothing here downloads one.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Literal

import numpy as np
import polars as pl
from beartype import BeartypeConf, beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells, csr_args, csr_subset
from bixverse._core.pathways import Hotspot, ScMatrixRes, _gene_idx, _streaming
from bixverse._core.repr import compact_repr
from bixverse._core.subset import ScOrMc
from bixverse.params import (
    CistargetParams,
    ScenicBinariseParams,
    ScenicParams,
    to_rust,
)

###########
# Globals #
###########

# meta cells aggregate tens of cells each, so a leaf of 20 or more meta cells
# starves the trees, so it is lowered to this
MC_MIN_SAMPLES_LEAF_CAP: int = 20
MC_MIN_SAMPLES_LEAF: int = 10

Mode = Literal["activating", "repressing", "both"]
MODE_SIGNS: dict[str, tuple[int, ...]] = {
    "activating": (1,),
    "repressing": (-1,),
    "both": (-1, 1),
}

_CONF = BeartypeConf(is_pep484_tower=True)

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class ScenicGrn:
    """SCENIC GRN inference results.

    Attributes:
        importance_matrix: `(n_genes, n_tfs)` `float64` importances.
        gene_ids: The target genes, the rows.
        tf_ids: The TFs, the columns.
        params: The parameters of the run.
        tf_to_gene_results: The TF to gene links once
            `identify_tf_to_genes` ran, later extended by the correlation and
            motif steps.
        cis_targets_results: The CisTarget motif enrichment once
            `tf_to_genes_motif_enrichment` ran.
    """

    importance_matrix: np.ndarray
    gene_ids: tuple[str, ...]
    tf_ids: tuple[str, ...]
    params: ScenicParams
    tf_to_gene_results: pl.DataFrame = field(default_factory=pl.DataFrame)
    cis_targets_results: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __repr__(self) -> str:
        """A short summary of the results."""
        return compact_repr(
            "ScenicGrn",
            n_genes=len(self.gene_ids),
            n_tfs=len(self.tf_ids),
            learner=self.params.learner_type,
            tf_to_gene=self.tf_to_gene_results.height > 0,
            cis_targets=self.cis_targets_results.height > 0,
        )

    def frame(self) -> pl.DataFrame:
        """The importances as `gene_id` and one column per TF."""
        return pl.DataFrame(
            {
                "gene_id": list(self.gene_ids),
                **dict(zip(self.tf_ids, self.importance_matrix.T, strict=True)),
            }
        )


@beartype
@dataclass(frozen=True, repr=False)
class RegulonBinarisation:
    """On/off regulon calls.

    Attributes:
        binary: `cell_id` and one boolean column per regulon, `True` where the
            score sits strictly above the threshold.
        thresholds: `regulon`, `threshold`, `bimodal` and `n_cells_on`.
    """

    binary: pl.DataFrame
    thresholds: pl.DataFrame

    def __repr__(self) -> str:
        return compact_repr(
            "RegulonBinarisation",
            n_cells=self.binary.height,
            n_regulons=self.thresholds.height,
            n_bimodal=int(self.thresholds["bimodal"].sum()),
        )


###############
# Gene filter #
###############


@beartype
def _cells(sc: ScOrMc, cells_to_take: Sequence[str] | None) -> np.ndarray:
    """0-based store cells (meta cell rows) to use: the kept ones by
    default."""
    if cells_to_take is None:
        if isinstance(sc, MetaCells):
            return np.arange(sc.n_cells, dtype=np.uint32)
        return np.asarray(sc.cells_to_keep, dtype=np.uint32)
    return np.asarray(sc.cell_indices(list(cells_to_take)), dtype=np.uint32)


@beartype
def _mc_params(params: ScenicParams, verbose: bool) -> ScenicParams:
    """Cap `min_samples_leaf` for meta cells."""
    # by design: with the single cell leaf sizes on the far fewer meta cells, the
    # trees are not very informative
    lp = params.learner
    if lp.min_samples_leaf < MC_MIN_SAMPLES_LEAF_CAP:
        return params
    if verbose:
        logger.info(
            "The mean leafs per sample is set quite high for meta cells. "
            "Reducing to %i.",
            MC_MIN_SAMPLES_LEAF,
        )
    return dataclasses.replace(
        params,
        learner_params=dataclasses.replace(lp, min_samples_leaf=MC_MIN_SAMPLES_LEAF),
    )


@beartype
def scenic_gene_filter(
    sc: ScOrMc,
    *,
    scenic_params: ScenicParams | None = None,
    cells_to_take: Sequence[str] | None = None,
    verbose: bool = False,
) -> list[str]:
    """Genes with enough counts and enough expressing cells.

    A gene passes with at least `min_counts` total counts and a share of at
    least `min_cells` expressing cells. Meta cells are filtered on their raw
    counts in memory.

    Args:
        sc: The experiment, a subset or meta cells.
        scenic_params: The parameters. `None` uses the defaults.
        cells_to_take: Cell (meta cell) ids to use. `None` for the kept cells.
        verbose: Controls verbosity.

    Returns:
        The passing gene ids, in gene order.

    Examples:
        >>> keep = bv.pathways.scenic_gene_filter(
        ...     sc, scenic_params=bv.params.ScenicParams(min_counts=50)
        ... )
    """
    params = scenic_params or ScenicParams()
    cells = _cells(sc, cells_to_take)
    if isinstance(sc, MetaCells):
        m = csr_subset(sc.raw, cells, None)
        total = np.bincount(m.indices, weights=m.data, minlength=sc.n_genes)
        expressed = np.bincount(m.indices[m.data != 0], minlength=sc.n_genes) / max(
            m.shape[0], 1
        )
        passing = np.flatnonzero(
            (total >= params.min_counts) & (expressed >= params.min_cells)
        )
        if verbose:
            logger.info(
                "SCENIC gene filter: %i / %i genes pass.", passing.size, sc.n_genes
            )
    else:
        passing = _bixverse.scenic_filter(
            str(sc.genes_path), cells, to_rust(params), verbose
        )
    return sc.gene_ids(passing).tolist()


#######
# GRN #
#######


@beartype
def _match_features(features: Sequence[str], available: Sequence[str]) -> list[str]:
    """Unknown features warn and drop.

    Raises:
        ValueError: If none match.
    """
    have = set(available)
    kept = [f for f in features if f in have]
    if len(kept) < len(features):
        warnings.warn(
            f"{len(features) - len(kept)} features could not be matched.",
            stacklevel=3,
        )
    if not kept:
        raise ValueError(
            "No features matched. Please double check provided parameters!"
        )
    return kept


@beartype
def scenic_grn(
    sc: ScOrMc,
    tf_ids: Sequence[str],
    *,
    scenic_params: ScenicParams | None = None,
    genes_to_take: Sequence[str] | None = None,
    cells_to_take: Sequence[str] | None = None,
    streaming: bool | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> ScenicGrn:
    """GRN inference, the GENIE3 / GRNBoost2 step of SCENIC.

    Regresses every target gene on the TFs with tree ensembles and keeps the
    feature importances. TFs must also pass as targets. On meta cells a
    `min_samples_leaf` of 20 or more is lowered to 10.

    Args:
        sc: The experiment, a subset or meta cells.
        tf_ids: The TF gene ids; ids not in the object drop, with a warning
            when `verbose` is on.
        scenic_params: The parameters. `None` uses the defaults.
        genes_to_take: Target gene ids. `None` runs `scenic_gene_filter`.
        cells_to_take: Cell (meta cell) ids. `None` for the kept cells.
        streaming: Stream the targets. `None` decides by size; unused for
            meta cells.
        seed: Seed for the learners.
        verbose: Controls verbosity.

    Returns:
        The `ScenicGrn`.

    Raises:
        ValueError: If no TF is in the object or none passes as a target.

    References:
        Aibar, et al., Nat. Methods, 2017; Moerman, et al., Bioinformatics,
        2019.

    Examples:
        >>> grn = bv.pathways.scenic_grn(sc, tf_ids, genes_to_take=keep)
    """
    return _scenic_grn(
        sc,
        tf_ids,
        scenic_params=scenic_params,
        genes_to_take=genes_to_take,
        cells_to_take=cells_to_take,
        streaming=streaming,
        seed=seed,
        verbose=verbose,
        ext=_bixverse,
    )


@beartype
def _scenic_grn(
    sc: ScOrMc,
    tf_ids: Sequence[str],
    *,
    scenic_params: ScenicParams | None = None,
    genes_to_take: Sequence[str] | None = None,
    cells_to_take: Sequence[str] | None = None,
    streaming: bool | None = None,
    seed: int = 42,
    verbose: bool = False,
    ext: ModuleType | SimpleNamespace,
) -> ScenicGrn:
    """The body of `scenic_grn`, with the extension module `ext` supplied, so
    the GPU twin shares it."""
    params = scenic_params or ScenicParams()
    if isinstance(sc, MetaCells):
        params = _mc_params(params, verbose)
    cells = _cells(sc, cells_to_take)
    if genes_to_take is None:
        if verbose:
            logger.info("No target genes supplied, running gene filter...")
        genes_to_take = scenic_gene_filter(
            sc, scenic_params=params, cells_to_take=cells_to_take, verbose=verbose
        )
    all_genes = sc.gene_ids().tolist()
    targets = _match_features(genes_to_take, all_genes)

    have = set(all_genes)
    tf_found = [t for t in tf_ids if t in have]
    n_dropped = len(tf_ids) - len(tf_found)
    if n_dropped and verbose:
        warnings.warn(
            f"{n_dropped} TF identifier(s) not found in the object and dropped.",
            stacklevel=2,
        )
    if not tf_found:
        raise ValueError("No provided TF identifiers match genes in the object.")
    target_set = set(targets)
    tfs = list(dict.fromkeys(t for t in tf_found if t in target_set))
    if not tfs:
        raise ValueError(
            "No TFs remain after intersecting with target gene indices. Consider "
            "relaxing min_counts / min_cells thresholds."
        )
    if verbose:
        logger.info(
            "SCENIC: %i target genes, %i TFs, %i cells",
            len(targets),
            len(tfs),
            cells.size,
        )

    gene_idx = _gene_idx(sc, targets)
    if isinstance(sc, MetaCells):
        m = csr_subset(sc.raw, cells, gene_idx)
        pos = {g: i for i, g in enumerate(targets)}
        tf_cols = np.asarray([pos[t] for t in tfs], dtype=np.uint32)
        imp = ext.mc_scenic_grn(*csr_args(m), tf_cols, to_rust(params), seed, verbose)
    else:
        imp = ext.scenic_grn(
            str(sc.genes_path),
            cells,
            gene_idx,
            _gene_idx(sc, tfs),
            to_rust(params),
            _streaming(cells.size, streaming, verbose),
            seed,
            verbose,
        )
    return ScenicGrn(
        importance_matrix=imp,
        gene_ids=tuple(targets),
        tf_ids=tuple(tfs),
        params=params,
    )


###########
# Getters #
###########


@beartype
def get_params(x: ScenicGrn | Hotspot) -> ScenicParams | dict[str, object]:
    """The parameters of the run.

    Args:
        x: A SCENIC GRN result or a HotSpot result.

    Returns:
        The `ScenicParams` for a GRN, a dict copy for HotSpot.
    """
    return x.params if isinstance(x, ScenicGrn) else dict(x.params)


@beartype
def get_tf_to_gene(x: ScenicGrn) -> pl.DataFrame:
    """The TF to gene links; empty with a warning
    before `identify_tf_to_genes`.

    Args:
        x: The SCENIC GRN result.
    """
    if x.tf_to_gene_results.height == 0:
        warnings.warn(
            "You did not seem to have run identify_tf_to_genes(). Returning an "
            "empty frame.",
            stacklevel=2,
        )
    return x.tf_to_gene_results.clone()


@beartype
def get_cistarget_res(x: ScenicGrn) -> pl.DataFrame:
    """The motif enrichment; empty with a warning
    before `tf_to_genes_motif_enrichment`.

    Args:
        x: The SCENIC GRN result.
    """
    if x.cis_targets_results.height == 0:
        warnings.warn(
            "You did not seem to have run tf_to_genes_motif_enrichment(). "
            "Returning an empty frame.",
            stacklevel=2,
        )
    return x.cis_targets_results.clone()


####################
# TF to gene links #
####################


@beartype
def _links_frame(x: ScenicGrn, links: dict[str, np.ndarray]) -> pl.DataFrame:
    """Rust's `(gene, tf, importance)` positions as named links."""
    genes = np.asarray(x.gene_ids)
    tfs = np.asarray(x.tf_ids)
    return pl.DataFrame(
        {
            "tf": tfs[links["tf"].astype(np.int64)],
            "gene": genes[links["gene"].astype(np.int64)],
            "importance": links["importance"],
        },
        schema={"tf": pl.String, "gene": pl.String, "importance": pl.Float64},
    )


@beartype(conf=_CONF)
def identify_tf_to_genes(
    x: ScenicGrn,
    *,
    method: Literal["threshold", "top_k"] = "threshold",
    k_tfs: int | None = None,
    k_genes: int | None = None,
    n_sd: float = 2.0,
    min_importance: float | None = None,
    verbose: bool = False,
) -> ScenicGrn:
    """TF to gene links from the importances.

    `"threshold"` keeps, per target gene, the TFs at or above `mean + n_sd *
    sd` of its importances. `"top_k"` keeps the `k_tfs` strongest regulators
    per gene and / or the `k_genes` strongest targets per TF. Duplicate links
    are dropped, the first kept.

    Args:
        x: The `scenic_grn` result.
        method: `"threshold"` or `"top_k"`.
        k_tfs: Regulators per gene for `"top_k"`.
        k_genes: Targets per TF for `"top_k"`.
        n_sd: Standard deviations above the mean for `"threshold"`.
        min_importance: Optional floor on the importance, in `[0, 1]`.
        verbose: Controls verbosity.

    Returns:
        A copy with `tf_to_gene_results` set: `tf`, `gene`, `importance`.

    Raises:
        ValueError: On `"top_k"` without `k_tfs` or `k_genes`, or a value out
            of range.

    Examples:
        >>> grn = bv.pathways.identify_tf_to_genes(grn, n_sd=1.5)
        >>> grn_topk = bv.pathways.identify_tf_to_genes(grn, method="top_k", k_tfs=3)
    """
    if n_sd <= 0:
        raise ValueError("n_sd must be positive.")
    if min_importance is not None and not 0 <= min_importance <= 1:
        raise ValueError("min_importance must be in [0, 1].")
    imp = np.ascontiguousarray(x.importance_matrix, dtype=np.float64)
    min_value = None if min_importance is None else float(min_importance)
    match method:
        case "top_k":
            if k_tfs is None and k_genes is None:
                raise ValueError(
                    "k_tfs and k_genes cannot both be None when method = 'top_k'."
                )
            if verbose:
                logger.info("Extracting TF to gene associations via top-k filtering.")
            parts = []
            if k_tfs is not None:
                parts.append(
                    _links_frame(
                        x, _bixverse.top_k_targets(imp, k_tfs, True, min_value)
                    )
                )
            if k_genes is not None:
                parts.append(
                    _links_frame(
                        x, _bixverse.top_k_targets(imp, k_genes, False, min_value)
                    )
                )
            links = pl.concat(parts)
        case "threshold":
            if verbose:
                logger.info(
                    "Extracting TF to gene associations via per-gene threshold "
                    "(mean + %.1f * SD).",
                    n_sd,
                )
            links = _links_frame(
                x, _bixverse.importance_threshold(imp, float(n_sd), min_value)
            )
        case _:
            raise ValueError(f"Unknown method '{method}'.")
    links = links.unique(subset=["tf", "gene"], keep="first", maintain_order=True)
    return dataclasses.replace(x, tf_to_gene_results=links)


@beartype(conf=_CONF)
def tf_to_genes_correlations(
    x: ScenicGrn,
    sc: ScOrMc,
    *,
    rho_threshold: float = 0.03,
    mode: Mode = "activating",
    remove_self: bool = True,
    spearman: bool = True,
    verbose: bool = False,
) -> ScenicGrn:
    """Sign the TF to gene links by correlation.

    Correlates each TF with its target over the kept cells (the normalised
    counts on meta cells), signs the link `+1` above `rho_threshold`, `-1`
    below its negative and `0` between, and keeps the signs `mode` asks
    for.

    Args:
        x: The result with TF to gene links.
        sc: The object the GRN ran on.
        rho_threshold: Correlation band around zero counted as unsigned.
        mode: Which links to keep.
        remove_self: Drop TF to itself links.
        spearman: Spearman rather than Pearson.
        verbose: Controls verbosity.

    Returns:
        A copy whose links gain `pairwise_cor` and `cor_sign`, filtered; `x`
        itself with a warning if there are no links.

    Raises:
        ValueError: If `rho_threshold` is outside `[0, 1]`.

    Examples:
        >>> grn = bv.pathways.tf_to_genes_correlations(grn, sc)
        >>> grn_both = bv.pathways.tf_to_genes_correlations(grn, sc, mode="both")
    """
    if not 0 <= rho_threshold <= 1:
        raise ValueError("rho_threshold must be in [0, 1].")
    links = x.tf_to_gene_results
    if links.height == 0:
        warnings.warn(
            "No TF to gene pairs found. Returning class as is. Did you run "
            "identify_tf_to_genes()?",
            stacklevel=2,
        )
        return x
    if verbose:
        logger.info("Calculating the pairwise correlations between the TFs and genes")
    g1 = _gene_idx(sc, links["tf"].to_list())
    g2 = _gene_idx(sc, links["gene"].to_list())
    if isinstance(sc, MetaCells):
        cors = _bixverse.mc_pairwise_gene_cors(
            *csr_args(sc.norm), g1, g2, spearman, verbose
        )
    else:
        cors = _bixverse.pairwise_gene_cors(
            str(sc.genes_path),
            g1,
            g2,
            np.asarray(sc.cells_to_keep, dtype=np.uint32),
            spearman,
            verbose,
        )
    links = links.with_columns(pairwise_cor=pl.Series(cors)).with_columns(
        cor_sign=(pl.col("pairwise_cor") > rho_threshold).cast(pl.Int32)
        - (pl.col("pairwise_cor") < -rho_threshold).cast(pl.Int32)
    )
    if verbose:
        logger.info("Keeping %s TF <> gene links at |rho| > %.3f", mode, rho_threshold)
    links = links.filter(pl.col("cor_sign").is_in(list(MODE_SIGNS[mode])))
    if remove_self:
        links = links.filter(pl.col("tf") != pl.col("gene"))
    return dataclasses.replace(x, tf_to_gene_results=links)


#############
# CisTarget #
#############


@beartype(conf=_CONF)
def run_cistarget(
    gs_list: Mapping[str, Sequence[str]],
    rankings: np.ndarray,
    rank_genes: Sequence[str],
    rank_motifs: Sequence[str],
    annot_data: pl.DataFrame,
    *,
    cis_target_params: CistargetParams | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """CisTarget motif enrichment of gene sets.

    Args:
        gs_list: Gene set name to gene ids.
        rankings: `(n_genes, n_motifs)` integer motif ranks, the database.
        rank_genes: The genes of the rows.
        rank_motifs: The motifs of the columns.
        annot_data: `motif`, `TF` and `annotationSource`, the motif to TF
            annotation.
        cis_target_params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        `motif`, `gs_name`, `nes`, `auc`, `rank_at_max`, `n_enriched`,
        `leading_edge_genes` (`;` separated), `TF_highConf` and `TF_lowConf`,
        sorted by gene set and descending NES. The TF lists are sorted in C
        order.

    Raises:
        ValueError: On a malformed ranking or annotation.

    References:
        Aibar, et al., Nat. Methods, 2017.

    Examples:
        >>> cis = bv.pathways.run_cistarget(
        ...     tf_targets, rankings, rank_genes, rank_motifs, annot
        ... )
    """
    p = cis_target_params or CistargetParams()
    ranks = np.ascontiguousarray(rankings, dtype=np.int32)
    if ranks.ndim != 2 or ranks.shape != (len(rank_genes), len(rank_motifs)):
        raise ValueError("rankings must be rank_genes x rank_motifs.")
    if not {"motif", "TF", "annotationSource"} <= set(annot_data.columns):
        raise ValueError("annot_data needs 'motif', 'TF' and 'annotationSource'.")
    motifs = np.asarray(rank_motifs)
    genes = np.asarray(rank_genes)
    annot = annot_data.filter(pl.col("motif").is_in(list(rank_motifs)))
    gs_idx = [
        np.flatnonzero(np.isin(genes, list(g))).astype(np.uint32)
        for g in gs_list.values()
    ]
    cats = set(p.low_conf_cats) | set(p.high_conf_cats)
    if not cats <= set(annot["annotationSource"].to_list()):
        warnings.warn(
            "Not all of the high and low confidence categories were found",
            stacklevel=2,
        )
    if any(i.size == 0 for i in gs_idx):
        warnings.warn(
            "Some of the gene sets have zero overlap with the ranking.", stacklevel=2
        )
    n_genes = ranks.shape[0]
    res = _bixverse.cistarget(
        ranks,
        gs_idx,
        int(p.auc_threshold * n_genes),
        float(p.nes_threshold),
        min(p.max_rank, n_genes),
        p.rcc_method,
        p.n_mean,
        verbose,
    )
    frames = []
    for name, r in zip(gs_list, res, strict=True):
        if r["nes"].size == 0:
            continue
        frames.append(
            pl.DataFrame(
                {
                    "gs_name": name,
                    "motif": motifs[r["motif_idx"].astype(np.int64)],
                    "nes": r["nes"],
                    "auc": r["auc"],
                    "rank_at_max": r["rank_at_max"].astype(np.int32),
                    "n_enriched": r["n_enriched"].astype(np.int32),
                    "leading_edge_genes": [
                        ";".join(genes[le.astype(np.int64)]) for le in r["leading_edge"]
                    ],
                }
            )
        )
    schema = {
        "gs_name": pl.String,
        "motif": pl.String,
        "nes": pl.Float64,
        "auc": pl.Float64,
        "rank_at_max": pl.Int32,
        "n_enriched": pl.Int32,
        "leading_edge_genes": pl.String,
    }
    cis = pl.concat(frames) if frames else pl.DataFrame(schema=schema)

    def tf_list(cats: Sequence[str], name: str) -> pl.DataFrame:
        return (
            annot.filter(pl.col("annotationSource").is_in(list(cats)))
            .group_by("motif")
            .agg(pl.col("TF").unique().sort().str.join(";").alias(name))
        )

    return (
        cis.join(tf_list(p.high_conf_cats, "TF_highConf"), on="motif", how="left")
        .join(tf_list(p.low_conf_cats, "TF_lowConf"), on="motif", how="left")
        .select(
            "motif", *[c for c in schema if c != "motif"], "TF_highConf", "TF_lowConf"
        )
        .sort(["gs_name", "nes", "motif"], descending=[False, True, False])
    )


@beartype
def _leading_edge_links(
    cis: pl.DataFrame, tf_col: str, symbol_to_id: Mapping[str, str] | None
) -> pl.DataFrame:
    """Leading edge genes of the motifs annotated back to their own gene
    set's TF."""
    empty = pl.DataFrame(schema={"tf": pl.String, "le_gene": pl.String})
    dt = cis.filter(pl.col(tf_col).is_not_null())
    if dt.height == 0:
        return empty
    le = (
        dt.with_columns(single_tf=pl.col(tf_col).str.split(";"))
        .explode("single_tf")
        .filter(pl.col("single_tf") == pl.col("gs_name"))
        .select(
            tf=pl.col("gs_name"), le_gene=pl.col("leading_edge_genes").str.split(";")
        )
        .explode("le_gene")
    )
    if le.height == 0:
        return empty
    if symbol_to_id is not None:
        m = dict(symbol_to_id)
        le = le.with_columns(
            tf=pl.col("tf").replace_strict(m, default=None),
            le_gene=pl.col("le_gene").replace_strict(m, default=None),
        ).drop_nulls()
    return le


@beartype
def tf_to_genes_motif_enrichment(
    x: ScenicGrn,
    rankings: np.ndarray,
    rank_genes: Sequence[str],
    rank_motifs: Sequence[str],
    annot_data: pl.DataFrame,
    *,
    cis_target_params: CistargetParams | None = None,
    gene_id_to_symbol: Mapping[str, str] | None = None,
    only_high_conf_tf: bool = True,
    verbose: bool = False,
) -> ScenicGrn:
    """Motif support for the TF to gene links.

    Runs `run_cistarget` on each TF's targets and flags a link
    `in_leading_edge` when a motif annotated to that TF (high confidence, or
    also low with `only_high_conf_tf = False`) has the gene in its leading
    edge.

    Args:
        x: The result with TF to gene links.
        rankings: `(n_genes, n_motifs)` integer motif ranks.
        rank_genes: The genes of the rows.
        rank_motifs: The motifs of the columns.
        annot_data: `motif`, `TF` and `annotationSource`.
        cis_target_params: The parameters. `None` uses the defaults.
        gene_id_to_symbol: Gene id to the symbol the database uses; links
            without a symbol are left out of the enrichment.
        only_high_conf_tf: Count only high confidence annotations.
        verbose: Controls verbosity.

    Returns:
        A copy with `cis_targets_results` set and the links gaining
        `in_leading_edge`, sorted by TF.

    Examples:
        >>> grn = bv.pathways.tf_to_genes_motif_enrichment(
        ...     grn, rankings, rank_genes, rank_motifs, annot
        ... )
        >>> motif_hits = bv.pathways.get_cistarget_res(grn)
    """
    links = get_tf_to_gene(x)
    symbol_to_id = None
    if gene_id_to_symbol is not None:
        sym = dict(gene_id_to_symbol)
        mapped = links.with_columns(
            tf_symbol=pl.col("tf").replace_strict(sym, default=None),
            gene_symbol=pl.col("gene").replace_strict(sym, default=None),
        )
        n_unmapped = (
            mapped.height - mapped.drop_nulls(["tf_symbol", "gene_symbol"]).height
        )
        if n_unmapped and verbose:
            logger.info(
                "  Dropping %i TF-gene pairs with no symbol mapping.", n_unmapped
            )
        mapped = mapped.drop_nulls(["tf_symbol", "gene_symbol"])
        tf_col, gene_col = "tf_symbol", "gene_symbol"
        symbol_to_id = {v: k for k, v in sym.items()}
    else:
        mapped, tf_col, gene_col = links, "tf", "gene"
    gs = {
        tf: g[gene_col].to_list()
        for (tf,), g in mapped.sort(tf_col).group_by(tf_col, maintain_order=True)
    }
    if verbose:
        logger.info(
            "Running CisTarget to associate TF to gene pairs with motif "
            "enrichment info."
        )
    cis = run_cistarget(
        gs,
        rankings,
        rank_genes,
        rank_motifs,
        annot_data,
        cis_target_params=cis_target_params,
        verbose=verbose,
    )
    le = _leading_edge_links(cis, "TF_highConf", symbol_to_id)
    if not only_high_conf_tf:
        le = pl.concat([le, _leading_edge_links(cis, "TF_lowConf", symbol_to_id)])
    hits = le.unique().with_columns(in_leading_edge=pl.lit(True))
    links = (
        links.join(hits, left_on=["tf", "gene"], right_on=["tf", "le_gene"], how="left")
        .with_columns(pl.col("in_leading_edge").fill_null(False))
        .sort("tf", maintain_order=True)
    )
    return dataclasses.replace(x, cis_targets_results=cis, tf_to_gene_results=links)


############
# Regulons #
############


@beartype
def build_regulons(
    x: ScenicGrn,
    *,
    use_leading_edge: bool = True,
    add_tf: bool = True,
    min_genes: int = 10,
    mode: Mode = "activating",
    verbose: bool = False,
) -> dict[str, list[str]] | None:
    """Regulons from the TF to gene links.

    Args:
        x: The result with TF to gene links.
        use_leading_edge: Keep only links with motif support; skipped with a
            warning before `tf_to_genes_motif_enrichment`.
        add_tf: Put the TF itself first in its regulon.
        min_genes: Minimum regulon size.
        mode: Which correlation signs to keep, if the links are signed. With
            `"both"` the regulons are named `<tf>_pos` and `<tf>_neg`.
        verbose: Controls verbosity.

    Returns:
        Regulon name (sorted) to gene ids, or `None` with a warning when
        nothing survives.

    Raises:
        ValueError: If `min_genes` is below 1.

    Examples:
        >>> regulons = bv.pathways.build_regulons(grn, min_genes=3)
        >>> regulons = bv.pathways.build_regulons(
        ...     grn, use_leading_edge=False, min_genes=5
        ... )
    """
    if min_genes < 1:
        raise ValueError("min_genes must be at least 1.")
    links = get_tf_to_gene(x)
    if links.height == 0:
        warnings.warn(
            "No TF to gene pairs found. Returning None. Did you run "
            "identify_tf_to_genes()?",
            stacklevel=2,
        )
        return None
    if use_leading_edge:
        if "in_leading_edge" not in links.columns:
            warnings.warn(
                "No `in_leading_edge` column found, so the CisTarget filter is "
                "skipped. Did you run tf_to_genes_motif_enrichment()?",
                stacklevel=2,
            )
        else:
            links = links.filter(pl.col("in_leading_edge"))
    signed = "cor_sign" in links.columns
    if signed:
        links = links.filter(pl.col("cor_sign").is_in(list(MODE_SIGNS[mode])))
    elif mode != "activating":
        warnings.warn(
            "No `cor_sign` column found, so `mode` is ignored. Did you run "
            "tf_to_genes_correlations()?",
            stacklevel=2,
        )
    if links.height == 0:
        warnings.warn(
            "No TF to gene pairs survived the filters. Returning None.", stacklevel=2
        )
        return None
    if mode == "both" and signed:
        key = pl.col("tf") + pl.when(pl.col("cor_sign") > 0).then(
            pl.lit("_pos")
        ).otherwise(pl.lit("_neg"))
    else:
        key = pl.col("tf")
    links = links.with_columns(key=key)
    regulons: dict[str, list[str]] = {}
    for k in sorted(links["key"].unique().to_list()):
        sub = links.filter(pl.col("key") == k)
        genes = sub["gene"].to_list()
        if add_tf:
            genes = [sub["tf"][0], *genes]
        regulons[k] = list(dict.fromkeys(genes))
    n_before = len(regulons)
    regulons = {k: v for k, v in regulons.items() if len(v) >= min_genes}
    if not regulons:
        warnings.warn(
            "No regulons left after the size filter. Returning None.", stacklevel=2
        )
        return None
    if verbose:
        logger.info(
            "Built %i regulons (%i dropped below %i genes). Median size: %.0f",
            len(regulons),
            n_before - len(regulons),
            min_genes,
            float(np.median([len(v) for v in regulons.values()])),
        )
    return regulons


@beartype
def binarise_regulon_activity(
    auc: ScMatrixRes,
    *,
    binarise_params: ScenicBinariseParams | None = None,
    verbose: bool = False,
) -> RegulonBinarisation:
    """On/off calls per regulon.

    Per regulon a two-component Gaussian mixture is compared against one
    Gaussian by BIC; if the mixture wins the threshold is the density trough
    between the means, else `mean + 2 * sd`. A cell is on strictly above it.

    Args:
        auc: The AUCell scores of the regulons, at least two cells.
        binarise_params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The `RegulonBinarisation`.

    Raises:
        ValueError: With fewer than two cells.

    References:
        Aibar, et al., Nat. Methods, 2017.

    Examples:
        >>> auc = bv.pathways.aucell(sc, regulons)
        >>> on_off = bv.pathways.binarise_regulon_activity(auc)
    """
    if auc.values.shape[0] < 2 or auc.values.shape[1] < 1:
        raise ValueError("auc needs at least two cells and one regulon.")
    params = binarise_params or ScenicBinariseParams()
    values = np.ascontiguousarray(auc.values, dtype=np.float64)
    res = _bixverse.regulon_thresholds(values, to_rust(params))
    binary = values > res["thresholds"][None, :]
    if verbose:
        logger.info(
            "Binarised %i regulons, %i of which were bimodal.",
            values.shape[1],
            sum(res["bimodal"]),
        )
    return RegulonBinarisation(
        binary=pl.DataFrame(
            {"cell_id": auc.cell_ids, **dict(zip(auc.columns, binary.T, strict=True))}
        ),
        thresholds=pl.DataFrame(
            {
                "regulon": list(auc.columns),
                "threshold": res["thresholds"],
                "bimodal": res["bimodal"],
                "n_cells_on": binary.sum(0).astype(np.int64),
            }
        ),
    )
