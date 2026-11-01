"""Differential expression, markers and differential abundance.

Wilcoxon markers, pseudobulk edgeR, NEBULA, Milo neighbourhoods and MELD.
The work runs in Rust.

Marker tests return polars frames; the specific markers, NEBULA, Milo and
MELD return frozen records. Nothing is written to obs.

    >>> markers = find_all_markers(sc, "leiden")
    >>> milo = get_milor_abundances(sc, "sample_id")
    >>> milo = test_nhoods(milo, "~ condition", design_df)

Designs are formulas over a frame, `"~ condition + age"`: additive main
effects, an intercept first, numeric columns as they are
and every other column treatment coded against its first (sorted) level,
named `<column><level>`. Interactions and transformations are not parsed.

Every index is 0-based, Milo's neighbourhoods (`Nhood`) and coefficients
(`coef`) included.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np
import polars as pl
from beartype import BeartypeConf, beartype

from bixverse import _bixverse
from bixverse._core.doublets import _as_character
from bixverse._core.graph import _embedding, _first_dims
from bixverse._core.integrate import _hard_state
from bixverse._core.pp import _factor_codes, _obs_aligned
from bixverse._core.repr import compact_repr
from bixverse._core.single_cells import Selection, SingleCells
from bixverse._core.state import CsrMatrix
from bixverse._core.subset import ScOrScSubset
from bixverse.params import EdgerQlParams, MeldParams, NebulaParams, to_rust
from bixverse.params._checks import check_range

###########
# Globals #
###########

# NEBULA's own "this likely failed" marker: a convergence code at or below.
NEBULA_FAILED_CONVERGENCE: int = -20

# Machine epsilon, the floor on a NEBULA offset.
DOUBLE_EPS: float = float(np.finfo(np.float64).eps)

INTERCEPT: str = "(Intercept)"

Alternative = Literal["twosided", "greater", "less"]
Method = Literal["wilcox"]
Coef = int | str | Sequence[int] | Sequence[str]
MiloNorm = Literal["TMM", "TMMwsp", "RLE", "upperquartile", "logMS"]
FdrWeighting = Literal["k-distance", "graph-overlap", "none"]

_CONF = BeartypeConf(is_pep484_tower=True)

logger = logging.getLogger(__name__)


##########
# Params #
##########


@beartype(conf=_CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class MiloRParams:
    """Parameters for the Milo neighbourhoods.

    It carries the kNN keys the neighbourhood refinement reads.

    Attributes:
        prop: Share of cells sampled as neighbourhood indices.
        k_refine: Neighbours whose median position an index is moved towards.
        refinement_strategy: `"index"` queries a kNN index for the cell nearest
            the median, `"approximate"` searches the `k_refine` neighbours,
            `"bruteforce"` every cell.
        index_type: The kNN index `"index"` builds. `"exhaustive"` returns the
            true nearest cell.
        ann_dist: Distance metric of the refinement.
        n_trees: Annoy: number of trees.
        search_budget: Annoy: search budget per query.
        m: HNSW: connections per layer.
        ef_construction: HNSW: construction budget.
        ef_search: HNSW: search budget.
        delta: NNDescent: early termination criterium.
        diversify_prob: NNDescent: diversification probability.
    """

    prop: float = 0.2
    k_refine: int = 20
    refinement_strategy: Literal["index", "approximate", "bruteforce"] = "index"
    index_type: Literal["nndescent", "hnsw", "annoy", "exhaustive"] = "nndescent"
    ann_dist: Literal["euclidean", "cosine"] = "euclidean"
    n_trees: int = 50
    search_budget: int | None = None
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 100
    delta: float = 0.001
    diversify_prob: float = 0.0

    def __post_init__(self) -> None:
        check_range("prop", self.prop, 0, 1, True, True)
        check_range("k_refine", self.k_refine, 1, None, False, True)
        check_range("n_trees", self.n_trees, 1, None, False, True)
        check_range("search_budget", self.search_budget, 1, None, False, True)
        check_range("m", self.m, 1, None, False, True)
        check_range("ef_construction", self.ef_construction, 1, None, False, True)
        check_range("ef_search", self.ef_search, 1, None, False, True)
        check_range("delta", self.delta, 0, 1, False, False)
        check_range("diversify_prob", self.diversify_prob, 0, 1, False, False)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class ScSpecificMarkers:
    """One-vs-many specific markers.

    Attributes:
        summary: Per reference group and gene, the summaries across every
            rival: `prop_ref`, `median_auroc`, `min_auroc`, `mean_auroc`,
            `max_auroc`, `worst_rival`, `min_rank`, `simes_p`, `simes_fdr`,
            `max_p` and `max_p_fdr`.
        per_comparison: Per reference, rival and gene, `auroc`, `lfc`,
            `prop_ref`, `prop_rival`, `z_scores`, `p_values` and `fdr`.
        params: The parameters of the run.
    """

    summary: pl.DataFrame
    per_comparison: pl.DataFrame
    params: dict[str, object]

    def __repr__(self) -> str:
        """Summary of the reference groups and their markers."""
        refs = self.summary["ref_grp"].unique(maintain_order=True).to_list()
        return compact_repr(
            "ScSpecificMarkers",
            n_references=len(refs),
            n_genes=self.summary["gene_id"].n_unique(),
            column=self.params["column_of_interest"],
            references=refs,
            alternative=self.params["alternative"],
            min_prop=self.params["min_prop"],
        )


@beartype
@dataclass(frozen=True, repr=False)
class ScNebula:
    """NEBULA fits.

    Attributes:
        results: One row per gene that survived NEBULA's expression filter:
            `gene_id`, `log_fc` (natural log), `effect_se`, `z`, `p_value`,
            `fdr`, `subject_overdispersion`, `cell_overdispersion`,
            `convergence`, `sigma_at_bound` and, with shrinkage,
            `cell_overdispersion_shrunk`.
        coefficients: `gene_id` and one column per design column, the fixed
            effects.
        se: The matching standard errors.
        params: The parameters of the run.
    """

    results: pl.DataFrame
    coefficients: pl.DataFrame
    se: pl.DataFrame
    params: dict[str, object]

    def __repr__(self) -> str:
        """Summary of the fit and its convergence."""
        res, p = self.results, self.params
        # the CPU or the GPU bundle
        method = to_rust(p["nebula_params"])["nebula_method"]
        failed = int((res["convergence"] <= NEBULA_FAILED_CONVERGENCE).sum())
        collapsed = int(res["sigma_at_bound"].sum())
        return compact_repr(
            "ScNebula",
            n_genes=res.height,
            n_coefficients=self.coefficients.width - 1,
            method=method,
            subject_col=p["subject_col"],
            tested=p["tested"],
            n_subjects=p["n_subjects"],
            n_cells=p["n_cells"],
            n_failed=failed,
            n_collapsed=collapsed,
        )


@beartype
@dataclass(frozen=True, repr=False)
class MiloR:
    """Milo neighbourhoods and their test.

    Attributes:
        nhoods: Membership, `(n_cells, n_nhoods)`; rows are the kept cells in
            `cells_to_keep` order.
        sample_counts: `(n_nhoods, n_samples)` cells per neighbourhood and
            sample.
        samples: The sample names, the columns of `sample_counts`.
        spatial_dist: Squared distance of each index cell to its last kNN
            neighbour, the `"k-distance"` weighting.
        nhood_overlap: Cells each neighbourhood shares with all the others,
            the `"graph-overlap"` weighting.
        params: The parameters, plus `used_emb`, `no_embd_to_use` and
            `index_cell`, the 0-based positions among the kept cells.
        nhoods_info: The differential abundance table once `test_nhoods` ran.
    """

    nhoods: CsrMatrix
    sample_counts: np.ndarray
    samples: tuple[str, ...]
    spatial_dist: np.ndarray
    nhood_overlap: np.ndarray
    params: dict[str, object]
    nhoods_info: pl.DataFrame | None = None

    def __repr__(self) -> str:
        return compact_repr(
            "MiloR",
            n_cells=self.nhoods.shape[0],
            n_nhoods=self.nhoods.shape[1],
            n_samples=len(self.samples),
            tested=self.nhoods_info is not None,
        )


@beartype
@dataclass(frozen=True, repr=False)
class MeldRes:
    """MELD densities.

    Attributes:
        raw_scores: `cell_id` and one column per sample, the filtered
            indicators.
        norm_scores: The same with negatives clamped to 0 and every row L1
            normalised, so probability-like.
    """

    raw_scores: pl.DataFrame
    norm_scores: pl.DataFrame

    def __repr__(self) -> str:
        return compact_repr(
            "MeldRes",
            n_cells=self.raw_scores.height,
            n_samples=self.raw_scores.width - 1,
        )


@beartype
@dataclass(frozen=True, repr=False)
class PseudoBulked:
    """Pseudobulked counts.

    Attributes:
        counts: `(n_samples, n_genes)`, a dense `float64` array or a CSR,
            whose values are `float32` as every `CsrMatrix`'s.
        samples: The pseudobulk sample names, the rows.
        gene_ids: The gene ids, the columns.
    """

    counts: np.ndarray | CsrMatrix
    samples: tuple[str, ...]
    gene_ids: np.ndarray

    def __repr__(self) -> str:
        return compact_repr(
            "PseudoBulked",
            n_samples=len(self.samples),
            n_genes=len(self.gene_ids),
            sparse=isinstance(self.counts, CsrMatrix),
        )


###########
# Helpers #
###########


@beartype(conf=_CONF)
def _require_columns(sc: ScOrScSubset, cols: Sequence[str]) -> None:
    """Check that every column is in obs.

    Raises:
        ValueError: If one is not in obs.
    """
    present = sc.obs_columns()
    missing = [c for c in cols if c not in present]
    if missing:
        raise ValueError(f"Not found in the obs table: {', '.join(missing)}.")


@beartype(conf=_CONF)
def _formula_terms(formula: str) -> list[str]:
    """The main effects of an additive formula, `"~ a + b"`.

    Raises:
        ValueError: On anything but `~` and `+`-separated column names.
    """
    lhs, sep, rhs = formula.partition("~")
    if not sep or lhs.strip():
        raise ValueError(f"The design must be a one-sided formula, got {formula!r}.")
    terms = [t.strip() for t in rhs.split("+")]
    terms = [t for t in terms if t != "1"]
    bad = [t for t in terms if not re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.]*", t)]
    if bad or not terms:
        raise ValueError(
            f"Only additive main effects of obs columns are supported, got {formula!r}."
        )
    return terms


@beartype(conf=_CONF)
def _model_matrix(frame: pl.DataFrame, formula: str) -> tuple[np.ndarray, list[str]]:
    """The design matrix for additive main effects with an intercept.

    Numeric columns enter as they are; any other column is treatment coded
    against its first level (Enum order, else sorted), one `<col><level>`
    column per other level present.

    Returns:
        The `(n_rows, n_coef)` `float64` design and its column names.
    """
    cols = [np.ones(frame.height)]
    names = [INTERCEPT]
    for term in _formula_terms(formula):
        values = frame[term]
        if values.dtype.is_numeric():
            cols.append(values.cast(pl.Float64).to_numpy())
            names.append(term)
            continue
        if values.dtype == pl.Boolean:
            values = values.cast(pl.String).str.to_uppercase()
        codes, levels = _factor_codes(values, term)
        for k, level in enumerate(levels[1:], start=1):
            cols.append((codes == k).astype(np.float64))
            names.append(f"{term}{level}")
    return np.ascontiguousarray(np.column_stack(cols)), names


@beartype(conf=_CONF)
def _design_array(design: np.ndarray | pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """A numeric design as a contiguous `float64` matrix and its names.

    An array has no names, so coefficients can only be picked by position.
    """
    if isinstance(design, pl.DataFrame):
        return np.ascontiguousarray(design.to_numpy(), dtype=np.float64), design.columns
    if design.ndim != 2:
        raise ValueError("The design must be a 2-D matrix.")
    return np.ascontiguousarray(design, dtype=np.float64), []


@beartype(conf=_CONF)
def _coef_positions(coef: Coef, names: Sequence[str], n_coef: int) -> list[int]:
    """0-based design columns from positions or names.

    Raises:
        ValueError: On an unknown name or a position outside the design.
    """
    wanted = [coef] if isinstance(coef, int | str) else list(coef)
    out = []
    for c in wanted:
        if isinstance(c, str):
            if c not in names:
                raise ValueError(f"Coefficient {c!r} is not a design column.")
            out.append(list(names).index(c))
        elif not 0 <= c < n_coef:
            raise ValueError(f"Coefficient {c} is outside 0..{n_coef - 1}.")
        else:
            out.append(c)
    return out


@beartype(conf=_CONF)
def _tested(
    names: Sequence[str],
    n_coef: int,
    coef: Coef | None,
    contrast: np.ndarray | Sequence[float] | None,
) -> tuple[list[int] | None, np.ndarray | None]:
    """The tested `(coef, contrast)`, `coef` defaulting to the
    last design column.

    Raises:
        ValueError: If both are given, or the contrast does not have one row
            per design column.
    """
    if coef is not None and contrast is not None:
        raise ValueError("Supply either `coef` or `contrast`, not both.")
    if contrast is not None:
        c = np.asarray(contrast, dtype=np.float64)
        c = c.reshape(-1, 1) if c.ndim == 1 else c
        if c.shape[0] != n_coef or np.isnan(c).any():
            raise ValueError(
                f"The contrast has {c.shape[0]} rows against {n_coef} design columns."
            )
        return None, c
    return _coef_positions(n_coef - 1 if coef is None else coef, names, n_coef), None


@beartype(conf=_CONF)
def _groups_to_cells(
    obs: pl.DataFrame, col: str, groups: Sequence[object]
) -> list[np.ndarray]:
    """0-based cells per group value, ascending."""
    idx = obs["cell_idx"].to_numpy()
    values = obs[col]
    return [
        np.ascontiguousarray(idx[(values == g).to_numpy()], dtype=np.uint32)
        for g in groups
    ]


###########
# Markers #
###########


@beartype(conf=_CONF)
def find_markers(
    sc: ScOrScSubset,
    cells_1: Selection,
    cells_2: Selection,
    *,
    method: Method = "wilcox",
    alternative: Alternative = "twosided",
    min_prop: float = 0.05,
    verbose: bool = False,
) -> pl.DataFrame:
    """Wilcoxon DGE between two groups of cells.

    Args:
        sc: The experiment.
        cells_1: 0-based cell indices or ids of group 1.
        cells_2: 0-based cell indices or ids of group 2.
        method: Only `"wilcox"`, reserved.
        alternative: The test alternative.
        min_prop: A gene is tested if this share of either group expresses it.
        verbose: Controls verbosity.

    Returns:
        `gene_id`, `lfc`, `prop1`, `prop2`, `z_scores`, `p_values` and `fdr`,
        one row per tested gene in gene order.

    Examples:
        >>> obs = sc.obs(filtered=True)
        >>> ct1 = obs.filter(obs["cell_grp"] == "cell_type_1")["cell_id"].to_list()
        >>> ct2 = obs.filter(obs["cell_grp"] == "cell_type_2")["cell_id"].to_list()
        >>> res = bv.de.find_markers(sc, ct1, ct2, alternative="greater")
    """
    check_range("min_prop", min_prop, 0, 1, False, False)
    res = _bixverse.dge_mann_whitney(
        str(sc.cells_path),
        sc._resolve_cells(cells_1),
        sc._resolve_cells(cells_2),
        min_prop,
        alternative,
        verbose,
    )
    return pl.DataFrame(
        {
            "gene_id": sc.gene_ids()[res["genes_to_keep"]],
            **{k: res[k] for k in ("lfc", "prop1", "prop2", "z_scores")},
            "p_values": res["p_values"],
            "fdr": res["fdr"],
        }
    )


@beartype(conf=_CONF)
def find_all_markers(
    sc: ScOrScSubset,
    column_of_interest: str,
    *,
    method: Method = "wilcox",
    alternative: Alternative = "greater",
    min_prop: float = 0.05,
    verbose: bool = False,
) -> pl.DataFrame:
    """One-vs-rest Wilcoxon DGE for every group of a column.

    All groups are tested in one pass over the genes. Cells without a label
    are left out of every group and of every rest.

    Args:
        sc: The experiment.
        column_of_interest: The obs column holding the groups.
        method: Only `"wilcox"`, reserved.
        alternative: The test alternative; `"greater"` asks for genes up in
            the group.
        min_prop: A gene is tested for a group if this share of the group or
            of the rest expresses it.
        verbose: Controls verbosity.

    Returns:
        `grp`, `gene_id`, `lfc`, `prop1`, `prop2`, `z_scores`, `p_values` and
        `fdr`; groups in order of first appearance, genes in gene order.

    Examples:
        >>> markers = bv.de.find_all_markers(sc, "cell_grp")
        >>> top = markers.filter(markers["fdr"] < 0.05)
    """
    check_range("min_prop", min_prop, 0, 1, False, False)
    _require_columns(sc, [column_of_interest])
    obs = sc.obs(cols=["cell_idx", column_of_interest], filtered=True).drop_nulls(
        column_of_interest
    )
    groups = obs[column_of_interest].unique(maintain_order=True)
    arms = _bixverse.dge_one_vs_rest(
        str(sc.genes_path),
        _groups_to_cells(obs, column_of_interest, groups.to_list()),
        min_prop,
        alternative,
        verbose,
    )
    gene_ids = sc.gene_ids()
    frames = [
        pl.DataFrame(
            {
                "grp": groups.gather([g] * int(arm["genes_to_keep"].sum())),
                "gene_id": gene_ids[arm["genes_to_keep"]],
                **{
                    k: arm[k]
                    for k in ("lfc", "prop1", "prop2", "z_scores", "p_values", "fdr")
                },
            }
        )
        for g, arm in enumerate(arms)
    ]
    return pl.concat(frames)


@beartype(conf=_CONF)
def find_specific_markers(
    sc: ScOrScSubset,
    column_of_interest: str,
    *,
    reference_group: str | None = None,
    method: Method = "wilcox",
    alternative: Alternative = "greater",
    min_prop: float = 0.05,
    verbose: bool = False,
) -> ScSpecificMarkers | None:
    """Markers that hold up against every rival group separately.

    Each reference is tested against each other group of the column on its
    own, then summarised per gene across the rivals: a gene marks the
    reference only if `min_auroc` stays high against all of them. Every arm
    comes out of one pass over the genes, and the genes are filtered once,
    globally, so every comparison's FDR covers the same gene set.

    Args:
        sc: The experiment.
        column_of_interest: The obs column holding the groups.
        reference_group: The one reference to report. `None` runs every group
            in turn.
        method: Only `"wilcox"`, reserved.
        alternative: The test alternative; `"greater"` asks for genes up in
            the reference.
        min_prop: A gene is tested if this share of any group expresses it.
        verbose: Controls verbosity.

    Returns:
        The `ScSpecificMarkers`, or `None` with a warning if no gene passed
        `min_prop`.

    Raises:
        ValueError: If the column is missing, has fewer than two groups, or
            `reference_group` is not one of them.

    References:
        Soneson and Robinson, Nat Methods, 2018; Lun, et al., F1000Research,
        2016.

    Examples:
        >>> res = bv.de.find_specific_markers(sc, "cell_grp")
        >>> summary = bv.de.get_marker_summary(res)
        >>> ranked = summary.sort("min_auroc", descending=True)
    """
    check_range("min_prop", min_prop, 0, 1, False, False)
    _require_columns(sc, [column_of_interest])
    obs = sc.obs(cols=["cell_idx", column_of_interest], filtered=True)
    labels = pl.Series(_as_character(obs[column_of_interest]), dtype=pl.String)
    n_missing = labels.null_count()
    if n_missing:
        warnings.warn(
            f"Dropping {n_missing} cell(s) with no '{column_of_interest}' annotation.",
            stacklevel=2,
        )
    obs = pl.DataFrame({"cell_idx": obs["cell_idx"], "grp": labels}).drop_nulls("grp")
    grp_names = sorted(obs["grp"].unique().to_list())
    if len(grp_names) < 2:
        raise ValueError(
            f"Column '{column_of_interest}' has fewer than two groups. Nothing to "
            "compare against."
        )
    if len(grp_names) == 2:
        warnings.warn(
            "Only two groups found. With a single rival this collapses to the "
            "pairwise case, see find_markers().",
            stacklevel=2,
        )
    if reference_group is not None and reference_group not in grp_names:
        raise ValueError(
            f"reference_group {reference_group!r} is not one of: {', '.join(grp_names)}"
        )
    refs = grp_names if reference_group is None else [reference_group]
    ref_idx = [grp_names.index(r) for r in refs]
    arms = _bixverse.dge_one_vs_many(
        str(sc.genes_path),
        _groups_to_cells(obs, "grp", grp_names),
        np.asarray(ref_idx, dtype=np.uint32),
        min_prop,
        alternative,
        verbose,
    )
    keep = arms[0]["genes_to_keep"]
    if not keep.any():
        warnings.warn(
            "No gene passed the proportion filter for any group. Consider "
            "lowering min_prop. Returning None.",
            stacklevel=2,
        )
        return None

    genes = sc.gene_ids()[keep]
    names = np.asarray(grp_names, dtype=object)
    summaries, comparisons = [], []
    for r, arm in zip(ref_idx, arms, strict=True):
        rivals = np.asarray([k for k in range(len(grp_names)) if k != r])
        summaries.append(
            pl.DataFrame(
                {
                    "ref_grp": [grp_names[r]] * len(genes),
                    "gene_id": genes,
                    **{
                        k: arm[k]
                        for k in (
                            "prop_ref",
                            "median_auroc",
                            "min_auroc",
                            "mean_auroc",
                            "max_auroc",
                        )
                    },
                    "worst_rival": names[rivals[arm["worst_comparison"]]].astype(str),
                    "min_rank": arm["min_rank"].astype(np.int32),
                    **{
                        k: arm[k]
                        for k in ("simes_p", "simes_fdr", "max_p", "max_p_fdr")
                    },
                }
            )
        )
        n_rivals = len(rivals)
        comparisons.append(
            pl.DataFrame(
                {
                    "ref_grp": [grp_names[r]] * (n_rivals * len(genes)),
                    "rival_grp": np.repeat(names[rivals], len(genes)).astype(str),
                    "gene_id": np.tile(genes, n_rivals),
                    "auroc": arm["auroc"].ravel(),
                    "lfc": arm["lfc"].ravel(),
                    "prop_ref": np.tile(arm["prop_ref"], n_rivals),
                    "prop_rival": arm["prop_other"].ravel(),
                    "z_scores": arm["z_scores"].ravel(),
                    "p_values": arm["p_values"].ravel(),
                    "fdr": arm["fdr"].ravel(),
                }
            )
        )
    return ScSpecificMarkers(
        summary=pl.concat(summaries),
        per_comparison=pl.concat(comparisons),
        params={
            "method": method,
            "alternative": alternative,
            "min_prop": min_prop,
            "column_of_interest": column_of_interest,
            "reference_group": reference_group,
        },
    )


@beartype(conf=_CONF)
def get_marker_summary(x: ScSpecificMarkers) -> pl.DataFrame:
    """The per-gene summaries across every rival.

    Args:
        x: The marker result.

    Returns:
        A copy of the summary table.
    """
    return x.summary.clone()


@beartype(conf=_CONF)
def get_marker_comparisons(x: ScSpecificMarkers) -> pl.DataFrame:
    """The per-rival statistics.

    Args:
        x: The marker result.

    Returns:
        A copy of the per-comparison table.
    """
    return x.per_comparison.clone()


##############
# Pseudobulk #
##############


@beartype(conf=_CONF)
def get_pseudobulked(
    sc: ScOrScSubset,
    cell_list: dict[str, Selection],
    *,
    return_format: Literal["dense", "sparse"] = "dense",
    assay: Literal["raw", "norm"] = "raw",
    verbose: bool = False,
) -> PseudoBulked:
    """Pseudobulk groups of cells.

    Args:
        sc: The experiment.
        cell_list: Pseudobulk sample name to its 0-based cell indices or ids.
        return_format: A dense array or a CSR matrix.
        assay: `"raw"` sums the raw counts, `"norm"` averages the normalised
            counts over the group, zeros included.
        verbose: Controls verbosity.

    Returns:
        The `PseudoBulked`, samples in `cell_list` order.

    Raises:
        ValueError: If a group matches no cell.

    Examples:
        >>> obs = sc.obs(filtered=True)
        >>> cell_list = {
        ...     s: obs.filter(obs["sample_id"] == s)["cell_id"].to_list()
        ...     for s in obs["sample_id"].unique().sort()
        ... }
        >>> pb = bv.de.get_pseudobulked(sc, cell_list, return_format="sparse")
    """
    groups = [sc._resolve_cells(cells) for cells in cell_list.values()]
    if any(len(g) == 0 for g in groups):
        raise ValueError(
            "Some of the elements in your cell_list could not be matched to any "
            "cell index. Please check your inputs!"
        )
    res = _bixverse.pseudobulk(
        str(sc.cells_path), groups, assay, return_format == "sparse", verbose
    )
    if return_format == "sparse":
        data, indices, indptr = res
        counts: np.ndarray | CsrMatrix = CsrMatrix(
            data.astype(np.float32), indices, indptr, (len(groups), sc.n_genes)
        )
    else:
        counts = res
    return PseudoBulked(counts=counts, samples=tuple(cell_list), gene_ids=sc.gene_ids())


@beartype(conf=_CONF)
def run_edger_ql(
    counts: np.ndarray,
    design: np.ndarray | pl.DataFrame,
    feature_ids: Sequence[str] | np.ndarray,
    *,
    coef: Coef | None = None,
    contrast: np.ndarray | Sequence[float] | None = None,
    edger_params: EdgerQlParams | None = None,
) -> pl.DataFrame:
    """edgeR's quasi-likelihood chain.

    `filterByExpr`, `calcNormFactors`, `glmQLFit` and `glmQLFTest`, via
    `edge-rs`. The rows do not have to be genes: Milo tests neighbourhoods
    through here with `filter=False`.

    Args:
        counts: `(n_features, n_samples)` raw counts.
        design: `(n_samples, n_coef)` design with an intercept, at least two
            columns. A frame's column names can be used in `coef`.
        feature_ids: One id per row of `counts`.
        coef: 0-based design column(s) or names to drop from the null model.
            `None` tests the last column.
        contrast: Weights over the design columns, a vector or an
            `(n_coef, n_contrasts)` matrix. Exclusive with `coef`.
        edger_params: The parameters. `None` uses the defaults.

    Returns:
        `feature_id`, `log_fc` (log2), `log_cpm`, `f_stat`, `p_value` and
        `fdr`, one row per feature that survived the filters.

    Raises:
        ValueError: On a design with fewer than two columns or a row count
            that does not match, negative counts, or both `coef` and
            `contrast`.

    References:
        Chen, Lun and Smyth, F1000Research, 2016.

    Examples:
        >>> pb = bv.de.get_pseudobulked(sc, cell_list)
        >>> res = bv.de.run_edger_ql(pb.counts.T, design, pb.gene_ids, coef="stim")
    """
    params = edger_params or EdgerQlParams()
    mm, names = _design_array(design)
    counts = np.ascontiguousarray(counts, dtype=np.float64)
    if mm.shape[1] < 2:
        raise ValueError(
            "The design needs at least two columns: the null model has to retain one."
        )
    if mm.shape[0] != counts.shape[1]:
        raise ValueError(
            f"The design has {mm.shape[0]} rows against {counts.shape[1]} samples "
            "in the counts."
        )
    if len(feature_ids) != counts.shape[0]:
        raise ValueError("feature_ids needs one id per row of the counts.")
    if (counts < 0).any():
        raise ValueError("`counts` holds negative values. This wants raw counts.")
    coefs, c = _tested(names, mm.shape[1], coef, contrast)
    res = _bixverse.edger_ql(
        counts,
        mm,
        to_rust(params),
        coefs,
        None if c is None else c.ravel(order="F").tolist(),
        1 if c is None else c.shape[1],
    )
    return pl.DataFrame(
        {
            "feature_id": np.asarray(feature_ids, dtype=object)[
                res["features_to_keep"]
            ].astype(str),
            "log_fc": res["log_fc"],
            "log_cpm": res["log_cpm"],
            "f_stat": res["f_stat"],
            "p_value": res["p_values"],
            "fdr": res["fdr"],
        }
    )


@beartype(conf=_CONF)
def pseudobulk_dge(
    sc: ScOrScSubset,
    cell_list: dict[str, Selection],
    design: np.ndarray | pl.DataFrame,
    *,
    coef: Coef | None = None,
    contrast: np.ndarray | Sequence[float] | None = None,
    edger_params: EdgerQlParams | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """edgeR quasi-likelihood DGE on pseudobulked cells.

    Sums the raw counts per sample and runs `run_edger_ql` on the result.

    Args:
        sc: The experiment.
        cell_list: Pseudobulk sample name to its 0-based cell indices or ids.
            The rows of `design` follow this order.
        design: `(n_samples, n_coef)` design with an intercept.
        coef: See `run_edger_ql`.
        contrast: See `run_edger_ql`.
        edger_params: See `run_edger_ql`.
        verbose: Controls verbosity.

    Returns:
        See `run_edger_ql`; `feature_id` holds the gene ids.

    Raises:
        ValueError: If the design rows do not match the samples.

    References:
        Squair, et al., Nat Commun, 2021; Chen, Lun and Smyth, F1000Research,
        2016.

    Examples:
        >>> res = bv.de.pseudobulk_dge(sc, cell_list, design, coef="stim")
        >>> sig = res.filter(res["fdr"] < 0.05)
    """
    n_rows = design.height if isinstance(design, pl.DataFrame) else design.shape[0]
    if n_rows != len(cell_list):
        raise ValueError(
            f"The design has {n_rows} rows against {len(cell_list)} pseudo-bulk "
            "samples."
        )
    agg = get_pseudobulked(sc, cell_list, assay="raw", verbose=verbose)
    assert isinstance(agg.counts, np.ndarray)
    return run_edger_ql(
        agg.counts.T,
        design,
        agg.gene_ids,
        coef=coef,
        contrast=contrast,
        edger_params=edger_params,
    )


##########
# NEBULA #
##########


@beartype(conf=_CONF)
def nebula(
    sc: ScOrScSubset,
    subject_col: str,
    design: str,
    *,
    coef: int | str | None = None,
    contrast: np.ndarray | Sequence[float] | None = None,
    genes_to_use: Selection | None = None,
    cells_to_use: Sequence[str] | None = None,
    offset: np.ndarray | Sequence[float] | None = None,
    nebula_params: NebulaParams | None = None,
    verbose: bool = False,
) -> ScNebula:
    """NEBULA's negative binomial gamma mixed model.

    The variance splits into a subject-level random effect and a cell-level
    overdispersion, so the donor structure is modelled rather than ignored.
    Genes stream through in batches, which is exact. A fit takes milliseconds
    to seconds per gene, so restrict `genes_to_use` to the HVGs or a
    candidate set.

    Args:
        sc: The experiment.
        subject_col: obs column with the subject (donor), the random effect.
        design: Formula over obs, e.g. `"~ condition + age"`.
        coef: The coefficient the Wald test reports, a 0-based design column
            or its name. `None` takes the last column.
        contrast: One weight per design column. Exclusive with `coef`.
        genes_to_use: 0-based gene indices or ids. `None` fits every gene.
        cells_to_use: Cell ids to fit. `None` takes the kept cells; ids that
            are not kept warn and drop.
        offset: Strictly positive scaling per cell, aligned to the cells that
            survive the design. `None` uses the library sizes.
        nebula_params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The `ScNebula`.

    Raises:
        ValueError: On missing columns, no cells left, both `coef` and
            `contrast`, or a bad offset.

    References:
        He, et al., Commun Biol, 2021.

    Examples:
        >>> bv.pp.find_hvg(sc, hvg_no=200)
        >>> hvgs = sc.gene_ids(sc.map.hvg)
        >>> res = bv.de.nebula(sc, "sample_id", "~ condition", genes_to_use=hvgs)
        >>> top = res.results.sort("p_value").head()
    """
    return _nebula(
        sc,
        subject_col,
        design,
        coef=coef,
        contrast=contrast,
        genes_to_use=genes_to_use,
        cells_to_use=cells_to_use,
        offset=offset,
        nebula_params=nebula_params,
        verbose=verbose,
        ext=_bixverse,
    )


@beartype(conf=_CONF)
def _nebula(
    sc: ScOrScSubset,
    subject_col: str,
    design: str,
    *,
    coef: int | str | None = None,
    contrast: np.ndarray | Sequence[float] | None = None,
    genes_to_use: Selection | None = None,
    cells_to_use: Sequence[str] | None = None,
    offset: np.ndarray | Sequence[float] | None = None,
    nebula_params: object = None,
    verbose: bool = False,
    ext: ModuleType,
) -> ScNebula:
    """The body of `nebula`, with the extension module `ext` supplied, so
    the GPU twin shares it. `nebula_params` is the CPU or the GPU bundle."""
    params = nebula_params or NebulaParams()
    terms = _formula_terms(design)
    needed = list(dict.fromkeys([*terms, subject_col]))
    _require_columns(sc, needed)
    obs = sc.obs(
        cols=list(dict.fromkeys(["cell_idx", "cell_id", *needed])), filtered=True
    )

    if cells_to_use is not None:
        keep = obs["cell_id"].is_in(list(cells_to_use))
        if not keep.any():
            raise ValueError("None of the provided ids could be matched.")
        n_missing = len(set(cells_to_use)) - int(keep.sum())
        if n_missing > 0:
            warnings.warn(
                f"{n_missing} of the provided ids could not be matched and are "
                "ignored.",
                stacklevel=2,
            )
        obs = obs.filter(keep)

    complete = obs.select(pl.all_horizontal(pl.col(needed).is_not_null())).to_series()
    if not complete.all():
        warnings.warn(
            f"Dropping {int((~complete).sum())} cell(s) with a missing design or "
            "subject value.",
            stacklevel=2,
        )
        obs = obs.filter(complete)
    if obs.height == 0:
        raise ValueError(
            "No cells left after dropping missing design or subject values."
        )

    mm, names = _model_matrix(obs, design)
    subject_codes, subjects = _factor_codes(
        pl.Series(_as_character(obs[subject_col]), dtype=pl.String), subject_col
    )
    if len(subjects) < 2:
        warnings.warn(
            "Only one subject found. The subject-level variance has nothing to "
            "estimate from and NEBULA collapses to a plain negative binomial.",
            stacklevel=2,
        )
    if coef is not None and contrast is not None:
        raise ValueError("Supply either `coef` or `contrast`, not both.")
    tested: dict[str, object]
    if contrast is not None:
        c = np.asarray(contrast, dtype=np.float64)
        if c.shape != (mm.shape[1],) or np.isnan(c).any():
            raise ValueError(f"The contrast needs {mm.shape[1]} finite weights.")
        tested = {"contrast": c.tolist()}
        label = "contrast"
    else:
        pos = _coef_positions(
            mm.shape[1] - 1 if coef is None else coef, names, mm.shape[1]
        )[0]
        tested = {"coef": pos}
        label = names[pos]

    cells = np.ascontiguousarray(obs["cell_idx"].to_numpy(), dtype=np.uint32)
    off = None
    if offset is not None:
        off = np.ascontiguousarray(offset, dtype=np.float64)
        if off.shape != (len(cells),) or not (off >= DOUBLE_EPS).all():
            raise ValueError(
                f"offset needs {len(cells)} values, each at least {DOUBLE_EPS}."
            )
    subject_ids = np.zeros(sc.dims[0], dtype=np.uint32)
    subject_ids[cells] = subject_codes
    genes = (
        np.arange(sc.n_genes, dtype=np.uint32)
        if genes_to_use is None
        else sc._resolve_genes(genes_to_use)
    )

    res = ext.nebula(
        str(sc.genes_path),
        str(sc.cells_path),
        cells,
        genes,
        subject_ids,
        mm,
        {**to_rust(params), **tested},
        off,
        verbose,
    )
    gene_ids = sc.gene_ids(res["gene_idx"])
    results = pl.DataFrame(
        {
            "gene_id": gene_ids,
            "log_fc": res["log_fc"],
            "effect_se": res["effect_se"],
            "z": res["z"],
            "p_value": res["p_values"],
            "fdr": res["fdr"],
            "subject_overdispersion": res["subject_overdispersion"],
            "cell_overdispersion": res["cell_overdispersion"],
            "convergence": res["convergence"],
            "sigma_at_bound": res["sigma_at_bound"],
        }
    )
    if res["cell_overdispersion_shrunk"] is not None:
        results = results.with_columns(
            cell_overdispersion_shrunk=pl.Series(res["cell_overdispersion_shrunk"])
        )

    def by_gene(mat: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {"gene_id": gene_ids, **{n: mat[:, j] for j, n in enumerate(names)}}
        )

    return ScNebula(
        results=results,
        coefficients=by_gene(res["coefficients"]),
        se=by_gene(res["se"]),
        params={
            "subject_col": subject_col,
            "design": design,
            "tested": label,
            "n_cells": len(cells),
            "n_subjects": len(subjects),
            "n_genes_requested": len(genes),
            "nebula_params": params,
        },
    )


########
# Milo #
########


@beartype(conf=_CONF)
def _nhood_csr(i: np.ndarray, j: np.ndarray, n_cells: int, n_nhoods: int) -> CsrMatrix:
    """The membership COO as a cells x neighbourhoods CSR, entries sorted."""
    order = np.lexsort((j, i))
    indptr = np.zeros(n_cells + 1, dtype=np.uint64)
    np.cumsum(np.bincount(i, minlength=n_cells), out=indptr[1:])
    return CsrMatrix(
        np.ones(len(i), dtype=np.float32),
        np.ascontiguousarray(j[order], dtype=np.uint32),
        indptr,
        (n_cells, n_nhoods),
    )


@beartype(conf=_CONF)
def get_milor_abundances(
    sc: SingleCells,
    sample_id_col: str,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    milor_params: MiloRParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> MiloR | None:
    """Milo neighbourhoods and their cell counts per sample.

    Samples a share of the cells, moves each towards the median of its kNN
    neighbourhood, and counts the cells of every sample in the neighbourhood
    of each refined index. Test them with `test_nhoods`.

    Args:
        sc: The experiment, with an embedding and the kNN built on it.
        sample_id_col: obs column with the sample per cell.
        embd_to_use: The embedding the kNN was built on.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        milor_params: The parameters. `None` uses the defaults; a `k_refine`
            above the embedding width is lowered to it with a warning.
        seed: Seed.
        verbose: Controls verbosity.

    Returns:
        The `MiloR`, or `None` with a warning if there is no kNN.

    Raises:
        ValueError: If the embedding is missing, it or the kNN is stale, or a
            kept cell has no sample.

    References:
        Dann, et al., Nat Biotechnol, 2022.

    Examples:
        >>> params = bv.de.MiloRParams(prop=0.1)
        >>> milo = bv.de.get_milor_abundances(sc, "sample_id", milor_params=params)
    """
    params = milor_params or MiloRParams()
    _hard_state(sc, [embd_to_use, "knn"])
    obs = _obs_aligned(sc, [sample_id_col])
    if sample_id_col not in obs.columns:
        raise ValueError(f"Column '{sample_id_col}' not found in the obs table.")
    n_missing = obs[sample_id_col].null_count()
    if n_missing:
        raise ValueError(
            f"Column '{sample_id_col}' holds {n_missing} missing value(s). Every "
            "cell needs a sample."
        )
    codes, samples = _factor_codes(obs[sample_id_col], sample_id_col)

    embd = _embedding(sc, embd_to_use)
    if embd is None:
        raise ValueError(f"No embedding '{embd_to_use}' found.")
    embd = _first_dims(embd, no_embd_to_use)
    knn = sc.cache.knn
    if knn is None:
        warnings.warn(
            "No kNN data could be found on the object. Please run "
            "find_neighbours(). Returning None.",
            stacklevel=2,
        )
        return None
    if params.k_refine > embd.shape[1]:
        warnings.warn(
            "You chose a k refinement larger than the number of features in the "
            "embedding. Will change to the embedding width.",
            stacklevel=2,
        )
        params = dataclasses.replace(params, k_refine=embd.shape[1])

    res = _bixverse.milor_nhoods(
        embd, knn.indices, codes, len(samples), to_rust(params), seed, verbose
    )
    n_nhoods = len(res["index_cell"])
    return MiloR(
        nhoods=_nhood_csr(res["nhoods_i"], res["nhoods_j"], res["n_cells"], n_nhoods),
        sample_counts=res["sample_counts"],
        samples=tuple(samples),
        spatial_dist=res["kth_distances"],
        nhood_overlap=res["nhood_overlap"],
        params={
            **to_rust(params),
            "used_emb": embd_to_use,
            "no_embd_to_use": no_embd_to_use,
            "index_cell": res["index_cell"],
        },
    )


@beartype(conf=_CONF)
def test_nhoods(
    x: MiloR,
    design: str,
    design_df: pl.DataFrame,
    *,
    sample_col: str = "sample_id",
    coef: Coef | None = None,
    contrast: np.ndarray | Sequence[float] | None = None,
    norm_method: MiloNorm = "TMM",
    min_mean: float = 0.0,
    robust: bool = True,
    legacy: bool = True,
    fdr_weighting: FdrWeighting = "k-distance",
) -> MiloR:
    """Differential abundance of the neighbourhoods.

    edgeR's quasi-likelihood test on the neighbourhood counts, with Milo's
    spatial FDR on top. `filterByExpr` stays off, it means nothing for a
    neighbourhood; use `min_mean` to drop sparse ones.

    Args:
        x: The `MiloR`.
        design: Formula over `design_df`, e.g. `"~ condition"`.
        design_df: One row per sample.
        sample_col: The `design_df` column with the sample names.
        coef: See `run_edger_ql`.
        contrast: See `run_edger_ql`.
        norm_method: Library size normalisation; Milo's `"logMS"` leaves every
            factor at one.
        min_mean: Neighbourhoods with a lower mean count are dropped.
        robust: Robust estimation of the quasi-likelihood dispersion.
        legacy: edgeR's pre-4.0 pipeline, which is what Milo itself runs.
        fdr_weighting: Spatial FDR weights: the k-th neighbour distance, the
            overlap with other neighbourhoods, or none.

    Returns:
        A new `MiloR` with `nhoods_info`: `Nhood` (0-based), `logFC`,
        `logCPM`, `F`, `PValue`, `FDR` and `SpatialFDR`.

    Raises:
        ValueError: If the design rows and the samples do not match.

    References:
        Dann, et al., Nat Biotechnol, 2022; Chen, Lun and Smyth,
        F1000Research, 2016.

    Examples:
        >>> cols = sc.obs().select("sample_id", "condition")
        >>> design_df = cols.unique().sort("sample_id")
        >>> milo = bv.de.test_nhoods(milo, "~ condition", design_df)
    """
    check_range("min_mean", min_mean, 0, None, False, True)
    mm, names = _model_matrix(design_df, design)
    rows = _as_character(design_df[sample_col])
    if len(rows) != len(x.samples):
        raise ValueError(
            f"Design matrix ({len(rows)}) and sample counts ({len(x.samples)}) "
            "dimensions don't match."
        )
    if rows != list(x.samples):
        if not set(x.samples) <= set(rows):
            raise ValueError("Sample names in counts and design matrix don't match.")
        warnings.warn("Reordering design matrix to match sample counts.", stacklevel=2)
        mm = mm[[rows.index(s) for s in x.samples]]

    params = EdgerQlParams(
        norm_method="none" if norm_method == "logMS" else norm_method,
        filter=False,
        min_mean=min_mean,
        robust=robust,
        legacy=legacy,
    )
    n_nhoods = x.sample_counts.shape[0]
    res = run_edger_ql(
        x.sample_counts,
        pl.DataFrame(mm, schema=names, orient="row"),
        [str(i) for i in range(n_nhoods)],
        coef=coef,
        contrast=contrast,
        edger_params=params,
    )
    nhood = res["feature_id"].cast(pl.Int64)
    match fdr_weighting:
        case "k-distance":
            connectivity: np.ndarray | None = x.spatial_dist[nhood.to_numpy()]
        case "graph-overlap":
            connectivity = x.nhood_overlap[nhood.to_numpy()]
        case "none":
            connectivity = None
        case _:
            raise ValueError(f"Unknown fdr_weighting {fdr_weighting!r}.")
    p_values = res["p_value"].to_numpy()
    spatial = (
        pl.Series("SpatialFDR", [None] * res.height, dtype=pl.Float64)
        if connectivity is None
        else pl.Series(
            "SpatialFDR",
            _bixverse.spatial_fdr(
                np.ascontiguousarray(p_values, dtype=np.float64),
                np.ascontiguousarray(connectivity, dtype=np.float64),
            ),
        )
    )
    info = pl.DataFrame(
        {
            "Nhood": nhood,
            "logFC": res["log_fc"],
            "logCPM": res["log_cpm"],
            "F": res["f_stat"],
            "PValue": res["p_value"],
            "FDR": res["fdr"],
            "SpatialFDR": spatial,
        }
    )
    return dataclasses.replace(x, nhoods_info=info)


@beartype(conf=_CONF)
def add_nhoods_info(
    x: MiloR, cell_info: Sequence[str] | np.ndarray | pl.Series
) -> MiloR:
    """The majority cell type of each neighbourhood.

    Adds `majority_celltype` (ties go to the first label in sorted order) and
    `majority_prop` to `nhoods_info`.

    Args:
        x: A tested `MiloR`.
        cell_info: One label per row of `nhoods`, i.e. per kept cell.

    Returns:
        A new `MiloR`, or `x` with a warning if it was not tested yet.

    Raises:
        ValueError: If `cell_info` does not have one label per cell.

    Examples:
        >>> milo = bv.de.add_nhoods_info(milo, sc.obs(filtered=True)["cell_grp"])
    """
    if x.nhoods_info is None:
        warnings.warn(
            "No neighbourhood information found. Did you run test_nhoods() on the "
            "object? Returning object as is.",
            stacklevel=2,
        )
        return x
    labels = pl.Series(cell_info, dtype=pl.String)
    n_cells = x.nhoods.shape[0]
    if len(labels) != n_cells:
        raise ValueError(
            f"Length of cell_info ({len(labels)}) doesn't match number of cells in "
            f"nhoods ({n_cells})"
        )
    rows = np.repeat(np.arange(n_cells), np.diff(x.nhoods.indptr).astype(np.int64))
    counts = (
        pl.DataFrame({"Nhood": x.nhoods.indices.astype(np.int64), "ct": labels[rows]})
        .group_by("Nhood", "ct")
        .len()
    )
    majority = (
        counts.sort(["Nhood", "len", "ct"], descending=[False, True, False])
        .group_by("Nhood", maintain_order=True)
        .agg(
            majority_celltype=pl.col("ct").first(),
            majority_prop=pl.col("len").first() / pl.col("len").sum(),
        )
    )
    info = x.nhoods_info.drop(
        [
            c
            for c in ("majority_celltype", "majority_prop")
            if c in x.nhoods_info.columns
        ]
    ).join(majority, on="Nhood", how="left", maintain_order="left")
    return dataclasses.replace(x, nhoods_info=info)


@beartype(conf=_CONF)
def get_differential_abundance_res(x: MiloR) -> pl.DataFrame | None:
    """The differential abundance table; warns and returns `None` if untested.

    Args:
        x: The Milo result.

    Returns:
        The neighbourhood table, or `None` before `test_nhoods()`.
    """
    if x.nhoods_info is None:
        warnings.warn(
            "No differential abundance results found in x. Did you run "
            "test_nhoods()? Returning None.",
            stacklevel=2,
        )
    return x.nhoods_info


@beartype(conf=_CONF)
def get_index_cells(x: MiloR) -> np.ndarray:
    """The neighbourhood index cells, 0-based positions among the kept cells.

    Args:
        x: The Milo result.

    Returns:
        The index cell positions.
    """
    index_cell = x.params["index_cell"]
    assert isinstance(index_cell, np.ndarray)
    return index_cell


@beartype(conf=_CONF)
def get_params(x: MiloR | ScNebula) -> dict[str, object]:
    """The parameters of the run.

    Args:
        x: The Milo or NEBULA result.

    Returns:
        A copy of the parameters.
    """
    return dict(x.params)


########
# MELD #
########


@beartype(conf=_CONF)
def meld(
    sc: SingleCells,
    sample_id_col: str,
    *,
    embd_to_use: str = "pca",
    meld_params: MeldParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> MeldRes | None:
    """MELD sample densities over the kNN graph.

    Smooths the one-hot sample indicators over the kNN graph with a
    Chebyshev-approximated spectral filter, giving a per-cell likelihood for
    each sample.

    Args:
        sc: The experiment, with the kNN.
        sample_id_col: obs column with the sample per cell.
        embd_to_use: The embedding the kNN was built on; only checked for
            staleness, as the kNN is what MELD reads.
        meld_params: The parameters. `None` uses the defaults; the kNN keys
            are unused, the cached kNN is taken as is.
        seed: Seed for the largest eigenvalue estimate.
        verbose: Controls verbosity.

    Returns:
        The `MeldRes` over the kept cells, or `None` with a warning if there
        is no kNN.

    Raises:
        ValueError: If the embedding or the kNN is stale, or a kept cell has
            no sample.

    References:
        Burkhardt, et al., Nat Biotechnol, 2021.

    Examples:
        >>> meld_res = bv.de.meld(sc, "condition")
        >>> scores = meld_res.norm_scores
    """
    params = meld_params or MeldParams()
    _hard_state(sc, [embd_to_use, "knn"])
    obs = _obs_aligned(sc, [sample_id_col])
    if sample_id_col not in obs.columns:
        raise ValueError(f"Column '{sample_id_col}' not found in the obs table.")
    codes, samples = _factor_codes(obs[sample_id_col], sample_id_col)
    knn = sc.cache.knn
    if knn is None:
        warnings.warn(
            "No kNN data could be found on the object. Please run "
            "find_neighbours(). Returning None.",
            stacklevel=2,
        )
        return None
    res = _bixverse.meld(
        knn.indices, knn.distances, codes, len(samples), to_rust(params), seed, verbose
    )
    cell_ids = sc.cell_ids(sc.cells_to_keep)

    def frame(mat: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {"cell_id": cell_ids, **{s: mat[:, j] for j, s in enumerate(samples)}}
        )

    return MeldRes(
        raw_scores=frame(res["raw_scores"]), norm_scores=frame(res["norm_scores"])
    )
