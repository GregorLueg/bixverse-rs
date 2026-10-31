"""Ligand-receptor analysis: the NicheNet-style ligand to target influence,
ligand activities and interaction prioritisation, and CellPhoneDB.

The port of R's `R/functions_sc_lr.R`: `generate_ligand_target_influence`,
`as_ligand_target_influence`, `ligand_activity_scores`,
`compute_expression_info_sc`, `prioritise_interactions`,
`get_cellphonedb_db` and `cellphonedb_sc`, and the `LigandTargetInfluence`
class. The numerics run in Rust; the table logic R does in data.table is
polars here.

    >>> inf = generate_ligand_target_influence(seeds, ppi, grn)
    >>> act = ligand_activity_scores(inf, {"degs": degs}, background=expressed)
    >>> lr_db = get_cellphonedb_db()
    >>> res = cellphonedb(sc, "cell_type", lr_db, method="simple")

Like R, the expression statistics and CellPhoneDB take a `SingleCells`.
Every index is 0-based.
"""

from __future__ import annotations

import io
import sys
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from beartype import BeartypeConf, beartype

from bixverse import _bixverse
from bixverse._core.doublets import _as_character
from bixverse._core.download import _cache, _download
from bixverse._core.single_cells import SingleCells
from bixverse.params import CellphonedbParams, LigandTargetParams, to_rust

###########
# Globals #
###########

WEIGHT_NAMES: tuple[str, ...] = (
    "de_ligand",
    "de_receptor",
    "activity_scaled",
    "exprs_ligand",
    "exprs_receptor",
    "ligand_condition_specificity",
    "receptor_condition_specificity",
)

COND_KEYS: tuple[str, ...] = (
    "ligand_condition_specificity",
    "receptor_condition_specificity",
)

# R's `.Machine$double.xmin`, the floor on a p-value before the log.
DOUBLE_XMIN: float = sys.float_info.min

CELLPHONEDB_URL: str = (
    "https://raw.githubusercontent.com/ventolab/cellphonedb-data/v5.0.0/cellphonedb.zip"
)

CELLPHONEDB_TABLES: tuple[str, ...] = (
    "interaction_table",
    "multidata_table",
    "protein_table",
    "gene_table",
    "complex_composition_table",
)

ACTIVITY_COLS: tuple[str, ...] = (
    "auroc",
    "aupr",
    "aupr_corrected",
    "pearson",
    "spearman",
)

Scenario = Literal["case_control", "one_condition"]
CpdbMethod = Literal["statistical", "degs", "simple"]

_CONF = BeartypeConf(is_pep484_tower=True)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class LigandTargetInfluence:
    """The ligands x genes influence matrix, R's `LigandTargetInfluence`.

    Attributes:
        influence: `(n_ligands, n_genes)` `float64` regulatory potential.
        ligand_seeds: The seed symbols of each row; a group is one complex.
        ligand_names: Row names.
        gene_ids: Column names, the gene universe of the run.
        params: The parameters used, `None` for a wrapped matrix.
    """

    influence: np.ndarray
    ligand_seeds: tuple[tuple[str, ...], ...]
    ligand_names: tuple[str, ...]
    gene_ids: tuple[str, ...]
    params: LigandTargetParams | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """`(n_ligands, n_genes)`, R's `dim()`."""
        return (self.influence.shape[0], self.influence.shape[1])

    def __repr__(self) -> str:
        """R's `print.LigandTargetInfluence`."""
        n, m = self.shape
        lines = [
            "LigandTargetInfluence",
            f"  No ligand seeds:    {n}",
            f"  No genes:           {m}",
        ]
        if self.params is not None:
            p = self.params
            lines += [
                f"  Damping factor:     {p.damping_factor:.3f}",
                f"  Max iter:           {p.max_iter}",
                f"  Secondary targets:  {str(p.secondary_targets).upper()}",
            ]
        return "\n".join(lines)


###########
# Helpers #
###########


@beartype
def _check_columns(df: pl.DataFrame, required: Sequence[str], name: str) -> None:
    """R's `checkmate::assertNames(must.include = ...)`.

    Raises:
        ValueError: On a missing column.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing the column(s): {missing}")


@beartype
def _string_col(df: pl.DataFrame, col: str, name: str) -> pl.DataFrame:
    """A text column as `String`, no nulls, as `checkmate::assertCharacter`.

    Raises:
        ValueError: If it is not text or holds nulls.
    """
    dtype = df[col].dtype
    if not (dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum))):
        raise ValueError(f"{name}${col} must be a string column, got {dtype}")
    if df[col].null_count():
        raise ValueError(f"{name}${col} holds missing values")
    return df.with_columns(pl.col(col).cast(pl.String))


@beartype
def _numeric_col(
    df: pl.DataFrame,
    col: str,
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> pl.DataFrame:
    """A numeric column as `Float64`, no missing values, optionally bounded.

    Raises:
        ValueError: If it is not numeric, holds missing values or leaves the
            bounds.
    """
    if not df[col].dtype.is_numeric():
        raise ValueError(f"{name}${col} must be numeric, got {df[col].dtype}")
    x = df[col].cast(pl.Float64)
    if x.null_count() or x.is_nan().any():
        raise ValueError(f"{name}${col} holds missing values")
    if (lower is not None and (x < lower).any()) or (
        upper is not None and (x > upper).any()
    ):
        raise ValueError(f"{name}${col} must lie in [{lower}, {upper}]")
    return df.with_columns(x)


@beartype
def _check_deg_table(
    x: pl.DataFrame, name: str, *, needs_cluster: bool = True
) -> pl.DataFrame:
    """R's `assertDegTable`: `gene`, `lfc`, `pval` (and `cluster_id`).

    Returns:
        The table with the text columns as `String` and the numbers as
        `Float64`.

    Raises:
        ValueError: On a missing column, a wrong type or a p-value outside
            `[0, 1]`.
    """
    required = ["gene", "lfc", "pval"]
    if needs_cluster:
        required = ["cluster_id", *required]
    _check_columns(x, required, name)
    x = _numeric_col(x, "lfc", name)
    x = _numeric_col(x, "pval", name, lower=0.0, upper=1.0)
    x = _string_col(x, "gene", name)
    if needs_cluster:
        x = _string_col(x, "cluster_id", name)
    return x


@beartype
def _check_exp_table(x: pl.DataFrame, name: str) -> pl.DataFrame:
    """R's `assertExpTable`: `cluster_id`, `gene`, `avg_expr`.

    Raises:
        ValueError: On a missing column or a wrong type.
    """
    _check_columns(x, ["cluster_id", "gene", "avg_expr"], name)
    x = _numeric_col(x, "avg_expr", name)
    x = _string_col(x, "gene", name)
    return _string_col(x, "cluster_id", name)


@beartype
def _rank_scale(x: np.ndarray) -> np.ndarray:
    """R's `.rank_scale`: average ranks over their maximum."""
    # numpy, not polars: polars divides by a scalar through its reciprocal,
    # which is off by an ulp and reorders near-tied prioritisation scores
    r: np.ndarray = pl.Series(x).rank("average").cast(pl.Float64).to_numpy()
    return r / np.max(r)


@beartype
def _scaling_zscore(x: np.ndarray) -> np.ndarray:
    """R's `.scaling_zscore`: zeros when the standard deviation is zero or
    undefined. `NaN` entries are left out of the moments and stay `NaN`."""
    ok = x[~np.isnan(x)]
    s = float(np.std(ok, ddof=1)) if ok.size > 1 else float("nan")
    if np.isnan(s) or s == 0:
        return np.zeros(x.size)
    return (x - ok.mean()) / s


@beartype
def _quantile7(x: np.ndarray, p: float) -> float:
    """R's default `quantile(type = 7)`, interpolating as R does."""
    s = np.sort(x)
    idx = (s.size - 1) * p
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    q = float(s[lo])
    h = idx - lo
    if h > 0 and s[hi] != q:
        q = (1 - h) * q + h * float(s[hi])
    return q


@beartype
def _scale_quantile_adapted(x: np.ndarray, outlier_cutoff: float = 0.05) -> np.ndarray:
    """R's `.scale_quantile_adapted`: clip to the `[cutoff, 1 - cutoff]`
    quantiles, map to `[0, 1]`, 0.5 everywhere if they coincide. `NaN`
    entries are left out of the quantiles and stay `NaN`."""
    ok = x[~np.isnan(x)]
    q_low = _quantile7(ok, outlier_cutoff)
    q_high = _quantile7(ok, 1 - outlier_cutoff)
    if q_high == q_low:
        return np.full(x.size, 0.5)
    return (np.clip(x, q_low, q_high) - q_low) / (q_high - q_low)


@beartype
def _scaled_by(df: pl.DataFrame, col: str, by: str, out: str) -> pl.DataFrame:
    """`.scale_quantile_adapted` of `col` within each level of `by`."""
    x = df[col].to_numpy().astype(np.float64)
    res = np.empty(x.size)
    for idx in df.with_row_index("_i").group_by(by).agg("_i")["_i"].to_list():
        res[idx] = _scale_quantile_adapted(x[idx])
    return df.with_columns(pl.Series(out, res))


@beartype
def _p_val_adapted(pval: str, lfc: str) -> pl.Expr:
    """`-log10(pmax(p, xmin)) * sign(lfc)`."""
    return -pl.col(pval).clip(lower_bound=DOUBLE_XMIN).log10() * pl.col(lfc).sign()


@beartype
def _with_rank_scaled(df: pl.DataFrame, src: str, out: str) -> pl.DataFrame:
    """`.rank_scale` of `src` over the whole table."""
    return df.with_columns(pl.Series(out, _rank_scale(df[src].to_numpy())))


@beartype
def _join_into(x: pl.DataFrame, out: pl.DataFrame, on: list[str]) -> pl.DataFrame:
    """data.table's `x[out, on = on, nomatch = NULL]`: the rows of `out`
    that match, in its order, `x`'s columns first."""
    res = out.join(x, on=on, how="inner", maintain_order="left")
    return res.select([*x.columns, *[c for c in out.columns if c not in x.columns]])


############
# NicheNet #
############


@beartype
def as_ligand_target_influence(
    x: np.ndarray,
    row_names: Sequence[str],
    col_names: Sequence[str],
    *,
    ligands_as_cols: bool = False,
    params: LigandTargetParams | None = None,
) -> LigandTargetInfluence:
    """Wrap an existing ligand-target matrix, as R's
    `as_ligand_target_influence`.

    For a regulatory potential matrix built elsewhere, for example with
    nichenetr's `construct_ligand_target_matrix()`, so it can be scored with
    `ligand_activity_scores`.

    Args:
        x: Numeric matrix.
        row_names: Unique row names.
        col_names: Unique column names.
        ligands_as_cols: `x` is genes x ligands, the nichenetr layout.
        params: Parameters used to build `x`, kept for the record.

    Returns:
        The influence, each ligand its own seed.

    Raises:
        ValueError: On a non-numeric matrix or names that do not fit it.
    """
    m = np.asarray(x)
    if m.ndim != 2 or m.dtype.kind not in "iuf":
        raise ValueError("x must be a numeric matrix")
    if len(row_names) != m.shape[0] or len(col_names) != m.shape[1]:
        raise ValueError(f"names do not fit the {m.shape} matrix")
    for names, what in ((row_names, "row_names"), (col_names, "col_names")):
        if len(set(names)) != len(names):
            raise ValueError(f"{what} must be unique")
    rows, cols = list(row_names), list(col_names)
    if ligands_as_cols:
        m, rows, cols = m.T, cols, rows
    return LigandTargetInfluence(
        influence=np.ascontiguousarray(m, dtype=np.float64),
        ligand_seeds=tuple((r,) for r in rows),
        ligand_names=tuple(rows),
        gene_ids=tuple(cols),
        params=params,
    )


@beartype
def get_influence(x: LigandTargetInfluence) -> np.ndarray:
    """The ligands x genes influence, R's `get_influence`; the names are
    `x.ligand_names` and `x.gene_ids`."""
    return x.influence


@beartype
def _network(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """A `from`, `to`, `weight` network with text node ids.

    Raises:
        ValueError: On a missing column.
    """
    _check_columns(df, ["from", "to", "weight"], name)
    return df.select(
        pl.col("from").cast(pl.String),
        pl.col("to").cast(pl.String),
        pl.col("weight").cast(pl.Float64),
    )


@beartype
def generate_ligand_target_influence(
    ligand_seeds: Mapping[str, Sequence[str] | str] | Sequence[Sequence[str] | str],
    ppi_network: pl.DataFrame,
    grn_network: pl.DataFrame,
    params: LigandTargetParams | None = None,
) -> LigandTargetInfluence:
    """The NicheNet-style ligand to target influence matrix, as R's
    `generate_ligand_target_influence`.

    The gene universe is the union of the nodes of both networks, in order
    of first appearance (`ppi$from`, `ppi$to`, `grn$from`, `grn$to`).

    Args:
        ligand_seeds: One ligand symbol or a group treated as one complex per
            row. A mapping names the rows; otherwise they are the symbols
            joined with `+`.
        ppi_network: `from`, `to`, `weight`. The signalling layer.
        grn_network: `from`, `to`, `weight`. The gene regulatory layer.
        params: The parameters. `None` uses the defaults.

    Returns:
        The influence, ligands x gene universe.

    Raises:
        ValueError: On no seeds, a missing column or a seed in neither
            network.
    """
    params = params or LigandTargetParams()
    if isinstance(ligand_seeds, Mapping):
        names: list[str] | None = [str(k) for k in ligand_seeds]
        groups = list(ligand_seeds.values())
    else:
        names = None
        groups = list(ligand_seeds)
    seeds = [(g,) if isinstance(g, str) else tuple(g) for g in groups]
    if not seeds:
        raise ValueError("ligand_seeds must hold at least one seed")
    ppi = _network(ppi_network, "ppi_network")
    grn = _network(grn_network, "grn_network")

    all_genes = (
        pl.concat([ppi["from"], ppi["to"], grn["from"], grn["to"]])
        .unique(maintain_order=True)
        .to_list()
    )
    pos = {g: i for i, g in enumerate(all_genes)}
    missing = list(dict.fromkeys(s for g in seeds for s in g if s not in pos))
    if missing:
        raise ValueError(
            f"Ligand seeds not present in either network: {', '.join(missing)}"
        )

    def idx(s: pl.Series) -> np.ndarray:
        return np.asarray([pos[v] for v in s.to_list()], dtype=np.uint32)

    res = _bixverse.ligand_target_influence(
        [np.asarray([pos[s] for s in g], dtype=np.uint32) for g in seeds],
        idx(ppi["from"]),
        idx(ppi["to"]),
        ppi["weight"].to_numpy(),
        idx(grn["from"]),
        idx(grn["to"]),
        grn["weight"].to_numpy(),
        len(all_genes),
        to_rust(params),
    )
    return LigandTargetInfluence(
        influence=res,
        ligand_seeds=tuple(seeds),
        ligand_names=tuple(names if names else ("+".join(g) for g in seeds)),
        gene_ids=tuple(all_genes),
        params=params,
    )


@beartype
def ligand_activity_scores(
    ligand_influence: LigandTargetInfluence,
    gene_sets: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | Sequence[str],
    *,
    background: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Ligand activities against gene sets, as R's `ligand_activity_scores`.

    The influence is restricted to the `background` columns first, so the
    metrics rank within the background (NicheNet practice: the genes
    expressed in the receiver, the gene set the DEGs).

    Args:
        ligand_influence: The influence.
        gene_sets: One gene set, a list of them, or a mapping naming them.
            Unnamed sets are called `"1"`, `"2"`, ...
        background: Genes to score against. `None` uses every column.
            Genes not in the influence are dropped silently, as in R.

    Returns:
        One row per gene set and ligand: `gene_set`, `ligand`, `auroc`,
        `aupr`, `aupr_corrected`, `pearson` and `spearman` (`NaN` where
        undefined).

    Raises:
        ValueError: If there are no gene sets or the background misses the
            influence entirely.
    """
    if isinstance(gene_sets, Mapping):
        set_names = list(gene_sets)
        sets = [list(s) for s in gene_sets.values()]
    elif all(isinstance(s, str) for s in gene_sets):
        set_names, sets = ["1"], [list(gene_sets)]
    else:
        sets = [list(s) for s in gene_sets]
        set_names = [str(i + 1) for i in range(len(sets))]
    if not sets:
        raise ValueError("gene_sets must hold at least one gene set")

    genes = ligand_influence.gene_ids
    bg = list(genes) if background is None else list(background)
    if not bg:
        raise ValueError("background must hold at least one gene")
    col = {g: j for j, g in enumerate(genes)}
    bg_present = [g for g in dict.fromkeys(bg) if g in col]
    if not bg_present:
        raise ValueError(
            "No background genes overlap with the influence matrix columns."
        )
    sub = np.ascontiguousarray(
        ligand_influence.influence[:, [col[g] for g in bg_present]]
    )
    members = [[g in set(s) for g in bg_present] for s in sets]
    res = _bixverse.ligand_activity_scores(sub, members)

    n_lig = sub.shape[0]
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "gene_set": [name] * n_lig,
                    "ligand": list(ligand_influence.ligand_names),
                    **{k: np.asarray(r[k], dtype=np.float64) for k in ACTIVITY_COLS},
                }
            )
            for name, r in zip(set_names, res, strict=True)
        ]
    )


##################
# Prioritisation #
##################


@beartype(conf=_CONF)
def resolve_weights(
    weights: Mapping[str, float] | None,
    scenario: Scenario,
    has_condition_de: bool,
) -> dict[str, float]:
    """The prioritisation weights, R's `resolve_weights`.

    Args:
        weights: One non-negative weight per name in `WEIGHT_NAMES`; extra
            names are ignored. `None` picks by `scenario`.
        scenario: `"case_control"` weighs everything 1, `"one_condition"`
            drops the condition specificity terms.
        has_condition_de: Whether a condition DE table was supplied.

    Returns:
        Name to weight, in `WEIGHT_NAMES` order.

    Raises:
        ValueError: On a missing name, a negative or non-finite weight, or
            non-zero condition weights without a condition DE table.
    """
    if weights is None:
        off = COND_KEYS if scenario == "one_condition" else ()
        w = {k: 0.0 if k in off else 1.0 for k in WEIGHT_NAMES}
    else:
        missing = [k for k in WEIGHT_NAMES if k not in weights]
        if missing:
            raise ValueError(f"weights is missing: {missing}")
        w = {k: float(weights[k]) for k in WEIGHT_NAMES}
        if any(not np.isfinite(v) or v < 0 for v in w.values()):
            raise ValueError("weights must be finite and non-negative")
    if not has_condition_de and any(w[k] > 0 for k in COND_KEYS):
        raise ValueError(
            "Non-zero condition-specificity weights require `condition_de`."
        )
    return w


@beartype
def compute_expression_info(
    sc: SingleCells,
    celltype_colname: str,
    genes: Sequence[str],
    *,
    condition_colname: str | None = None,
    condition_oi: str | None = None,
) -> pl.DataFrame:
    """Per-cluster mean expression and expressing fraction, as R's
    `compute_expression_info_sc`.

    Streams the genes off the gene-major store and aggregates over the kept
    cells per cluster; cells with a missing label are ignored. With a
    condition only that condition's cells count.

    Args:
        sc: The experiment.
        celltype_colname: The cluster column in obs.
        genes: Gene ids. Unknown ids warn and drop.
        condition_colname: Optional condition column in obs.
        condition_oi: The condition level to keep.

    Returns:
        `cluster_id`, `gene`, `avg_expr` and `frac_expr`, clusters sorted,
        genes in the order given within each.

    Raises:
        ValueError: On half a condition, no known gene or no cells left.
    """
    if (condition_colname is None) != (condition_oi is None):
        raise ValueError(
            "Provide both `condition_colname` and `condition_oi`, or neither."
        )
    if not genes:
        raise ValueError("genes must hold at least one gene")
    gene_ids = sc.gene_ids().tolist()
    pos: dict[str, int] = {}
    for i, g in enumerate(gene_ids):
        pos.setdefault(g, i)
    genes = list(genes)
    n_missing = len(set(genes) - set(pos))
    if n_missing:
        warnings.warn(
            f"Dropping {n_missing} gene(s) not in the var table.", stacklevel=2
        )
        genes = [g for g in dict.fromkeys(genes) if g in pos]
        if not genes:
            raise ValueError("No requested genes are present in `object`.")

    cols = ["cell_idx", celltype_colname]
    if condition_colname is not None:
        cols.append(condition_colname)
    obs = sc.obs(cols=list(dict.fromkeys(cols)), filtered=True)
    if condition_colname is not None:
        cond = pl.Series(_as_character(obs[condition_colname]), dtype=pl.String)
        obs = obs.filter(cond == condition_oi)
    if obs.height == 0:
        raise ValueError("No cells remain after filtering by `to_keep` and condition.")

    obs = obs.with_columns(
        pl.Series("_lab", _as_character(obs[celltype_colname]), dtype=pl.String)
    )
    levels = sorted(obs["_lab"].drop_nulls().unique().to_list())
    clusters = [
        obs.filter(pl.col("_lab") == cl)["cell_idx"].to_numpy().astype(np.uint32)
        for cl in levels
    ]
    res = _bixverse.cluster_expr_stats(
        str(sc.genes_path),
        np.asarray([pos[g] for g in genes], dtype=np.uint32),
        clusters,
    )
    return pl.DataFrame(
        {
            "cluster_id": np.repeat(levels, len(genes)).tolist(),
            "gene": genes * len(levels),
            "avg_expr": res["mean"].T.ravel(),
            "frac_expr": res["frac"].T.ravel(),
        },
        schema_overrides={"cluster_id": pl.String, "gene": pl.String},
    )


@beartype(conf=_CONF)
def prioritise_interactions(
    celltype_de: pl.DataFrame,
    expression_info: pl.DataFrame,
    ligand_activities: pl.DataFrame,
    lr_network: pl.DataFrame,
    senders_oi: Sequence[str],
    receivers_oi: Sequence[str],
    *,
    condition_de: pl.DataFrame | None = None,
    weights: Mapping[str, float] | None = None,
    scenario: Scenario = "case_control",
) -> pl.DataFrame:
    """Prioritise sender-ligand-receiver-receptor interactions, as R's
    `prioritise_interactions`.

    Each (sender, ligand, receiver, receptor) from the L/R network gets a
    weighted score over ligand DE in the sender, receptor DE in the
    receiver, the ligand activity, the ligand (receptor) expression
    specificity across senders (receivers) and the condition specificity of
    both. A component with weight zero is skipped, its join included; every
    other one is an inner join, so a tuple missing from it drops out. DE and
    expression terms count half, as nichenetr's.

    Args:
        celltype_de: `cluster_id`, `gene`, `lfc`, `pval`, one row per
            cluster and gene, covering senders and receivers.
        expression_info: `cluster_id`, `gene`, `avg_expr`, as
            `compute_expression_info` returns.
        ligand_activities: `ligand` and `aupr_corrected`, one row per
            ligand: one gene set of `ligand_activity_scores`.
        lr_network: `ligand`, `receptor`.
        senders_oi: Sender clusters.
        receivers_oi: Receiver clusters.
        condition_de: Optional `gene`, `lfc`, `pval` from a condition
            contrast. Needed for non-zero condition weights.
        weights: Weight per name in `WEIGHT_NAMES`. `None` picks by
            `scenario`.
        scenario: `"case_control"` (all weights 1) or `"one_condition"`
            (condition weights 0). Ignored if `weights` is given.

    Returns:
        One row per surviving tuple, `sender`, `receiver`, `ligand`,
        `receptor` first, by descending `prioritisation_score` with a
        `prioritisation_rank` (ties share the lowest). A score that is
        undefined, from a `NaN` activity, is null and sorts last. Empty,
        with a warning, if no tuple survives.

    Raises:
        ValueError: On malformed tables or more than one row per ligand in
            `ligand_activities`.
    """
    celltype_de = _check_deg_table(celltype_de, "celltype_de")
    expression_info = _check_exp_table(expression_info, "expression_info")
    _check_columns(ligand_activities, ["ligand", "aupr_corrected"], "ligand_activities")
    _check_columns(lr_network, ["ligand", "receptor"], "lr_network")
    if not senders_oi or not receivers_oi:
        raise ValueError("senders_oi and receivers_oi must not be empty")
    if condition_de is not None:
        condition_de = _check_deg_table(
            condition_de, "condition_de", needs_cluster=False
        )
    if ligand_activities["ligand"].is_duplicated().any():
        raise ValueError(
            "`ligand_activities` must have one row per ligand; "
            "filter to a single gene set first."
        )
    w = resolve_weights(weights, scenario, condition_de is not None)

    lr = lr_network.select(
        pl.col("ligand").cast(pl.String), pl.col("receptor").cast(pl.String)
    )
    ligands = lr["ligand"].unique().to_list()
    receptors = lr["receptor"].unique().to_list()
    senders, receivers = list(senders_oi), list(receivers_oi)

    # 1. / 2. ligand DE in the senders, receptor DE in the receivers
    sender_de = celltype_de.filter(
        pl.col("cluster_id").is_in(senders) & pl.col("gene").is_in(ligands)
    ).select(sender="cluster_id", ligand="gene", lfc_ligand="lfc", pval_ligand="pval")
    if w["de_ligand"] > 0 and sender_de.height:
        sender_de = sender_de.with_columns(
            _p_val_adapted("pval_ligand", "lfc_ligand").alias("p_val_adapted_ligand")
        )
        sender_de = _with_rank_scaled(
            sender_de, "p_val_adapted_ligand", "scaled_p_val_adapted_ligand"
        )
    receiver_de = celltype_de.filter(
        pl.col("cluster_id").is_in(receivers) & pl.col("gene").is_in(receptors)
    ).select(
        receiver="cluster_id",
        receptor="gene",
        lfc_receptor="lfc",
        pval_receptor="pval",
    )
    if w["de_receptor"] > 0 and receiver_de.height:
        receiver_de = receiver_de.with_columns(
            _p_val_adapted("pval_receptor", "lfc_receptor").alias(
                "p_val_adapted_receptor"
            )
        )
        receiver_de = _with_rank_scaled(
            receiver_de, "p_val_adapted_receptor", "scaled_p_val_adapted_receptor"
        )

    # 3. / 4. expression, scaled within each ligand (receptor)
    ligand_expr = expression_info.filter(
        pl.col("cluster_id").is_in(senders) & pl.col("gene").is_in(ligands)
    ).select(sender="cluster_id", ligand="gene", avg_ligand="avg_expr")
    if w["exprs_ligand"] > 0 and ligand_expr.height:
        ligand_expr = _scaled_by(
            ligand_expr, "avg_ligand", "ligand", "scaled_avg_exprs_ligand"
        )
    receptor_expr = expression_info.filter(
        pl.col("cluster_id").is_in(receivers) & pl.col("gene").is_in(receptors)
    ).select(receiver="cluster_id", receptor="gene", avg_receptor="avg_expr")
    if w["exprs_receptor"] > 0 and receptor_expr.height:
        receptor_expr = _scaled_by(
            receptor_expr, "avg_receptor", "receptor", "scaled_avg_exprs_receptor"
        )

    # 5. ligand activity
    la = ligand_activities.select(
        pl.col("ligand").cast(pl.String),
        pl.col("aupr_corrected").cast(pl.Float64).alias("activity"),
    )
    if w["activity_scaled"] > 0:
        act = la["activity"].fill_null(float("nan")).to_numpy()
        la = la.with_columns(
            pl.Series("activity_zscore", _scaling_zscore(act)),
            pl.Series("scaled_activity", _scale_quantile_adapted(act, 0.01)),
        )

    # 6. condition DE
    ligand_cond = receptor_cond = None
    if condition_de is not None:
        if w["ligand_condition_specificity"] > 0:
            ligand_cond = (
                condition_de.filter(pl.col("gene").is_in(ligands))
                .select(ligand="gene", lfc_ligand_group="lfc", pval_ligand_group="pval")
                .with_columns(
                    _p_val_adapted("pval_ligand_group", "lfc_ligand_group").alias(
                        "p_val_adapted_ligand_group"
                    )
                )
            )
            ligand_cond = _with_rank_scaled(
                ligand_cond,
                "p_val_adapted_ligand_group",
                "scaled_p_val_adapted_ligand_group",
            )
        if w["receptor_condition_specificity"] > 0:
            receptor_cond = (
                condition_de.filter(pl.col("gene").is_in(receptors))
                .select(
                    receptor="gene",
                    lfc_receptor_group="lfc",
                    pval_receptor_group="pval",
                )
                .with_columns(
                    _p_val_adapted("pval_receptor_group", "lfc_receptor_group").alias(
                        "p_val_adapted_receptor_group"
                    )
                )
            )
            receptor_cond = _with_rank_scaled(
                receptor_cond,
                "p_val_adapted_receptor_group",
                "scaled_p_val_adapted_receptor_group",
            )

    # 7. base table: data.table's CJ sorts each side, the L/R pairs keep order
    sr = pl.DataFrame({"sender": sorted(senders)}).join(
        pl.DataFrame({"receiver": sorted(receivers)}), how="cross"
    )
    out = sr.join(lr, how="cross")

    # 8. inner joins of the active components
    if w["de_ligand"] > 0:
        out = _join_into(sender_de, out, ["sender", "ligand"])
    if w["de_receptor"] > 0:
        out = _join_into(receiver_de, out, ["receiver", "receptor"])
    if w["exprs_ligand"] > 0:
        out = _join_into(ligand_expr, out, ["sender", "ligand"])
    if w["exprs_receptor"] > 0:
        out = _join_into(receptor_expr, out, ["receiver", "receptor"])
    if w["activity_scaled"] > 0:
        out = _join_into(la, out, ["ligand"])
    if ligand_cond is not None:
        out = _join_into(ligand_cond, out, ["ligand"])
    if receptor_cond is not None:
        out = _join_into(receptor_cond, out, ["receptor"])

    if out.height == 0:
        warnings.warn(
            "No interactions remain after all joins. Check inputs.", stacklevel=2
        )
        return out

    # 9. weighted score; DE and expression are halved as each has two sides
    terms = {
        "scaled_p_val_adapted_ligand": 0.5 * w["de_ligand"],
        "scaled_p_val_adapted_receptor": 0.5 * w["de_receptor"],
        "scaled_activity": w["activity_scaled"],
        "scaled_avg_exprs_ligand": 0.5 * w["exprs_ligand"],
        "scaled_avg_exprs_receptor": 0.5 * w["exprs_receptor"],
        "scaled_p_val_adapted_ligand_group": w["ligand_condition_specificity"],
        "scaled_p_val_adapted_receptor_group": w["receptor_condition_specificity"],
    }
    sum_w = (
        0.5 * w["de_ligand"]
        + 0.5 * w["de_receptor"]
        + w["activity_scaled"]
        + 0.5 * w["exprs_ligand"]
        + 0.5 * w["exprs_receptor"]
        + w["ligand_condition_specificity"]
        + w["receptor_condition_specificity"]
    )
    score = np.zeros(out.height)
    for col, wt in terms.items():
        if wt > 0 and col in out.columns:
            score = score + wt * out[col].to_numpy()
    out = (
        out.with_columns(
            pl.Series("prioritisation_score", score / sum_w).fill_nan(None)
        )
        .with_columns(
            pl.col("prioritisation_score")
            .rank("min", descending=True)
            .alias("prioritisation_rank")
        )
        .sort(
            "prioritisation_score",
            descending=True,
            nulls_last=True,
            maintain_order=True,
        )
    )
    ids = ["sender", "receiver", "ligand", "receptor"]
    return out.select([*ids, *[c for c in out.columns if c not in ids]])


###############
# CellPhoneDB #
###############


@beartype
def _read_cellphonedb_zip(zip_file: Path) -> dict[str, pl.DataFrame]:
    """The five CellPhoneDB tables out of the release zip, read in place."""
    with zipfile.ZipFile(zip_file) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        out = {}
        for tab in CELLPHONEDB_TABLES:
            member = names.get(f"{tab}.csv")
            if member is None:
                raise ValueError(f"{zip_file} holds no {tab}.csv")
            out[tab] = pl.read_csv(
                io.BytesIO(zf.read(member)), infer_schema_length=None
            )
    return out


@beartype
def _as_bool(s: pl.Series) -> pl.Series:
    """CellPhoneDB's `True` / `False` flags as booleans, however they parse."""
    if s.dtype == pl.String:
        return s.str.to_lowercase() == "true"
    return s.cast(pl.Boolean)


@beartype
def _flatten_cellphonedb(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """The five CellPhoneDB tables as one row per interaction, see
    `get_cellphonedb_db`."""
    protein_genes = (
        tables["protein_table"]
        .select("id_protein", multidata_id="protein_multidata_id")
        .join(
            tables["gene_table"].select("protein_id", "gene_name").unique(),
            left_on="id_protein",
            right_on="protein_id",
        )
        .select("multidata_id", "gene_name")
    )
    complex_genes = (
        tables["complex_composition_table"]
        .select(multidata_id="complex_multidata_id", pid="protein_multidata_id")
        .join(
            protein_genes.rename({"multidata_id": "pid"}),
            on="pid",
        )
        .select("multidata_id", "gene_name")
    )
    multidata = tables["multidata_table"]
    partners = (
        pl.concat([protein_genes, complex_genes])
        .group_by("multidata_id")
        .agg(genes=pl.col("gene_name").cast(pl.String).unique().sort())
        .join(
            multidata.select(
                multidata_id="id_multidata",
                name=pl.col("name").cast(pl.String),
                is_complex=_as_bool(multidata["is_complex"]),
            ),
            on="multidata_id",
        )
        .with_columns(
            name=pl.when(pl.col("is_complex"))
            .then(pl.col("name"))
            .otherwise(pl.col("genes").list.first())
        )
    )

    def side(k: str, s: str) -> pl.DataFrame:
        return partners.select(
            pl.col("multidata_id").alias(f"multidata_{k}_id"),
            pl.col("name").alias(f"partner_{s}"),
            pl.col("genes").alias(f"genes_{s}"),
            pl.col("is_complex").alias(f"is_complex_{s}"),
        )

    return (
        tables["interaction_table"]
        .select(
            pl.col("id_cp_interaction").cast(pl.String).alias("interaction_id"),
            "multidata_1_id",
            "multidata_2_id",
            pl.col("directionality").cast(pl.String),
            pl.col("classification").cast(pl.String),
        )
        .join(side("1", "a"), on="multidata_1_id")
        .join(side("2", "b"), on="multidata_2_id")
        .select(
            "interaction_id",
            "partner_a",
            "partner_b",
            "genes_a",
            "genes_b",
            "is_complex_a",
            "is_complex_b",
            "directionality",
            "classification",
        )
        .sort("interaction_id", maintain_order=True)
    )


@beartype
def get_cellphonedb_db(
    cache_dir: Path | str | None = None, *, verbose: bool = True
) -> pl.DataFrame:
    """The CellPhoneDB v5.0.0 ligand-receptor database, as R's
    `get_cellphonedb_db`.

    Fetches `cellphonedb.zip` from the
    [cellphonedb-data](https://github.com/ventolab/cellphonedb-data)
    repository unless it is already in `cache_dir`, and flattens it to one
    row per interaction. Each partner resolves to the gene symbols of its
    subunits: one for a single protein, several for a complex.

    Args:
        cache_dir: Where to keep `cellphonedb.zip`. `None` uses the bixverse
            cache.
        verbose: Log the download.

    Returns:
        `interaction_id`, `partner_a` / `partner_b` (the gene symbol of a
        single protein, the complex name otherwise), `genes_a` / `genes_b`
        (the subunit symbols, sorted), `is_complex_a` / `is_complex_b`,
        `directionality` and `classification`, by `interaction_id`.

    References:
        Troulé, et al., Nat. Protoc., 2025.
    """
    zip_file = _cache(cache_dir) / "cellphonedb.zip"
    if not zip_file.exists():
        _download([CELLPHONEDB_URL], zip_file, quiet=not verbose)
    return _flatten_cellphonedb(_read_cellphonedb_zip(zip_file))


@beartype
def _check_lr_db(lr_db: pl.DataFrame) -> None:
    """R's `checkLrDb`.

    Raises:
        ValueError: On a missing column, missing or duplicated interaction
            ids, or a partner without genes.
    """
    _check_columns(
        lr_db,
        ["interaction_id", "partner_a", "partner_b", "genes_a", "genes_b"],
        "lr_db",
    )
    ids = lr_db["interaction_id"]
    if ids.dtype != pl.String or ids.null_count():
        raise ValueError("lr_db$interaction_id must be strings without missing values")
    if ids.is_duplicated().any():
        raise ValueError("lr_db$interaction_id: must be unique.")
    for col in ("genes_a", "genes_b"):
        s = lr_db[col]
        if s.dtype != pl.List(pl.String):
            raise ValueError(f"lr_db${col} must be a list of strings")
        if (s.list.len().fill_null(0) == 0).any() or s.list.eval(
            pl.element().is_null()
        ).list.any().any():
            raise ValueError(f"{col}: every partner needs at least one gene.")


@beartype
def cellphonedb(
    sc: SingleCells,
    celltype_colname: str,
    lr_db: pl.DataFrame,
    *,
    method: CpdbMethod = "statistical",
    deg_table: pl.DataFrame | None = None,
    senders: Sequence[str] | None = None,
    receivers: Sequence[str] | None = None,
    gene_id_col: str = "gene_id",
    params: CellphonedbParams | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """CellPhoneDB ligand-receptor analysis, as R's `cellphonedb_sc`.

    For every interaction and ordered (sender, receiver) cluster pair,
    partner a is measured in the sender and partner b in the receiver. The
    interaction mean is the average of the two partner means, zero if either
    is not expressed; a complex takes the minimum over its subunits. The gate
    needs both partners expressed in more than `threshold` of their cluster's
    cells.

    - `"statistical"`: permutes the cluster labels, `p = #(perm > real) /
      n_perm`, 1 where the mean is zero or the gate fails.
    - `"degs"`: no permutations; the gate also needs partner a to be a DEG
      of the sender or partner b of the receiver, as `deg_table` says.
    - `"simple"`: the means and the expression gate only.

    Reads the normalised layer of the L/R genes once. Interactions with a
    subunit missing from `sc` drop with a warning.

    Args:
        sc: The experiment.
        celltype_colname: The cluster column in obs. Cells with a missing
            label are ignored.
        lr_db: The database, see `get_cellphonedb_db`.
        method: See above.
        deg_table: `cluster_id` and `gene`, the DEGs per cluster. Needed for
            `"degs"`, ignored with a warning otherwise.
        senders: Restrict the pairs to these sender clusters. `None` uses all.
        receivers: The same for the receivers.
        gene_id_col: The var column holding the symbols used in `lr_db`. A
            duplicated symbol resolves to its first gene.
        params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        One row per interaction and pair: `interaction_id`, `partner_a`,
        `partner_b`, `sender`, `receiver`, `mean`, `pval` (null unless
        `"statistical"`) and `gate`.

    Raises:
        ValueError: On a malformed database, no testable interaction, an
            unknown sender or receiver, or `"degs"` without `deg_table`.

    References:
        Efremova, et al., Nat. Protoc., 2020; Troulé, et al., Nat. Protoc.,
        2025.
    """
    params = params or CellphonedbParams()
    _check_lr_db(lr_db)
    if method == "degs":
        if deg_table is None:
            raise ValueError("method = 'degs' needs a `deg_table`")
        _check_columns(deg_table, ["cluster_id", "gene"], "deg_table")
    elif deg_table is not None:
        warnings.warn(
            "`deg_table` is only used with method = 'degs'. Ignoring it.", stacklevel=2
        )

    var = sc.var(cols=[gene_id_col])
    pos: dict[str, int] = {}
    for i, g in enumerate(var[gene_id_col].cast(pl.String).to_list()):
        if g is not None:
            pos.setdefault(g, i)

    present = np.array(
        [
            all(g in pos for g in a + b)
            for a, b in zip(
                lr_db["genes_a"].to_list(), lr_db["genes_b"].to_list(), strict=True
            )
        ],
        dtype=bool,
    )
    if not present.any():
        raise ValueError("No interaction has all of its subunits in `object`.")
    if not present.all():
        warnings.warn(
            f"Dropping {int((~present).sum())} of {present.size} interactions with "
            "subunits missing from `object`.",
            stacklevel=2,
        )
    lr_db = lr_db.filter(pl.Series(present))

    def to_idx(genes: Sequence[str]) -> np.ndarray:
        return np.asarray([pos[g] for g in genes], dtype=np.uint32)

    obs = sc.obs(cols=["cell_idx", celltype_colname], filtered=True)
    obs = obs.with_columns(
        pl.Series("_lab", _as_character(obs[celltype_colname]), dtype=pl.String)
    )
    obs = obs.filter(pl.col("_lab").is_not_null())
    levels = sorted(obs["_lab"].unique().to_list())
    clusters = [
        obs.filter(pl.col("_lab") == cl)["cell_idx"].to_numpy().astype(np.uint32)
        for cl in levels
    ]

    level_pos = {cl: i for i, cl in enumerate(levels)}
    for side, name in ((senders, "senders"), (receivers, "receivers")):
        bad = [s for s in side or [] if s not in level_pos]
        if bad:
            raise ValueError(
                f"{name} holds clusters not in `{celltype_colname}`: {bad}"
            )
    pairs = None
    if senders is not None or receivers is not None:
        a = sorted(level_pos[s] for s in (senders if senders is not None else levels))
        b = sorted(
            level_pos[r] for r in (receivers if receivers is not None else levels)
        )
        pairs = [(i, j) for i in a for j in b]

    deg_genes = None
    if method == "degs":
        assert deg_table is not None
        deg = deg_table.select(
            pl.col("cluster_id").cast(pl.String), pl.col("gene").cast(pl.String)
        ).filter(pl.col("gene").is_in(list(pos)))
        deg_genes = [
            to_idx(
                deg.filter(pl.col("cluster_id") == cl)["gene"]
                .unique(maintain_order=True)
                .to_list()
            )
            for cl in levels
        ]

    res = _bixverse.cellphonedb(
        str(sc.genes_path),
        [to_idx(g) for g in lr_db["genes_a"].to_list()],
        [to_idx(g) for g in lr_db["genes_b"].to_list()],
        clusters,
        pairs,
        deg_genes,
        method == "statistical",
        to_rust(params),
        int(verbose),
    )

    # the matrices are (interactions, pairs); R flattens them column-major
    n_inter = lr_db.height
    n_pairs = len(res["pair_a"])
    n = n_inter * n_pairs
    pvals = res["pvals"]
    return pl.DataFrame(
        {
            "interaction_id": lr_db["interaction_id"].to_list() * n_pairs,
            "partner_a": lr_db["partner_a"].cast(pl.String).to_list() * n_pairs,
            "partner_b": lr_db["partner_b"].cast(pl.String).to_list() * n_pairs,
            "sender": np.repeat([levels[i] for i in res["pair_a"]], n_inter).tolist(),
            "receiver": np.repeat([levels[i] for i in res["pair_b"]], n_inter).tolist(),
            "mean": res["means"].T.ravel(),
            "pval": np.full(n, np.nan) if pvals is None else pvals.T.ravel(),
            "gate": res["gate"].T.ravel(),
        },
        schema_overrides={"sender": pl.String, "receiver": pl.String},
    ).with_columns(pl.col("pval").fill_nan(None) if pvals is None else pl.col("pval"))
