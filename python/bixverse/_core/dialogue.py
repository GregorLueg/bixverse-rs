"""DIALOGUE: multicellular programmes, genes that are cell type specific but
whose activity covaries across the samples the cell types share.

The port of R's `dialogue_sc` over `SingleCells`, `SingleCellsSubset`
(`R/methods_sc_dialogue.R`) and `MetaCells` (`R/methods_mc_analysis.R`), and
the `DialogueResult` class. The work runs in Rust.

    >>> res = dialogue(sc, "cell_type", "sample_id", features)
    >>> res.programmes

`features` is not computed here: per cell type a frame of `cell_id` (the
meta cell id for meta cells) plus at least two numeric feature columns, rows
matched to cells by id. Use a PCA per cell type, not a slice of a global
one. Programmes are numbered from 1, matching their `mcp_01`, ... names;
every other index is 0-based.

References:
    Jerby-Arnon & Regev, Nat. Biotechnol., 2022.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells, csr_args
from bixverse._core.single_cells import SingleCells
from bixverse._core.subset import ScOrMc
from bixverse.params import (
    DialogueHlmParams,
    DialoguePmdParams,
    DialogueRefineParams,
    to_rust,
)

###########
# Globals #
###########

# Mirrors MIN_SHARED_SAMPLES on the Rust side, checked here so the error
# names the study design rather than surfacing from the decomposition.
DIALOGUE_MIN_SHARED_SAMPLES: int = 5


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class DialogueResult:
    """A DIALOGUE run, R's `DialogueResult`.

    Attributes:
        programmes: One row per programme and cell type pair: `programme`,
            `cell_type_a`, `cell_type_b`, the empirical `emp_p` and the
            canonical correlation `pair_cor`.
        mcp_cell_types: Per programme the cell types it spans.
        scores: Per cell type the final programme scores, `cell_id` plus
            `mcp_01`, ...
        cca_scores: The same for stage one's canonical scores.
        refit_fidelity: `cell_type` plus one column per programme: the
            correlation between the two. Low means the gene-level refit
            drifted from the programme the decomposition found.
        ws: Per cell type the sparse canonical weights, `feature` plus one
            column per programme.
        kept_features: Per cell type the feature columns that survived the
            ANOVA filter.
        verdicts: What the meta-analysis concluded per gene.
        signatures: The `permissive` and `strict` gene lists: `cell_type`,
            `programme`, `gene_id`, `direction`, `list`.
        shared_samples: Samples present in every cell type.
        cell_types: The analysed cell types, in order.
        source_class: The class it ran on.
        params: The parameters of the run.
    """

    programmes: pl.DataFrame
    mcp_cell_types: tuple[tuple[str, ...], ...]
    scores: dict[str, pl.DataFrame]
    cca_scores: dict[str, pl.DataFrame]
    refit_fidelity: pl.DataFrame
    ws: dict[str, pl.DataFrame]
    kept_features: dict[str, tuple[str, ...]]
    verdicts: pl.DataFrame
    signatures: pl.DataFrame
    shared_samples: tuple[str, ...]
    cell_types: tuple[str, ...]
    source_class: str
    params: dict[str, object]

    def __repr__(self) -> str:
        """R's `print.DialogueResult`. The worst pair decides whether a
        programme is really multicellular, so it reports the largest p."""
        k = len(self.mcp_cell_types)
        worst = (
            self.programmes.group_by("programme")
            .agg(pl.col("emp_p").max())
            .sort("programme")["emp_p"]
            .to_list()
        )
        sig = self.signatures["list"]
        lines = [
            "DialogueResult (multicellular programmes)",
            f"  Source class:     {self.source_class}",
            f"  Cell types:       {len(self.cell_types)} "
            f"({', '.join(self.cell_types)})",
            f"  Shared samples:   {len(self.shared_samples)}",
            f"  Programmes:       {k}",
            *(
                f"    mcp_{i + 1:02d} - worst pair p: {worst[i]:.4g} | spans "
                f"{len(self.mcp_cell_types[i])} cell type(s)"
                for i in range(k)
            ),
            f"  Genes with a verdict: {self.verdicts.height}",
            f"  Signature genes:  {(sig == 'permissive').sum()} permissive | "
            f"{(sig == 'strict').sum()} strict",
        ]
        return "\n".join(lines)


@beartype
def get_params(x: DialogueResult) -> dict[str, object]:
    """The parameters of the run, R's `get_params`."""
    return x.params


@beartype
def get_results(x: DialogueResult) -> pl.DataFrame:
    """The programme table, R's `get_results`."""
    return x.programmes


###########
# Helpers #
###########


@dataclass(frozen=True)
class _Prepped:
    """What `_prep_dialogue_inputs` hands to Rust and to the result.

    Attributes:
        cell_type_indices: 0-based cell (meta cell) positions per cell type.
        features: Per cell type the `float64` features, rows aligned.
        feature_names: Per cell type the feature column names.
        sample_ids: 0-based sample code per global position.
        cell_quality: Quality covariate per global position.
        cell_types: The analysed cell types, in order.
        cell_ids: Per cell type the cell ids.
        sample_levels: Sample labels, position `i` being code `i`.
    """

    cell_type_indices: list[np.ndarray]
    features: list[np.ndarray]
    feature_names: list[list[str]]
    sample_ids: np.ndarray
    cell_quality: np.ndarray
    cell_types: list[str]
    cell_ids: list[list[str]]
    sample_levels: list[str]


@beartype
def _r_scale(x: np.ndarray) -> np.ndarray:
    """R's `scale()` of one column. The sums run left to right as R's do
    (`cumsum`; numpy's `sum` is pairwise, Python's compensated): the stage
    two mixed models amplify last-ulp differences in the covariate."""
    centred = x - np.cumsum(x)[-1] / x.size
    return centred / np.sqrt(np.cumsum(centred * centred)[-1] / max(1, x.size - 1))


@beartype
def _labels(obs: pl.DataFrame, col: str) -> list[str | None]:
    """A label column as text, nulls kept.

    Raises:
        KeyError: If the column is missing.
    """
    if col not in obs.columns:
        raise KeyError(f"column '{col}' not found in obs")
    return obs[col].cast(pl.String).to_list()


@beartype
def _check_features(
    feature: pl.DataFrame, cell_ids: list[str], cell_type: str, id_col: str
) -> tuple[np.ndarray, list[str]]:
    """R's `.assert_dialogue_features`: one cell type's features, rows put in
    `cell_ids` order.

    Returns:
        The `(n_cells, k)` `float64` matrix and its column names.

    Raises:
        ValueError: On a missing id column, fewer than two numeric feature
            columns, a missing cell or missing values.
    """
    if id_col not in feature.columns:
        raise ValueError(
            f"features['{cell_type}'] has no '{id_col}' column. Rows are matched "
            "to cells by id, not by position."
        )
    cols = [c for c in feature.columns if c != id_col]
    if any(not feature[c].dtype.is_numeric() for c in cols):
        raise ValueError(f"features['{cell_type}'] holds non-numeric columns.")
    if len(cols) < 2:
        raise ValueError(
            f"features['{cell_type}'] has {len(cols)} column(s); DIALOGUE needs "
            "at least 2."
        )
    row: dict[str, int] = {}
    for i, c in enumerate(feature[id_col].cast(pl.String).to_list()):
        row.setdefault(c, i)
    missing = len(set(cell_ids) - set(row))
    if missing:
        raise ValueError(
            f"features['{cell_type}'] is missing {missing} of the {len(cell_ids)} "
            "cells of that cell type."
        )
    m = feature.select(cols).cast(pl.Float64).to_numpy()[[row[c] for c in cell_ids]]
    if np.isnan(m).any():
        raise ValueError(f"features['{cell_type}'] holds NAs.")
    return np.ascontiguousarray(m), cols


@beartype
def _prep_dialogue_inputs(
    sc: ScOrMc,
    obs: pl.DataFrame,
    cell_type_col: str,
    sample_col: str,
    features: Mapping[str, pl.DataFrame],
    quality_col: str | None,
    cell_id_col: str,
    default_quality: np.ndarray,
) -> _Prepped:
    """R's `.prep_dialogue_inputs`: the flat 0-based vectors Rust takes.

    The cell types are `features`' keys, in that order; any other label is
    ignored. `sample_ids` and `cell_quality` are indexed by global position
    and reach the largest one in use; positions of no analysed cell type are
    never read.

    Raises:
        ValueError: On fewer than two cell types, one absent from the labels,
            a missing sample label or quality value, or too few samples
            shared by every cell type.
    """
    cell_types = list(features)
    if len(cell_types) < 2:
        raise ValueError(
            f"DIALOGUE needs at least two cell types, `features` names "
            f"{len(cell_types)}. Nothing to correlate across."
        )
    labels = _labels(obs, cell_type_col)
    samples = _labels(obs, sample_col)
    ids = _labels(obs, cell_id_col)
    missing_types = [ct for ct in cell_types if ct not in set(labels)]
    if missing_types:
        raise ValueError(
            f"`features` names cell types absent from `{cell_type_col}`: "
            f"{', '.join(missing_types)}."
        )
    rows = {ct: [i for i, lab in enumerate(labels) if lab == ct] for ct in cell_types}
    if any(samples[i] is None for r in rows.values() for i in r):
        raise ValueError(f"`{sample_col}` holds missing values.")

    if quality_col is None:
        quality = _r_scale(np.log1p(default_quality.astype(np.float64)))
    else:
        q = obs[quality_col].cast(pl.Float64)
        if q.null_count() or q.is_nan().any():
            raise ValueError(f"`{quality_col}` holds missing values.")
        quality = q.to_numpy()

    cell_ids = [[str(ids[i]) for i in rows[ct]] for ct in cell_types]
    indices = [np.asarray(sc.cell_indices(cids), dtype=np.uint32) for cids in cell_ids]
    checked = [
        _check_features(features[ct], cids, ct, cell_id_col)
        for ct, cids in zip(cell_types, cell_ids, strict=True)
    ]

    shared = set.intersection(*({samples[i] for i in rows[ct]} for ct in cell_types))
    if len(shared) < DIALOGUE_MIN_SHARED_SAMPLES:
        raise ValueError(
            f"Only {len(shared)} sample(s) are present in every cell type, DIALOGUE "
            f"needs at least {DIALOGUE_MIN_SHARED_SAMPLES}. Check `{sample_col}` and "
            "whether the rarer cell types were captured in every sample."
        )

    sample_levels = sorted({s for s in samples if s is not None})
    code = {s: i for i, s in enumerate(sample_levels)}
    n_slots = int(np.max(np.concatenate(indices))) + 1
    sample_ids = np.zeros(n_slots, dtype=np.uint32)
    cell_quality = np.zeros(n_slots, dtype=np.float64)
    for ct, idx in zip(cell_types, indices, strict=True):
        r = rows[ct]
        sample_ids[idx] = [code[str(samples[i])] for i in r]
        cell_quality[idx] = quality[r]

    return _Prepped(
        cell_type_indices=indices,
        features=[m for m, _ in checked],
        feature_names=[c for _, c in checked],
        sample_ids=sample_ids,
        cell_quality=cell_quality,
        cell_types=cell_types,
        cell_ids=cell_ids,
        sample_levels=sample_levels,
    )


@beartype
def _resolve_dialogue_genes(sc: ScOrMc, gene_ids: Sequence[str] | None) -> np.ndarray:
    """R's `.resolve_dialogue_genes`: the given genes, else the HVGs.

    Raises:
        ValueError: If that leaves no gene.
    """
    if gene_ids is None:
        hvg = sc.hvg if isinstance(sc, MetaCells) else sc.map.hvg
        idx = np.empty(0, dtype=np.uint32) if hvg is None else hvg
    else:
        idx = sc.gene_indices(list(gene_ids))
    if len(idx) == 0:
        raise ValueError(
            "No genes to run DIALOGUE on. Either run find_hvg() first or pass "
            "`gene_ids` explicitly."
        )
    return np.asarray(idx, dtype=np.uint32)


@beartype
def _named_rows(
    m: np.ndarray, id_name: str, ids: Sequence[str], mcp: list[str]
) -> pl.DataFrame:
    """A matrix as a frame: an id column, then one column per programme."""
    return pl.DataFrame(
        {id_name: list(ids), **{c: m[:, j] for j, c in enumerate(mcp)}},
        schema_overrides={id_name: pl.String},
    )


@beartype
def _signature_frame(
    sigs: list, cell_types: list[str], genes: np.ndarray, list_name: str
) -> pl.DataFrame:
    """R's `.dialogue_signature_dt`: the `[cell_type][programme]` lists flat.

    `sigs` is a plain list because it is nested FFI output.
    """
    rows: dict[str, list] = {
        "cell_type": [],
        "programme": [],
        "gene_id": [],
        "direction": [],
    }
    for ct, per_type in zip(cell_types, sigs, strict=True):
        for p, sig in enumerate(per_type, start=1):
            for direction in ("up", "down"):
                g = sig[direction]
                rows["cell_type"] += [ct] * len(g)
                rows["programme"] += [p] * len(g)
                rows["gene_id"] += genes[g].tolist()
                rows["direction"] += [direction] * len(g)
    return pl.DataFrame(
        {**rows, "list": [list_name] * len(rows["gene_id"])},
        schema={
            "cell_type": pl.String,
            "programme": pl.Int32,
            "gene_id": pl.String,
            "direction": pl.String,
            "list": pl.String,
        },
    )


@beartype
def _dialogue_result(
    res: dict,
    prepped: _Prepped,
    genes: np.ndarray,
    source_class: str,
    params: dict[str, object],
) -> DialogueResult:
    """R's `new_dialogue_result`: every index mapped back to its name.

    `res` is a plain dict because it is heterogeneous FFI output.
    """
    cell_types = prepped.cell_types
    k = res["emp_p"].shape[0]
    mcp = [f"mcp_{i + 1:02d}" for i in range(k)]
    # Rust orders the pair columns as R's `combn(.., 2)` does
    pairs = list(itertools.combinations(cell_types, 2))
    programmes = pl.DataFrame(
        {
            "programme": list(range(1, k + 1)) * len(pairs),
            "cell_type_a": [a for a, _ in pairs for _ in range(k)],
            "cell_type_b": [b for _, b in pairs for _ in range(k)],
            "emp_p": res["emp_p"].T.ravel(),
            "pair_cor": res["pair_cor"].T.ravel(),
        },
        schema_overrides={
            "programme": pl.Int32,
            "cell_type_a": pl.String,
            "cell_type_b": pl.String,
        },
    )

    def scores(mats: list[np.ndarray]) -> dict[str, pl.DataFrame]:
        return {
            ct: _named_rows(m, "cell_id", ids, mcp)
            for ct, m, ids in zip(cell_types, mats, prepped.cell_ids, strict=True)
        }

    kept = {
        ct: tuple(names[i] for i in kf.tolist())
        for ct, names, kf in zip(
            cell_types, prepped.feature_names, res["kept_features"], strict=True
        )
    }
    v = res["verdicts"]
    verdicts = pl.DataFrame(
        {
            "cell_type": [cell_types[i] for i in v["cell_type"].tolist()],
            "programme": (v["programme"].astype(np.int64) + 1).tolist(),
            "gene_id": genes[v["gene"]].tolist(),
            "up": v["up"],
            "n_supporting": v["n_supporting"].astype(np.int64),
            "support_fraction": v["support_fraction"],
            "p_up": v["p_up"],
            "p_down": v["p_down"],
            "coefficient": v["coefficient"],
        },
        schema_overrides={
            "cell_type": pl.String,
            "programme": pl.Int32,
            "gene_id": pl.String,
            "up": pl.Boolean,
            "n_supporting": pl.Int32,
            "support_fraction": pl.Float64,
            "p_up": pl.Float64,
            "p_down": pl.Float64,
            "coefficient": pl.Float64,
        },
    )
    return DialogueResult(
        programmes=programmes,
        mcp_cell_types=tuple(
            tuple(cell_types[i] for i in t.tolist()) for t in res["mcp_cell_types"]
        ),
        scores=scores(res["scores"]),
        cca_scores=scores(res["cca_scores"]),
        refit_fidelity=_named_rows(res["refit_fidelity"], "cell_type", cell_types, mcp),
        ws={
            ct: _named_rows(m, "feature", kept[ct], mcp)
            for ct, m in zip(cell_types, res["ws"], strict=True)
        },
        kept_features=kept,
        verdicts=verdicts,
        signatures=pl.concat(
            [
                _signature_frame(res["permissive"], cell_types, genes, "permissive"),
                _signature_frame(res["strict"], cell_types, genes, "strict"),
            ]
        ),
        shared_samples=tuple(
            prepped.sample_levels[i] for i in res["shared_samples"].tolist()
        ),
        cell_types=tuple(cell_types),
        source_class=source_class,
        params=params,
    )


########
# Main #
########


@beartype
def dialogue(
    sc: ScOrMc,
    cell_type_col: str,
    sample_col: str,
    features: Mapping[str, pl.DataFrame],
    *,
    quality_col: str | None = None,
    gene_ids: Sequence[str] | None = None,
    pmd_params: DialoguePmdParams | None = None,
    hlm_params: DialogueHlmParams | None = None,
    refine_params: DialogueRefineParams | None = None,
    verbose: bool = False,
) -> DialogueResult:
    """Multicellular programmes, as R's `dialogue_sc`.

    Three stages. Each cell type's features are collapsed to one row per
    sample and put through a sparse multi-CCA, giving every programme a
    weight vector per cell type plus a provisional gene signature. For every
    ordered pair of cell types and every candidate gene, a mixed model asks
    whether a cell's programme score tracks the partner cell type's
    expression of that gene in the same sample. The partners are then
    meta-analysed and the scores refit onto the surviving genes by
    non-negative least squares. Reads the normalised counts only.

    The method needs at least two cell types, five samples present in every
    one of them, and enough cells per sample per cell type to clear `abn_c`.
    For meta cells, build them within samples, or the sample of a meta cell
    is not well-defined.

    Args:
        sc: The kept cells of the experiment, a subset, or meta cells.
        cell_type_col: obs column with the cell type labels.
        sample_col: obs column with the sample labels; the random intercept
            of stage two is over these.
        features: Per analysed cell type (the keys, in order) a frame of
            `cell_id` (`meta_cell_id` for meta cells) plus at least two
            numeric feature columns covering that cell type's cells.
        quality_col: obs column with the cell quality covariate. `None` uses
            the z-scored `log1p` library size (`lib_size` in obs; the summed
            raw counts for meta cells).
        gene_ids: Genes to build the signatures from. `None` uses the HVGs.
        pmd_params: Stage one parameters. `None` uses the defaults.
        hlm_params: Stage two parameters. `None` uses the defaults.
        refine_params: Stage three parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The `DialogueResult`.

    Raises:
        ValueError: On a design DIALOGUE cannot run on, malformed features,
            or no genes.

    References:
        Jerby-Arnon & Regev, Nat. Biotechnol., 2022.
    """
    pmd = pmd_params or DialoguePmdParams()
    hlm = hlm_params or DialogueHlmParams()
    refine = refine_params or DialogueRefineParams()

    if isinstance(sc, MetaCells):
        obs = sc.obs()
        id_col = "meta_cell_id"
        # meta cells carry no `lib_size`, but the raw counts are in memory
        rows = np.repeat(np.arange(sc.n_cells), np.diff(sc.raw.indptr))
        default_quality = np.bincount(
            rows, weights=sc.raw.data.astype(np.float64), minlength=sc.n_cells
        )
        source = "MetaCells"
    else:
        obs = sc.obs(filtered=True)
        id_col = "cell_id"
        if quality_col is None and "lib_size" not in obs.columns:
            raise ValueError(
                "No `lib_size` column in the obs table to derive the cell quality "
                "covariate from. Pass `quality_col` instead."
            )
        default_quality = (
            obs["lib_size"].cast(pl.Float64).to_numpy()
            if "lib_size" in obs.columns
            else np.zeros(obs.height)
        )
        source = "SingleCells" if isinstance(sc, SingleCells) else "SingleCellsSubset"

    prepped = _prep_dialogue_inputs(
        sc,
        obs,
        cell_type_col,
        sample_col,
        features,
        quality_col,
        id_col,
        default_quality,
    )
    genes = _resolve_dialogue_genes(sc, gene_ids)
    args = (
        prepped.cell_type_indices,
        prepped.features,
        prepped.sample_ids,
        prepped.cell_quality,
        genes,
        to_rust(pmd),
        to_rust(hlm),
        to_rust(refine),
        int(verbose),
    )
    if isinstance(sc, MetaCells):
        # DIALOGUE reads the normalised layer only
        res = _bixverse.mc_dialogue(*csr_args(sc.norm), *args)
    else:
        res = _bixverse.dialogue_sc(str(sc.genes_path), *args)

    return _dialogue_result(
        res,
        prepped,
        np.asarray(sc.gene_ids()),
        source,
        {
            "cell_type_col": cell_type_col,
            "sample_col": sample_col,
            "quality_col": quality_col,
            "pmd_params": pmd,
            "hlm_params": hlm,
            "refine_params": refine,
        },
    )
