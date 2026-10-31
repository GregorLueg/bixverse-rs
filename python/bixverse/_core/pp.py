"""Processing: HVGs, PCA, residual models and MAGIC.

The port of R's `find_hvg_sc`, `get_hvg_data_sc`, `find_hvg_batch_aware_sc`,
`calculate_pca_sc`, `fit_residuals_sc`, `sct_corrected_counts_sc` and
`run_magic_sc`, plus the `MetaCells` methods of the first four. The work
runs in Rust over the binary stores, or the in-memory meta cell counts; this
layer
resolves the inputs, writes what R writes (the HVG statistics to var, the
HVG set to `sc.map.hvg`, the PCA, residual model and MAGIC layer to
`sc.cache`) and stamps the cached artefacts as R stamps them.

    >>> find_hvg(sc, hvg_no=2000)
    >>> calculate_pca(sc, 30)

The residual path fits a model once and reuses it for the HVGs and the PCA:

    >>> fit_residuals(sc, method="analytic_pearson")
    >>> find_hvg(sc, hvg_params=HvgParams(method="residual"))
    >>> calculate_pca(
    ...     sc, 30, pca_params=PcaParams(normalise_variance=False), residuals=True
    ... )
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.ingest import _prepare
from bixverse._core.metacells import MetaCells, csr_args
from bixverse._core.multimodal import Modality, modality_cache, require_fresh, stamp
from bixverse._core.qc import _auto_streaming
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN, SingleCells
from bixverse._core.state import MagicLayer
from bixverse._core.subset import ScOrMc, ScOrScSubset
from bixverse.params import (
    AprParams,
    HvgParams,
    MagicParams,
    PcaParams,
    SctransformParams,
    to_rust,
)

###########
# Globals #
###########

# Above this many cells the dense PCA path is swapped for the sparse one.
N_CELLS_DENSE_PCA_LIMIT: int = 500_000

# Cells held in memory per phase and genes per batch when transposing a
# corrected store; R's `.CORRECTED_CELLS_PER_PHASE` / `.CORRECTED_GENE_BATCH`.
CORRECTED_CELLS_PER_PHASE: int = 50_000
CORRECTED_GENE_BATCH: int = 1000

# The residual PCA densifies cells x genes in f64; warn above this many GB.
RESIDUAL_PCA_WARN_GB: float = 8.0

# HVG method to the statistic genes are ranked on, highest first.
HVG_RANK_COL: dict[str, str] = {
    "vst": "var_std",
    "dispersion": "dispersion",
    "meanvarbin": "dispersion_scaled",
    "scran": "scran_residual",
}

ResidualMethod = Literal["sctransform", "analytic_pearson"]
GeneCombMethod = Literal["union", "average", "intersection"]

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True)
class ResidualFit:
    """A fitted residual model, R's `ScResidualFit`.

    Attributes:
        model: The fitted model, held in Rust.
        method: `"sctransform"` or `"analytic_pearson"`.
        params: The parameters the fit ran with.
        group_column: The obs column the cells were grouped by, if any.
        covariate_names: The covariates, in design order.
    """

    model: _bixverse.ResidualFit
    method: ResidualMethod
    params: SctransformParams | AprParams
    group_column: str | None
    covariate_names: tuple[str, ...]

    @property
    def cell_indices(self) -> np.ndarray:
        """0-based store indices of the fitted cells, in fitting order."""
        return self.model.cell_indices

    @property
    def genes(self) -> np.ndarray:
        """0-based store genes modelled in every group, ascending."""
        return self.model.genes

    @property
    def n_groups(self) -> int:
        """Number of groups, one model each."""
        return self.model.n_groups


@beartype
@dataclass(frozen=True)
class BatchHvg:
    """Batch-aware HVGs, R's `find_hvg_batch_aware_sc` result.

    Attributes:
        hvg_genes: Gene ids of the HVGs.
        hvg_gene_idx: Their 0-based gene indices.
        hvg_data: Per batch and gene statistics, with `batch` (the label) and
            `gene_idx` (0-based).
    """

    hvg_genes: np.ndarray
    hvg_gene_idx: np.ndarray
    hvg_data: pl.DataFrame


@beartype
@dataclass(frozen=True)
class CorrectedCounts:
    """A corrected gene-major store written without its cell-major twin.

    Attributes:
        f_path: The gene-major store.
        genes: 0-based source genes written, the new store's gene axis.
        n_genes: Number of genes written.
        n_cells: Number of cells written.
    """

    f_path: Path
    genes: np.ndarray
    n_genes: int
    n_cells: int


###########
# Helpers #
###########


@beartype
def _top(score: np.ndarray, n: int) -> np.ndarray:
    """0-based positions of the `n` highest scores, as R's
    `order(score, decreasing = TRUE)[1:n]`: ties keep their order, NaN last.

    Raises:
        ValueError: If `n` is not in `[1, len(score)]`.
    """
    if not 1 <= n <= score.shape[0]:
        raise ValueError(f"hvg_no must be in [1, {score.shape[0]}], got {n}")
    return np.argsort(-score, kind="stable")[:n].astype(np.uint32)


@beartype
def _factor_codes(values: pl.Series, name: str) -> tuple[np.ndarray, list[str]]:
    """0-based codes of `as.integer(factor(x)) - 1`, with the levels.

    An Enum keeps its level order, minus unused levels; anything else is
    sorted. R sorts strings in the locale's collation, here by code point.
    Float levels print as R's `as.character` does (15 significant digits, so
    `1` rather than `1.0`).

    Raises:
        ValueError: On a missing value.
    """
    if values.null_count():
        raise ValueError(f"Column '{name}' holds missing values.")
    if isinstance(values.dtype, pl.Enum):
        present = set(values.cast(pl.String).to_list())
        levels = [c for c in values.dtype.categories.to_list() if c in present]
        lookup = {v: i for i, v in enumerate(levels)}
        codes = [lookup[v] for v in values.cast(pl.String).to_list()]
    else:
        raw = values.to_list()
        uniq = sorted(set(raw))
        lookup = {v: i for i, v in enumerate(uniq)}
        codes = [lookup[v] for v in raw]
        levels = [f"{v:.15g}" if isinstance(v, float) else str(v) for v in uniq]
    return np.asarray(codes, dtype=np.uint32), levels


@beartype
def _obs_aligned(sc: ScOrScSubset, cols: Sequence[str] | None = None) -> pl.DataFrame:
    """obs rows of the kept cells in `cells_to_keep` order, R's
    `.residual_obs`.

    Labels and covariates go to Rust by position, so the rows have to follow
    the kept cell order. `cols` that are not in obs are left out, so callers
    can name them.

    Raises:
        ValueError: If a kept cell has no obs row.
    """
    if cols is not None:
        present = sc.obs_columns()
        cols = [c for c in dict.fromkeys(["cell_idx", *cols]) if c in present]
    obs = sc.obs(cols=None if cols is None else list(cols), filtered=True)
    order = pl.DataFrame({"cell_idx": sc.cells_to_keep})
    aligned = order.join(obs, on="cell_idx", how="inner", maintain_order="left")
    if aligned.height != order.height:
        raise ValueError(
            "Some selected cells are missing from the observation table. "
            "The object state and the database have drifted apart."
        )
    return aligned


@beartype
def _fit_usable(sc: ScOrScSubset) -> ResidualFit:
    """The cached residual model, if it is safe to compute with, R's
    `.assert_fit_usable`.

    Raises:
        ValueError: If nothing was fitted, or the cell filter moved since.
    """
    fit = sc.cache.residual_fit
    if not isinstance(fit, ResidualFit):
        raise ValueError(
            "No fitted residual model found on the object. Run fit_residuals() first."
        )
    if not np.array_equal(fit.cell_indices, sc.cells_to_keep):
        raise ValueError(
            f"The residual model was fitted on {len(fit.cell_indices)} cells but "
            f"{sc.n_cells} are selected now. The cell filter moved after "
            "fitting; re-run fit_residuals()."
        )
    return fit


@beartype
def _fit_usable_mc(mc: MetaCells) -> ResidualFit:
    """The cached residual model of meta cells, R's `.assert_fit_usable_mc`.

    Meta cells have no cell filter to drift, so this is the presence check.

    Raises:
        ValueError: If nothing was fitted.
    """
    fit = mc.cache.residual_fit
    if not isinstance(fit, ResidualFit):
        raise ValueError(
            "No fitted residual model found on the object. Run fit_residuals() first."
        )
    return fit


@beartype
def _check_not_residual(params: HvgParams, fn: str, hint: str) -> None:
    """Refuse the residual HVG method where it does not apply.

    Raises:
        ValueError: If `params.method` is `"residual"`.
    """
    if params.method == "residual":
        raise ValueError(f"{fn}() does not support method = 'residual'. {hint}")


#######
# HVG #
#######


@beartype
def _hvg_stats(
    sc: ScOrScSubset,
    cell_indices: np.ndarray,
    params: HvgParams,
    streaming: bool | None,
    verbose: bool,
) -> dict[str, np.ndarray]:
    """`rs_sc_hvg` over a cell selection."""
    return _bixverse.hvg(
        str(sc.genes_path),
        str(sc.cells_path),
        cell_indices,
        to_rust(params),
        _auto_streaming(len(cell_indices), streaming),
        verbose,
    )


@beartype
def _find_hvg_residual(sc: ScOrScSubset, hvg_no: int, verbose: bool) -> None:
    """Residual variance HVGs, R's `.find_hvg_residual`.

    Writes `residual_variance` (the mean over groups, NULL for genes the
    model does not cover) to var and the union of the per-group top `hvg_no`
    to `sc.map.hvg`.
    """
    fit = _fit_usable(sc)
    genes, variance, hvg = _bixverse.residual_variance(
        str(sc.genes_path),
        str(sc.cells_path),
        fit.model,
        sc.cells_to_keep,
        hvg_no,
        None,
        verbose,
    )
    # a subset leaves var alone, as R's `write_var = FALSE`
    if isinstance(sc, SingleCells):
        full = np.full(sc.n_genes, np.nan)
        full[genes.astype(np.int64)] = variance.mean(axis=1)
        sc.add_var(pl.DataFrame({"residual_variance": full}).fill_nan(None))
    if verbose and fit.n_groups > 1:
        logger.info(
            "Selected %i variable features: the union of the top %i of each of "
            "%i groups.",
            len(hvg),
            hvg_no,
            fit.n_groups,
        )
    sc.map.hvg = hvg


@beartype
def _find_hvg_mc(mc: MetaCells, hvg_no: int, params: HvgParams, verbose: bool) -> None:
    """`find_hvg` on meta cells, R's `find_hvg_sc` MetaCells method.

    The statistics go to the meta cell var, the selection to `mc.hvg`. The
    residual method writes no statistics, as in R.
    """
    if params.method == "residual":
        fit = _fit_usable_mc(mc)
        _, _, hvg = _bixverse.mc_residual_variance(
            *csr_args(mc.raw), fit.model, hvg_no, verbose
        )
        if verbose and fit.n_groups > 1:
            logger.info(
                "Selected %i variable features across %i groups.",
                len(hvg),
                fit.n_groups,
            )
        mc.hvg = hvg
        return
    m = mc.raw if params.method == "vst" else mc.norm
    res = _bixverse.mc_hvg(*csr_args(m), to_rust(params))
    mc.add_var(pl.DataFrame(res))
    mc.hvg = _top(res[HVG_RANK_COL[params.method]], hvg_no)


@beartype
def find_hvg(
    sc: ScOrMc,
    *,
    hvg_no: int = 2000,
    hvg_params: HvgParams | None = None,
    streaming: bool | None = None,
    verbose: bool = False,
) -> None:
    """Select highly variable genes over the kept cells, as R's `find_hvg_sc`.

    The per-gene statistics go to var under R's column names (`mean`, `var`,
    `var_exp`, `var_std` for `"vst"`; `mean`, `dispersion`,
    `dispersion_scaled`, `bin` for `"meanvarbin"` / `"dispersion"`;
    `scran_mean`, `scran_var`, `scran_fitted`, `scran_residual` for
    `"scran"`), the top `hvg_no` genes on the ranking statistic to
    `sc.map.hvg`. On a `SingleCellsSubset` only the selection is stored, as
    in R.

    `method = "residual"` ranks on the residual variance of the model from
    `fit_residuals`, writes `residual_variance` instead, and treats `hvg_no`
    as a per-group count whose union is kept.

    On `MetaCells` the statistics come from the in-memory counts (raw for
    `"vst"`, normalised otherwise) and go to its var, the selection to
    `mc.hvg`; the residual method writes no statistics there. `streaming` is
    ignored.

    Args:
        sc: The experiment.
        hvg_no: Number of HVGs (per group for `"residual"`).
        hvg_params: The HVG parameters. `None` uses the defaults.
        streaming: Read the genes in batches. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` kept cells.
        verbose: Controls verbosity.

    Raises:
        ValueError: If `hvg_no` exceeds the gene count, or the residual model
            is missing or stale.
    """
    params = hvg_params or HvgParams()
    if isinstance(sc, MetaCells):
        _find_hvg_mc(sc, hvg_no, params, verbose)
        return
    if params.method == "residual":
        _find_hvg_residual(sc, hvg_no, verbose)
        return
    res = _hvg_stats(sc, sc.cells_to_keep, params, streaming, verbose)
    hvg = _top(res[HVG_RANK_COL[params.method]], hvg_no)
    if isinstance(sc, SingleCells):
        sc.add_var(pl.DataFrame(res))
    sc.map.hvg = hvg


@beartype
def get_hvg_data(
    sc: SingleCells | MetaCells,
    *,
    cell_ids: Sequence[str] | None = None,
    hvg_no: int = 3000,
    hvg_params: HvgParams | None = None,
    streaming: bool | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """HVG statistics without touching the experiment, as R's
    `get_hvg_data_sc`.

    Args:
        sc: The experiment, or meta cells.
        cell_ids: The cells (meta cell ids on `MetaCells`) to compute over.
            `None` for the kept cells, or every meta cell.
        hvg_no: Number of genes to flag.
        hvg_params: The HVG parameters; `"residual"` is refused, it needs a
            model fitted on a fixed cell set.
        streaming: Read the genes in batches. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` cells.
        verbose: Controls verbosity.

    Returns:
        `gene_idx` (0-based), `gene_id`, the statistics, `is_hvg` and
        `hvg_rank` (1 for the most variable, null outside the HVGs).

    Raises:
        ValueError: On method `"residual"` or an `hvg_no` past the gene
            count.
    """
    params = hvg_params or HvgParams()
    _check_not_residual(
        params,
        "get_hvg_data",
        "Use fit_residuals() followed by find_hvg().",
    )
    if isinstance(sc, MetaCells):
        rows = None if cell_ids is None else sc.cell_indices(list(cell_ids))
        layer = "raw" if params.method == "vst" else "norm"
        res = _bixverse.mc_hvg(
            *csr_args(sc.counts(cells=rows, layer=layer)), to_rust(params)
        )
    else:
        cells = (
            sc.cells_to_keep if cell_ids is None else sc.cell_indices(list(cell_ids))
        )
        res = _hvg_stats(sc, cells, params, streaming, verbose)
    hvg = _top(res[HVG_RANK_COL[params.method]], hvg_no)
    rank = np.full(sc.n_genes, None, dtype=object)
    rank[hvg.astype(np.int64)] = np.arange(1, hvg_no + 1)
    return sc.var(cols=["gene_idx", "gene_id"]).with_columns(
        *[pl.Series(k, v) for k, v in res.items()],
        pl.Series("is_hvg", np.isin(np.arange(sc.n_genes), hvg)),
        pl.Series("hvg_rank", rank.tolist(), dtype=pl.Int32),
    )


@beartype
def _combine_batches(
    data: pl.DataFrame, score: str, hvg_no: int, how: GeneCombMethod
) -> np.ndarray:
    """Combine per-batch rankings into one gene set, as R does with
    data.table: batches and genes in first-appearance order, ties stable.
    """

    def top_genes(frame: pl.DataFrame, col: str) -> list[int]:
        pos = np.argsort(-frame[col].to_numpy(), kind="stable")[:hvg_no]
        return frame["gene_idx"].to_numpy()[pos].tolist()

    batches = data.partition_by("batch", maintain_order=True)
    match how:
        case "union":
            genes = list(dict.fromkeys(g for b in batches for g in top_genes(b, score)))
        case "average":
            avg = data.group_by("gene_idx", maintain_order=True).agg(
                pl.col(score).mean().alias("score_avg")
            )
            genes = top_genes(avg, "score_avg")
        case "intersection":
            tops = [top_genes(b, score) for b in batches]
            counts: dict[int, int] = {}
            for top in tops:
                for g in top:
                    counts[g] = counts.get(g, 0) + 1
            genes = [g for g, n in counts.items() if n == len(batches)]
        case _:
            raise ValueError(f"Unknown gene_comb_method '{how}'")
    return np.asarray(genes, dtype=np.uint32)


@beartype
def find_hvg_batch_aware(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    hvg_no: int = 2000,
    gene_comb_method: GeneCombMethod = "union",
    hvg_params: HvgParams | None = None,
    streaming: bool | None = None,
    verbose: bool = False,
) -> BatchHvg:
    """HVGs per batch, combined across batches, as R's
    `find_hvg_batch_aware_sc`.

    Does not touch the experiment: pass `hvg_gene_idx` to `calculate_pca`.

    Args:
        sc: The experiment.
        batch_column: obs column holding the batch.
        hvg_no: HVGs per batch (`"union"`, `"intersection"`) or in total
            (`"average"`).
        gene_comb_method: `"union"` of the per-batch top genes, top genes of
            the `"average"` score, or the `"intersection"` of the per-batch
            top genes.
        hvg_params: The HVG parameters; `"residual"` is refused, fit per
            batch with `fit_residuals(group_column=...)` instead.
        streaming: Read the genes in batches. `None` streams above
            `N_CELLS_STREAMING_THRESHOLD` kept cells.
        verbose: Controls verbosity.

    Returns:
        The HVGs and the per-batch statistics.

    Raises:
        ValueError: On method `"residual"`, or a missing or incomplete batch
            column.
    """
    params = hvg_params or HvgParams()
    _check_not_residual(
        params,
        "find_hvg_batch_aware",
        "Fit per batch with fit_residuals(group_column=...), then find_hvg().",
    )
    obs = _obs_aligned(sc, [batch_column])
    if batch_column not in obs.columns:
        raise ValueError(f"Column '{batch_column}' not found in the obs table.")
    codes, levels = _factor_codes(obs[batch_column], batch_column)
    res = _bixverse.hvg_batch_aware(
        str(sc.genes_path),
        str(sc.cells_path),
        sc.cells_to_keep,
        codes,
        to_rust(params),
        _auto_streaming(sc.n_cells, streaming),
        verbose,
    )
    batch = np.asarray(levels, dtype=object)[res["batch"]]
    data = pl.DataFrame(res).with_columns(
        pl.Series("batch", batch.tolist(), dtype=pl.String)
    )
    genes = _combine_batches(
        data, HVG_RANK_COL[params.method], hvg_no, gene_comb_method
    )
    return BatchHvg(hvg_genes=sc.gene_ids(genes), hvg_gene_idx=genes, hvg_data=data)


#######
# PCA #
#######


@beartype
def _check_residual_pca_params(params: PcaParams) -> None:
    """Refuse PCA settings the residual path cannot honour, R's
    `.assert_residual_pca_params`.

    Raises:
        ValueError: On `clr` or `normalise_variance`.
    """
    if params.clr:
        raise ValueError(
            "Residual PCA cannot apply the PFlogPF transformation, which belongs "
            "to the normalised layer. Pass PcaParams(clr=False)."
        )
    if params.normalise_variance:
        raise ValueError(
            "Residual PCA cannot normalise the variance: the residuals already "
            "carry the biological signal as variance, and scaling it away is the "
            "one thing the transform exists to avoid. Pass "
            "PcaParams(normalise_variance=False)."
        )


@beartype
def _calculate_pca_residual(
    sc: ScOrScSubset,
    no_pcs: int,
    params: PcaParams,
    selected: np.ndarray,
    sparse_svd: bool,
    seed: int,
    verbose: bool,
) -> None:
    """PCA on residuals, R's `.calculate_pca_residual`."""
    if sparse_svd:
        raise ValueError(
            "Residual PCA has no sparse solver: a residual column is dense even "
            "where the counts are not. Pass sparse_svd=False."
        )
    _check_residual_pca_params(params)
    fit = _fit_usable(sc)

    genes = np.intersect1d(selected, fit.genes).astype(np.uint32)
    if genes.size == 0:
        raise ValueError(
            "None of the requested genes are covered by the fitted residual "
            "model. The model only covers genes detected in every group."
        )
    if genes.size < selected.size and verbose:
        logger.info(
            "Dropping %i of %i genes not covered by the residual model. They were "
            "filtered out in at least one group.",
            selected.size - genes.size,
            selected.size,
        )
    gb = sc.n_cells * genes.size * 8 / 1024**3
    if gb > RESIDUAL_PCA_WARN_GB:
        warnings.warn(
            f"Residual PCA densifies the matrix: {sc.n_cells} cells by "
            f"{genes.size} genes is about {gb:.1f} GB. Reduce the HVG count or "
            "subset the cells if that is too much for this machine.",
            stacklevel=3,
        )

    scores, loadings, sv = _bixverse.pca_residuals(
        str(sc.genes_path),
        str(sc.cells_path),
        fit.model,
        sc.cells_to_keep,
        genes,
        no_pcs,
        to_rust(params),
        seed,
        verbose,
    )
    _set_pca(sc, scores, loadings, sv[:no_pcs], parents=("residuals",))


@beartype
def _set_pca(
    sc: ScOrMc,
    scores: np.ndarray,
    loadings: np.ndarray,
    singular_vals: np.ndarray,
    *,
    parents: tuple[str, ...] = (),
) -> None:
    """Cache a PCA and stamp it, as R's `set_pca_*` chain."""
    sc.cache.pca_factors = scores
    sc.cache.pca_loadings = loadings
    sc.cache.pca_singular_vals = singular_vals
    sc.stamp("pca", parents=parents)


@beartype
def _calculate_pca_mc(
    mc: MetaCells,
    no_pcs: int,
    params: PcaParams,
    residuals: bool,
    seed: int,
    verbose: bool,
) -> None:
    """`calculate_pca` on meta cells, R's `calculate_pca_sc` MetaCells method.

    Dense, in memory; the CLR offsets come off the full raw panel.
    """
    selected = mc.hvg
    assert selected is not None
    if residuals:
        _check_residual_pca_params(params)
        fit = _fit_usable_mc(mc)
        genes = np.intersect1d(selected, fit.genes).astype(np.uint32)
        if genes.size == 0:
            raise ValueError(
                "None of the requested genes are covered by the fitted residual "
                "model. The model only covers genes detected in every group."
            )
        if genes.size < selected.size and verbose:
            logger.info(
                "Dropping %i of %i genes not covered by the residual model.",
                selected.size - genes.size,
                selected.size,
            )
        scores, loadings, sv = _bixverse.mc_pca_residuals(
            *csr_args(mc.raw), fit.model, genes, no_pcs, to_rust(params), seed, verbose
        )
        _set_pca(mc, scores, loadings, sv[:no_pcs], parents=("residuals",))
        return
    offsets = _clr_offsets(mc) if params.clr else None
    scores, loadings, sv = _bixverse.mc_pca(
        *csr_args(mc.counts(genes=selected, layer="norm")),
        no_pcs,
        to_rust(params),
        offsets,
        seed,
    )
    _set_pca(mc, scores, loadings, sv[:no_pcs])


@beartype
def _clr_offsets(mc: MetaCells) -> np.ndarray:
    """Per meta cell `mean(log1p(x / rowSum(x)))` over every gene of the raw
    counts, R's `mc_get_clr_offsets`."""
    m = mc.raw
    n_rows, n_genes = m.shape
    lengths = np.diff(m.indptr.astype(np.int64))
    row_of = np.repeat(np.arange(n_rows), lengths)
    data = m.data.astype(np.float64)
    totals = np.bincount(row_of, weights=data, minlength=n_rows)
    logs = np.log1p(data / totals[row_of])
    return np.bincount(row_of, weights=logs, minlength=n_rows) / n_genes


@beartype
def calculate_pca(
    sc: ScOrMc,
    no_pcs: int,
    *,
    pca_params: PcaParams | None = None,
    sparse_svd: bool | None = None,
    hvg: np.ndarray | Sequence[int] | Sequence[str] | None = None,
    seed: int = 42,
    residuals: bool = False,
    verbose: bool = False,
) -> None:
    """PCA over the kept cells and the HVGs, as R's `calculate_pca_sc`.

    Scores, loadings and the first `no_pcs` singular values go to `sc.cache`,
    stamped `"pca"` (derived from `"residuals"` on the residual path).

    Args:
        sc: The experiment.
        no_pcs: Number of components.
        pca_params: The PCA parameters. `None` uses the defaults.
        sparse_svd: Solve on the sparse matrix. `None` means `True` on the
            normalised path and `False` on the residual one; above
            `N_CELLS_DENSE_PCA_LIMIT` cells the normalised path is always
            sparse.
        hvg: 0-based gene indices or gene ids to use instead of
            `sc.map.hvg`; they replace it.
        seed: Seed for the randomised SVD.
        residuals: Run on the residuals of the model from `fit_residuals`.
            Dense only, and needs `clr` and `normalise_variance` off.
        verbose: Controls verbosity.

    On `MetaCells` the PCA is dense over the in-memory normalised counts
    (raw for the residual path) and `sparse_svd` is ignored, as in R.

    Raises:
        ValueError: On a residual run that cannot be honoured, or a missing
            or stale residual model.
    """
    params = pca_params or PcaParams()
    if isinstance(sc, MetaCells):
        if sc.hvg is None and hvg is None:
            warnings.warn(
                "No HVGs identified in the object nor provided. Please run "
                "find_hvg() or provide the indices of the HVG. Returning object "
                "as is.",
                stacklevel=2,
            )
            return
        if hvg is not None:
            if verbose:
                logger.info("HVGs provided. Setting the internal HVG to them.")
            sc.hvg = sc._resolve_genes(hvg)
        _calculate_pca_mc(sc, no_pcs, params, residuals, seed, verbose)
        return
    if sc.map.hvg is None and hvg is None:
        warnings.warn(
            "No HVGs identified in the object nor provided. Please run find_hvg() "
            "or provide the indices of the HVG. Returning object as is.",
            stacklevel=2,
        )
        return
    if hvg is not None:
        if verbose:
            logger.info("HVGs provided. Setting the internal HVG to them.")
        sc.map.hvg = sc._resolve_genes(hvg)
    selected = sc.map.hvg
    assert selected is not None

    if residuals:
        _calculate_pca_residual(
            sc, no_pcs, params, selected, sparse_svd is True, seed, verbose
        )
        return

    sparse = True if sparse_svd is None else sparse_svd
    if sc.n_cells > N_CELLS_DENSE_PCA_LIMIT and not sparse:
        logger.info(
            "More than %i cells with sparse_svd=False. Setting it to True to avoid "
            "high memory pressure.",
            N_CELLS_DENSE_PCA_LIMIT,
        )
        sparse = True
    if verbose:
        logger.info(
            "Using %s SVD solving on scaled data on %i HVG.",
            "sparse" if sparse else "dense",
            selected.size,
        )
    scores, loadings, sv = _bixverse.pca(
        str(sc.genes_path),
        str(sc.cells_path),
        sc.cells_to_keep,
        selected,
        no_pcs,
        to_rust(params),
        sparse,
        seed,
        verbose,
    )
    _set_pca(sc, scores, loadings, sv[:no_pcs])


#############
# Residuals #
#############


@beartype
def _covariates(
    obs: pl.DataFrame, columns: Sequence[str], method: ResidualMethod
) -> list[tuple[str, np.ndarray]]:
    """Numeric covariate columns, R's `.residual_covariates`.

    Raises:
        ValueError: For analytic Pearson, a missing, non-numeric or
            incomplete column.
    """
    if not columns:
        return []
    if method != "sctransform":
        raise ValueError(
            "Covariates are only supported for method = 'sctransform'. The "
            "analytic Pearson model has no design matrix."
        )
    missing = [c for c in columns if c not in obs.columns]
    if missing:
        raise ValueError(
            "Covariate columns not found in the observation table: "
            f"{', '.join(missing)}."
        )
    out = []
    for col in columns:
        values = obs[col]
        if not values.dtype.is_numeric():
            raise ValueError(
                f"Covariate '{col}' is not numeric. Dummy code it yourself and "
                "pass the resulting columns; bixverse will not guess a contrast "
                "for you."
            )
        arr = values.cast(pl.Float64).fill_null(np.nan).to_numpy()
        if not np.isfinite(arr).all():
            raise ValueError(f"Covariate '{col}' has missing or non-finite values.")
        out.append((col, np.ascontiguousarray(arr)))
    return out


@beartype
def _fit_residuals_mc(
    mc: MetaCells,
    method: ResidualMethod,
    params: SctransformParams | AprParams,
    group_column: str | None,
    covariate_columns: list[str],
    seed: int,
    verbose: bool,
) -> None:
    """`fit_residuals` on meta cells, R's `fit_residuals_sc` MetaCells
    method: the in-memory raw counts, obs in row order."""
    obs = mc.obs()
    groups = None
    if group_column is not None:
        if group_column not in obs.columns:
            raise ValueError(
                f"Column '{group_column}' not found in the observation table."
            )
        groups, _ = _factor_codes(obs[group_column], group_column)
    covariates = _covariates(obs, covariate_columns, method)
    model = _bixverse.mc_fit_residuals(
        *csr_args(mc.raw),
        method,
        groups,
        covariates,
        to_rust(params),
        seed,
        verbose,
    )
    if verbose:
        logger.info(
            "Fitted %i model(s) over %i genes.", model.n_groups, len(model.genes)
        )
    mc.cache.residual_fit = ResidualFit(
        model=model,
        method=method,
        params=params,
        group_column=group_column,
        covariate_names=tuple(c for c, _ in covariates),
    )
    mc.stamp("residuals")


@beartype
def fit_residuals(
    sc: ScOrMc,
    *,
    method: ResidualMethod = "sctransform",
    group_column: str | None = None,
    covariate_columns: Sequence[str] | None = None,
    residual_params: SctransformParams | AprParams | None = None,
    gene_batch_size: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Fit a scTransform (v2) or analytic Pearson residual model over the
    kept cells, as R's `fit_residuals_sc`.

    The model goes to `sc.cache.residual_fit`, stamped `"residuals"`, and is
    what `find_hvg(method="residual")`, `calculate_pca(residuals=True)` and
    `sct_corrected_counts` read. It is bound to the current cell set.

    Args:
        sc: The experiment.
        method: `"sctransform"` or `"analytic_pearson"`.
        group_column: obs column to fit one model per group by (sample,
            donor), Seurat v5 style.
        covariate_columns: Numeric obs columns to regress out
            (`"sctransform"` only). The library size is never one: it is the
            model's offset.
        residual_params: `SctransformParams` or `AprParams` matching
            `method`. `None` uses the defaults.
        gene_batch_size: Genes per disk batch. `None` uses the crate default.
        seed: Seed for the step-1 subsampling.
        verbose: Controls verbosity.

    On `MetaCells` the model is fitted on the summed counts (still negative
    binomial, at a far greater depth; `"analytic_pearson"` is the better
    choice there) with obs in row order; `gene_batch_size` does not apply and
    warns.

    Raises:
        TypeError: If `residual_params` does not match `method`.
        ValueError: On a missing group column or unusable covariates.
    """
    if isinstance(sc, MetaCells):
        if gene_batch_size is not None:
            warnings.warn(
                "`gene_batch_size` does not apply to meta cells, whose counts are "
                "already in memory. Ignoring it.",
                stacklevel=2,
            )
        expected = SctransformParams if method == "sctransform" else AprParams
        params = residual_params or expected()
        if not isinstance(params, expected):
            raise TypeError(
                f"method '{method}' takes {expected.__name__}, got "
                f"{type(params).__name__}"
            )
        _fit_residuals_mc(
            sc,
            method,
            params,
            group_column,
            list(covariate_columns or []),
            seed,
            verbose,
        )
        return
    if sc.n_cells == 0:
        warnings.warn(
            "No cells kept in the object. Run set_cells_to_keep() first. "
            "Returning object as is.",
            stacklevel=2,
        )
        return
    expected = SctransformParams if method == "sctransform" else AprParams
    params = residual_params or expected()
    if not isinstance(params, expected):
        raise TypeError(
            f"method '{method}' takes {expected.__name__}, got {type(params).__name__}"
        )

    covariate_columns = list(covariate_columns or [])
    obs = _obs_aligned(
        sc, [c for c in (group_column, *covariate_columns) if c is not None]
    )
    groups = None
    if group_column is not None:
        if group_column not in obs.columns:
            raise ValueError(
                f"Column '{group_column}' not found in the observation table."
            )
        groups, _ = _factor_codes(obs[group_column], group_column)
    covariates = _covariates(obs, covariate_columns, method)

    if verbose:
        logger.info(
            "Fitting %s over %i cells%s.",
            method,
            sc.n_cells,
            "" if groups is None else f" in {int(groups.max()) + 1} groups",
        )
    model = _bixverse.fit_residuals(
        str(sc.genes_path),
        str(sc.cells_path),
        method,
        sc.cells_to_keep,
        groups,
        covariates,
        to_rust(params),
        gene_batch_size,
        seed,
        verbose,
    )
    if verbose:
        logger.info(
            "Fitted %i model(s) over %i genes.", model.n_groups, len(model.genes)
        )
    sc.cache.residual_fit = ResidualFit(
        model=model,
        method=method,
        params=params,
        group_column=group_column,
        covariate_names=tuple(c for c, _ in covariates),
    )
    sc.stamp("residuals")


@beartype
def sct_corrected_counts(
    sc: ScOrScSubset,
    *,
    dir_out: Path | str | None = None,
    build_cell_store: bool = True,
    overwrite: bool = False,
    gene_batch_size: int | None = None,
    verbose: bool = False,
) -> SingleCells | CorrectedCounts:
    """scTransform corrected counts as a new experiment, as R's
    `sct_corrected_counts_sc`.

    The counts each kept cell would have at the median library size under
    the fitted model, written to a fresh directory. The gene axis is the
    model's, so the new var table holds only the genes that survived, and
    the obs table the kept cells in order.

    Args:
        sc: The experiment, with a scTransform fit.
        dir_out: Output directory. `None` uses `sct_corrected` inside the
            experiment directory.
        build_cell_store: Also build the cell-major store and the DuckDB and
            open the result. `False` writes only the gene-major store.
        overwrite: Replace an existing store at `dir_out`.
        gene_batch_size: Genes per disk batch. `None` uses the crate default.
        verbose: Controls verbosity.

    Returns:
        The new experiment, or with `build_cell_store=False` what was
        written.

    Raises:
        FileExistsError: If a store exists at `dir_out` and `overwrite` is
            false.
        ValueError: Without a usable scTransform fit.
    """
    fit = _fit_usable(sc)
    if fit.method != "sctransform":
        raise ValueError(
            "Corrected counts need a scTransform fit; the analytic Pearson model "
            "has no corrected-count equivalent. Refit with method='sctransform'."
        )
    out = sc.dir / "sct_corrected" if dir_out is None else Path(dir_out).expanduser()
    f_path_gene = out / GENES_BIN
    if f_path_gene.exists() and not overwrite:
        raise FileExistsError(
            f"A store already exists at '{out}'. Pass overwrite=True to replace it."
        )
    out = _prepare(out)
    genes = _bixverse.sct_corrected_counts(
        str(sc.genes_path),
        fit.model,
        sc.cells_to_keep,
        str(f_path_gene),
        gene_batch_size,
        verbose,
    )
    if not build_cell_store:
        return CorrectedCounts(
            f_path=f_path_gene, genes=genes, n_genes=len(genes), n_cells=sc.n_cells
        )

    if verbose:
        logger.info("Building the cell-major companion store.")
    _bixverse.gene_store_to_cell_store(
        str(f_path_gene),
        str(out / CELLS_BIN),
        CORRECTED_CELLS_PER_PHASE,
        gene_batch_size or CORRECTED_GENE_BATCH,
        verbose,
    )
    # both axes are renumbered, and the old indices would point into the
    # source store
    obs = _obs_aligned(sc).drop("cell_idx")
    var = sc.var(indices=genes).drop("gene_idx")
    with db.connect(db.db_path(out)) as con:
        db.populate_from_frame(con, "obs", obs)
        db.populate_from_frame(con, "var", var)
        db.set_to_keep_column(con)
    return SingleCells(out)


#########
# MAGIC #
#########


@beartype
def run_magic(
    sc: ScOrScSubset,
    features: Sequence[str],
    *,
    modality: Modality = "rna",
    magic_params: MagicParams | None = None,
    verbose: bool = False,
) -> None:
    """Impute genes with MAGIC over the cached kNN graph, as R's
    `run_magic_sc`.

    The imputed layer is RNA whichever graph smoothed it, so it goes to
    `sc.cache.magic`, stamped as derived from the modality's `"knn"`. Do not
    feed it to correlation-based methods: the smoothing inflates gene-gene
    correlation.

    Args:
        sc: The experiment, with a kNN graph over the kept cells.
        features: Gene ids to impute. Unknown ids warn and drop.
        modality: The kNN graph: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
        magic_params: The MAGIC parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the kNN graph is stale or does not cover the kept
            cells.

    References:
        van Dijk, et al., Cell, 2018.
    """
    params = magic_params or MagicParams()
    require_fresh(sc, ["knn"], modality)
    knn = modality_cache(sc, modality).knn
    if knn is None:
        warnings.warn(
            "No kNN data found. Did you run find_neighbours()? Returning the "
            "object as is.",
            stacklevel=2,
        )
        return

    genes = sc.gene_indices(list(features))
    if knn.indices.shape[0] != sc.n_cells:
        raise ValueError(
            f"The kNN graph covers {knn.indices.shape[0]} cells but {sc.n_cells} "
            "are kept. Regenerate the neighbours so the two agree."
        )
    if verbose:
        logger.info(
            "Running MAGIC over the %s kNN graph (%i cells, %i genes, %i steps).",
            modality,
            sc.n_cells,
            genes.size,
            params.n_steps,
        )
    imputed = _bixverse.magic_impute(
        str(sc.genes_path),
        np.ascontiguousarray(knn.indices),
        np.ascontiguousarray(knn.distances),
        sc.cells_to_keep,
        genes,
        to_rust(params),
        verbose,
    )
    sc.cache.magic = MagicLayer(data=imputed, features=sc.gene_ids(genes))
    stamp(sc, "magic", parents=(f"{modality}:knn",))
