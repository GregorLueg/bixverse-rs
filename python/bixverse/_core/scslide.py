"""scSLIDE: one vector per sample from landmark densities, then a diffusion map
over the samples.

`run_scslide` sketches every sample, fits a PLS against a sample-level
phenotype, picks landmark cells by leverage score and counts each cell against
its nearest landmarks (WNN over the PCA and the PLS). `sample_diffusion_map`
turns the landmark x sample counts into sample coordinates.

    >>> res = run_scslide(sc, "donor", "disease", cell_type_col="cell_type")
    >>> dm = sample_diffusion_map(res)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import BeartypeConf, beartype

from bixverse import _bixverse
from bixverse._core.graph import _embedding, _first_dims
from bixverse._core.integrate import _hard_state
from bixverse._core.pp import _factor_codes, _obs_aligned
from bixverse._core.repr import compact_repr
from bixverse._core.subset import ScOrScSubset
from bixverse._validate import Verbose
from bixverse.params import to_rust
from bixverse.params._checks import check_range

_CONF = BeartypeConf(is_pep484_tower=True)

logger = logging.getLogger(__name__)


##########
# Params #
##########


@beartype(conf=_CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class ScSlideParams:
    """Parameters for `run_scslide`.

    Defaults follow Seurat's `PrepareSampleObject`, except `n_comp`, which
    takes the vignette's 20.

    Attributes:
        n_cells_per_donor: Training sketch, cells drawn uniformly per sample.
        n_landmarks: Landmark cells drawn from the sketch by leverage score.
        n_comp: PLS components.
        add_cor_genes: Add the genes most correlated with the phenotype per
            cell type to the HVGs (Seurat's `QuickCorTest`).
        cor_top_prop: Share of the eligible genes kept per cell type and
            phenotype column.
        cor_min_pct: A gene is eligible in a cell type when it is non-zero in
            this share of its cells.
        cor_min_cells: Cell types with fewer training cells are skipped.
        scale_max: Upper clip of the scaled training data.
        n_folds: Sample folds for cross-fitting the PLS. `1` fits once on all
            samples, as the reference does; above 1 every sample is embedded
            by a model, gene selection included, that never saw it.
        wnn_k: Landmark neighbours the WNN keeps per cell.
        knn_range: Landmark candidates per cell and modality.
        sigma_k: Self kNN size, self included, for the WNN bandwidths.
        modality_weights: Fixed weights of the PCA and the PLS.
        k_count: Landmarks each cell is counted against.
        leverage_n_sketch: CountSketch rows of the leverage scores.
        leverage_eps: Johnson-Lindenstrauss distortion of the leverage scores.
        knn_method: Index of the two self kNN searches.
    """

    n_cells_per_donor: int = 1000
    n_landmarks: int = 2000
    n_comp: int = 20
    add_cor_genes: bool = True
    cor_top_prop: float = 0.01
    cor_min_pct: float = 0.1
    cor_min_cells: int = 100
    scale_max: float = 10.0
    n_folds: int = 1
    wnn_k: int = 20
    knn_range: int = 200
    sigma_k: int = 20
    modality_weights: tuple[float, float] = (0.5, 0.5)
    k_count: int = 5
    leverage_n_sketch: int = 5000
    leverage_eps: float = 0.5
    knn_method: Literal["kmknn", "exhaustive", "annoy", "hnsw", "ivf", "nndescent"] = (
        "kmknn"
    )

    def __post_init__(self) -> None:
        check_range("n_cells_per_donor", self.n_cells_per_donor, 1, None, False, True)
        check_range("n_landmarks", self.n_landmarks, 2, None, False, True)
        check_range("n_comp", self.n_comp, 1, None, False, True)
        check_range("cor_top_prop", self.cor_top_prop, 0, 1, True, False)
        check_range("cor_min_pct", self.cor_min_pct, 0, 1, False, False)
        check_range("cor_min_cells", self.cor_min_cells, 2, None, False, True)
        check_range("scale_max", self.scale_max, 0, None, True, True)
        check_range("n_folds", self.n_folds, 1, None, False, True)
        check_range("wnn_k", self.wnn_k, 1, None, False, True)
        check_range("knn_range", self.knn_range, 1, None, False, True)
        check_range("sigma_k", self.sigma_k, 2, None, False, True)
        check_range("modality_weights", self.modality_weights, 0, None, False, True)
        check_range("k_count", self.k_count, 1, self.wnn_k, False, False)
        check_range("leverage_n_sketch", self.leverage_n_sketch, 2, None, False, True)
        check_range("leverage_eps", self.leverage_eps, 0, 1, True, False)


@beartype(conf=_CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class SampleDmParams:
    """Parameters for `sample_diffusion_map`.

    Attributes:
        n_dcs: Diffusion components.
        k: Neighbours per sample in the kernel; the `k`-th sets the adaptive
            bandwidth. Must stay below the number of samples.
        t: Diffusion time; components are scaled by `lambda^t`.
        alpha_norm: Density normalisation exponent, 1 for Laplace-Beltrami.
    """

    n_dcs: int = 10
    k: int = 10
    t: int = 1
    alpha_norm: float = 1.0

    def __post_init__(self) -> None:
        check_range("n_dcs", self.n_dcs, 1, None, False, True)
        check_range("k", self.k, 1, None, False, True)
        check_range("t", self.t, 1, None, False, True)
        check_range("alpha_norm", self.alpha_norm, 0, 1, False, False)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class ScSlideRes:
    """Landmark densities of scSLIDE.

    Attributes:
        counts: `(n_landmarks, n_samples)` cells per landmark and sample.
            Landmark cells are not counted.
        samples: Sample names, the columns of `counts`.
        sample_meta: One row per sample: `sample` and the phenotype columns.
        landmark_ids: Cell id of each landmark, the rows of `counts`.
        landmark_labels: Cell type of each landmark, or `None` without
            `cell_type_col`.
        cell_ids: The kept cells, in `cells_to_keep` order.
        cell_landmarks: `(n_cells, k_count)` landmarks each cell was counted
            against, 0-based rows of `counts`, best first.
        pls_embedding: `(n_cells, n_comp)` PLS scores; with `n_folds > 1` each
            cell's from the model that did not see its sample.
        sample_folds: Fold per sample, in `samples` order.
        training_cell_ids: The training sketch.
        pls_genes: Gene ids the PLS of each fold was fitted on.
        params: The parameters, plus `used_emb` and `phenotype`.
    """

    counts: np.ndarray
    samples: tuple[str, ...]
    sample_meta: pl.DataFrame
    landmark_ids: np.ndarray
    landmark_labels: np.ndarray | None
    cell_ids: np.ndarray
    cell_landmarks: np.ndarray
    pls_embedding: np.ndarray
    sample_folds: np.ndarray
    training_cell_ids: np.ndarray
    pls_genes: tuple[np.ndarray, ...]
    params: dict[str, object]

    def __repr__(self) -> str:
        return compact_repr(
            "ScSlideRes",
            n_landmarks=self.counts.shape[0],
            n_samples=len(self.samples),
            n_cells=len(self.cell_ids),
            n_folds=len(self.pls_genes),
        )


@beartype
@dataclass(frozen=True, repr=False)
class SampleDiffusionRes:
    """Sample coordinates from scSLIDE's landmark densities.

    Attributes:
        dcs: `sample`, the phenotype columns and `DC_1` ... `DC_n`.
        residuals: `(n_landmarks, n_samples)` chi-squared residuals of the
            counts, before centring.
        params: The parameters.
    """

    dcs: pl.DataFrame
    residuals: np.ndarray
    params: dict[str, object]

    def __repr__(self) -> str:
        n_dcs = sum(c.startswith("DC_") for c in self.dcs.columns)
        return compact_repr(
            "SampleDiffusionRes", n_samples=self.dcs.height, n_dcs=n_dcs
        )


#############
# Functions #
#############


@beartype
def _phenotype_matrix(
    obs: pl.DataFrame, sample_col: str, phenotype: Sequence[str], samples: list[str]
) -> tuple[np.ndarray, pl.DataFrame]:
    """The phenotype per sample: numeric columns as they are, the rest one-hot
    over all levels.

    Raises:
        ValueError: On a missing value or a phenotype that varies within a
            sample.
    """
    meta = obs.select(
        pl.col(sample_col).cast(pl.String).alias("sample"), *phenotype
    ).unique(maintain_order=True)
    if meta.height != meta["sample"].n_unique():
        bad = meta.filter(pl.col("sample").is_duplicated())["sample"].unique().to_list()
        raise ValueError(
            f"The phenotype must be constant within a sample; it varies in {bad[:5]}."
        )
    meta = pl.DataFrame({"sample": samples}).join(meta, on="sample", how="left")
    blocks = []
    for col in phenotype:
        values = meta[col]
        if values.null_count():
            raise ValueError(f"Phenotype column '{col}' holds missing values.")
        if values.dtype.is_numeric():
            blocks.append(values.cast(pl.Float64).to_numpy()[:, None])
        else:
            codes, levels = _factor_codes(values, col)
            blocks.append(np.eye(len(levels))[codes])
    return np.ascontiguousarray(np.hstack(blocks), dtype=np.float64), meta


@beartype
def run_scslide(
    sc: ScOrScSubset,
    sample_col: str,
    phenotype: str | Sequence[str],
    *,
    cell_type_col: str | None = None,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    scslide_params: ScSlideParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> ScSlideRes:
    """Landmark x sample densities, scSLIDE's first half.

    Sketches `n_cells_per_donor` cells per sample, fits a PLS of the scaled
    HVGs (plus the phenotype-correlated genes per cell type) against the
    phenotype, projects every cell, draws landmarks from the sketch by leverage
    score and counts each non-landmark cell against its `k_count` nearest
    landmarks under a fixed-weight WNN over the embedding and the PLS.

    The phenotype is a sample label, so an in-sample PLS separates the samples
    by it partly by construction. Set `n_folds` above 1 to cross-fit: each fold
    of samples is embedded by a model that never saw it.

    Args:
        sc: The experiment, with HVGs and the embedding.
        sample_col: obs column with the sample (donor) per cell.
        phenotype: obs column(s) holding a sample-level phenotype, constant
            within a sample. Numeric columns enter as they are, others one-hot.
        cell_type_col: obs column with cell types, for the correlated genes and
            the landmark labels. `None` treats all cells as one group.
        embd_to_use: The cell-state embedding of the first WNN modality.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        scslide_params: The parameters. `None` uses the defaults.
        seed: Seed.
        verbose: Print progress.

    Returns:
        The `ScSlideRes`.

    Raises:
        ValueError: If the HVGs or the embedding are missing or stale, a kept
            cell lacks a sample, or the phenotype varies within a sample.

    References:
        Jiang, et al., Cell, 2026.

    Examples:
        >>> res = bv.traj.run_scslide(sc, "donor", "disease", cell_type_col="celltype")
        >>> res.counts.shape
    """
    params = scslide_params or ScSlideParams()
    phenotype = [phenotype] if isinstance(phenotype, str) else list(phenotype)
    _hard_state(sc, [embd_to_use])
    hvg = sc.map.hvg
    if hvg is None:
        raise ValueError("No HVGs found. Run bv.pp.find_hvg() first.")
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        raise ValueError(f"No embedding '{embd_to_use}' found.")
    embd = _first_dims(embd, no_embd_to_use)

    cols = [sample_col, *phenotype, *([cell_type_col] if cell_type_col else [])]
    obs = _obs_aligned(sc, cols)
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in the obs table.")
    donors, samples = _factor_codes(obs[sample_col].cast(pl.String), sample_col)
    y, meta = _phenotype_matrix(obs, sample_col, phenotype, samples)
    types, type_levels = (
        _factor_codes(obs[cell_type_col].cast(pl.String), cell_type_col)
        if cell_type_col
        else (None, None)
    )

    res = _bixverse.scslide(
        str(sc.cells_path),
        sc.cells_to_keep,
        donors,
        y,
        types,
        np.ascontiguousarray(hvg, dtype=np.uint32),
        embd,
        to_rust(params),
        seed,
        bool(verbose),
    )

    cell_ids = sc.cell_ids(sc.cells_to_keep)
    landmarks = res["landmarks"]
    landmark_labels = (
        None
        if types is None or type_levels is None
        else np.asarray(type_levels, dtype=object)[types[landmarks]]
    )
    return ScSlideRes(
        counts=res["counts"],
        samples=tuple(samples),
        sample_meta=meta,
        landmark_ids=cell_ids[landmarks],
        landmark_labels=landmark_labels,
        cell_ids=cell_ids,
        cell_landmarks=res["cell_landmarks"],
        pls_embedding=res["pls_embedding"],
        sample_folds=res["donor_folds"],
        training_cell_ids=cell_ids[res["training_cells"]],
        pls_genes=tuple(sc.gene_ids(g) for g in res["pls_genes"]),
        params={**to_rust(params), "used_emb": embd_to_use, "phenotype": phenotype},
    )


@beartype
def sample_diffusion_map(
    res: ScSlideRes,
    *,
    dm_params: SampleDmParams | None = None,
    seed: int = 42,
) -> SampleDiffusionRes:
    """Sample coordinates from the landmark densities, scSLIDE's second half.

    Chi-squared residuals of the landmark x sample counts (correspondence
    analysis without the SVD), centred per landmark, then a diffusion map over
    the samples with cosine distance. Seurat runs destiny on `1 - Pearson`
    instead, so components agree with it up to sign and scale, not exactly.

    Args:
        res: The `run_scslide` result.
        dm_params: The parameters. `None` uses the defaults; `k` is lowered to
            the number of samples minus two when it does not fit.
        seed: Seed of the eigensolver.

    Returns:
        The `SampleDiffusionRes`.

    Raises:
        ValueError: With too few samples for the requested components.

    References:
        Coifman & Lafon, Appl Comput Harmon Anal, 2006; Jiang, et al., Cell,
        2026.

    Examples:
        >>> params = bv.traj.SampleDmParams(k=8)
        >>> dm = bv.traj.sample_diffusion_map(res, dm_params=params)
        >>> dm.dcs.select("sample", "DC_1", "DC_2")
    """
    params = dm_params or SampleDmParams()
    n_samples = len(res.samples)
    if params.k >= n_samples - 1:
        k = max(n_samples - 2, 1)
        logger.warning(
            "k = %i does not fit %i samples; using %i.", params.k, n_samples, k
        )
        params = SampleDmParams(
            n_dcs=params.n_dcs, k=k, t=params.t, alpha_norm=params.alpha_norm
        )
    out = _bixverse.scslide_diffusion_map(
        np.ascontiguousarray(res.counts, dtype=np.float64), to_rust(params), seed
    )
    dcs = pl.DataFrame(
        {f"DC_{i + 1}": out["dcs"][:, i] for i in range(out["dcs"].shape[1])}
    )
    return SampleDiffusionRes(
        dcs=pl.concat([res.sample_meta, dcs], how="horizontal"),
        residuals=out["residuals"],
        params=to_rust(params),
    )
