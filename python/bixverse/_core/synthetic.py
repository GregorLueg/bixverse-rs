"""Synthetic single cell data for tests and examples.

Ports of R's `generate_*_test_data()`. The generators are the crate's, so a
seed gives the same counts in R and Python; this layer only names cells and
genes and builds the obs / var tables the way R does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.state import CsrMatrix
from bixverse.params import (
    SyntheticCellsweepParams,
    SyntheticDataAdtParams,
    SyntheticDataParams,
    SyntheticDialogueParams,
    to_rust,
)

###########
# Records #
###########


@dataclass(frozen=True)
class SyntheticData:
    """Synthetic counts with their obs and var tables.

    Attributes:
        counts: Cells x genes raw counts.
        obs: One row per cell, `cell_id` first.
        var: One row per gene, `gene_id` first.
    """

    counts: CsrMatrix
    obs: pl.DataFrame
    var: pl.DataFrame


@dataclass(frozen=True)
class SyntheticAdtData:
    """Synthetic ADT counts.

    Attributes:
        counts: `(n_cells, n_proteins)` `float64` raw counts.
        obs: One row per cell, `cell_id` first.
        var: One row per protein: `protein_id`, `is_isotype`.
    """

    counts: np.ndarray
    obs: pl.DataFrame
    var: pl.DataFrame


@dataclass(frozen=True)
class SyntheticCellsweepData(SyntheticData):
    """Synthetic counts with a planted ambient profile.

    Attributes:
        ambient_true: The soup per gene, in var order, summing to one.
        celltype_profiles_true: `(n_celltypes, n_genes)`, rows sum to one.
    """

    ambient_true: np.ndarray = field(kw_only=True)
    celltype_profiles_true: np.ndarray = field(kw_only=True)


@dataclass(frozen=True)
class SyntheticDialogueData(SyntheticData):
    """Synthetic counts with a planted multicellular programme.

    Attributes:
        features: Per cell type a frame of `cell_id` plus `feature_01`, ...,
            the input DIALOGUE takes.
        latent: The per-sample latent of the planted programme.
        planted: Per cell type the planted gene ids.
    """

    features: dict[str, pl.DataFrame] = field(kw_only=True)
    latent: dict[str, float] = field(kw_only=True)
    planted: dict[str, list[str]] = field(kw_only=True)


###########
# Helpers #
###########


@beartype
def _labels(prefix: str, n: int) -> list[str]:
    """`prefix_001` style labels, zero padded to the digits of `n`."""
    width = len(str(n))
    return [f"{prefix}_{i:0{width}d}" for i in range(1, n + 1)]


@beartype
def _csr(res: dict[str, Any]) -> CsrMatrix:
    """The generator's CSR buffers as a `CsrMatrix` of `float32` counts.

    `Any` because the dict is heterogeneous FFI output.
    """
    return CsrMatrix(
        np.asarray(res["data"]).astype(np.float32),
        np.asarray(res["indices"], dtype=np.uint32),
        np.asarray(res["indptr"], dtype=np.uint32),
        (int(res["nrow"]), int(res["ncol"])),
    )


@beartype
def _var(n_genes: int) -> pl.DataFrame:
    """`gene_id` and `ensembl_id` for `n_genes` genes."""
    return pl.DataFrame(
        {"gene_id": _labels("gene", n_genes), "ensembl_id": _labels("ens", n_genes)}
    )


@beartype
def _cell_configs(marker_genes: dict[str, dict[str, list[int]]]) -> list[dict]:
    """The marker gene configs in list order, as R hands them to Rust."""
    return [dict(v) for v in marker_genes.values()]


##############
# Generators #
##############


@beartype
def generate_single_cell_test_data(
    params: SyntheticDataParams | None = None, *, seed: int = 42
) -> SyntheticData:
    """Synthetic RNA counts with planted cell types.

    Args:
        params: Shape of the data.
        seed: Seed.

    Returns:
        Counts, obs (`cell_id`, `cell_grp`, `batch_index`, plus `sample_id`
        with sample membership) and var (`gene_id`, `ensembl_id`).
    """
    p = params or SyntheticDataParams()
    res = _bixverse.synthetic_sc_data(
        p.n_cells,
        p.n_genes,
        p.n_batches,
        p.n_samples,
        _cell_configs(p.marker_genes),
        p.batch_effect_strength,
        p.sample_bias,
        seed,
    )
    obs = pl.DataFrame(
        {
            "cell_id": _labels("cell", p.n_cells),
            "cell_grp": [
                f"cell_type_{i + 1}" for i in res["cell_type_indices"].tolist()
            ],
            # R adds a double one to the 0-based integer batch
            "batch_index": res["batch_indices"].astype(np.float64) + 1.0,
        }
    )
    if res["sample_indices"] is not None:
        obs = obs.with_columns(
            pl.Series(
                "sample_id", [f"sample_{i + 1}" for i in res["sample_indices"].tolist()]
            )
        )
    return SyntheticData(counts=_csr(res), obs=obs, var=_var(p.n_genes))


@beartype
def generate_single_cell_test_data_adt(
    params: SyntheticDataAdtParams | None = None, *, seed: int = 42
) -> SyntheticAdtData:
    """Synthetic ADT counts, cell for cell with `generate_single_cell_test_data`.

    Args:
        params: Shape of the data.
        seed: Seed.

    Returns:
        Dense counts, obs (`cell_id`, `cell_grp`, `batch_index`) and var
        (`protein_id`, `is_isotype`).
    """
    p = params or SyntheticDataAdtParams()
    res = _bixverse.synthetic_adt_data(
        p.n_cells,
        p.n_proteins,
        p.n_batches,
        list(p.isotype_controls),
        _cell_configs(p.marker_genes),
        p.batch_effect_strength,
        seed,
    )
    proteins = _labels("protein", p.n_proteins)
    isotypes = {proteins[i] for i in p.isotype_controls if 0 <= i < p.n_proteins}
    obs = pl.DataFrame(
        {
            "cell_id": _labels("cell", p.n_cells),
            "cell_grp": [
                f"cell_type_{i + 1}" for i in res["cell_type_indices"].tolist()
            ],
            "batch_index": res["batch_indices"].astype(np.float64) + 1.0,
        }
    )
    var = pl.DataFrame(
        {"protein_id": proteins, "is_isotype": [x in isotypes for x in proteins]}
    )
    return SyntheticAdtData(
        counts=np.asarray(res["counts"], dtype=np.float64), obs=obs, var=var
    )


@beartype
def generate_cellsweep_test_data(
    params: SyntheticCellsweepParams | None = None, *, seed: int = 42
) -> SyntheticCellsweepData:
    """Synthetic counts with a planted ambient profile, for CellSweep.

    Real barcodes come first, empty droplets after. Load them with a fully
    permissive `MinQualityParams`, or the ingest drops the empty droplets
    the model trains on.

    Args:
        params: Shape of the data.
        seed: Seed.

    Returns:
        Counts, obs (`cell_id`, `cell_grp` null for empties, `sample_id`,
        `is_empty`, `alpha_true` null for empties), var and the ground truth.
    """
    p = params or SyntheticCellsweepParams()
    res = _bixverse.synthetic_cellsweep_data(to_rust(p), seed)
    n_cells, n_genes = int(res["nrow"]), int(res["ncol"])
    n_empty = n_cells - p.n_real
    obs = pl.DataFrame(
        {
            "cell_id": _labels("cell", n_cells),
            "cell_grp": [
                f"cell_type_{i + 1}" for i in res["cell_type_indices"].tolist()
            ]
            + [None] * n_empty,
            "sample_id": ["sample_01"] * n_cells,
            "is_empty": res["is_empty"],
            "alpha_true": res["alpha_true"].tolist() + [None] * n_empty,
        },
        schema_overrides={"cell_grp": pl.String, "alpha_true": pl.Float64},
    )
    return SyntheticCellsweepData(
        counts=_csr(res),
        obs=obs,
        var=_var(n_genes),
        ambient_true=np.asarray(res["ambient_true"]),
        celltype_profiles_true=np.asarray(res["celltype_profiles_true"]),
    )


@beartype
def generate_dialogue_test_data(
    params: SyntheticDialogueParams | None = None, *, seed: int = 42
) -> SyntheticDialogueData:
    """Synthetic counts with a planted multicellular programme, for DIALOGUE.

    Cells are laid out by cell type and, within one, by sample. Only the
    first feature column and the planted genes carry the shared per-sample
    latent.

    Args:
        params: Shape of the data.
        seed: Seed.

    Returns:
        Counts, obs (`cell_id`, `cell_grp`, `sample_id`, `cell_quality`, the
        last pure noise), var, the per cell type features, the latent and the
        planted genes.
    """
    p = params or SyntheticDialogueParams()
    res = _bixverse.synthetic_dialogue_data(to_rust(p), seed)
    n_cells, n_genes = int(res["nrow"]), int(res["ncol"])
    cell_ids = _labels("cell", n_cells)
    gene_ids = _labels("gene", n_genes)
    cell_types = [f"cell_type_{i + 1}" for i in range(len(res["cell_type_indices"]))]

    cell_grp = [""] * n_cells
    for name, cells in zip(cell_types, res["cell_type_indices"], strict=True):
        for c in cells.tolist():
            cell_grp[c] = name
    obs = pl.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_grp": cell_grp,
            "sample_id": [f"sample_{i + 1:02d}" for i in res["sample_ids"].tolist()],
            "cell_quality": res["quality"],
        }
    )

    features = {}
    for name, cells, mat in zip(
        cell_types, res["cell_type_indices"], res["features"], strict=True
    ):
        cols = {f"feature_{j + 1:02d}": mat[:, j] for j in range(mat.shape[1])}
        features[name] = pl.DataFrame(
            {"cell_id": [cell_ids[c] for c in cells.tolist()], **cols}
        )
    latent = {f"sample_{i + 1:02d}": float(v) for i, v in enumerate(res["latent"])}
    planted = {
        name: [gene_ids[g] for g in genes.tolist()]
        for name, genes in zip(cell_types, res["planted"], strict=True)
    }
    return SyntheticDialogueData(
        counts=_csr(res),
        obs=obs,
        var=_var(n_genes),
        features=features,
        latent=latent,
        planted=planted,
    )
