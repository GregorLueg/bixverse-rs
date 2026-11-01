"""`SingleCellsMultiModal`: RNA in the store, ADT counts in memory, a WNN slot.

The RNA side is a `SingleCells` in every respect, so every RNA method takes
one unchanged. On top of it sit three things:

- `adt`: the ADT counts of the cells kept when they were added, raw and
  normalised (CLR or DSB), keyed by barcode. They are not in the store and
  never written to it; `bixverse.io.save` keeps them in Python's own memory
  file. Their feature table is the `var_adt` table in the DuckDB and their
  per-cell totals land in obs.
- `adt_cache`: PCA, embeddings, kNN and graph over the ADT counts.
- `wnn`: the fused kNN, its graph, the per-cell modality weights and any
  embeddings laid out on it.

Methods with a `modality` argument read and write the cache it names, `"wnn"`
included. Provenance labels are qualified by modality (`"adt:pca"`,
`"wnn:knn"`), so a stamp can name parents in another cache, as the WNN kNN
does with its two embeddings.
"""

from __future__ import annotations

import dataclasses
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse._core import db
from bixverse._core.single_cells import Counts, SingleCells, _check_range
from bixverse._core.state import (
    CsrMatrix,
    ScCache,
    Stamp,
    cache_from_payload,
    cache_payload,
    check_state,
    state_report,
)
from bixverse._core.subset import ScOrMc
from bixverse.params import DsbParams

###########
# Globals #
###########

Modality = Literal["rna", "adt", "wnn"]
CacheModality = Literal["rna", "adt"]

MODALITY_ONLY_MM = "modality = '{}' is only supported for SingleCellsMultiModal."


###########
# Records #
###########


@beartype
@dataclass(frozen=True)
class DsbInfo:
    """What a DSB normalisation estimated on the way.

    Attributes:
        protein_background_mean: Per-protein background mean.
        protein_background_sd: Per-protein background SD; empty without empty
            droplets.
        technical_component: Per-cell covariate regressed out in step II;
            empty without denoising.
        cellwise_background_mean: Per-cell background mean; empty without
            denoising.
        isotype_names: The isotype controls used.
        dsb_params: The parameters.
        scale_factor: `"standardise"` or `"mean_subtract"`.
    """

    protein_background_mean: np.ndarray
    protein_background_sd: np.ndarray
    technical_component: np.ndarray
    cellwise_background_mean: np.ndarray
    isotype_names: tuple[str, ...]
    dsb_params: DsbParams
    scale_factor: str


@beartype
@dataclass(frozen=True)
class AdtCounts:
    """ADT counts with their normalisation.

    Attributes:
        raw_counts: `(n_cells, n_features)` `float64` raw counts.
        norm_counts: `(n_cells, n_features)` `float64` normalised counts.
        cell_ids: Barcode of each row.
        cell_idx: 0-based store index of each row.
        feature_ids: Id of each column.
        type: `"CLR"` or `"DSB"`.
        dsb: The DSB estimates, for `type == "DSB"`.
    """

    raw_counts: np.ndarray
    norm_counts: np.ndarray
    cell_ids: np.ndarray
    cell_idx: np.ndarray
    feature_ids: np.ndarray
    type: Literal["CLR", "DSB"]
    dsb: DsbInfo | None = None

    def __post_init__(self) -> None:
        """Check that every matrix and label vector lines up.

        Raises:
            ValueError: On a shape mismatch.
        """
        shape = (len(self.cell_ids), len(self.feature_ids))
        for name in ("raw_counts", "norm_counts"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must be {shape}")
        if len(self.cell_idx) != shape[0]:
            raise ValueError("cell_idx must have one entry per row")

    def __repr__(self) -> str:
        n, p = self.raw_counts.shape
        return f"AdtCounts(cells={n}, proteins={p}, type={self.type!r})"


@beartype
@dataclass
class WnnSlot:
    """The weighted nearest neighbour integration.

    Attributes:
        cache: The fused kNN, its graph and embeddings laid out on it. Its
            PCA slots stay empty.
        weights: `cell_id`, `weight_modality_1`, `weight_modality_2`, one row
            per kept cell.
        modality_1: The first modality fused.
        modality_2: The second modality fused.
    """

    cache: ScCache
    weights: pl.DataFrame
    modality_1: str
    modality_2: str


#########################
# SingleCellsMultiModal #
#########################


class SingleCellsMultiModal(SingleCells):
    """A single cell experiment with RNA and ADT.

        >>> import bixverse as bv
        >>> mm = bv.SingleCellsMultiModal("my_experiment")
        >>> bv.mm.add_adt_counts(mm, adt, barcodes, proteins, method="clr")
        >>> bv.mm.calculate_pca_adt(mm, no_pcs=10)
        >>> bv.mm.generate_wnn_graph(mm)
        >>> bv.graph.find_clusters(mm, modality="wnn")

    Attributes:
        dir: The experiment directory.
        map: Ids and index sets.
        cache: The RNA cache.
        adt: The ADT counts, once added.
        adt_cache: The ADT cache.
        wnn: The WNN integration, once generated.
    """

    @beartype
    def __init__(self, directory: Path | str) -> None:
        """Open an experiment directory, with no ADT counts yet.

        Args:
            directory: Holds `sc_duckdb.db` and `counts_cells.bin`.
        """
        super().__init__(directory)
        self.adt: AdtCounts | None = None
        self.adt_cache = ScCache()
        self.wnn: WnnSlot | None = None

    ##########
    # Caches #
    ##########

    @beartype
    def modality_cache(self, modality: Modality) -> ScCache:
        """The cache of a modality.

        Raises:
            ValueError: For `"wnn"` before `generate_wnn_graph`.
        """
        match modality:
            case "rna":
                return self.cache
            case "adt":
                return self.adt_cache
            case "wnn":
                if self.wnn is None:
                    raise ValueError(
                        "No 'wnn' integration found. Did you run the relevant "
                        "integration step?"
                    )
                return self.wnn.cache
            case _:
                raise ValueError(f"No cache for modality '{modality}'.")

    def _caches(self) -> dict[str, ScCache]:
        """Every cache present, by modality."""
        out = {"rna": self.cache, "adt": self.adt_cache}
        if self.wnn is not None:
            out["wnn"] = self.wnn.cache
        return out

    def _artefacts(self) -> dict[str, Stamp | None]:
        """Every artefact over all caches, labels qualified by modality."""
        return {
            f"{m}:{label}": stamp
            for m, cache in self._caches().items()
            for label, stamp in cache.artefacts().items()
        }

    def _clear_cache(self) -> None:
        """Drop every cache: all modalities share the kept cell set."""
        self.cache = ScCache()
        self.adt_cache = ScCache()
        self.wnn = None

    ##############
    # Provenance #
    ##############

    @beartype
    def stamp(
        self,
        artefact: str,
        *,
        parents: Sequence[str] = (),
        modality: Modality = "rna",
    ) -> Stamp:
        """Stamp a cached artefact of a modality against the current cell set.

        Args:
            artefact: `"pca"`, `"knn"`, `"connectivities"`, `"magic"`,
                `"residuals"` or an embedding name.
            parents: Labels of the artefacts it was derived from; unqualified
                ones are taken from `modality`, `"rna:pca"` names another.
                Absent or unstamped parents are dropped.
            modality: The cache the artefact sits in.

        Raises:
            KeyError: If no artefact sits under the label.
        """
        present = self._artefacts()
        ids = tuple(s.id for p in parents if (s := present.get(_qualify(p, modality))))
        new = Stamp(cells=self.cells_hash(), id=_new_id(), parents=ids)
        self.modality_cache(modality).set_stamp(artefact, new)
        return new

    @beartype
    def check_state(
        self, artefacts: Sequence[str], *, modality: Modality = "rna"
    ) -> bool | str:
        """`True` if the artefacts are fresh, else what is stale and why.

        Args:
            artefacts: Labels, unqualified ones taken from `modality`.
            modality: The modality to qualify against.
        """
        labels = [_qualify(a, modality) for a in artefacts]
        return check_state(self._artefacts(), self.cells_hash(), labels)

    def cache_status(self) -> pl.DataFrame:
        """One row per cached artefact over all modalities.

        Returns:
            As `SingleCells.cache_status`, with a leading `modality` column;
            `artefact` is unqualified.
        """
        report = state_report(self._artefacts(), self.cells_hash())
        parts = pl.col("artefact").str.splitn(":", 2)
        return report.select(
            parts.struct.field("field_0").alias("modality"),
            parts.struct.field("field_1").alias("artefact"),
            pl.exclude("artefact"),
        )

    #######
    # var #
    #######

    @beartype
    def var(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
        modality: CacheModality = "rna",
    ) -> pl.DataFrame:
        """The var table of a modality, `gene_idx` / `feature_idx` 0-based.

        Args:
            indices: Optional 0-based gene or feature indices.
            cols: Optional columns.
            modality: `"rna"` for `var`, `"adt"` for `var_adt`.

        Raises:
            TableNotFoundError: For `"adt"` before ADT counts were added.
        """
        if modality == "rna":
            return super().var(indices=indices, cols=cols)
        idx = None if indices is None else np.asarray(indices, dtype=np.uint32)
        with db.connect(self.db_path, read_only=True) as con:
            return db.get_table(con, "var_adt", indices=idx, cols=cols)

    def __repr__(self) -> str:
        base = super().__repr__().removeprefix("SingleCells(").rstrip(")\n")
        c = self.adt_cache
        adt = "none" if self.adt is None else repr(self.adt)
        return (
            "SingleCellsMultiModal("
            f"{base}\n"
            f"  adt={adt}, adt_pca={c.pca_factors is not None}, "
            f"adt_knn={c.knn is not None}, adt_graph={c.connectivities is not None},\n"
            f"  adt_embeddings={sorted(c.embeddings) or 'none'}, "
            f"wnn={self.wnn is not None}\n"
            ")"
        )


###########
# Helpers #
###########


@beartype
def _qualify(label: str, modality: str) -> str:
    """A label qualified by modality."""
    return label if ":" in label else f"{modality}:{label}"


def _new_id() -> str:
    """A fresh stamp id, as `ScCache.stamp` mints them."""
    return uuid.uuid4().hex[:16]


@beartype
def check_modality(sc: object, modality: str) -> None:
    """Non-RNA modalities need a multi-modal object.

    Raises:
        ValueError: On a non-RNA modality of anything else.
    """
    if modality != "rna" and not isinstance(sc, SingleCellsMultiModal):
        raise ValueError(MODALITY_ONLY_MM.format(modality))


@beartype
def modality_cache(sc: ScOrMc, modality: Modality) -> ScCache:
    """The cache a modality reads and writes.

    Raises:
        ValueError: On a non-RNA modality of a single modality object, or
            `"wnn"` before the integration.
    """
    check_modality(sc, modality)
    if isinstance(sc, SingleCellsMultiModal):
        return sc.modality_cache(modality)
    return sc.cache


@beartype
def stamp(
    sc: ScOrMc,
    label: str,
    *,
    parents: Sequence[str] = (),
    modality: Modality = "rna",
) -> None:
    """Stamp an artefact in a modality's cache, see `SingleCellsMultiModal.stamp`.

    On a single modality object `"rna:"` qualified parents are unqualified.
    """
    if isinstance(sc, SingleCellsMultiModal):
        sc.stamp(label, parents=parents, modality=modality)
        return
    check_modality(sc, modality)
    sc.stamp(label, parents=[p.removeprefix("rna:") for p in parents])


@beartype
def require_fresh(sc: ScOrMc, labels: Sequence[str], modality: Modality) -> None:
    """Hard state tier: raise unless the artefacts are fresh.

    Raises:
        ValueError: If one is stale.
    """
    if isinstance(sc, SingleCellsMultiModal):
        state = sc.check_state(labels, modality=modality)
    else:
        check_modality(sc, modality)
        state = sc.check_state([x.removeprefix("rna:") for x in labels])
    if state is not True:
        raise ValueError(state)


#######
# ADT #
#######


@beartype
def require_adt(mm: SingleCellsMultiModal) -> AdtCounts:
    """The ADT counts.

    Raises:
        ValueError: If none were added.
    """
    if mm.adt is None:
        raise ValueError(
            "No ADT counts in this object. Add them with add_adt_counts()."
        )
    return mm.adt


@beartype
def adt_rows(
    mm: SingleCellsMultiModal,
    cells: np.ndarray | Sequence[int] | Sequence[str] | None,
    use_cells_to_keep: bool,
) -> np.ndarray:
    """Rows of the ADT matrix for a cell selection.

    The ADT matrix is keyed by barcode and frozen at add time, so the cells
    resolve to barcodes first. Cells it does not hold warn and drop.

    Args:
        mm: The experiment, with ADT counts.
        cells: 0-based store indices or ids. `None` for the kept cells (or
            every ADT row without `use_cells_to_keep`).
        use_cells_to_keep: Restrict to the kept cells.

    Returns:
        Row positions into the ADT matrices.

    Raises:
        ValueError: Without ADT counts, or if no cell is left.
    """
    adt = require_adt(mm)
    if cells is None:
        if not use_cells_to_keep:
            return np.arange(len(adt.cell_ids))
        idx = mm.cells_to_keep
    else:
        idx = mm._resolve_cells(cells)
        if use_cells_to_keep:
            idx = idx[np.isin(idx, mm.cells_to_keep)]
    pos = {c: i for i, c in enumerate(adt.cell_ids)}
    found = [pos.get(c) for c in mm.cell_ids(idx)]
    n_missing = sum(f is None for f in found)
    if n_missing:
        warnings.warn(
            f"{n_missing} requested cells are not present in the ADT matrix; dropping.",
            stacklevel=3,
        )
    rows = np.asarray([f for f in found if f is not None], dtype=np.int64)
    if rows.size == 0:
        raise ValueError("No valid cells for the ADT matrix.")
    return rows


@beartype
def adt_cols(
    adt: AdtCounts, features: np.ndarray | Sequence[int] | Sequence[str]
) -> np.ndarray:
    """0-based ADT columns from indices or ids; unknown ids warn and drop.

    Raises:
        ValueError: If nothing matched or an index is out of range.
    """
    arr = np.asarray(features)
    if arr.dtype.kind not in "USO":
        return _check_range(arr, len(adt.feature_ids), "features").astype(np.int64)
    pos = {f: i for i, f in enumerate(adt.feature_ids)}
    found = [pos.get(str(f)) for f in arr]
    n_missing = sum(f is None for f in found)
    if n_missing:
        warnings.warn(
            f"{n_missing} ADT feature names not found; dropping.", stacklevel=3
        )
    cols = [f for f in found if f is not None]
    if not cols:
        raise ValueError("None of the ADT feature names matched.")
    return np.asarray(cols, dtype=np.int64)


@beartype
def adt_counts_csr(
    mm: SingleCellsMultiModal,
    *,
    cells: np.ndarray | Sequence[int] | Sequence[str] | None,
    features: np.ndarray | Sequence[int] | Sequence[str] | None,
    layer: Literal["raw", "norm"],
    use_cells_to_keep: bool,
) -> Counts:
    """The ADT counts as a `Counts`, for `get_counts(assay="adt")`.

    The values are narrowed to `float32` to fit `CsrMatrix`; `bv.mm.get_adt_counts`
    returns the dense `float64` matrix instead. `Counts.genes` holds the
    0-based ADT feature indices.
    """
    rows = adt_rows(mm, cells, use_cells_to_keep)
    adt = require_adt(mm)
    cols = (
        np.arange(len(adt.feature_ids), dtype=np.int64)
        if features is None
        else adt_cols(adt, features)
    )
    mat = (adt.raw_counts if layer == "raw" else adt.norm_counts)[np.ix_(rows, cols)]
    nz_r, nz_c = np.nonzero(mat)
    indptr = np.zeros(mat.shape[0] + 1, dtype=np.uint32)
    np.cumsum(np.bincount(nz_r, minlength=mat.shape[0]), out=indptr[1:])
    return Counts(
        matrix=CsrMatrix(
            mat[nz_r, nz_c].astype(np.float32),
            nz_c.astype(np.uint32),
            indptr,
            (int(mat.shape[0]), int(mat.shape[1])),
        ),
        cells=adt.cell_idx[rows],
        genes=cols.astype(np.uint32),
    )


###############
# Persistence #
###############

_DSB_ARRAYS: tuple[str, ...] = (
    "protein_background_mean",
    "protein_background_sd",
    "technical_component",
    "cellwise_background_mean",
)


@beartype
def mm_payload(mm: SingleCellsMultiModal) -> tuple[dict[str, np.ndarray], dict]:
    """The ADT counts, the ADT cache and the WNN slot as npz arrays and JSON."""
    arrays, adt_meta = cache_payload(mm.adt_cache, "adt_cache")
    meta: dict = {"adt_cache": adt_meta, "adt": None, "wnn": None}
    if mm.adt is not None:
        a = mm.adt
        arrays |= {
            "adt/raw_counts": a.raw_counts,
            "adt/norm_counts": a.norm_counts,
            "adt/cell_ids": np.asarray(a.cell_ids, dtype=str),
            "adt/cell_idx": a.cell_idx,
            "adt/feature_ids": np.asarray(a.feature_ids, dtype=str),
        }
        dsb = None
        if a.dsb is not None:
            arrays |= {f"adt/dsb/{k}": getattr(a.dsb, k) for k in _DSB_ARRAYS}
            dsb = {
                "isotype_names": list(a.dsb.isotype_names),
                "dsb_params": dataclasses.asdict(a.dsb.dsb_params),
                "scale_factor": a.dsb.scale_factor,
            }
        meta["adt"] = {"type": a.type, "dsb": dsb}
    if mm.wnn is not None:
        w = mm.wnn
        wnn_arrays, wnn_meta = cache_payload(w.cache, "wnn")
        arrays |= wnn_arrays
        arrays["wnn/weights/cell_id"] = np.asarray(w.weights["cell_id"], dtype=str)
        arrays["wnn/weights/1"] = w.weights["weight_modality_1"].to_numpy()
        arrays["wnn/weights/2"] = w.weights["weight_modality_2"].to_numpy()
        meta["wnn"] = {
            "cache": wnn_meta,
            "modality_1": w.modality_1,
            "modality_2": w.modality_2,
        }
    return arrays, meta


@beartype
def mm_from_payload(
    mm: SingleCellsMultiModal, arrays: dict[str, np.ndarray], meta: dict
) -> None:
    """Restore what `mm_payload` wrote onto an experiment."""
    mm.adt_cache = cache_from_payload(arrays, meta["adt_cache"], "adt_cache")
    if meta["adt"] is not None:
        dsb = meta["adt"]["dsb"]
        mm.adt = AdtCounts(
            raw_counts=arrays["adt/raw_counts"],
            norm_counts=arrays["adt/norm_counts"],
            cell_ids=arrays["adt/cell_ids"].astype(object),
            cell_idx=arrays["adt/cell_idx"],
            feature_ids=arrays["adt/feature_ids"].astype(object),
            type=meta["adt"]["type"],
            dsb=None
            if dsb is None
            else DsbInfo(
                **{k: arrays[f"adt/dsb/{k}"] for k in _DSB_ARRAYS},
                isotype_names=tuple(dsb["isotype_names"]),
                dsb_params=DsbParams(**dsb["dsb_params"]),
                scale_factor=dsb["scale_factor"],
            ),
        )
    if meta["wnn"] is not None:
        w = meta["wnn"]
        mm.wnn = WnnSlot(
            cache=cache_from_payload(arrays, w["cache"], "wnn"),
            weights=pl.DataFrame(
                {
                    "cell_id": pl.Series(
                        arrays["wnn/weights/cell_id"].tolist(), dtype=pl.String
                    ),
                    "weight_modality_1": arrays["wnn/weights/1"],
                    "weight_modality_2": arrays["wnn/weights/2"],
                }
            ),
            modality_1=w["modality_1"],
            modality_2=w["modality_2"],
        )
