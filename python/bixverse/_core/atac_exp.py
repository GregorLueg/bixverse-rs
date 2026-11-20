"""`SingleCellsAtac`: an ATAC-only experiment, and the ATAC state both kinds of
experiment share.

A `SingleCellsAtac` is a `SingleCells` whose store holds peaks or genome
tiles. Its var table is keyed like any other (`gene_idx`, `gene_id`, the
region as `chr:start-end`) plus `chrom`, `start`, `end`, so the store
machinery reads it unchanged; `var(modality="atac")` gives it the
`feature_idx` / `feature_id` shape of a multi-modal experiment's `var_atac`.

Both `SingleCellsAtac` and `SingleCellsMultiModal` expose the ATAC store
through the same attributes (`atac_dir`, `atac`, `atac_cache`,
`atac_var_table`, `atac_lib_col`), which is all the `bixverse.atac`
functions use, so one implementation serves both.

`AtacInfo` is rebuilt from the store on open: the feature kind and the norm
kind are header fields, the count mode is store metadata. Only the LSI
arrays live in Python's memory file.
"""

from __future__ import annotations

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
from bixverse._core.repr import compact_repr
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN, Selection, SingleCells
from bixverse._core.state import ScCache

###########
# Globals #
###########

ATAC_DIR: str = "atac"
ATAC_KINDS: tuple[str, ...] = ("peak", "tile")
GENE_ACTIVITY_DIR: str = "gene_activity"

CountMode = Literal["insertion", "paired-insertion", "fragment"]
COUNT_MODES: tuple[str, ...] = ("insertion", "paired-insertion", "fragment")

# store metadata key of the count mode
COUNT_MODE_KEY: str = "count_mode"


###########
# Records #
###########


@beartype
@dataclass(repr=False)
class AtacInfo:
    """What the ATAC store holds and how far it was processed.

    Attributes:
        kind: `"peak"` or `"tile"`.
        count_mode: What one count is: `"insertion"` (up to two per
            fragment), `"paired-insertion"` or `"fragment"` (one per fragment
            and feature). The 10x `Peaks` matrix counts insertions.
        tf_idf: Whether the store is TF-IDF normalised.
        depth_cor: Correlation of each LSI component with the library size,
            all components including any dropped one.
        lsi_dropped: 0-based LSI components left out of the `"lsi"` embedding.
        lsi_features: 0-based ATAC feature rows the LSI was fitted on.
        lsi_loadings: Features x components loadings of the LSI, all
            components, for projecting other cells onto it.
        lsi_singular_values: Singular values of the LSI.
    """

    kind: Literal["peak", "tile"]
    count_mode: CountMode
    tf_idf: bool = False
    depth_cor: np.ndarray | None = None
    lsi_dropped: tuple[int, ...] = ()
    lsi_features: np.ndarray | None = None
    lsi_loadings: np.ndarray | None = None
    lsi_singular_values: np.ndarray | None = None

    def __repr__(self) -> str:
        return compact_repr(
            "AtacInfo",
            kind=self.kind,
            count_mode=self.count_mode,
            tf_idf=self.tf_idf,
            lsi=self.depth_cor is not None,
            lsi_dropped=list(self.lsi_dropped),
        )


@beartype
def read_atac_info(store_dir: Path) -> AtacInfo:
    """The ATAC state an ATAC store records: kind, count mode, TF-IDF.

    Args:
        store_dir: The directory holding the ATAC store's two files.

    Returns:
        The state, with no LSI.

    Raises:
        ValueError: If the store holds neither peaks nor tiles.
    """
    cells = store_dir / CELLS_BIN
    kind = _bixverse.store_feature_kind(cells)
    if kind not in ATAC_KINDS:
        raise ValueError(f"{store_dir} holds {kind} features, not peaks or tiles.")
    mode = _bixverse.store_metadata(cells).get(COUNT_MODE_KEY)
    if mode not in COUNT_MODES:
        # stores written before the count mode was recorded: 10x counts and
        # the add_atac_fragments default are insertions
        warnings.warn(
            f"The ATAC store in {store_dir} records no count mode; assuming "
            "'insertion'. Re-count it to record one.",
            stacklevel=3,
        )
        mode = "insertion"
    return AtacInfo(
        kind=kind,  # ty: ignore[invalid-argument-type]
        count_mode=mode,  # ty: ignore[invalid-argument-type]
        tf_idf=_bixverse.store_norm_kind(cells) == "tf_idf",
    )


@beartype
def write_count_mode(store_dir: Path, count_mode: CountMode) -> None:
    """Record the count mode in both files of an ATAC store."""
    _bixverse.set_store_metadata(
        store_dir / CELLS_BIN, store_dir / GENES_BIN, [(COUNT_MODE_KEY, count_mode)]
    )


_LSI_ARRAYS = ("depth_cor", "lsi_features", "lsi_loadings", "lsi_singular_values")


@beartype
def atac_info_payload(info: AtacInfo) -> tuple[dict[str, np.ndarray], dict]:
    """The LSI state of an `AtacInfo` as npz arrays and JSON."""
    arrays = {
        f"atac/{k}": v for k in _LSI_ARRAYS if (v := getattr(info, k)) is not None
    }
    return arrays, {"lsi_dropped": list(info.lsi_dropped)}


@beartype
def restore_lsi(info: AtacInfo, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """Put the LSI state of a memory file back onto an `AtacInfo`.

    The store's own kind, count mode and TF-IDF flag stand; the memory file
    only carries what the store cannot.
    """
    for k in _LSI_ARRAYS:
        setattr(info, k, arrays.get(f"atac/{k}"))
    info.lsi_dropped = tuple(meta.get("lsi_dropped", ()))


#################
# Gene activity #
#################


@beartype
def sync_gene_activity_filter(owner: SingleCells) -> None:
    """Copy an experiment's cell filter into its default gene activity store.

    The gene activity store at `<dir>/gene_activity` shares the cell axis;
    without this a filter set after it was written would not reach it.
    """
    ga = owner.dir / GENE_ACTIVITY_DIR
    if not (ga / CELLS_BIN).is_file():
        return
    with db.connect(db.db_path(ga)) as con:
        db.set_cells_to_keep(con, owner.cells_to_keep)


####################
# SingleCellsAtac #
####################


class SingleCellsAtac(SingleCells):
    """A single cell ATAC experiment on disk: peaks or tiles, no RNA.

        >>> import bixverse as bv
        >>> x = bv.io.load_atac_fragments(
        ...     "exp", "fragments.tsv.gz", peaks="peaks.bed", min_fragments=1000
        ... )
        >>> bv.atac.fragment_qc(x, "fragments.tsv.gz", "genes.gtf.gz")
        >>> bv.atac.run_tfidf(x)
        >>> bv.atac.calculate_lsi(x)
        >>> bv.graph.find_neighbours(x, embd_to_use="lsi", modality="atac")

    `modality="atac"` and the default `"rna"` both name this object's own
    store and cache, so code written for the ATAC side of a
    `SingleCellsMultiModal` runs unchanged.

    Attributes:
        dir: The experiment directory.
        map: Ids (cell barcodes, region ids) and index sets.
        cache: The ATAC cache; also `atac_cache`.
        atac: The ATAC store state.
    """

    root_modality = "atac"
    atac_var_table = "var"
    atac_lib_col = "lib_size"

    @beartype
    def __init__(self, directory: Path | str) -> None:
        """Open an ATAC experiment directory.

        Args:
            directory: Holds `sc_duckdb.db` and the peak or tile store.

        Raises:
            FileNotFoundError: If either is missing.
            ValueError: If the store holds neither peaks nor tiles.
        """
        super().__init__(directory)
        self.atac: AtacInfo = read_atac_info(self.dir)

    @property
    def atac_dir(self) -> Path:
        """Where the ATAC store lies: the experiment directory itself."""
        return self.dir

    @property
    def atac_cache(self) -> ScCache:
        """The ATAC cache, which is this object's only cache."""
        return self.cache

    @atac_cache.setter
    def atac_cache(self, value: ScCache) -> None:
        self.cache = value

    @beartype
    def modality_cache(self, modality: str) -> ScCache:
        """The cache: `"rna"` and `"atac"` both name it.

        Raises:
            ValueError: For any other modality.
        """
        if modality not in ("rna", "atac"):
            raise ValueError(f"A SingleCellsAtac has no '{modality}' modality.")
        return self.cache

    @beartype
    def var(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
        modality: Literal["rna", "atac"] = "rna",
    ) -> pl.DataFrame:
        """The var table; with `"atac"` keyed `feature_idx` / `feature_id`.

        Args:
            indices: Optional 0-based feature indices.
            cols: Optional columns.
            modality: `"rna"` for the store's own keys, `"atac"` for the
                multi-modal `var_atac` shape.
        """
        if modality == "atac" and cols is not None:
            back = {"feature_id": "gene_id", "feature_idx": "gene_idx"}
            cols = [back.get(c, c) for c in cols]
        frame = super().var(indices=indices, cols=cols)
        if modality == "atac":
            frame = frame.rename(
                {"gene_idx": "feature_idx", "gene_id": "feature_id"}, strict=False
            )
        return frame

    @beartype
    def set_cells_to_keep(self, cells: Selection) -> None:
        """Restrict the experiment to a set of cells, gene activity included.

        Args:
            cells: 0-based store indices or cell ids.
        """
        super().set_cells_to_keep(cells)
        sync_gene_activity_filter(self)

    @beartype
    def reset_cells_to_keep(self, *, force: bool = False) -> None:
        """Restore every cell, gene activity included; see `SingleCells`."""
        super().reset_cells_to_keep(force=force)
        sync_gene_activity_filter(self)

    def __repr__(self) -> str:
        base = (
            super()
            .__repr__()
            .removeprefix("SingleCells(")
            .rstrip(")\n")
            .replace(", genes=", ", features=")
        )
        ga = (self.dir / GENE_ACTIVITY_DIR / CELLS_BIN).is_file()
        return f"SingleCellsAtac({base}\n  atac={self.atac!r}, gene_activity={ga}\n)"
