"""The `SingleCells` handle: a directory shared with the bixverse R package.

A directory holds the binary stores (`counts_cells.bin`, `counts_genes.bin`)
and the obs / var DuckDB (`sc_duckdb.db`), in exactly the layout R writes, so
either front end opens what the other wrote. Analysis state lives in memory
(`map`, `cache`) and is persisted by `bixverse.io.save` next to R's own
memory file, never in it.

Only state access lives here, no analysis.
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
from bixverse._core.state import (
    CsrMatrix,
    ScCache,
    ScMap,
    Stamp,
    cells_hash,
    check_state,
    stale_artefacts,
    state_report,
)

###########
# Globals #
###########

CELLS_BIN: str = "counts_cells.bin"
GENES_BIN: str = "counts_genes.bin"

Selection = np.ndarray | Sequence[int] | Sequence[str]


###############
# SingleCells #
###############


class SingleCells:
    """A single cell experiment on disk.

    The constructor rebuilds the map from the DuckDB (ids from `obs` / `var`,
    the cell filter from `to_keep`) with an empty cache, which is what R's
    `load_existing()` does when no memory file is present. Use
    `bixverse.io.load_existing` to also restore a saved cache.

        >>> import bixverse as bv
        >>> sc = bv.io.load_existing("my_experiment")
        >>> sc.obs(filtered=True)
        >>> counts = bv.get_counts(sc, genes=["CD3E", "MS4A1"])

    Attributes:
        dir: The experiment directory.
        map: Ids and index sets.
        cache: Analysis results held in memory.
    """

    @beartype
    def __init__(self, directory: Path | str) -> None:
        """Open an experiment directory.

        Args:
            directory: Holds `sc_duckdb.db` and `counts_cells.bin`.

        Raises:
            FileNotFoundError: If either is missing.
            ValueError: If the obs / var tables disagree with the store.
        """
        self.dir = Path(directory).expanduser()
        for path in (self.db_path, self.cells_path):
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist")

        self.reader = _bixverse.CellReader(str(self.cells_path))
        with db.connect(self.db_path, read_only=True) as con:
            cell_ids = db.get_ids(con, "obs")
            gene_ids = db.get_ids(con, "var")
            cells_to_keep = db.get_cells_to_keep(con)

        if (len(cell_ids), len(gene_ids)) != self.dims:
            raise ValueError(
                "The data in the observation table and or var table do not match "
                f"what is stored on disk: {len(cell_ids)} x {len(gene_ids)} in the "
                f"DuckDB, {self.dims[0]} x {self.dims[1]} in {CELLS_BIN}."
            )
        self.map = ScMap(cell_ids, gene_ids, cells_to_keep)
        self.cache = ScCache()

    #########
    # Paths #
    #########

    @property
    def db_path(self) -> Path:
        """The obs / var DuckDB."""
        return db.db_path(self.dir)

    @property
    def cells_path(self) -> Path:
        """The cell-major store."""
        return self.dir / CELLS_BIN

    @property
    def genes_path(self) -> Path:
        """The gene-major store."""
        return self.dir / GENES_BIN

    ########
    # Dims #
    ########

    @property
    def dims(self) -> tuple[int, int]:
        """`(cells, genes)` in the store, before any cell filter."""
        return (self.reader.total_cells, self.reader.total_genes)

    @property
    def n_cells(self) -> int:
        """Number of kept cells."""
        return len(self.map.cells_to_keep)

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self.reader.total_genes

    @property
    def shape(self) -> tuple[int, int]:
        """`(kept cells, genes)`, as R's `dim()`."""
        return (self.n_cells, self.n_genes)

    @property
    def cells_to_keep(self) -> np.ndarray:
        """0-based store indices of the kept cells."""
        return self.map.cells_to_keep

    ##############
    # Id lookups #
    ##############

    @beartype
    def cell_ids(
        self,
        indices: np.ndarray | Sequence[int] | None = None,
        *,
        filtered: bool = False,
    ) -> np.ndarray:
        """Cell ids, all, kept only, or for 0-based store indices."""
        if indices is not None:
            return self.map.cell_ids[np.asarray(indices, dtype=np.int64)]
        return self.map.cell_names(filtered=filtered)

    @beartype
    def gene_ids(self, indices: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        """Gene ids, all or for 0-based store indices."""
        if indices is None:
            return self.map.gene_ids
        return self.map.gene_ids[np.asarray(indices, dtype=np.int64)]

    @beartype
    def cell_indices(self, cell_ids: Sequence[str]) -> np.ndarray:
        """0-based store indices of cells by id. Unknown ids warn and drop."""
        return self.map.cell_indices(cell_ids)

    @beartype
    def gene_indices(self, gene_ids: Sequence[str]) -> np.ndarray:
        """0-based store indices of genes by id. Unknown ids warn and drop."""
        return self.map.gene_indices(gene_ids)

    @beartype
    def _resolve_cells(self, cells: Selection) -> np.ndarray:
        """0-based cell indices from indices or ids."""
        arr = np.asarray(cells)
        if arr.dtype.kind in "USO":
            return self.cell_indices([str(c) for c in arr])
        return _check_range(arr, self.dims[0], "cells")

    @beartype
    def _resolve_genes(self, genes: Selection) -> np.ndarray:
        """0-based gene indices from indices or ids."""
        arr = np.asarray(genes)
        if arr.dtype.kind in "USO":
            return self.gene_indices([str(g) for g in arr])
        return _check_range(arr, self.dims[1], "genes")

    ##############
    # obs / var #
    ##############

    @beartype
    def obs(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
        filtered: bool = False,
    ) -> pl.DataFrame:
        """The obs table, `cell_idx` 0-based.

        Args:
            indices: Optional 0-based cell indices.
            cols: Optional columns.
            filtered: Only the kept cells.
        """
        idx = (
            None if indices is None else _check_range(indices, self.dims[0], "indices")
        )
        with db.connect(self.db_path, read_only=True) as con:
            return db.get_table(con, "obs", indices=idx, cols=cols, filtered=filtered)

    @beartype
    def var(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
    ) -> pl.DataFrame:
        """The var table, `gene_idx` 0-based.

        Args:
            indices: Optional 0-based gene indices.
            cols: Optional columns.
        """
        idx = (
            None if indices is None else _check_range(indices, self.dims[1], "indices")
        )
        with db.connect(self.db_path, read_only=True) as con:
            return db.get_table(con, "var", indices=idx, cols=cols)

    @beartype
    def add_obs(self, frame: pl.DataFrame) -> None:
        """Add or replace obs columns.

        With a 0-based `cell_idx` column the frame joins on it. Without one it
        must hold one row per kept cell, in ascending cell order, as R's
        `set_sc_new_obs_col()`. Cells the frame does not cover get NULL.
        """
        if "cell_idx" not in frame.columns:
            if frame.height != self.n_cells:
                raise ValueError(
                    f"frame has {frame.height} rows; there are {self.n_cells} kept "
                    "cells. Pass a cell_idx column to address cells directly."
                )
            frame = frame.with_columns(
                pl.Series("cell_idx", np.sort(self.map.cells_to_keep))
            )
        with db.connect(self.db_path) as con:
            db.add_data(con, "obs", frame)

    @beartype
    def add_var(self, frame: pl.DataFrame) -> None:
        """Add or replace var columns: one row per gene, or a 0-based `gene_idx`."""
        with db.connect(self.db_path) as con:
            db.add_data(con, "var", frame)

    @beartype
    def drop_obs_cols(self, cols: list[str]) -> None:
        """Drop obs columns. Protected and missing ones warn and are skipped."""
        with db.connect(self.db_path) as con:
            db.drop_columns(con, "obs", cols)

    @beartype
    def drop_var_cols(self, cols: list[str]) -> None:
        """Drop var columns. Protected and missing ones warn and are skipped."""
        with db.connect(self.db_path) as con:
            db.drop_columns(con, "var", cols)

    @beartype
    def rename_obs_col(self, old: str, new: str) -> None:
        """Rename an obs column. A missing one warns."""
        with db.connect(self.db_path) as con:
            db.rename_column(con, "obs", old, new)

    @beartype
    def rename_var_col(self, old: str, new: str) -> None:
        """Rename a var column. A missing one warns."""
        with db.connect(self.db_path) as con:
            db.rename_column(con, "var", old, new)

    ###############
    # Cell filter #
    ###############

    @beartype
    def set_cells_to_keep(self, cells: Selection) -> None:
        """Restrict the experiment to a set of cells.

        Updates the map and the `to_keep` column. The order given is the order
        later results align to. Anything cached was computed on the old cell
        set; it is kept, not deleted, and a warning names what went stale.

        Args:
            cells: 0-based store indices or cell ids.
        """
        idx = self._resolve_cells(cells)
        if idx.size == 0:
            raise ValueError("cells must be non-empty")
        self.map.cells_to_keep = idx
        with db.connect(self.db_path) as con:
            db.set_cells_to_keep(con, idx)

        stale = stale_artefacts(self.cache, self.cells_hash())
        if stale:
            warnings.warn(
                "The cell set changed. These cached artefacts are now stale and "
                f"need re-running: {', '.join(stale)}. See `cache_status()`.",
                stacklevel=2,
            )

    @beartype
    def reset_cells_to_keep(self, *, force: bool = False) -> None:
        """Restore every cell, clearing the HVG selection and the cache.

        A PCA or kNN computed on a subset does not describe the full cell
        set, so nothing cached survives. R asks interactively; here a
        populated cache needs `force=True`.

        Raises:
            ValueError: If something would be dropped and `force` is false.
        """
        destroyed = list(self.cache.artefacts())
        if self.map.hvg is not None:
            destroyed.append("the HVG selection")
        if destroyed and not force:
            raise ValueError(
                f"Resetting would drop {', '.join(destroyed)}. Pass force=True "
                "if that is what you want."
            )
        self.map.cells_to_keep = np.arange(self.dims[0], dtype=np.uint32)
        self.map.hvg = None
        with db.connect(self.db_path) as con:
            db.reset_cells_to_keep(con)
        self.cache = ScCache()

    ##############
    # Provenance #
    ##############

    def cells_hash(self) -> str:
        """Hash of the current cell set, as recorded in stamps."""
        return cells_hash(self.map.cells_to_keep)

    @beartype
    def stamp(self, artefact: str, *, parents: Sequence[str] = ()) -> Stamp:
        """Stamp a cached artefact against the current cell set.

        Args:
            artefact: `"pca"`, `"knn"`, `"connectivities"`, `"magic"`,
                `"residuals"` or an embedding name.
            parents: Labels of the artefacts it was derived from.
        """
        return self.cache.stamp(artefact, self.cells_hash(), parents)

    def cache_status(self) -> pl.DataFrame:
        """One row per cached artefact: stamped, stale and why."""
        return state_report(self.cache, self.cells_hash())

    @beartype
    def check_state(self, artefacts: Sequence[str]) -> bool | str:
        """`True` if the artefacts are fresh, else what is stale and why."""
        return check_state(self.cache, self.cells_hash(), artefacts)

    def __repr__(self) -> str:
        c = self.cache
        stale = stale_artefacts(c, self.cells_hash())
        return (
            "SingleCells(\n"
            f"  dir={str(self.dir)!r},\n"
            f"  cells={self.dims[0]} (kept {self.n_cells}), genes={self.n_genes},\n"
            f"  hvg={self.map.hvg is not None}, pca={c.pca_factors is not None}, "
            f"embeddings={sorted(c.embeddings) or 'none'},\n"
            f"  knn={c.knn is not None}, graph={c.connectivities is not None}, "
            f"magic={c.magic is not None}, residuals={c.residual_fit is not None},\n"
            f"  stale={', '.join(stale) or 'none'}\n"
            ")"
        )


###########
# Helpers #
###########


@beartype
def _check_range(indices: np.ndarray | Sequence[int], n: int, name: str) -> np.ndarray:
    """Coerce 0-based indices to `uint32`, checking they fall in `[0, n)`.

    Raises:
        TypeError: If they are not integers.
        ValueError: If one is out of range.
    """
    arr = np.asarray(indices)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if arr.dtype.kind not in "iu" or arr.ndim != 1:
        raise TypeError(f"{name} must be a 1-D integer array, got {arr.dtype}")
    if arr.min() < 0 or arr.max() >= n:
        raise ValueError(f"{name} holds an index outside [0, {n})")
    return np.ascontiguousarray(arr, dtype=np.uint32)


##########
# Counts #
##########


@beartype
@dataclass(frozen=True)
class Counts:
    """A cells x genes count matrix in CSR, with what its rows and columns are.

    Attributes:
        matrix: The CSR components.
        cells: 0-based store index of each row.
        genes: 0-based store index of each column.
    """

    matrix: CsrMatrix
    cells: np.ndarray
    genes: np.ndarray

    def scipy_csr(self) -> object:
        """The matrix as a `scipy.sparse.csr_matrix`. Needs scipy."""
        return self.matrix.scipy_csr()


@beartype
def get_counts(
    sc: SingleCells,
    *,
    cells: Selection | None = None,
    genes: Selection | None = None,
    layer: Literal["raw", "norm"] = "raw",
    assay: Literal["rna"] = "rna",
    use_cells_to_keep: bool = True,
) -> Counts:
    """Read counts for a selection of cells and genes, as R's `get_sc_counts`.

    With `use_cells_to_keep` the rows are the kept cells, or the given cells
    that are kept, in the order given. The decode runs in Rust with the GIL
    released.

    Args:
        sc: The experiment.
        cells: 0-based store indices or ids. `None` for all (kept) cells.
        genes: 0-based store indices or ids, in the column order wanted.
            `None` for every gene.
        layer: `"raw"` counts or the `"norm"` layer.
        assay: Only `"rna"`; `SingleCells` holds one modality.
        use_cells_to_keep: Restrict to the kept cells.

    Returns:
        The CSR matrix with its row cells and column genes.

    Raises:
        ValueError: If no cell is left to read.
    """
    del assay  # one modality; the argument mirrors R's `modality`
    if cells is None:
        idx = (
            sc.map.cells_to_keep
            if use_cells_to_keep
            else np.arange(sc.dims[0], dtype=np.uint32)
        )
    else:
        idx = sc._resolve_cells(cells)
        if use_cells_to_keep:
            idx = idx[np.isin(idx, sc.map.cells_to_keep)]
    if idx.size == 0:
        raise ValueError("no cells to read")

    gene_idx = None if genes is None else sc._resolve_genes(genes)
    data, indices, indptr, shape = sc.reader.read_csr(idx, layer, gene_idx)
    return Counts(
        matrix=CsrMatrix(data, indices, indptr, shape),
        cells=idx,
        genes=np.arange(sc.n_genes, dtype=np.uint32) if gene_idx is None else gene_idx,
    )
