"""`MetaCells`: aggregated cells held in memory.

The meta cell generators aggregate a `SingleCells` or `SingleCellsSubset`
into far fewer meta cells, so the data lives in memory: obs and var as polars
frames, the summed raw counts and the re-normalised counts as two CSR matrices
sharing one index structure, and a `ScCache` for the PCA, kNN, neighbour graph
and embeddings, exactly as on the on-disk classes.

Indices are 0-based throughout. `obs["meta_cell_idx"]` is the row,
`obs["original_cell_idx"]` lists the store cells of the source each meta cell
aggregates, and gene indices into `var` are positions in the source's gene
space. The cache is stamped against the meta cell ids: a meta cell set is
fixed at construction, so only a renaming (`merge_meta_cells`) moves it.

`save_metacells` / `load_metacells` write a directory of Parquet (obs, var,
the frames in `other_data`), one npz (counts, cache, the arrays) and one JSON
(everything else), no pickle. The residual model is not persisted, as for
`SingleCells`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype

from bixverse._core.repr import compact_repr
from bixverse._core.state import (
    CsrMatrix,
    ScCache,
    Stamp,
    _lookup,
    cache_from_payload,
    cache_payload,
    check_state,
    stale_artefacts,
    state_report,
)
from bixverse._validate import Verbose

###########
# Globals #
###########

MC_FORMAT: int = 1
MC_OBS: str = "obs.parquet"
MC_VAR: str = "var.parquet"
MC_NPZ: str = "data.npz"
MC_JSON: str = "meta.json"

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(repr=False)
class MetaCellAssignment:
    """Where the meta cells came from.

    The per-cell assignments are derived from the memberships in obs on
    demand, see `MetaCells.cell_assignments`.

    Attributes:
        n_cells: Cells (obs rows) in the source store.
        n_metacells: Number of meta cells.
        n_unassigned: Source cells in no meta cell, QC-dropped ones included.
        cells_to_keep: The source's `cells_to_keep` at generation, the row
            order of its embeddings, kNN and diffusion maps. `None` for a
            merged object.
        per_source: For a merged object, `n_cells`, `n_metacells` and
            `n_unassigned` per source id.
    """

    n_cells: int
    n_metacells: int
    n_unassigned: int
    cells_to_keep: np.ndarray | None = None
    per_source: dict[str, dict[str, int]] | None = None

    def __repr__(self) -> str:
        return compact_repr(
            "MetaCellAssignment",
            n_cells=self.n_cells,
            n_metacells=self.n_metacells,
            n_unassigned=self.n_unassigned,
            sources=None if self.per_source is None else list(self.per_source),
        )


#############
# MetaCells #
#############


class MetaCells:
    """Meta cells in memory.

    Built by the generators in `bixverse.metacells` and by
    `merge_meta_cells`, not by hand.

        >>> import bixverse as bv
        >>> mc = bv.metacells.generate_bt_meta_cells(sc)
        >>> bv.pp.find_hvg(mc, hvg_no=2000)
        >>> bv.pp.calculate_pca(mc, 30)
        >>> bv.graph.find_neighbours(mc)

    Attributes:
        raw: Summed raw counts, meta cells x genes, CSR.
        norm: The counts normalised to the target size and `log1p`'d, CSR
            sharing `raw`'s indices and pointers.
        cache: Analysis results.
        hvg: 0-based gene indices of the highly variable genes, if selected.
        original_assignment: Provenance of the meta cells.
        other_data: Method-specific extras (SEACells `rss` / `archetypes`,
            MetaCells2 `cell_info` / `rare_gene_modules` / `selected_genes`,
            diffusion `dcs` / `density_dist` / `regions`, merge `sources`).
        meta_cell_method: The generator.
        is_merged: Built by `merge_meta_cells`.
    """

    @beartype
    def __init__(
        self,
        obs: pl.DataFrame,
        var: pl.DataFrame,
        raw: CsrMatrix,
        norm: CsrMatrix,
        original_assignment: MetaCellAssignment,
        meta_cell_method: str,
        *,
        other_data: dict[str, object] | None = None,
        is_merged: bool = False,
    ) -> None:
        """Assemble a meta cell object; the generators and the loader call it.

        Raises:
            ValueError: If the tables and matrices disagree.
        """
        for col in ("meta_cell_idx", "meta_cell_id", "original_cell_idx"):
            if col not in obs.columns:
                raise ValueError(f"obs needs a '{col}' column")
        for col in ("gene_idx", "gene_id"):
            if col not in var.columns:
                raise ValueError(f"var needs a '{col}' column")
        shape = (obs.height, var.height)
        if raw.shape != shape or norm.shape != shape:
            raise ValueError(
                f"raw {raw.shape} and norm {norm.shape} must both be {shape}"
            )
        if not (
            np.array_equal(raw.indices, norm.indices)
            and np.array_equal(raw.indptr, norm.indptr)
        ):
            raise ValueError("raw and norm must share one index structure")
        if obs["meta_cell_id"].is_duplicated().any():
            raise ValueError("meta_cell_id holds duplicates")
        self._obs = obs
        self._var = var
        self.raw = raw
        self.norm = norm
        self.cache = ScCache()
        self.hvg: np.ndarray | None = None
        self.original_assignment = original_assignment
        self.other_data: dict[str, object] = other_data or {}
        self.meta_cell_method = meta_cell_method
        self.is_merged = is_merged

    ########
    # Dims #
    ########

    @property
    def n_cells(self) -> int:
        """Number of meta cells."""
        return self._obs.height

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self._var.height

    @property
    def shape(self) -> tuple[int, int]:
        """`(meta cells, genes)`."""
        return (self.n_cells, self.n_genes)

    ##############
    # Id lookups #
    ##############

    def cell_ids(self) -> np.ndarray:
        """The meta cell ids, in row order."""
        return self._obs["meta_cell_id"].to_numpy()

    @beartype
    def gene_ids(self, indices: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        """Gene ids, all or for 0-based gene indices."""
        ids = self._var["gene_id"].to_numpy()
        return ids if indices is None else ids[np.asarray(indices, dtype=np.int64)]

    @beartype
    def cell_indices(self, cell_ids: Sequence[str]) -> np.ndarray:
        """0-based rows of meta cells by id. Unknown ids warn and drop."""
        return _lookup(self.cell_ids(), cell_ids, "meta_cell")

    @beartype
    def gene_indices(self, gene_ids: Sequence[str]) -> np.ndarray:
        """0-based gene indices by id. Unknown ids warn and drop."""
        return _lookup(self._var["gene_id"].to_numpy(), gene_ids, "gene")

    @beartype
    def _resolve(
        self, sel: np.ndarray | Sequence[int] | Sequence[str], n: int, what: str
    ) -> np.ndarray:
        """0-based positions from positions or ids.

        Raises:
            ValueError: On a position outside `[0, n)`.
        """
        arr = np.asarray(sel)
        if arr.dtype.kind in "USO":
            ids = [str(x) for x in arr]
            return self.cell_indices(ids) if what == "cells" else self.gene_indices(ids)
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            raise ValueError(f"{what} holds an index outside [0, {n})")
        return arr.astype(np.uint32)

    @beartype
    def _resolve_cells(
        self, cells: np.ndarray | Sequence[int] | Sequence[str]
    ) -> np.ndarray:
        """0-based meta cell rows from rows or ids."""
        return self._resolve(cells, self.n_cells, "cells")

    @beartype
    def _resolve_genes(
        self, genes: np.ndarray | Sequence[int] | Sequence[str]
    ) -> np.ndarray:
        """0-based gene indices from indices or ids."""
        return self._resolve(genes, self.n_genes, "genes")

    #############
    # obs / var #
    #############

    @beartype
    def obs(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
        filtered: bool = False,
    ) -> pl.DataFrame:
        """The meta cell obs table.

        Args:
            indices: Optional 0-based rows.
            cols: Optional columns.
            filtered: Ignored; meta cells have no cell filter.

        Raises:
            KeyError: If a column is not in obs.
        """
        del filtered
        return _select(self._obs, indices, cols, "obs")

    def obs_columns(self) -> list[str]:
        """The obs column names."""
        return self._obs.columns

    @beartype
    def var(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
    ) -> pl.DataFrame:
        """The var table.

        Raises:
            KeyError: If a column is not in var.
        """
        return _select(self._var, indices, cols, "var")

    @beartype
    def add_obs(self, frame: pl.DataFrame) -> None:
        """Add or replace obs columns.

        With a 0-based `meta_cell_idx` column the frame joins on it, other
        meta cells get NULL. Without one it must hold one row per meta cell,
        in row order.

        Raises:
            ValueError: On a wrong row count or a protected column.
        """
        protected = [
            c
            for c in frame.columns
            if c in ("meta_cell_id", "original_cell_idx", "no_originating_cells")
        ]
        if protected:
            raise ValueError(f"protected obs column(s): {protected}")
        if "meta_cell_idx" not in frame.columns:
            if frame.height != self.n_cells:
                raise ValueError(
                    f"frame has {frame.height} rows; there are {self.n_cells} meta "
                    "cells. Pass a meta_cell_idx column to address them directly."
                )
            frame = frame.with_columns(self._obs["meta_cell_idx"])
        frame = frame.with_columns(pl.col("meta_cell_idx").cast(pl.UInt32))
        if frame["meta_cell_idx"].is_duplicated().any():
            raise ValueError("meta_cell_idx holds repeated values")
        new = [c for c in frame.columns if c != "meta_cell_idx"]
        self._obs = self._obs.drop([c for c in new if c in self._obs.columns]).join(
            frame, on="meta_cell_idx", how="left", maintain_order="left"
        )

    @beartype
    def add_var(self, frame: pl.DataFrame) -> None:
        """Add or replace var columns: one row per gene, in var order.

        Raises:
            ValueError: On a wrong row count or a protected column.
        """
        if frame.height != self.n_genes:
            raise ValueError(
                f"frame has {frame.height} rows; there are {self.n_genes} genes"
            )
        protected = [c for c in frame.columns if c in ("gene_idx", "gene_id")]
        if protected:
            raise ValueError(f"protected var column(s): {protected}")
        self._var = self._var.drop(
            [c for c in frame.columns if c in self._var.columns]
        ).hstack(frame)

    ##########
    # Counts #
    ##########

    @beartype
    def counts(
        self,
        *,
        cells: np.ndarray | Sequence[int] | Sequence[str] | None = None,
        genes: np.ndarray | Sequence[int] | Sequence[str] | None = None,
        layer: str = "raw",
    ) -> CsrMatrix:
        """The counts, optionally restricted.

        Args:
            cells: 0-based rows or meta cell ids, in the row order wanted.
            genes: 0-based gene indices or ids, in the column order wanted.
            layer: `"raw"` or `"norm"`.

        Raises:
            ValueError: On an unknown layer.
        """
        match layer:
            case "raw":
                m = self.raw
            case "norm":
                m = self.norm
            case _:
                raise ValueError(f"layer must be 'raw' or 'norm', got '{layer}'")
        rows = None if cells is None else self._resolve_cells(cells)
        cols = None if genes is None else self._resolve_genes(genes)
        return csr_subset(m, rows, cols)

    ##############
    # Provenance #
    ##############

    def cells_hash(self) -> str:
        """Hash of the meta cell ids, what the cache stamps record.

        Meta cells have no cell filter; their set is fixed at construction and
        identified by the ids, which a merge rewrites.
        """
        data = "\x1f".join(self.cell_ids().tolist()).encode()
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    @beartype
    def stamp(self, artefact: str, *, parents: Sequence[str] = ()) -> Stamp:
        """Stamp a cached artefact against the meta cell set."""
        return self.cache.stamp(artefact, self.cells_hash(), parents)

    def cache_status(self) -> pl.DataFrame:
        """One row per cached artefact: stamped, stale and why."""
        return state_report(self.cache, self.cells_hash())

    @beartype
    def check_state(self, artefacts: Sequence[str]) -> bool | str:
        """`True` if the artefacts are fresh, else what is stale and why."""
        return check_state(self.cache, self.cells_hash(), artefacts)

    ################
    # Memberships #
    ################

    def cell_assignments(self) -> pl.DataFrame:
        """One row per (source cell, meta cell) membership, in long form.

        Columns `cell_idx` (0-based store index in the source) and
        `meta_cell_idx`. Bootstrapped meta cells overlap, so a cell can
        appear more than once; unassigned cells do not appear.
        """
        return (
            self._obs.select(
                "meta_cell_idx", pl.col("original_cell_idx").alias("cell_idx")
            )
            .explode("cell_idx")
            .drop_nulls("cell_idx")
            .select("cell_idx", "meta_cell_idx")
            .sort("cell_idx", "meta_cell_idx")
        )

    @beartype
    def _artefact_rows(self, n_rows: int) -> list[np.ndarray]:
        """Memberships as rows of a source artefact.

        `original_cell_idx` holds store indices; embeddings, kNN graphs and
        diffusion maps have one row per kept cell, in the source's
        `cells_to_keep` order. This maps the one onto the other.

        Raises:
            ValueError: If the artefact does not cover the recorded cells, or
                a member is not among them.
        """
        a = self.original_assignment
        members = [
            np.asarray(m, dtype=np.int64)
            for m in self._obs["original_cell_idx"].to_list()
        ]
        keep = a.cells_to_keep
        if keep is None:
            if n_rows != a.n_cells:
                raise ValueError(
                    "This object does not record the source cell mapping, and the "
                    f"artefact covers {n_rows} of the {a.n_cells} original cells, so "
                    "the meta cell memberships cannot be resolved against it."
                )
            return members
        if n_rows != len(keep):
            raise ValueError(
                f"The artefact covers {n_rows} cells but the meta cells were built "
                f"over the {len(keep)} kept cells of the source. Regenerate it so "
                "the two agree."
            )
        lookup = np.full(a.n_cells, -1, dtype=np.int64)
        lookup[keep.astype(np.int64)] = np.arange(len(keep))
        rows = [lookup[m] for m in members]
        if any((r < 0).any() for r in rows):
            raise ValueError(
                "Some meta cell members are absent from the recorded cell mapping. "
                "The object is inconsistent; regenerate the meta cells."
            )
        return rows

    def __repr__(self) -> str:
        c = self.cache
        a = self.original_assignment
        stale = stale_artefacts(c, self.cells_hash())
        return (
            "MetaCells(\n"
            f"  method={self.meta_cell_method}, merged={self.is_merged},\n"
            f"  meta_cells={self.n_cells}, genes={self.n_genes},\n"
            f"  cells_aggregated={a.n_cells - a.n_unassigned}, "
            f"source_obs_rows={a.n_cells},\n"
            f"  hvg={self.hvg is not None}, pca={c.pca_factors is not None}, "
            f"embeddings={sorted(c.embeddings) or 'none'},\n"
            f"  knn={c.knn is not None}, graph={c.connectivities is not None}, "
            f"residuals={c.residual_fit is not None},\n"
            f"  stale={', '.join(stale) or 'none'}\n"
            ")"
        )


###########
# Helpers #
###########


@beartype
def _select(
    frame: pl.DataFrame,
    indices: np.ndarray | Sequence[int] | None,
    cols: list[str] | None,
    what: str,
) -> pl.DataFrame:
    """Rows by 0-based position and columns by name.

    Raises:
        KeyError: If a column is missing.
        ValueError: On a position out of range.
    """
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size and (np.min(idx) < 0 or np.max(idx) >= frame.height):
            raise ValueError(f"indices hold a row outside [0, {frame.height})")
        frame = frame[idx]
    if cols is not None:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"column(s) not found in '{what}': {missing}")
        frame = frame.select(cols)
    return frame


@beartype
def csr_args(
    m: CsrMatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """A CSR matrix as the `(data, indices, indptr, shape)` the meta cell
    Rust entry points take, `indptr` as `uint32`.

    Raises:
        ValueError: If the matrix overflows `uint32` row pointers.
    """
    if int(m.indptr[-1]) > np.iinfo(np.uint32).max:
        raise ValueError("the matrix has more entries than uint32 row pointers hold")
    return (
        np.ascontiguousarray(m.data),
        np.ascontiguousarray(m.indices),
        np.ascontiguousarray(m.indptr, dtype=np.uint32),
        m.shape,
    )


@beartype
def csr_subset(
    m: CsrMatrix, rows: np.ndarray | None, cols: np.ndarray | None
) -> CsrMatrix:
    """Rows and columns of a CSR matrix, in the order given.

    Column indices stay ascending within each row, as a CSR consumer expects.
    """
    data, indices, indptr = m.data, m.indices, m.indptr.astype(np.int64)
    n_rows, n_cols = m.shape
    if rows is not None:
        rows = rows.astype(np.int64)
        starts, ends = indptr[rows], indptr[rows + 1]
        lengths = ends - starts
        take = np.repeat(
            starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths
        ) + np.arange(lengths.sum())
        data, indices = data[take], indices[take]
        indptr = np.concatenate(([0], np.cumsum(lengths)))
        n_rows = len(rows)
    if cols is not None:
        new_pos = np.full(n_cols, -1, dtype=np.int64)
        new_pos[cols.astype(np.int64)] = np.arange(len(cols))
        mapped = new_pos[indices]
        keep = mapped >= 0
        row_of = np.repeat(np.arange(n_rows), np.diff(indptr))[keep]
        mapped = mapped[keep]
        order = np.lexsort((mapped, row_of))
        data, indices = data[keep][order], mapped[order]
        indptr = np.concatenate(([0], np.cumsum(np.bincount(row_of, minlength=n_rows))))
        n_cols = len(cols)
    return CsrMatrix(
        np.ascontiguousarray(data, dtype=np.float32),
        np.ascontiguousarray(indices, dtype=np.uint32),
        np.ascontiguousarray(indptr, dtype=np.uint32),
        (n_rows, n_cols),
    )


@beartype
def meta_cell_ids(n: int) -> list[str]:
    """Default ids, `meta_cell_<zero padded 1-based number>`."""
    width = len(str(n))
    return [f"meta_cell_{i:0{width}d}" for i in range(1, n + 1)]


@beartype
def build_metacells(
    res: dict,
    var: pl.DataFrame,
    meta_cell_method: str,
    *,
    cells_to_keep: np.ndarray | None,
    obs_ids: Sequence[str] | None = None,
    other_data: dict[str, object] | None = None,
) -> MetaCells:
    """A `MetaCells` from a generator's Rust result.

    Args:
        res: `metacells` (store cells per meta cell), the CSR parts `indptr`,
            `indices`, `raw`, `norm`, `shape` and `n_total`.
        var: `gene_idx` and `gene_id` of the source.
        meta_cell_method: The generator.
        cells_to_keep: The source's `cells_to_keep`.
        obs_ids: Meta cell ids; `None` for the default ids.
        other_data: Method-specific extras.
    """
    members: list[np.ndarray] = list(res["metacells"])
    n = len(members)
    ids = meta_cell_ids(n) if obs_ids is None else list(obs_ids)
    obs = pl.DataFrame(
        {
            "meta_cell_idx": pl.Series(np.arange(n, dtype=np.uint32)),
            "meta_cell_id": pl.Series(ids, dtype=pl.String),
            "no_originating_cells": pl.Series(
                [len(m) for m in members], dtype=pl.UInt32
            ),
            "original_cell_idx": pl.Series(
                [m.tolist() for m in members], dtype=pl.List(pl.UInt32)
            ),
        }
    )
    shape = tuple(res["shape"])
    raw = CsrMatrix(res["raw"], res["indices"], res["indptr"], shape)
    norm = CsrMatrix(res["norm"], res["indices"], res["indptr"], shape)
    n_total = int(res["n_total"])
    covered = np.zeros(n_total, dtype=bool)
    for m in members:
        covered[m] = True
    assignment = MetaCellAssignment(
        n_cells=n_total,
        n_metacells=n,
        n_unassigned=int(n_total - covered.sum()),
        cells_to_keep=None if cells_to_keep is None else cells_to_keep.copy(),
    )
    return MetaCells(
        obs,
        var.select("gene_idx", "gene_id"),
        raw,
        norm,
        assignment,
        meta_cell_method,
        other_data=other_data,
    )


###########
# Merging #
###########


@beartype
def _gene_space(gene_lists: list[list[str]], feature_space: str) -> list[str]:
    """The merged gene space, ordered by the first input.

    Raises:
        ValueError: On an unknown `feature_space` or an empty intersection.
    """
    match feature_space:
        case "intersect":
            shared = set(gene_lists[0]).intersection(*gene_lists[1:])
            target = [g for g in dict.fromkeys(gene_lists[0]) if g in shared]
        case "union":
            target = list(dict.fromkeys(g for genes in gene_lists for g in genes))
        case _:
            raise ValueError(
                f"feature_space must be 'intersect' or 'union', got '{feature_space}'"
            )
    if not target:
        raise ValueError("The gene intersection of the inputs is empty.")
    return target


@beartype
def _remap(m: CsrMatrix, gene_map: np.ndarray, n_cols: int) -> CsrMatrix:
    """A CSR matrix on another gene axis of `n_cols` genes; `gene_map[j]` is
    column `j`'s new position or `-1` to drop it. Rows stay sorted."""
    mapped = gene_map[m.indices.astype(np.int64)]
    keep = mapped >= 0
    n_rows = m.shape[0]
    row_of = np.repeat(np.arange(n_rows), np.diff(m.indptr.astype(np.int64)))[keep]
    mapped = mapped[keep]
    order = np.lexsort((mapped, row_of))
    indptr = np.concatenate(([0], np.cumsum(np.bincount(row_of, minlength=n_rows))))
    return CsrMatrix(
        np.ascontiguousarray(m.data[keep][order]),
        np.ascontiguousarray(mapped[order], dtype=np.uint32),
        np.ascontiguousarray(indptr, dtype=np.uint32),
        (n_rows, n_cols),
    )


@beartype
def merge_meta_cells(
    inputs: Sequence[MetaCells] | dict[str, MetaCells],
    *,
    source_ids: Sequence[str] | None = None,
    feature_space: str = "intersect",
    prefix_ids: bool = True,
    verbose: Verbose = True,
) -> MetaCells:
    """Row-bind several `MetaCells` into one.

    The typical case is sample-pure meta cells, generated per patient and
    merged so that downstream methods run across the full set. The normalised
    counts carry over: they are normalised per meta cell, so row-binding
    leaves them valid. Caches and HVGs are per source and dropped.

    `original_cell_idx` stays in each source's index space, which is why the
    diffusion and manifold metrics refuse a merged object.

    Args:
        inputs: The meta cells; a dict names them by source.
        source_ids: Source ids, unique, one per input. `None` takes the dict
            keys, else `source_01`, `source_02`, ...
        feature_space: `"intersect"` or `"union"` of the gene spaces; with
            `"union"` a gene an input lacks is a structural zero for its meta
            cells.
        prefix_ids: Prefix the meta cell ids with `"<source_id>__"`. Without
            it, ids repeated across inputs are an error.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The merged meta cells, `is_merged` set, a `source_id` obs column and
        the source ids in `other_data["sources"]`.

    Raises:
        ValueError: On no inputs, mismatched methods or ids, or an empty gene
            intersection.

    Examples:
        >>> mc_a = bv.metacells.generate_supercells(sc)
        >>> mc_b = bv.metacells.generate_supercells(sc, seed=1)
        >>> merged = bv.metacells.merge_meta_cells({"a": mc_a, "b": mc_b})
    """
    if isinstance(inputs, dict):
        if source_ids is None:
            source_ids = [str(k) for k in inputs]
        mcs = list(inputs.values())
    else:
        mcs = list(inputs)
    if not mcs:
        raise ValueError("inputs must be non-empty")
    if source_ids is None:
        source_ids = [f"source_{i:02d}" for i in range(1, len(mcs) + 1)]
    source_ids = [str(s) for s in source_ids]
    if len(source_ids) != len(mcs) or len(set(source_ids)) != len(source_ids):
        raise ValueError("source_ids must be unique, one per input")
    methods = list(dict.fromkeys(mc.meta_cell_method for mc in mcs))
    if len(methods) > 1:
        raise ValueError(
            f"All inputs need the same meta cell method. Found: {', '.join(methods)}."
        )

    gene_lists = [mc.var()["gene_id"].to_list() for mc in mcs]
    target = _gene_space(gene_lists, feature_space)
    pos = {g: i for i, g in enumerate(target)}
    gene_maps = [
        np.array([pos.get(g, -1) for g in genes], dtype=np.int64)
        for genes in gene_lists
    ]
    if verbose:
        logger.info(
            "Merging %i meta cell objects over %i genes (%s).",
            len(mcs),
            len(target),
            feature_space,
        )
    dropped = max(int((m < 0).sum()) for m in gene_maps)
    if dropped:
        warnings.warn(
            f"Dropping up to {dropped} genes from individual inputs. The normalised "
            "counts of the affected meta cells no longer sum to their original "
            "target library size.",
            stacklevel=2,
        )

    # observations
    shared = set(mcs[0].obs_columns()).intersection(
        *(mc.obs_columns() for mc in mcs[1:])
    )
    cols = [c for c in mcs[0].obs_columns() if c in shared]
    lost = [
        c
        for c in dict.fromkeys(c for mc in mcs for c in mc.obs_columns())
        if c not in shared
    ]
    if lost:
        warnings.warn(
            "Dropping observation columns not shared by all inputs: "
            f"{', '.join(lost)}.",
            stacklevel=2,
        )
    obs = pl.concat(
        [
            mc.obs(cols=cols).with_columns(pl.lit(sid).alias("source_id"))
            for mc, sid in zip(mcs, source_ids, strict=True)
        ],
        how="vertical_relaxed",
    )
    if prefix_ids:
        obs = obs.with_columns(
            pl.format("{}__{}", "source_id", "meta_cell_id").alias("meta_cell_id")
        )
    elif obs["meta_cell_id"].is_duplicated().any():
        raise ValueError(
            "Duplicated meta cell identifiers across the inputs. Use "
            "prefix_ids=True or rename them."
        )
    obs = obs.with_columns(
        pl.Series("meta_cell_idx", np.arange(obs.height, dtype=np.uint32))
    ).select(
        "meta_cell_idx",
        "meta_cell_id",
        "source_id",
        *[c for c in cols if c not in ("meta_cell_idx", "meta_cell_id")],
    )

    # counts: raw and norm share one index structure, so remap it once
    raws, norms = [], []
    n_target = len(target)
    for mc, gmap in zip(mcs, gene_maps, strict=True):
        identity = len(gmap) == n_target and np.array_equal(gmap, np.arange(n_target))
        if identity:
            raws.append(mc.raw)
            norms.append(mc.norm)
            continue
        raws.append(_remap(mc.raw, gmap, n_target))
        norms.append(_remap(mc.norm, gmap, n_target))
    raw = _rbind(raws, n_target)
    norm = _rbind(norms, n_target)

    # provenance: sources off one parent share its obs space
    per_source = {
        sid: {
            "n_cells": mc.original_assignment.n_cells,
            "n_metacells": mc.original_assignment.n_metacells,
            "n_unassigned": mc.original_assignment.n_unassigned,
        }
        for mc, sid in zip(mcs, source_ids, strict=True)
    }
    source_cells = {v["n_cells"] for v in per_source.values()}
    if len(source_cells) == 1:
        n_cells = source_cells.pop()
        seen = np.zeros(n_cells, dtype=bool)
        for m in obs["original_cell_idx"].to_list():
            seen[np.asarray(m, dtype=np.int64)] = True
        n_unassigned = int(n_cells - seen.sum())
    else:
        n_cells = sum(v["n_cells"] for v in per_source.values())
        n_unassigned = sum(v["n_unassigned"] for v in per_source.values())

    other_keys = list(
        dict.fromkeys(
            k
            for mc in mcs
            for k in [*mc.other_data, *(["hvg"] if mc.hvg is not None else [])]
        )
    )
    if other_keys:
        warnings.warn(
            f"Dropping per-source entries from `other_data`: {', '.join(other_keys)}.",
            stacklevel=2,
        )

    return MetaCells(
        obs,
        pl.DataFrame(
            {
                "gene_idx": pl.Series(np.arange(n_target, dtype=np.uint32)),
                "gene_id": pl.Series(target, dtype=pl.String),
            }
        ),
        raw,
        norm,
        MetaCellAssignment(
            n_cells=n_cells,
            n_metacells=obs.height,
            n_unassigned=n_unassigned,
            cells_to_keep=None,
            per_source=per_source,
        ),
        methods[0],
        other_data={"sources": source_ids},
        is_merged=True,
    )


@beartype
def _rbind(mats: list[CsrMatrix], n_cols: int) -> CsrMatrix:
    """Stack CSR matrices with the same column count.

    Raises:
        ValueError: If the merged matrix overflows `uint32` row pointers.
    """
    nnz = sum(len(m.data) for m in mats)
    if nnz > np.iinfo(np.uint32).max:
        raise ValueError(
            f"The merged matrices would hold {nnz} stored values, which overflows "
            "the index of a sparse matrix. Merge fewer sources."
        )
    offsets = np.cumsum([0, *[len(m.data) for m in mats]])
    indptr = np.concatenate(
        [np.zeros(1, dtype=np.int64)]
        + [
            m.indptr[1:].astype(np.int64) + off
            for m, off in zip(mats, offsets, strict=False)
        ]
    )
    return CsrMatrix(
        np.concatenate([m.data for m in mats]),
        np.concatenate([m.indices for m in mats]),
        indptr.astype(np.uint32),
        (sum(m.shape[0] for m in mats), n_cols),
    )


###############
# Persistence #
###############


@beartype
def save_metacells(mc: MetaCells, directory: Path | str) -> None:
    """Write a `MetaCells` to a directory: Parquet, npz and JSON, no pickle.

    The residual model is not written; a warning says so.

    Args:
        mc: The meta cells.
        directory: Created if missing; existing files of the format are
            replaced.

    Raises:
        TypeError: On an `other_data` entry that is not an array, a frame, a
            list of strings or a number.

    Examples:
        >>> bv.metacells.save_metacells(mc, "meta_cells")
    """
    out = Path(directory).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    arrays, cache_meta = cache_payload(mc.cache)
    for name, m in (("raw", mc.raw), ("norm", mc.norm)):
        arrays[f"counts/{name}"] = m.data
    arrays["counts/indices"] = mc.raw.indices
    arrays["counts/indptr"] = mc.raw.indptr
    if mc.hvg is not None:
        arrays["hvg"] = mc.hvg
    a = mc.original_assignment
    if a.cells_to_keep is not None:
        arrays["assignment/cells_to_keep"] = a.cells_to_keep

    other: dict[str, str] = {}
    other_plain: dict[str, object] = {}
    for key, value in mc.other_data.items():
        match value:
            case np.ndarray():
                arrays[f"other/{key}"] = value
                other[key] = "array"
            case pl.DataFrame():
                value.write_parquet(out / f"other_{key}.parquet")
                other[key] = "frame"
            case list() | int() | float() | str():
                other_plain[key] = value
                other[key] = "json"
            case _:
                raise TypeError(
                    f"other_data['{key}'] is a {type(value).__name__}, which "
                    "save_metacells cannot write"
                )

    meta = {
        "format": MC_FORMAT,
        "meta_cell_method": mc.meta_cell_method,
        "is_merged": mc.is_merged,
        "shape": list(mc.shape),
        "assignment": {
            "n_cells": a.n_cells,
            "n_metacells": a.n_metacells,
            "n_unassigned": a.n_unassigned,
            "per_source": a.per_source,
        },
        "other": other,
        "other_plain": other_plain,
        "cache": cache_meta,
    }
    mc.obs().write_parquet(out / MC_OBS)
    mc.var().write_parquet(out / MC_VAR)
    tmp = out / (MC_NPZ + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)  # ty: ignore[invalid-argument-type]
    os.replace(tmp, out / MC_NPZ)
    (out / MC_JSON).write_text(json.dumps(meta, indent=1))


@beartype
def load_metacells(directory: Path | str) -> MetaCells:
    """Read a `MetaCells` written by `save_metacells`.

    Args:
        directory: The directory `save_metacells` wrote.

    Returns:
        The meta cells.

    Raises:
        FileNotFoundError: If a file of the format is missing.
        ValueError: On an unknown format version.

    Examples:
        >>> mc = bv.metacells.load_metacells("meta_cells")
    """
    src = Path(directory).expanduser()
    for name in (MC_OBS, MC_VAR, MC_NPZ, MC_JSON):
        if not (src / name).is_file():
            raise FileNotFoundError(f"{src / name} does not exist")
    meta = json.loads((src / MC_JSON).read_text())
    if meta.get("format") != MC_FORMAT:
        raise ValueError(f"unknown {MC_JSON} format {meta.get('format')}")
    with np.load(src / MC_NPZ, allow_pickle=False) as fh:
        arrays = {k: fh[k] for k in fh.files}

    shape = (int(meta["shape"][0]), int(meta["shape"][1]))
    idx, ptr = arrays["counts/indices"], arrays["counts/indptr"]
    raw = CsrMatrix(arrays["counts/raw"], idx, ptr, shape)
    norm = CsrMatrix(arrays["counts/norm"], idx, ptr, shape)
    a = meta["assignment"]
    assignment = MetaCellAssignment(
        n_cells=a["n_cells"],
        n_metacells=a["n_metacells"],
        n_unassigned=a["n_unassigned"],
        cells_to_keep=arrays.get("assignment/cells_to_keep"),
        per_source=a["per_source"],
    )
    other: dict[str, object] = {}
    for key, kind in meta["other"].items():
        match kind:
            case "array":
                other[key] = arrays[f"other/{key}"]
            case "frame":
                other[key] = pl.read_parquet(src / f"other_{key}.parquet")
            case "json":
                other[key] = meta["other_plain"][key]
            case _:
                raise ValueError(f"unknown other_data kind '{kind}'")

    mc = MetaCells(
        pl.read_parquet(src / MC_OBS),
        pl.read_parquet(src / MC_VAR),
        raw,
        norm,
        assignment,
        meta["meta_cell_method"],
        other_data=other,
        is_merged=meta["is_merged"],
    )
    mc.hvg = arrays.get("hvg")
    mc.cache = cache_from_payload(arrays, meta["cache"])
    return mc
