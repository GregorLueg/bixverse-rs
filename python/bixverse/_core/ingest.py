"""Loading counts into a `SingleCells` directory, and maintaining the store.

Ports of R's `methods_sc_io.R` loaders. Each one writes the cell-based store
in Rust (applying the cell and gene QC), converts it to the gene-based
store, loads obs / var into the DuckDB, then adds the per-cell `nnz` and
`lib_size` and the per-gene `no_cells_exp` columns, as R does. The result is
a directory R's `load_existing()` opens.

Where R takes a `SingleCells` object pointing at an empty directory, these
take the directory and return the opened `SingleCells`.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import duckdb
import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.io import ARCHIVE_NAME, MEMORY_JSON, MEMORY_NPZ
from bixverse._core.scan import (
    Prescan,
    RawSlot,
    detect_raw_count_slot,
    get_h5ad_dimensions,
    get_tenx_h5_metadata,
    warn_skipped,
)
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN, SingleCells
from bixverse.params import MinQualityParams, MtxIoParams, to_rust

###########
# Globals #
###########

logger = logging.getLogger(__name__)


@runtime_checkable
class CsrLike(Protocol):
    """Anything with the three CSR buffers and a shape: a `CsrMatrix` or a
    `scipy.sparse.csr_matrix` / `csr_array`. Read-only members, so the
    frozen `CsrMatrix` satisfies it."""

    @property
    def data(self) -> np.ndarray: ...
    @property
    def indices(self) -> np.ndarray: ...
    @property
    def indptr(self) -> np.ndarray: ...
    @property
    def shape(self) -> tuple[int, int]: ...


###########
# Helpers #
###########


@beartype
def _prepare(directory: Path | str) -> Path:
    """Create the directory and drop Python memory files of an earlier load.

    The memory files describe the cells of whatever was loaded before, so
    they would mislead `load_existing` after a fresh ingest.
    """
    path = Path(directory).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    for name in (MEMORY_NPZ, MEMORY_JSON):
        (path / name).unlink(missing_ok=True)
    return path


@beartype
def csr_buffers(counts: CsrLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The CSR buffers as integral `uint32` counts.

    Raises:
        TypeError: If the matrix is not CSR.
        ValueError: If a count is negative or fractional.
    """
    fmt = getattr(counts, "format", "csr")
    if fmt != "csr":
        raise TypeError(f"counts must be a CSR matrix, got format {fmt!r}")
    data = np.asarray(counts.data)
    if data.size and (data.min() < 0 or not np.array_equal(data, np.floor(data))):
        raise ValueError("counts must be non-negative integers")
    return (
        np.ascontiguousarray(data, dtype=np.uint32),
        np.ascontiguousarray(counts.indices, dtype=np.uint32),
        np.ascontiguousarray(counts.indptr, dtype=np.uint32),
    )


@beartype
def _make_unique(names: list[str]) -> list[str]:
    """R's `make.unique()`: repeats get `.1`, `.2`, ... until unique."""
    seen = set(names)
    out: list[str] = []
    counts: dict[str, int] = {}
    used: set[str] = set()
    for name in names:
        if name not in used:
            used.add(name)
            out.append(name)
            continue
        i = counts.get(name, 0)
        while True:
            i += 1
            candidate = f"{name}.{i}"
            if candidate not in seen and candidate not in used:
                break
        counts[name] = i
        used.add(candidate)
        out.append(candidate)
    return out


@beartype
def _parquet_info(info: dict[str, Any]) -> db.ParquetFrameInfo:
    """The Rust Parquet writer's description as a `ParquetFrameInfo`.

    `Any` because the dict is heterogeneous FFI output; `ParquetFrameInfo`
    checks the fields.
    """
    categories = [None if c is None else tuple(c) for c in info["categories"]]
    return db.ParquetFrameInfo(
        names=tuple(info["names"]),
        types=tuple(info["types"]),
        categories=tuple(categories),
        skipped=tuple(info["skipped"]),
        n_rows=int(info["n_rows"]),
    )


@beartype
def _finish(
    directory: Path,
    nnz: np.ndarray,
    lib_size: np.ndarray,
    *,
    csc_mem_gb: float | None,
    verbose: bool,
    populate: Callable[[duckdb.DuckDBPyConnection], None],
) -> SingleCells:
    """The shared tail of every loader.

    Builds the gene-based store, lets `populate` write obs / var, then adds
    the QC columns and `to_keep`, as R's loaders do.

    Args:
        directory: The experiment directory.
        nnz: Genes detected per written cell.
        lib_size: Library size per written cell.
        csc_mem_gb: Memory for the CSR to CSC conversion, `None` for one pass.
        verbose: Controls verbosity.
        populate: Called with an open connection to write obs and var.

    Returns:
        The opened experiment.
    """
    if verbose:
        logger.info("Converting the cell-based data into the gene-based format.")
    _bixverse.generate_gene_based_data(
        directory / CELLS_BIN, directory / GENES_BIN, csc_mem_gb, verbose
    )
    gene_nnz = _bixverse.gene_nnz(directory / GENES_BIN)

    with db.connect(db.db_path(directory)) as con:
        populate(con)
        db.add_data(
            con,
            "obs",
            pl.DataFrame(
                {
                    "nnz": nnz.astype(np.float64),
                    "lib_size": lib_size.astype(np.float64),
                }
            ),
        )
        db.add_data(
            con,
            "var",
            pl.DataFrame({"no_cells_exp": gene_nnz.astype(np.int32)}),
        )
        db.set_to_keep_column(con)
    return SingleCells(directory)


@beartype
def _h5ad_obs_var(
    con: duckdb.DuckDBPyConnection,
    h5_path: Path,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray | None,
    cell_id_col: str | None,
) -> None:
    """Load the obs and (unless `gene_indices` is `None`) var of an h5ad file."""
    with tempfile.TemporaryDirectory() as tmp:
        frames: list[tuple[Literal["obs", "var"], np.ndarray, str | None]] = [
            ("obs", cell_indices, cell_id_col)
        ]
        if gene_indices is not None:
            frames.append(("var", gene_indices, None))
        for frame, keep, id_col in frames:
            out = Path(tmp) / f"{frame}.parquet"
            info = _bixverse.h5ad_obs_var_to_parquet(h5_path, frame, keep, id_col, out)
            warn_skipped(info["skipped"], f"/{frame}")
            db.parquet_to_duckdb(con, frame, out, _parquet_info(info))


@beartype
def _tenx_obs_var(
    con: duckdb.DuckDBPyConnection,
    h5_path: Path,
    version: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
) -> None:
    """Load the barcodes and features of a 10x h5 file."""
    with tempfile.TemporaryDirectory() as tmp:
        pairs: tuple[tuple[Literal["obs", "var"], np.ndarray], ...] = (
            ("obs", cell_indices),
            ("var", gene_indices),
        )
        for frame, keep in pairs:
            out = Path(tmp) / f"{frame}.parquet"
            info = _bixverse.tenx_obs_var_to_parquet(h5_path, version, frame, keep, out)
            db.parquet_to_duckdb(con, frame, out, _parquet_info(info))


@beartype
def _per_file_qc(per_file: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated `nnz` and `lib_size` of a multi-file load."""
    nnz = np.concatenate([f["nnz"] for f in per_file])
    lib = np.concatenate([f["lib_size"] for f in per_file])
    return nnz, lib


@beartype
def _final_genes(prescan: Prescan, global_gene_indices: np.ndarray) -> list[str]:
    """Gene ids of the genes a multi-file load kept, in store order."""
    return [prescan.universe[i] for i in global_gene_indices.tolist()]


@beartype
def _check_tasks(prescan: Prescan) -> None:
    """Raise unless the prescan has at least two tasks."""
    if len(prescan.file_tasks) < 2:
        raise ValueError("a multi-file load needs at least two file tasks")


#############
# In memory #
#############


@beartype
def load_csr(
    directory: Path | str,
    counts: CsrLike,
    obs: pl.DataFrame,
    var: pl.DataFrame,
    *,
    qc: MinQualityParams | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load an in-memory cells x genes count matrix, as R's `load_r_data()`.

    obs and var go in as R's `populate_*_from_data.table` writes them: every
    column name snake_cased, the first column renamed to `cell_id` /
    `gene_id` (obs names then made unique), only the rows that survive QC.

    Args:
        directory: The experiment directory, created if missing.
        counts: Cells x genes CSR matrix of integral counts.
        obs: One row per matrix row, identifier first.
        var: One row per matrix column, identifier first.
        qc: Cell and gene QC thresholds and the target size.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
            `None` converts in one pass.
        verbose: Controls verbosity.

    Returns:
        The experiment.

    Raises:
        ValueError: If obs / var do not match the matrix or the counts are
            not integral.
    """
    qc = qc or MinQualityParams()
    n_cells, n_genes = (int(x) for x in counts.shape)
    if obs.height != n_cells or var.height != n_genes:
        raise ValueError(
            f"obs has {obs.height} and var {var.height} rows for a "
            f"{n_cells} x {n_genes} matrix"
        )
    path = _prepare(directory)
    data, indices, indptr = csr_buffers(counts)
    res = _bixverse.csr_to_file(
        path / CELLS_BIN,
        data,
        indices,
        indptr,
        (n_cells, n_genes),
        to_rust(qc),
        verbose,
    )

    def populate(con: duckdb.DuckDBPyConnection) -> None:
        names = _bixverse.snake_case(obs.columns)
        names[0] = "cell_id"
        obs_f = obs.rename(dict(zip(obs.columns, _make_unique(names), strict=True)))
        names = _bixverse.snake_case(var.columns)
        names[0] = "gene_id"
        var_f = var.rename(dict(zip(var.columns, names, strict=True)))
        db.populate_from_frame(con, "obs", obs_f[res["cell_indices"].astype(np.int64)])
        db.populate_from_frame(con, "var", var_f[res["gene_indices"].astype(np.int64)])

    return _finish(
        path,
        res["nnz"],
        res["lib_size"],
        csc_mem_gb=csc_mem_gb,
        verbose=verbose,
        populate=populate,
    )


########
# h5ad #
########


@beartype
def load_h5ad(
    directory: Path | str,
    h5_path: Path | str,
    *,
    qc: MinQualityParams | None = None,
    raw_count_slot: Literal["auto"] | RawSlot = "auto",
    cell_id_col: str | None = None,
    h5ad_streaming: bool = True,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load an h5ad file, as R's `load_h5ad()`.

    Args:
        directory: The experiment directory, created if missing.
        h5_path: The h5ad file.
        qc: Cell and gene QC thresholds and the target size.
        raw_count_slot: Where the raw counts are. `"auto"` detects it, see
            `detect_raw_count_slot`.
        cell_id_col: obs column to use as cell identifier when the index is
            not it.
        h5ad_streaming: Stream the counts in batches instead of materialising
            the filtered matrix first.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The experiment.

    Raises:
        ValueError: If `"auto"` finds no slot with raw counts.
    """
    qc = qc or MinQualityParams()
    h5 = Path(h5_path).expanduser()
    slot = detect_raw_count_slot(h5) if raw_count_slot == "auto" else raw_count_slot
    if slot is None:
        raise ValueError(
            "No raw count slot could be found in the object! Please validate the "
            "h5ad file"
        )
    dims = get_h5ad_dimensions(h5)
    path = _prepare(directory)
    res = _bixverse.h5ad_to_file(
        path / CELLS_BIN,
        h5,
        dims.type,
        dims.n_obs,
        dims.n_var,
        to_rust(qc),
        slot,
        h5ad_streaming,
        verbose,
    )
    return _finish(
        path,
        res["nnz"],
        res["lib_size"],
        csc_mem_gb=csc_mem_gb,
        verbose=verbose,
        populate=lambda con: _h5ad_obs_var(
            con, h5, res["cell_indices"], res["gene_indices"], cell_id_col
        ),
    )


@beartype
def stream_h5ad(
    directory: Path | str,
    h5_path: Path | str,
    *,
    qc: MinQualityParams | None = None,
    raw_count_slot: Literal["auto"] | RawSlot = "auto",
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """`load_h5ad(h5ad_streaming=True)`, kept as R keeps it."""
    return load_h5ad(
        directory,
        h5_path,
        qc=qc,
        raw_count_slot=raw_count_slot,
        h5ad_streaming=True,
        csc_mem_gb=csc_mem_gb,
        verbose=verbose,
    )


@beartype
def load_h5ad_norm(
    directory: Path | str,
    h5_path: Path | str,
    obs_lib_size_col: str,
    target_size: float,
    *,
    qc: MinQualityParams | None = None,
    cell_id_col: str | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load an h5ad file holding only normalised counts, as R's `load_h5ad_norm()`.

    Raw counts are reconstructed assuming `X = log1p(x / lib_size *
    target_size)`, with the library sizes read from an obs column.

    Args:
        directory: The experiment directory, created if missing.
        h5_path: The h5ad file.
        obs_lib_size_col: obs column with the total counts per cell.
        target_size: Target size of the original normalisation.
        qc: Cell and gene QC thresholds and the target size.
        cell_id_col: obs column to use as cell identifier.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The experiment.
    """
    qc = qc or MinQualityParams()
    h5 = Path(h5_path).expanduser()
    dims = get_h5ad_dimensions(h5)
    path = _prepare(directory)
    res = _bixverse.norm_h5ad_to_file(
        path / CELLS_BIN,
        h5,
        dims.type,
        dims.n_obs,
        dims.n_var,
        obs_lib_size_col,
        target_size,
        to_rust(qc),
        verbose,
    )
    return _finish(
        path,
        res["nnz"],
        res["lib_size"],
        csc_mem_gb=csc_mem_gb,
        verbose=verbose,
        populate=lambda con: _h5ad_obs_var(
            con, h5, res["cell_indices"], res["gene_indices"], cell_id_col
        ),
    )


@beartype
def load_multi_h5ad(
    directory: Path | str,
    prescan: Prescan,
    *,
    qc: MinQualityParams | None = None,
    cell_id_col: str | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load several h5ad files into one experiment, as R's `load_multi_h5ad()`.

    Gene QC is global across files and cells are indexed file after file.
    obs gains `exp_id` and `cell_id` is prefixed with it; var holds only
    `gene_id`.

    Args:
        directory: The experiment directory, created if missing.
        prescan: Output of `prescan_h5ad_files`.
        qc: Cell and gene QC thresholds and the target size.
        cell_id_col: obs column to use as cell identifier.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The experiment.
    """
    qc = qc or MinQualityParams()
    _check_tasks(prescan)
    path = _prepare(directory)
    res = _bixverse.multi_h5ad_to_file(
        path / CELLS_BIN,
        list(prescan.file_tasks),
        prescan.universe_size,
        to_rust(qc),
        verbose,
    )
    per_file = res["per_file"]

    def populate(con: duckdb.DuckDBPyConnection) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parts = []
            for i, f in enumerate(per_file):
                h5 = Path(str(prescan.task(f["exp_id"])["h5_path"]))
                out = Path(tmp) / f"obs_{i}.parquet"
                info = _bixverse.h5ad_obs_var_to_parquet(
                    h5, "obs", f["cell_indices"], cell_id_col, out
                )
                warn_skipped(info["skipped"], "/obs")
                parts.append(db.ParquetPart(out, _parquet_info(info), f["exp_id"]))
            db.parquet_parts_to_duckdb(con, "obs", parts)
        db.populate_var_minimal(con, _final_genes(prescan, res["global_gene_indices"]))

    nnz, lib = _per_file_qc(per_file)
    return _finish(
        path, nnz, lib, csc_mem_gb=csc_mem_gb, verbose=verbose, populate=populate
    )


#######
# mtx #
#######


@beartype
def get_cell_ranger_params(
    directory: Path | str, *, cells_as_rows: bool = False, has_hdr: bool = False
) -> MtxIoParams:
    """The mtx parameters for a Cell Ranger output directory.

    Picks the first file present of `matrix.mtx` / `mat.mtx`, `barcodes.tsv` /
    `barcodes.csv` and `features.*` / `genes.*` (Cell Ranger v3 and v2).

    Args:
        directory: The directory holding the three files.
        cells_as_rows: Do cells represent the rows of the mtx. Cell Ranger
            writes genes as rows.
        has_hdr: Do the barcode and feature files have headers.

    Returns:
        The parameters for `load_mtx()`.

    Raises:
        FileNotFoundError: If one of the three files is missing.
    """
    directory = Path(directory).expanduser()

    def pick(candidates: tuple[str, ...], label: str) -> str:
        for c in candidates:
            if (directory / c).exists():
                return str(directory / c)
        raise FileNotFoundError(
            f"No {label} file found in '{directory}'. "
            f"Looked for: {', '.join(candidates)}."
        )

    return MtxIoParams(
        path_mtx=pick(("matrix.mtx", "mat.mtx"), "matrix"),
        path_obs=pick(("barcodes.tsv", "barcodes.csv"), "barcode"),
        path_var=pick(
            ("features.tsv", "features.csv", "genes.tsv", "genes.csv"), "feature"
        ),
        cells_as_rows=cells_as_rows,
        has_hdr=has_hdr,
    )


@beartype
def load_mtx(
    directory: Path | str,
    mtx_params: MtxIoParams,
    *,
    qc: MinQualityParams | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load an mtx file with its barcodes and features, as R's `load_mtx()`.

    The `.mtx` (plain or gzipped) is read in one pass with its entries
    spilled next to the store. obs and var come from the plain text files,
    the first column as identifier.

    Args:
        directory: The experiment directory, created if missing.
        mtx_params: The three files and their layout.
        qc: Cell and gene QC thresholds and the target size.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The experiment.
    """
    qc = qc or MinQualityParams()
    path = _prepare(directory)
    res = _bixverse.mtx_to_file(
        path / CELLS_BIN,
        Path(mtx_params.path_mtx).expanduser(),
        to_rust(qc),
        mtx_params.cells_as_rows,
        True,
        verbose,
    )

    def populate(con: duckdb.DuckDBPyConnection) -> None:
        for table, file, keep in (
            ("obs", mtx_params.path_obs, res["cell_indices"]),
            ("var", mtx_params.path_var, res["gene_indices"]),
        ):
            db.plain_text_to_duckdb(
                con,
                Path(file).expanduser(),
                has_hdr=mtx_params.has_hdr,
                keep=keep,
                table=table,
            )

    return _finish(
        path,
        res["nnz"],
        res["lib_size"],
        csc_mem_gb=csc_mem_gb,
        verbose=verbose,
        populate=populate,
    )


@beartype
def load_multi_mtx(
    directory: Path | str,
    prescan: Prescan,
    *,
    qc: MinQualityParams | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load several mtx directories into one experiment, as R's `load_multi_mtx()`.

    Args:
        directory: The experiment directory, created if missing.
        prescan: Output of `prescan_mtx_dirs`.
        qc: Cell and gene QC thresholds and the target size.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The experiment.
    """
    qc = qc or MinQualityParams()
    _check_tasks(prescan)
    path = _prepare(directory)
    rust_tasks = [
        {
            k: t[k]
            for k in ("exp_id", "mtx_path", "cells_as_rows", "gene_local_to_universe")
        }
        for t in prescan.file_tasks
    ]
    res = _bixverse.multi_mtx_to_file(
        path / CELLS_BIN, rust_tasks, prescan.universe_size, to_rust(qc), verbose
    )
    per_file = res["per_file"]

    def populate(con: duckdb.DuckDBPyConnection) -> None:
        parts = [
            (
                Path(str(t["barcodes_path"])),
                str(t["exp_id"]),
                bool(t["has_hdr"]),
                f["cell_indices"],
            )
            for t, f in zip(prescan.file_tasks, per_file, strict=True)
        ]
        db.obs_from_multi_plain_text(con, parts)
        db.populate_var_minimal(con, _final_genes(prescan, res["global_gene_indices"]))

    nnz, lib = _per_file_qc(per_file)
    return _finish(
        path, nnz, lib, csc_mem_gb=csc_mem_gb, verbose=verbose, populate=populate
    )


##########
# 10x h5 #
##########


@beartype
def load_tenx_h5(
    directory: Path | str,
    h5_path: Path | str,
    *,
    qc: MinQualityParams | None = None,
    feature_type: str | None = "Gene Expression",
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load a 10x CellRanger h5 file, as R's `load_tenx_h5()`.

    Only the `feature_type` modality is loaded (v3); v2 is single modality.

    Args:
        directory: The experiment directory, created if missing.
        h5_path: The 10x h5 file.
        qc: Cell and gene QC thresholds and the target size.
        feature_type: Modality to keep. `None` is gene expression.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The experiment.
    """
    qc = qc or MinQualityParams()
    h5 = Path(h5_path).expanduser()
    meta = get_tenx_h5_metadata(h5)
    path = _prepare(directory)
    res = _bixverse.tenx_h5_to_file(
        path / CELLS_BIN,
        h5,
        meta.version,
        meta.n_cells,
        meta.n_genes,
        to_rust(qc),
        feature_type,
        verbose,
    )
    return _finish(
        path,
        res["nnz"],
        res["lib_size"],
        csc_mem_gb=csc_mem_gb,
        verbose=verbose,
        populate=lambda con: _tenx_obs_var(
            con, h5, meta.version, res["cell_indices"], res["gene_indices"]
        ),
    )


@beartype
def load_multi_tenx_h5(
    directory: Path | str,
    prescan: Prescan,
    *,
    qc: MinQualityParams | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Load several 10x h5 files into one experiment, as R's `load_multi_tenx_h5()`.

    Args:
        directory: The experiment directory, created if missing.
        prescan: Output of `prescan_tenx_h5_files`.
        qc: Cell and gene QC thresholds and the target size.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The experiment.
    """
    qc = qc or MinQualityParams()
    _check_tasks(prescan)
    path = _prepare(directory)
    res = _bixverse.multi_tenx_h5_to_file(
        path / CELLS_BIN,
        list(prescan.file_tasks),
        prescan.universe_size,
        to_rust(qc),
        verbose,
    )
    per_file = res["per_file"]

    def populate(con: duckdb.DuckDBPyConnection) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parts = []
            for i, f in enumerate(per_file):
                task = prescan.task(f["exp_id"])
                out = Path(tmp) / f"obs_{i}.parquet"
                info = _bixverse.tenx_obs_var_to_parquet(
                    Path(str(task["h5_path"])),
                    str(task["version"]),
                    "obs",
                    f["cell_indices"],
                    out,
                )
                parts.append(db.ParquetPart(out, _parquet_info(info), f["exp_id"]))
            db.parquet_parts_to_duckdb(con, "obs", parts)
        db.populate_var_minimal(con, _final_genes(prescan, res["global_gene_indices"]))

    nnz, lib = _per_file_qc(per_file)
    return _finish(
        path, nnz, lib, csc_mem_gb=csc_mem_gb, verbose=verbose, populate=populate
    )


###########
# Merging #
###########


@beartype
def merge_sc_experiments(
    directory: Path | str,
    inputs: Sequence[SingleCells],
    exp_ids: Sequence[str],
    *,
    renormalise: bool = False,
    qc: MinQualityParams | None = None,
    csc_mem_gb: float | None = None,
    verbose: bool = False,
) -> SingleCells:
    """Merge several experiments into a new one, as R's `merge_sc_experiments()`.

    The genes are the intersection of the inputs' genes, in the order of the
    first input. Only each input's kept cells are carried over. obs gains
    `exp_id`; var comes from the first input. Nothing cached is carried over.

    Args:
        directory: The target directory, created if missing.
        inputs: The experiments, at least two.
        exp_ids: One unique experiment id per input.
        renormalise: Recompute the normalised layer against `qc.target_size`
            from the surviving raw counts. Without it the stored layer is
            copied, which is only right if every input used the same target
            size.
        qc: Only `target_size` is used; no QC filtering happens in a merge.
        csc_mem_gb: Memory in GB for the CSR to CSC conversion buffers.
        verbose: Controls verbosity.

    Returns:
        The merged experiment.

    Raises:
        ValueError: On fewer than two inputs, mismatched or repeated ids, an
            input that was itself merged, or an empty intersection.
    """
    qc = qc or MinQualityParams()
    if len(inputs) < 2:
        raise ValueError("merging needs at least two inputs")
    if len(exp_ids) != len(inputs) or len(set(exp_ids)) != len(exp_ids):
        raise ValueError("exp_ids must be unique, one per input")
    for i, (inp, exp_id) in enumerate(zip(inputs, exp_ids, strict=True)):
        with db.connect(inp.db_path, read_only=True) as con:
            if "exp_id" in db.columns(con, "obs"):
                raise ValueError(
                    f"Input {i + 1} ('{exp_id}') already has an exp_id column. "
                    "Merging already-merged objects is not supported."
                )

    gene_sets = [inp.gene_ids().tolist() for inp in inputs]
    shared = set(gene_sets[0]).intersection(*gene_sets[1:])
    universe = list(dict.fromkeys(g for g in gene_sets[0] if g in shared))
    if not universe:
        raise ValueError("Gene intersection across inputs is empty.")
    lookup = {g: i for i, g in enumerate(universe)}

    tasks = [
        {
            "exp_id": exp_id,
            "bin_cells_path": str(inp.cells_path),
            "cells_to_keep": np.sort(inp.cells_to_keep).tolist(),
            "gene_local_to_universe": [lookup.get(g) for g in genes],
        }
        for inp, exp_id, genes in zip(inputs, exp_ids, gene_sets, strict=True)
    ]
    path = _prepare(directory)
    res = _bixverse.merge_sc_files(
        path / CELLS_BIN, tasks, len(universe), renormalise, qc.target_size, verbose
    )

    def populate(con: duckdb.DuckDBPyConnection) -> None:
        db.obs_from_multi_duckdb(
            con,
            [(inp.db_path, e) for inp, e in zip(inputs, exp_ids, strict=True)],
        )
        db.var_from_duckdb_reordered(con, inputs[0].db_path, universe)

    nnz, lib = _per_file_qc(res["per_file"])
    return _finish(
        path, nnz, lib, csc_mem_gb=csc_mem_gb, verbose=verbose, populate=populate
    )


#####################
# Store maintenance #
#####################


@beartype
def archive_sc_exp(
    sc: SingleCells,
    *,
    level: int = 3,
    remove_bins: bool = True,
    verbose: bool = False,
) -> dict[str, int]:
    """Archive the counts for cold storage, as R's `archive_sc_exp()`.

    Writes `counts.bxa`, a zstd archive of the cell-based store; the
    gene-based store is rebuilt on restore. `load_existing` restores the
    stores from the archive when they are missing.

    Args:
        sc: The experiment.
        level: zstd level, 1 to 22.
        remove_bins: Delete both stores afterwards.
        verbose: Controls verbosity.

    Returns:
        `n_cells`, `nnz`, `n_norm_stored` (cells whose normalised values
        were stored verbatim) and `archive_bytes`.

    Raises:
        ValueError: If `level` is outside 1 to 22.
    """
    if not 1 <= level <= 22:
        raise ValueError(f"level must lie in 1..22, got {level}")
    stats = _bixverse.archive_cells(
        sc.cells_path, sc.dir / ARCHIVE_NAME, level, verbose
    )
    if remove_bins:
        sc.cells_path.unlink(missing_ok=True)
        sc.genes_path.unlink(missing_ok=True)
    return stats


@beartype
def upgrade_sc_store(
    sc: SingleCells,
    *,
    codec: Literal["lz4", "zstd", "none"] = "lz4",
    keep_backup: bool = True,
    verbose: bool = False,
) -> SingleCells:
    """Convert a v3 store to the v4 format in place, as R's `upgrade_sc_store()`.

    Converts into temporary files, checks them and swaps them in, so a
    failure leaves the store untouched. Nothing else may hold the files open
    meanwhile. A store already at v4 is left alone.

    Args:
        sc: The experiment.
        codec: Chunk codec of the new files.
        keep_backup: Keep the originals as `<path>.v3.bak`.
        verbose: Controls verbosity.

    Returns:
        The experiment, reading the converted store.
    """
    versions = [_bixverse.file_version(p) for p in (sc.cells_path, sc.genes_path)]
    if min(versions) >= 4:
        if verbose:
            logger.info("The store is already in the v4 format. Nothing to do.")
        return sc
    _bixverse.upgrade_to_v4(sc.cells_path, sc.genes_path, codec, keep_backup, verbose)
    sc.reader = _bixverse.CellReader(str(sc.cells_path))
    return sc
