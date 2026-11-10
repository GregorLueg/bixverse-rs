"""Writing h5ad, 10x h5 and CellRanger mtx files.

`save_h5ad` exports an experiment for ScanPy. The other three write fixtures
from in-memory counts; tests and examples use them to produce input for the
loaders.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.ingest import CsrLike, csr_buffers
from bixverse._core.single_cells import SingleCells
from bixverse._core.state import CsrMatrix
from bixverse._validate import Verbose

###########
# Globals #
###########

logger = logging.getLogger(__name__)

H5adColumn = tuple[str, str, object]

INT32_MAX: int = 2**31 - 1


###########
# Helpers #
###########


@beartype
def _check_output(path: Path, *, overwrite: bool) -> None:
    """Remove an existing output file, or refuse to.

    Raises:
        FileExistsError: If it exists and `overwrite` is false.
    """
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists and overwrite = False.")
        path.unlink()


@beartype
def h5ad_columns(frame: pl.DataFrame, *, factorise: bool) -> list[H5adColumn]:
    """Encode frame columns for the h5ad writer.

    - with `factorise`, a string column with repeated values becomes a
      categorical, which is what pandas would hold it as (levels sorted by
      code point)
    - a categorical without levels becomes the literal string `"NA"`
    - integers and booleans with missing values are widened to float, the
      only one of the three with a representation for them; so are integers
      beyond `int32`
    - strings keep missing values as `"NA"`, anything else is stringified

    Args:
        frame: The columns, without the index.
        factorise: Turn repeated string columns into categoricals.

    Returns:
        One `(name, kind, payload)` per column, see the Rust writer.
    """
    out: list[H5adColumn] = []
    for name in frame.columns:
        s = frame[name]
        if factorise and s.dtype == pl.String and s.n_unique() < s.len():
            s = s.cast(pl.Enum(sorted(s.drop_nulls().unique().to_list())))
        dtype = s.dtype
        if isinstance(dtype, (pl.Enum, pl.Categorical)):
            levels = (
                dtype.categories.to_list()
                if isinstance(dtype, pl.Enum)
                else sorted(s.drop_nulls().cast(pl.String).unique().to_list())
            )
            if not levels:
                out.append((name, "string", ["NA"] * s.len()))
                continue
            codes = (
                s.cast(pl.String)
                .cast(pl.Enum(levels))
                .to_physical()
                .cast(pl.Int32)
                .fill_null(-1)
            )
            out.append((name, "categorical", (codes.to_numpy(), levels)))
        elif dtype == pl.String:
            out.append((name, "string", s.fill_null("NA").to_list()))
        elif dtype == pl.Boolean:
            if s.null_count():
                out.append(
                    (name, "float", s.cast(pl.Float64).fill_null(np.nan).to_numpy())
                )
            else:
                out.append((name, "bool", s.to_numpy()))
        elif dtype.is_integer():
            fits = s.null_count() == 0 and bool(
                s.is_between(-INT32_MAX, INT32_MAX).all()
            )
            if fits:
                out.append((name, "int", s.cast(pl.Int32).to_numpy()))
            else:
                out.append(
                    (name, "float", s.cast(pl.Float64).fill_null(np.nan).to_numpy())
                )
        elif dtype.is_float():
            out.append((name, "float", s.cast(pl.Float64).fill_null(np.nan).to_numpy()))
        elif dtype == pl.Null:
            out.append((name, "string", ["NA"] * s.len()))
        else:
            out.append((name, "string", s.cast(pl.String).fill_null("NA").to_list()))
    return out


@beartype
def _ids(frame: pl.DataFrame) -> list[str]:
    """The first column of a frame as strings, the index of the written table."""
    return frame[frame.columns[0]].cast(pl.String).to_list()


@beartype
def _permute_square(m: CsrMatrix, order: np.ndarray) -> CsrMatrix:
    """Rows and columns of a square CSR matrix reordered: new `i` is old `order[i]`.

    Indices come back sorted within rows, as sprs requires.
    """
    n = m.shape[0]
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    counts = np.diff(m.indptr.astype(np.int64))
    rows = inverse[np.repeat(np.arange(n), counts)]
    cols = inverse[m.indices.astype(np.int64)]
    perm = np.lexsort((cols, rows))
    indptr = np.zeros(n + 1, dtype=np.uint32)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return CsrMatrix(
        m.data[perm],
        cols[perm].astype(np.uint32),
        indptr,
        m.shape,
    )


@beartype
def _fresh(sc: SingleCells, label: str, n_rows: int, n_cells: int) -> bool:
    """Whether a cached artefact may be exported; warns when it may not."""
    if n_rows != n_cells:
        warnings.warn(
            f"'{label}' has {n_rows} rows but {n_cells} cells are being written. "
            "Skipping it in the h5ad export.",
            stacklevel=3,
        )
        return False
    state = sc.check_state([label])
    if state is not True:
        warnings.warn(f"Skipping '{label}' in the h5ad export: {state}", stacklevel=3)
        return False
    return True


@beartype
def _export_cache(
    sc: SingleCells, order: np.ndarray
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    str,
]:
    """The cached embeddings and graph.

    PCA factors go to `obsm/X_pca`, the loadings (padded out to every gene
    from the genes the PCA ran on, and only with a fresh `X_pca`) to
    `varm/PCs`, every other embedding to
    `obsm/X_<name>`, the neighbour graph to `obsp/connectivities` with the
    `uns/neighbors` block ScanPy looks for. Cached rows follow
    `cells_to_keep`; they are reordered to the ascending cell order of the
    file.

    Args:
        sc: The experiment.
        order: Positions in `cells_to_keep` of the written cells, in file
            order.

    Returns:
        `obsm`, `varm`, `obsp` and the `uns` JSON.
    """
    c = sc.cache
    n_cells, n_genes = len(order), sc.n_genes
    obsm: dict[str, np.ndarray] = {}
    varm: dict[str, np.ndarray] = {}
    obsp: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    uns: dict[str, object] = {}

    embeddings = dict(c.embeddings)
    if c.pca_factors is not None:
        embeddings = {"pca": c.pca_factors, **embeddings}
    for name, emb in embeddings.items():
        if _fresh(sc, name, emb.shape[0], n_cells):
            obsm[f"X_{name}"] = np.asarray(emb, dtype=np.float64)[order]

    if c.pca_loadings is not None and "X_pca" in obsm:
        # the genes the PCA ran on, not the current HVGs, which may have moved
        genes = c.pca_genes
        if genes is not None and len(genes) == c.pca_loadings.shape[0]:
            padded = np.zeros((n_genes, c.pca_loadings.shape[1]), dtype=np.float64)
            padded[genes] = c.pca_loadings
            varm["PCs"] = padded
        else:
            warnings.warn(
                "The cached PCA loadings do not record the genes they cover. "
                "Skipping varm/PCs in the h5ad export; re-run calculate_pca.",
                stacklevel=2,
            )

    g = c.connectivities
    if g is not None and _fresh(sc, "connectivities", g.shape[0], n_cells):
        g = _permute_square(g, order)
        obsp["connectivities"] = (g.data, g.indices, g.indptr.astype(np.uint32))
        uns["neighbors"] = {"connectivities_key": "connectivities"}

    return obsm, varm, obsp, json.dumps(uns)


###########
# Writers #
###########


@beartype
def save_h5ad(
    sc: SingleCells,
    h5_path: Path | str,
    *,
    assay: Literal["raw", "norm"] = "raw",
    chunk_size: int = 10_000,
    overwrite: bool = True,
    verbose: Verbose = True,
) -> Path:
    """Write an experiment to a spec compliant h5ad file.

    Writes the kept cells in ascending store order, the obs / var tables
    without the bookkeeping columns, and whatever cached embeddings and
    graph still match the cells. Counts are streamed from the store in
    batches and stored as `float32`. A failed write leaves no file behind.

    Args:
        sc: The experiment.
        h5_path: File to write.
        assay: Which counts go into `X`.
        chunk_size: Cells per streaming batch.
        overwrite: Replace an existing file.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The written path.

    Raises:
        FileExistsError: If the file exists and `overwrite` is false.
        ValueError: If `chunk_size` is not positive.

    Examples:
        >>> bv.io.save_h5ad(sc, "path/to/out.h5ad", assay="norm")
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    path = Path(h5_path).expanduser()
    _check_output(path, overwrite=overwrite)

    keep = sc.cells_to_keep
    order = np.argsort(keep, kind="stable")
    cells = np.ascontiguousarray(keep[order], dtype=np.uint32)

    obs = sc.obs(filtered=True).sort("cell_idx")
    obs = obs.select(
        "cell_id",
        *[c for c in obs.columns if c not in ("cell_idx", "cell_id", "to_keep")],
    )
    var = sc.var().sort("gene_idx").drop("gene_idx")
    var = var.select("gene_id", *[c for c in var.columns if c != "gene_id"])
    if obs.height != len(cells) or var.height != sc.n_genes:
        raise ValueError("The obs / var tables do not match the cells being written.")

    obsm, varm, obsp, uns_json = _export_cache(sc, order)
    if verbose:
        logger.info("Writing %i cells x %i genes to h5ad.", obs.height, var.height)
    try:
        _bixverse.save_h5ad(
            sc.cells_path,
            path,
            cells,
            assay == "norm",
            _ids(obs),
            h5ad_columns(obs.drop("cell_id"), factorise=True),
            _ids(var),
            h5ad_columns(var.drop("gene_id"), factorise=True),
            obsm,
            varm,
            obsp,
            uns_json,
            chunk_size,
        )
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


@beartype
def write_h5ad_sc(
    f_path: Path | str,
    counts: CsrLike,
    obs: pl.DataFrame,
    var: pl.DataFrame,
    *,
    overwrite: bool = True,
) -> None:
    """Write in-memory counts to a sparse h5ad file.

    The first column of `obs` / `var` becomes the index, the others are
    written as they are typed. `X` is CSR and stored as `float32`.

    Args:
        f_path: File to write.
        counts: Cells x genes CSR matrix.
        obs: One row per cell, identifier first.
        var: One row per gene, identifier first.
        overwrite: Replace an existing file.

    Raises:
        ValueError: If obs / var do not match the matrix.

    Examples:
        >>> bv.io.write_h5ad_sc("path/to/file.h5ad", sim.counts, sim.obs, sim.var)
    """
    n_cells, n_genes = (int(x) for x in counts.shape)
    if obs.height != n_cells or var.height != n_genes:
        raise ValueError("obs and var must have one row per cell and per gene")
    fmt = getattr(counts, "format", "csr")
    if fmt != "csr":
        raise TypeError(f"counts must be a CSR matrix, got format {fmt!r}")
    path = Path(f_path).expanduser()
    _check_output(path, overwrite=overwrite)
    _bixverse.write_h5ad(
        path,
        np.ascontiguousarray(counts.data, dtype=np.float32),
        np.ascontiguousarray(counts.indices, dtype=np.uint32),
        np.ascontiguousarray(counts.indptr, dtype=np.uint64),
        _ids(obs),
        h5ad_columns(obs.drop(obs.columns[0]), factorise=False),
        _ids(var),
        h5ad_columns(var.drop(var.columns[0]), factorise=False),
    )


@beartype
def write_tenx_h5_sc(
    f_path: Path | str,
    counts: CsrLike,
    barcodes: Sequence[str],
    features: pl.DataFrame,
    *,
    version: Literal["v3", "v2"] = "v3",
    overwrite: bool = True,
) -> None:
    """Write in-memory counts to a 10x h5 file.

    Args:
        f_path: File to write.
        counts: Cells x features CSR matrix of integral counts.
        barcodes: One per cell.
        features: `id` and `name`, plus optionally `feature_type` (v3; all
            `"Gene Expression"` when absent), one row per feature.
        version: Layout to write.
        overwrite: Replace an existing file.

    Raises:
        ValueError: If the inputs do not match the matrix.

    Examples:
        >>> barcodes = sim.obs["cell_id"].to_list()
        >>> features = sim.var.select(id="gene_id", name="gene_id")
        >>> bv.io.write_tenx_h5_sc("path/to/file.h5", sim.counts, barcodes, features)
    """
    n_cells, n_features = (int(x) for x in counts.shape)
    if len(barcodes) != n_cells or features.height != n_features:
        raise ValueError("barcodes and features must match the matrix")
    if not {"id", "name"} <= set(features.columns):
        raise ValueError("features needs `id` and `name` columns")
    path = Path(f_path).expanduser()
    _check_output(path, overwrite=overwrite)
    data, indices, indptr = csr_buffers(counts)
    types = (
        features["feature_type"].cast(pl.String).to_list()
        if "feature_type" in features.columns
        else None
    )
    _bixverse.write_tenx(
        path,
        version,
        data,
        indices,
        indptr,
        (n_cells, n_features),
        list(barcodes),
        features["id"].cast(pl.String).to_list(),
        features["name"].cast(pl.String).to_list(),
        types,
    )


@beartype
def write_cellranger_output(
    f_path: Path | str,
    counts: CsrLike,
    obs: pl.DataFrame,
    var: pl.DataFrame,
    *,
    format_type: Literal["csv", "tsv"] = "csv",
    rows: Literal["cells", "genes"] = "cells",
    overwrite: bool = True,
) -> None:
    """Write counts as a CellRanger style trio.

    Writes `matrix.mtx` (integer coordinate format, 1-based) plus
    `barcodes.<format_type>` and `features.<format_type>` with a header row.

    Args:
        f_path: Directory to write into, created if missing.
        counts: Cells x genes CSR matrix of integral counts.
        obs: One row per cell.
        var: One row per gene.
        format_type: `"csv"` or `"tsv"` for the two tables.
        rows: Whether the rows of the `.mtx` are cells or genes.
        overwrite: Replace existing files.

    Raises:
        FileExistsError: If a file exists and `overwrite` is false.

    Examples:
        >>> out = "path/to/mtx_dir"
        >>> bv.io.write_cellranger_output(
        ...     out, sim.counts, sim.obs, sim.var, rows="genes"
        ... )
    """
    n_cells, n_genes = (int(x) for x in counts.shape)
    if obs.height != n_cells or var.height != n_genes:
        raise ValueError("obs and var must have one row per cell and per gene")
    out = Path(f_path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    f_mtx = out / "matrix.mtx"
    f_obs, f_var = out / f"barcodes.{format_type}", out / f"features.{format_type}"
    if not overwrite and any(p.exists() for p in (f_mtx, f_obs, f_var)):
        raise FileExistsError(
            "The to be written files already exist and overwrite = False."
        )

    data, indices, indptr = csr_buffers(counts)
    row_idx = np.repeat(np.arange(1, n_cells + 1), np.diff(indptr.astype(np.int64)))
    col_idx = indices.astype(np.int64) + 1
    if rows == "cells":
        dims = (n_cells, n_genes)
    else:
        dims = (n_genes, n_cells)
        row_idx, col_idx = col_idx, row_idx

    with f_mtx.open("w") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n")
        fh.write(
            f"%Rows={rows} ({dims[0]}), "
            f"Cols={'genes' if rows == 'cells' else 'cells'} ({dims[1]})\n"
        )
        fh.write(f"{dims[0]} {dims[1]} {len(data)}\n")
        np.savetxt(fh, np.column_stack([row_idx, col_idx, data]), fmt="%d")

    sep = "," if format_type == "csv" else "\t"
    obs.write_csv(f_obs, separator=sep)
    var.write_csv(f_var, separator=sep)
