"""Metadata of h5ad, 10x h5 and mtx inputs, and the prescans of multi-file loads.

The reads go through the Rust core, which statically links its own HDF5, so
no h5py is needed.
"""

from __future__ import annotations

import re
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db

###########
# Globals #
###########

RawSlot = Literal["X", "raw.X", "layers.counts"]
GeneUniverse = Literal["intersection", "union"]

RAW_SLOT_CANDIDATES: tuple[RawSlot, ...] = ("layers.counts", "raw.X", "X")

InputPaths = Sequence[Path | str] | Mapping[str, Path | str]


###########
# Records #
###########


@dataclass(frozen=True)
class H5adDims:
    """Dimensions and storage of an h5ad file, as `get_h5ad_dimensions()`.

    Attributes:
        n_obs: Rows of obs.
        n_var: Rows of var.
        type: `"CSR"`, `"CSC"`, `"DENSE_ROW"` or `"DENSE_COL"`.
    """

    n_obs: int
    n_var: int
    type: str


@dataclass(frozen=True)
class TenxH5Meta:
    """Layout and dimensions of a 10x h5 file, as `get_tenx_h5_metadata()`.

    Attributes:
        version: `"v2"` or `"v3"`.
        n_cells: Barcodes.
        n_genes: Features, every modality included.
    """

    version: str
    n_cells: int
    n_genes: int


@dataclass(frozen=True)
class Prescan:
    """The gene universe and per-file tasks of a multi-file load.

    Attributes:
        universe: Gene ids in the order of the final var table.
        file_tasks: One task dict per input, in input order. The keys are the
            ones the prescans produce and the Rust loaders read;
            `gene_local_to_universe` is 0-based with `None` for genes outside
            the universe.
    """

    universe: tuple[str, ...]
    file_tasks: tuple[dict[str, object], ...] = field(repr=False)

    @property
    def universe_size(self) -> int:
        """Number of genes in the universe."""
        return len(self.universe)

    @beartype
    def task(self, exp_id: str) -> dict[str, object]:
        """The task of one input.

        Raises:
            KeyError: If no input has that `exp_id`.
        """
        for t in self.file_tasks:
            if t["exp_id"] == exp_id:
                return t
        raise KeyError(exp_id)


###########
# Helpers #
###########


@beartype
def _exp_ids(paths: InputPaths) -> tuple[list[str], list[Path]]:
    """Experiment ids (mapping keys, else file stems) and expanded paths.

    Raises:
        ValueError: On fewer than two inputs or repeated ids.
        FileNotFoundError: If an input is missing.
    """
    if isinstance(paths, Mapping):
        ids, files = list(paths.keys()), [Path(p) for p in paths.values()]
    else:
        files = [Path(p) for p in paths]
        ids = [p.stem for p in files]
    files = [p.expanduser() for p in files]
    if len(files) < 2:
        raise ValueError("a multi-file load needs at least two inputs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"experiment ids must be unique, got {ids}")
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
    return ids, files


@beartype
def _universe(gene_sets: list[list[str]], how: GeneUniverse) -> list[str]:
    """Intersection or union of gene sets, sorted.

    Sorts by code point, so mixed case gene ids order upper case first.
    """
    match how:
        case "intersection":
            out = set(gene_sets[0]).intersection(*gene_sets[1:])
        case "union":
            out = set().union(*gene_sets)
        case _:
            raise ValueError(f"unknown gene_universe {how!r}")
    return sorted(out)


@beartype
def _mapping(genes: list[str], lookup: dict[str, int]) -> list[int | None]:
    """0-based universe index per local gene, `None` outside the universe."""
    return [lookup.get(g) for g in genes]


########
# h5ad #
########


@beartype
def get_h5ad_dimensions(f_path: Path | str) -> H5adDims:
    """Dimensions and storage layout of an h5ad file.

    Sparse `X` is `CSR` or `CSC` by its encoding attribute (or the length of
    `indptr`). Dense `X` is `DENSE_ROW` (cells x genes on disk) or
    `DENSE_COL`, with ties going to `DENSE_ROW`.

    Args:
        f_path: The h5ad file.

    Returns:
        The dimensions and layout.

    Raises:
        FileNotFoundError: If the file is missing.
        BixverseError: If `X` is missing or its shape fits neither ordering.
    """
    path = Path(f_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    n_obs, n_var, kind = _bixverse.h5ad_dimensions(path)
    return H5adDims(n_obs=n_obs, n_var=n_var, type=kind)


@beartype
def detect_raw_count_slot(
    f_path: Path | str,
    candidates: Sequence[RawSlot] = RAW_SLOT_CANDIDATES,
    *,
    n_sample: int = 10_000,
    threshold: float = 0.99,
) -> RawSlot | None:
    """Detect which slot of an h5ad file holds raw integer counts.

    Samples the non-zero values of each candidate in order and returns the
    first that is integer valued, so dedicated count slots win over `X`.

    Args:
        f_path: The h5ad file.
        candidates: Slots to test, in priority order.
        n_sample: Values to sample per slot.
        threshold: Minimum fraction of non-zeros that must be whole numbers.

    Returns:
        The slot, or `None` if none qualifies.

    Examples:
        >>> slot = bv.io.detect_raw_count_slot("path/to/file.h5ad")
    """
    path = Path(f_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    found = _bixverse.raw_count_slot(path, list(candidates), n_sample, threshold)
    match found:
        case None | "X" | "raw.X" | "layers.counts":
            return found
        case _:
            raise AssertionError(f"unexpected slot {found!r}")


@beartype
def _h5ad_frame(path: Path, frame: Literal["obs", "var"]) -> pl.DataFrame:
    """An obs / var dataframe of an h5ad file, through the Parquet exporter."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{frame}.parquet"
        info = _bixverse.h5ad_obs_var_to_parquet(path, frame, None, None, out)
        warn_skipped(info["skipped"], f"/{frame}")
        table = pl.read_parquet(out)
    idx_col, _ = db.TABLE_KEYS[frame]
    return table.drop(idx_col)


@beartype
def warn_skipped(skipped: Sequence[str], group: str) -> None:
    """Warn once per metadata column the Parquet writer left out."""
    for col in skipped:
        warnings.warn(
            f"Skipping {group}/{col}: unsupported h5ad encoding.", stacklevel=3
        )


@beartype
def read_h5ad_metadata(f_path: Path | str) -> dict[str, object]:
    """Read the obs and var tables and the layout of an h5ad file.

    The tables come through the same reader the loaders use, so column
    names are snake_cased and the index is `cell_id` / `gene_id`.

    Args:
        f_path: The h5ad file.

    Returns:
        `obs` and `var` (polars), `dims` (`(n_obs, n_var)`) and `type`.

    Examples:
        >>> meta = bv.io.read_h5ad_metadata("path/to/file.h5ad")
    """
    path = Path(f_path).expanduser()
    dims = get_h5ad_dimensions(path)
    return {
        "obs": _h5ad_frame(path, "obs"),
        "var": _h5ad_frame(path, "var"),
        "dims": (dims.n_obs, dims.n_var),
        "type": dims.type,
    }


@beartype
def prescan_h5ad_files(
    h5_paths: InputPaths,
    *,
    gene_universe: GeneUniverse = "intersection",
    var_index: str = "_index",
    raw_count_slot: Literal["auto"] | RawSlot = "auto",
) -> Prescan:
    """Prescan several h5ad files for a multi-file load.

    Args:
        h5_paths: The files. A mapping supplies the experiment ids, otherwise
            the file stems are used.
        gene_universe: `"intersection"` or `"union"` of the gene names.
        var_index: Dataset in `/var` holding the gene names.
        raw_count_slot: `"auto"` detects per file, see
            `detect_raw_count_slot`.

    Returns:
        The universe (sorted) and one task per file with `exp_id`,
        `h5_path`, `cs_type`, `no_cells`, `no_genes`, `raw_slot` and
        `gene_local_to_universe`.

    Examples:
        >>> files = {"b1": "path/to/b1.h5ad", "b2": "path/to/b2.h5ad"}
        >>> prescan = bv.io.prescan_h5ad_files(files, gene_universe="union")
    """
    ids, files = _exp_ids(h5_paths)
    metas = []
    for exp_id, path in zip(ids, files, strict=True):
        dims = get_h5ad_dimensions(path)
        genes = _bixverse.h5_strings(path, f"var/{var_index}")
        slot = (
            detect_raw_count_slot(path) if raw_count_slot == "auto" else raw_count_slot
        )
        if slot is None:
            warnings.warn(f"Could not detect a raw count slot for {path}", stacklevel=2)
        metas.append((exp_id, path, dims, genes, slot))

    universe = _universe([m[3] for m in metas], gene_universe)
    lookup = {g: i for i, g in enumerate(universe)}
    tasks: tuple[dict[str, object], ...] = tuple(
        {
            "exp_id": exp_id,
            "h5_path": str(path),
            "cs_type": dims.type,
            "no_cells": dims.n_obs,
            "no_genes": dims.n_var,
            "raw_slot": slot,
            "gene_local_to_universe": _mapping(genes, lookup),
        }
        for exp_id, path, dims, genes, slot in metas
    )
    return Prescan(universe=tuple(universe), file_tasks=tasks)


#######
# 10x #
#######


@beartype
def get_tenx_h5_metadata(f_path: Path | str) -> TenxH5Meta:
    """Detect the layout of a 10x h5 file and read its dimensions.

    Args:
        f_path: The 10x h5 file.

    Returns:
        The version and dimensions.

    Raises:
        FileNotFoundError: If the file is missing.
        BixverseError: If neither the v2 nor the v3 layout is found.
    """
    path = Path(f_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    version, n_cells, n_genes = _bixverse.tenx_h5_metadata(path)
    return TenxH5Meta(version=version, n_cells=n_cells, n_genes=n_genes)


@beartype
def read_tenx_h5_metadata(f_path: Path | str) -> dict[str, object]:
    """Read the barcode and feature tables and the layout of a 10x h5 file.

    Args:
        f_path: The 10x h5 file.

    Returns:
        `obs` (`.id`), `var` (`.id`, `name`, plus `feature_type` for v3),
        `dims` (`(n_cells, n_genes)`), `version` and `feature_types` (counts
        per type, v3 only, else `None`).

    Examples:
        >>> meta = bv.io.read_tenx_h5_metadata("path/to/file.h5")
    """
    path = Path(f_path).expanduser()
    meta = get_tenx_h5_metadata(path)
    v3 = meta.version == "v3"
    obs = pl.DataFrame(
        {".id": _bixverse.h5_strings(path, "matrix/barcodes" if v3 else "barcodes")}
    )
    if v3:
        var = pl.DataFrame(
            {
                ".id": _bixverse.h5_strings(path, "matrix/features/id"),
                "name": _bixverse.h5_strings(path, "matrix/features/name"),
                "feature_type": _bixverse.h5_strings(
                    path, "matrix/features/feature_type"
                ),
            }
        )
        counts = var["feature_type"].value_counts().sort("feature_type")
        feature_types: dict[str, int] | None = dict(
            zip(
                counts["feature_type"].to_list(), counts["count"].to_list(), strict=True
            )
        )
    else:
        var = pl.DataFrame(
            {
                ".id": _bixverse.h5_strings(path, "genes"),
                "name": _bixverse.h5_strings(path, "gene_names"),
            }
        )
        feature_types = None
    return {
        "obs": obs,
        "var": var,
        "dims": (meta.n_cells, meta.n_genes),
        "version": meta.version,
        "feature_types": feature_types,
    }


@beartype
def prescan_tenx_h5_files(
    h5_paths: InputPaths,
    *,
    feature_type: str = "Gene Expression",
    gene_universe: GeneUniverse = "intersection",
) -> Prescan:
    """Prescan several 10x h5 files for a multi-file load.

    Genes are matched on the feature id. v2 and v3 files can be mixed; the
    `feature_type` filter only applies to v3.

    Args:
        h5_paths: The files. A mapping supplies the experiment ids, otherwise
            the file stems are used.
        feature_type: Modality to keep (v3).
        gene_universe: `"intersection"` or `"union"` of the feature ids.

    Returns:
        The universe (sorted) and one task per file with `exp_id`,
        `h5_path`, `version`, `no_cells`, `no_genes`, `feature_type` and
        `gene_local_to_universe`.

    Raises:
        ValueError: If a v3 file has no feature of the type, or the universe
            is empty.

    Examples:
        >>> files = ["path/to/t1.h5", "path/to/t2.h5"]
        >>> prescan = bv.io.prescan_tenx_h5_files(files)
    """
    ids, files = _exp_ids(h5_paths)
    metas = []
    for exp_id, path in zip(ids, files, strict=True):
        meta = get_tenx_h5_metadata(path)
        if meta.version == "v3":
            all_ids = _bixverse.h5_strings(path, "matrix/features/id")
            types = _bixverse.h5_strings(path, "matrix/features/feature_type")
            keep = [t == feature_type for t in types]
            if not any(keep):
                raise ValueError(
                    f"No features of type '{feature_type}' found in {path}. "
                    f"Available: {', '.join(dict.fromkeys(types))}"
                )
        else:
            all_ids = _bixverse.h5_strings(path, "genes")
            keep = [True] * len(all_ids)
        metas.append((exp_id, path, meta, all_ids, keep))

    universe = _universe(
        [[g for g, k in zip(m[3], m[4], strict=True) if k] for m in metas],
        gene_universe,
    )
    if not universe:
        raise ValueError("Gene universe across inputs is empty.")
    lookup = {g: i for i, g in enumerate(universe)}
    tasks: tuple[dict[str, object], ...] = tuple(
        {
            "exp_id": exp_id,
            "h5_path": str(path),
            "version": meta.version,
            "no_cells": meta.n_cells,
            "no_genes": meta.n_genes,
            "feature_type": feature_type,
            "gene_local_to_universe": [
                lookup.get(g) if k else None for g, k in zip(all_ids, keep, strict=True)
            ],
        }
        for exp_id, path, meta, all_ids, keep in metas
    )
    return Prescan(universe=tuple(universe), file_tasks=tasks)


#######
# mtx #
#######


@beartype
def _locate(directory: Path, pattern: str) -> Path:
    """The one file in a directory matching a pattern, case insensitive.

    Raises:
        FileNotFoundError: If none matches.
        ValueError: If several do.
    """
    hits = sorted(
        p for p in directory.iterdir() if re.search(pattern, p.name, re.IGNORECASE)
    )
    if not hits:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    if len(hits) > 1:
        raise ValueError(f"Multiple files matching '{pattern}' in {directory}")
    return hits[0]


@beartype
def prescan_mtx_dirs(
    dirs: Sequence[Path | str],
    exp_ids: Sequence[str],
    *,
    cells_as_rows: bool = False,
    has_hdr: bool = False,
) -> Prescan:
    """Prescan several CellRanger style mtx directories for a multi-file load.

    Each directory holds one `.mtx` (or `.mtx.gz`), one barcodes and one
    features (or genes) file. The universe is the intersection of the first
    feature column, in the order of the first directory.

    Args:
        dirs: The directories, at least two.
        exp_ids: One unique experiment id per directory.
        cells_as_rows: Applied to every input. `False` is the 10x convention.
        has_hdr: Whether the barcodes and features files have a header.

    Returns:
        The universe and one task per directory with `exp_id`, `mtx_path`,
        `barcodes_path`, `features_path`, `cells_as_rows`, `has_hdr` and
        `gene_local_to_universe`.

    Raises:
        ValueError: On fewer than two inputs, mismatched or repeated ids, or
            an empty intersection.

    Examples:
        >>> dirs = ["path/to/mtx_a", "path/to/mtx_b"]
        >>> prescan = bv.io.prescan_mtx_dirs(dirs, ["a", "b"], has_hdr=True)
    """
    paths = [Path(d).expanduser() for d in dirs]
    if len(paths) < 2:
        raise ValueError("a multi-file load needs at least two inputs")
    if len(exp_ids) != len(paths) or len(set(exp_ids)) != len(exp_ids):
        raise ValueError("exp_ids must be unique, one per directory")
    for d in paths:
        if not d.is_dir():
            raise FileNotFoundError(f"{d} is not a directory")

    tasks: list[dict[str, object]] = []
    gene_sets: list[list[str]] = []
    for exp_id, d in zip(exp_ids, paths, strict=True):
        features = _locate(d, r"(features|genes)\.(tsv|csv)(\.gz)?$")
        feats = pl.read_csv(
            features,
            separator=db.plain_text_delimiter(features),
            has_header=has_hdr,
            infer_schema=False,
        )
        genes = feats[feats.columns[0]].to_list()
        gene_sets.append(genes)
        tasks.append(
            {
                "exp_id": exp_id,
                "mtx_path": str(_locate(d, r"\.mtx(\.gz)?$")),
                "barcodes_path": str(_locate(d, r"barcodes\.(tsv|csv)(\.gz)?$")),
                "features_path": str(features),
                "cells_as_rows": cells_as_rows,
                "has_hdr": has_hdr,
            }
        )

    shared = set(gene_sets[0]).intersection(*gene_sets[1:])
    universe = list(dict.fromkeys(g for g in gene_sets[0] if g in shared))
    if not universe:
        raise ValueError("Gene intersection across inputs is empty.")
    lookup = {g: i for i, g in enumerate(universe)}
    for task, genes in zip(tasks, gene_sets, strict=True):
        task["gene_local_to_universe"] = _mapping(genes, lookup)
    return Prescan(universe=tuple(universe), file_tasks=tuple(tasks))
