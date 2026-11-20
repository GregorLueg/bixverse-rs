"""Single cell ATAC: the implementations behind `bixverse.atac`.

Every function takes either kind of ATAC experiment:

- a `SingleCellsMultiModal` with an ATAC store under `<dir>/atac/`, written
  for every cell of the RNA store in store order, so both stores share
  `cell_idx` and the kept cell set of the RNA side applies unchanged;
- a `SingleCellsAtac`, whose own store holds the peaks or tiles.

Both expose the ATAC store through `atac_dir`, `atac`, `atac_cache`,
`atac_var_table` and `atac_lib_col`; nothing here asks which class it got,
except the `add_atac_*` functions, which add an ATAC store next to RNA.

Fragment files are read with the cell ids as barcodes. Where the two differ
(a `-1` suffix, a sample prefix), pass `barcode_map`: a mapping or function
from cell id to the barcode in the fragment file.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.atac_exp import (
    GENE_ACTIVITY_DIR,
    AtacInfo,
    CountMode,
    SingleCellsAtac,
    read_atac_info,
    sync_gene_activity_filter,
    write_count_mode,
)
from bixverse._core.ingest import _prepare
from bixverse._core.integrate import _labels
from bixverse._core.multimodal import SingleCellsMultiModal
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN, SingleCells
from bixverse._core.state import ScCache
from bixverse._validate import Verbose
from bixverse.params import KnnParams, to_rust

logger = logging.getLogger(__name__)

###########
# Globals #
###########

# Signac and muon
TF_IDF_SCALE: float = 1e4

MITO_CHROMS: tuple[str, ...] = ("chrM", "MT", "chrMT", "M")

# warn when more of the requested cells than this have no fragment
MISSING_CELLS_WARN: float = 0.01
# warn when more of the fragments than this lie on chromosomes without targets
OFF_TARGET_WARN: float = 0.5

TagSource = Literal["insertion", "fragment_start"]
BarcodeMap = Mapping[str, str] | Callable[[str], str] | None
AtacHost = SingleCellsMultiModal | SingleCellsAtac

###########
# Helpers #
###########


@beartype
def atac_paths(x: AtacHost) -> tuple[Path, Path]:
    """The cell-based and feature-major files of the ATAC store."""
    return x.atac_dir / CELLS_BIN, x.atac_dir / GENES_BIN


@beartype
def require_atac(x: AtacHost) -> AtacInfo:
    """The ATAC state.

    Raises:
        ValueError: If no ATAC store was added.
    """
    if x.atac is None:
        raise ValueError(
            "No ATAC data in this object. Add it with add_atac_tenx() or "
            "add_atac_fragments()."
        )
    return x.atac


@beartype
def atac_var(x: AtacHost, cols: list[str] | None = None) -> pl.DataFrame:
    """The ATAC feature table, keyed `feature_idx` / `feature_id`."""
    return x.var(modality="atac", cols=cols)


@beartype
def _add_atac_var(x: AtacHost, cols: pl.DataFrame) -> None:
    """Add or replace columns of the ATAC feature table, one row per feature."""
    with db.connect(x.db_path) as con:
        if x.atac_var_table == "var":
            db.add_data(con, "var", cols)
            return
        var = db.get_table(con, "var_atac").drop("feature_idx", strict=False)
        var = var.drop([c for c in cols.columns if c in var.columns]).hstack(cols)
        db.populate_from_frame(con, "var_atac", var)


@beartype
def _region_ids(chrom: Sequence[str], start: np.ndarray, end: np.ndarray) -> list[str]:
    """`chrom:start-end` ids, 0-based half-open."""
    return [f"{c}:{s}-{e}" for c, s, e in zip(chrom, start, end, strict=True)]


@beartype
def _fragment_barcodes(
    ids: Sequence[str] | np.ndarray, barcode_map: BarcodeMap
) -> list[str]:
    """The fragment-file barcode of every cell id.

    Raises:
        KeyError: If a mapping lacks a cell id.
    """
    ids = [str(i) for i in ids]
    if barcode_map is None:
        return ids
    if callable(barcode_map):
        return [str(barcode_map(i)) for i in ids]
    return [str(barcode_map[i]) for i in ids]


@beartype
def _check_fragments(fragments: Path, verbose: int) -> None:
    """Note what the fragment file lacks for a fast read.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not fragments.is_file():
        raise FileNotFoundError(f"{fragments} does not exist")
    with fragments.open("rb") as fh:
        head = fh.read(16)
    tbi = Path(f"{fragments}.tbi")
    bgzf = head[:2] == b"\x1f\x8b" and head[12:14] == b"BC"
    if bgzf and tbi.is_file():
        if tbi.stat().st_mtime < fragments.stat().st_mtime:
            warnings.warn(
                f"{tbi} is older than the fragment file. If the file was rewritten, "
                "regenerate the index (tabix -p bed); a stale one is refused or reads "
                "the wrong chromosomes.",
                stacklevel=3,
            )
        return
    if verbose >= 1:
        what = (
            "no tabix index next to it"
            if bgzf
            else "plain gzip, which inflates on one thread"
            if head[:2] == b"\x1f\x8b"
            else "uncompressed"
        )
        logger.info(
            "%s: %s, so passes read the whole file at once.", fragments.name, what
        )


@beartype
def _report_coverage(res: Mapping, n_cells: int, what: str) -> None:
    """Warn about cells without fragments and fragments off the targets."""
    missing = int(res["cells_missing"])
    if n_cells and missing / n_cells > MISSING_CELLS_WARN:
        warnings.warn(
            f"{missing} of {n_cells} cells have no fragment in the file. If the cell "
            "ids and the fragment barcodes differ (a '-1' suffix, a sample prefix), "
            "pass barcode_map.",
            stacklevel=3,
        )
    off = float(res["off_target_fraction"])
    if off > OFF_TARGET_WARN:
        chroms = ", ".join(res["off_target_chroms"])
        warnings.warn(
            f"{off:.0%} of the fragments lie on chromosomes without {what} ({chroms}). "
            "Check the chromosome naming (chr1 against 1) and the genome build.",
            stacklevel=3,
        )


@beartype
def _frip(x: AtacHost, n_fragments: np.ndarray) -> np.ndarray | None:
    """Fraction of fragments in peaks from the store's library sizes.

    In the store's count unit: with insertion counts the share of insertions
    in peaks (two per fragment), otherwise Signac's fragments in peaks over
    fragments, which can exceed 1 when a fragment overlaps two peaks. `None`
    for tiles, which cover everything.
    """
    info = x.atac
    if info is None or info.kind != "peak":
        return None
    lib = x.obs(cols=[x.atac_lib_col])[x.atac_lib_col].to_numpy().astype(np.float64)
    per_fragment = 2.0 if info.count_mode == "insertion" else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return lib / (per_fragment * n_fragments.astype(np.float64))


@beartype
def _write_atac_tables(
    mm: SingleCellsMultiModal,
    feature_ids: list[str],
    chrom: list[str],
    start: np.ndarray,
    end: np.ndarray,
    lib_size: np.ndarray,
    nnz: np.ndarray,
    count_mode: CountMode,
) -> None:
    """Write `var_atac`, the per-cell totals to obs, and reset the ATAC state.

    A FRiP from an earlier fragment QC is recomputed against the new counts.
    """
    var = pl.DataFrame(
        {
            "feature_id": pl.Series(feature_ids, dtype=pl.String),
            "chrom": pl.Series(chrom, dtype=pl.String),
            "start": pl.Series(np.asarray(start, dtype=np.int64)),
            "end": pl.Series(np.asarray(end, dtype=np.int64)),
        }
    )
    with db.connect(mm.db_path) as con:
        db.populate_from_frame(con, "var_atac", var)
    mm.add_obs(
        pl.DataFrame(
            {
                "cell_idx": mm._all_cells(),
                "atac_lib_size": np.asarray(lib_size, dtype=np.int64),
                "atac_nnz": np.asarray(nnz, dtype=np.int64),
            }
        )
    )
    write_count_mode(mm.atac_dir, count_mode)
    mm.atac = read_atac_info(mm.atac_dir)
    mm.atac_cache = ScCache()
    _refresh_frip(mm)


@beartype
def _refresh_frip(x: AtacHost) -> None:
    """Recompute `atac_frip` when a fragment QC left `atac_n_fragments`."""
    if "atac_n_fragments" not in x.obs_columns():
        return
    n = x.obs(cols=["atac_n_fragments"])["atac_n_fragments"].to_numpy()
    frip = _frip(x, n)
    if frip is None:
        x.drop_obs_cols([c for c in ["atac_frip"] if c in x.obs_columns()])
        return
    x.add_obs(pl.DataFrame({"cell_idx": x._all_cells(), "atac_frip": frip}))


@beartype
def _check_overwrite(x: SingleCellsMultiModal, overwrite: bool) -> None:
    """Refuse to replace an ATAC store unless asked.

    Raises:
        FileExistsError: If one exists and `overwrite` is false.
    """
    if x.atac is not None and not overwrite:
        raise FileExistsError(
            "This experiment has an ATAC store. Pass overwrite=True to replace it."
        )


############
# Ingest   #
############


@beartype
def add_atac_tenx(
    mm: SingleCellsMultiModal,
    h5_path: Path | str,
    *,
    overwrite: bool = False,
    max_mem_gb: float | None = None,
    verbose: Verbose = 1,
) -> None:
    """Add the `Peaks` of a 10x multiome h5 as the ATAC store.

    Args:
        mm: The experiment, its RNA loaded from the same h5 (or with the same
            barcodes).
        h5_path: The Cell Ranger ARC `filtered_feature_bc_matrix.h5`.
        overwrite: Replace an existing ATAC store.
        max_mem_gb: Memory budget of the peak-major conversion.
        verbose: `0` silent, `1` normal, `2` detailed.

    Raises:
        FileExistsError: If an ATAC store exists and `overwrite` is false.
        ValueError: If a cell of the store is not in the h5.
    """
    _check_overwrite(mm, overwrite)
    cells, genes = mm.atac_dir / CELLS_BIN, mm.atac_dir / GENES_BIN
    cells.parent.mkdir(exist_ok=True)
    res = _bixverse.atac_tenx_peaks_to_store(
        Path(h5_path),
        cells,
        genes,
        [str(b) for b in mm.cell_ids()],
        max_mem_gb,
        verbose,
    )
    _write_atac_tables(
        mm,
        list(res["name"]),
        list(res["chrom"]),
        res["start"],
        res["end"],
        res["lib_size"],
        res["nnz"],
        "insertion",
    )


@beartype
def _features(
    peaks: Path | str | pl.DataFrame | None,
    tile_width: int | None,
    chrom_sizes: Mapping[str, int] | None,
) -> tuple[list[str], np.ndarray, np.ndarray, Literal["peak", "tile"]]:
    """The features to count into: a BED, a frame of regions, or tiles.

    Raises:
        ValueError: Unless exactly one of `peaks` and `tile_width` is given, or
            tiles come without `chrom_sizes`.
    """
    exactly_one = "Pass exactly one of peaks and tile_width."
    if peaks is not None and tile_width is not None:
        raise ValueError(exactly_one)
    if peaks is None:
        if tile_width is None:
            raise ValueError(exactly_one)
        if not chrom_sizes:
            raise ValueError("Tiles need chrom_sizes.")
        r = _bixverse.atac_genome_tiles(
            list(chrom_sizes), [int(v) for v in chrom_sizes.values()], tile_width
        )
        return list(r["chrom"]), r["start"], r["end"], "tile"
    if isinstance(peaks, pl.DataFrame):
        missing = {"chrom", "start", "end"} - set(peaks.columns)
        if missing:
            raise ValueError(f"peaks frame lacks {sorted(missing)}")
        return (
            peaks["chrom"].cast(pl.String).to_list(),
            peaks["start"].to_numpy().astype(np.uint32),
            peaks["end"].to_numpy().astype(np.uint32),
            "peak",
        )
    r = _bixverse.atac_read_bed(Path(peaks))
    return list(r["chrom"]), r["start"], r["end"], "peak"


@beartype
def _count_store(
    fragments: Path,
    store_dir: Path,
    barcodes: list[str],
    chrom: list[str],
    start: np.ndarray,
    end: np.ndarray,
    mode: CountMode,
    kind: str,
    target_size: float,
    max_mem_gb: float | None,
    verbose: int,
) -> dict:
    """Count a fragment file into a store and report on the pass."""
    _check_fragments(fragments, verbose)
    store_dir.mkdir(parents=True, exist_ok=True)
    res = _bixverse.atac_fragments_to_store(
        fragments,
        store_dir / CELLS_BIN,
        store_dir / GENES_BIN,
        barcodes,
        chrom,
        [int(x) for x in start],
        [int(x) for x in end],
        mode,
        kind,
        target_size,
        max_mem_gb,
        verbose,
    )
    _report_coverage(res, len(barcodes), "features")
    return res


@beartype
def add_atac_fragments(
    mm: SingleCellsMultiModal,
    fragments: Path | str,
    *,
    peaks: Path | str | pl.DataFrame | None = None,
    tile_width: int | None = None,
    chrom_sizes: Mapping[str, int] | None = None,
    mode: CountMode = "insertion",
    barcode_map: BarcodeMap = None,
    overwrite: bool = False,
    max_mem_gb: float | None = None,
    verbose: Verbose = 1,
) -> None:
    """Count a fragment file into peaks or genome tiles as the ATAC store.

    Args:
        mm: The experiment; every cell of its store is counted.
        fragments: The fragment file (BGZF, gzip or plain).
        peaks: A BED file or a frame with `chrom`, `start`, `end` (0-based,
            half-open).
        tile_width: Tile width in bp, instead of `peaks`.
        chrom_sizes: Chromosome lengths, for the tiles.
        mode: `"insertion"` (each Tn5 insertion, Cell Ranger and ArchR),
            `"paired-insertion"` (once per fragment and feature, SnapATAC2) or
            `"fragment"` (each overlapping fragment, Signac).
        barcode_map: Cell id to fragment barcode, where they differ.
        overwrite: Replace an existing ATAC store.
        max_mem_gb: Memory budget of the peak-major conversion.
        verbose: `0` silent, `1` normal, `2` detailed.

    Raises:
        FileExistsError: If an ATAC store exists and `overwrite` is false.
        ValueError: Unless exactly one of `peaks` and `tile_width` is given,
            if tiles come without `chrom_sizes`, if a `peaks` frame lacks
            `chrom`, `start` or `end`, if no cell id is a barcode of the file,
            or if no fragment lies on a chromosome of the features.
    """
    chrom, start, end, kind = _features(peaks, tile_width, chrom_sizes)
    _check_overwrite(mm, overwrite)
    res = _count_store(
        Path(fragments),
        mm.atac_dir,
        _fragment_barcodes(mm.cell_ids(), barcode_map),
        chrom,
        np.asarray(start),
        np.asarray(end),
        mode,
        kind,
        TF_IDF_SCALE,
        max_mem_gb,
        verbose,
    )
    _write_atac_tables(
        mm,
        _region_ids(chrom, start, end),
        chrom,
        np.asarray(start),
        np.asarray(end),
        res["lib_size"],
        res["nnz"],
        mode,
    )


@beartype
def read_cellranger_barcodes(path: Path | str) -> list[str]:
    """The cell barcodes Cell Ranger ATAC called.

    Args:
        path: `singlecell.csv` (rows with `is__cell_barcode == 1`) or a
            filtered `*_bc_matrix.h5` (its `matrix/barcodes`).

    Returns:
        The barcodes, in file order.

    Raises:
        ValueError: For another file type, or a CSV without
            `is__cell_barcode`.
    """
    path = Path(path).expanduser()
    if path.suffix == ".h5":
        return [str(b) for b in _bixverse.h5_strings(path, "matrix/barcodes")]
    if path.suffix != ".csv":
        raise ValueError(f"Expected singlecell.csv or a filtered h5, got {path.name}.")
    frame = pl.read_csv(path, infer_schema_length=0)
    if "is__cell_barcode" not in frame.columns:
        raise ValueError(f"{path.name} has no is__cell_barcode column.")
    calls = frame.filter(pl.col("is__cell_barcode") == "1")
    return [b for b in calls["barcode"].to_list() if b != "NO_BARCODE"]


@beartype
def barcode_fragment_counts(
    fragments: Path | str, *, exclude_chroms: Sequence[str] = MITO_CHROMS
) -> pl.DataFrame:
    """Fragments per barcode, every barcode in the file, for cell calling.

    Args:
        fragments: The fragment file.
        exclude_chroms: Chromosomes not counted.

    Returns:
        `barcode` and `n_fragments`, most fragments first (for a knee plot).
    """
    r = _bixverse.atac_barcode_fragment_counts(Path(fragments), list(exclude_chroms))
    return pl.DataFrame(
        {
            "barcode": pl.Series(r["barcode"], dtype=pl.String),
            "n_fragments": r["n_fragments"],
        }
    ).sort(["n_fragments", "barcode"], descending=[True, False])


@beartype
def _resolve_barcodes(
    fragments: Path,
    barcodes: Sequence[str] | np.ndarray | None,
    cellranger: Path | str | None,
    min_fragments: int | None,
    exclude_chroms: Sequence[str],
) -> list[str]:
    """The cells of an ATAC-only experiment, from exactly one source.

    Raises:
        ValueError: Unless exactly one source is given, or if it yields no cell.
    """
    given = [barcodes is not None, cellranger is not None, min_fragments is not None]
    if sum(given) != 1:
        raise ValueError("Pass exactly one of barcodes, cellranger and min_fragments.")
    if barcodes is not None:
        out = [str(b) for b in barcodes]
    elif cellranger is not None:
        out = read_cellranger_barcodes(cellranger)
    else:
        counts = barcode_fragment_counts(fragments, exclude_chroms=exclude_chroms)
        out = sorted(counts.filter(pl.col("n_fragments") >= min_fragments)["barcode"])
    if not out:
        raise ValueError("No cell barcode to load.")
    return out


@beartype
def _finish_atac_root(
    directory: Path,
    barcodes: list[str],
    feature_ids: list[str],
    chrom: list[str],
    start: np.ndarray,
    end: np.ndarray,
    lib_size: np.ndarray,
    nnz: np.ndarray,
    count_mode: CountMode,
) -> SingleCellsAtac:
    """obs, var and the count mode of a freshly written ATAC-only store."""
    gene_nnz = _bixverse.gene_nnz(directory / GENES_BIN)
    with db.connect(db.db_path(directory)) as con:
        db.populate_from_frame(con, "obs", pl.DataFrame({"cell_id": barcodes}))
        db.populate_from_frame(
            con,
            "var",
            pl.DataFrame(
                {
                    "gene_id": pl.Series(feature_ids, dtype=pl.String),
                    "chrom": pl.Series(chrom, dtype=pl.String),
                    "start": pl.Series(np.asarray(start, dtype=np.int64)),
                    "end": pl.Series(np.asarray(end, dtype=np.int64)),
                    "no_cells_exp": pl.Series(gene_nnz.astype(np.int32)),
                }
            ),
        )
        db.add_data(
            con,
            "obs",
            pl.DataFrame(
                {
                    "nnz": np.asarray(nnz, dtype=np.float64),
                    "lib_size": np.asarray(lib_size, dtype=np.float64),
                }
            ),
        )
        db.set_to_keep_column(con)
    write_count_mode(directory, count_mode)
    x = SingleCellsAtac(directory)
    empty = int((np.asarray(lib_size) == 0).sum())
    if empty:
        logger.info(
            "%d cells have no count in any feature; they sit at the origin of the "
            "LSI. Filter them (lib_size > 0) unless they are meant to be there.",
            empty,
        )
    return x


@beartype
def load_atac_fragments(
    directory: Path | str,
    fragments: Path | str,
    *,
    peaks: Path | str | pl.DataFrame | None = None,
    tile_width: int | None = None,
    chrom_sizes: Mapping[str, int] | None = None,
    mode: CountMode = "insertion",
    barcodes: Sequence[str] | np.ndarray | None = None,
    cellranger: Path | str | None = None,
    min_fragments: int | None = None,
    exclude_chroms: Sequence[str] = MITO_CHROMS,
    max_mem_gb: float | None = None,
    verbose: Verbose = 1,
) -> SingleCellsAtac:
    """Load an ATAC-only experiment from a fragment file.

    The cells come from exactly one of `barcodes`, `cellranger` or
    `min_fragments`; with the last, every barcode with at least that many
    fragments (outside `exclude_chroms`), sorted, as Cell Ranger orders
    them. See `barcode_fragment_counts` for the knee plot to pick it from.

    Args:
        directory: The experiment directory, created if missing.
        fragments: The fragment file (BGZF, gzip or plain).
        peaks: A BED file or a frame with `chrom`, `start`, `end`.
        tile_width: Tile width in bp, instead of `peaks`.
        chrom_sizes: Chromosome lengths, for the tiles.
        mode: What one count is, see `add_atac_fragments`.
        barcodes: The cell barcodes.
        cellranger: Cell Ranger ATAC's `singlecell.csv` or filtered h5.
        min_fragments: Call cells with at least this many fragments.
        exclude_chroms: Chromosomes the `min_fragments` count skips.
        max_mem_gb: Memory budget of the feature-major conversion.
        verbose: `0` silent, `1` normal, `2` detailed.

    Returns:
        The experiment: obs holds `cell_id`, `lib_size`, `nnz`, `to_keep`;
        var the features as `gene_id` (`chr:start-end`) with `chrom`,
        `start`, `end`, `no_cells_exp`.

    Raises:
        ValueError: Unless exactly one cell source and exactly one of `peaks`
            and `tile_width` are given, or if no fragment of the cells lies
            on a feature's chromosome.
    """
    fragments = Path(fragments).expanduser()
    cells = _resolve_barcodes(
        fragments, barcodes, cellranger, min_fragments, exclude_chroms
    )
    chrom, start, end, kind = _features(peaks, tile_width, chrom_sizes)
    path = _prepare(directory)
    res = _count_store(
        fragments,
        path,
        cells,
        chrom,
        np.asarray(start),
        np.asarray(end),
        mode,
        kind,
        TF_IDF_SCALE,
        max_mem_gb,
        verbose,
    )
    return _finish_atac_root(
        path,
        cells,
        _region_ids(chrom, start, end),
        chrom,
        np.asarray(start),
        np.asarray(end),
        res["lib_size"],
        res["nnz"],
        mode,
    )


@beartype
def load_atac_tenx(
    directory: Path | str,
    h5_path: Path | str,
    *,
    barcodes: Sequence[str] | np.ndarray | None = None,
    max_mem_gb: float | None = None,
    verbose: Verbose = 1,
) -> SingleCellsAtac:
    """Load an ATAC-only experiment from a 10x peak matrix h5.

    Args:
        directory: The experiment directory, created if missing.
        h5_path: Cell Ranger ATAC's `filtered_peak_bc_matrix.h5`, or the
            Cell Ranger ARC h5 (its `Peaks` rows).
        barcodes: Cells to load; `None` for every barcode of the h5.
        max_mem_gb: Memory budget of the feature-major conversion.
        verbose: `0` silent, `1` normal, `2` detailed.

    Returns:
        The experiment, as `load_atac_fragments` describes it; the counts
        are insertions.

    Raises:
        ValueError: If a requested barcode is not in the h5.
    """
    h5 = Path(h5_path).expanduser()
    cells = (
        [str(b) for b in _bixverse.h5_strings(h5, "matrix/barcodes")]
        if barcodes is None
        else [str(b) for b in barcodes]
    )
    path = _prepare(directory)
    res = _bixverse.atac_tenx_peaks_to_store(
        h5, path / CELLS_BIN, path / GENES_BIN, cells, max_mem_gb, verbose
    )
    return _finish_atac_root(
        path,
        cells,
        list(res["name"]),
        list(res["chrom"]),
        np.asarray(res["start"]),
        np.asarray(res["end"]),
        np.asarray(res["lib_size"]),
        np.asarray(res["nnz"]),
        "insertion",
    )


@beartype
def merge_fragments(
    fragments: Sequence[Path | str],
    out: Path | str,
    *,
    prefixes: Sequence[str] | None = None,
    suffixes: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Merge per-sample fragment files into one, as `cellranger-arc aggr` does.

    Each file's barcodes get its prefix and suffix, so the same barcode in
    two samples stays two cells; give every file its own. The output is a
    sorted BGZF file with a tabix index, which every function here reads
    as one experiment. Files with a tabix index are read one chromosome at a
    time; others are held in memory whole.

    Args:
        fragments: The fragment files.
        out: The merged file; the index goes next to it as `.tbi`.
        prefixes: Barcode prefix per file, e.g. `["s1_", "s2_"]`.
        suffixes: Barcode suffix per file, e.g. `["-1", "-2"]` after
            stripping 10x's own suffix with a prefix of `""`.

    Returns:
        `fragments`, `prefix`, `suffix` and `n_fragments` per input.

    Raises:
        ValueError: If neither `prefixes` nor `suffixes` is given, or their
            lengths do not match the files.
    """
    n = len(fragments)
    if prefixes is None and suffixes is None:
        raise ValueError(
            "Pass prefixes or suffixes, so barcodes of different files stay apart."
        )
    pre = list(prefixes) if prefixes is not None else [""] * n
    suf = list(suffixes) if suffixes is not None else [""] * n
    if len(pre) != n or len(suf) != n:
        raise ValueError("prefixes and suffixes need one entry per fragment file.")
    r = _bixverse.atac_merge_fragments(
        [Path(f).expanduser() for f in fragments], pre, suf, Path(out).expanduser()
    )
    return pl.DataFrame(
        {
            "fragments": [str(f) for f in fragments],
            "prefix": pre,
            "suffix": suf,
            "n_fragments": r["n_fragments"],
        }
    )


#################
# Gene activity #
#################


@beartype
def gene_activity(
    sc: SingleCells,
    fragments: Path | str,
    gtf: Path | str,
    *,
    dir_out: Path | str | None = None,
    upstream: int = 2000,
    downstream: int = 0,
    max_width: int = 500_000,
    biotypes: Sequence[str] = ("protein_coding",),
    target_size: float | None = None,
    barcode_map: BarcodeMap = None,
    overwrite: bool = False,
    max_mem_gb: float | None = None,
    verbose: Verbose = 1,
) -> SingleCells:
    """Count fragments over gene body plus promoter into a gene activity store.

    Signac's `GeneActivity`: per gene, the longest span of its biotype, under
    `max_width`, extended `upstream` of the TSS; each overlapping fragment
    counts once. The result is a `SingleCells` over every cell of `sc`, in its
    order and with its obs and cell filter, so it serves as the query of
    `bv.annotate.transfer_labels_seurat` and of the RNA tools. Methods that
    assume RNA counts refuse it or warn. Written to the default directory,
    later cell filters on `sc` reach it; reopen it with `get_gene_activity`.

    Args:
        sc: The experiment whose cell ids are the fragment barcodes.
        fragments: The fragment file (BGZF, gzip or plain).
        gtf: Gene annotation (GENCODE, Ensembl, or a transcript-only GTF).
        dir_out: Output directory. `None` uses `gene_activity` inside the
            experiment directory.
        upstream: Bases added upstream of the TSS.
        downstream: Bases added downstream of the gene end.
        max_width: Genes this long or longer are dropped.
        biotypes: Biotypes to keep, empty for all.
        target_size: Target of the log-CPM norm. `None` uses the median RNA
            `lib_size` of the kept cells, as the Signac vignettes do, or 1e4
            on an ATAC-only experiment, whose `lib_size` counts peaks.
        barcode_map: Cell id to fragment barcode, where they differ.
        overwrite: Replace an existing store at `dir_out`.
        max_mem_gb: Memory budget of the gene-major conversion.
        verbose: `0` silent, `1` normal, `2` detailed.

    Returns:
        The gene activity experiment. var holds `gene_id` (the GTF gene ID
        without its version, as Cell Ranger keys the RNA), `gene_name` (made
        unique), `chrom`, `start`, `end`; obs's `lib_size` and `nnz` are
        the gene activity's.

    Raises:
        FileExistsError: If a store exists at `dir_out` and `overwrite` is
            false.
        ValueError: If `target_size` is `None` and obs has no `lib_size`, two
            gene IDs collide once their versions are stripped, no gene has a
            requested biotype, or the fragments and the GTF share no
            chromosome.
    """
    out = sc.dir / GENE_ACTIVITY_DIR if dir_out is None else Path(dir_out).expanduser()
    genes = out / GENES_BIN
    if genes.exists() and not overwrite:
        raise FileExistsError(
            f"A store already exists at '{out}'. Pass overwrite=True to replace it."
        )
    obs = sc.obs()
    if target_size is None:
        if sc.root_modality == "atac":
            target_size = TF_IDF_SCALE
        else:
            if "lib_size" not in obs.columns:
                raise ValueError("obs has no lib_size; pass target_size.")
            kept = obs.filter(pl.col("cell_idx").is_in(sc.cells_to_keep))
            target_size = float(np.median(kept["lib_size"].to_numpy()))

    r = _bixverse.atac_gene_activity_regions(
        Path(gtf), upstream, downstream, max_width, list(biotypes)
    )
    res = _count_store(
        Path(fragments),
        out,
        _fragment_barcodes(sc.cell_ids(), barcode_map),
        list(r["chrom"]),
        np.asarray(r["start"]),
        np.asarray(r["end"]),
        "fragment",
        "gene_activity",
        target_size,
        max_mem_gb,
        verbose,
    )
    # Unversioned, as Cell Ranger keys the RNA, so RNA and gene activity share
    # gene_id. `_PAR_Y` survives and keeps the chrY copies apart.
    gene_id = pl.Series(list(r["gene_id"]), dtype=pl.String).str.replace(r"\.\d+", "")
    if gene_id.is_duplicated().any():
        raise ValueError("Gene IDs collide once their versions are stripped.")
    var = pl.DataFrame(
        {
            "gene_id": gene_id,
            "gene_name": pl.Series(list(r["name"]), dtype=pl.String),
            "chrom": pl.Series(list(r["chrom"]), dtype=pl.String),
            "start": pl.Series(np.asarray(r["start"], dtype=np.int64)),
            "end": pl.Series(np.asarray(r["end"], dtype=np.int64)),
            "no_cells_exp": pl.Series(_bixverse.gene_nnz(genes).astype(np.int32)),
        }
    )
    obs = obs.sort("cell_idx").drop("cell_idx")
    obs = obs.with_columns(
        lib_size=pl.Series(np.asarray(res["lib_size"], dtype=np.float64)),
        nnz=pl.Series(np.asarray(res["nnz"], dtype=np.float64)),
    )
    with db.connect(db.db_path(out)) as con:
        db.populate_from_frame(con, "obs", obs)
        db.populate_from_frame(con, "var", var)
    return SingleCells(out)


@beartype
def get_gene_activity(
    x: SingleCells, directory: Path | str | None = None
) -> SingleCells:
    """Open an experiment's gene activity store, its cell filter brought in line.

    Args:
        x: The experiment the gene activity was counted for.
        directory: Where it was written; `None` for the default
            `<dir>/gene_activity`.

    Returns:
        The gene activity experiment, with `x`'s kept cells.

    Raises:
        FileNotFoundError: If there is no store.
        ValueError: If it was written for other cells (a re-ingest since).
    """
    out = (
        x.dir / GENE_ACTIVITY_DIR if directory is None else Path(directory).expanduser()
    )
    if not (out / CELLS_BIN).is_file():
        raise FileNotFoundError(
            f"No gene activity store at {out}. Run gene_activity()."
        )
    theirs = _bixverse.cell_id_hash(out / CELLS_BIN)
    ours = _bixverse.cell_id_hash(x.cells_path)
    if theirs is not None and ours is not None and theirs != ours:
        raise ValueError(
            f"The gene activity at {out} was counted for other cells. Run "
            "gene_activity(..., overwrite=True) again."
        )
    if directory is None:
        sync_gene_activity_filter(x)
        return SingleCells(out)
    ga = SingleCells(out)
    ga.set_cells_to_keep(x.cells_to_keep)
    return ga


########
# QC   #
########


@beartype
def _tss(gtf: Path | str, biotypes: Sequence[str]) -> dict:
    """TSSs of the GTF genes of the biotypes, with strand, gene id and name."""
    return _bixverse.atac_tss_positions(Path(gtf), list(biotypes))


@beartype
def fragment_qc(
    x: AtacHost,
    fragments: Path | str,
    gtf: Path | str,
    *,
    biotypes: Sequence[str] = ("protein_coding",),
    mito_chroms: Sequence[str] = MITO_CHROMS,
    barcode_map: BarcodeMap = None,
) -> pl.DataFrame:
    """Per-cell QC from the fragment file, written to obs.

    Adds `atac_n_fragments`, `atac_nucleosome_signal`, `atac_tss_enrichment`
    (Signac's fast definition) and `atac_mito_fraction` for every cell, plus
    `atac_frip` on a peak store: its library size over the fragments, in the
    store's count unit, so with insertion counts the share of insertions in
    peaks (two per fragment) and otherwise Signac's fragments in peaks over
    fragments.

    The nucleosome signal is mononucleosomal over nucleosome-free fragments,
    as in Signac: infinite for a cell with mononucleosomal but no
    nucleosome-free fragments (a bad cell, which any upper threshold drops),
    and 0 rather than Signac's `NaN` for a cell with neither, so robust
    statistics over the column stay defined.

    Args:
        x: The experiment.
        fragments: The fragment file.
        gtf: Gene annotation for the TSSs.
        biotypes: Gene biotypes whose TSSs count (Signac: protein coding).
        mito_chroms: Names of the mitochondrial genome.
        barcode_map: Cell id to fragment barcode, where they differ.

    Returns:
        The obs columns added, with `cell_idx`.

    Raises:
        ValueError: If no cell id is a barcode of the file, or the fragments
            and the GTF share no chromosome.
    """
    fragments = Path(fragments)
    _check_fragments(fragments, 1)
    tss = _tss(gtf, biotypes)
    barcodes = _fragment_barcodes(x.cell_ids(), barcode_map)
    qc = _bixverse.atac_fragment_qc(
        fragments,
        barcodes,
        list(tss["chrom"]),
        [int(v) for v in tss["pos"]],
        list(mito_chroms),
    )
    _report_coverage(qc, len(barcodes), "TSSs")
    ns = np.asarray(qc["nucleosome_signal"], dtype=np.float64)
    frame = pl.DataFrame(
        {
            "cell_idx": x._all_cells(),
            "atac_n_fragments": qc["n_fragments"].astype(np.int64),
            "atac_nucleosome_signal": np.where(np.isnan(ns), 0.0, ns),
            "atac_tss_enrichment": qc["tss_enrichment"],
            "atac_mito_fraction": qc["mito_fraction"],
        }
    )
    frip = _frip(x, qc["n_fragments"])
    if frip is not None:
        frame = frame.with_columns(pl.Series("atac_frip", frip))
    x.add_obs(frame)
    return frame


@beartype
def fragment_profiles(
    x: AtacHost,
    fragments: Path | str,
    gtf: Path | str,
    *,
    group_by: str | None = None,
    half_width: int = 2000,
    max_len: int = 1000,
    flank: int = 100,
    biotypes: Sequence[str] = ("protein_coding",),
    barcode_map: BarcodeMap = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """TSS insertion profile and fragment length histogram of the kept cells.

    One pass over the fragment file, per level of `group_by` (SnapATAC2's
    `TSS_profile` and `frag_size_distr`, which these reproduce): insertions
    are a fragment's start and `end - 1`, offsets are strand aware, positive
    downstream.

    Args:
        x: The experiment.
        fragments: The fragment file.
        gtf: Gene annotation for the TSSs.
        group_by: obs column to split by; `None` for one group, `"all"`.
        half_width: Profile half width in bp.
        max_len: Longest fragment length in the histogram.
        flank: Bases at each end of the profile whose mean is the background
            of the enrichment.
        biotypes: Gene biotypes whose TSSs count.
        barcode_map: Cell id to fragment barcode, where they differ.

    Returns:
        `(tss, lengths)`: `group`, `offset`, `insertions` and `enrichment`
        (insertions over the mean of both flanks); and `group`, `length`,
        `fragments`, `fraction` (of the group's fragments up to `max_len`).
        For `bv.pl.tss_profile_plot` and `bv.pl.fragment_length_plot`.
    """
    kept = x.cells_to_keep
    if group_by is None:
        codes = np.zeros(len(kept), dtype=np.uint32)
        levels = ["all"]
    else:
        codes, levels = _labels(x, group_by)
    tss = _tss(gtf, biotypes)
    barcodes = _fragment_barcodes(x.cell_ids(kept), barcode_map)
    r = _bixverse.atac_fragment_profiles(
        Path(fragments),
        barcodes,
        [int(c) for c in codes],
        list(tss["chrom"]),
        [int(v) for v in tss["pos"]],
        [int(v) for v in tss["strand"]],
        half_width,
        max_len,
    )
    _report_coverage(r, len(barcodes), "TSSs")
    prof = np.asarray(r["tss"], dtype=np.float64)
    width = 2 * half_width + 1
    background = np.concatenate([prof[:, :flank], prof[:, -flank:]], axis=1).mean(
        axis=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        enrich = prof / background[:, None]
    group = pl.Enum(levels)
    tss_frame = pl.DataFrame(
        {
            "group": pl.Series(np.repeat(levels, width), dtype=group),
            "offset": np.tile(np.arange(-half_width, half_width + 1), len(levels)),
            "insertions": prof.ravel().astype(np.int64),
            "enrichment": enrich.ravel(),
        }
    )
    lengths = np.asarray(r["lengths"], dtype=np.float64)[:, 1:]
    total = lengths.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = lengths / total
    len_frame = pl.DataFrame(
        {
            "group": pl.Series(np.repeat(levels, max_len), dtype=group),
            "length": np.tile(np.arange(1, max_len + 1), len(levels)),
            "fragments": lengths.ravel().astype(np.int64),
            "fraction": frac.ravel(),
        }
    )
    return tss_frame, len_frame


################
# Peak calling #
################


@beartype
def call_peaks(
    sc: SingleCells,
    fragments: Path | str,
    *,
    group_by: str | None = None,
    combine: bool = True,
    tag_source: TagSource = "insertion",
    keep_dup: int | None = None,
    shift: int = -100,
    extsize: int = 200,
    genome_size: float = 2.7e9,
    llocal: int = 10_000,
    q_value: float = 0.05,
    min_length: int | None = None,
    max_gap: int | None = None,
    barcode_map: BarcodeMap = None,
    max_mem_gb: float | None = None,
    tmp_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Call peaks on the kept cells, pooled or per group (MACS3 `callpeak`).

    A native port of MACS3 `callpeak --nomodel` without a control track, one
    call per group, as Signac's `CallPeaks(group.by = ...)` runs MACS. Against
    MACS3 3.1.0 the peak coordinates, summits and pileups are exact and the
    scores agree to MACS's printed precision. Pass the result as `peaks` to
    `add_atac_fragments` or `load_atac_fragments` to count against it.

    Memory: with a tabix index next to the fragment file (`.tbi`, which Cell
    Ranger writes), the file is read one chromosome at a time and the sorted
    tags wait in a scratch file (4 bytes per tag) between passes. Without an
    index every tag is held at once while reading.

    Args:
        sc: The experiment; its kept cells are pooled.
        fragments: The fragment file.
        group_by: obs column to call per level of (clusters, samples); `None`
            pools all kept cells into one group, `"all"`.
        combine: Merge overlapping peaks across groups into one set, with the
            groups each was called in. `False` returns every group's peaks.
        tag_source: `"insertion"` (both Tn5 insertions per fragment) or
            `"fragment_start"` (the fragment start only: what MACS reads from
            the fragment file Signac hands it).
        keep_dup: Tags kept per position; `None` keeps all (MACS's default,
            and so Signac's, is 1).
        shift: Tag shift before extension (`--shift`).
        extsize: Tag extension, the fragment size (`--extsize`).
        genome_size: Effective genome size (`-g`); the default is Signac's,
            MACS3's own `hs` is 2.913e9.
        llocal: Window of the local lambda (`--llocal`).
        q_value: q-value cutoff (`-q`), in `(0, 1)`.
        min_length: Minimum peak length; `None` for `extsize`.
        max_gap: Largest gap merged into a peak; `None` for MACS's tag size (1
            for insertions, the mean length of the group's first fragments
            for fragment starts).
        barcode_map: Cell id to fragment barcode, where they differ.
        max_mem_gb: Tag budget of the reading pass; `None` reads one
            chromosome at a time.
        tmp_dir: Directory of the scratch file; `None` for the system
            temporary directory.

    Returns:
        With `combine`: `chrom`, `start`, `end` (0-based, half-open) and
        `peak_called_in`, the comma-separated groups. Without: one row per
        group and peak, with `group`, `summit` and the summit's `pileup`,
        `pscore` (`-log10 p`), `fold` and `qscore` (`-log10 q`). A group
        none of whose cells has a fragment has no peaks.

    Raises:
        ValueError: If `group_by` is not an obs column or has missing values,
            or `q_value` is not in `(0, 1)`.
    """
    fragments = Path(fragments)
    _check_fragments(fragments, 1)
    kept = sc.cells_to_keep
    if group_by is None:
        codes = np.zeros(len(kept), dtype=np.uint32)
        levels = ["all"]
    else:
        codes, levels = _labels(sc, group_by)
    r = _bixverse.atac_call_peaks(
        fragments,
        _fragment_barcodes(sc.cell_ids(kept), barcode_map),
        [int(c) for c in codes],
        tag_source,
        keep_dup,
        shift,
        extsize,
        genome_size,
        llocal,
        q_value,
        min_length,
        max_gap,
        max_mem_gb,
        None if tmp_dir is None else Path(tmp_dir),
    )
    peaks = pl.DataFrame(
        {
            "group": pl.Series([levels[g] for g in r["group"]], dtype=pl.Enum(levels)),
            "chrom": pl.Series(r["chrom"], dtype=pl.String),
            "start": r["start"],
            "end": r["end"],
            "summit": r["summit"],
            "pileup": r["pileup"],
            "pscore": r["pscore"],
            "fold": r["fold"],
            "qscore": r["qscore"],
        }
    )
    if peaks.height == 0:
        warnings.warn(
            "No peak was called. Check the barcodes (barcode_map) and the q_value.",
            stacklevel=2,
        )
    if not combine:
        return peaks
    merged = _bixverse.atac_reduce_regions(
        peaks["chrom"].to_list(),
        [int(x) for x in peaks["start"]],
        [int(x) for x in peaks["end"]],
    )
    group = peaks["group"].cast(pl.String).to_list()
    called_in = []
    for members in merged["members"]:
        present = {group[i] for i in members}
        called_in.append(",".join(lev for lev in levels if lev in present))
    return pl.DataFrame(
        {
            "chrom": pl.Series(merged["chrom"], dtype=pl.String),
            "start": merged["start"],
            "end": merged["end"],
            "peak_called_in": called_in,
        }
    )


@beartype
def _regions(
    regions: Path | str | pl.DataFrame,
) -> tuple[list[str], list[int], list[int]]:
    """`chrom`, `start`, `end` of a BED file or a frame."""
    if isinstance(regions, pl.DataFrame):
        return (
            regions["chrom"].cast(pl.String).to_list(),
            [int(v) for v in regions["start"]],
            [int(v) for v in regions["end"]],
        )
    r = _bixverse.atac_read_bed(Path(regions))
    return list(r["chrom"]), [int(v) for v in r["start"]], [int(v) for v in r["end"]]


@beartype
def remove_blacklisted(
    peaks: pl.DataFrame, blacklist: Path | str | pl.DataFrame
) -> pl.DataFrame:
    """Drop the peaks that overlap a blacklist region.

    Args:
        peaks: Regions with `chrom`, `start`, `end`, e.g. from `call_peaks`.
        blacklist: A BED file (plain or gzip) or a frame of regions, e.g. the
            ENCODE blacklist.

    Returns:
        The peaks that touch no blacklist region, in order.
    """
    c, s, e = _regions(peaks)
    bc, bs, be = _regions(blacklist)
    hit = _bixverse.atac_overlaps_any(c, s, e, bc, bs, be)
    return peaks.filter(~pl.Series(hit))


@beartype
def annotate_peaks(
    x: AtacHost, gtf: Path | str, *, biotypes: Sequence[str] = ("protein_coding",)
) -> pl.DataFrame:
    """The closest gene of every ATAC feature (Signac `ClosestFeature`).

    Written to the ATAC feature table as `closest_gene`, `closest_gene_id`,
    `distance_to_gene` (bases to the gene body, 0 inside or touching it)
    and `distance_to_tss` (feature centre to TSS, positive downstream).

    Args:
        x: The experiment.
        gtf: Gene annotation.
        biotypes: Biotypes to consider, empty for all.

    Returns:
        `feature_id` and the four columns.
    """
    require_atac(x)
    var = atac_var(x, cols=["feature_id", "chrom", "start", "end"])
    r = _bixverse.atac_closest_genes(
        Path(gtf),
        list(biotypes),
        var["chrom"].cast(pl.String).to_list(),
        [int(v) for v in var["start"]],
        [int(v) for v in var["end"]],
    )
    cols = pl.DataFrame(
        {
            "closest_gene": pl.Series(r["gene_name"], dtype=pl.String),
            "closest_gene_id": pl.Series(r["gene_id"], dtype=pl.String),
            "distance_to_gene": pl.Series(np.asarray(r["distance"], dtype=np.int64)),
            "distance_to_tss": pl.Series(np.asarray(r["tss_distance"], dtype=np.int64)),
        }
    )
    _add_atac_var(x, cols)
    return pl.concat([var.select("feature_id"), cols], how="horizontal")


##############
# Multiplets #
##############


@beartype
def amulet(
    sc: SingleCells,
    fragments: Path | str,
    *,
    restrict_beds: Sequence[Path | str] = (),
    max_overlap: int = 2,
    max_fragment_len: int = 900,
    repeat_q: float = 0.01,
    multiplet_q: float = 0.01,
    chroms: Sequence[str] | None = None,
    barcode_map: BarcodeMap = None,
    max_mem_gb: float | None = None,
    count_trailing_loci: bool = False,
) -> pl.DataFrame:
    """Flag multiplets from fragment overlaps (AMULET), written to obs.

    A diploid nucleus yields at most two overlapping fragments at a locus.
    Per cell, the loci with more are counted (autosomes, restriction regions
    dropped), sites shared by too many cells are dropped as repeats, and the
    count is tested against the mean (Poisson, Benjamini-Hochberg).
    Implemented from Thibodeau et al. (Genome Biology, 2021); on the test
    data it reproduces the reference implementation's counts and p-values.
    Needs no clustering and no RNA, so it works for ATAC-only experiments.
    Run it per library: multiplets are a property of one capture.

    Adds `amulet_n_overlaps`, `amulet_n_sites`, `amulet_p`, `amulet_q` and
    `amulet_multiplet` for every cell of the store.

    Args:
        sc: The experiment; every cell of its store is tested.
        fragments: The fragment file.
        restrict_beds: BED files of regions to ignore: repeats, segmental
            duplications, blacklists. The paper uses all three.
        max_overlap: Overlapping fragments a cell can explain; 2 for diploid.
        max_fragment_len: Longest valid fragment.
        repeat_q: q-value under which a site is a repeat.
        multiplet_q: q-value under which a cell is a multiplet.
        chroms: Chromosomes to use; `None` for the autosomes.
        barcode_map: Cell id to fragment barcode, where they differ.
        max_mem_gb: Read budget; `None` reads one chromosome at a time (with a
            tabix index).
        count_trailing_loci: Count a cell's last loci on the last chromosome
            used, which the reference implementation drops.

    Returns:
        The obs columns added, with `cell_idx`.

    Raises:
        ValueError: If no fragment lies on a chromosome AMULET uses, or no
            restriction region does.
    """
    fragments = Path(fragments)
    _check_fragments(fragments, 1)
    r = _bixverse.atac_amulet(
        fragments,
        _fragment_barcodes(sc.cell_ids(), barcode_map),
        [Path(b) for b in restrict_beds],
        max_overlap,
        max_fragment_len,
        repeat_q,
        multiplet_q,
        None if chroms is None else list(chroms),
        max_mem_gb,
        count_trailing_loci,
    )
    frame = pl.DataFrame(
        {
            "cell_idx": sc._all_cells(),
            "amulet_n_overlaps": r["n_overlaps"].astype(np.int64),
            "amulet_n_sites": r["n_sites"].astype(np.int64),
            "amulet_p": r["p_value"],
            "amulet_q": r["q_value"],
            "amulet_multiplet": r["multiplet"],
        }
    )
    sc.add_obs(frame)
    return frame


####################
# TF-IDF, features #
####################


@beartype
def run_tfidf(x: AtacHost, *, scale: float = TF_IDF_SCALE) -> np.ndarray:
    """Switch the ATAC store to TF-IDF over the kept cells (Signac method 1).

    Header metadata only; no count is rewritten. Run again after the kept
    cells change. Everything in the ATAC cache (LSI, embeddings, kNN) was
    built on the old norm and is dropped; rerun `calculate_lsi`.

    Args:
        x: The experiment, with ATAC.
        scale: Scale factor of the term frequencies.

    Returns:
        The IDF per feature.

    Raises:
        ValueError: If no ATAC store was added.
    """
    info = require_atac(x)
    cells, genes = atac_paths(x)
    idf = _bixverse.atac_set_tf_idf(
        cells, genes, np.asarray(x.cells_to_keep, dtype=np.uint32), scale
    )
    info.tf_idf = True
    x.atac_cache = ScCache()
    return idf


@beartype
def find_top_features(
    x: AtacHost, *, min_cutoff: str | float | None = "q5"
) -> np.ndarray:
    """The most accessible features (Signac `FindTopFeatures`).

    Totals are summed over the kept cells; the ATAC feature table gains
    `count`, `percentile` and `top_feature`.

    Args:
        x: The experiment, with ATAC.
        min_cutoff: `"qN"` keeps features above the N-th percentile of the
            totals, a number those with more counts, `None` or `"all"` all.

    Returns:
        0-based indices of the top features.

    Raises:
        ValueError: If no ATAC store was added, or for a malformed
            `min_cutoff`.
    """
    require_atac(x)
    _, genes = atac_paths(x)
    t = _bixverse.atac_feature_totals(
        genes,
        np.asarray(x.cells_to_keep, dtype=np.uint32),
        None if min_cutoff is None else str(min_cutoff),
    )
    keep = t["top_feature"]
    _add_atac_var(
        x,
        pl.DataFrame(
            {
                "count": t["count"],
                "percentile": t["percentile"],
                "top_feature": keep,
            }
        ),
    )
    return np.flatnonzero(keep).astype(np.uint32)


#######
# LSI #
#######


@beartype
def calculate_lsi(
    x: AtacHost,
    *,
    n_comp: int = 50,
    features: np.ndarray | Sequence[int] | None = None,
    drop_first: bool = True,
    solver: Literal["auto", "exact", "randomised", "covariance"] = "auto",
    scale_max: float | None = None,
    seed: int = 42,
    verbose: Verbose = 1,
) -> np.ndarray:
    """LSI over the TF-IDF of the kept cells (Signac `RunSVD`).

    The embedding (components z-scored over the cells) goes to
    `x.atac_cache.embeddings["lsi"]`, stamped `"atac:lsi"`, for
    `find_neighbours`, `harmony`, `umap` and `generate_wnn_graph` with
    `modality="atac"`, `embd_to_use="lsi"`.

    Args:
        x: The experiment, after `run_tfidf`.
        n_comp: Components to compute.
        features: 0-based features; `None` takes `top_feature` from the
            feature table if `find_top_features` ran, else all.
        drop_first: Leave component 1 out of the embedding. It usually tracks
            sequencing depth; `get_lsi_depth_cor` and `bv.pl.lsi_depth_plot`
            show whether it does.
        solver: SVD solver.
        scale_max: Clip the z-scores to `+-scale_max`.
        seed: Seed of the solver.
        verbose: `0` silent, `1` normal, `2` detailed.

    Returns:
        The correlation of every component with the library size.

    Raises:
        ValueError: Before `run_tfidf`.
    """
    info = require_atac(x)
    if not info.tf_idf:
        raise ValueError("Run run_tfidf() before calculate_lsi().")
    cells, genes = atac_paths(x)
    if features is None:
        var = atac_var(x)
        feats = (
            np.flatnonzero(var["top_feature"].to_numpy())
            if "top_feature" in var.columns
            else np.arange(var.height)
        )
    else:
        feats = np.asarray(features)
    res = _bixverse.atac_lsi(
        cells,
        genes,
        np.asarray(x.cells_to_keep, dtype=np.uint32),
        feats.astype(np.uint32),
        n_comp,
        solver,
        scale_max=scale_max,
        seed=seed,
        verbose=verbose,
    )
    emb = res["embedding"]
    dropped: tuple[int, ...] = ()
    if drop_first:
        emb, dropped = emb[:, 1:], (0,)
    x.atac_cache.embeddings["lsi"] = np.ascontiguousarray(emb)
    _stamp(x, "lsi")
    info.depth_cor = np.asarray(res["depth_cor"])
    info.lsi_dropped = dropped
    info.lsi_features = feats.astype(np.uint32)
    info.lsi_loadings = np.asarray(res["loadings"], dtype=np.float32)
    info.lsi_singular_values = np.asarray(res["singular_values"], dtype=np.float32)
    return info.depth_cor


@beartype
def _stamp(x: AtacHost, label: str, parents: Sequence[str] = ()) -> None:
    """Stamp an ATAC artefact in whichever cache holds it."""
    if isinstance(x, SingleCellsMultiModal):
        x.stamp(label, parents=parents, modality="atac")
    else:
        x.stamp(label, parents=parents)


@beartype
def project_lsi(reference: AtacHost, query: AtacHost) -> np.ndarray:
    """Project the query's kept cells onto the reference LSI.

    Seurat's `ProjectSVD(mode = "lsi")` as `FindTransferAnchors(reduction =
    "lsiproject")` runs it: the query's own TF-IDF times the reference
    loadings, divided by the singular values, with no centring or scaling.
    The reference `"lsi"` embedding is z-scored, the projection is not; L2
    normalise both per cell before comparing them, as Seurat does. Components
    the reference dropped are dropped here too. Either side may be
    multi-modal or ATAC-only.

    Written to `query.atac_cache.embeddings` as `"lsi_projected"`.

    Args:
        reference: The experiment after `calculate_lsi`.
        query: The experiment to project, after `run_tfidf`, counted over the
            reference features (`peaks=reference.var(modality="atac")`).

    Returns:
        Kept query cells x components.

    Raises:
        ValueError: Without a reference LSI or a query TF-IDF, or if a
            reference LSI feature is not in the query.
    """
    ref = require_atac(reference)
    if (
        ref.lsi_loadings is None
        or ref.lsi_features is None
        or ref.lsi_singular_values is None
    ):
        raise ValueError("Run calculate_lsi() on the reference first.")
    if not require_atac(query).tf_idf:
        raise ValueError("Run run_tfidf() on the query first.")

    @beartype
    def regions(v: pl.DataFrame) -> list[tuple[str, int, int]]:
        """The `(chrom, start, end)` of every row of a var table."""
        return list(zip(v["chrom"], v["start"], v["end"], strict=True))

    ref_regions = regions(atac_var(reference))
    lookup = {r: i for i, r in enumerate(regions(atac_var(query)))}
    wanted = [ref_regions[i] for i in ref.lsi_features]
    missing = [r for r in wanted if r not in lookup]
    if missing:
        c, s, e = missing[0]
        raise ValueError(
            f"{len(missing)} reference LSI features are not in the query, e.g. "
            f"{c}:{s}-{e}. Count the query over the reference features."
        )
    cells, genes = atac_paths(query)
    emb = _bixverse.atac_project_lsi(
        cells,
        genes,
        np.asarray(query.cells_to_keep, dtype=np.uint32),
        np.asarray([lookup[r] for r in wanted], dtype=np.uint32),
        np.ascontiguousarray(ref.lsi_loadings, dtype=np.float32),
        np.ascontiguousarray(ref.lsi_singular_values, dtype=np.float32),
    )
    keep = [j for j in range(emb.shape[1]) if j not in ref.lsi_dropped]
    emb = np.ascontiguousarray(emb[:, keep])
    query.atac_cache.embeddings["lsi_projected"] = emb
    _stamp(query, "lsi_projected")
    return emb


@beartype
def run_rlsi(
    x: AtacHost,
    batch_column: str,
    *,
    embedding: str = "lsi",
    features: np.ndarray | Sequence[int] | None = None,
    dims: Sequence[int] = tuple(range(1, 30)),
    n_comp: int = 30,
    k_anchor: int = 5,
    k_score: int = 30,
    k_weight: int = 100,
    sd: float = 1.0,
    l2_norm: bool = True,
    solver: Literal["auto", "exact", "randomised", "covariance"] = "auto",
    knn_params: KnnParams | None = None,
    seed: int = 42,
    verbose: Verbose = 1,
) -> np.ndarray:
    """rLSI integration, Signac's `FindIntegrationAnchors(reduction = "rlsi")`.

    Then `IntegrateEmbeddings`. Per batch: IDF over its own cells, TF-IDF
    and LSI of `features`; anchors between every batch pair in the
    reciprocal LSI spaces; then `embedding` is corrected along the anchor
    tree. Written to `x.atac_cache.embeddings` as `"rlsi"`.

    Signac integrates all LSI components and leaves component 1 out
    afterwards: `calculate_lsi(x, drop_first=False)`, `run_rlsi(x, ...)`,
    then `"rlsi"[:, 1:]`. With the default `drop_first=True` the depth
    component is left out of the integration too, which changes the
    correction weights.

    Args:
        x: The experiment, after `calculate_lsi`.
        batch_column: obs column with the batch of each cell.
        embedding: The ATAC embedding to integrate.
        features: 0-based ATAC feature rows of the per-batch LSIs. `None`
            uses those of `calculate_lsi`.
        dims: 0-based per-batch LSI components the anchors use; the default
            is Signac's `dims = 2:30`.
        n_comp: Components of the per-batch LSIs; must exceed `max(dims)`.
        k_anchor: Neighbours for the anchors.
        k_score: Neighbours for the anchor scores.
        k_weight: Anchors per cell in the correction weights.
        sd: Bandwidth divisor of the weights.
        l2_norm: Scale and L2 normalise the anchor spaces.
        solver: SVD solver of the per-batch LSIs.
        knn_params: kNN backend. `None` uses the defaults.
        seed: Seed of the solvers and the kNN backend.
        verbose: `0` silent, `1` normal, `2` detailed.

    Returns:
        The integrated embedding, kept cells x its dimensions.

    Raises:
        ValueError: Without the embedding, with fewer than two batches, or if
            `batch_column` is not an obs column or has missing values.
    """
    info = require_atac(x)
    emb = x.atac_cache.embeddings.get(embedding)
    if emb is None:
        raise ValueError(f"No '{embedding}' ATAC embedding. Run calculate_lsi() first.")
    if features is None:
        if info.lsi_features is None:
            raise ValueError("Run calculate_lsi() first or pass features.")
        feats = info.lsi_features
    else:
        feats = np.asarray(features)
    codes, _ = _labels(x, batch_column)
    cells, genes = atac_paths(x)
    out = _bixverse.atac_rlsi(
        cells,
        genes,
        np.asarray(x.cells_to_keep, dtype=np.uint32),
        np.asarray(feats, dtype=np.uint32),
        codes,
        np.ascontiguousarray(emb, dtype=np.float32),
        np.asarray(dims, dtype=np.uint32),
        n_comp,
        solver,
        k_anchor,
        k_score,
        k_weight,
        sd,
        l2_norm,
        to_rust(knn_params or KnnParams()),
        seed,
        verbose,
    )
    x.atac_cache.embeddings["rlsi"] = out
    _stamp(x, "rlsi", parents=[embedding])
    return out


@beartype
def get_lsi_depth_cor(x: AtacHost) -> pl.DataFrame:
    """Correlation of each LSI component with the library size.

    Args:
        x: The experiment, after `calculate_lsi`.

    Returns:
        `component` (1-based), `depth_cor`, `dropped`.

    Raises:
        ValueError: Before `calculate_lsi`.
    """
    info = require_atac(x)
    if info.depth_cor is None:
        raise ValueError("Run calculate_lsi() first.")
    n = len(info.depth_cor)
    return pl.DataFrame(
        {
            "component": np.arange(1, n + 1),
            "depth_cor": info.depth_cor,
            "dropped": [i in info.lsi_dropped for i in range(n)],
        }
    )


################
# Peak to gene #
################


@beartype
def peak_to_gene(
    mm: SingleCellsMultiModal,
    gtf: Path | str,
    group_by: str,
    *,
    max_distance: int = 250_000,
    biotypes: Sequence[str] = ("protein_coding",),
    genes: Sequence[str] | None = None,
    min_cells: int = 10,
    fdr: float | None = 0.05,
    verbose: Verbose = 1,
) -> pl.DataFrame:
    """Link peaks to nearby genes by correlation over groups of cells.

    ArchR's `addPeak2GeneLinks`. Peak and gene counts are summed per level of
    `group_by` (fine clusters, or metacell assignments, of the kept cells),
    log-normalised per group,
    `log2(x / total * 1e4 + 1)`, and correlated over the groups for every
    peak within `max_distance` of a gene's TSS. p-values are the two-sided
    t-test of each correlation, FDR Benjamini-Hochberg over all pairs. Unlike
    Signac's `LinkPeaks` there is no GC-matched background, which needs the
    genome sequence.

    Args:
        mm: A multi-modal experiment with RNA and ATAC.
        gtf: Gene annotation for the TSSs, matched to the RNA on `gene_id`
            without its version.
        group_by: obs column whose levels are the groups; ArchR uses a few
            hundred groups of ~100 cells.
        max_distance: Largest distance from a peak to a TSS.
        biotypes: Gene biotypes to link.
        genes: Gene ids to restrict to; `None` for every gene with a TSS.
        min_cells: Groups with fewer kept cells are left out.
        fdr: Keep links below this FDR; `None` keeps every tested pair.
        verbose: `0` silent, `1` normal, `2` detailed.

    Returns:
        `peak_id`, `gene_id`, `gene_name`, `distance` (peak centre minus
        TSS), `r`, `p`, `fdr`, sorted by gene and distance.

    Raises:
        ValueError: With fewer than three groups.
    """
    require_atac(mm)
    codes, levels = _labels(mm, group_by)
    kept = mm.cells_to_keep
    groups = [kept[codes == g] for g in range(len(levels))]
    groups = [g.astype(np.uint32) for g in groups if len(g) >= min_cells]
    if len(groups) < 3:
        raise ValueError(f"{len(groups)} groups of at least {min_cells} cells; need 3.")
    tss = _tss(gtf, biotypes)
    tss_frame = pl.DataFrame(
        {
            "gene_id": pl.Series(tss["gene_id"], dtype=pl.String).str.replace(
                r"\.\d+", ""
            ),
            "gene_name": pl.Series(tss["gene_name"], dtype=pl.String),
            "chrom": pl.Series(tss["chrom"], dtype=pl.String),
            "pos": np.asarray(tss["pos"], dtype=np.int64),
        }
    ).unique("gene_id", keep="first", maintain_order=True)
    rna_ids = mm.gene_ids().tolist()
    lookup = {g: i for i, g in enumerate(rna_ids)}
    tss_frame = tss_frame.filter(pl.col("gene_id").is_in(list(lookup)))
    if genes is not None:
        tss_frame = tss_frame.filter(pl.col("gene_id").is_in(list(genes)))
    gene_cols = [lookup[g] for g in tss_frame["gene_id"]]

    rna = np.asarray(
        _bixverse.pseudobulk(str(mm.cells_path), groups, "raw", False, verbose),
        dtype=np.float64,
    )[:, gene_cols]
    cells, _ = atac_paths(mm)
    atac = np.asarray(
        _bixverse.pseudobulk(str(cells), groups, "raw", False, verbose),
        dtype=np.float64,
    )
    var = atac_var(mm, cols=["feature_id", "chrom", "start", "end"])
    r = _bixverse.atac_peak_to_gene(
        np.ascontiguousarray(atac),
        np.ascontiguousarray(rna),
        var["chrom"].cast(pl.String).to_list(),
        [int(v) for v in var["start"]],
        [int(v) for v in var["end"]],
        tss_frame["chrom"].to_list(),
        [int(v) for v in tss_frame["pos"]],
        max_distance,
        1e4,
    )
    peak_ids = var["feature_id"].to_numpy()
    out = pl.DataFrame(
        {
            "peak_id": pl.Series(peak_ids[r["peak"]], dtype=pl.String),
            "gene_id": pl.Series(
                tss_frame["gene_id"].to_numpy()[r["gene"]], dtype=pl.String
            ),
            "gene_name": pl.Series(
                tss_frame["gene_name"].to_numpy()[r["gene"]], dtype=pl.String
            ),
            "distance": r["distance"],
            "r": r["r"],
            "p": r["p"],
            "fdr": r["fdr"],
        }
    )
    if fdr is not None:
        out = out.filter(pl.col("fdr") < fdr)
    return out
