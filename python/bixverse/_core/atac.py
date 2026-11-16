"""Single cell ATAC on a `SingleCellsMultiModal`: the implementations behind
`bixverse.atac`.

The ATAC store sits in `<dir>/atac/` and is always written for every cell of
the RNA store, in store order, so both stores share `cell_idx` and the kept
cell set of the RNA side applies to it unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.multimodal import AtacInfo, SingleCellsMultiModal
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN
from bixverse._core.state import ScCache
from bixverse._validate import Verbose

###########
# Globals #
###########

ATAC_DIR: str = "atac"

# Signac and muon
TF_IDF_SCALE: float = 1e4

MITO_CHROMS: tuple[str, ...] = ("chrM", "MT", "chrMT", "M")

CountMode = Literal["insertion", "paired-insertion", "fragment"]

###########
# Helpers #
###########


@beartype
def atac_paths(mm: SingleCellsMultiModal) -> tuple[Path, Path]:
    """The cell-based and gene-based (peak-major) files of the ATAC store."""
    d = mm.dir / ATAC_DIR
    return d / CELLS_BIN, d / GENES_BIN


@beartype
def require_atac(mm: SingleCellsMultiModal) -> AtacInfo:
    """The ATAC state.

    Raises:
        ValueError: If no ATAC store was added.
    """
    if mm.atac is None:
        raise ValueError(
            "No ATAC data in this object. Add it with add_atac_tenx() or "
            "add_atac_fragments()."
        )
    return mm.atac


def _region_ids(chrom: Sequence[str], start: np.ndarray, end: np.ndarray) -> list[str]:
    """`chrom:start-end` ids, 0-based half-open."""
    return [f"{c}:{s}-{e}" for c, s, e in zip(chrom, start, end, strict=True)]


@beartype
def _write_atac_tables(
    mm: SingleCellsMultiModal,
    feature_ids: list[str],
    chrom: list[str],
    start: np.ndarray,
    end: np.ndarray,
    lib_size: np.ndarray,
    nnz: np.ndarray,
    kind: Literal["peak", "tile"],
) -> None:
    """Write `var_atac`, the per-cell totals to obs, and reset the ATAC state."""
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
    mm.atac = AtacInfo(kind=kind)
    mm.atac_cache = ScCache()


############
# Ingest   #
############


@beartype
def add_atac_tenx(
    mm: SingleCellsMultiModal,
    h5_path: Path | str,
    *,
    max_mem_gb: float | None = None,
    verbose: Verbose = 1,
) -> None:
    """Add the `Peaks` of a 10x multiome h5 as the ATAC store.

    Args:
        mm: The experiment, its RNA loaded from the same h5 (or with the same
            barcodes).
        h5_path: The Cell Ranger ARC `filtered_feature_bc_matrix.h5`.
        max_mem_gb: Memory budget of the peak-major conversion.
        verbose: `0` silent, `1` normal.

    Raises:
        ValueError: If a cell of the store is not in the h5.
    """
    cells, genes = atac_paths(mm)
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
        "peak",
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
def add_atac_fragments(
    mm: SingleCellsMultiModal,
    fragments: Path | str,
    *,
    peaks: Path | str | pl.DataFrame | None = None,
    tile_width: int | None = None,
    chrom_sizes: Mapping[str, int] | None = None,
    mode: CountMode = "insertion",
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
        max_mem_gb: Memory budget of the peak-major conversion.
        verbose: `0` silent, `1` normal.
    """
    chrom, start, end, kind = _features(peaks, tile_width, chrom_sizes)
    cells, genes = atac_paths(mm)
    cells.parent.mkdir(exist_ok=True)
    res = _bixverse.atac_fragments_to_store(
        Path(fragments),
        cells,
        genes,
        [str(b) for b in mm.cell_ids()],
        chrom,
        [int(x) for x in start],
        [int(x) for x in end],
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
        kind,
    )


########
# QC   #
########


@beartype
def atac_fragment_qc(
    mm: SingleCellsMultiModal,
    fragments: Path | str,
    gtf: Path | str,
    *,
    biotypes: Sequence[str] = ("protein_coding",),
    mito_chroms: Sequence[str] = MITO_CHROMS,
) -> pl.DataFrame:
    """Per-cell QC from the fragment file, written to obs.

    Adds `atac_n_fragments`, `atac_nucleosome_signal`, `atac_tss_enrichment`
    (Signac's fast definition) and `atac_mito_fraction` for every cell of the
    store, plus `atac_frip` when an ATAC store exists (its library size over
    the fragments; insertion counts give two per fragment).

    Args:
        mm: The experiment.
        fragments: The fragment file.
        gtf: Gene annotation for the TSSs.
        biotypes: Gene biotypes whose TSSs count (Signac: protein coding).
        mito_chroms: Names of the mitochondrial genome.

    Returns:
        The obs columns added, with `cell_idx`.
    """
    tss = _bixverse.atac_tss_positions(Path(gtf), list(biotypes))
    qc = _bixverse.atac_fragment_qc(
        Path(fragments),
        [str(b) for b in mm.cell_ids()],
        list(tss["chrom"]),
        [int(x) for x in tss["pos"]],
        list(mito_chroms),
    )
    frame = pl.DataFrame(
        {
            "cell_idx": mm._all_cells(),
            "atac_n_fragments": qc["n_fragments"].astype(np.int64),
            "atac_nucleosome_signal": qc["nucleosome_signal"],
            "atac_tss_enrichment": qc["tss_enrichment"],
            "atac_mito_fraction": qc["mito_fraction"],
        }
    )
    if mm.atac is not None and "atac_lib_size" in mm.obs_columns():
        lib = mm.obs(cols=["atac_lib_size"])["atac_lib_size"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            frame = frame.with_columns(
                pl.Series("atac_frip", lib / qc["n_fragments"].astype(np.float64))
            )
    mm.add_obs(frame)
    return frame


####################
# TF-IDF, features #
####################


@beartype
def run_tfidf(mm: SingleCellsMultiModal, *, scale: float = TF_IDF_SCALE) -> np.ndarray:
    """Switch the ATAC store to TF-IDF over the kept cells (Signac method 1).

    Header metadata only; no count is rewritten. Run again after the kept
    cells change.

    Returns:
        The IDF per feature.
    """
    info = require_atac(mm)
    cells, genes = atac_paths(mm)
    idf = _bixverse.atac_set_tf_idf(
        cells, genes, np.sort(mm.cells_to_keep).astype(np.uint32), scale
    )
    info.tf_idf = True
    mm.atac_cache = ScCache()
    return idf


@beartype
def _parse_cutoff(min_cutoff: str | float | None) -> tuple[str, float]:
    """`("quantile", q)`, `("count", c)` or `("all", 0)`.

    Raises:
        ValueError: For a malformed `"qN"`.
    """
    if min_cutoff is None:
        return "all", 0.0
    if isinstance(min_cutoff, str):
        if not min_cutoff.startswith("q"):
            raise ValueError(
                f"min_cutoff must be 'qN', a number or None, got '{min_cutoff}'"
            )
        q = float(min_cutoff[1:])
        if not 0 <= q <= 100:
            raise ValueError("the percentile of min_cutoff must be in [0, 100]")
        return "quantile", q / 100
    return "count", float(min_cutoff)


@beartype
def find_top_features(
    mm: SingleCellsMultiModal, *, min_cutoff: str | float | None = "q5"
) -> np.ndarray:
    """The most accessible features (Signac `FindTopFeatures`).

    Totals are summed over the kept cells; `var_atac` gains `count`,
    `percentile` and `top_feature`.

    Args:
        mm: The experiment, with ATAC.
        min_cutoff: `"qN"` keeps features above the N-th percentile of the
            totals, a number those with more counts, `None` all.

    Returns:
        0-based indices of the top features.
    """
    require_atac(mm)
    _, genes = atac_paths(mm)
    t = _bixverse.atac_feature_totals(
        genes, np.sort(mm.cells_to_keep).astype(np.uint32)
    )
    how, x = _parse_cutoff(min_cutoff)
    match how:
        case "quantile":
            keep = t["percentile"] > x
        case "count":
            keep = t["count"] > x
        case _:
            keep = np.ones(len(t["count"]), dtype=bool)
    with db.connect(mm.db_path) as con:
        var = db.get_table(
            con, "var_atac", cols=["feature_id", "chrom", "start", "end"]
        )
        var = var.drop("feature_idx", strict=False).with_columns(
            pl.Series("count", t["count"]),
            pl.Series("percentile", t["percentile"]),
            pl.Series("top_feature", keep),
        )
        db.populate_from_frame(con, "var_atac", var)
    return np.flatnonzero(keep).astype(np.uint32)


#######
# LSI #
#######


@beartype
def calculate_lsi(
    mm: SingleCellsMultiModal,
    *,
    n_comp: int = 50,
    features: np.ndarray | Sequence[int] | None = None,
    drop_first: bool = True,
    solver: Literal["auto", "exact", "randomised", "covariance"] = "auto",
    scale_max: float | None = None,
    seed: int = 42,
    verbose: Verbose = 0,
) -> np.ndarray:
    """LSI over the TF-IDF of the kept cells (Signac `RunSVD`).

    The embedding (components z-scored over the cells) goes to
    `mm.atac_cache.embeddings["lsi"]`, stamped `"atac:lsi"`, for
    `find_neighbours`, `harmony`, `umap` and `generate_wnn_graph` with
    `modality="atac"`, `embd_to_use="lsi"`.

    Args:
        mm: The experiment, after `run_tfidf`.
        n_comp: Components to compute.
        features: 0-based features; `None` takes `top_feature` from
            `var_atac` if `find_top_features` ran, else all.
        drop_first: Leave component 1 out of the embedding. It usually tracks
            sequencing depth; check `get_lsi_depth_cor` before relying on it.
        solver: SVD solver.
        scale_max: Clip the z-scores to `+-scale_max`.
        seed: Seed of the solver.
        verbose: `0` silent, `1` normal, `2` detailed.

    Returns:
        The correlation of every component with the library size.

    Raises:
        ValueError: Before `run_tfidf`.
    """
    info = require_atac(mm)
    if not info.tf_idf:
        raise ValueError("Run run_tfidf() before calculate_lsi().")
    cells, genes = atac_paths(mm)
    if features is None:
        var = mm.var(modality="atac")
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
        np.asarray(mm.cells_to_keep, dtype=np.uint32),
        feats.astype(np.uint32),
        n_comp,
        solver,
        True,
        scale_max,
        seed,
        verbose,
    )
    emb = res["embedding"]
    dropped: tuple[int, ...] = (0,) if drop_first else ()
    if drop_first:
        emb = emb[:, 1:]
    mm.atac_cache.embeddings["lsi"] = np.ascontiguousarray(emb)
    mm.stamp("lsi", modality="atac")
    info.depth_cor = np.asarray(res["depth_cor"])
    info.lsi_dropped = dropped
    return info.depth_cor


@beartype
def get_lsi_depth_cor(mm: SingleCellsMultiModal) -> pl.DataFrame:
    """Correlation of each LSI component with the library size.

    Returns:
        `component` (1-based), `depth_cor`, `dropped`.

    Raises:
        ValueError: Before `calculate_lsi`.
    """
    info = require_atac(mm)
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
