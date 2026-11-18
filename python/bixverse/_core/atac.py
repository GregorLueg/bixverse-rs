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
from bixverse._core.pp import _factor_codes, _obs_aligned
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN, SingleCells
from bixverse._core.state import ScCache
from bixverse._validate import Verbose
from bixverse.params import KnnParams, to_rust

###########
# Globals #
###########

ATAC_DIR: str = "atac"

# Signac and muon
TF_IDF_SCALE: float = 1e4

MITO_CHROMS: tuple[str, ...] = ("chrM", "MT", "chrMT", "M")

CountMode = Literal["insertion", "paired-insertion", "fragment"]

TagSource = Literal["insertion", "fragment_start"]

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
    assume RNA counts refuse it or warn.

    Args:
        sc: The experiment whose cell IDs are the fragment barcodes.
        fragments: The fragment file (BGZF, gzip or plain).
        gtf: Gene annotation (GENCODE or Ensembl GTF).
        dir_out: Output directory. `None` uses `gene_activity` inside the
            experiment directory.
        upstream: Bases added upstream of the TSS.
        downstream: Bases added downstream of the gene end.
        max_width: Genes this long or longer are dropped.
        biotypes: Biotypes to keep, empty for all.
        target_size: Target of the log-CPM norm. `None` uses the median RNA
            `lib_size` of the kept cells, as the Signac vignettes do.
        overwrite: Replace an existing store at `dir_out`.
        max_mem_gb: Memory budget of the gene-major conversion.
        verbose: `0` silent, `1` normal.

    Returns:
        The gene activity experiment. var holds `gene_id` (the gene name, made
        unique), `gtf_gene_id`, `chrom`, `start`, `end`; obs's `lib_size` and
        `nnz` are the gene activity's.

    Raises:
        FileExistsError: If a store exists at `dir_out` and `overwrite` is
            false.
        ValueError: If `target_size` is `None` and obs has no `lib_size`.
    """
    out = sc.dir / "gene_activity" if dir_out is None else Path(dir_out).expanduser()
    cells, genes = out / CELLS_BIN, out / GENES_BIN
    if genes.exists() and not overwrite:
        raise FileExistsError(
            f"A store already exists at '{out}'. Pass overwrite=True to replace it."
        )
    obs = sc.obs()
    if target_size is None:
        if "lib_size" not in obs.columns:
            raise ValueError("obs has no lib_size; pass target_size.")
        kept = obs.filter(pl.col("cell_idx").is_in(sc.cells_to_keep))
        target_size = float(np.median(kept["lib_size"].to_numpy()))
    out.mkdir(parents=True, exist_ok=True)

    r = _bixverse.atac_gene_activity_regions(
        Path(gtf), upstream, downstream, max_width, list(biotypes)
    )
    res = _bixverse.atac_fragments_to_store(
        Path(fragments),
        cells,
        genes,
        [str(b) for b in sc.cell_ids()],
        list(r["chrom"]),
        [int(x) for x in r["start"]],
        [int(x) for x in r["end"]],
        "fragment",
        "gene_activity",
        target_size,
        max_mem_gb,
        verbose,
    )
    var = pl.DataFrame(
        {
            "gene_id": pl.Series(list(r["name"]), dtype=pl.String),
            "gtf_gene_id": pl.Series(list(r["gene_id"]), dtype=pl.String),
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
    max_mem_gb: float | None = None,
    tmp_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Call peaks on the kept cells, pooled or per group (MACS3 `callpeak`).

    A native port of MACS3 `callpeak --nomodel` without a control track, one
    call per group, as Signac's `CallPeaks(group.by = ...)` runs MACS. It
    reproduces MACS3 3.1.0's peaks exactly. Pass the result as `peaks` to
    `add_atac_fragments` to count against it.

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
        genome_size: Effective genome size (`-g`).
        llocal: Window of the local lambda (`--llocal`).
        q_value: q-value cutoff (`-q`).
        min_length: Minimum peak length; `None` for `extsize`.
        max_gap: Largest gap merged into a peak; `None` for MACS's tag size (1
            for insertions, the mean length of the group's first fragments
            for fragment starts).
        max_mem_gb: Tag budget of the reading pass; `None` reads one
            chromosome at a time.
        tmp_dir: Directory of the scratch file; `None` for the system
            temporary directory.

    Returns:
        With `combine`: `chrom`, `start`, `end` (0-based, half-open) and
        `peak_called_in`, the comma-separated groups. Without: one row per
        group and peak, with `group`, `summit` and the summit's `pileup`,
        `pscore` (`-log10 p`), `fold` and `qscore` (`-log10 q`).
    """
    kept = sc.cells_to_keep
    if group_by is None:
        codes = np.zeros(len(kept), dtype=np.uint32)
        levels = ["all"]
    else:
        codes, levels = _factor_codes(_obs_aligned(sc, [group_by])[group_by], group_by)
    r = _bixverse.atac_call_peaks(
        Path(fragments),
        [str(b) for b in sc.cell_ids(kept)],
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
    max_mem_gb: float | None = None,
    count_trailing_loci: bool = False,
) -> pl.DataFrame:
    """Flag multiplets from fragment overlaps (AMULET), written to obs.

    A diploid nucleus yields at most two overlapping fragments at a locus.
    Per cell, the loci with more are counted (autosomes, restriction regions
    dropped), sites shared by too many cells are dropped as repeats, and the
    count is tested against the mean (Poisson, Benjamini-Hochberg).
    Implemented from Thibodeau et al. (Genome Biology, 2021); it reproduces
    the reference implementation's counts and p-values. Needs no clustering
    and no RNA, so it works for ATAC-only experiments too.

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
        max_mem_gb: Read budget; `None` reads one chromosome at a time (with a
            tabix index).
        count_trailing_loci: Count a cell's last loci on a chromosome that no
            later fragment passes; the reference implementation does not.

    Returns:
        The obs columns added, with `cell_idx`.
    """
    r = _bixverse.atac_amulet(
        Path(fragments),
        [str(b) for b in sc.cell_ids()],
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
    info.lsi_features = feats.astype(np.uint32)
    info.lsi_loadings = np.asarray(res["loadings"], dtype=np.float32)
    info.lsi_singular_values = np.asarray(res["singular_values"], dtype=np.float32)
    return info.depth_cor


@beartype
def project_lsi(
    reference: SingleCellsMultiModal, query: SingleCellsMultiModal
) -> np.ndarray:
    """Project the query's kept cells onto the reference LSI.

    Seurat's `ProjectSVD(mode = "lsi")` as `FindTransferAnchors(reduction =
    "lsiproject")` runs it: the query's own TF-IDF times the reference
    loadings, divided by the singular values, with no centring or scaling.
    The reference `"lsi"` embedding is z-scored, the projection is not; L2
    normalise both per cell before comparing them, as Seurat does. Components
    the reference dropped are dropped here too.

    Written to `query.atac_cache.embeddings` as `"lsi_projected"`.

    Args:
        reference: The experiment after `calculate_lsi`.
        query: The experiment to project, after `run_tfidf`, counted over the
            reference features (`add_atac_fragments(query, ...,
            peaks=reference.var(modality="atac"))`).

    Returns:
        Kept query cells x components.

    Raises:
        ValueError: Without a reference LSI or a query TF-IDF, or if a
            reference LSI feature is not in the query.
    """
    ref = require_atac(reference)
    if ref.lsi_loadings is None or ref.lsi_features is None:
        raise ValueError("Run calculate_lsi() on the reference first.")
    if ref.lsi_singular_values is None:
        raise ValueError("Run calculate_lsi() on the reference first.")
    if not require_atac(query).tf_idf:
        raise ValueError("Run run_tfidf() on the query first.")

    def regions(v: pl.DataFrame) -> list[tuple[str, int, int]]:
        return list(zip(v["chrom"], v["start"], v["end"], strict=True))

    ref_regions = regions(reference.var(modality="atac"))
    lookup = {r: i for i, r in enumerate(regions(query.var(modality="atac")))}
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
    query.stamp("lsi_projected", modality="atac")
    return emb


@beartype
def run_rlsi(
    mm: SingleCellsMultiModal,
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
    verbose: Verbose = 0,
) -> np.ndarray:
    """rLSI integration (Signac's `FindIntegrationAnchors(reduction = "rlsi")`
    plus `IntegrateEmbeddings`).

    Per batch: IDF over its own cells, TF-IDF and LSI of `features`; anchors
    between every batch pair in the reciprocal LSI spaces; then `embedding`
    is corrected along the anchor tree. Written to `mm.atac_cache.embeddings`
    as `"rlsi"`.

    Signac integrates all LSI components and leaves component 1 out
    afterwards: `calculate_lsi(mm, drop_first=False)`, `run_rlsi(mm, ...)`,
    then `"rlsi"[:, 1:]`. With the default `drop_first=True` the depth
    component is left out of the integration too, which changes the
    correction weights.

    Args:
        mm: The experiment, after `calculate_lsi`.
        batch_column: obs column with the batch of each cell.
        embedding: The ATAC embedding to integrate.
        features: 0-based `var_atac` rows of the per-batch LSIs. `None` uses
            those of `calculate_lsi`.
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
        ValueError: Without the embedding, or with fewer than two batches.
    """
    info = require_atac(mm)
    emb = mm.atac_cache.embeddings.get(embedding)
    if emb is None:
        raise ValueError(f"No '{embedding}' ATAC embedding. Run calculate_lsi() first.")
    if features is None:
        if info.lsi_features is None:
            raise ValueError("Run calculate_lsi() first or pass features.")
        feats = info.lsi_features
    else:
        feats = np.asarray(features)
    codes, _ = _factor_codes(
        _obs_aligned(mm, [batch_column])[batch_column], batch_column
    )
    cells, genes = atac_paths(mm)
    out = _bixverse.atac_rlsi(
        cells,
        genes,
        np.asarray(mm.cells_to_keep, dtype=np.uint32),
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
    mm.atac_cache.embeddings["rlsi"] = out
    mm.stamp("rlsi", parents=[embedding], modality="atac")
    return out


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
