"""MACS3 peak calls on the chr22 fixture, the reference for the native caller.

Run from the repository root, after gen_atac_inputs.R:

    uv run --python 3.12 --with macs3==3.1.0 --with beartype \\
        python dev/atac/gen_atac_macs3.py

Cells are split into three groups by their position in cells.txt (i % 3), as
the pseudobulks a per-cluster call would see. Every group and all cells
together are called with each combination of tag source, duplicate handling
and genome size:

- `insertion`: both Tn5 insertions of a fragment (raw start and raw end, the
  Cell Ranger convention) as 1 bp plus strand reads.
- `start`: the fragment file itself, as Signac's `CallPeaks` hands it to MACS
  (no strand column, so each fragment is a plus strand read at its start).

Always `--nomodel --extsize 200 --shift -100 -q 0.05`, Signac's settings.
Writes macs3/ under crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/: one
`<group>_<source>_dup<keep>_g<gsize>.tsv.gz` per call with 0-based half-open
coordinates, and `groups.txt`.
"""

import gzip
import shutil
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from beartype import beartype

###########
# Globals #
###########

FIX_DIR: Path = Path("crates/bixverse-atac/tests/fixtures/pbmc3k_chr22")
OUT_DIR: Path = FIX_DIR / "macs3"
N_GROUPS: int = 3
SOURCES: list[str] = ["insertion", "start"]
KEEP_DUPS: list[str] = ["all", "1"]
# 2.7e9 is Signac's default, under which the 10 kb local lambda dominates on
# one chromosome; 1e7 lifts the genome background above it.
GSIZES: list[str] = ["2.7e9", "1e7"]


@beartype
def read_fragments(cell_group: dict[str, int]) -> list[tuple[str, int, int, str, str, int]]:
    """Fragments of the fixture cells, in file order.

    Args:
        cell_group: Barcode to group.

    Returns:
        `(chrom, start, end, barcode, count, group)` per fragment line.
    """
    out = []
    with gzip.open(FIX_DIR / "fragments.tsv.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            chrom, start, end, bc, count = line.rstrip("\n").split("\t")[:5]
            if bc in cell_group:
                out.append((chrom, int(start), int(end), bc, count, cell_group[bc]))
    return out


@beartype
def write_bed(
    frags: list[tuple[str, int, int, str, str, int]],
    group: int | None,
    source: str,
    path: Path,
) -> None:
    """Write the MACS input of one group.

    Args:
        frags: From `read_fragments`.
        group: Group to keep, `None` for all cells.
        source: `insertion` or `start`.
        path: Output BED.
    """
    with path.open("w") as fh:
        for chrom, start, end, bc, count, g in frags:
            if group is not None and g != group:
                continue
            match source:
                case "start":
                    fh.write(f"{chrom}\t{start}\t{end}\t{bc}\t{count}\n")
                case "insertion":
                    fh.write(f"{chrom}\t{start}\t{start + 1}\t.\t0\t+\n")
                    fh.write(f"{chrom}\t{end}\t{end + 1}\t.\t0\t+\n")
                case _:
                    raise ValueError(f"unknown source '{source}'")


@beartype
def call_macs3(bed: Path, keep_dup: str, gsize: str, work: Path) -> list[str]:
    """Run `macs3 callpeak` and merge its narrowPeak and xls rows.

    Args:
        bed: Input BED.
        keep_dup: `--keep-dup` value.
        gsize: `-g` value.
        work: Scratch directory.

    Returns:
        TSV rows `chrom start end summit pileup pscore fold qscore`, start
        0-based, summit absolute and 0-based.
    """
    macs3 = shutil.which("macs3")
    if macs3 is None:
        raise RuntimeError("macs3 not on PATH")
    subprocess.run(
        [
            macs3, "callpeak", "-t", str(bed), "-f", "BED", "-g", gsize,
            "--nomodel", "--extsize", "200", "--shift", "-100",
            "--keep-dup", keep_dup, "-q", "0.05", "-n", "x",
            "--outdir", str(work),
        ],
        check=True,
        capture_output=True,
    )
    narrow = [
        line.rstrip("\n").split("\t")
        for line in (work / "x_peaks.narrowPeak").read_text().splitlines()
    ]
    xls = [
        line.split("\t")
        for line in (work / "x_peaks.xls").read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("chr\t")
    ]
    if len(narrow) != len(xls):
        raise RuntimeError("narrowPeak and xls disagree on the peak count")
    rows = []
    for n, x in zip(narrow, xls, strict=True):
        chrom, start, end = n[0], int(n[1]), int(n[2])
        summit = start + int(n[9])
        if int(x[2]) != end or int(x[4]) - 1 != summit:
            raise RuntimeError(f"xls row {x} does not match narrowPeak {n}")
        rows.append(f"{chrom}\t{start}\t{end}\t{summit}\t{x[5]}\t{n[7]}\t{n[6]}\t{n[8]}")
    return rows


@beartype
def main() -> None:
    """Write every call and the group assignment."""
    cells = (FIX_DIR / "cells.txt").read_text().split()
    cell_group = {bc: i % N_GROUPS for i, bc in enumerate(cells)}
    frags = read_fragments(cell_group)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "groups.txt").write_text("".join(f"{i % N_GROUPS}\n" for i in range(len(cells))))

    groups: list[tuple[str, int | None]] = [("all", None)] + [
        (f"g{g}", g) for g in range(N_GROUPS)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for label, group in groups:
            for source in SOURCES:
                bed = work / f"{label}_{source}.bed"
                write_bed(frags, group, source, bed)
                for keep_dup in KEEP_DUPS:
                    for gsize in GSIZES:
                        rows = call_macs3(bed, keep_dup, gsize, work)
                        name = f"{label}_{source}_dup{keep_dup}_g{gsize}.tsv.gz"
                        header = "chrom\tstart\tend\tsummit\tpileup\tpscore\tfold\tqscore\n"
                        with gzip.open(OUT_DIR / name, "wt") as fh:
                            fh.write(header + "".join(r + "\n" for r in rows))
                        print(f"{name}: {len(rows)} peaks")
    (OUT_DIR / "versions.txt").write_text(f"macs3 {version('macs3')}\n")


if __name__ == "__main__":
    main()
