"""AMULET probe for trailing loci on a multi-chromosome file.

chr21 and chr22 each carry the chr22 fixture's fragments plus, for five
cells, three overlapping fragments past everything else: a locus no later
fragment of the cell on that chromosome starts beyond. chrX and a contig
follow, which AMULET skips. The output says on which chromosomes the
reference counts such a locus. Writes BGZF plus tabix and a three-column
blacklist on chr21 and chr22.

Run from the repository root, then AMULET as in gen_atac_amulet.sh and keep
its OverlapSummary.txt (gzipped) next to the input:

    uv run --with pysam python dev/atac/gen_atac_amulet_trailing.py \\
        crates/bixverse-atac/tests/fixtures/pbmc3k_chr22 \\
        crates/bixverse-atac/tests/fixtures/amulet_trailing 4000
"""
import gzip
import sys
from pathlib import Path

import pysam

fix = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

header, body = [], []
with gzip.open(fix / "fragments.tsv.gz", "rt") as fh:
    for line in fh:
        (header if line.startswith("#") else body).append(line.rstrip("\n").split("\t"))
body = body[: int(sys.argv[3])]
cells = [l.strip() for l in open(fix / "cells.txt") if l.strip()][:5]
top = max(int(p[1]) for p in body)
pos = top + 10_000
extra = []
for i, bc in enumerate(cells):
    p = pos + i * 5_000
    for d in (0, 50, 100):
        extra.append([None, str(p + d), str(p + d + 200), bc, "1"])

chroms = ["chr21", "chr22", "chrX", "chrUn_KI270742v1"]
plain = out / "fragments.tsv"
with open(plain, "w") as fh:
    fh.writelines("\t".join(h) + "\n" for h in header)
    for c in chroms:
        rows = body + (extra if c in ("chr21", "chr22") else [])
        for p in rows:
            fh.write("\t".join([c] + p[1:]) + "\n")
pysam.tabix_compress(str(plain), str(out / "fragments.tsv.gz"), force=True)
pysam.tabix_index(str(out / "fragments.tsv.gz"), preset="bed", force=True)
plain.unlink()

with open(fix / "blacklist.bed") as fh:
    rows = [l.rstrip("\n").split("\t") for l in fh if l.strip()]
with open(out / "blacklist.bed", "w") as fh:
    for c in ["chr21", "chr22"]:
        for p in rows:
            fh.write("\t".join([c] + p[1:]) + "\n")
with open(out / "probe_cells.txt", "w") as fh:
    fh.write("\n".join(cells) + "\n")
print("ok", cells)
