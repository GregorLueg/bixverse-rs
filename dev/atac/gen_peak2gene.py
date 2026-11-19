"""Reference peak to gene links for bixverse-atac, from numpy and scipy.

ArchR's recipe (`addPeak2GeneLinks`) on a seeded toy: groups x peaks and
groups x genes summed counts, log2(x / total * 1e4 + 1) per group, Pearson
over the groups (`scipy.stats.pearsonr`, two-sided t-test p-value) for every
peak whose span reaches within max_distance of a TSS, Benjamini-Hochberg
(`scipy.stats.false_discovery_control`) over all tested pairs.

Run from the repository root:

    uv run --with numpy==2.5.5 --with scipy==1.18.1 \\
        python dev/atac/gen_peak2gene.py

Writes crates/bixverse-atac/tests/fixtures/peak2gene/.
"""

from pathlib import Path

import numpy as np
import scipy
from scipy import stats

OUT = Path("crates/bixverse-atac/tests/fixtures/peak2gene")
MAX_DISTANCE = 50_000
N_GROUPS, N_PEAKS, N_GENES = 40, 60, 8


def main() -> None:
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)
    chroms = rng.choice(["chr1", "chr2"], size=N_PEAKS)
    starts = rng.integers(0, 400_000, size=N_PEAKS)
    ends = starts + rng.integers(200, 1_500, size=N_PEAKS)
    peaks = rng.poisson(rng.gamma(1.0, 3.0, size=(N_GROUPS, N_PEAKS)))
    genes = rng.poisson(rng.gamma(1.0, 6.0, size=(N_GROUPS, N_GENES)))
    # a few genes driven by a nearby peak, so some links are real
    genes[:, 0] += 3 * peaks[:, 0]
    genes[:, 1] += 2 * peaks[:, 5]
    peaks[:, 7] = 0  # constant peak: never tested
    tss_chrom = rng.choice(["chr1", "chr2", "chr3"], size=N_GENES)
    tss_chrom[0], tss_chrom[1] = chroms[0], chroms[5]
    tss_pos = rng.integers(0, 400_000, size=N_GENES)
    tss_pos[0], tss_pos[1] = starts[0] + 20_000, ends[5] + 1_000

    def norm(x: np.ndarray) -> np.ndarray:
        return np.log2(x / x.sum(axis=1, keepdims=True) * 1e4 + 1)

    zp, zg = norm(peaks.astype(float)), norm(genes.astype(float))
    rows = []
    for g in range(N_GENES):
        if zg[:, g].std() == 0:
            continue
        lo, hi = tss_pos[g] - MAX_DISTANCE, tss_pos[g] + MAX_DISTANCE + 1
        hit = [
            p
            for p in range(N_PEAKS)
            if chroms[p] == tss_chrom[g] and starts[p] < hi and ends[p] > lo
        ]
        for p in sorted(hit, key=lambda p: (starts[p], p)):
            if zp[:, p].std() == 0:
                continue
            r, pv = stats.pearsonr(zp[:, p], zg[:, g])
            centre = (starts[p] + ends[p]) // 2
            rows.append((p, g, int(centre - tss_pos[g]), float(r), float(pv)))
    fdr = stats.false_discovery_control([r[4] for r in rows], method="bh")

    with open(OUT / "peaks.tsv", "w") as f:
        f.write("chrom\tstart\tend\n")
        f.writelines(f"{c}\t{s}\t{e}\n" for c, s, e in zip(chroms, starts, ends))
    with open(OUT / "tss.tsv", "w") as f:
        f.write("chrom\tpos\n")
        for c, p in zip(tss_chrom, tss_pos):
            f.write(f"{c}\t{p}\n")
    for name, m in (("peak_counts.tsv", peaks), ("gene_counts.tsv", genes)):
        np.savetxt(OUT / name, m, fmt="%d", delimiter="\t")
    with open(OUT / "links.tsv", "w") as f:
        f.write("peak\tgene\tdistance\tr\tp\tfdr\n")
        for (p, g, d, r, pv), q in zip(rows, fdr):
            f.write(f"{p}\t{g}\t{d}\t{r:.17g}\t{pv:.17g}\t{q:.17g}\n")
    with open(OUT / "versions.txt", "w") as f:
        f.write(f"numpy {np.__version__}, scipy {scipy.__version__}, max_distance {MAX_DISTANCE}\n")


if __name__ == "__main__":
    main()
