"""SnapATAC2 reference values on the chr22 fixture inputs.

Run from the repository root, after gen_atac_inputs.R and gen_atac_signac.R:

    RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \\
    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    uv run --python 3.12 --with snapatac2==2.10.0 \\
        --with beartype python dev/atac/gen_atac_scverse.py

Writes snapatac2/ under
crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/. Coordinates are 0-based
half-open throughout (SnapATAC2 works in BED coordinates already).
"""

import gzip
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import snapatac2 as snap
from beartype import beartype

###########
# Globals #
###########

FIX_DIR: Path = Path("crates/bixverse-atac/tests/fixtures/pbmc3k_chr22")
STRATEGIES: list[str] = ["fragment", "insertion", "paired-insertion"]


@beartype
def fmt(x: float) -> str:
    """Float with 17 significant digits."""
    return f"{x:.17g}"


@beartype
def parse_range(name: str) -> tuple[str, int, int]:
    """Parse SnapATAC2's `chr:start-end` feature names (0-based start)."""
    chrom, rest = name.rsplit(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


@beartype
def write_sparse(
    mat: sp.csr_matrix,
    features: list[tuple[str, int, int, str]] | None,
    out: Path,
) -> None:
    """Write a cells x features matrix as feature-major triplets plus features.tsv.gz.

    Tiles pass `features=None`: they follow from bin size and chromosome length.
    """
    out.mkdir(parents=True, exist_ok=True)
    if features is not None:
        with gzip.open(out / "features.tsv.gz", "wt") as f:
            f.write("chrom\tstart\tend\tname\n")
            for c, s, e, n in features:
                f.write(f"{c}\t{s}\t{e}\t{n}\n")
    coo = sp.coo_matrix(mat.T)
    order = np.lexsort((coo.col, coo.row))
    vals = coo.data[order]
    is_int = bool(np.all(vals == np.round(vals)))
    with gzip.open(out / "matrix.tsv.gz", "wt") as f:
        f.write("feature_idx\tcell_idx\tvalue\n")
        for r, c, v in zip(coo.row[order], coo.col[order], vals):
            val = str(int(v)) if is_int else fmt(float(v))
            f.write(f"{r}\t{c}\t{val}\n")


@beartype
def write_dense(header: list[str], rows: list[list[str]], path: Path) -> None:
    """Write a TSV with a header, gzipped when the name ends in .gz."""
    with gzip.open(path, "wt") if path.suffix == ".gz" else path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


@beartype
def read_triplets(path: Path, n_features: int, n_cells: int) -> sp.csr_matrix:
    """Read feature-major triplets into a cells x features matrix."""
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return sp.csr_matrix(
        (data[:, 2], (data[:, 1].astype(int), data[:, 0].astype(int))),
        shape=(n_cells, n_features),
    )


@beartype
def align_cells(x: sp.csr_matrix, obs_names: list[str], cells: list[str]) -> sp.csr_matrix:
    """Reorder rows to cells.txt order, zero rows for cells SnapATAC2 did not keep."""
    pos = {b: i for i, b in enumerate(obs_names)}
    rows = sp.lil_matrix((len(cells), x.shape[1]), dtype=x.dtype)
    for i, b in enumerate(cells):
        if b in pos:
            rows[i] = x[pos[b]]
    return rows.tocsr()


@beartype
def read_tss_like_snapatac2(gtf: Path) -> list[tuple[str, int, str]]:
    """Mirror snapatac2-core `read_tss` plus the `.unique()` in the binding."""
    seen: set[tuple[str, int, bool]] = set()
    out: list[tuple[str, int, str]] = []
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            el = line.rstrip("\n").split("\t")
            if el[2] != "transcript":
                continue
            fwd = el[6] != "-"
            tss = int(el[3]) - 1 if fwd else int(el[4]) - 1
            key = (el[0], tss, fwd)
            if key not in seen:
                seen.add(key)
                out.append((el[0], tss, "+" if fwd else "-"))
    return out


@beartype
def assert_equal_to_signac(x: sp.csr_matrix, rel: str) -> None:
    """Fragment-mode outputs duplicate Signac's; check and skip writing them."""
    ref = read_triplets(FIX_DIR / "signac" / rel / "matrix.tsv.gz", x.shape[1], x.shape[0])
    assert (ref != x).nnz == 0, rel


@beartype
def run_snapatac2(cells: list[str], out: Path) -> None:
    """Import, peak and tile matrices in every counting strategy, TSSe, size distribution."""
    out.mkdir(parents=True, exist_ok=True)
    chrom, length = (FIX_DIR / "chrom_sizes.tsv").read_text().split()
    data = snap.pp.import_fragments(
        FIX_DIR / "fragments.tsv.gz",
        chrom_sizes={chrom: int(length)},
        sorted_by_barcode=False,
        whitelist=cells,
        min_num_fragments=0,
        chunk_size=2000,
    )
    obs_names = list(data.obs_names)

    peak_lines = (FIX_DIR / "peaks.bed").read_text().splitlines()
    peaks = [tuple(line.split("\t")) for line in peak_lines]
    peak_key = {(c, int(s), int(e)): i for i, (c, s, e) in enumerate(peaks)}
    peak_feats = [(c, int(s), int(e), f"{c}:{s}-{e}") for c, s, e in peaks]

    for strat in STRATEGIES:
        pm = snap.pp.make_peak_matrix(
            data, peak_file=FIX_DIR / "peaks.bed", counting_strategy=strat
        )
        idx = [peak_key[parse_range(v)] for v in pm.var_names]
        x = sp.csr_matrix(pm.X)
        perm = np.empty(len(idx), dtype=int)
        perm[idx] = np.arange(len(idx))
        x = align_cells(x[:, perm], obs_names, cells)
        if strat == "fragment":
            assert_equal_to_signac(x, "peaks")
        else:
            write_sparse(x, peak_feats, out / f"peaks_{strat}")

        for bin_size in (5000, 500):
            tm = snap.pp.add_tile_matrix(
                data, bin_size=bin_size, counting_strategy=strat, inplace=False
            )
            feats = [parse_range(v) for v in tm.var_names]
            n_len = int(length)
            assert feats == [
                (chrom, s, min(s + bin_size, n_len)) for s in range(0, n_len, bin_size)
            ]
            x = align_cells(sp.csr_matrix(tm.X), obs_names, cells)
            # no fragment here spans more than two 5 kb bins, so paired-insertion
            # equals fragment counting at that bin size
            if strat != "insertion" and bin_size == 5000:
                assert_equal_to_signac(x, "bins_5000")
            else:
                write_sparse(x, None, out / f"tiles_{bin_size}_{strat}")

    snap.metrics.tsse(data, FIX_DIR / "genes.gtf.gz", exclude_chroms=["chrM", "M"])
    snap.metrics.frag_size_distr(data, max_recorded_size=1000)

    obs = {b: i for i, b in enumerate(obs_names)}
    rows = []
    for b in cells:
        if b in obs:
            i = obs[b]
            rows.append(
                [
                    b,
                    "1",
                    str(int(data.obs["n_fragment"].iloc[i])),
                    fmt(float(data.obs["frac_dup"].iloc[i])),
                    fmt(float(data.obs["frac_mito"].iloc[i])),
                    fmt(float(data.obs["tsse"].iloc[i])),
                ]
            )
        else:
            rows.append([b, "0", "0", "nan", "nan", "nan"])
    write_dense(
        ["barcode", "imported", "n_fragment", "frac_dup", "frac_mito", "tsse"],
        rows,
        out / "cell_metrics.tsv",
    )

    profile = np.asarray(data.uns["TSS_profile"])
    write_dense(
        ["offset", "insertions"],
        [[str(i - 2000), str(int(v))] for i, v in enumerate(profile)],
        out / "tss_profile.tsv",
    )
    distr = np.asarray(data.uns["frag_size_distr"])
    write_dense(
        ["size", "fragments"],
        [[str(i), str(int(v))] for i, v in enumerate(distr)],
        out / "frag_size_distr.tsv",
    )
    tss = read_tss_like_snapatac2(FIX_DIR / "genes.gtf.gz")
    with gzip.open(out / "tss.bed.gz", "wt") as f:
        for c, t, s in tss:
            f.write(f"{c}\t{t}\t{t + 1}\t.\t.\t{s}\n")


@beartype
def check_signac_overlap(cells: list[str]) -> None:
    """Independent count of half-open fragment / peak overlaps against Signac's FeatureMatrix."""
    cell_idx = {b: i for i, b in enumerate(cells)}
    peaks = [
        (int(s), int(e))
        for _, s, e in (line.split("\t") for line in (FIX_DIR / "peaks.bed").read_text().splitlines())
    ]
    ref = np.zeros((len(peaks), len(cells)), dtype=np.int64)
    p_start = np.array([p[0] for p in peaks])
    p_end = np.array([p[1] for p in peaks])
    with gzip.open(FIX_DIR / "fragments.tsv.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            _, s, e, b, _ = line.rstrip("\n").split("\t")
            hit = np.nonzero((p_start < int(e)) & (p_end > int(s)))[0]
            ref[hit, cell_idx[b]] += 1
    signac = read_triplets(
        FIX_DIR / "signac" / "peaks" / "matrix.tsv.gz", len(peaks), len(cells)
    ).toarray().T
    diff = np.abs(ref - signac)
    print(f"signac peaks vs half-open overlap: {int((diff > 0).sum())} entries differ, max {int(diff.max())}")


@beartype
def main() -> None:
    cells = (FIX_DIR / "cells.txt").read_text().split()
    run_snapatac2(cells, FIX_DIR / "snapatac2")
    check_signac_overlap(cells)
    for out, pkgs in (
        (FIX_DIR / "snapatac2" / "versions.txt", ["snapatac2", "anndata", "numpy", "scipy"]),
    ):
        out.write_text("".join(f"{p} {version(p)}\n" for p in pkgs))


if __name__ == "__main__":
    main()
