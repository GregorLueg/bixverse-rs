"""ATAC-only experiments (`SingleCellsAtac`), the ATAC state on reopen, and the
fragment passes added with them, on the chr22 PBMC fixtures.

The ATAC-only paths share their implementation with the multi-modal ones, so
most checks here are that both give the same numbers on the same cells.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import atac
from bixverse.params import MinQualityParams, SyntheticDataParams

FIX = Path(__file__).parents[2] / "crates/bixverse-atac/tests/fixtures/pbmc3k_chr22"
FRAGS = FIX / "fragments.tsv.gz"
GTF = FIX / "genes.gtf.gz"
QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)


def _tsv(rel: str) -> pl.DataFrame:
    return pl.read_csv(FIX / rel, separator="\t", null_values=["NaN"])


@pytest.fixture(scope="module")
def barcodes():
    return (FIX / "cells.txt").read_text().split()


@pytest.fixture
def x(barcodes, tmp_path):
    """ATAC only, the fixture's cells, Signac's fragment counting."""
    return bv.io.load_atac_fragments(
        tmp_path / "atac",
        FRAGS,
        peaks=FIX / "peaks.bed",
        mode="fragment",
        barcodes=barcodes,
        verbose=0,
    )


@pytest.fixture
def mm(barcodes, tmp_path):
    """RNA (synthetic) plus the same ATAC counts."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=len(barcodes), n_genes=60, n_batches=1), seed=3
    )
    obs = data.obs.with_columns(pl.Series("cell_id", barcodes))
    bv.io.load_csr(tmp_path / "mm", data.counts, obs, data.var, qc=QC)
    mm = bv.SingleCellsMultiModal(tmp_path / "mm")
    atac.add_atac_fragments(mm, FRAGS, peaks=FIX / "peaks.bed", mode="fragment")
    return mm


def test_load_matches_signac_and_the_multimodal_store(x, mm):
    obs = x.obs(cols=["cell_id", "lib_size"])
    ref = _tsv("signac/cell_metrics.tsv")
    assert obs["cell_id"].to_list() == ref["barcode"].to_list()
    assert obs["lib_size"].to_list() == ref["nCount_peaks"].to_list()
    assert x.feature_kind == "peak" and x.atac.kind == "peak"
    assert x.atac.count_mode == "fragment" and not x.atac.tf_idf
    ours = x.var(modality="atac")
    theirs = mm.var(modality="atac")
    assert ours.columns[:2] == ["feature_idx", "feature_id"]
    assert ours["feature_id"].to_list() == theirs["feature_id"].to_list()


def test_barcode_sources(barcodes, tmp_path):
    counts = atac.barcode_fragment_counts(FRAGS)
    assert counts["n_fragments"].is_sorted(descending=True)
    cut = int(counts["n_fragments"][100])
    x = bv.io.load_atac_fragments(
        tmp_path / "cut", FRAGS, peaks=FIX / "peaks.bed", min_fragments=cut, verbose=0
    )
    want = sorted(counts.filter(pl.col("n_fragments") >= cut)["barcode"])
    assert x.cell_ids().tolist() == want

    csv = tmp_path / "singlecell.csv"
    rows = ["barcode,is__cell_barcode", "NO_BARCODE,0"]
    rows += [f"{b},{int(i % 2 == 0)}" for i, b in enumerate(barcodes)]
    csv.write_text("\n".join(rows) + "\n")
    assert atac.read_cellranger_barcodes(csv) == barcodes[::2]

    with pytest.raises(ValueError, match="exactly one"):
        bv.io.load_atac_fragments(tmp_path / "n", FRAGS, peaks=FIX / "peaks.bed")
    with pytest.raises(ValueError, match="exactly one"):
        bv.io.load_atac_fragments(
            tmp_path / "n",
            FRAGS,
            peaks=FIX / "peaks.bed",
            barcodes=barcodes,
            min_fragments=10,
        )


def test_barcode_counts_match_fragment_qc(x):
    qc = atac.fragment_qc(x, FRAGS, GTF, mito_chroms=[])
    counts = atac.barcode_fragment_counts(FRAGS, exclude_chroms=())
    lookup = dict(zip(counts["barcode"], counts["n_fragments"], strict=True))
    want = [lookup.get(b, 0) for b in x.cell_ids()]
    assert qc["atac_n_fragments"].to_list() == want


def test_lsi_is_one_code_path(x, mm):
    for e in (x, mm):
        atac.run_tfidf(e)
        atac.find_top_features(e, min_cutoff="q5")
        atac.calculate_lsi(e, n_comp=10, solver="exact", verbose=0)
    np.testing.assert_allclose(
        x.atac_cache.embeddings["lsi"], mm.atac_cache.embeddings["lsi"], atol=1e-5
    )
    np.testing.assert_allclose(x.atac.depth_cor, mm.atac.depth_cor, atol=1e-6)


def test_modality_names_the_root(x):
    atac.run_tfidf(x)
    atac.calculate_lsi(x, n_comp=10, solver="exact", verbose=0)
    bv.graph.find_neighbours(x, embd_to_use="lsi", modality="atac", verbose=0)
    assert x.cache.knn is not None
    bv.graph.find_neighbours(x, embd_to_use="lsi", verbose=0)
    assert x.check_state(["knn"]) is True
    with pytest.raises(ValueError, match="only supported"):
        bv.graph.find_neighbours(x, embd_to_use="lsi", modality="adt", verbose=0)


def test_differential_accessibility_defaults_to_lr(x, mm):
    a, b = list(range(250)), list(range(250, 500))
    ours = bv.de.find_markers(x, a, b, verbose=0)
    theirs = bv.de.find_markers(mm, a, b, modality="atac", verbose=0)
    assert ours.columns[0] == "feature_id"
    assert ours.equals(theirs)


def test_rna_methods_refuse_peaks(x):
    with pytest.raises(ValueError, match="works on genes"):
        bv.pp.find_hvg(x)
    with pytest.raises(ValueError, match="works on genes"):
        bv.pp.calculate_pca(x, no_pcs=5)
    with pytest.raises(ValueError, match="needs RNA counts"):
        bv.doublets.scrublet(x)


def test_peak_accessibility_plots_on_both(x, mm):
    peak = x.var(modality="atac")["feature_id"][0]
    for e in (x, mm):
        expr = bv.pl.extract_gene_expression(e, [peak], modality="atac")
        assert expr.columns == ["cell_id", peak]
        assert expr.height == 500


def test_state_is_rebuilt_from_the_store(x, mm):
    atac.run_tfidf(x)
    atac.calculate_lsi(x, n_comp=10, solver="exact", verbose=0)
    bv.io.save(x)
    back = bv.io.load_existing(x.dir)
    assert isinstance(back, bv.SingleCellsAtac)
    assert back.atac.tf_idf and back.atac.count_mode == "fragment"
    np.testing.assert_array_equal(back.atac.depth_cor, x.atac.depth_cor)
    assert "lsi" in back.atac_cache.embeddings

    # without the memory file: kind, count mode and TF-IDF from the store
    for e in (x, mm):
        for f in ("memory_py.npz", "memory_py.json"):
            (e.dir / f).unlink(missing_ok=True)
    assert bv.SingleCellsAtac(x.dir).atac.tf_idf
    reopened = bv.SingleCellsMultiModal(mm.dir)
    assert reopened.atac is not None and reopened.atac.count_mode == "fragment"

    with pytest.raises(ValueError, match="ATAC-only"):
        bv.io.load_existing(x.dir, multi_modal=True)
    with pytest.raises(ValueError, match="SingleCellsAtac"):
        bv.SingleCellsMultiModal(x.dir)


def test_gene_activity_follows_the_filter(x):
    ga = atac.gene_activity(x, FRAGS, GTF, verbose=0)
    assert ga.n_cells == x.n_cells
    x.set_cells_to_keep(list(range(100)))
    assert atac.get_gene_activity(x).n_cells == 100
    assert bv.SingleCells(x.dir / "gene_activity").n_cells == 100


def test_frip_after_recount(mm):
    atac.fragment_qc(mm, FRAGS, GTF)
    before = mm.obs(cols=["atac_frip"])["atac_frip"].to_numpy()
    atac.add_atac_fragments(
        mm, FRAGS, peaks=FIX / "peaks.bed", mode="insertion", overwrite=True
    )
    after = mm.obs(cols=["atac_frip"])["atac_frip"].to_numpy()
    n = mm.obs(cols=["atac_n_fragments"])["atac_n_fragments"].to_numpy()
    lib = mm.obs(cols=["atac_lib_size"])["atac_lib_size"].to_numpy()
    ok = n > 0
    np.testing.assert_allclose(after[ok], lib[ok] / (2.0 * n[ok]))
    assert not np.allclose(before[ok], after[ok])


def test_mismatched_inputs_fail_loudly(x, barcodes, tmp_path):
    suffixless = [b.removesuffix("-1") for b in barcodes]
    with pytest.raises(ValueError, match="barcodes are not in the input"):
        bv.io.load_atac_fragments(
            tmp_path / "bad",
            FRAGS,
            peaks=FIX / "peaks.bed",
            barcodes=suffixless,
            verbose=0,
        )
    ok = bv.io.load_atac_fragments(
        tmp_path / "mapped",
        FRAGS,
        peaks=FIX / "peaks.bed",
        barcodes=barcodes,
        verbose=0,
    )
    # a barcode map reaches every fragment pass
    plain = atac.fragment_qc(ok, FRAGS, GTF)
    mapped = atac.fragment_qc(
        ok, FRAGS, GTF, barcode_map=dict(zip(barcodes, barcodes, strict=True))
    )
    assert plain.equals(mapped)
    with pytest.raises(ValueError, match="barcodes are not in the input"):
        atac.fragment_qc(ok, FRAGS, GTF, barcode_map=lambda c: c.removesuffix("-1"))

    renamed = pl.DataFrame(
        {"chrom": ["22"], "start": [16_000_000], "end": [16_001_000]}
    )
    with pytest.raises(ValueError, match="chromosome naming"):
        bv.io.load_atac_fragments(
            tmp_path / "chr", FRAGS, peaks=renamed, barcodes=barcodes, verbose=0
        )


def test_profiles_annotation_blacklist(x):
    tss, lengths = atac.fragment_profiles(x, FRAGS, GTF)
    assert tss.height == 4001 and lengths.height == 1000
    centre = tss.filter(pl.col("offset").abs() < 100)["enrichment"].mean()
    assert centre > 3
    assert abs(lengths["fraction"].sum() - 1) < 1e-9

    ann = atac.annotate_peaks(x, GTF)
    assert ann.height == x.var().height
    assert (ann["distance_to_gene"] >= 0).all()
    assert "closest_gene" in x.var().columns

    peaks = pl.read_csv(
        FIX / "peaks.bed",
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end"],
    )
    black = pl.read_csv(
        FIX / "blacklist.bed",
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end"],
    )
    kept = atac.remove_blacklisted(peaks, black)
    assert 0 < kept.height <= peaks.height


def test_merge_fragments_keeps_samples_apart(barcodes, tmp_path):
    out = tmp_path / "merged.tsv.gz"
    summary = atac.merge_fragments([FRAGS, FRAGS], out, prefixes=["s1_", "s2_"])
    assert summary["n_fragments"][0] == summary["n_fragments"][1] > 0
    assert Path(f"{out}.tbi").is_file()
    cells = [f"s1_{b}" for b in barcodes[:50]] + [f"s2_{b}" for b in barcodes[:50]]
    m = bv.io.load_atac_fragments(
        tmp_path / "m", out, peaks=FIX / "peaks.bed", barcodes=cells, verbose=0
    )
    lib = m.obs(cols=["lib_size"])["lib_size"].to_numpy()
    np.testing.assert_array_equal(lib[:50], lib[50:])


def test_peak_to_gene_runs_on_the_multiome(mm, barcodes):
    groups = pl.DataFrame(
        {"cell_idx": mm._all_cells(), "grp": [f"g{i % 20}" for i in range(500)]}
    )
    mm.add_obs(groups)
    links = atac.peak_to_gene(mm, GTF, "grp", fdr=None, verbose=0)
    assert set(links.columns) == {
        "peak_id",
        "gene_id",
        "gene_name",
        "distance",
        "r",
        "p",
        "fdr",
    }
