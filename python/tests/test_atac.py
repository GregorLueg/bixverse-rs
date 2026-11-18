"""ATAC: the store next to the RNA, QC, TF-IDF, top features, LSI and WNN.

Numbers against Signac on the 500 PBMC multiome cells (chr22) under
`crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/`, the fixtures the Rust
parity tests use; the Rust tests cover the definitions, these the wiring.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import atac, mm
from bixverse.params import MinQualityParams, SyntheticDataParams

FIX = Path(__file__).parents[2] / "crates/bixverse-atac/tests/fixtures/pbmc3k_chr22"
QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)


def _tsv(rel: str) -> pl.DataFrame:
    return pl.read_csv(FIX / rel, separator="\t", null_values=["NaN"])


@pytest.fixture(scope="module")
def barcodes():
    return (FIX / "cells.txt").read_text().split()


@pytest.fixture
def exp(barcodes, tmp_path):
    """An experiment whose RNA is synthetic, keyed by the fixture barcodes."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=len(barcodes), n_genes=60, n_batches=1), seed=3
    )
    obs = data.obs.with_columns(pl.Series("cell_id", barcodes))
    bv.io.load_csr(tmp_path / "exp", data.counts, obs, data.var, qc=QC)
    exp = bv.SingleCellsMultiModal(tmp_path / "exp")
    assert exp.n_cells == len(barcodes)
    return exp


@pytest.fixture
def with_peaks(exp):
    atac.add_atac_fragments(
        exp, FIX / "fragments.tsv.gz", peaks=FIX / "peaks.bed", mode="fragment"
    )
    return exp


def test_fragments_into_peaks_match_signac_library_size(with_peaks):
    obs = with_peaks.obs(cols=["cell_id", "atac_lib_size"])
    ref = _tsv("signac/cell_metrics.tsv")
    assert obs["cell_id"].to_list() == ref["barcode"].to_list()
    assert obs["atac_lib_size"].to_list() == ref["nCount_peaks"].to_list()
    var = with_peaks.var(modality="atac")
    assert var.height == 1883
    assert var["feature_id"][0].startswith("chr22:")


def test_fragment_qc_matches_signac(with_peaks):
    qc = atac.atac_fragment_qc(
        with_peaks, FIX / "fragments.tsv.gz", FIX / "genes.gtf.gz"
    )
    ref = _tsv("signac/cell_metrics.tsv")
    assert qc["atac_n_fragments"].to_list() == ref["frequency_count"].to_list()
    np.testing.assert_allclose(
        qc["atac_tss_enrichment"].to_numpy(),
        ref["tss_enrichment_fast"].to_numpy(),
        atol=1e-9,
    )
    frip = qc["atac_frip"].to_numpy()
    want = ref["frip"].to_numpy()
    ok = ~np.isnan(want)
    np.testing.assert_allclose(frip[ok], want[ok], atol=1e-12)


def test_lsi_matches_signac_and_feeds_wnn(with_peaks):
    exp = with_peaks
    atac.run_tfidf(exp)
    top = atac.find_top_features(exp, min_cutoff="q5")
    ref_top = _tsv("signac/top_features.tsv")
    assert top.tolist() == np.flatnonzero(ref_top["kept"].to_numpy() == 1).tolist()

    depth = atac.calculate_lsi(exp, n_comp=10, drop_first=False, solver="exact")
    ref = pl.read_csv(FIX / "signac/lsi_embeddings.tsv.gz", separator="\t")
    emb = exp.atac_cache.embeddings["lsi"]
    assert emb.shape == (500, 10)
    for j in range(10):
        r = np.corrcoef(emb[:, j], ref[f"LSI_{j + 1}"].to_numpy())[0, 1]
        assert abs(r) > 0.999, (j, r)
    ref_depth = _tsv("signac/lsi_components.tsv")["depth_cor"].to_numpy()
    np.testing.assert_allclose(np.abs(depth), np.abs(ref_depth), atol=1e-4)

    atac.calculate_lsi(exp, n_comp=10)
    assert exp.atac_cache.embeddings["lsi"].shape == (500, 9)
    assert atac.get_lsi_depth_cor(exp)["dropped"].to_list()[:2] == [True, False]
    assert exp.check_state(["lsi"], modality="atac") is True

    bv.pp.find_hvg(exp, hvg_no=40)
    bv.pp.calculate_pca(exp, 10)
    mm.generate_wnn_graph(exp, modality_2="atac", embd_to_use_2="lsi")
    assert exp.wnn is not None
    assert exp.wnn.weights.height == 500


def test_lsi_needs_tfidf(with_peaks):
    with pytest.raises(ValueError, match="run_tfidf"):
        atac.calculate_lsi(with_peaks, n_comp=5)


def test_atac_needs_a_store(exp):
    with pytest.raises(ValueError, match="No ATAC"):
        atac.run_tfidf(exp)


def test_tiles_and_cutoffs(exp):
    atac.add_atac_fragments(
        exp,
        FIX / "fragments.tsv.gz",
        tile_width=5000,
        chrom_sizes={"chr22": 50_818_468},
        mode="fragment",
    )
    assert exp.atac is not None and exp.atac.kind == "tile"
    assert exp.var(modality="atac").height == 10164
    with pytest.raises(ValueError, match="exactly one"):
        atac.add_atac_fragments(exp, FIX / "fragments.tsv.gz")
    with pytest.raises(ValueError, match="min_cutoff"):
        atac.find_top_features(exp, min_cutoff="x5")


def test_gene_activity_matches_signac(exp):
    ga = atac.gene_activity(exp, FIX / "fragments.tsv.gz", FIX / "genes.gtf.gz")
    assert ga.feature_kind == "gene_activity"
    assert exp.feature_kind == "gene"
    assert ga.cell_ids().tolist() == exp.cell_ids().tolist()

    names = _tsv("signac/gene_activity/features.tsv.gz")["name"].to_list()
    ref = _tsv("signac/gene_activity/matrix.tsv.gz")
    want = {
        (names[f], c): v
        for f, c, v in ref.select("feature_idx", "cell_idx", "value").iter_rows()
    }
    gene_ids = ga.var()["gene_id"].to_list()
    assert sorted(gene_ids) == sorted(names)
    m = bv.get_counts(ga, layer="raw").scipy_csr().tocoo()
    got = {
        (gene_ids[g], int(c)): int(v)
        for c, g, v in zip(m.row, m.col, m.data, strict=True)
    }
    assert got == want
    np.testing.assert_array_equal(
        ga.obs()["lib_size"].to_numpy(), np.asarray(m.tocsr().sum(axis=1)).ravel()
    )


def test_rna_methods_refuse_or_warn_on_gene_activity(exp):
    from bixverse._core.single_cells import check_rna_counts

    ga = atac.gene_activity(exp, FIX / "fragments.tsv.gz", FIX / "genes.gtf.gz")
    with pytest.raises(ValueError, match="needs RNA counts"):
        bv.doublets.scrublet(ga)
    with pytest.warns(UserWarning, match="holds gene activity"):
        check_rna_counts(ga, "aucell", refuse=False)
    check_rna_counts(exp, "scrublet", refuse=True)
