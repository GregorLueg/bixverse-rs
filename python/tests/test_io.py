"""Ingestion, writers, store maintenance and downloads, without R.

Parity with the R loaders is in `test_io_r.py`; these cover the Python side
on its own: round trips through the writers and loaders, the encoding rules,
the archive, and the download cache.
"""

import urllib.request
import warnings

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

import bixverse as bv
from bixverse._core import db
from bixverse._core.ingest import _make_unique
from bixverse._core.state import CsrMatrix
from bixverse._core.writers import _permute_square, h5ad_columns
from bixverse.params import MinQualityParams, MtxIoParams, SyntheticDataParams

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=120, n_genes=40), seed=9
    )


def _dense(sc: bv.SingleCells) -> np.ndarray:
    return bv.get_counts(sc).scipy_csr().toarray()


def _truth(d) -> np.ndarray:
    return d.counts.scipy_csr().toarray()


###########
# Loaders #
###########


def test_load_csr_takes_scipy_and_keeps_every_cell(data, tmp_path):
    csr = sp.csr_matrix(_truth(data))
    sc = bv.io.load_csr(tmp_path / "sc", csr, data.obs, data.var, qc=QC)
    assert sc.shape == (120, 40)
    np.testing.assert_array_equal(_dense(sc), _truth(data))
    assert sc.obs()["cell_id"].to_list() == data.obs["cell_id"].to_list()
    lib = sc.obs(cols=["lib_size"])["lib_size"].to_numpy()
    np.testing.assert_array_equal(lib, _truth(data).sum(axis=1))


def test_load_csr_snake_cases_and_dedups_obs_names(data, tmp_path):
    obs = data.obs.rename({"cell_id": "Barcode", "cell_grp": "CellGrp"}).with_columns(
        pl.lit(1).alias("cell_grp")
    )
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, obs, data.var, qc=QC)
    cols = sc.obs().columns
    assert cols[:2] == ["cell_idx", "cell_id"]
    assert "cell_grp" in cols and "cell_grp.1" in cols


def test_load_csr_rejects_fractional_counts(data, tmp_path):
    bad = CsrMatrix(
        data.counts.data + 0.5, data.counts.indices, data.counts.indptr, (120, 40)
    )
    with pytest.raises(ValueError, match="integers"):
        bv.io.load_csr(tmp_path / "sc", bad, data.obs, data.var, qc=QC)
    with pytest.raises(ValueError, match="rows"):
        bv.io.load_csr(tmp_path / "sc", data.counts, data.obs[:5], data.var, qc=QC)


def test_qc_drops_cells_and_genes(data, tmp_path):
    strict = MinQualityParams(min_unique_genes=25, min_lib_size=1, min_cells=1)
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=strict)
    kept = (_truth(data) > 0).sum(axis=1) >= 25
    assert sc.dims[0] == int(kept.sum())
    assert sc.obs()["cell_id"].to_list() == data.obs["cell_id"].filter(kept).to_list()


def test_h5ad_round_trip_and_auto_slot(data, tmp_path):
    f = tmp_path / "x.h5ad"
    bv.io.write_h5ad_sc(f, data.counts, data.obs, data.var)
    assert bv.io.detect_raw_count_slot(f) == "X"
    meta = bv.io.read_h5ad_metadata(f)
    obs = meta["obs"]
    assert isinstance(obs, pl.DataFrame)
    assert meta["dims"] == (120, 40)
    assert obs["cell_id"].to_list() == data.obs["cell_id"].to_list()
    sc = bv.io.load_h5ad(tmp_path / "sc", f, qc=QC)
    np.testing.assert_array_equal(_dense(sc), _truth(data))
    with pytest.raises(FileExistsError):
        bv.io.write_h5ad_sc(f, data.counts, data.obs, data.var, overwrite=False)


def test_load_h5ad_without_raw_counts_raises(data, tmp_path):
    frac = CsrMatrix(
        data.counts.data * 0.37, data.counts.indices, data.counts.indptr, (120, 40)
    )
    f = tmp_path / "frac.h5ad"
    bv.io.write_h5ad_sc(f, frac, data.obs, data.var)
    assert bv.io.detect_raw_count_slot(f) is None
    with pytest.raises(ValueError, match="raw count slot"):
        bv.io.load_h5ad(tmp_path / "sc", f, qc=QC)


def test_gene_major_mtx_round_trip(data, tmp_path):
    bv.io.write_cellranger_output(
        tmp_path / "cr",
        data.counts,
        data.obs,
        data.var,
        rows="genes",
        format_type="tsv",
    )
    params = MtxIoParams(
        path_mtx=str(tmp_path / "cr" / "matrix.mtx"),
        path_obs=str(tmp_path / "cr" / "barcodes.tsv"),
        path_var=str(tmp_path / "cr" / "features.tsv"),
        cells_as_rows=False,
        has_hdr=True,
    )
    sc = bv.io.load_mtx(tmp_path / "sc", params, qc=QC)
    np.testing.assert_array_equal(_dense(sc), _truth(data))
    assert sc.var()["ensembl_id"].to_list() == data.var["ensembl_id"].to_list()


def test_tenx_v2_round_trip(data, tmp_path):
    f = tmp_path / "v2.h5"
    features = pl.DataFrame({"id": data.var["gene_id"], "name": data.var["ensembl_id"]})
    bv.io.write_tenx_h5_sc(
        f, data.counts, data.obs["cell_id"].to_list(), features, version="v2"
    )
    meta = bv.io.read_tenx_h5_metadata(f)
    assert meta["version"] == "v2"
    assert meta["feature_types"] is None
    var = meta["var"]
    assert isinstance(var, pl.DataFrame)
    assert var[".id"].to_list() == data.var["gene_id"].to_list()
    sc = bv.io.load_tenx_h5(tmp_path / "sc", f, qc=QC)
    np.testing.assert_array_equal(_dense(sc), _truth(data))


def test_merge_rejects_merged_input(data, tmp_path):
    a = bv.io.load_csr(tmp_path / "a", data.counts, data.obs, data.var, qc=QC)
    b = bv.io.load_csr(tmp_path / "b", data.counts, data.obs, data.var, qc=QC)
    m = bv.io.merge_sc_experiments(tmp_path / "m", [a, b], ["a", "b"])
    assert m.shape == (240, 40)
    np.testing.assert_array_equal(_dense(m), np.vstack([_truth(data)] * 2))
    with pytest.raises(ValueError, match="already has an exp_id"):
        bv.io.merge_sc_experiments(tmp_path / "m2", [m, a], ["m", "a"])


#####################
# Store maintenance #
#####################


def test_archive_restores_on_load_existing(data, tmp_path):
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    before = _dense(sc)
    stats = bv.io.archive_sc_exp(sc)
    assert stats["n_cells"] == 120
    assert not sc.cells_path.exists() and not sc.genes_path.exists()
    np.testing.assert_array_equal(_dense(bv.io.load_existing(tmp_path / "sc")), before)


def test_upgrade_is_a_no_op_on_v4(data, tmp_path):
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    assert bv._bixverse.file_version(sc.cells_path) == 4
    assert bv.io.upgrade_sc_store(sc) is sc
    assert not (sc.dir / "counts_cells.bin.v3.bak").exists()


###########
# Writers #
###########


def test_h5ad_column_encoding_follows_r():
    frame = pl.DataFrame(
        {
            "grp": ["b", "a", "b"],
            "uniq": ["x", None, "z"],
            "n": [1, None, 3],
            "flag": [True, False, True],
            "empty": pl.Series([None, None, None], dtype=pl.String),
        }
    )
    cols = {
        name: (kind, payload)
        for name, kind, payload in h5ad_columns(frame, factorise=True)
    }
    kind, payload = cols["grp"]
    assert isinstance(payload, tuple)
    codes, levels = payload
    assert (kind, codes.tolist(), levels) == ("categorical", [1, 0, 1], ["a", "b"])
    assert cols["uniq"] == ("string", ["x", "NA", "z"])
    n_kind, n_payload = cols["n"]
    assert isinstance(n_payload, list | np.ndarray)
    assert n_kind == "float" and np.isnan(n_payload[1])
    assert cols["flag"][0] == "bool"
    # repeated values factorise, but there is no level to write
    assert cols["empty"] == ("string", ["NA", "NA", "NA"])


def test_permute_square_matches_dense():
    rng = np.random.default_rng(0)
    dense = (rng.random((6, 6)) < 0.4) * rng.random((6, 6)).astype(np.float32)
    s = sp.csr_matrix(dense.astype(np.float32))
    m = CsrMatrix(
        s.data.astype(np.float32),
        s.indices.astype(np.uint32),
        s.indptr.astype(np.uint32),
        (6, 6),
    )
    order = np.array([3, 0, 5, 1, 4, 2])
    out = _permute_square(m, order)
    np.testing.assert_allclose(out.scipy_csr().toarray(), dense[np.ix_(order, order)])


def test_save_h5ad_skips_stale(data, tmp_path):
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    sc.set_cells_to_keep(np.array([7, 3, 50, 20], dtype=np.uint32))
    sc.cache.embeddings["old"] = np.zeros((4, 2), dtype=np.float32)
    sc.stamp("old")
    keep = np.array([7, 3, 50, 1], dtype=np.uint32)
    with pytest.warns(UserWarning, match="stale"):
        sc.set_cells_to_keep(keep)
    sc.cache.embeddings["umap"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    sc.stamp("umap")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = bv.io.save_h5ad(sc, tmp_path / "s.h5ad")
    assert any("Skipping 'old'" in str(w.message) for w in caught)
    back = bv.io.load_h5ad(tmp_path / "back", out, qc=QC)
    # gene QC on four cells drops the genes none of them express
    sub = _truth(data)[np.sort(keep)]
    np.testing.assert_array_equal(_dense(back), sub[:, sub.sum(axis=0) > 0])


#############
# Synthetic #
#############


def test_synthetic_samples_and_seed():
    p = SyntheticDataParams(n_cells=90, n_genes=30, n_samples=3, sample_bias="even")
    a = bv.io.generate_single_cell_test_data(p, seed=1)
    b = bv.io.generate_single_cell_test_data(p, seed=1)
    assert a.obs.columns == ["cell_id", "cell_grp", "batch_index", "sample_id"]
    np.testing.assert_array_equal(a.counts.data, b.counts.data)
    assert a.obs["cell_id"][0] == "cell_01"


##########
# Helpers #
##########


def test_make_unique_matches_r():
    assert _make_unique(["a", "a", "b", "a", "a.1"]) == ["a", "a.2", "b", "a.3", "a.1"]


def test_rbind_fill_unions_levels():
    one = pl.DataFrame({"g": pl.Series(["x", "y"], dtype=pl.Enum(["y", "x"]))})
    two = pl.DataFrame({"g": ["z", "x"], "n": [1, 2]})
    out = db.rbind_fill([one, two])
    assert out.columns == ["g", "n"]
    assert out["g"].dtype == pl.Enum(["y", "x", "z"])
    assert out["n"].to_list() == [None, None, 1, 2]


#############
# Downloads #
#############


def test_downloads_hit_the_cache(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError("a cached download touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    cached = {
        bv.io.download_pbmc3k: "filtered_gene_bc_matrices/hg19",
        bv.io.download_pbmc8k: "pmbc-8k",
        bv.io.download_demuxlet_pbmc: "demuxlet_PBMCs",
        bv.io.download_pbmc_batches: "pbmc_batches",
        bv.io.download_pbmc_1k_5p: "sc5p_v2_hs_PBMC_1k_raw_feature_bc_matrix.h5",
        bv.io.download_cd34_data: "cd34_multiome_rna.h5ad",
        bv.io.download_dialogue_uc: "dialogue_uc.h5ad",
        bv.io.download_marrow_cd34: "marrow_sample_scseq_counts.h5ad",
        bv.io.download_pbmc_totalseq_data: "10k_Human_PBMC_TotalSeqB.h5",
        bv.io.download_kang_pbmc: "kang18_8vs8.h5ad",
        bv.io.download_thymus_ageing: "thymus_ageing_droplet.h5ad",
    }
    for fn, rel in cached.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"cached")
        assert fn(tmp_path) == target
