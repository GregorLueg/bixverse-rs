"""Batch correction and integration metrics without R: state handling,
persistence and edge cases.

Parity with the R functions is in `test_integrate_r.py`.
"""

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import graph, integrate, pp
from bixverse.params import (
    BbknnParams,
    FastmnnParams,
    HarmonyV2Params,
    MinQualityParams,
    SeuratRpcaParams,
    SyntheticDataParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
BATCH = "batch_index"


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment with two batches."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=300, n_genes=80, n_batches=2), seed=5
    )


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment with HVGs and a PCA."""
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    pp.find_hvg(sc, hvg_no=50)
    pp.calculate_pca(sc, 10)
    return sc


@pytest.fixture
def one_batch(sc):
    """The experiment with a constant batch column."""
    sc.add_obs(pl.DataFrame({"cell_idx": sc.cells_to_keep, "single": [1] * sc.n_cells}))
    return sc


###############
# Corrections #
###############


def test_harmony_stamps_from_pca(sc):
    integrate.harmony(sc, BATCH)
    embd = sc.cache.embeddings["harmony"]
    assert embd.shape == (sc.n_cells, 10)
    assert embd.dtype == np.float32
    assert sc.cache.stamps["harmony"].parents == (sc.cache.stamps["pca"].id,)
    # the default k is min(round(n / 30), 100): same result when given
    auto = embd.copy()
    integrate.harmony(
        sc, BATCH, harmony_params=HarmonyV2Params(k=round(sc.n_cells / 30))
    )
    np.testing.assert_array_equal(sc.cache.embeddings["harmony"], auto)


def test_harmony_feeds_find_neighbours(sc):
    integrate.harmony_v1(sc, BATCH)
    graph.find_neighbours(sc, embd_to_use="harmony_v1")
    ids = {a: s.id for a, s in sc.cache.stamps.items()}
    assert sc.cache.stamps["knn"].parents == (ids["harmony_v1"],)
    pp.calculate_pca(sc, 10)
    stale = sc.cache_status().filter(pl.col("stale"))["artefact"].to_list()
    assert stale == ["harmony_v1", "knn", "connectivities"]


def test_harmony_refuses_stale_pca(sc):
    with pytest.warns(UserWarning, match="stale"):
        sc.set_cells_to_keep(sc.cells_to_keep[10:])
    with pytest.raises(ValueError, match="stale"):
        integrate.harmony(sc, BATCH)


def test_harmony_without_pca_warns(sc):
    sc.cache.pca_factors = None
    with pytest.warns(UserWarning, match="No PCA"):
        integrate.harmony(sc, BATCH)
    assert "harmony" not in sc.cache.embeddings


@pytest.mark.parametrize(
    "run",
    [
        lambda sc: integrate.harmony(sc, "nope"),
        lambda sc: integrate.harmony(sc, BATCH, additional_batch_columns=["nope"]),
        lambda sc: integrate.bbknn(sc, "nope"),
        lambda sc: integrate.fast_mnn(sc, "nope", sc.map.hvg),
        lambda sc: integrate.seurat_cca(sc, "nope", sc.map.hvg),
        lambda sc: integrate.calculate_kbet(sc, "nope"),
        lambda sc: integrate.calculate_pcr(sc, "nope"),
    ],
)
def test_missing_batch_column_raises(sc, run):
    with pytest.raises(ValueError, match="not found"):
        run(sc)


def test_single_batch_leaves_the_experiment_alone(one_batch):
    sc = one_batch
    with pytest.warns(UserWarning, match="only has one batch"):
        integrate.bbknn(sc, "single")
    with pytest.warns(UserWarning, match="only has one batch"):
        integrate.seurat_rpca(sc, "single", sc.map.hvg)
    assert sc.cache.knn is None
    assert "rpca" not in sc.cache.embeddings


def test_single_batch_fast_mnn_raises(one_batch):
    with pytest.raises(bv.BixverseError):
        integrate.fast_mnn(one_batch, "single", one_batch.map.hvg)


def test_single_batch_metrics_return_none(one_batch):
    graph.find_neighbours(one_batch)
    for run in (
        integrate.calculate_kbet,
        integrate.calculate_lisi,
        integrate.calculate_batch_asw,
        integrate.calculate_pcr,
        integrate.calculate_cell_type_asw,
    ):
        with pytest.warns(UserWarning, match="only has one"):
            assert run(one_batch, "single") is None


def test_bbknn_writes_knn_and_graph(sc):
    integrate.bbknn(
        sc, BATCH, no_neighbours_to_keep=4, bbknn_params=BbknnParams(trim=20)
    )
    knn, g = sc.cache.knn, sc.cache.connectivities
    assert knn is not None and g is not None
    assert knn.indices.shape == (sc.n_cells, 4)
    assert knn.dist_metric == "euclidean"
    dense = g.scipy_csr().toarray()
    np.testing.assert_array_equal(dense, dense.T)
    ids = {a: s.id for a, s in sc.cache.stamps.items()}
    assert sc.cache.stamps["knn"].parents == (ids["pca"],)
    assert sc.cache.stamps["connectivities"].parents == (ids["knn"],)
    graph.find_clusters(sc, name="bbknn_leiden")
    assert sc.obs(filtered=True)["bbknn_leiden"].null_count() == 0


def test_bbknn_caps_neighbours_and_warns_on_prior_knn(sc):
    graph.find_neighbours(sc)
    with pytest.warns(UserWarning) as record:
        integrate.bbknn(sc, BATCH, no_neighbours_to_keep=50)
    messages = " ".join(str(w.message) for w in record)
    assert "Prior kNN" in messages
    assert "too few generated neighbours" in messages
    assert sc.cache.knn is not None
    # two batches x 3 neighbours within each
    assert sc.cache.knn.indices.shape[1] == 6


def test_fast_mnn_stamps_follow_the_pca_choice(sc):
    params = FastmnnParams(no_pcs=8, k=10)
    integrate.fast_mnn(sc, BATCH, sc.map.hvg, fastmnn_params=params)
    assert sc.cache.embeddings["mnn"].shape == (sc.n_cells, 8)
    assert sc.cache.stamps["mnn"].parents == ()
    integrate.fast_mnn(
        sc, BATCH, sc.map.hvg, fastmnn_params=params, use_precomputed_pca=True
    )
    assert sc.cache.embeddings["mnn"].shape == (sc.n_cells, 10)
    assert sc.cache.stamps["mnn"].parents == (sc.cache.stamps["pca"].id,)


def test_seurat_rpca_runs_with_the_randomised_solver(sc):
    integrate.seurat_rpca(
        sc,
        BATCH,
        sc.gene_ids(sc.map.hvg).tolist(),
        rpca_params=SeuratRpcaParams(dims=5, k_weight=20, svd_solver="randomised"),
    )
    assert sc.cache.embeddings["rpca"].shape == (sc.n_cells, 5)


def test_corrections_survive_save_and_load(sc):
    integrate.harmony(sc, BATCH)
    integrate.bbknn(sc, BATCH)
    bv.io.save(sc)
    back = bv.io.load_existing(sc.dir)
    np.testing.assert_array_equal(
        back.cache.embeddings["harmony"], sc.cache.embeddings["harmony"]
    )
    assert back.cache.knn is not None and sc.cache.knn is not None
    np.testing.assert_array_equal(back.cache.knn.indices, sc.cache.knn.indices)
    assert back.cache.stamps == sc.cache.stamps
    assert not back.cache_status()["stale"].any()


###########
# Metrics #
###########


def test_metrics_without_knn_warn(sc):
    with pytest.warns(UserWarning, match="No kNN"):
        assert integrate.calculate_kbet(sc, BATCH) is None
    with pytest.warns(UserWarning, match="No kNN"):
        assert integrate.calculate_graph_connectivity(sc, "cell_grp") is None


def test_metric_missing_embedding_warns(sc):
    with pytest.warns(UserWarning, match="not found"):
        assert integrate.calculate_batch_asw(sc, BATCH, embd_to_use="mnn") is None
    with pytest.warns(UserWarning, match="not found"):
        assert integrate.calculate_pcr(sc, BATCH, embd_to_use="mnn") is None


def test_metrics_warn_on_stale_knn(sc):
    graph.find_neighbours(sc)
    pp.calculate_pca(sc, 10)
    with pytest.warns(UserWarning, match="stale"):
        integrate.calculate_kbet(sc, BATCH)


def test_metrics_check_ranges(sc):
    with pytest.raises(ValueError, match="threshold"):
        integrate.calculate_kbet(sc, BATCH, threshold=1.5)
    with pytest.raises(ValueError, match="perplexity"):
        integrate.calculate_lisi(sc, BATCH, perplexity=0.0)


def test_kbet_score_is_the_rejection_rate(sc):
    graph.find_neighbours(sc)
    res = integrate.calculate_kbet(sc, BATCH, threshold=0.5)
    assert res is not None
    assert res.kbet_score == pytest.approx(np.mean(res.p_values <= 0.5))
    assert res.n_batches == 2


def test_pcr_on_pca_has_no_comparison(sc):
    res = integrate.calculate_pcr(sc, BATCH)
    assert res is not None
    assert res.pcr_comparison is None
    assert res.pcr == res.pcr_pca
    integrate.harmony(sc, BATCH)
    res = integrate.calculate_pcr(sc, BATCH, embd_to_use="harmony")
    assert res is not None
    assert res.pcr_comparison == pytest.approx((res.pcr_pca - res.pcr) / res.pcr_pca)


def test_integration_metrics_one_row(sc):
    integrate.harmony(sc, BATCH)
    graph.find_neighbours(sc, embd_to_use="harmony")
    res = integrate.calculate_integration_metrics(
        sc, BATCH, cell_type_column="cell_grp", embd_to_use="harmony"
    )
    assert res.shape == (1, 8)
    assert res["embedding"].item() == "harmony"
    assert res.null_count().sum_horizontal().item() == 0
    bare = integrate.calculate_integration_metrics(sc, BATCH, embd_to_use=None)
    assert bare.select("batch_asw", "pcr_comparison", "clisi").null_count().row(0) == (
        1,
        1,
        1,
    )


######################
# Review regressions #
######################


@pytest.mark.parametrize(
    "params", [HarmonyV2Params(theta=(2.0, 2.0)), HarmonyV2Params(sigma=(0.1, 0.2))]
)
def test_harmony_wrong_param_length_is_an_error_not_a_panic(sc, params):
    # theta is per batch variable, sigma per cluster: both used to assert
    with pytest.raises(bv.BixverseError, match="must be 1 or"):
        integrate.harmony(sc, BATCH, harmony_params=params)


def test_graph_connectivity_with_one_cell_type_is_none(one_batch):
    graph.find_neighbours(one_batch)
    with pytest.warns(UserWarning, match="only has one cell type"):
        assert integrate.calculate_graph_connectivity(one_batch, "single") is None


def test_fortran_ordered_knn_reads_the_same(sc):
    graph.find_neighbours(sc)
    knn = sc.cache.knn
    lisi = integrate.calculate_lisi(sc, BATCH)
    assert knn is not None and lisi is not None
    want = lisi.per_cell
    sc.cache.knn = bv.KnnGraph(
        indices=np.asfortranarray(knn.indices),
        distances=np.asfortranarray(knn.distances),
        dist_metric=knn.dist_metric,
    )
    sc.stamp("knn", parents=["pca"])
    assert sc.cache.knn.indices.flags.c_contiguous
    lisi = integrate.calculate_lisi(sc, BATCH)
    assert lisi is not None
    np.testing.assert_array_equal(lisi.per_cell, want)


def test_bindings_refuse_fortran_order(sc):
    from bixverse import _bixverse

    graph.find_neighbours(sc)
    idx = np.asfortranarray(sc.cache.knn.indices)
    codes = np.zeros(sc.n_cells, dtype=np.uint32) + 1
    with pytest.raises(ValueError, match="C-contiguous"):
        _bixverse.graph_connectivity(idx, codes)


########################
# Independent oracles #
########################


def _ilisi(sc) -> float:
    lisi = integrate.calculate_lisi(sc, BATCH, verbose=False)
    assert lisi is not None
    return lisi.mean_lisi


def test_corrections_mix_planted_batches(sc):
    # strong planted batch effect: every method must mix the two batches
    # better than the uncorrected PCA does (iLISI, 1 = separate, 2 = mixed)
    graph.find_neighbours(sc)
    base = _ilisi(sc)
    assert base < 1.5

    integrate.harmony(sc, BATCH, verbose=False)
    graph.find_neighbours(sc, embd_to_use="harmony")
    harmony = _ilisi(sc)

    # on the cached PCA: fastMNN's own 30-PC PCA of 50 HVGs shrinks the
    # corrected batch and re-separates it under euclidean neighbours
    integrate.fast_mnn(sc, BATCH, sc.map.hvg, use_precomputed_pca=True, verbose=False)
    graph.find_neighbours(sc, embd_to_use="mnn")
    mnn = _ilisi(sc)

    integrate.bbknn(sc, BATCH, no_neighbours_to_keep=15, verbose=False)
    bbknn = _ilisi(sc)

    for name, v in (("harmony", harmony), ("mnn", mnn), ("bbknn", bbknn)):
        assert v > base + 0.2, (name, v, base)


@pytest.fixture
def toy(sc):
    """Planted 10-D embeddings: one with the batches mixed, one with them
    30 standard deviations apart."""
    rng = np.random.default_rng(2)
    batch = sc.obs(cols=[BATCH], filtered=True)[BATCH].to_numpy().astype(np.int64)
    noise = rng.normal(size=(sc.n_cells, 10)).astype(np.float32)
    sc.cache.embeddings["mixed"] = noise
    sc.stamp("mixed")
    shifted = noise.copy()
    shifted[:, 0] += 30.0 * (batch - batch.min())
    sc.cache.embeddings["apart"] = shifted
    sc.stamp("apart")
    return sc


def test_batch_metrics_on_planted_embeddings(toy):
    out = {}
    for name in ("mixed", "apart"):
        graph.find_neighbours(toy, embd_to_use=name)
        kbet = integrate.calculate_kbet(toy, BATCH, verbose=False)
        asw = integrate.calculate_batch_asw(toy, BATCH, embd_to_use=name, verbose=False)
        assert kbet is not None and asw is not None
        out[name] = (_ilisi(toy), kbet.kbet_score, asw.mean_asw)
    (lisi_m, kbet_m, asw_m), (lisi_a, kbet_a, asw_a) = out["mixed"], out["apart"]
    # two balanced batches: iLISI near 2 when mixed, 1 when apart
    assert lisi_m > 1.8 and lisi_a < 1.05
    # kBET rejection share: near the 5 % level when mixed, all when apart
    assert kbet_m < 0.2 and kbet_a > 0.95
    # raw silhouette by batch: near 0 when mixed, near 1 when apart
    assert abs(asw_m) < 0.1 and asw_a > 0.8
