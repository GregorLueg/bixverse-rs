"""Neighbours and clustering without R: state handling, persistence and edge
cases.

Parity with the R helpers is in `test_graph_r.py`.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import graph, pp
from bixverse.params import (
    FastClusterParams,
    KnnParams,
    MinQualityParams,
    NeighboursParams,
    SyntheticDataParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=200, n_genes=60, n_batches=1), seed=5
    )


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment with HVGs and a PCA."""
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    pp.find_hvg(sc, hvg_no=40)
    pp.calculate_pca(sc, 10)
    return sc


#######
# kNN #
#######


def test_generate_knn_leaves_the_experiment_alone(sc):
    knn = graph.generate_knn(sc, neighbours_params=KnnParams(k=7))
    assert knn is not None
    assert knn.indices.shape == (sc.n_cells, 7)
    assert knn.indices.dtype == np.uint32
    assert knn.distances.dtype == np.float32
    assert np.all(knn.indices != np.arange(sc.n_cells)[:, None])
    assert np.all(np.diff(knn.distances, axis=1) >= 0)
    assert sc.cache.knn is None


def test_generate_knn_missing_embedding_warns(sc):
    with pytest.warns(UserWarning, match="not found"):
        assert graph.generate_knn(sc, embd_to_use="umap") is None


def test_generate_knn_truncates_dims(sc):
    knn = graph.generate_knn(
        sc, no_embd_to_use=3, neighbours_params=KnnParams(knn_method="exhaustive")
    )
    assert sc.cache.pca_factors is not None
    pcs = sc.cache.pca_factors[:, :3]
    d = np.linalg.norm(pcs[:, None, :] - pcs[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert knn is not None
    assert knn.indices[:, 0].tolist() == d.argmin(axis=1).tolist()


def test_extract_knn_runs_the_descent_graph(sc):
    p = KnnParams(knn_method="nndescent", extract_knn=True, k=10)
    knn = graph.generate_knn(sc, neighbours_params=p)
    exact = graph.generate_knn(sc, neighbours_params=KnnParams(k=10))
    assert knn is not None
    assert exact is not None
    metrics = graph.calc_knn_metrics(exact, knn)
    assert metrics.final_recall > 0.8


def test_calc_knn_metrics_of_itself(sc):
    knn = graph.generate_knn(sc)
    assert knn is not None
    res = graph.calc_knn_metrics(knn, knn)
    assert res.final_recall == 1.0
    assert res.final_ratio == pytest.approx(1.0)
    assert res.all_matches.tolist() == [knn.indices.shape[1]] * sc.n_cells


def test_calc_knn_metrics_rejects_mismatched_k(sc):
    a = graph.generate_knn(sc, neighbours_params=KnnParams(k=5))
    b = graph.generate_knn(sc, neighbours_params=KnnParams(k=6))
    assert a is not None
    assert b is not None
    with pytest.raises(ValueError, match="neighbours"):
        graph.calc_knn_metrics(a, b)


##############
# Neighbours #
##############


def test_find_neighbours_writes_and_stamps(sc):
    graph.find_neighbours(sc, neighbours_params=NeighboursParams(k=8))
    knn, g = sc.cache.knn, sc.cache.connectivities
    assert knn is not None
    assert g is not None
    assert g.shape == (sc.n_cells, sc.n_cells)
    dense = g.scipy_csr().toarray()  # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(dense, dense.T)
    assert sc.cache.stamps["knn"].parents == (sc.cache.stamps["pca"].id,)
    assert sc.cache.stamps["connectivities"].parents == (sc.cache.stamps["knn"].id,)


def test_find_neighbours_missing_embedding_warns(sc):
    with pytest.warns(UserWarning, match="not found"):
        graph.find_neighbours(sc, embd_to_use="umap")
    assert sc.cache.knn is None


def test_filter_change_makes_the_graph_stale(sc):
    graph.find_neighbours(sc)
    with pytest.warns(UserWarning, match="stale"):
        sc.set_cells_to_keep(sc.cells_to_keep[10:])
    stale = sc.cache_status().filter("stale")["artefact"].to_list()
    assert stale == ["pca", "knn", "connectivities"]
    with pytest.raises(ValueError, match="stale"):
        graph.find_neighbours(sc)
    with pytest.raises(ValueError, match="stale"):
        graph.find_clusters(sc)


def test_recomputed_pca_makes_the_graph_stale(sc):
    graph.find_neighbours(sc)
    pp.calculate_pca(sc, 8)
    status = sc.cache_status()
    assert status.filter(status["artefact"] == "connectivities")["stale"].item()
    with pytest.raises(ValueError, match="stale"):
        graph.find_clusters(sc)


def test_knn_and_graph_survive_save_and_load(sc):
    graph.find_neighbours(sc)
    bv.io.save(sc)
    back = bv.io.load_existing(sc.dir)
    a, b = sc.cache.knn, back.cache.knn
    assert a is not None
    assert b is not None
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.distances, b.distances)
    assert a.dist_metric == b.dist_metric
    g, h = sc.cache.connectivities, back.cache.connectivities
    assert g is not None
    assert h is not None
    assert g.shape == h.shape
    for part in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(g, part), getattr(h, part))
    assert back.cache.stamps == sc.cache.stamps
    assert back.cache_status()["stale"].to_list() == [False, False, False]

    graph.find_clusters(back, name="after_load")
    graph.find_clusters(sc, name="before_save")
    obs = sc.obs()
    assert obs["after_load"].to_list() == obs["before_save"].to_list()


##############
# Clustering #
##############


def test_find_clusters_writes_obs(sc):
    graph.find_neighbours(sc)
    graph.find_clusters(sc, name="clusters")
    col = sc.obs()["clusters"]
    assert col.dtype == pl.Int32
    counts = np.bincount(col.to_numpy())
    assert counts.min() > 0
    assert np.all(np.diff(counts) <= 0)  # 0 is the largest


def test_find_clusters_only_labels_kept_cells(sc):
    with pytest.warns(UserWarning):
        sc.set_cells_to_keep(sc.cells_to_keep[20:])
    pp.calculate_pca(sc, 10)
    graph.find_neighbours(sc)
    graph.find_clusters(sc, cluster_algorithm="louvain", name="lv")
    obs = sc.obs()
    assert obs["lv"][:20].is_null().all()
    assert not obs["lv"][20:].is_null().any()


def test_find_clusters_is_seeded(sc):
    graph.find_neighbours(sc)
    graph.find_clusters(sc, name="a", seed=1)
    graph.find_clusters(sc, name="b", seed=1)
    obs = sc.obs()
    assert obs["a"].to_list() == obs["b"].to_list()


def test_find_clusters_without_graph_warns(sc):
    with pytest.warns(UserWarning, match="No neighbour graph"):
        graph.find_clusters(sc)
    assert "leiden_clustering" not in sc.obs().columns


def test_find_clusters_rejects_a_graph_of_other_cells(sc):
    graph.find_neighbours(sc)
    g = sc.cache.connectivities
    assert g is not None
    n = g.shape[0] - 1
    sc.cache.connectivities = bv.CsrMatrix(
        g.data[: g.indptr[n]], g.indices[: g.indptr[n]], g.indptr[: n + 1], (n, n)
    )
    with pytest.raises(ValueError, match="covers"):
        graph.find_clusters(sc)


def test_find_clusters_rejects_zero_iterations(sc):
    graph.find_neighbours(sc)
    with pytest.raises(ValueError, match="n_iterations"):
        graph.find_clusters(sc, n_iterations=0)


def test_get_connectivities_without_graph_warns(sc):
    with pytest.warns(UserWarning, match="No neighbour graph"):
        assert graph.get_connectivities(sc) is None


###################
# Fast clustering #
###################


def test_fast_cluster_shapes(sc):
    res = graph.fast_cluster(
        sc,
        resolutions=[1.0, 0.25],
        fc_params=FastClusterParams(n_centroids=20),
        return_kmeans=True,
    )
    assert res.memberships.columns == ["cell_idx", "res_1", "res_0.25"]
    assert res.memberships["cell_idx"].to_list() == sc.cells_to_keep.tolist()
    assert res.stats is None
    assert res.k_means_cluster is not None
    assert res.k_means_cluster.shape == (sc.n_cells,)
    assert res.centroids is not None
    assert res.centroids.shape == (20, 10)
    assert "res_1" not in sc.obs().columns


def test_fast_cluster_seed_stats(sc):
    res = graph.fast_cluster(sc, resolutions=[1.0], n_seeds=3)
    assert res.stats is not None
    assert res.stats.columns == [
        "resolution",
        "mean_ari",
        "median_ari",
        "mean_conductance",
        "median_conductance",
        "mean_n_comms",
    ]
    assert res.k_means_cluster is None


def test_fast_cluster_missing_embedding_raises(sc):
    with pytest.raises(ValueError, match="not found"):
        graph.fast_cluster(sc, embd_to_use="umap")


def test_no_warnings_on_the_happy_path(sc):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph.find_neighbours(sc)
        graph.find_clusters(sc)
