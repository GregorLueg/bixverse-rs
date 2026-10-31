"""kNN, neighbour graph and clustering parity with the bixverse R package.

R loads seeded synthetic data into two directories, runs its HVG and PCA on
one and from that PCA the kNN searches, graphs and clusterings
(`dev/sc_graph_fixtures.R`), leaving the other untouched. Python copies the
untouched one, applies the same cell filter, takes R's PCA scores as its
embedding and has to reproduce R's results. Skipped when Rscript or the
package is missing.

Tolerances. Both sides run the same Rust numerics on the same f32 embedding,
so indices, graph structure and labels are compared exactly. Distances and
edge weights are f32 printed by jsonlite at 15 significant digits; they are
compared at `rtol=1e-6`, `atol=1e-7`, a few f32 ulps. The exact methods
(`exhaustive`, `kmknn`) must give identical indices. The approximate ones are
seeded, but the HNSW build races benignly, so they are held to a recall of
`MIN_RECALL` against R's own result (measured: 1.0 for every method and
metric). Leiden and Louvain depend on the seed only, so labels are exact.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import bixverse as bv
from bixverse import graph
from bixverse.params import FastClusterParams, NeighboursParams

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_graph_fixtures.R"

RTOL_F32 = 1e-6
ATOL_F32 = 1e-7
MIN_RECALL = 0.95
EXACT = ("exhaustive", "kmknn")
APPROX = ("annoy", "hnsw", "nndescent", "ivf")


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        [
            "Rscript",
            "-e",
            'quit(status = !requireNamespace("bixverse", quietly = TRUE))',
        ],
        capture_output=True,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _r_available(), reason="needs Rscript and bixverse")


def _run_r(*args: str) -> None:
    res = subprocess.run(
        ["Rscript", str(SCRIPT), *args], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the kNN, graph and clustering steps."""
    root = tmp_path_factory.mktemp("r_graph")
    _run_r("write", str(root), str(root / "reference.json"))
    return json.loads((root / "reference.json").read_text())


@pytest.fixture
def sc(ref, tmp_path):
    """A fresh copy of R's untouched directory, R's cell filter and PCA."""
    directory = tmp_path / "py_graph"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    return sc


###########
# Helpers #
###########


def _same_knn(got: bv.KnnGraph, want: dict, *, exact: bool = True) -> None:
    assert got.dist_metric == want["metric"]
    wi = np.array(want["indices"], dtype=np.uint32)
    assert got.indices.shape == wi.shape
    if exact:
        np.testing.assert_array_equal(got.indices, wi)
        np.testing.assert_allclose(
            got.distances, np.array(want["dist"]), rtol=RTOL_F32, atol=ATOL_F32
        )
    else:
        recall = np.mean(
            [
                len(set(a) & set(b)) / len(a)
                for a, b in zip(got.indices, wi, strict=True)
            ]
        )
        assert recall >= MIN_RECALL, recall


def _same_graph(got: bv.CsrMatrix, want: dict) -> None:
    assert got.shape == (want["n"], want["n"])
    np.testing.assert_array_equal(got.indptr, want["indptr"])
    np.testing.assert_array_equal(got.indices, want["indices"])
    np.testing.assert_allclose(
        got.data, np.array(want["data"]), rtol=RTOL_F32, atol=ATOL_F32
    )
    # every undirected edge stored twice, self-loops once
    n_loops = int(
        np.sum(np.repeat(np.arange(got.shape[0]), np.diff(got.indptr)) == got.indices)
    )
    assert (len(got.data) + n_loops) // 2 == want["n_edges"]


def _same_fast(got: graph.FastClusters, want: dict) -> None:
    wm = want["memberships"]
    assert got.memberships.columns == list(wm)
    assert [c + 1 for c in got.memberships["cell_idx"].to_list()] == wm["cell_idx"]
    for col in got.memberships.columns[1:]:
        assert got.memberships[col].to_list() == wm[col], col
    if want["stats"] is None:
        assert got.stats is None
    else:
        assert got.stats is not None
        assert got.stats.columns == list(want["stats"])
        for col, values in want["stats"].items():
            np.testing.assert_allclose(
                got.stats[col].to_numpy(), values, rtol=RTOL_F32, err_msg=col
            )
    if want["k_means_cluster"] is not None:
        assert got.k_means_cluster is not None
        assert got.k_means_cluster.tolist() == want["k_means_cluster"]
        assert got.centroids is not None
        np.testing.assert_allclose(
            got.centroids, np.array(want["centroids"]), rtol=1e-5, atol=1e-6
        )


#######
# kNN #
#######


@pytest.mark.parametrize("dist", ["euclidean", "cosine"])
@pytest.mark.parametrize("method", EXACT + APPROX)
def test_generate_knn_matches_r(ref, sc, method, dist):
    knn = graph.generate_knn(
        sc,
        neighbours_params=NeighboursParams(
            knn_method=method, ann_dist=dist, k=ref["k"]
        ),
        seed=42,
        validate_index=False,
    )
    assert knn is not None
    _same_knn(knn, ref["knn"][f"{method}_{dist}"], exact=method in EXACT)
    assert sc.cache.knn is None


def test_generate_knn_cell_subset_matches_r(ref, sc):
    knn = graph.generate_knn(
        sc,
        cells_to_use=ref["subset_cells"],
        neighbours_params=NeighboursParams(knn_method="exhaustive"),
        validate_index=False,
    )
    assert knn is not None
    _same_knn(knn, ref["knn_subset"])


def test_calc_knn_metrics_matches_r(ref):
    want = ref["metrics"]
    exact = ref["knn"]["exhaustive_euclidean"]
    query = want["query"]

    def as_knn(d: dict) -> bv.KnnGraph:
        return bv.KnnGraph(
            indices=np.array(d["indices"], dtype=np.uint32),
            distances=np.array(d["dist"], dtype=np.float32),
            dist_metric=d["metric"],
        )

    res = graph.calc_knn_metrics(as_knn(exact), as_knn(query))
    assert res.all_matches.tolist() == want["all_matches"]
    np.testing.assert_allclose(res.all_ratios, want["all_ratios"], rtol=RTOL_F32)
    np.testing.assert_allclose(res.final_recall, want["final_recall"], rtol=1e-12)
    np.testing.assert_allclose(res.final_ratio, want["final_ratio"], rtol=RTOL_F32)


##########
# Graphs #
##########

GRAPH_RUNS = {
    "snn_jaccard": {"knn_sym": "snn", "snn_similarity": "jaccard"},
    "snn_rank": {"knn_sym": "snn", "snn_similarity": "rank", "pruning": 0.1},
    "snn_no_rescue": {"knn_sym": "snn", "rescue": False, "pruning": 0.2},
    "umap": {"knn_sym": "umap"},
    "umap_mixed": {"knn_sym": "umap", "local_connectivity": 1.5, "mix_ratio": 0.5},
    "knn": {"knn_sym": "knn"},
    "knn_binary": {"knn_sym": "knn", "binary": True},
}


@pytest.mark.parametrize("run", list(GRAPH_RUNS))
def test_find_neighbours_matches_r(ref, sc, run):
    graph.find_neighbours(sc, neighbours_params=NeighboursParams(**GRAPH_RUNS[run]))
    g = graph.get_connectivities(sc)
    assert g is not None
    _same_graph(g, ref["graphs"][run])
    status = sc.cache_status()
    assert status["artefact"].to_list() == ["pca", "knn", "connectivities"]
    assert status["stale"].to_list() == [False, False, False]
    ids = {a: s.id for a, s in sc.cache.stamps.items()}
    assert status["parents"].to_list() == [[], [ids["pca"]], [ids["knn"]]]


def test_find_neighbours_default_matches_r(ref, sc):
    graph.find_neighbours(sc, no_embd_to_use=10)
    assert sc.cache.knn is not None
    _same_knn(sc.cache.knn, ref["knn_default"])
    assert sc.cache.connectivities is not None
    _same_graph(sc.cache.connectivities, ref["graph_default"])


############
# Clusters #
############

CLUSTER_RUNS = {
    "leiden": {"cluster_algorithm": "leiden", "res": 1.0, "seed": 42},
    "leiden_hi": {"cluster_algorithm": "leiden", "res": 2.0, "seed": 7},
    "louvain": {"cluster_algorithm": "louvain", "res": 1.0, "seed": 42},
    "leiden_stable": {
        "cluster_algorithm": "leiden",
        "res": 0.5,
        "seed": 3,
        "n_iterations": -1,
        "beta": 0.05,
    },
}


def test_find_clusters_matches_r(ref, sc):
    graph.find_neighbours(sc, no_embd_to_use=10)
    for run, kwargs in CLUSTER_RUNS.items():
        graph.find_clusters(sc, name=run, **kwargs)
    obs = sc.obs(filtered=True)
    for run in CLUSTER_RUNS:
        assert obs[run].to_list() == ref["clusters"][run], run
    assert ref["clusters_class"] == "integer"


def test_r_reads_python_clusters(ref, sc, tmp_path):
    """R opens the Python directory and sees the labels as integers."""
    graph.find_neighbours(sc, no_embd_to_use=10)
    graph.find_clusters(sc, name="leiden")
    out = tmp_path / "out.json"
    _run_r("read", str(sc.dir), str(out))
    got = json.loads(out.read_text())
    assert got["class"]["leiden"] == "integer"
    kept = {c + 1 for c in sc.cells_to_keep.tolist()}
    labels = [
        v
        for idx, v in zip(got["obs"]["cell_idx"], got["obs"]["leiden"], strict=True)
        if idx in kept
    ]
    assert labels == ref["clusters"]["leiden"]


###################
# Fast clustering #
###################


def test_fast_cluster_single_matches_r(ref, sc):
    res = graph.fast_cluster(
        sc,
        resolutions=[1.0, 0.5],
        fc_params=FastClusterParams(n_centroids=30),
        return_kmeans=True,
    )
    _same_fast(res, ref["fast"]["single"])


def test_fast_cluster_seeds_matches_r(ref, sc):
    res = graph.fast_cluster(
        sc,
        resolutions=[2.0, 1.0],
        fc_params=FastClusterParams(
            n_centroids=40, algorithm="leiden", km_type="minibatch", batch_size=128
        ),
        n_seeds=3,
        seed=11,
    )
    _same_fast(res, ref["fast"]["seeds"])


def test_fast_cluster_cell_mediated_matches_r(ref, sc):
    res = graph.fast_cluster(
        sc,
        no_embd_to_use=8,
        resolutions=[1.0],
        fc_params=FastClusterParams(cell_mediated=True, dist="cosine", knn_sym="umap"),
    )
    _same_fast(res, ref["fast"]["cell_mediated"])


def test_fast_cluster_umap_matches_r(ref, sc):
    res = graph.fast_cluster(
        sc,
        resolutions=[1.5],
        fc_params=FastClusterParams(n_centroids=50, centroid_k=10, knn_sym="umap"),
    )
    _same_fast(res, ref["fast"]["umap"])
