"""Batch correction and integration metric parity with the bixverse R package.

R loads seeded synthetic data with three batches into two directories, runs
its HVG and PCA on one and from there every correction and metric
(`dev/sc_integrate_fixtures.R`), leaving the other untouched. Python copies
the untouched one, applies the same cell filter, takes R's PCA scores and
R's batch-aware HVGs, and has to reproduce R's results. Skipped when Rscript
or the package is missing.

Tolerances. Both sides run the same Rust numerics on the same inputs, so the
results agree to the 15 significant digits jsonlite prints (measured: at most
5e-15 absolute, 4e-15 relative, on every embedding, kNN distance, graph
weight and metric). Embeddings are f32 and compared at `rtol=1e-6` with an
absolute floor of `1e-7` times the largest entry; f64 metrics at
`rtol=1e-12`. kNN indices, graph structure and kBET decisions are exact.
"""

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

import bixverse as bv
from bixverse import graph, integrate
from bixverse.params import (
    BbknnParams,
    FastmnnParams,
    HarmonyParams,
    HarmonyV2Params,
    SeuratCcaParams,
    SeuratRpcaParams,
)

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_integrate_fixtures.R"

RTOL_F32 = 1e-6
ATOL_F32 = 1e-7
RTOL_F64 = 1e-12
BATCH = "batch_index"
CCA = SeuratCcaParams(num_cc=10, dims=10, k_filter=50, k_weight=50, n_top_features=50)
RPCA = SeuratRpcaParams(dims=10, k_weight=50)
MNN = FastmnnParams(no_pcs=15, k=10)
# R's harmony_sc is still v1, so its "harmony" is our "harmony_v1"
R_HARMONY = "harmony_v1"


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the corrections and metrics."""
    root = tmp_path_factory.mktemp("r_integrate")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), "write", str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture
def sc(ref, tmp_path):
    """A fresh copy of R's untouched directory, R's cell filter and PCA."""
    directory = tmp_path / "py_integrate"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    return sc


@pytest.fixture
def sc_corrected(ref, sc):
    """The experiment with R's metric inputs: Harmony, fastMNN and the
    default neighbours on the PCA."""
    integrate.harmony_v1(sc, BATCH)
    integrate.fast_mnn(sc, BATCH, ref["batch_hvg"], fastmnn_params=MNN)
    graph.find_neighbours(sc)
    return sc


###########
# Helpers #
###########


def _same_embedding(got: np.ndarray, want: list) -> None:
    w = np.array(want)
    assert got.dtype == np.float32
    assert got.shape == w.shape
    np.testing.assert_allclose(got, w, rtol=RTOL_F32, atol=ATOL_F32 * np.abs(w).max())


def _same_bbknn(sc: bv.SingleCells, want: dict) -> None:
    knn, g = sc.cache.knn, sc.cache.connectivities
    assert knn is not None and g is not None
    assert knn.dist_metric == want["knn"]["metric"]
    np.testing.assert_array_equal(knn.indices, want["knn"]["indices"])
    np.testing.assert_allclose(
        knn.distances, want["knn"]["dist"], rtol=RTOL_F32, atol=ATOL_F32
    )
    wg = want["graph"]
    assert g.shape == (wg["n"], wg["n"])
    np.testing.assert_array_equal(g.indptr, wg["indptr"])
    np.testing.assert_array_equal(g.indices, wg["indices"])
    np.testing.assert_allclose(g.data, wg["data"], rtol=RTOL_F32, atol=ATOL_F32)
    n_loops = int(
        np.sum(np.repeat(np.arange(g.shape[0]), np.diff(g.indptr)) == g.indices)
    )
    assert (len(g.data) + n_loops) // 2 == wg["n_edges"]


def _close(got: object, want: object, rtol: float = RTOL_F64) -> None:
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, rtol=rtol)


###########
# Harmony #
###########


def test_harmony_v1_matches_r(ref, sc):
    integrate.harmony_v1(sc, BATCH, seed=42)
    _same_embedding(sc.cache.embeddings["harmony_v1"], ref["harmony"])
    status = sc.cache_status()
    assert status["artefact"].to_list() == ["pca", "harmony_v1"]
    assert status["parents"].to_list()[1] == [sc.cache.stamps["pca"].id]


def test_harmony_v1_multi_covariate_matches_r(ref, sc):
    integrate.harmony_v1(
        sc,
        BATCH,
        additional_batch_columns=["sample_id"],
        harmony_params=HarmonyParams(k=12, theta=(2.0, 1.0), k_means_init="random"),
        seed=7,
    )
    _same_embedding(sc.cache.embeddings["harmony_v1"], ref["harmony_multi"])


def test_harmony_v2_matches_r(ref, sc):
    integrate.harmony(sc, BATCH, seed=42)
    _same_embedding(sc.cache.embeddings["harmony"], ref["harmony_v2"])


def test_harmony_v2_fixed_lambda_matches_r(ref, sc):
    integrate.harmony(
        sc,
        BATCH,
        harmony_params=HarmonyV2Params(
            use_dynamic_lambda=False, tau=5.0, lambda_=(0.5,)
        ),
        seed=3,
    )
    _same_embedding(sc.cache.embeddings["harmony"], ref["harmony_v2_fixed"])


###########
# Anchors #
###########


def test_fast_mnn_matches_r(ref, sc):
    integrate.fast_mnn(sc, BATCH, ref["batch_hvg"], fastmnn_params=MNN)
    _same_embedding(sc.cache.embeddings["mnn"], ref["mnn"])
    assert sc.cache.stamps["mnn"].parents == ()


def test_fast_mnn_precomputed_matches_r(ref, sc):
    integrate.fast_mnn(
        sc,
        BATCH,
        ref["batch_hvg"],
        fastmnn_params=FastmnnParams(
            no_pcs=15, k=10, cos_norm=False, ann_dist="euclidean"
        ),
        use_precomputed_pca=True,
    )
    _same_embedding(sc.cache.embeddings["mnn"], ref["mnn_pre"])
    assert sc.cache.stamps["mnn"].parents == (sc.cache.stamps["pca"].id,)


def test_seurat_cca_matches_r(ref, sc):
    integrate.seurat_cca(sc, BATCH, ref["batch_hvg"], cca_params=CCA)
    _same_embedding(sc.cache.embeddings["cca"], ref["cca"])


def test_seurat_rpca_matches_r(ref, sc):
    integrate.seurat_rpca(sc, BATCH, ref["batch_hvg"], rpca_params=RPCA)
    _same_embedding(sc.cache.embeddings["rpca"], ref["rpca"])


def test_seurat_rpca_precomputed_matches_r(ref, sc):
    integrate.seurat_rpca(
        sc, BATCH, ref["batch_hvg"], rpca_params=RPCA, use_precomputed_pca=True
    )
    _same_embedding(sc.cache.embeddings["rpca"], ref["rpca_pre"])


#########
# BBKNN #
#########


def test_bbknn_matches_r(ref, sc):
    integrate.bbknn(sc, BATCH, no_neighbours_to_keep=5)
    _same_bbknn(sc, ref["bbknn"])
    status = sc.cache_status()
    assert status["artefact"].to_list() == ["pca", "knn", "connectivities"]
    assert status["stale"].to_list() == [False, False, False]


def test_bbknn_custom_matches_r(ref, sc):
    integrate.bbknn(
        sc,
        BATCH,
        no_neighbours_to_keep=8,
        no_embd_to_use=10,
        bbknn_params=BbknnParams(
            neighbours_within_batch=4,
            set_op_mix_ratio=0.5,
            trim=15,
            knn_method="exhaustive",
            ann_dist="cosine",
        ),
    )
    _same_bbknn(sc, ref["bbknn_custom"])


###########
# Metrics #
###########


def test_metric_knn_matches_r(ref, sc_corrected):
    knn = sc_corrected.cache.knn
    assert knn is not None
    np.testing.assert_array_equal(knn.indices, ref["knn"]["indices"])


@pytest.mark.parametrize(
    ("key", "column", "threshold"),
    [("kbet", BATCH, 0.05), ("kbet_strict", "sample_id", 0.2)],
)
def test_kbet_matches_r(ref, sc_corrected, key, column, threshold):
    want = ref[key]
    got = integrate.calculate_kbet(sc_corrected, column, threshold=threshold)
    assert got is not None
    assert got.significant_tests.tolist() == want["significant_tests"]
    _close(got.kbet_score, want["kbet_score"])
    _close(got.p_values, want["p_values"])
    _close(got.chi_square_stats, want["chi_square_stats"])
    _close(got.mean_chi_square, want["mean_chi_square"])
    _close(got.median_chi_square, want["median_chi_square"])
    assert got.n_batches == want["n_batches"]


@pytest.mark.parametrize(
    ("key", "column", "kwargs"),
    [
        ("lisi_batch", BATCH, {}),
        ("lisi_weighted", BATCH, {"weighted": True, "perplexity": 10.0}),
        ("lisi_cell_type", "cell_grp", {"type": "cell_type"}),
    ],
)
def test_lisi_matches_r(ref, sc_corrected, key, column, kwargs):
    want = ref[key]
    got = integrate.calculate_lisi(sc_corrected, column, **kwargs)
    assert got is not None
    _close(got.per_cell, want["per_cell"], RTOL_F32)
    _close(got.mean_lisi, want["mean_lisi"], RTOL_F32)
    _close(got.median_lisi, want["median_lisi"], RTOL_F32)
    _close(got.lisi_norm, want["lisi_norm"], RTOL_F32)
    assert got.n_labels == want["n_labels"]
    assert got.type == want["type"]


@pytest.mark.parametrize(
    ("key", "kwargs"),
    [
        ("batch_asw", {}),
        ("batch_asw_sub", {"embd_to_use": R_HARMONY, "max_cells": 300, "seed": 11}),
    ],
)
def test_batch_asw_matches_r(ref, sc_corrected, key, kwargs):
    want = ref[key]
    got = integrate.calculate_batch_asw(sc_corrected, BATCH, **kwargs)
    assert got is not None
    _close(got.per_cell, want["per_cell"], RTOL_F32)
    _close(got.mean_asw, want["mean_asw"], RTOL_F32)
    _close(got.median_asw, want["median_asw"], RTOL_F32)
    assert got.n_batches == want["n_batches"]
    assert got.embedding_used == want["embedding_used"].replace("harmony", R_HARMONY)


@pytest.mark.parametrize(
    ("key", "embd"), [("pcr_pca", "pca"), ("pcr_harmony", R_HARMONY)]
)
def test_pcr_matches_r(ref, sc_corrected, key, embd):
    want = ref[key]
    got = integrate.calculate_pcr(sc_corrected, BATCH, embd_to_use=embd)
    assert got is not None
    _close(got.pcr, want["pcr"])
    _close(got.pcr_pca, want["pcr_pca"])
    _close(got.var_explained, want["var_explained"])
    _close(got.r_squared, want["r_squared"])
    if want["pcr_comparison"] is None:
        assert got.pcr_comparison is None
    else:
        _close(got.pcr_comparison, want["pcr_comparison"])


def test_cell_type_asw_matches_r(ref, sc_corrected):
    want = ref["cell_type_asw"]
    got = integrate.calculate_cell_type_asw(
        sc_corrected, "cell_grp", embd_to_use="mnn", max_cells=None
    )
    assert got is not None
    _close(got.per_cell, want["per_cell"], RTOL_F32)
    _close(got.mean_asw, want["mean_asw"], RTOL_F32)
    assert got.n_cell_types == want["n_cell_types"]


def test_graph_connectivity_matches_r(ref, sc_corrected):
    want = ref["connectivity"]
    got = integrate.calculate_graph_connectivity(sc_corrected, "cell_grp")
    assert got is not None
    assert got.per_cell_type["cell_type"].to_list() == want["names"]
    _close(got.per_cell_type["connectivity"].to_numpy(), want["values"], RTOL_F32)
    _close(got.mean_connectivity, want["mean"], RTOL_F32)
    _close(got.median_connectivity, want["median"], RTOL_F32)


@pytest.mark.parametrize(
    ("key", "kwargs"),
    [
        ("summary", {"cell_type_column": "cell_grp", "embd_to_use": R_HARMONY}),
        ("summary_no_embedding", {"embd_to_use": None}),
    ],
)
def test_integration_metrics_match_r(ref, sc_corrected, key, kwargs):
    want = ref[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = integrate.calculate_integration_metrics(sc_corrected, BATCH, **kwargs)
    assert got.columns == list(want)
    row = got.row(0, named=True)
    want_embd = want["embedding"]
    assert row["embedding"] == (want_embd and want_embd.replace("harmony", R_HARMONY))
    for col in got.columns[1:]:
        if want[col] is None:
            assert row[col] is None, col
        else:
            np.testing.assert_allclose(row[col], want[col], rtol=RTOL_F32, err_msg=col)
