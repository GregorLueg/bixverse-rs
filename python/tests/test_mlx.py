"""MLX twins (`backend="mlx"`) against their CPU twins, which carry the R
parity.

The parity tests skip unless the `bixverse-gpu` build carries the MLX twins
(`maturin develop --release --features mlx` on Apple Silicon with mlx-c), so
CI only runs the error path. The tolerances are those of the wgpu tests in
`test_gpu.py`; they have not been measured on MLX yet.
"""

import numpy as np
import pytest

import bixverse as bv
from bixverse import graph, integrate, metacells, pp
from bixverse.params import (
    HarmonyV2Params,
    MinQualityParams,
    NeighboursParams,
    PcaParams,
    SeacellsParams,
    SyntheticDataParams,
)

bg = pytest.importorskip("bixverse_gpu")

from bixverse_gpu.params import (  # noqa: E402
    FastClusterGpuParams,
    HarmonyV2GpuParams,
    KnnGpuDefaults,
)

mlx = pytest.mark.skipif(
    not bg.mlx_available(), reason="bixverse-gpu built without the mlx feature"
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
BATCH = "batch_index"
EXACT = PcaParams(svd_solver="exact")


@pytest.fixture(scope="module")
def data():
    """Six samples in two batches."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(
            n_cells=600, n_genes=80, n_samples=6, sample_bias="even", n_batches=2
        ),
        seed=5,
    )
    return data, data.obs


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment with HVGs and an exact CPU PCA."""
    data, obs = data
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, obs, data.var, qc=QC)
    pp.find_hvg(sc, hvg_no=50)
    pp.calculate_pca(sc, 10, pca_params=EXACT)
    return sc


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


##############
# Error path #
##############


@pytest.mark.skipif(bg.mlx_available(), reason="the MLX twins are compiled in")
def test_mlx_backend_without_the_feature_raises(sc):
    with pytest.raises(bv.BixverseError, match="--features mlx"):
        bg.pp.calculate_pca(sc, 10, backend="mlx")
    assert not hasattr(bg._bixverse_gpu, "pca_mlx")


def test_mlx_available_is_a_bool():
    assert isinstance(bg.mlx_available(), bool)


##########
# Parity #
##########


@mlx
def test_lanczos_pca_matches_cpu(sc):
    want = (sc.cache.pca_factors, sc.cache.pca_loadings, sc.cache.pca_singular_vals)
    bg.pp.calculate_pca(sc, 10, pca_params=EXACT, backend="mlx")
    np.testing.assert_allclose(sc.cache.pca_singular_vals, want[2], rtol=1e-5)
    assert _rel(sc.cache.pca_factors, want[0]) < 1e-4
    assert _rel(sc.cache.pca_loadings, want[1]) < 1e-4
    assert "pca" in sc.cache.stamps


@mlx
def test_exhaustive_knn_matches_cpu(sc):
    want = graph.generate_knn(
        sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
    )
    got = bg.graph.generate_knn(
        sc, knn_params=KnnGpuDefaults(knn_method="exhaustive"), backend="mlx"
    )
    assert want is not None and got is not None
    assert graph.calc_knn_metrics(want, got).final_recall == 1.0
    np.testing.assert_allclose(
        np.sort(got.distances, axis=1), np.sort(want.distances, axis=1), atol=1e-5
    )


@mlx
def test_neighbours_write_the_cache(sc):
    bg.graph.find_neighbours(
        sc, knn_params=KnnGpuDefaults(knn_method="exhaustive"), backend="mlx"
    )
    assert sc.cache.knn is not None
    assert sc.cache.connectivities is not None
    assert "connectivities" in sc.cache.stamps


@mlx
def test_fast_cluster_structure(sc):
    got = bg.graph.fast_cluster(
        sc,
        fc_params=FastClusterGpuParams(n_centroids=40),
        return_kmeans=True,
        backend="mlx",
    )
    assert got.memberships["cell_idx"].to_list() == sc.cells_to_keep.tolist()
    assert got.centroids is not None and got.k_means_cluster is not None
    assert got.centroids.shape == (40, 10)
    embd = sc.cache.pca_factors
    assert embd is not None
    d = ((embd[:, None, :] - got.centroids[None, :, :]) ** 2).sum(-1)
    assert (d.argmin(1) == got.k_means_cluster).mean() > 0.99
    for col in got.memberships.columns[1:]:
        assert got.memberships[col].min() == 0


@mlx
def test_harmony_v2_tracks_cpu(sc):
    integrate.harmony(sc, BATCH, harmony_params=HarmonyV2Params(k=10))
    bg.integrate.harmony(
        sc,
        BATCH,
        harmony_params=HarmonyV2GpuParams(k=10, use_dynamic_lambda=True),
        backend="mlx",
    )
    want = sc.cache.embeddings["harmony"]
    got = sc.cache.embeddings["harmony_gpu"]
    assert got.shape == want.shape
    corr = [np.corrcoef(got[:, i], want[:, i])[0, 1] for i in range(got.shape[1])]
    assert min(corr) >= 0.95
    assert sc.cache.stamps["harmony_gpu"].parents == (sc.cache.stamps["pca"].id,)


@mlx
def test_seacells_matches_cpu(sc):
    graph.find_neighbours(
        sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
    )
    params = SeacellsParams(n_sea_cells=20)
    want = metacells.generate_seacells(sc, params)
    got = bg.metacells.generate_seacells(sc, params, backend="mlx")
    assert got.n_cells == want.n_cells
    assert (
        got.obs()["original_cell_idx"].to_list()
        == want.obs()["original_cell_idx"].to_list()
    )
    np.testing.assert_allclose(
        np.asarray(got.other_data["rss"]),
        np.asarray(want.other_data["rss"]),
        rtol=1e-4,
    )
