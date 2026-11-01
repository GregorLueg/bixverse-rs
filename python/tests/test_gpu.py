"""GPU twins against their CPU twins, which carry the R parity.

Skipped without the `bixverse-gpu` wheel or a GPU adapter; the plane-reducing
methods (Leiden, Bonsai's Sanity, NEBULA) also need plane operations.

Tolerances, measured on this data on Apple Silicon (wgpu / Metal), with
headroom for another device:

- Exact paths agree to f32 rounding: Lanczos PCA scores 2.3e-6 relative,
  exhaustive kNN distances 4.8e-7, fastMNN 6.5e-8, NMF factors
  2e-6, NEBULA log fold changes 1e-5 and subject overdispersions 2e-4
  relative. BBKNN, the neighbour graph, Leiden and SEACells came back
  identical.
- Approximate or differently ordered paths are held to what they measured:
  the randomised PCA draws a different sketch on each side, so both are held
  to the exact solve (0.5% and 0.6% on the singular values); IVF recall
  0.80, CAGRA recall 1.0, Harmony (Jacobi instead of blocked updates) column
  correlation 0.989, Scrublet score correlation 0.994,
  SCENIC importance Spearman 0.91 (different tree RNG streams), Bonsai
  log-likelihood 4e-4 relative.
- Fast clustering and the 2D embeddings start from a different k-means or
  optimiser stream, so they are checked for structure and for neighbourhood
  preservation instead.
"""

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import (
    _bixverse,
    de,
    doublets,
    embed,
    graph,
    integrate,
    metacells,
    nmf,
    pathways,
    pp,
    trees,
)
from bixverse.params import (
    BbknnParams,
    BtMetacellsParams,
    FastClusterParams,
    FastmnnParams,
    HarmonyV2Params,
    KnnDefaults,
    MinQualityParams,
    NebulaParams,
    NeighboursParams,
    PcaParams,
    ScenicGradientBoostingDefaults,
    ScenicParams,
    ScenicRandomForestDefaults,
    ScrubletParams,
    SeacellsParams,
    SyntheticDataParams,
    to_rust,
)

bg = pytest.importorskip("bixverse_gpu")

from bixverse_gpu.params import (  # noqa: E402
    BbknnGpuParams,
    FastClusterGpuParams,
    HarmonyV2GpuParams,
    KnnGpuDefaults,
    NebulaGpuParams,
    ScrubletGpuParams,
)

pytestmark = pytest.mark.skipif(not bg.gpu_available(), reason="no GPU adapter")
planes = pytest.mark.skipif(
    not bg.gpu_plane_ops(), reason="the GPU adapter does not run plane operations"
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
TRT = ("sample_4", "sample_5", "sample_6")
BATCH = "batch_index"
EXACT = PcaParams(svd_solver="exact")


@pytest.fixture(scope="module")
def data():
    """Six samples in two batches, a condition and a per-cell covariate."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(
            n_cells=600, n_genes=80, n_samples=6, sample_bias="even", n_batches=2
        ),
        seed=5,
    )
    obs = data.obs.with_columns(
        condition=pl.when(pl.col("sample_id").is_in(TRT))
        .then(pl.lit("trt"))
        .otherwise(pl.lit("ctr")),
        score=pl.Series(np.random.default_rng(1).normal(size=data.obs.height)),
    )
    return data, obs


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment with HVGs and an exact CPU PCA."""
    data, obs = data
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, obs, data.var, qc=QC)
    pp.find_hvg(sc, hvg_no=50)
    pp.calculate_pca(sc, 10, pca_params=EXACT)
    return sc


@pytest.fixture
def mc(sc):
    """Meta cells with HVGs."""
    graph.find_neighbours(sc)
    mc = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=40, k=8)
    )
    assert mc is not None
    pp.find_hvg(mc, hvg_no=40)
    return mc


###########
# Helpers #
###########


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index."""
    _, ai = np.unique(np.asarray(a), return_inverse=True)
    _, bi = np.unique(np.asarray(b), return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)

    def pairs(x: np.ndarray) -> float:
        return float((x * (x - 1) / 2).sum())

    total = pairs(table)
    sa, sb = pairs(table.sum(1)), pairs(table.sum(0))
    expected = sa * sb / (len(ai) * (len(ai) - 1) / 2)
    return (total - expected) / (0.5 * (sa + sb) - expected)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel()))
    rb = np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


def _knn_preserved(embd: np.ndarray, ref: np.ndarray, k: int = 15) -> float:
    """Share of each cell's `k` nearest neighbours in `embd` that are also
    among its reference neighbours."""
    idx, _ = _bixverse.knn_with_dist(
        np.ascontiguousarray(embd, dtype=np.float32),
        {"k": k, "knn_method": "exhaustive", "ann_dist": "euclidean"},
        False,
        42,
        False,
    )
    return float(np.mean([len(set(idx[i]) & set(ref[i])) / k for i in range(len(idx))]))


##########
# Device #
##########


def test_device_probes_are_bools():
    assert bg.gpu_available() is True
    assert isinstance(bg.gpu_plane_ops(), bool)


def test_gpu_errors_are_bixverse_errors(sc):
    p = ScenicParams(
        learner_type="grnboost2", learner_params=ScenicGradientBoostingDefaults()
    )
    cells = sc.cells_to_keep
    genes = np.arange(10, dtype=np.uint32)
    with pytest.raises(bv.BixverseError, match="no GPU path"):
        bg._bixverse_gpu.scenic_grn(
            str(sc.genes_path), cells, genes, genes[:2], to_rust(p), False
        )
    with pytest.raises(ValueError, match="not supported on the GPU"):
        bg.pathways.scenic_grn(sc, ["gene_01"], scenic_params=p)


def test_gpu_params_serialise():
    d = to_rust(HarmonyV2GpuParams())
    assert d["lambda"] == (1.0,)
    assert "lambda_" not in d
    assert to_rust(ScrubletGpuParams())["k"] == 0
    with pytest.raises(ValueError, match="KnnGpuDefaults"):
        ScrubletGpuParams(knn=KnnDefaults())
    with pytest.raises(ValueError, match="k_means_init"):
        FastClusterGpuParams(k_means_init="nope")


######
# PP #
######


@pytest.mark.parametrize("solver", ["exact", "auto"])
def test_lanczos_pca_matches_cpu(sc, solver):
    params = PcaParams(svd_solver=solver)
    pp.calculate_pca(sc, 10, pca_params=params)
    want = (sc.cache.pca_factors, sc.cache.pca_loadings, sc.cache.pca_singular_vals)
    bg.pp.calculate_pca(sc, 10, pca_params=params)
    np.testing.assert_allclose(sc.cache.pca_singular_vals, want[2], rtol=1e-5)
    # the Lanczos fixes the signs on both sides, so no flips
    assert _rel(sc.cache.pca_factors, want[0]) < 1e-4
    assert _rel(sc.cache.pca_loadings, want[1]) < 1e-4
    assert "pca" in sc.cache.stamps


def test_randomised_pca_tracks_exact(sc):
    # The two sides draw different sketches, so each is held to the exact
    # solve: measured 0.5% (CPU) and 0.6% (GPU) on the singular values.
    exact = sc.cache.pca_singular_vals
    exact_pc1 = sc.cache.pca_factors[:, 0]
    bg.pp.calculate_pca(sc, 10, pca_params=PcaParams(svd_solver="randomised"))
    np.testing.assert_allclose(sc.cache.pca_singular_vals, exact, rtol=2e-2)
    assert abs(np.corrcoef(sc.cache.pca_factors[:, 0], exact_pc1)[0, 1]) > 0.999


def test_pca_without_hvg_warns(data, tmp_path):
    data, obs = data
    sc = bv.io.load_csr(tmp_path / "bare", data.counts, obs, data.var, qc=QC)
    with pytest.warns(UserWarning, match="No HVGs"):
        bg.pp.calculate_pca(sc, 10)
    assert sc.cache.pca_factors is None


#########
# Graph #
#########


def test_exhaustive_knn_matches_cpu(sc):
    want = graph.generate_knn(
        sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
    )
    got = bg.graph.generate_knn(sc, knn_params=KnnGpuDefaults(knn_method="exhaustive"))
    assert want is not None and got is not None
    assert graph.calc_knn_metrics(want, got).final_recall == 1.0
    np.testing.assert_allclose(
        np.sort(got.distances, axis=1), np.sort(want.distances, axis=1), atol=1e-5
    )


@pytest.mark.parametrize(
    ("params", "recall"),
    [
        (KnnGpuDefaults(knn_method="ivf"), 0.7),
        (KnnGpuDefaults(knn_method="nndescent"), 0.95),
        (KnnGpuDefaults(knn_method="cagra"), 0.9),
    ],
)
def test_approximate_knn_recall(sc, params, recall):
    want = graph.generate_knn(
        sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
    )
    got = bg.graph.generate_knn(sc, knn_params=params)
    assert want is not None and got is not None
    assert got.indices.shape == want.indices.shape
    assert graph.calc_knn_metrics(want, got).final_recall >= recall


def test_neighbours_default_to_nndescent(sc):
    # the GPU descent is not deterministic run to run, so no exact comparison
    exact = KnnGpuDefaults(knn_method="exhaustive")
    want = bg.graph.generate_knn(sc, knn_params=exact)
    got = bg.graph.generate_knn(sc)
    assert want is not None and got is not None
    assert graph.calc_knn_metrics(want, got).final_recall >= 0.95


def test_neighbours_and_clusters_match_cpu(sc):
    graph.find_neighbours(
        sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
    )
    want = sc.cache.connectivities
    graph.find_clusters(sc, name="cpu")
    bg.graph.find_neighbours(sc, knn_params=KnnGpuDefaults(knn_method="exhaustive"))
    got = sc.cache.connectivities
    assert got is not None and want is not None
    np.testing.assert_array_equal(got.indptr, want.indptr)
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_allclose(got.data, want.data, rtol=1e-5)
    stamps = sc.cache.stamps
    assert stamps["knn"].parents == (stamps["pca"].id,)
    assert stamps["connectivities"].parents == (stamps["knn"].id,)
    if bg.gpu_plane_ops():
        bg.graph.find_clusters(sc, name="gpu")
        obs = sc.obs(filtered=True)
        assert obs["gpu"].dtype == pl.Int32
        assert _ari(obs["cpu"].to_numpy(), obs["gpu"].to_numpy()) >= 0.9


def test_fast_cluster_structure(sc):
    want = graph.fast_cluster(sc, fc_params=FastClusterParams(n_centroids=40))
    got = bg.graph.fast_cluster(
        sc, fc_params=FastClusterGpuParams(n_centroids=40), return_kmeans=True
    )
    assert got.memberships.columns == want.memberships.columns
    assert got.memberships["cell_idx"].to_list() == sc.cells_to_keep.tolist()
    assert got.centroids is not None and got.k_means_cluster is not None
    assert got.centroids.shape == (40, 10)
    # every cell sits with its nearest centroid
    embd = sc.cache.pca_factors
    assert embd is not None
    d = ((embd[:, None, :] - got.centroids[None, :, :]) ** 2).sum(-1)
    assert (d.argmin(1) == got.k_means_cluster).mean() > 0.99
    for col in got.memberships.columns[1:]:
        assert got.memberships[col].min() == 0


#############
# Integrate #
#############


def test_harmony_v2_tracks_cpu(sc):
    integrate.harmony_v2(sc, BATCH, harmony_params=HarmonyV2Params(k=10))
    bg.integrate.harmony_v2(
        sc, BATCH, harmony_params=HarmonyV2GpuParams(k=10, use_dynamic_lambda=True)
    )
    want = sc.cache.embeddings["harmony_v2"]
    got = sc.cache.embeddings["harmony_gpu"]
    assert got.shape == want.shape
    corr = [np.corrcoef(got[:, i], want[:, i])[0, 1] for i in range(got.shape[1])]
    assert min(corr) >= 0.95
    assert sc.cache.stamps["harmony_gpu"].parents == (sc.cache.stamps["pca"].id,)


def test_bbknn_matches_cpu(sc):
    integrate.bbknn(sc, BATCH, bbknn_params=BbknnParams(knn_method="exhaustive"))
    want_knn, want_graph = sc.cache.knn, sc.cache.connectivities
    with pytest.warns(UserWarning, match="Prior kNN"):
        bg.integrate.bbknn(sc, BATCH, bbknn_params=BbknnGpuParams())
    got_knn, got_graph = sc.cache.knn, sc.cache.connectivities
    assert want_knn is not None and got_knn is not None
    assert want_graph is not None and got_graph is not None
    np.testing.assert_array_equal(got_knn.indices, want_knn.indices)
    np.testing.assert_allclose(got_knn.distances, want_knn.distances, atol=1e-5)
    np.testing.assert_array_equal(got_graph.indices, want_graph.indices)
    np.testing.assert_allclose(got_graph.data, want_graph.data, rtol=1e-5)


def test_fast_mnn_matches_cpu(sc):
    assert sc.map.hvg is not None
    integrate.fast_mnn(
        sc,
        BATCH,
        sc.map.hvg,
        fastmnn_params=FastmnnParams(knn_method="exhaustive"),
        use_precomputed_pca=True,
    )
    bg.integrate.fast_mnn(sc, BATCH)
    got, want = sc.cache.embeddings["mnn_gpu"], sc.cache.embeddings["mnn"]
    assert _rel(got, want) < 1e-5
    assert sc.cache.stamps["mnn_gpu"].parents == (sc.cache.stamps["pca"].id,)


############
# Doublets #
############


def test_scrublet_tracks_cpu(sc):
    want = doublets.scrublet(
        sc, scrublet_params=ScrubletParams(knn_method="exhaustive", no_pcs=10)
    )
    got = bg.doublets.scrublet(sc, scrublet_params=ScrubletGpuParams(no_pcs=10))
    assert np.array_equal(got.cell_indices, want.cell_indices)
    corr = np.corrcoef(got.doublet_scores_obs, want.doublet_scores_obs)[0, 1]
    assert corr >= 0.98
    assert (got.predicted_doublets == want.predicted_doublets).mean() >= 0.97
    cpu_knn = bg.doublets.scrublet(
        sc,
        scrublet_params=ScrubletGpuParams(
            no_pcs=10, knn_backend="cpu", knn=KnnDefaults(k=0, knn_method="exhaustive")
        ),
    )
    np.testing.assert_allclose(
        cpu_knn.doublet_scores_obs, got.doublet_scores_obs, atol=1e-6
    )


@pytest.mark.filterwarnings("ignore:Group\\(s\\) with fewer than")
def test_scrublet_group_by(sc):
    res = bg.doublets.scrublet(
        sc, scrublet_params=ScrubletGpuParams(no_pcs=10), group_by=BATCH
    )
    assert res.group_by == BATCH
    assert isinstance(res.threshold, dict) and len(res.threshold) == 2


##############
# Meta cells #
##############


def test_seacells_matches_cpu(sc):
    graph.find_neighbours(
        sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
    )
    params = SeacellsParams(n_sea_cells=20)
    want = metacells.generate_seacells(sc, params)
    got = bg.metacells.generate_seacells(sc, params)
    assert got.n_cells == want.n_cells
    assert (
        got.obs()["original_cell_idx"].to_list()
        == want.obs()["original_cell_idx"].to_list()
    )
    np.testing.assert_allclose(got.other_data["rss"], want.other_data["rss"], rtol=1e-4)


#########
# Trees #
#########


@planes
def test_bonsai_tracks_cpu(sc, mc):
    for x in (sc, mc):
        want = trees.bonsai(x)
        got = bg.trees.bonsai(x)
        np.testing.assert_array_equal(got.genes_used, want.genes_used)
        assert abs(got.loglik - want.loglik) / abs(want.loglik) < 2e-3


#######
# NMF #
#######


def test_nmf_matches_cpu(sc, mc):
    for x in (sc, mc):
        want = nmf.nmf(x, 4)
        got = bg.nmf.nmf(x, 4)
        assert _rel(got.w, want.w) < 1e-4
        assert _rel(got.h, want.h) < 1e-4
        assert got.gene_ids == want.gene_ids
    want = nmf.stabilised_nmf(sc, 3, n_runs=4)
    got = bg.nmf.stabilised_nmf(sc, 3, n_runs=4)
    np.testing.assert_allclose(got.losses, want.losses, rtol=1e-5)
    want = nmf.consensus_nmf(sc, 3, n_runs=6)
    got = bg.nmf.consensus_nmf(sc, 3, n_runs=6)
    assert abs(got.stability - want.stability) < 1e-4
    want = nmf.nmf_k_sweep(sc, [2, 3], n_runs=4)
    got = bg.nmf.nmf_k_sweep(sc, [2, 3], n_runs=4)
    np.testing.assert_allclose(
        got.frame["best_error"].to_numpy(),
        want.frame["best_error"].to_numpy(),
        rtol=1e-5,
    )


def test_nmf_rank_cap(sc):
    with pytest.raises(ValueError, match="caps the rank"):
        bg.nmf.nmf(sc, bg.nmf.NMF_GPU_MAX_RANK + 1)


############
# Pathways #
############


def test_scenic_tracks_cpu(sc, mc):
    p = ScenicParams(
        min_counts=10,
        min_cells=0.05,
        gene_batch_strategy="random",
        learner_params=ScenicRandomForestDefaults(n_trees=50, min_samples_leaf=5),
    )
    for x in (sc, mc):
        tfs = x.gene_ids().tolist()[:5]
        want = pathways.scenic_grn(x, tfs, scenic_params=p)
        got = bg.pathways.scenic_grn(x, tfs, scenic_params=p)
        assert got.gene_ids == want.gene_ids and got.tf_ids == want.tf_ids
        assert _spearman(got.importance_matrix, want.importance_matrix) >= 0.8


######
# DE #
######


@planes
def test_nebula_matches_cpu(sc):
    # a per-cell covariate, so the fits do not collapse onto the CPU's
    # zero-count tables and stage two runs on the device
    kw = {"genes_to_use": list(range(30))}
    want = de.nebula(sc, "sample_id", "~ condition + score", **kw)
    got = bg.de.nebula(sc, "sample_id", "~ condition + score", **kw)
    assert got.results["gene_id"].to_list() == want.results["gene_id"].to_list()
    for col, rtol in (("log_fc", 1e-3), ("effect_se", 1e-4), ("p_value", 1e-3)):
        np.testing.assert_allclose(
            got.results[col].to_numpy(),
            want.results[col].to_numpy(),
            rtol=rtol,
            atol=1e-6,
            err_msg=col,
        )
    np.testing.assert_allclose(
        got.results["subject_overdispersion"].to_numpy(),
        want.results["subject_overdispersion"].to_numpy(),
        rtol=1e-2,
        atol=1e-6,
    )
    assert "method='ln'" in repr(got)


def test_nebula_params_differ_only_in_reml():
    cpu = set(to_rust(NebulaParams()))
    gpu = set(to_rust(NebulaGpuParams()))
    assert cpu - gpu == {"reml"}
    assert gpu <= cpu


##############
# Embeddings #
##############


def test_umap_and_tsne_preserve_neighbours(sc):
    graph.find_neighbours(sc)
    ref = sc.cache.knn
    assert ref is not None
    embed.umap(sc)
    bg.embed.umap(sc, slot_name="umap_gpu")
    embed.tsne(sc)
    bg.embed.tsne(sc, slot_name="tsne_gpu")
    for cpu, gpu in (("umap", "umap_gpu"), ("tsne", "tsne_gpu")):
        got = sc.cache.embeddings[gpu]
        assert got.shape == (sc.n_cells, 2)
        assert np.isfinite(got).all()
        assert sc.cache.stamps[gpu].parents == (
            sc.cache.stamps["pca"].id,
            *((sc.cache.stamps["knn"].id,) if gpu == "umap_gpu" else ()),
        )
        want = _knn_preserved(sc.cache.embeddings[cpu], ref.indices)
        assert _knn_preserved(got, ref.indices) >= 0.8 * want
