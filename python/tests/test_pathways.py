"""Pathway activity, HotSpot and SCENIC without R: structure, guards, the
hand-written `ScenicParams` and the object plumbing on synthetic data.

Parity with the R package is in `test_pathways_r.py`.
"""

import json
import shutil
import subprocess
from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import bixverse as bv
from bixverse import metacells, pathways, pp
from bixverse._core.pathways import bh_adjust, lm_padj_per_obs
from bixverse.params import (
    AucellParams,
    BtMetacellsParams,
    HotspotParams,
    LmActivityParams,
    MinQualityParams,
    NeighboursParams,
    ScenicExtraTreesDefaults,
    ScenicGradientBoostingDefaults,
    ScenicParams,
    ScenicRandomForestDefaults,
    SyntheticDataParams,
    VisionParams,
    to_rust,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
GS = {
    "type_1": [f"gene_{i:02d}" for i in range(1, 11)],
    "type_2": [f"gene_{i:02d}" for i in range(11, 21)],
}
VGS = {
    "type_1": {"pos": GS["type_1"]},
    "type_1_vs_2": {"pos": GS["type_1"], "neg": GS["type_2"]},
}
HS = HotspotParams(k=10, knn_method="exhaustive")


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    """A small synthetic experiment with PCA and an exhaustive kNN."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=300, n_genes=60), seed=11
    )
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("pw") / "sc", data.counts, data.obs, data.var, qc=QC
    )
    sc.set_cells_to_keep(sc.cell_ids()[10:])
    pp.find_hvg(sc, hvg_no=30)
    pp.calculate_pca(sc, 8)
    bv.graph.find_neighbours(sc, neighbours_params=NeighboursParams(k=10))
    return sc


@pytest.fixture(scope="module")
def mc(sc):
    mc = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=30, k=8)
    )
    assert mc is not None
    pp.find_hvg(mc, hvg_no=30)
    pp.calculate_pca(mc, 5)
    bv.graph.find_neighbours(mc, neighbours_params=NeighboursParams(k=6))
    return mc


################
# ScenicParams #
################


@pytest.mark.parametrize(
    ("learner", "block"),
    [
        ("randomforest", ScenicRandomForestDefaults),
        ("extratrees", ScenicExtraTreesDefaults),
        ("grnboost2", ScenicGradientBoostingDefaults),
    ],
)
def test_scenic_params_flatten_top_level_then_learner(learner, block):
    p = ScenicParams(learner_type=learner)
    assert isinstance(p.learner, block)
    flat = to_rust(p)
    top = [f.name for f in fields(ScenicParams) if f.name != "learner_params"]
    assert list(flat)[: len(top)] == top
    assert list(flat)[len(top) :] == [f.name for f in fields(block)]
    assert all(flat[k] == getattr(p.learner, k) for k in list(flat)[len(top) :])


def test_scenic_params_reject_a_mismatched_learner_block():
    with pytest.raises(ValueError, match="needs ScenicExtraTreesDefaults"):
        ScenicParams(
            learner_type="extratrees", learner_params=ScenicRandomForestDefaults()
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learner_params": ScenicRandomForestDefaults(subsample_rate=0.0)},
        {"learner_params": ScenicRandomForestDefaults(max_depth=0)},
        {"learner_params": ScenicRandomForestDefaults(min_samples_leaf=0)},
        {"learner_params": ScenicRandomForestDefaults(subsample_frac=1.5)},
        {
            "learner_type": "extratrees",
            "learner_params": ScenicExtraTreesDefaults(n_thresholds=0),
        },
        {
            "learner_type": "grnboost2",
            "learner_params": ScenicGradientBoostingDefaults(learning_rate=2.0),
        },
        {"min_cells": 0.0},
        {"gene_batch_size": 0},
    ],
)
def test_scenic_params_port_the_r_checks(kwargs):
    with pytest.raises(ValueError):
        ScenicParams(**kwargs)


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


@pytest.mark.skipif(not _r_available(), reason="needs Rscript and bixverse")
@pytest.mark.parametrize(
    ("r_call", "py"),
    [
        ("params_scenic()", ScenicParams()),
        (
            'params_scenic(learner_type = "extratrees", '
            "learner_params = list(n_trees = 20L))",
            ScenicParams(
                learner_type="extratrees",
                learner_params=ScenicExtraTreesDefaults(n_trees=20),
            ),
        ),
        (
            'params_scenic(min_counts = 10L, learner_type = "grnboost2", '
            'gene_batch_strategy = "random", gene_batch_size = 8L)',
            ScenicParams(
                min_counts=10,
                learner_type="grnboost2",
                gene_batch_strategy="random",
                gene_batch_size=8,
            ),
        ),
    ],
)
def test_scenic_params_flatten_to_r_list(r_call, py):
    out = subprocess.run(
        [
            "Rscript",
            "-e",
            "cat(jsonlite::toJSON(bixverse::"
            + r_call
            + ', auto_unbox = TRUE, null = "null", digits = NA))',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert to_rust(py) == json.loads(out.stdout)
    assert list(to_rust(py)) == list(json.loads(out.stdout))


################
# Serde guards #
################


def test_bad_param_strings_are_errors_not_fallbacks(sc):
    keep = np.asarray(sc.cells_to_keep, dtype=np.uint32)
    with pytest.raises(ValueError, match="binarisation"):
        bv._bixverse.regulon_thresholds(np.zeros((3, 1)), {"n_grid": "x"})
    with pytest.raises(ValueError, match="auc_type"):
        bv._bixverse.aucell(
            str(sc.cells_path),
            [np.array([0], dtype=np.uint32)],
            keep,
            {"auc_type": "x"},
            False,
        )
    with pytest.raises(ValueError, match="learner_type"):
        bv._bixverse.scenic_filter(str(sc.genes_path), keep, {"learner_type": "x"})
    with pytest.raises(ValueError, match="method"):
        bv._bixverse.lm_activity(
            str(sc.cells_path),
            [(np.array([0, 1], dtype=np.uint32), np.ones(2))],
            keep,
            {"method": "x"},
        )
    embd = np.zeros((keep.size, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="model"):
        bv._bixverse.hotspot_autocor(
            str(sc.genes_path),
            str(sc.cells_path),
            embd,
            None,
            None,
            {"model": "x"},
            keep,
            np.array([0], dtype=np.uint32),
            False,
        )


###########
# Helpers #
###########


def test_bh_adjust_matches_the_sorted_definition():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=25)
    order = np.argsort(p)
    n = p.size
    naive = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, p[order[i]] * n / (i + 1))
        naive[order[i]] = running
    np.testing.assert_allclose(bh_adjust(p), naive, rtol=0, atol=0)
    m = rng.uniform(size=(4, 6))
    np.testing.assert_allclose(lm_padj_per_obs(m)[2], bh_adjust(m[2]))


def test_generate_null_perm_gs_matches_sizes_and_balance():
    genes = [f"g{i}" for i in range(200)]
    random, clusters = pathways.generate_null_perm_gs(VGS, genes, n_perm=7, seed=3)
    assert clusters.tolist() == [0, 1]
    assert [len(c) for c in random] == [7, 7]
    assert all(len(s["pos"]) == 10 and not s["neg"] for s in random[0])
    assert all(len(s["pos"]) == 10 and len(s["neg"]) == 10 for s in random[1])
    again, _ = pathways.generate_null_perm_gs(VGS, genes, n_perm=7, seed=3)
    assert again == random


def test_generate_null_perm_gs_clusters_many_signatures():
    sigs = {f"s{i}": {"pos": [f"g{j}" for j in range(5 + i)]} for i in range(12)}
    random, clusters = pathways.generate_null_perm_gs(
        sigs, [f"g{i}" for i in range(100)], n_perm=3, n_comp=4
    )
    assert len(random) == 4
    assert set(clusters.tolist()) == {0, 1, 2, 3}


def test_prep_lm_network_drops_and_warns():
    net = pl.DataFrame(
        {
            "source": ["a", "a", "a", "b", "b", "a"],
            "target": ["x", "y", "nope", "x", "y", "x"],
        }
    )
    with pytest.warns(UserWarning, match="duplicated"):
        res = pathways.prep_lm_network(net, ["x", "y", "z"], 2)
    assert res is not None
    assert list(res) == ["a", "b"]
    assert res["a"][0].tolist() == [0, 1]
    assert res["a"][1].tolist() == [1.0, 1.0]
    with pytest.warns(UserWarning, match="tmin"):
        assert (
            pathways.prep_lm_network(net.unique(maintain_order=True), ["x", "y"], 5)
            is None
        )


###########
# Results #
###########


def test_scores_and_get_data(sc):
    res = pathways.aucell(sc, GS)
    assert res.shape == (sc.n_cells, 2)
    assert res.cell_ids.tolist() == sc.cell_ids(filtered=True).tolist()
    obs = pathways.get_data(res, columns=["type_2"])
    assert obs.columns == ["type_2", "cell_idx"]
    sc.add_obs(obs.rename({"type_2": "auc_type_2"}))
    assert "auc_type_2" in sc.obs_columns()
    with pytest.raises(ValueError, match="Unknown gene set"):
        pathways.get_data(res, columns=["nope"])
    assert "ScMatrixRes" in repr(res)


def test_module_scores_vision_and_lm_shapes(sc):
    assert pathways.module_scores(sc, GS, n_ctrl=5).shape == (sc.n_cells, 2)
    assert pathways.vision(sc, VGS).shape == (sc.n_cells, 2)
    net = pl.DataFrame({"source": ["a"] * 6, "target": GS["type_1"][:6]})
    res = pathways.lm_activity(sc, net, lm_params=LmActivityParams(p_values=True))
    assert isinstance(res, pathways.LmActivityRes)
    assert (res.fdr.values >= res.pvals.values - 1e-15).all()


def test_subset_scores_equal_the_parent_rows(sc):
    sub = bv.SingleCellsSubset(sc, "cell_grp", "cell_type_1")
    parent = pathways.aucell(sc, GS, aucell_params=AucellParams(auc_type="wilcox"))
    got = pathways.aucell(sub, GS, aucell_params=AucellParams(auc_type="wilcox"))
    rows = {c: i for i, c in enumerate(parent.cell_ids.tolist())}
    idx = [rows[c] for c in got.cell_ids.tolist()]
    np.testing.assert_array_equal(got.values, parent.values[idx])


def test_vision_w_autocor_cached_and_searched_knn(sc):
    vp = VisionParams(n_perm=20, k=10, knn_method="exhaustive")
    a = pathways.vision_w_autocor(sc, VGS, "pca", vision_params=vp)
    b = pathways.vision_w_autocor(sc, VGS, "pca", vision_params=vp, use_knn=False)
    assert a.auto_cor.columns == ["gene_set_name", "auto_cor", "p_val", "fdr"]
    np.testing.assert_array_equal(a.vision_matrix.values, b.vision_matrix.values)
    # the cached kNN (exact, k = 10) and the searched one (exhaustive, k = 10)
    # are the same graph, and only the autocorrelation reads it
    np.testing.assert_allclose(
        a.auto_cor.sort("gene_set_name")["auto_cor"].to_numpy(),
        b.auto_cor.sort("gene_set_name")["auto_cor"].to_numpy(),
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="No embedding"):
        pathways.vision_w_autocor(sc, VGS, "umap_nope")


def test_hotspot_round_trip(sc):
    genes = GS["type_1"] + GS["type_2"]
    auto = pathways.hotspot_autocor(sc, hotspot_params=HS, genes_to_take=genes)
    assert auto.columns == ["gene_id", "gearys_c", "z_score", "pval", "fdr"]
    assert sorted(auto["gene_id"].to_list()) == sorted(genes)
    streamed = pathways.hotspot_autocor(
        sc, hotspot_params=HS, genes_to_take=genes, streaming=True
    )
    np.testing.assert_allclose(
        streamed.sort("gene_id")["z_score"].to_numpy(),
        auto.sort("gene_id")["z_score"].to_numpy(),
        rtol=1e-5,
    )
    hs = pathways.hotspot_gene_cor(sc, hotspot_params=HS, genes_to_take=genes)
    assert hs.z.shape == (20, 20)
    np.testing.assert_allclose(hs.z, hs.z.T, atol=1e-6)
    assert "n_modules=None" in repr(hs)
    assert pathways.get_hotspot_membership(hs) is None
    hs = pathways.generate_hotspot_membership(hs, min_size=3)
    m = pathways.get_hotspot_membership(hs)
    assert m is not None and m.height == 20
    assert "n_modules=None" not in repr(hs)
    params = pathways.get_params(hs)
    assert isinstance(params, dict)
    assert params["used_cells"] == tuple(sc.cell_ids(filtered=True))


def test_hotspot_cell_subset_drops_the_cached_knn(sc):
    cells = sc.cell_ids(filtered=True)[::2].tolist()
    res = pathways.hotspot_autocor(
        sc, hotspot_params=HS, cells_to_take=cells, genes_to_take=GS["type_1"]
    )
    assert res.height == 10


def test_scenic_chain(sc):
    p = ScenicParams(
        min_counts=10,
        min_cells=0.05,
        gene_batch_strategy="random",
        learner_params=ScenicRandomForestDefaults(n_trees=10, min_samples_leaf=5),
    )
    with pytest.raises(ValueError, match="No provided TF"):
        pathways.scenic_grn(sc, ["not_a_gene"], scenic_params=p)
    grn = pathways.scenic_grn(sc, ["gene_01", "gene_11", "nope"], scenic_params=p)
    assert grn.importance_matrix.shape == (len(grn.gene_ids), 2)
    assert "n_tfs=2" in repr(grn)
    with pytest.warns(UserWarning, match="identify_tf_to_genes"):
        assert pathways.build_regulons(grn) is None
    grn = pathways.identify_tf_to_genes(grn, method="top_k", k_genes=5)
    assert grn.tf_to_gene_results.height == 10
    with pytest.raises(ValueError, match="cannot both be None"):
        pathways.identify_tf_to_genes(grn, method="top_k")
    grn = pathways.tf_to_genes_correlations(grn, sc, mode="both", rho_threshold=0.0)
    assert {"pairwise_cor", "cor_sign"} <= set(grn.tf_to_gene_results.columns)
    with pytest.warns(UserWarning, match="in_leading_edge"):
        reg = pathways.build_regulons(grn, min_genes=1, mode="both")
    assert reg is not None and all(k.endswith(("_pos", "_neg")) for k in reg)


def test_binarise_regulon_activity(sc):
    res = pathways.binarise_regulon_activity(pathways.aucell(sc, GS))
    assert res.binary.columns == ["cell_id", "type_1", "type_2"]
    assert res.thresholds["n_cells_on"].to_list() == [
        int(res.binary[c].sum()) for c in ("type_1", "type_2")
    ]


def test_meta_cell_twins_run(mc):
    assert pathways.aucell(mc, GS).index_col == "meta_cell_idx"
    assert pathways.vision(mc, VGS).shape == (mc.n_cells, 2)
    auto = pathways.hotspot_autocor(
        mc,
        hotspot_params=HotspotParams(k=6, knn_method="exhaustive"),
        genes_to_take=GS["type_1"],
    )
    assert auto.height == 10
    with pytest.raises(BeartypeCallHintParamViolation):
        pathways.module_scores(mc, GS)


######################
# Review regressions #
######################


def test_vision_kmeans_populates_every_cluster_with_tied_rows():
    from bixverse._core.pathways import _kmeans

    # 16 signatures of one (size, balance), as top-n marker sets are, plus 4
    x = np.array(
        [[np.log10(50), 1.0]] * 16
        + [
            [np.log10(200), 1.0],
            [np.log10(150), 0.6],
            [np.log10(100), 0.8],
            [np.log10(30), 1.0],
        ]
    )
    for seed in range(50):
        labels = _kmeans(x, 5, np.random.default_rng(seed))
        assert np.unique(labels).size == 5, seed


def test_aucell_recovery_matches_pyscenic(tmp_path):
    # dev/gen_aucell_fixture.py: pySCENIC's aucell on tie-free cells
    import scipy.sparse as sp

    f = np.load(Path(__file__).parent / "fixtures" / "aucell_pyscenic.npz")
    counts = f["counts"]
    n, g = counts.shape
    sc = bv.io.load_csr(
        tmp_path / "sc",
        sp.csr_matrix(counts),
        pl.DataFrame({"cell_id": [f"cell_{i}" for i in range(n)]}),
        pl.DataFrame({"gene_id": [f"gene_{j}" for j in range(g)]}),
        qc=QC,
    )
    names = ["set_5", "set_15", "set_30"]
    sets = {k: [f"gene_{j}" for j in f[k]] for k in names}
    res = pathways.aucell(sc, sets)
    assert res.cell_ids.tolist() == [f"cell_{i}" for i in range(n)]
    np.testing.assert_allclose(res.values, f["auc"], atol=1e-6)
