"""Pathway activity, HotSpot and SCENIC parity with the bixverse R package.

R loads seeded synthetic data into two directories, drops the first cells on
one, runs HVG, PCA and an exhaustive kNN there and hands both over
(`dev/sc_pathways_fixtures.R`). Python copies the untouched directory, applies
the same filter, takes R's PCA and kNN, and reruns every method on the three
planted marker blocks: module scores, AUCell, VISION (with autocorrelation on
R's null gene sets), ULM / MLM, HotSpot, the SCENIC gene filter, GRN
inference, TF to gene links, CisTarget on a synthetic ranking, regulons and
their binarisation, then the meta cell twins on bootstrapped meta cells.
Skipped when Rscript or the package is missing.

Both sides run the same Rust on the same inputs with the same seeds, so ids,
gene filters, links and regulons are compared exactly and every float at
`RTOL`, the tree importances included.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import metacells, pathways
from bixverse._core.pathways import _vision_autocor
from bixverse.params import (
    AucellParams,
    BtMetacellsParams,
    CistargetParams,
    HotspotParams,
    LmActivityParams,
    ScenicExtraTreesDefaults,
    ScenicGradientBoostingDefaults,
    ScenicParams,
    ScenicRandomForestDefaults,
    VisionParams,
)

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_pathways_fixtures.R"

RTOL = 1e-12


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


###########
# Helpers #
###########


def _as_list(x) -> list:
    """R's `auto_unbox` turns length-one vectors into scalars."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _close(got, want, name: str, rtol: float = RTOL) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(
        [np.nan if v is None else v for v in np.ravel(want)], dtype=np.float64
    ).reshape(got.shape)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-12, err_msg=name)


def _knn(want: dict) -> bv.KnnGraph:
    return bv.KnnGraph(
        indices=np.array(want["indices"], dtype=np.uint32),
        distances=np.array(want["dist"], dtype=np.float32),
        dist_metric=want["metric"],
    )


def _vgs(ref) -> dict:
    return {
        k: {s: _as_list(v.get(s)) for s in ("pos", "neg") if s in v}
        for k, v in ref["vgs"].items()
    }


def _null(ref) -> tuple[list, np.ndarray]:
    random = [
        [{k: _as_list(v) for k, v in sig.items()} for sig in cl]
        for cl in ref["vision_null"]["random"]
    ]
    return random, np.array(_as_list(ref["vision_null"]["clusters"]), dtype=np.uint32)


def _frame(d: dict) -> pl.DataFrame:
    return pl.DataFrame({k: _as_list(v) for k, v in d.items()})


def _check_links(got: pl.DataFrame, want: dict, name: str) -> None:
    want = _frame(want)
    assert got["tf"].to_list() == want["tf"].to_list(), name
    assert got["gene"].to_list() == want["gene"].to_list(), name
    for col in ("importance", "pairwise_cor"):
        if col in want.columns:
            _close(got[col].to_numpy(), want[col].to_list(), f"{name} {col}")
    if "cor_sign" in want.columns:
        assert got["cor_sign"].to_list() == want["cor_sign"].to_list(), name
    if "in_leading_edge" in want.columns:
        assert got["in_leading_edge"].to_list() == want["in_leading_edge"].to_list()


def _scenic(
    learner: str, n: int, *, leaf: int = 10, counts: int = 700, cells: float = 0.47
) -> ScenicParams:
    match learner:
        case "rf":
            return ScenicParams(
                min_counts=counts,
                min_cells=cells,
                gene_batch_strategy="random",
                learner_params=ScenicRandomForestDefaults(
                    n_trees=n, min_samples_leaf=leaf
                ),
            )
        case "et":
            return ScenicParams(
                min_counts=counts,
                min_cells=cells,
                learner_type="extratrees",
                gene_batch_strategy="random",
                learner_params=ScenicExtraTreesDefaults(
                    n_trees=n, min_samples_leaf=leaf
                ),
            )
        case _:
            return ScenicParams(
                min_counts=counts,
                min_cells=cells,
                learner_type="grnboost2",
                learner_params=ScenicGradientBoostingDefaults(
                    n_trees_max=n, min_samples_leaf=leaf
                ),
            )


############
# Fixtures #
############


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the pathways."""
    root = tmp_path_factory.mktemp("r_pw")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched store with R's filter, PCA and kNN."""
    directory = tmp_path_factory.mktemp("py_pw") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    sc.cache.knn = _knn(ref["knn"])
    sc.stamp("knn", parents=["pca"])
    return sc


@pytest.fixture(scope="module")
def mc(ref, sc):
    """Bootstrapped meta cells with R's PCA and kNN."""
    mc = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=60, k=10)
    )
    assert mc is not None
    assert mc.cell_ids().tolist() == ref["mc_ids"]
    mc.cache.pca_factors = np.array(ref["mc_pca"], dtype=np.float32)
    mc.stamp("pca")
    mc.cache.knn = _knn(ref["mc_knn"])
    mc.stamp("knn", parents=["pca"])
    return mc


@pytest.fixture(scope="module")
def grn_rf(ref, sc):
    return pathways.scenic_grn(sc, ref["tfs"], scenic_params=_scenic("rf", 20))


##########################
# Module scores / AUCell #
##########################


def test_module_scores_match_r(ref, sc):
    res = pathways.module_scores(sc, ref["gs"], n_ctrl=5)
    assert res.columns == ("type_1", "type_2", "type_3")
    assert res.cell_ids.tolist() == ref["keep"]
    _close(res.values, ref["module_scores"], "module scores")


@pytest.mark.parametrize("auc_type", ["recovery", "wilcox", "ap"])
def test_aucell_matches_r(ref, sc, auc_type):
    res = pathways.aucell(
        sc,
        ref["gs"],
        aucell_params=AucellParams(auc_type=auc_type, standardise=auc_type == "ap"),
    )
    _close(res.values, ref["aucell"][auc_type], auc_type)


def test_aucell_max_rank_matches_r(ref, sc):
    res = pathways.aucell(sc, ref["gs"], aucell_params=AucellParams(max_rank=15))
    _close(res.values, ref["aucell_max_rank"], "max_rank")


def test_aucell_streaming_equals_in_memory(ref, sc):
    a = pathways.aucell(sc, ref["gs"], streaming=True)
    b = pathways.aucell(sc, ref["gs"], streaming=False)
    np.testing.assert_array_equal(a.values, b.values)


##########
# VISION #
##########


def test_vision_matches_r(ref, sc):
    res = pathways.vision(sc, _vgs(ref))
    _close(res.values, ref["vision"], "vision")


def _check_autocor(res, want, name: str) -> None:
    _close(res.vision_matrix.values, want["vision"], f"{name} scores")
    for col in ("auto_cor", "p_val", "fdr"):
        _close(res.auto_cor[col].to_numpy(), want[col], f"{name} {col}")


@pytest.mark.parametrize("use_knn", [True, False])
def test_vision_autocor_matches_r(ref, sc, use_knn):
    random, clusters = _null(ref)
    embd = sc.cache.pca_factors if use_knn else sc.cache.pca_factors[:, :5]
    res = _vision_autocor(
        sc,
        _vgs(ref),
        np.ascontiguousarray(embd),
        random,
        clusters,
        use_knn=use_knn,
        vision_params=VisionParams(n_perm=50, k=15, knn_method="exhaustive"),
        streaming=None,
        seed=42,
        verbose=False,
    )
    want = ref["vision_autocor"] if use_knn else ref["vision_autocor_search"]
    _check_autocor(res, want, f"use_knn={use_knn}")


def test_vision_w_autocor_runs_with_its_own_null(ref, sc):
    res = pathways.vision_w_autocor(
        sc, _vgs(ref), "pca", vision_params=VisionParams(n_perm=50, k=15)
    )
    # scores do not depend on the null, the p-values only on its draws
    _close(res.vision_matrix.values, ref["vision"], "scores")
    assert res.auto_cor["gene_set_name"].to_list() == list(ref["vgs"])
    assert ((res.auto_cor["p_val"] > 0) & (res.auto_cor["p_val"] <= 1)).all()


###############
# ULM / MLM   #
###############


@pytest.mark.parametrize("method", ["ulm", "mlm"])
def test_lm_activity_matches_r(ref, sc, method):
    net = _frame(ref["net"])
    res = pathways.lm_activity(
        sc, net, lm_params=LmActivityParams(method=method, p_values=True)
    )
    assert isinstance(res, pathways.LmActivityRes)
    assert res.es.columns == ("tf_a", "tf_b", "tf_c")
    want = ref[method]
    _close(res.es.values, want["es"], f"{method} es")
    _close(res.pvals.values, want["pvals"], f"{method} pvals")
    _close(res.fdr.values, want["fdr"], f"{method} fdr")


def test_lm_activity_coef_matches_r(ref, sc):
    res = pathways.lm_activity(
        sc, _frame(ref["net"]), lm_params=LmActivityParams(stat="coef")
    )
    assert isinstance(res, pathways.ScMatrixRes)
    _close(res.values, ref["ulm_coef"], "coef")


###########
# HotSpot #
###########


def _check_hs_autocor(got: pl.DataFrame, want: dict, name: str) -> None:
    want = _frame(want)
    assert got["gene_id"].to_list() == want["gene_id"].to_list(), name
    for col in ("gaerys_c", "z_score", "pval", "fdr"):
        _close(got[col].to_numpy(), want[col].to_list(), f"{name} {col}")


def test_hotspot_autocor_matches_r(ref, sc):
    res = pathways.hotspot_autocor(
        sc,
        hotspot_params=HotspotParams(k=15, knn_method="exhaustive"),
        genes_to_take=ref["hs_genes"],
    )
    _check_hs_autocor(res, ref["hs_autocor"], "cached kNN")


def test_hotspot_autocor_cell_subset_matches_r(ref, sc):
    res = pathways.hotspot_autocor(
        sc,
        hotspot_params=HotspotParams(model="normal", k=10, knn_method="exhaustive"),
        no_embd_to_use=5,
        cells_to_take=ref["hs_cells"],
        genes_to_take=ref["hs_genes"],
    )
    _check_hs_autocor(res, ref["hs_autocor_sub"], "searched kNN")


def test_hotspot_gene_cor_and_modules_match_r(ref, sc):
    hs = pathways.hotspot_gene_cor(
        sc,
        hotspot_params=HotspotParams(k=15, knn_method="exhaustive"),
        genes_to_take=ref["gs"]["type_1"] + ref["gs"]["type_2"] + ref["gs"]["type_3"],
    )
    want = ref["hs_pair"]
    _close(hs.z, want["z"], "z")
    _close(hs.cor, want["cor"], "cor")
    hs = pathways.generate_hotspot_membership(hs, min_size=3)
    assert hs.module_membership is not None
    assert hs.module_membership["cluster_member"].to_list() == want["member"]


##########
# SCENIC #
##########


def test_scenic_gene_filter_matches_r(ref, sc):
    got = pathways.scenic_gene_filter(
        sc, scenic_params=ScenicParams(min_counts=700, min_cells=0.47)
    )
    assert 0 < len(got) < sc.n_genes
    assert got == ref["scenic_filter"]


@pytest.mark.parametrize(("learner", "n"), [("rf", 20), ("et", 20), ("gbm", 50)])
def test_scenic_grn_matches_r(ref, sc, learner, n):
    grn = pathways.scenic_grn(sc, ref["tfs"], scenic_params=_scenic(learner, n))
    want = ref["grn"][learner]
    assert list(grn.gene_ids) == want["genes"]
    assert list(grn.tf_ids) == want["tfs"]
    _close(grn.importance_matrix, want["imp"], learner)


def test_scenic_grn_streaming_matches_r(ref, sc):
    grn = pathways.scenic_grn(
        sc, ref["tfs"], scenic_params=_scenic("rf", 20), streaming=True
    )
    _close(grn.importance_matrix, ref["grn_streaming"]["imp"], "streaming")


def test_tf_to_gene_links_match_r(ref, sc, grn_rf):
    thr = pathways.identify_tf_to_genes(grn_rf, n_sd=1)
    _check_links(thr.tf_to_gene_results, ref["links_threshold"], "threshold")
    top = pathways.identify_tf_to_genes(
        grn_rf, method="top_k", k_tfs=2, k_genes=5, min_importance=0.01
    )
    _check_links(top.tf_to_gene_results, ref["links_top_k"], "top_k")
    cor = pathways.tf_to_genes_correlations(thr, sc, mode="both")
    _check_links(cor.tf_to_gene_results, ref["links_cor"], "correlations")


def _ranking(ref) -> tuple[np.ndarray, list[str], list[str], pl.DataFrame]:
    return (
        np.array(ref["rankings"], dtype=np.int32),
        [f"gene_{i:03d}" for i in range(1, len(ref["rankings"]) + 1)],
        ref["motifs"],
        _frame(ref["annot"]),
    )


CIS = CistargetParams(nes_threshold=1, auc_threshold=0.2)


def _check_cis(got: pl.DataFrame, want: dict) -> None:
    want = _frame(want)
    for col in ("motif", "gs_name", "rank_at_max", "n_enriched", "leading_edge_genes"):
        assert got[col].to_list() == want[col].to_list(), col
    for col in ("TF_highConf", "TF_lowConf"):
        assert got[col].to_list() == want[col].to_list(), col
    _close(got["nes"].to_numpy(), want["nes"].to_list(), "nes")
    _close(got["auc"].to_numpy(), want["auc"].to_list(), "auc")


def test_run_cistarget_matches_r(ref, sc):
    ranks, genes, motifs, annot = _ranking(ref)
    g = ref["gs"]
    with pytest.warns(UserWarning, match="confidence categories"):
        got = pathways.run_cistarget(
            {
                "set_a": g["type_1"],
                "set_b": g["type_2"],
                "set_c": g["type_1"] + g["type_2"] + g["type_3"],
            },
            ranks,
            genes,
            motifs,
            annot,
            cis_target_params=CIS,
        )
    _check_cis(got, ref["cistarget"])


def test_motif_enrichment_and_regulons_match_r(ref, sc, grn_rf):
    ranks, genes, motifs, annot = _ranking(ref)
    grn = pathways.identify_tf_to_genes(grn_rf, n_sd=1)
    grn = pathways.tf_to_genes_correlations(grn, sc, mode="both")
    with pytest.warns(UserWarning, match="confidence categories"):
        grn = pathways.tf_to_genes_motif_enrichment(
            grn,
            ranks,
            genes,
            motifs,
            annot,
            cis_target_params=CIS,
            only_high_conf_tf=False,
        )
    _check_links(grn.tf_to_gene_results, ref["motif_links"], "motif links")
    _check_cis(grn.cis_targets_results, ref["motif_cis"])
    reg = pathways.build_regulons(grn, min_genes=2, mode="both")
    assert reg == {k: _as_list(v) for k, v in ref["regulons"].items()}
    reg = pathways.build_regulons(grn, use_leading_edge=False, min_genes=3)
    assert reg == {k: _as_list(v) for k, v in ref["regulons_no_le"].items()}


def test_binarise_matches_r(ref, sc):
    auc = pathways.aucell(sc, ref["gs"])
    res = pathways.binarise_regulon_activity(auc)
    want = ref["binarise"]
    _close(res.thresholds["threshold"].to_numpy(), want["thresholds"], "thresholds")
    assert res.thresholds["bimodal"].to_list() == want["bimodal"]
    assert res.thresholds["n_cells_on"].to_list() == want["n_cells_on"]
    got = res.binary.drop("cell_id").to_numpy().astype(int)
    np.testing.assert_array_equal(got, np.array(want["binary"]))


##############
# Meta cells #
##############


def test_mc_scores_match_r(ref, mc):
    _close(pathways.aucell(mc, ref["gs"]).values, ref["mc_aucell"], "mc aucell")
    _close(pathways.vision(mc, _vgs(ref)).values, ref["mc_vision"], "mc vision")
    res = pathways.lm_activity(
        mc, _frame(ref["net"]), lm_params=LmActivityParams(p_values=True)
    )
    assert isinstance(res, pathways.LmActivityRes)
    for k in ("es", "pvals", "fdr"):
        _close(getattr(res, k).values, ref["mc_ulm"][k], f"mc ulm {k}")


def test_mc_vision_autocor_matches_r(ref, mc):
    random, clusters = _null(ref)
    res = _vision_autocor(
        mc,
        _vgs(ref),
        np.ascontiguousarray(mc.cache.pca_factors),
        random,
        clusters,
        use_knn=True,
        vision_params=VisionParams(n_perm=50, k=8, knn_method="exhaustive"),
        streaming=None,
        seed=42,
        verbose=False,
    )
    # R draws a second null for the meta cells with the same seed, so the
    # random gene sets are the same as for the cells
    _check_autocor(res, ref["mc_vision_autocor"], "mc")


def test_mc_hotspot_matches_r(ref, mc):
    params = HotspotParams(k=8, knn_method="exhaustive")
    res = pathways.hotspot_autocor(
        mc, hotspot_params=params, genes_to_take=ref["hs_genes"]
    )
    _check_hs_autocor(res, ref["mc_hs_autocor"], "mc")
    hs = pathways.hotspot_gene_cor(
        mc,
        hotspot_params=params,
        genes_to_take=ref["gs"]["type_1"] + ref["gs"]["type_2"] + ref["gs"]["type_3"],
    )
    _close(hs.z, ref["mc_hs_pair"]["z"], "mc z")
    _close(hs.cor, ref["mc_hs_pair"]["cor"], "mc cor")


def test_mc_scenic_matches_r(ref, mc):
    p = _scenic("rf", 20, leaf=50, counts=1100, cells=0.1)
    kept = pathways.scenic_gene_filter(mc, scenic_params=p)
    assert kept == ref["mc_scenic_filter"]
    assert 0 < len(kept) < mc.n_genes
    grn = pathways.scenic_grn(mc, ref["tfs"], scenic_params=p)
    want = ref["mc_grn"]
    assert list(grn.gene_ids) == want["genes"]
    assert list(grn.tf_ids) == want["tfs"]
    _close(grn.importance_matrix, want["imp"], "mc grn")
    grn = pathways.identify_tf_to_genes(grn, n_sd=1)
    grn = pathways.tf_to_genes_correlations(grn, mc, mode="both")
    _check_links(grn.tf_to_gene_results, ref["mc_links_cor"], "mc links")
