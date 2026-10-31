"""MetaCells parity with the bixverse R package.

R loads seeded synthetic data into two directories, drops the first cells on
one and runs HVGs, PCA and neighbours (`dev/sc_metacells_fixtures.R`). Python
copies the untouched directory, applies the same filter and takes R's PCA and
kNN, so every generator sees the same inputs. Then both run the four
generators, the MetaCells chain (HVGs, PCA, residuals, neighbours, Leiden),
purity, the diffusion map, the manifold metrics, a merge, and meta cells on a
subset and per group. Skipped when Rscript or the package is missing.

Both sides run the same Rust on the same inputs with the same seeds, so the
memberships, the summed raw counts, the HVG selections, kNN indices, cluster
labels, regions and labels are compared exactly, every float at `RTOL`
(measured: at most 5.3e-15, relative for the normalised counts, SEACells
RSS, HVG statistics, PCA, residual PCA, density distances, compactness,
separation and purity, absolute for the kNN distances), PCA and diffusion
components up to a sign each, scaled by the largest entry. The subset and
per group runs build their own HVGs, PCA and kNN on both sides first.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import graph, metacells, pp
from bixverse.params import (
    BtMetacellsParams,
    HvgParams,
    Metacells2DacParams,
    Metacells2Params,
    NeighboursParams,
    PcaParams,
    SeacellsParams,
    SupercellParams,
)

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_metacells_fixtures.R"

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


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the meta cells."""
    root = tmp_path_factory.mktemp("r_mc")
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
    directory = tmp_path_factory.mktemp("py_mc") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    sc.cache.knn = bv.KnnGraph(
        indices=np.array(ref["knn"]["indices"], dtype=np.uint32),
        distances=np.array(ref["knn"]["dist"], dtype=np.float32),
        dist_metric=ref["knn"]["metric"],
    )
    sc.stamp("knn", parents=["pca"])
    return sc


BT = BtMetacellsParams(target_no_metacells=40, k=10)


@pytest.fixture(scope="module")
def mc_bt(sc):
    mc = metacells.generate_bt_meta_cells(sc, params=BT)
    assert mc is not None
    return mc


@pytest.fixture(scope="module")
def chain(mc_bt):
    """The MetaCells chain on a fresh copy of the bootstrapped meta cells."""
    mc = _copy(mc_bt)
    pp.find_hvg(mc, hvg_no=50)
    pp.calculate_pca(mc, 10)
    graph.find_neighbours(mc, neighbours_params=NeighboursParams(k=8))
    graph.find_clusters(mc, res=1.0, name="mc_leiden")
    return mc


def _copy(mc: metacells.MetaCells) -> metacells.MetaCells:
    """A fresh MetaCells over the same data, empty cache."""
    return metacells.MetaCells(
        mc.obs(),
        mc.var(cols=["gene_idx", "gene_id"]),
        mc.raw,
        mc.norm,
        mc.original_assignment,
        mc.meta_cell_method,
        other_data=dict(mc.other_data),
    )


def _check_mc(mc: metacells.MetaCells, want: dict) -> None:
    """Ids, memberships, provenance and both count matrices."""
    obs = mc.obs()
    assert obs["meta_cell_id"].to_list() == want["ids"]
    assert obs["original_cell_idx"].to_list() == want["members"]
    a = mc.original_assignment
    assert (a.n_cells, a.n_metacells, a.n_unassigned) == (
        want["n_cells"],
        want["n_metacells"],
        want["n_unassigned"],
    )
    for layer in ("raw", "norm"):
        got, exp = getattr(mc, layer), want[layer]
        assert list(got.shape) == exp["dim"]
        np.testing.assert_array_equal(got.indptr, exp["p"])
        np.testing.assert_array_equal(got.indices, exp["j"])
        if layer == "raw":
            np.testing.assert_array_equal(got.data, exp["x"])
        else:
            np.testing.assert_allclose(got.data, exp["x"], rtol=RTOL)


def _same_up_to_sign(got: np.ndarray, want: np.ndarray, err_msg: str) -> None:
    """Columns equal up to a sign each, relative to the largest entry."""
    assert got.shape == want.shape, err_msg
    signs = np.sign(np.sum(got * want, axis=0))
    signs[signs == 0] = 1
    np.testing.assert_allclose(
        got * signs,
        want,
        rtol=RTOL,
        atol=RTOL * np.abs(want).max(),
        err_msg=err_msg,
    )


def _check_pca(mc: metacells.MetaCells, want: dict) -> None:
    np.testing.assert_allclose(
        mc.cache.pca_singular_vals, want["singular_values"], rtol=RTOL
    )
    _same_up_to_sign(mc.cache.pca_factors, np.array(want["factors"]), "factors")
    _same_up_to_sign(mc.cache.pca_loadings, np.array(want["loadings"]), "loadings")


##############
# Generators #
##############


def test_bootstrapped_matches_r(ref, mc_bt):
    assert mc_bt.meta_cell_method == "meta_cells_hdwgcna"
    _check_mc(mc_bt, ref["bt"])


def test_bootstrapped_regenerated_knn_matches_r(ref, sc):
    mc = metacells.generate_bt_meta_cells(
        sc, params=BT, regenerate_knn=True, no_embd_to_use=10
    )
    assert mc is not None
    _check_mc(mc, ref["bt_regen"])


def test_bootstrapped_cells_to_use_matches_r(ref, sc):
    mc = metacells.generate_bt_meta_cells(sc, params=BT, cells_to_use=ref["type_cells"])
    assert mc is not None
    _check_mc(mc, ref["bt_restricted"])


def test_seacells_matches_r(ref, sc):
    mc = metacells.generate_seacells(
        sc, SeacellsParams(n_sea_cells=30, min_iter=5, max_iter=20, k=10)
    )
    want = ref["sea"]
    _check_mc(mc, want)
    np.testing.assert_allclose(mc.other_data["rss"], want["rss"], rtol=RTOL)
    np.testing.assert_array_equal(mc.other_data["archetypes"], want["archetypes"])


def test_supercells_matches_r(ref, sc):
    mc = metacells.generate_supercells(
        sc, params=SupercellParams(graining_factor=10.0, k=10)
    )
    assert mc is not None
    _check_mc(mc, ref["super"])


def test_metacells2_matches_r(ref, sc):
    mc = metacells.generate_metacells2(
        sc,
        mc2_params=Metacells2Params(
            target_metacell_size=20, target_metacell_umis=20000
        ),
        dac_params=Metacells2DacParams(rare_gene_modules=False),
    )
    want = ref["mc2"]
    _check_mc(mc, want)
    info = mc.other_data["cell_info"]
    assert isinstance(info, pl.DataFrame)
    exp = want["cell_info"]
    assert info["original_cell_idx"].to_list() == [
        i - 1 for i in exp["original_cell_idx"]
    ]
    assert info["dissolved"].to_list() == exp["dissolved"]
    assert info["level"].to_list() == exp["level"]
    assert mc.other_data["selected_genes"] == want["selected_genes"]


#########
# Chain #
#########


def test_mc_hvg_matches_r(ref, chain):
    assert chain.hvg is not None
    assert chain.hvg.tolist() == ref["mc_hvg"]
    var = chain.var()
    for col in ("mean", "var", "var_exp", "var_std"):
        np.testing.assert_allclose(
            var[col].to_numpy(), ref["mc_var"][col], rtol=RTOL, err_msg=col
        )


def test_mc_hvg_scran_and_data_match_r(ref, mc_bt):
    mc = _copy(mc_bt)
    pp.find_hvg(mc, hvg_no=50, hvg_params=HvgParams(method="scran"))
    assert mc.hvg is not None
    assert mc.hvg.tolist() == ref["mc_hvg_scran"]

    want = ref["mc_hvg_data"]
    got = pp.get_hvg_data(
        mc_bt,
        cell_ids=mc_bt.cell_ids()[:30].tolist(),
        hvg_no=20,
        hvg_params=HvgParams(method="dispersion"),
    )
    assert got["is_hvg"].to_list() == want["is_hvg"]
    assert got["hvg_rank"].to_list() == want["hvg_rank"]
    np.testing.assert_allclose(
        got["dispersion"].to_numpy(), want["dispersion"], rtol=RTOL
    )


def test_mc_pca_matches_r(ref, chain):
    _check_pca(chain, ref["mc_pca"])


def test_mc_pca_clr_matches_r(ref, chain):
    mc = _copy(chain)
    mc.hvg = chain.hvg
    pp.calculate_pca(mc, 10, pca_params=PcaParams(clr=True))
    _check_pca(mc, ref["mc_pca_clr"])


def test_mc_residuals_match_r(ref, mc_bt):
    mc = _copy(mc_bt)
    pp.fit_residuals(mc, method="analytic_pearson")
    pp.find_hvg(mc, hvg_no=50, hvg_params=HvgParams(method="residual"))
    assert mc.hvg is not None
    assert mc.hvg.tolist() == ref["mc_res_hvg"]
    pp.calculate_pca(
        mc, 10, pca_params=PcaParams(normalise_variance=False), residuals=True
    )
    _check_pca(mc, ref["mc_res_pca"])
    assert mc.check_state(["pca"]) is True


def test_mc_neighbours_and_clusters_match_r(ref, chain):
    knn = chain.cache.knn
    assert knn is not None
    np.testing.assert_array_equal(knn.indices, np.array(ref["mc_knn"]["indices"]))
    np.testing.assert_allclose(
        knn.distances, np.array(ref["mc_knn"]["dist"]), rtol=RTOL, atol=RTOL
    )
    assert chain.obs()["mc_leiden"].to_list() == ref["mc_leiden"]


###########
# Metrics #
###########


def test_purity_matches_r(ref, sc, chain):
    mc = _copy(chain)
    labels = sc.obs()["cell_grp"].cast(pl.String).to_list()
    metacells.calc_meta_cell_purity(
        mc, labels, add_additional_info="top_two_labels", add_entropy=True
    )
    obs = mc.obs()
    want = ref["purity"]
    for col in ("mc_purity", "mc_second_frac", "mc_entropy"):
        np.testing.assert_allclose(
            obs[col].to_numpy(), want[col], rtol=RTOL, err_msg=col
        )
    for col in ("mc_top_label", "mc_second_label"):
        assert obs[col].to_list() == want[col], col

    filtered = sc.obs(filtered=True)
    order = pl.DataFrame({"cell_idx": sc.cells_to_keep}).join(
        filtered, on="cell_idx", maintain_order="left"
    )
    got = metacells.get_meta_cell_purity(
        mc, order["cell_grp"].cast(pl.String).to_list(), add_additional_info="top_label"
    )
    want = ref["purity_filtered"]
    assert got["meta_cell_idx"].to_list() == [i - 1 for i in want["meta_cell_idx"]]
    np.testing.assert_allclose(
        got["mc_purity"].to_numpy(), want["mc_purity"], rtol=RTOL
    )
    assert got["mc_top_label"].to_list() == want["mc_top_label"]


def test_diffusion_and_manifold_match_r(ref, sc, chain):
    mc = _copy(chain)
    assert sc.cache.knn is not None
    metacells.calc_diffusion_coordinates(mc, sc.cache.knn, n_dcs=6, k_density=30)
    metacells.calc_manifold_metrics(mc)
    want = ref["diffusion"]
    dcs = mc.other_data["dcs"]
    assert isinstance(dcs, np.ndarray)
    _same_up_to_sign(dcs, np.array(want["dcs"]), "dcs")
    np.testing.assert_allclose(
        mc.other_data["density_dist"], want["density_dist"], rtol=RTOL
    )
    assert mc.other_data["regions"] == want["regions"]
    obs = mc.obs()
    assert obs["density_region"].to_list() == want["density_region"]
    np.testing.assert_allclose(
        obs["compactness"].to_numpy(), want["compactness"], rtol=RTOL
    )
    np.testing.assert_allclose(
        obs["separation"].to_numpy(), want["separation"], rtol=RTOL
    )

    free_c = metacells.calc_meta_cell_compactness(dcs, ref["free_assignment"])
    free_s = metacells.calc_meta_cell_separation(dcs, ref["free_assignment"])
    assert free_c["meta_cell"].to_list() == ref["free"]["names"]
    np.testing.assert_allclose(
        free_c["compactness"].to_numpy(), ref["free"]["compactness"], rtol=RTOL
    )
    np.testing.assert_allclose(
        free_s["separation"].to_numpy(), ref["free"]["separation"], rtol=RTOL
    )


##################
# Merge / groups #
##################


def test_merge_matches_r(ref, sc, chain):
    regen = metacells.generate_bt_meta_cells(
        sc, params=BT, regenerate_knn=True, no_embd_to_use=10
    )
    assert regen is not None
    with pytest.warns(UserWarning):
        merged = metacells.merge_meta_cells({"donor_a": chain, "donor_b": regen})
    want = ref["merged"]
    _check_mc(merged, want)
    assert merged.is_merged
    obs = merged.obs()
    assert obs["source_id"].to_list() == want["source_id"]
    assert obs.columns == want["obs_cols"]
    assert merged.other_data["sources"] == want["sources"]


def test_subset_bootstrapped_matches_r(ref, sc):
    sub = bv.SingleCellsSubset(sc, "cell_grp", "cell_type_1")
    pp.find_hvg(sub, hvg_no=100)
    pp.calculate_pca(sub, 10)
    graph.find_neighbours(sub)
    mc = metacells.generate_bt_meta_cells(
        sub, params=BtMetacellsParams(target_no_metacells=15, k=10)
    )
    assert mc is not None
    _check_mc(mc, ref["sub_bt"])


def test_per_group_matches_r(ref, sc):
    def prep(sub: bv.SingleCellsSubset) -> None:
        pp.find_hvg(sub, hvg_no=100)
        pp.calculate_pca(sub, 10)
        graph.find_neighbours(sub)

    mc = metacells.meta_cells_per_group(
        sc,
        "cell_grp",
        mc_params={"params": BtMetacellsParams(target_no_metacells=10, k=10)},
        prep=prep,
    )
    want = ref["per_group"]
    _check_mc(mc, want)
    assert mc.obs()["source_id"].to_list() == want["source_id"]
