"""Trajectory and Bonsai parity with the bixverse R package.

R loads seeded synthetic data into two directories, drops the first cells on
one and plants a kNN graph over the kept cells: a noisy Y, since the
synthetic generator makes cell types and no continuous process
(`dev/sc_traj_fixtures.R`). A noisy line goes over a subset and over
bootstrapped meta cells. Python copies the untouched directory, applies the
same filter, takes the planted graphs and the bin labels, and runs
Palantir, PAGA, the gene trends (normalised, fate weighted, MAGIC), the same
on the subset and the meta cells, Bonsai over the cells and the meta cells,
the relayouts and the Bonsai embedding. Skipped when Rscript or the package
is missing.

Both sides run the same Rust on the same inputs with the same seeds, so cell
ids, terminal states, waypoints, PAGA graph structure, branch cells, tree
topology, leaf order and genes used are compared exactly, every float at
`RTOL`.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import metacells, pp, traj, trees
from bixverse.params import (
    BonsaiParams,
    BtMetacellsParams,
    GeneTrendsParams,
    PalantirParams,
)

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_traj_fixtures.R"

RTOL = 1e-12


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


############
# Fixtures #
############


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the trajectories and trees."""
    root = tmp_path_factory.mktemp("r_traj")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


def _knn(want: dict) -> bv.KnnGraph:
    return bv.KnnGraph(
        indices=np.array(want["indices"], dtype=np.uint32),
        distances=np.array(want["dist"], dtype=np.float32),
        dist_metric=want["metric"],
    )


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched store with R's filter, the planted Y kNN and
    the bins."""
    directory = tmp_path_factory.mktemp("py_traj") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.knn = _knn(ref["knn"])
    sc.stamp("knn")
    sc.add_obs(pl.DataFrame({"cell_idx": sc.cells_to_keep, "traj_bin": ref["bins"]}))
    return sc


@pytest.fixture(scope="module")
def sub(ref, sc):
    """The subset with its planted line."""
    sub = bv.SingleCellsSubset(sc, ref["grouping_column"], ref["group"])
    sub.cache.knn = _knn(ref["sub_knn"])
    sub.stamp("knn")
    sub.add_obs(pl.DataFrame({"cell_idx": sub.cells_to_keep, "seg": ref["sub_bins"]}))
    return sub


@pytest.fixture(scope="module")
def mc(ref, sc):
    """Bootstrapped meta cells with their planted line."""
    mc = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=60, k=10)
    )
    assert mc is not None
    assert mc.cell_ids().tolist() == ref["mc_ids"]
    mc.cache.knn = _knn(ref["mc_knn"])
    mc.stamp("knn")
    mc.add_obs(pl.DataFrame({"seg": ref["mc_bins"]}))
    return mc


PALANTIR = PalantirParams(
    knn=24,
    num_waypoints=200,
    n_eigs=3,
    use_early_cell_as_start=True,
    knn_method="exhaustive",
)


@pytest.fixture(scope="module")
def pal(ref, sc):
    res = traj.run_palantir(sc, ref["early_cell"], palantir_params=PALANTIR)
    assert res is not None
    return res


@pytest.fixture(scope="module")
def pal_lab(ref, sc):
    states = dict(
        zip(ref["labels"], ref["palantir_labelled"]["terminal_states"], strict=True)
    )
    res = traj.run_palantir(
        sc, ref["early_cell"], terminal_states=states, palantir_params=PALANTIR
    )
    assert res is not None
    return res


@pytest.fixture(scope="module")
def tree(sc):
    return trees.bonsai(sc)


###########
# Helpers #
###########


def _vec(x) -> list:
    """A JSON vector R may have unboxed to a scalar (`auto_unbox`)."""
    return x if isinstance(x, list) else [x]


def _close(got, want, what: str) -> None:
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
        rtol=RTOL,
        atol=0,
        err_msg=what,
    )


def _check_palantir(res: traj.PalantirRes, want: dict) -> None:
    assert res.pseudotime["cell_id"].to_list() == want["cell_id"]
    _close(res.pseudotime["pseudotime"], want["pseudotime"], "pseudotime")
    _close(res.pseudotime["entropy"], want["entropy"], "entropy")
    assert res.branches == _vec(want["branches"])
    assert list(res.terminal_states.values()) == _vec(want["terminal_states"])
    _close(
        res.branch_probs.select(res.branches).to_numpy(),
        np.array(want["branch_probs"]).reshape(len(want["cell_id"]), -1),
        "branch_probs",
    )
    assert list(res.waypoints) == want["waypoints"]
    assert res.start_cell == want["start_cell"]
    _close(res.multiscale.drop("cell_id").to_numpy(), want["multiscale"], "multiscale")
    info = want["run_info"]
    for key in (
        "iterations",
        "converged",
        "eigen_converged",
        "repair_edges",
        "stranded_waypoints",
        "n_waypoints",
        "modality",
    ):
        assert res.run_info[key] == info[key], key
    _close(res.run_info["eigen_residual"], info["eigen_residual"], "eigen_residual")


def _check_paga(res: traj.PagaRes, want: dict) -> None:
    assert list(res.clusters) == want["clusters"]
    assert res.sizes["n_cells"].to_list() == want["sizes"]
    for name in ("connectivities", "connectivities_tree"):
        got, exp = getattr(res, name), np.array(want[name])
        np.testing.assert_array_equal(got != 0, exp != 0, err_msg=name)
        _close(got, exp, name)


def _check_trends(res: traj.GeneTrendsRes, want: dict) -> None:
    t = res.trends
    assert t["branch"].cast(pl.String).to_list() == want["branch"]
    assert t["gene"].cast(pl.String).to_list() == want["gene"]
    _close(t["pseudotime"], want["pseudotime"], "trend grid")
    _close(t["expression"], want["expression"], "trend expression")
    assert list(res.branches) == _vec(want["branches"])
    assert [list(res.branch_cells[b]) for b in res.branches] == want["branch_cells"]
    run_info = res.run_info
    n_cells, jitter = run_info["n_cells"], run_info["jitter_used"]
    assert isinstance(n_cells, dict) and isinstance(jitter, dict)
    assert [n_cells[b] for b in res.branches] == _vec(want["n_cells"])
    _close([jitter[b] for b in res.branches], _vec(want["jitter_used"]), "jitter")
    assert run_info["resolution"] == want["resolution"]


def _check_nodes(nodes: pl.DataFrame, want: dict) -> None:
    assert nodes["parent"].to_list() == want["parent"]
    assert nodes["is_leaf"].to_list() == want["is_leaf"]
    assert nodes["cell_id"].to_list() == want["cell_id"]
    assert nodes["n_cells"].to_list() == want["n_cells"]
    for col in ("branch", "x", "y"):
        _close(nodes[col], want[col], col)


def _check_bonsai(tree: trees.BonsaiTree, want: dict) -> None:
    _check_nodes(tree.nodes, want["nodes"])
    _close(tree.loglik, want["loglik"], "loglik")
    assert tree.steps["step"].to_list() == want["steps"]
    _close(tree.steps["loglik"], want["steps_loglik"], "step loglik")
    assert tree.timings["stage"].to_list() == want["stages"]
    assert list(tree.genes_used) == want["genes_used"]
    assert list(tree.genes_dropped) == want["genes_dropped"]
    assert tree.cell_idx.tolist() == want["cell_idx"]


############
# Palantir #
############


def test_palantir_detected_matches_r(ref, pal):
    _check_palantir(pal, ref["palantir"])


def test_palantir_labelled_matches_r(ref, pal_lab):
    want = ref["palantir_labelled"]
    _check_palantir(pal_lab, want)
    assert pal_lab.branches == want["state_names"]


def test_palantir_subset_matches_r(ref, sub):
    want = ref["sub_palantir"]
    res = traj.run_palantir(
        sub,
        want["cell_id"][0],
        terminal_states=[want["cell_id"][-1]],
        palantir_params=PalantirParams(
            knn=15,
            num_waypoints=80,
            n_eigs=3,
            use_early_cell_as_start=True,
            knn_method="exhaustive",
        ),
    )
    assert res is not None
    _check_palantir(res, want)


def test_palantir_metacells_matches_r(ref, mc):
    want = ref["mc_palantir"]
    res = traj.run_palantir(
        mc,
        want["cell_id"][0],
        terminal_states=[want["cell_id"][-1]],
        palantir_params=PalantirParams(
            knn=10,
            num_waypoints=30,
            n_eigs=3,
            use_early_cell_as_start=True,
            knn_method="exhaustive",
        ),
    )
    assert res is not None
    _check_palantir(res, want)
    trends = traj.run_gene_trends(
        mc,
        res,
        features=ref["features"],
        gene_trend_params=GeneTrendsParams(resolution=25),
    )
    _check_trends(trends, ref["mc_trends"])


########
# PAGA #
########


def test_paga_matches_r(ref, sc):
    res = traj.run_paga(sc, "traj_bin")
    assert res is not None
    _check_paga(res, ref["paga"])


def test_paga_subset_matches_r(ref, sub):
    res = traj.run_paga(sub, "seg")
    assert res is not None
    _check_paga(res, ref["sub_paga"])


def test_paga_metacells_matches_r(ref, mc):
    res = traj.run_paga(mc, "seg")
    assert res is not None
    _check_paga(res, ref["mc_paga"])


###############
# Gene trends #
###############


def test_gene_trends_matches_r(ref, sc, pal_lab):
    res = traj.run_gene_trends(
        sc,
        pal_lab,
        features=ref["features"],
        gene_trend_params=GeneTrendsParams(resolution=50),
    )
    _check_trends(res, ref["trends"])


def test_gene_trends_weighted_matches_r(ref, sc, pal_lab):
    res = traj.run_gene_trends(
        sc,
        pal_lab,
        features=ref["features"],
        gene_trend_params=GeneTrendsParams(
            resolution=30, weighting="fate_probability", length_scale=0.3
        ),
    )
    _check_trends(res, ref["trends_weighted"])


def test_gene_trends_magic_matches_r(ref, sc, pal_lab):
    pp.run_magic(sc, ref["features"])
    assert sc.cache.magic is not None
    _close(sc.cache.magic.data, ref["magic"], "magic")
    res = traj.run_gene_trends(
        sc, pal_lab, use_magic=True, gene_trend_params=GeneTrendsParams(resolution=50)
    )
    _check_trends(res, ref["trends_magic"])


def test_gene_trends_subset_matches_r(ref, sub):
    want = ref["sub_palantir"]
    pal = traj.run_palantir(
        sub,
        want["cell_id"][0],
        terminal_states=[want["cell_id"][-1]],
        palantir_params=PalantirParams(
            knn=15,
            num_waypoints=80,
            n_eigs=3,
            use_early_cell_as_start=True,
            knn_method="exhaustive",
        ),
    )
    assert pal is not None
    res = traj.run_gene_trends(
        sub,
        pal,
        features=ref["features"],
        gene_trend_params=GeneTrendsParams(resolution=25),
    )
    _check_trends(res, ref["sub_trends"])


##########
# Bonsai #
##########


def test_bonsai_matches_r(ref, tree):
    _check_bonsai(tree, ref["bonsai"])


def test_bonsai_candidates_matches_r(ref, sc):
    res = trees.bonsai(
        sc,
        hvg=[g - 1 for g in ref["hvg"]],
        bonsai_params=BonsaiParams(
            start="greedy_merge", nni_search="exact", layout="dendrogram", seed=3
        ),
    )
    _check_bonsai(res, ref["bonsai_hvg"])


def test_bonsai_relayout_matches_r(ref, tree):
    _check_nodes(
        trees.relayout_bonsai(tree, layout="dendrogram").nodes,
        ref["relayout_dendrogram"],
    )
    _check_nodes(
        trees.relayout_bonsai(tree, layout="equal_daylight", hyperbolic=True).nodes,
        ref["relayout_hyperbolic"],
    )


def test_bonsai_embedding_matches_r(ref, sc, tree):
    trees.set_bonsai_embedding(sc, tree)
    np.testing.assert_allclose(
        sc.cache.embeddings["bonsai"], np.array(ref["bonsai_embedding"]), rtol=1e-6
    )


def test_bonsai_metacells_matches_r(ref, mc):
    _check_bonsai(trees.bonsai(mc), ref["mc_bonsai"])
