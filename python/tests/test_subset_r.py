"""SingleCellsSubset parity with the bixverse R package.

R loads seeded synthetic data into two directories, drops the first cells on
one, cuts a subset on one cell type and runs the subset chain on it
(`dev/sc_subset_fixtures.R`): HVGs, PCA, neighbours, Leiden, fast clustering,
markers on the sub-clusters and PHATE. Python copies the untouched directory,
applies the same filter, cuts the same subset, runs the same chain from
scratch and has to reproduce R's numbers. Both sides then check the parent
came through untouched. Skipped when Rscript or the package is missing.

Tolerances, as the earlier families. Both sides run the same Rust numerics
on the same store and cell set (measured: at most 5e-15, relative for the
PCA, singular values and markers, absolute for the kNN distances, edge
weights and PHATE). Index sets, HVGs, kNN indices, graph
structure and cluster labels are compared exactly. f64 marker statistics at
`rtol=1e-12`. f32 outputs (PCA, kNN distances, edge weights) at `rtol=1e-5`,
`atol=1e-6` (scaled by the largest entry for the PCA). PHATE at `atol=1e-6`, as
in `test_embed_r.py`.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import bixverse as bv
from bixverse import de, graph, pp
from bixverse.params import FastClusterParams

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_subset_fixtures.R"

RTOL_F64 = 1e-12
RTOL_F32 = 1e-5
ATOL_F32 = 1e-6


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the subset chain."""
    root = tmp_path_factory.mktemp("r_subset")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def run(ref, tmp_path_factory):
    """The parent, its state before, and the subset after the Python chain."""
    directory = tmp_path_factory.mktemp("py_subset") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    before = {
        "obs": sc.obs_columns(),
        "var": sc.var().columns,
        "keep": sc.cells_to_keep.copy(),
    }

    sub = bv.SingleCellsSubset(sc, ref["grouping_column"], ref["group"])
    pp.find_hvg(sub, hvg_no=100)
    pp.calculate_pca(sub, 15)
    graph.find_neighbours(sub)
    graph.find_clusters(sub, res=1.0, name="sub_leiden", seed=42)
    fast = graph.fast_cluster(
        sub, resolutions=[1.0, 0.5], fc_params=FastClusterParams(n_centroids=20)
    )
    markers = de.find_all_markers(sub, "sub_leiden")
    return {"sc": sc, "sub": sub, "before": before, "fast": fast, "markers": markers}


def _same_columns(got: np.ndarray, want: np.ndarray, err_msg: str) -> None:
    """Equal relative to the largest entry."""
    assert got.shape == want.shape, err_msg
    np.testing.assert_allclose(
        got,
        want,
        rtol=RTOL_F32,
        atol=ATOL_F32 * np.abs(want).max(),
        err_msg=err_msg,
    )


#########
# Tests #
#########


def test_subset_cells_match_r(ref, run):
    sub = run["sub"]
    assert sub.subset_to_original.tolist() == ref["subset_to_original"]
    assert sub.cells_to_keep.tolist() == ref["cells_to_keep"]
    assert sub.cell_ids().tolist() == ref["cell_ids"]
    assert sub.shape[0] == len(ref["cell_ids"])


def test_subset_hvg_and_pca_match_r(ref, run):
    sub = run["sub"]
    assert sub.map.hvg is not None
    assert sub.map.hvg.tolist() == ref["hvg"]
    want = ref["pca"]
    np.testing.assert_allclose(
        sub.cache.pca_singular_vals, want["singular_values"], rtol=RTOL_F32
    )
    _same_columns(sub.cache.pca_factors, np.array(want["factors"]), "factors")
    _same_columns(sub.cache.pca_loadings, np.array(want["loadings"]), "loadings")


def test_subset_neighbours_match_r(ref, run):
    knn = run["sub"].cache.knn
    assert knn is not None
    assert knn.dist_metric == ref["knn"]["metric"]
    np.testing.assert_array_equal(knn.indices, np.array(ref["knn"]["indices"]))
    np.testing.assert_allclose(
        knn.distances, np.array(ref["knn"]["dist"]), rtol=RTOL_F32, atol=ATOL_F32
    )
    g = run["sub"].cache.connectivities
    want = ref["graph"]
    assert g is not None
    assert g.shape == (want["n"], want["n"])
    np.testing.assert_array_equal(g.indptr, want["indptr"])
    np.testing.assert_array_equal(g.indices, want["indices"])
    np.testing.assert_allclose(
        g.data, np.array(want["data"]), rtol=RTOL_F32, atol=ATOL_F32
    )


def test_subset_clusters_match_r(ref, run):
    assert run["sub"].obs()["sub_leiden"].to_list() == ref["leiden"]
    fast = run["fast"].memberships
    want = ref["fast"]
    assert fast.columns == list(want)
    assert [c + 1 for c in fast["cell_idx"].to_list()] == want["cell_idx"]
    for col in fast.columns[1:]:
        assert fast[col].to_list() == want[col], col


def test_subset_markers_match_r(ref, run):
    got = run["markers"]
    want = ref["markers"]
    assert got.columns == list(want)
    assert got["grp"].to_list() == want["grp"]
    assert got["gene_id"].to_list() == want["gene_id"]
    for col in ("lfc", "prop1", "prop2", "z_scores", "p_values", "fdr"):
        np.testing.assert_allclose(
            got[col].to_numpy(), want[col], rtol=RTOL_F64, err_msg=col
        )


def test_subset_phate_matches_r(ref, run):
    embed = pytest.importorskip("bixverse.embed", exc_type=ImportError)
    if not ref["has_phate"]:
        pytest.skip("R side has no manifoldsR")
    sub = run["sub"]
    embed.phate(sub)
    np.testing.assert_allclose(
        sub.cache.embeddings["phate"], np.array(ref["phate"]), rtol=0, atol=1e-6
    )


def test_parent_unchanged_in_both(ref, run):
    want = ref["parent"]
    assert want == {
        "hvg": 0,
        "pca": False,
        "knn": False,
        "obs_cols_unchanged": True,
        "var_cols_unchanged": True,
        "keep_unchanged": True,
    }
    sc, before = run["sc"], run["before"]
    assert sc.map.hvg is None
    assert sc.cache.artefacts() == {}
    assert sc.obs_columns() == before["obs"]
    assert sc.var().columns == before["var"]
    np.testing.assert_array_equal(sc.cells_to_keep, before["keep"])
    # the subset's labels live in the subset alone
    assert "sub_leiden" not in sc.obs_columns()
