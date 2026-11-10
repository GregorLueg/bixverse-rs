"""Pipeline parity with the bixverse R package.

R loads seeded synthetic data into two directories, drops the first cells on
one and applies four pipelines (`dev/sc_pipeline_fixtures.R`): HVGs, PCA,
neighbours and Leiden on the experiment; the same chain per cell type through
`apply_pipeline_per_group`; meta cells followed by the MetaCells chain; and
per cell type pre-processing ending on meta cells, merged. Python copies the
untouched directory, applies the same filter and the same pipelines and has
to reproduce R's numbers. Skipped when Rscript or the package is missing.

Tolerances, as `test_subset_r.py` and `test_metacells_r.py`: both sides run
the same Rust on the same store and cell set. Index sets, HVGs, kNN indices,
cluster labels, memberships and raw counts are compared exactly. The f32 PCA
and kNN distances at `rtol=1e-5`, `atol=1e-6` (scaled by the largest entry for
the PCA); the f64 meta cell normalised
counts at `rtol=1e-12`.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import bixverse as bv
from bixverse import metacells, pipeline
from bixverse.params import BtMetacellsParams, NeighboursParams

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_pipeline_fixtures.R"

RTOL_F64 = 1e-12
RTOL_F32 = 1e-5
ATOL_F32 = 1e-6


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the pipelines."""
    root = tmp_path_factory.mktemp("r_pipe")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched store with R's filter and the experiment
    pipeline applied."""
    directory = tmp_path_factory.mktemp("py_pipe") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    p = (
        pipeline.step_hvg(hvg_no=100)
        >> pipeline.step_pca(15)
        >> pipeline.step_neighbours()
        >> pipeline.step_clusters(res=1.0, name="pipe_leiden")
    )
    assert pipeline.apply_pipeline(p, sc) is sc
    return sc


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


def _check_pca(obj: bv.SingleCells | bv.SingleCellsSubset | bv.MetaCells, want):
    assert obj.cache.pca_singular_vals is not None
    assert obj.cache.pca_factors is not None
    assert obj.cache.pca_loadings is not None
    np.testing.assert_allclose(
        obj.cache.pca_singular_vals, want["singular_values"], rtol=RTOL_F32
    )
    _same_columns(obj.cache.pca_factors, np.array(want["factors"]), "factors")
    _same_columns(obj.cache.pca_loadings, np.array(want["loadings"]), "loadings")


def _check_knn(obj: bv.SingleCells | bv.SingleCellsSubset, want: dict) -> None:
    knn = obj.cache.knn
    assert knn is not None
    assert knn.dist_metric == want["metric"]
    np.testing.assert_array_equal(knn.indices, np.array(want["indices"]))
    np.testing.assert_allclose(
        knn.distances, np.array(want["dist"]), rtol=RTOL_F32, atol=ATOL_F32
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
            np.testing.assert_allclose(got.data, exp["x"], rtol=RTOL_F64)


#########
# Tests #
#########


def test_experiment_pipeline_matches_r(ref, sc):
    want = ref["sc"]
    assert sc.map.hvg is not None
    assert sc.map.hvg.tolist() == want["hvg"]
    _check_pca(sc, want["pca"])
    _check_knn(sc, want["knn"])
    assert sc.obs(filtered=True)["pipe_leiden"].to_list() == want["leiden"]


def test_per_group_pipeline_matches_r(ref, sc):
    p = (
        pipeline.step_hvg(hvg_no=80)
        >> pipeline.step_pca(10)
        >> pipeline.step_neighbours()
        >> pipeline.step_clusters(res=1.0, name="grp_leiden")
    )
    out = pipeline.apply_pipeline_per_group(p, sc, "cell_grp")
    assert list(out) == ref["groups"]
    for g, sub in out.items():
        want = ref["per_group"][g]
        assert isinstance(sub, bv.SingleCellsSubset)
        assert sub.cell_ids().tolist() == want["cell_ids"]
        assert sub.map.hvg is not None
        assert sub.map.hvg.tolist() == want["hvg"], g
        _check_pca(sub, want["pca"])
        _check_knn(sub, want["knn"])
        assert sub.obs()["grp_leiden"].to_list() == want["leiden"], g
    assert "grp_leiden" not in sc.obs_columns()

    by_batch = pipeline.apply_pipeline_per_group(p, sc, "batch_index")
    assert list(by_batch) == ref["batches"]
    for g, sub in by_batch.items():
        assert isinstance(sub, bv.SingleCellsSubset)
        assert sub.map.hvg is not None
        assert sub.map.hvg.tolist() == ref["per_batch_hvg"][g], g


def test_metacell_pipeline_matches_r(ref, sc):
    p = (
        pipeline.step_metacells(
            "bootstrapped", params=BtMetacellsParams(target_no_metacells=40, k=10)
        )
        >> pipeline.step_hvg(hvg_no=50)
        >> pipeline.step_pca(10)
        >> pipeline.step_neighbours(neighbours_params=NeighboursParams(k=8))
        >> pipeline.step_clusters(res=1.0, name="mc_leiden")
    )
    end = pipeline.validate_pipeline(p, bv.SingleCells)
    assert end.__name__ == ref["mc_end_class"]
    mc = pipeline.apply_pipeline(p, sc)
    assert isinstance(mc, bv.MetaCells)
    want = ref["mc"]
    _check_mc(mc, want)
    assert mc.hvg is not None
    assert mc.hvg.tolist() == want["hvg"]
    _check_pca(mc, want["pca"])
    assert mc.obs()["mc_leiden"].to_list() == want["leiden"]


def test_per_group_metacells_match_r(ref, sc):
    p = (
        pipeline.step_hvg(hvg_no=100)
        >> pipeline.step_pca(10)
        >> pipeline.step_neighbours()
        >> pipeline.step_metacells(
            "bootstrapped", params=BtMetacellsParams(target_no_metacells=10, k=10)
        )
    )
    out = pipeline.apply_pipeline_per_group(p, sc, "cell_grp")
    assert all(isinstance(mc, bv.MetaCells) for mc in out.values())
    merged = metacells.merge_meta_cells(out)  # ty: ignore[invalid-argument-type]
    want = ref["grp_mc"]
    _check_mc(merged, want)
    assert merged.obs()["source_id"].to_list() == want["source_id"]
