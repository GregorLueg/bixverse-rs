"""Doublet detection, empty droplet and CellSweep parity with the bixverse R
package.

R loads seeded synthetic data into two directories per family and runs its
methods on one (`dev/sc_doublets_fixtures.R`), leaving the other untouched.
Python copies the untouched one, applies the same cell filter and has to
reproduce R's results. Skipped when Rscript or the package is missing.

R plants 60 heterotypic doublets (summed type 1 and type 2 cells) so every
caller has something to find. Both sides run the same Rust on the same inputs
with the same seeds, so the results agree to the digits jsonlite prints
(measured: at most 5e-15 relative on every score, error, z-score, threshold,
rate and scDblFinder feature). Scores are compared at `rtol=1e-12`; calls,
cluster labels, pairs and empty droplet masks exactly. CellSweep writes through
the same writer with the same seeded rounding, so obs and var match column for
column and both count
layers of a sample of cells match exactly (measured: bit for bit on every
cell).

"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import doublets
from bixverse.params import (
    BoostParams,
    CellsweepParams,
    EmptyDropletsParams,
    MinQualityParams,
    ScdblfinderParams,
    ScrubletParams,
)

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_doublets_fixtures.R"

RTOL = 1e-12
SCRUBLET = ScrubletParams(no_pcs=10, min_gene_var_pctl=0.0, n_bins_histogram=20)
BOOST = BoostParams(min_gene_var_pctl=0.0, no_pcs=10, n_iters=5)
SCDBL = ScdblfinderParams(no_pcs=10, n_genes=50, cxds_genes=50)


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the doublet callers and CellSweep."""
    root = tmp_path_factory.mktemp("r_doublets")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), "write", str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched directory with R's cell filter. The doublet
    callers leave it alone, so one copy serves every test."""
    directory = tmp_path_factory.mktemp("py") / "py_doublets"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    return sc


@pytest.fixture(scope="module")
def cs(ref, tmp_path_factory):
    """A copy of R's untouched CellSweep directory with R's cell filter."""
    directory = tmp_path_factory.mktemp("py") / "py_cs"
    shutil.copytree(ref["cs_base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["cs_keep"])
    return sc


###########
# Helpers #
###########


def _close(got: object, want: object) -> None:
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, rtol=RTOL)


def _scalar(got: object, want: object) -> None:
    """R hands a scalar as a one-element list, a named vector as a dict."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and list(got) == list(want)
        for g in want:
            _close(got[g], want[g])
    else:
        _close(got, want[0])


def _same_scrublet(got: doublets.ScrubletRes, want: dict) -> None:
    np.testing.assert_array_equal(got.predicted_doublets, want["predicted_doublets"])
    _close(got.doublet_scores_obs, want["doublet_scores_obs"])
    _close(got.doublet_errors_obs, want["doublet_errors_obs"])
    _close(got.z_scores, want["z_scores"])
    for key in (
        "threshold",
        "detected_doublet_rate",
        "detectable_doublet_fraction",
        "overall_doublet_rate",
    ):
        _scalar(getattr(got, key), want[key])
    if isinstance(want["doublet_scores_sim"], dict):
        assert isinstance(got.doublet_scores_sim, dict)
        for g, sim in want["doublet_scores_sim"].items():
            _close(got.doublet_scores_sim[g], sim)
    else:
        _close(got.doublet_scores_sim, want["doublet_scores_sim"])
    np.testing.assert_array_equal(got.cell_indices, want["cell_indices"])
    if want["cell_groups"] is not None:
        assert got.cell_groups is not None
        assert got.cell_groups.tolist() == want["cell_groups"]
    assert repr(got).splitlines() == want["print"]
    assert doublets.get_data(got).columns == want["get_data_cols"]


def _same_store(got: bv.SingleCells, want: bv.SingleCells, rng_seed: int) -> None:
    """Same obs, var and, on a sample of cells, counts in both layers."""
    obs_got, obs_want = got.obs(), want.obs()
    assert obs_got.columns == obs_want.columns
    assert obs_got.height == obs_want.height
    for col in obs_want.columns:
        g, w = obs_got[col], obs_want[col]
        if w.dtype.is_float():
            _close(g.to_numpy(), w.to_numpy())
        else:
            assert g.cast(pl.String).to_list() == w.cast(pl.String).to_list(), col
    var_got, var_want = got.var(), want.var()
    assert var_got.columns == var_want.columns
    _close(var_got["cellsweep_ambient"], var_want["cellsweep_ambient"].to_numpy())
    assert var_got["no_cells_exp"].to_list() == var_want["no_cells_exp"].to_list()

    rng = np.random.default_rng(rng_seed)
    cells = np.sort(rng.choice(got.dims[0], size=min(60, got.dims[0]), replace=False))
    for layer in ("raw", "norm"):
        a = bv.get_counts(got, cells=cells, layer=layer).matrix
        b = bv.get_counts(want, cells=cells, layer=layer).matrix
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.indices, b.indices)
        if layer == "raw":
            np.testing.assert_array_equal(a.data, b.data)
        else:
            _close(a.data, b.data)


############
# Scrublet #
############


def test_scrublet_matches_r(ref, sc):
    res = doublets.scrublet(
        sc, scrublet_params=SCRUBLET, return_combined_pca=True, return_pairs=True
    )
    _same_scrublet(res, ref["scrublet"])
    np.testing.assert_array_equal(res.pair_1, ref["scrublet"]["pair_1"])
    np.testing.assert_array_equal(res.pair_2, ref["scrublet"]["pair_2"])
    pca, want = res.pca, np.array(ref["scrublet"]["pca"])
    assert isinstance(pca, np.ndarray) and pca.shape == want.shape
    np.testing.assert_allclose(pca, want, rtol=1e-6, atol=1e-7 * np.abs(want).max())


def test_scrublet_manual_matches_r(ref, sc):
    res = doublets.scrublet(
        sc, scrublet_params=SCRUBLET, return_combined_pca=True, return_pairs=True
    )
    _same_scrublet(doublets.call_doublets_manual(res, 0.3), ref["scrublet_manual"])


def test_scrublet_grouped_matches_r(ref, sc):
    res = doublets.scrublet(sc, scrublet_params=SCRUBLET, group_by="sample_id", seed=7)
    _same_scrublet(res, ref["scrublet_grouped"])
    _same_scrublet(
        doublets.call_doublets_manual(res, 0.25, for_sample="sample_2"),
        ref["scrublet_grouped_manual"],
    )


#########
# Boost #
#########


def _same_boost(got: doublets.BoostRes, want: dict) -> None:
    np.testing.assert_array_equal(got.doublet, want["doublet"])
    _close(got.doublet_score, want["doublet_score"])
    _close(got.voting_avg, want["voting_avg"])
    np.testing.assert_array_equal(got.cell_indices, want["cell_indices"])
    if want["cell_groups"] is not None:
        assert got.cell_groups is not None
        assert got.cell_groups.tolist() == want["cell_groups"]
    assert repr(got).splitlines() == want["print"]
    assert doublets.get_data(got).columns == want["get_data_cols"]


def test_boost_matches_r(ref, sc):
    _same_boost(doublets.doublet_detection_boost(sc, boost_params=BOOST), ref["boost"])


def test_boost_grouped_matches_r(ref, sc):
    res = doublets.doublet_detection_boost(sc, boost_params=BOOST, group_by="sample_id")
    _same_boost(res, ref["boost_grouped"])


###############
# scDblFinder #
###############


def _same_scdbl(got: doublets.ScDblFinderRes, want: dict) -> None:
    np.testing.assert_array_equal(got.predicted_doublets, want["predicted_doublets"])
    for key, r_key in (
        ("doublet_score", "doublet_score"),
        ("cxds_scores", "cxds_scores"),
        ("weighted", "weighted"),
    ):
        _close(getattr(got, key), want[r_key])
    _scalar(got.threshold, want["threshold"])
    _scalar(got.detected_doublet_rate, want["detected_doublet_rate"])
    assert [str(c) for c in got.cluster_labels.tolist()] == [
        str(c) for c in want["cluster_labels"]
    ]
    np.testing.assert_array_equal(got.cell_indices, want["cell_indices"])
    assert repr(got).splitlines() == want["print"]
    assert doublets.get_data(got).columns == want["get_data_cols"]


def test_scdblfinder_matches_r(ref, sc):
    res = doublets.scdblfinder(sc, scdblfinder_params=SCDBL, return_features=True)
    _same_scdbl(res, ref["scdblfinder"])
    features = doublets.get_feature_mat(res)
    assert features is not None
    assert features.columns[1:] == ref["scdblfinder"]["feature_names"]
    assert features["cell_id"].to_list() == sc.cell_ids(res.cell_indices).tolist()
    _close(features.drop("cell_id").to_numpy(), ref["scdblfinder"]["features"])


def test_scdblfinder_gbm_rounds_match_r(ref, sc):
    res = doublets.scdblfinder(
        sc,
        scdblfinder_params=ScdblfinderParams(
            no_pcs=10, n_genes=50, cxds_genes=50, gbm_n_trees=3
        ),
    )
    _same_scdbl(res, ref["scdblfinder_short"])
    short, full = ref["scdblfinder_short"], ref["scdblfinder"]
    assert short["doublet_score"] != full["doublet_score"]


def test_scdblfinder_grouped_matches_r(ref, sc):
    res = doublets.scdblfinder(sc, scdblfinder_params=SCDBL, group_by="sample_id")
    _same_scdbl(res, ref["scdblfinder_grouped"])


##################
# Empty droplets #
##################


@pytest.mark.parametrize(
    ("params", "key"),
    [
        (EmptyDropletsParams(method="knee"), "empty_knee"),
        (EmptyDropletsParams(method="umi_cutoff", umi_cutoff=400), "empty_cutoff"),
        (
            EmptyDropletsParams(method="expected_cells", expected_cells=380),
            "empty_expected",
        ),
    ],
)
def test_empty_droplets_match_r(ref, cs, params, key):
    got = doublets.infer_empty_droplets(cs.obs()["lib_size"], params)
    np.testing.assert_array_equal(got, ref[key])


#############
# CellSweep #
#############


def test_cellsweep_supplied_matches_r(ref, cs, tmp_path):
    out = doublets.cellsweep(
        tmp_path / "out",
        cs,
        "cell_grp",
        "sample_id",
        EmptyDropletsParams(method="supplied", is_empty_column="is_empty"),
    )
    _same_store(out, bv.SingleCells(ref["cs_out_supplied"]), rng_seed=1)


def test_cellsweep_cutoff_two_samples_matches_r(ref, cs, tmp_path):
    out = doublets.cellsweep(
        tmp_path / "out",
        cs,
        "cell_grp",
        "sample_two",
        EmptyDropletsParams(method="umi_cutoff", umi_cutoff=400),
        cellsweep_params=CellsweepParams(max_iter=300, seed=3),
        qc=MinQualityParams(target_size=5000.0),
    )
    _same_store(out, bv.SingleCells(ref["cs_out_cutoff"]), rng_seed=2)
