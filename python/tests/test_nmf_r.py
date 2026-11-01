"""NMF parity with the bixverse R package.

R loads seeded synthetic data into two directories, drops the first cells on
one and picks HVGs there (`dev/sc_nmf_fixtures.R`). Python copies the
untouched directory, applies the same filter, takes R's HVGs and reruns one
HALS run (NNDSVD and random initialisation, raw and normalised counts, a cell
and gene selection with sd scaling), stabilised restarts, consensus NMF (on
the spectra and on the usages) and the k sweep; then the same on
bootstrapped meta cells over R's meta cell HVGs. Skipped when Rscript or the
package is missing.

Both sides run the same Rust on the same inputs with the same seeds, so ids,
cluster labels and counts are compared exactly and every factor at `RTOL`;
the factors come out identical, so no component matching is needed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import bixverse as bv
from bixverse import metacells
from bixverse import nmf as bvnmf
from bixverse.params import BtMetacellsParams, NmfConsensusParams, NmfHalsParams

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_nmf_fixtures.R"

RTOL = 1e-12

HALS_100 = NmfHalsParams(max_iter=100)


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


###########
# Helpers #
###########


def _close(got, want, name: str) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(
        [np.nan if v is None else v for v in np.ravel(want)], dtype=np.float64
    ).reshape(got.shape)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=1e-12, err_msg=name)


def _check_single(res: bvnmf.NmfResult, want: dict) -> None:
    assert list(res.gene_ids) == want["genes"]
    assert list(res.cell_ids) == want["cells"]
    _close(res.w, want["w"], "w")
    _close(res.h, want["h"], "h")
    _close(res.final_loss, want["final_loss"], "loss")
    assert res.n_iter == want["n_iter"]
    assert res.converged == want["converged"]


def _check_multi(res: bvnmf.StabilisedNmfResult, want: dict) -> None:
    _close(res.w_all, want["w_all"], "w_all")
    assert list(res.w_all_columns) == want["w_all_columns"]
    for got, h in zip(res.h_per_run, want["h_per_run"].values(), strict=True):
        _close(got, h, "h_per_run")
    _close(res.losses, want["losses"], "losses")
    assert list(res.converged) == want["converged"]
    assert res.best_idx == want["best_idx"]


def _check_consensus(res: bvnmf.ConsensusNmfResult, want: dict) -> None:
    _close(res.w, want["w"], "w")
    _close(res.h, want["h"], "h")
    _close(res.stability, want["stability"], "stability")
    _close(res.rel_error, want["rel_error"], "rel_error")
    _close(res.rel_run_errors, want["rel_run_errors"], "rel_run_errors")
    cl = res.clusters
    assert cl["component_id"].to_list() == want["component_id"]
    assert cl["cluster"].to_list() == want["cluster"]
    assert cl["kept"].to_list() == want["kept"]
    _close(cl["local_density"].to_numpy(), want["local_density"], "density")
    _close(cl["silhouette"].to_numpy(), want["silhouette"], "silhouette")
    assert res.cluster_sizes["n"].to_list() == want["cluster_sizes"]
    assert res.n_dropped == want["n_dropped"]
    assert res.n_empty_clusters == want["n_empty_clusters"]


def _check_sweep(res: bvnmf.NmfKSweepResult, want: dict) -> None:
    f = res.frame
    for col in (
        "k",
        "consensus_failed",
        "n_dropped",
        "n_empty_clusters",
        "n_converged",
    ):
        assert f[col].to_list() == want[col], col
    for col in ("stability", "best_error", "median_error"):
        _close(f[col].to_numpy(), want[col], col)


############
# Fixtures #
############


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the NMF runs."""
    root = tmp_path_factory.mktemp("r_nmf")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched store with R's filter, HVGs, PCA and kNN."""
    directory = tmp_path_factory.mktemp("py_nmf") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.map.hvg = np.array(ref["hvg"], dtype=np.uint32)
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    sc.cache.knn = bv.KnnGraph(
        indices=np.array(ref["knn"]["indices"], dtype=np.uint32),
        distances=np.array(ref["knn"]["dist"], dtype=np.float32),
        dist_metric=ref["knn"]["metric"],
    )
    sc.stamp("knn", parents=["pca"])
    return sc


@pytest.fixture(scope="module")
def mc(ref, sc):
    """Bootstrapped meta cells with R's meta cell HVGs (R's are 1-based)."""
    mc = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=60, k=10)
    )
    assert mc is not None
    assert mc.cell_ids().tolist() == ref["mc_ids"]
    mc.hvg = np.array(ref["mc_hvg"], dtype=np.uint32) - 1
    return mc


#########
# Cells #
#########


def test_nmf_matches_r(ref, sc):
    _check_single(bvnmf.nmf(sc, 3), ref["single"])


def test_nmf_random_raw_matches_r(ref, sc):
    res = bvnmf.nmf(
        sc,
        3,
        use_second_layer=False,
        nmf_hals_params=NmfHalsParams(nmf_init="random", max_iter=100),
        seed=7,
    )
    _check_single(res, ref["single_random_raw"])


def test_nmf_selection_matches_r(ref, sc):
    res = bvnmf.nmf(
        sc,
        3,
        cell_ids=ref["cell_sel"],
        gene_ids=ref["gene_sel"],
        preprocessing="sd",
    )
    _check_single(res, ref["single_sel"])


def test_stabilised_nmf_matches_r(ref, sc):
    res = bvnmf.stabilised_nmf(sc, 3, n_runs=4, nmf_hals_params=HALS_100)
    _check_multi(res, ref["multi"])


def test_consensus_nmf_matches_r(ref, sc):
    res = bvnmf.consensus_nmf(sc, 3, n_runs=6, nmf_hals_params=HALS_100)
    _check_consensus(res, ref["consensus"])


def test_consensus_nmf_on_usages_matches_r(ref, sc):
    res = bvnmf.consensus_nmf(
        sc,
        3,
        n_runs=6,
        preprocessing="sqrt_sd",
        nmf_hals_params=HALS_100,
        nmf_consensus_params=NmfConsensusParams(
            consensus_target="w", density_threshold=2
        ),
        seed=3,
    )
    _check_consensus(res, ref["consensus_w"])


def test_nmf_k_sweep_matches_r(ref, sc):
    res = bvnmf.nmf_k_sweep(sc, range(2, 5), n_runs=4, nmf_hals_params=HALS_100)
    _check_sweep(res, ref["sweep"])


##############
# Meta cells #
##############


def test_mc_nmf_matches_r(ref, mc):
    _check_single(bvnmf.nmf(mc, 3), ref["mc_single"])
    res = bvnmf.nmf(mc, 3, use_second_layer=False, preprocessing="sd")
    _check_single(res, ref["mc_single_raw"])


def test_mc_stabilised_and_consensus_match_r(ref, mc):
    _check_multi(
        bvnmf.stabilised_nmf(mc, 3, n_runs=4, nmf_hals_params=HALS_100),
        ref["mc_multi"],
    )
    _check_consensus(
        bvnmf.consensus_nmf(mc, 3, n_runs=6, nmf_hals_params=HALS_100),
        ref["mc_consensus"],
    )


def test_mc_nmf_k_sweep_matches_r(ref, mc):
    res = bvnmf.nmf_k_sweep(mc, [2, 3, 4], n_runs=4, nmf_hals_params=HALS_100)
    _check_sweep(res, ref["mc_sweep"])
