"""GPU PCA parity with the bixverse.gpu R package.

R loads seeded synthetic data into two directories, finds the HVGs and runs
`calculate_pca_gpu_sc` with the block Lanczos and the randomised solver on one
(`dev/sc_gpu_fixtures.R`) and leaves the other untouched. Python runs the same
steps on a copy of the untouched one. Skipped without Rscript, bixverse.gpu,
the GPU wheel or a GPU adapter.

Tolerances. Both sides run the GPU SVD in `f32` on the same store and seed,
so the components agree to a few f32 ulps.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import bixverse as bv
from bixverse.params import PcaParams

bg = pytest.importorskip("bixverse_gpu")

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_gpu_fixtures.R"

RTOL = 1e-6
ATOL = 1e-7


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        [
            "Rscript",
            "-e",
            'quit(status = !requireNamespace("bixverse.gpu", quietly = TRUE))',
        ],
        capture_output=True,
    )
    return probe.returncode == 0


pytestmark = [
    pytest.mark.skipif(not bg.gpu_available(), reason="no GPU adapter"),
    pytest.mark.skipif(not _r_available(), reason="needs Rscript and bixverse.gpu"),
]


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R's GPU PCA made of the store."""
    root = tmp_path_factory.mktemp("r_gpu")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture
def sc(ref, tmp_path):
    """A fresh copy of R's untouched directory with R's cell filter and HVGs."""
    directory = tmp_path / "py_gpu"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    bv.pp.find_hvg(sc, hvg_no=ref["hvg_no"])
    assert sc.map.hvg is not None
    assert sc.map.hvg.tolist() == ref["hvg"]
    return sc


def _same_columns(got: np.ndarray, want: np.ndarray, err_msg: str) -> None:
    """Equal relative to the largest entry."""
    assert got.shape == want.shape, err_msg
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=ATOL * np.abs(want).max(), err_msg=err_msg
    )


@pytest.mark.parametrize("solver", ["exact", "randomised"])
def test_gpu_pca_matches_r(ref, sc, solver):
    want = ref["pca"][solver]
    bg.pp.calculate_pca(sc, 10, pca_params=PcaParams(svd_solver=solver), seed=42)
    np.testing.assert_allclose(
        sc.cache.pca_singular_vals, want["singular_values"], rtol=RTOL
    )
    _same_columns(
        sc.cache.pca_factors, np.asarray(want["factors"]), f"{solver}: factors"
    )
    _same_columns(
        sc.cache.pca_loadings, np.asarray(want["loadings"]), f"{solver}: loadings"
    )
