"""A SingleCells directory written by R opens in Python, and the reverse.

The R side is `dev/sc_interop.R`, run against the installed bixverse R
package. Skipped when Rscript or the package is missing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_interop.R"


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


def _run_r(mode: str, directory: Path, out: Path) -> dict:
    res = subprocess.run(
        ["Rscript", str(SCRIPT), mode, str(directory), str(out)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def r_dir(tmp_path_factory):
    """A directory R built, filtered, annotated and saved, plus its reference."""
    root = tmp_path_factory.mktemp("r_interop")
    ref = _run_r("write", root / "sc", root / "reference.json")
    return root / "sc", ref


def _copy(r_dir, tmp_path, *, keep_r_memory: bool) -> Path:
    src, _ = r_dir
    dst = tmp_path / "sc"
    shutil.copytree(src, dst)
    if not keep_r_memory:
        (dst / "memory.qs2").unlink()
    return dst


###############
# R to Python #
###############


def test_python_opens_the_r_directory(r_dir):
    path, ref = r_dir
    sc = bv.io.load_existing(path)

    assert sc.dims == (ref["n_cells_total"], len(ref["gene_ids"]))
    assert list(sc.shape) == ref["dims"]
    assert sc.cell_ids().tolist() == ref["cell_ids"]
    assert sc.gene_ids().tolist() == ref["gene_ids"]


def test_r_cell_filter_arrives_zero_based(r_dir):
    path, ref = r_dir
    sc = bv.io.load_existing(path)
    # the fixture hands R a descending filter; R's map is 0-based and ascending
    assert ref["cells_to_keep_r"] == sorted(ref["cells_to_keep_r"])
    assert sc.cells_to_keep.tolist() == ref["cells_to_keep_r"]

    obs = sc.obs()
    assert obs["cell_idx"].to_list() == [i - 1 for i in ref["obs_cell_idx"]]
    assert obs["to_keep"].to_list() == ref["obs_to_keep"]
    kept = sc.obs(filtered=True)["cell_idx"].to_list()
    assert kept == ref["cells_to_keep_r"]


def test_r_obs_column_arrives(r_dir):
    path, ref = r_dir
    obs = bv.io.load_existing(path).obs(cols=["arm", "cell_grp"])
    assert obs["arm"].to_list() == ref["obs_arm"]
    assert obs["cell_grp"].to_list() == ref["obs_cell_grp"]


@pytest.mark.parametrize("layer", ["raw", "norm"])
def test_counts_match_what_r_reads(r_dir, layer):
    path, ref = r_dir
    sc = bv.io.load_existing(path)
    cells = np.array(ref["probe_cell_idx"]) - 1
    got = bv.get_counts(sc, cells=cells, layer=layer).scipy_csr().toarray()
    np.testing.assert_allclose(got, np.array(ref[f"probe_{layer}"]), rtol=1e-6)


###############
# Python to R #
###############


def _modify(path: Path) -> tuple[np.ndarray, pl.DataFrame]:
    sc = bv.io.load_existing(path)
    keep = sc.cells_to_keep[::3][:40]
    sc.set_cells_to_keep(keep)
    frame = pl.DataFrame(
        {
            "cell_idx": keep,
            "py_score": np.linspace(0.0, 1.0, len(keep)),
            "py_label": pl.Series(
                ["lo" if i % 2 else "hi" for i in range(len(keep))],
                dtype=pl.Enum(["lo", "hi"]),
            ),
        }
    )
    sc.add_obs(frame)
    bv.io.save(sc)
    return keep, frame


def _check_obs(back: dict, keep: np.ndarray, frame: pl.DataFrame, n_total: int) -> None:
    to_keep = np.zeros(n_total, dtype=bool)
    to_keep[keep] = True
    assert back["obs_to_keep"] == to_keep.tolist()

    score = [None] * n_total
    label = [None] * n_total
    for i, s, lab in frame.iter_rows():
        score[i], label[i] = s, lab
    assert back["obs_py_score"] == pytest.approx(score)
    assert back["obs_py_label"] == label
    assert back["py_label_is_factor"] is True
    assert back["py_label_levels"] == ["lo", "hi"]


def test_r_reads_python_filter_and_obs(r_dir, tmp_path):
    _, ref = r_dir
    path = _copy(r_dir, tmp_path, keep_r_memory=False)
    keep, frame = _modify(path)

    back = _run_r("read", path, tmp_path / "back.json")
    assert back["dims"] == [len(keep), len(ref["gene_ids"])]
    assert back["cells_to_keep_r"] == sorted(keep.tolist())
    _check_obs(back, keep, frame, ref["n_cells_total"])


def test_python_filter_wins_over_r_memory_file(r_dir, tmp_path):
    # obs.to_keep is authoritative on both sides: a stale memory.qs2 loses
    path = _copy(r_dir, tmp_path, keep_r_memory=True)
    keep, frame = _modify(path)

    back = _run_r("read", path, tmp_path / "back.json")
    assert back["cells_to_keep_r"] == keep.tolist()
    _check_obs(back, keep, frame, r_dir[1]["n_cells_total"])
