"""QC parity with the bixverse R package.

R loads seeded synthetic data into two directories, runs its QC helpers on
one (`dev/sc_qc_fixtures.R`) and leaves the other untouched. Python runs the
same pipeline on a copy of the untouched one and has to end up with the same
obs table, column for column and type for type, and the same `CellQc`
internals. Python also opens the R directory, and R opens the Python one.
Skipped when Rscript or the package is missing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import qc

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_qc_fixtures.R"


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


def _run_r(*args: str) -> None:
    res = subprocess.run(
        ["Rscript", str(SCRIPT), *args], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the QC pipeline."""
    root = tmp_path_factory.mktemp("r_qc")
    _run_r("write", str(root), str(root / "reference.json"))
    return json.loads((root / "reference.json").read_text())


###########
# Helpers #
###########


def _types(directory: Path) -> dict[str, str]:
    con = duckdb.connect(str(directory / "sc_duckdb.db"), read_only=True)
    try:
        return {r[0]: r[1] for r in con.execute("DESCRIBE obs").fetchall()}
    finally:
        con.close()


def _same_column(name: str, got: list, want: list) -> None:
    assert len(got) == len(want), name
    if name == "cell_idx":
        assert [g + 1 for g in got] == want, name
    elif any(isinstance(v, float) for v in [*got, *want] if v is not None):
        # jsonlite writes whole doubles as integers and NA as null
        np.testing.assert_allclose(
            np.array([np.nan if g is None else g for g in got], dtype=np.float64),
            np.array([np.nan if w is None else w for w in want], dtype=np.float64),
            rtol=1e-12,
            err_msg=name,
        )
    else:
        assert got == want, name


def _same_frame(frame: pl.DataFrame, want: dict) -> None:
    assert frame.columns == list(want)
    for name, values in want.items():
        got = frame[name]
        if isinstance(got.dtype, pl.Enum):
            got = got.cast(pl.String)
        _same_column(name, got.to_list(), values)


def _same_obs(sc: bv.SingleCells, rep: dict) -> None:
    assert _types(sc.dir) == rep["obs_types"]
    _same_frame(sc.obs(), rep["obs"])


def _same_qc(got: qc.CellQc, want: dict) -> None:
    assert got.combined.tolist() == want["combined"]
    _same_frame(qc.get_data(got), want["data"])
    if want["thresholds"] is None:
        assert got.thresholds is None
    else:
        assert got.thresholds is not None
        _same_frame(got.thresholds, want["thresholds"])
    if want["per_group_stats"] is None:
        assert got.per_group_stats is None
    else:
        assert got.per_group_stats is not None
        _same_frame(got.per_group_stats, want["per_group_stats"])


def _bounds(raw: dict) -> dict[str, dict[str, float]]:
    return {m: {k: float(v) for k, v in b.items()} for m, b in raw.items()}


#########
# Tests #
#########


def test_per_cell_qc_outlier_matches_r(ref):
    x = np.array(ref["outlier_input"], dtype=np.float64)
    for direction in ("twosided", "below", "above"):
        for std in (True, False):
            want = ref["outlier"][f"{direction}_{str(std).upper()}"]
            res = qc.per_cell_qc_outlier(
                x, threshold=2.0, direction=direction, r_standard=std
            )
            assert res.outlier.tolist() == want["outlier"]
            got = {
                "median": res.median,
                "upper_threshold": res.upper_threshold,
                "lower_threshold": res.lower_threshold,
            }
            for k, v in want["metrics"].items():
                if v is None:
                    assert got[k] is None, k
                else:
                    assert got[k] == pytest.approx(v, rel=1e-12), k


def test_qc_pipeline_matches_r(ref, tmp_path):
    """Metric passes, outlier calls, filter and column edits, end to end."""
    directory = tmp_path / "py_qc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sets = ref["gene_sets"]

    qc.gene_set_proportions(sc, {k: sets[k] for k in ("set_a", "set_b")})
    qc.top_genes_perc(sc, top_n_vals=ref["top_n_vals"])

    obs = sc.obs(filtered=True)
    metrics = {
        "lib_size": obs["lib_size"],
        "nnz": obs["nnz"],
        "set_a": obs["set_a"],
        "top_10": obs["top_10_genes_percentage"],
    }
    cells = obs["cell_idx"].to_numpy()

    res = qc.run_cell_qc(
        metrics,
        cells,
        directions=ref["directions"],
        threshold=2.0,
        groups=obs[ref["group_col"]],
    )
    _same_qc(res, ref["qc_mad"])
    res = qc.flag_cells(res, _bounds(ref["hard"]))
    _same_qc(res, ref["qc_flagged"])
    res = qc.rescue_cells(res, _bounds(ref["rescue"]))
    _same_qc(res, ref["qc_rescued"])

    _same_qc(qc.run_cell_qc(metrics, cells, threshold=2.0), ref["qc_ungrouped"])
    _same_qc(
        qc.run_cell_qc_fixed(metrics, cells, _bounds(ref["hard"])), ref["qc_fixed"]
    )

    sc.add_obs(qc.get_data(res))
    keep = obs.filter(~pl.Series(res.combined))["cell_id"].to_list()
    assert keep == ref["keep"]
    sc.set_cells_to_keep(keep)
    qc.gene_set_proportions(sc, {"set_c": sets["set_c"]})
    sc.rename_obs_col("set_b", "set_b_renamed")
    sc.drop_obs_cols(["top_20_genes_percentage"])

    _same_obs(sc, ref["final"])
    _same_frame(qc.get_available_features(sc), ref["features"])

    # R opens the Python directory and sees the same table
    out = tmp_path / "out.json"
    _run_r("read", str(directory), str(out))
    _same_obs(sc, json.loads(out.read_text()))


def test_python_opens_r_qc_directory(ref):
    sc = bv.SingleCells(ref["r_dir"])
    _same_obs(sc, ref["final"])
    assert sc.n_cells == len(ref["keep"])
