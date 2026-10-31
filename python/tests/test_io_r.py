"""Ingestion parity with the bixverse R package.

R writes h5ad, mtx and 10x h5 inputs and ingests them (`dev/sc_io_fixtures.R`);
Python ingests the same files and has to end up with the same directory:
the same cells and genes, obs / var tables of the same columns, types and
values, and the same counts. The reverse direction has R open what Python
wrote. Skipped when Rscript or the package is missing.
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
from bixverse.params import (
    MinQualityParams,
    MtxIoParams,
    SyntheticCellsweepParams,
    SyntheticDataAdtParams,
    SyntheticDataParams,
    SyntheticDialogueParams,
)

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_io_fixtures.R"
QC = MinQualityParams(min_unique_genes=5, min_lib_size=25, min_cells=5)
INDEX_COLS = ("cell_idx", "gene_idx")


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
    """The R written inputs and what R made of them."""
    root = tmp_path_factory.mktemp("r_io")
    _run_r("write", str(root), str(root / "reference.json"))
    out = json.loads((root / "reference.json").read_text())
    out["root"] = root
    return out


###########
# Helpers #
###########


def _types(directory: Path, table: str) -> dict[str, str]:
    con = duckdb.connect(str(directory / "sc_duckdb.db"), read_only=True)
    try:
        return {r[0]: r[1] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    finally:
        con.close()


def _same_column(name: str, got: list, want: list) -> None:
    assert len(got) == len(want), name
    if name in INDEX_COLS:
        assert [g + 1 for g in got] == want, name
    elif any(isinstance(g, float) for g in got):
        # jsonlite writes whole doubles as integers and NA as null
        np.testing.assert_allclose(
            np.array([np.nan if g is None else g for g in got], dtype=np.float64),
            np.array([np.nan if w is None else w for w in want], dtype=np.float64),
            err_msg=name,
        )
    else:
        assert got == want, name


def _assert_same(sc: bv.SingleCells, rep: dict) -> None:
    """A Python opened experiment matches an R report of the same data."""
    assert list(sc.shape) == rep["dims"]
    assert _types(sc.dir, "obs") == rep["obs_types"]
    assert _types(sc.dir, "var") == rep["var_types"]

    for frame, want in ((sc.obs(), rep["obs"]), (sc.var(), rep["var"])):
        assert frame.columns == list(want)
        for name, values in want.items():
            got = frame[name]
            if isinstance(got.dtype, pl.Enum):
                got = got.cast(pl.String)
            _same_column(name, got.to_list(), values)

    cells = np.array(rep["probe"], dtype=np.int64) - 1
    for layer in ("raw", "norm"):
        got = bv.get_counts(sc, cells=cells, layer=layer).scipy_csr().toarray()
        np.testing.assert_allclose(got, np.array(rep[layer]), rtol=1e-6, atol=1e-7)


def _rna(ref) -> bv.io.SyntheticData:
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(
            n_cells=300, n_genes=60, n_samples=3, sample_bias="slightly_uneven"
        ),
        seed=123,
    )


#############
# Synthetic #
#############


def test_rna_generator_matches_r(ref):
    want = ref["synthetic"]["rna"]
    d = _rna(ref)
    assert list(d.counts.shape) == want["shape"]
    assert d.counts.indptr.tolist() == want["indptr"]
    assert d.counts.indices.tolist() == want["indices"]
    np.testing.assert_array_equal(d.counts.data, want["data"])
    for frame, cols in ((d.obs, want["obs"]), (d.var, want["var"])):
        assert frame.columns == list(cols)
        for name, values in cols.items():
            _same_column(name, frame[name].to_list(), values)


def test_adt_generator_matches_r(ref):
    want = ref["synthetic"]["adt"]
    d = bv.io.generate_single_cell_test_data_adt(
        SyntheticDataAdtParams(n_cells=300), seed=123
    )
    np.testing.assert_array_equal(d.counts, np.array(want["counts"]))
    for frame, cols in ((d.obs, want["obs"]), (d.var, want["var"])):
        assert frame.columns == list(cols)
        for name, values in cols.items():
            _same_column(name, frame[name].to_list(), values)


def test_cellsweep_generator_matches_r(ref):
    want = ref["synthetic"]["cellsweep"]
    d = bv.io.generate_cellsweep_test_data(
        SyntheticCellsweepParams(n_real=60, n_empty=100, n_genes=60), seed=3
    )
    assert d.counts.indptr.tolist() == want["indptr"]
    assert d.counts.indices.tolist() == want["indices"]
    np.testing.assert_array_equal(d.counts.data, want["data"])
    assert d.obs.columns == list(want["obs"])
    for name, values in want["obs"].items():
        _same_column(name, d.obs[name].to_list(), values)
    np.testing.assert_allclose(d.ambient_true, want["ambient_true"])
    np.testing.assert_allclose(
        d.celltype_profiles_true, np.array(want["celltype_profiles_true"])
    )


def test_dialogue_generator_matches_r(ref):
    want = ref["synthetic"]["dialogue"]
    d = bv.io.generate_dialogue_test_data(
        SyntheticDialogueParams(n_samples=6, cells_per_sample=5, n_genes=40), seed=5
    )
    assert d.counts.indptr.tolist() == want["indptr"]
    assert d.counts.indices.tolist() == want["indices"]
    np.testing.assert_array_equal(d.counts.data, want["data"])
    for name, values in want["obs"].items():
        _same_column(name, d.obs[name].to_list(), values)
    assert d.latent == pytest.approx(want["latent"])
    assert d.planted == want["planted"]
    first = d.features["cell_type_1"]
    assert first["cell_id"].to_list() == want["features_ct1_cells"]
    np.testing.assert_allclose(
        first.drop("cell_id").to_numpy(), np.array(want["features_ct1"])
    )


#########
# Scans #
#########


def test_h5ad_dimensions_and_slot_match_r(ref):
    f = ref["files"]
    for case in ("csr", "csc", "dense"):
        dims = bv.io.get_h5ad_dimensions(f[f"h5ad_{case}"])
        want = ref["h5ad_dims"][case]
        assert [dims.n_obs, dims.n_var] == want["dims"]
        assert dims.type == want["type"]
    assert bv.io.detect_raw_count_slot(f["h5ad_csr"]) == ref["raw_slot"]
    assert bv.io.detect_raw_count_slot(f["h5ad_norm"]) == ref["raw_slot_norm"]
    meta = bv.io.get_tenx_h5_metadata(f["tenx_v3"])
    assert (meta.version, meta.n_cells, meta.n_genes) == (
        ref["tenx_meta"]["version"],
        ref["tenx_meta"]["n_cells"],
        ref["tenx_meta"]["n_genes"],
    )


def test_prescans_match_r(ref):
    f = ref["files"]
    scan = bv.io.prescan_h5ad_files({"a": f["h5ad_csr"], "b": f["h5ad_b"]})
    assert list(scan.universe) == ref["prescan_h5ad"]["universe"]
    assert scan.task("b")["gene_local_to_universe"] == ref["prescan_h5ad"]["mapping_b"]
    scan = bv.io.prescan_tenx_h5_files({"a": f["tenx_v3"], "b": f["tenx_v2"]})
    assert list(scan.universe) == ref["prescan_tenx"]["universe"]
    assert scan.task("a")["gene_local_to_universe"] == ref["prescan_tenx"]["mapping_a"]


###########
# Loaders #
###########


def test_load_csr_matches_load_r_data(ref, tmp_path):
    d = _rna(ref)
    sc = bv.io.load_csr(tmp_path / "sc", d.counts, d.obs, d.var, qc=QC)
    _assert_same(sc, ref["r_data"])


@pytest.mark.parametrize(
    ("case", "streaming"),
    [("h5ad_csr", True), ("h5ad_csc", False), ("h5ad_dense", True)],
)
def test_load_h5ad_matches_r(ref, tmp_path, case, streaming):
    sc = bv.io.load_h5ad(
        tmp_path / "sc", ref["files"][case], qc=QC, h5ad_streaming=streaming
    )
    _assert_same(sc, ref[case])


def test_load_h5ad_norm_matches_r(ref, tmp_path):
    sc = bv.io.load_h5ad_norm(
        tmp_path / "sc", ref["files"]["h5ad_norm"], "total_counts", 1e4, qc=QC
    )
    _assert_same(sc, ref["h5ad_norm"])


def test_load_mtx_matches_r(ref, tmp_path):
    src = Path(ref["files"]["mtx_a"])
    params = MtxIoParams(
        path_mtx=str(src / "matrix.mtx"),
        path_obs=str(src / "barcodes.csv"),
        path_var=str(src / "features.csv"),
        cells_as_rows=True,
        has_hdr=True,
    )
    _assert_same(bv.io.load_mtx(tmp_path / "sc", params, qc=QC), ref["mtx"])


@pytest.mark.parametrize("case", ["tenx_v3", "tenx_v2"])
def test_load_tenx_h5_matches_r(ref, tmp_path, case):
    sc = bv.io.load_tenx_h5(tmp_path / "sc", ref["files"][case], qc=QC)
    _assert_same(sc, ref[case])


def test_load_multi_h5ad_matches_r(ref, tmp_path):
    f = ref["files"]
    scan = bv.io.prescan_h5ad_files({"a": f["h5ad_csr"], "b": f["h5ad_b"]})
    sc = bv.io.load_multi_h5ad(tmp_path / "sc", scan, qc=QC)
    _assert_same(sc, ref["multi_h5ad"])


def test_load_multi_mtx_matches_r(ref, tmp_path):
    f = ref["files"]
    scan = bv.io.prescan_mtx_dirs(
        [f["mtx_a"], f["mtx_b"]], ["a", "b"], cells_as_rows=True, has_hdr=True
    )
    sc = bv.io.load_multi_mtx(tmp_path / "sc", scan, qc=QC)
    _assert_same(sc, ref["multi_mtx"])


def test_load_multi_tenx_h5_matches_r(ref, tmp_path):
    f = ref["files"]
    scan = bv.io.prescan_tenx_h5_files({"a": f["tenx_v3"], "b": f["tenx_v2"]})
    sc = bv.io.load_multi_tenx_h5(tmp_path / "sc", scan, qc=QC)
    _assert_same(sc, ref["multi_tenx"])


def _python_merge(ref, tmp_path) -> bv.SingleCells:
    """Python's run of the R merge: the CSR h5ad without cell_type_2, plus v2."""
    a = bv.io.load_h5ad(tmp_path / "a", ref["files"]["h5ad_csr"], qc=QC)
    obs = a.obs()
    a.set_cells_to_keep(
        obs.filter(pl.col("cell_grp") != "cell_type_2")["cell_idx"].to_numpy()
    )
    b = bv.io.load_tenx_h5(tmp_path / "b", ref["files"]["tenx_v2"], qc=QC)
    return bv.io.merge_sc_experiments(tmp_path / "merged", [a, b], ["a", "b"])


def test_merge_matches_r(ref, tmp_path):
    _assert_same(_python_merge(ref, tmp_path), ref["merge"])


###############
# Python to R #
###############


def test_r_opens_what_python_wrote(ref, tmp_path):
    """R `load_existing` and R loaders on Python written directories and files."""
    d = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=150, n_genes=30), seed=11
    )
    ingested = bv.io.load_h5ad(tmp_path / "py_h5ad", ref["files"]["h5ad_csr"], qc=QC)
    merged = _python_merge(ref, tmp_path)

    h5ad = tmp_path / "py.h5ad"
    bv.io.write_h5ad_sc(h5ad, d.counts, d.obs, d.var)
    tenx = tmp_path / "py.h5"
    bv.io.write_tenx_h5_sc(
        tenx,
        d.counts,
        d.obs["cell_id"].to_list(),
        pl.DataFrame({"id": d.var["gene_id"], "name": d.var["ensembl_id"]}),
    )
    saved = bv.io.save_h5ad(merged, tmp_path / "saved.h5ad")

    tasks = {
        "ingested": {"kind": "existing", "dir": str(ingested.dir)},
        "merged": {"kind": "existing", "dir": str(merged.dir)},
        "written_h5ad": {"kind": "h5ad", "path": str(h5ad), "root": str(tmp_path)},
        "written_tenx": {"kind": "tenx", "path": str(tenx), "root": str(tmp_path)},
        "saved_h5ad": {"kind": "h5ad", "path": str(saved), "root": str(tmp_path)},
    }
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    _run_r("read", str(tmp_path / "tasks.json"), str(tmp_path / "out.json"))
    out = json.loads((tmp_path / "out.json").read_text())

    _assert_same(ingested, out["ingested"])
    _assert_same(merged, out["merged"])
    _assert_same(bv.io.load_h5ad(tmp_path / "p1", h5ad, qc=QC), out["written_h5ad"])
    _assert_same(bv.io.load_tenx_h5(tmp_path / "p2", tenx, qc=QC), out["written_tenx"])
    _assert_same(bv.io.load_h5ad(tmp_path / "p3", saved, qc=QC), out["saved_h5ad"])
