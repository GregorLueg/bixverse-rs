"""Processing parity with the bixverse R package.

R loads seeded synthetic data into two directories, runs its HVG, PCA,
residual and MAGIC helpers on one (`dev/sc_pp_fixtures.R`) and leaves the
other untouched. Python runs the same steps on a copy of the untouched one,
after the same cell filter, and has to reproduce R's numbers. Skipped when
Rscript or the package is missing.

Tolerances. Both sides run the same Rust numerics on the same store, so the
results agree to the 15 significant digits jsonlite prints (measured: at most
5e-15 relative, randomised solvers and the default sparse Lanczos path
included). f64 outputs (HVG statistics, residual variances) are compared at
`rtol=1e-12`. f32 outputs (PCA, MAGIC, corrected counts) at `rtol=1e-5` and
`atol=1e-6` (scaled by the largest entry for the PCA): a few f32 ulps, room
for a different reduction order on another machine or thread count without
letting a wrong algorithm through. The randomised solvers run with R's seed.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import bixverse as bv
from bixverse import pp
from bixverse.params import AprParams, HvgParams, MagicParams, PcaParams

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_pp_fixtures.R"

RTOL_F64 = 1e-12
RTOL_F32 = 1e-5
ATOL_F32 = 1e-6


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


def _run_r(*args: str) -> None:
    res = subprocess.run(
        ["Rscript", str(SCRIPT), *args], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the processing steps."""
    root = tmp_path_factory.mktemp("r_pp")
    _run_r("write", str(root), str(root / "reference.json"))
    return json.loads((root / "reference.json").read_text())


@pytest.fixture
def sc(ref, tmp_path):
    """A fresh copy of R's untouched directory with R's cell filter."""
    directory = tmp_path / "py_pp"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    return sc


###########
# Helpers #
###########


def _arr(x: list) -> np.ndarray:
    """A JSON array as float64, null as NaN."""
    return np.array(
        [[np.nan if v is None else v for v in r] for r in x]
        if x and isinstance(x[0], list)
        else [np.nan if v is None else v for v in x],
        dtype=np.float64,
    )


def _same_stats(got: dict, want: dict) -> None:
    for name, values in want.items():
        g = np.asarray(got[name], dtype=np.float64)
        np.testing.assert_allclose(g, _arr(values), rtol=RTOL_F64, atol=0, err_msg=name)


def _same_frame(frame, want: dict, *, gene_idx_offset: int = 0) -> None:
    """Column for column; `gene_idx_offset=1` where R's `gene_idx` is 1-based."""
    assert frame.columns == list(want)
    for name, values in want.items():
        col = frame[name]
        if name == "gene_idx":
            assert [g + gene_idx_offset for g in col.to_list()] == values, name
        elif col.dtype.is_numeric():
            np.testing.assert_allclose(
                col.cast(float).fill_null(np.nan).to_numpy(),
                _arr(values),
                rtol=RTOL_F64,
                atol=0,
                err_msg=name,
            )
        else:
            assert col.to_list() == values, name


def _same_columns(got: np.ndarray, want: np.ndarray, err_msg: str) -> None:
    """Equal relative to the largest entry."""
    assert got.shape == want.shape, err_msg
    scale = np.abs(want).max()
    np.testing.assert_allclose(
        got, want, rtol=RTOL_F32, atol=ATOL_F32 * scale, err_msg=err_msg
    )


def _same_pca(sc: bv.SingleCells, want: dict, name: str) -> None:
    assert sc.cache.pca_singular_vals is not None
    assert sc.cache.pca_factors is not None
    assert sc.cache.pca_loadings is not None
    np.testing.assert_allclose(
        sc.cache.pca_singular_vals,
        want["singular_values"],
        rtol=RTOL_F32,
        err_msg=f"{name}: singular values",
    )
    _same_columns(sc.cache.pca_factors, _arr(want["factors"]), f"{name}: factors")
    _same_columns(sc.cache.pca_loadings, _arr(want["loadings"]), f"{name}: loadings")


#########
# Tests #
#########


@pytest.mark.parametrize("method", ["vst", "meanvarbin", "dispersion", "scran"])
def test_find_hvg_matches_r(ref, sc, method):
    want = ref["hvg"][method]
    pp.find_hvg(sc, hvg_no=ref["hvg_no"], hvg_params=HvgParams(method=method))
    assert sc.map.hvg is not None
    assert sc.map.hvg.tolist() == want["hvg"]
    var = sc.var(cols=list(want["var"]))
    _same_stats({c: var[c].to_numpy() for c in var.columns}, want["var"])


def test_get_hvg_data_matches_r(ref, sc):
    p = HvgParams(method="meanvarbin", bin_method="equal_freq")
    _same_frame(
        pp.get_hvg_data(sc, hvg_no=40, hvg_params=p),
        ref["hvg_equal_freq"],
        gene_idx_offset=1,
    )
    # few enough cells to take the in-memory subset path
    _same_frame(
        pp.get_hvg_data(sc, cell_ids=ref["subset_cells"], hvg_no=25),
        ref["hvg_data_subset"],
        gene_idx_offset=1,
    )
    assert sc.map.hvg is None


@pytest.mark.parametrize("comb", ["union", "average", "intersection"])
def test_find_hvg_batch_aware_matches_r(ref, sc, comb):
    want = ref["batch_aware"][comb]
    res = pp.find_hvg_batch_aware(sc, "batch_index", hvg_no=40, gene_comb_method=comb)
    assert res.hvg_gene_idx.tolist() == want["hvg_gene_idx"]
    assert res.hvg_genes.tolist() == want["hvg_genes"]
    _same_frame(res.hvg_data, want["hvg_data"])


def test_find_hvg_batch_aware_scran_matches_r(ref, sc):
    want = ref["batch_aware_scran"]
    res = pp.find_hvg_batch_aware(
        sc, "sample_id", hvg_no=40, hvg_params=HvgParams(method="scran")
    )
    assert res.hvg_gene_idx.tolist() == want["hvg_gene_idx"]
    _same_frame(res.hvg_data, want["hvg_data"])


PCA_RUNS: dict[
    str, tuple[Literal["auto", "covariance", "randomised", "exact"], bool, bool]
] = {
    "dense_covariance": ("covariance", False, False),
    "dense_exact": ("exact", False, False),
    "dense_randomised": ("randomised", False, False),
    "sparse_exact": ("exact", True, False),
    "sparse_randomised": ("randomised", True, False),
    "dense_clr": ("covariance", False, True),
}


@pytest.mark.parametrize("run", list(PCA_RUNS))
def test_calculate_pca_matches_r(ref, sc, run):
    pp.find_hvg(sc, hvg_no=100)
    assert sc.map.hvg is not None
    assert sc.map.hvg.tolist() == ref["pca_hvg"]
    solver, sparse, clr = PCA_RUNS[run]
    pp.calculate_pca(
        sc,
        ref["no_pcs"],
        pca_params=PcaParams(svd_solver=solver, clr=clr),
        sparse_svd=sparse,
        seed=7,
    )
    _same_pca(sc, ref["pca"][run], run)
    assert sc.cache_status()["stamped"].to_list() == [True]


def test_calculate_pca_default_matches_r(ref, sc):
    pp.find_hvg(sc, hvg_no=100)
    pp.calculate_pca(sc, ref["no_pcs"])
    _same_pca(sc, ref["pca"]["default"], "default")


@pytest.mark.parametrize("layer", ["norm", "raw"])
def test_run_magic_matches_r(ref, sc, layer):
    knn = ref["knn"]
    sc.cache.knn = bv.KnnGraph(
        indices=np.array(knn["indices"], dtype=np.uint32),
        distances=np.array(knn["dist"], dtype=np.float32),
        dist_metric=knn["metric"],
    )
    sc.stamp("knn")
    pp.run_magic(sc, ref["magic_features"], magic_params=MagicParams(layer=layer))
    want = ref["magic"][layer]
    assert sc.cache.magic is not None
    assert sc.cache.magic.features.tolist() == want["features"]
    np.testing.assert_allclose(
        sc.cache.magic.data, _arr(want["data"]), rtol=RTOL_F32, atol=ATOL_F32
    )
    status = sc.cache_status()
    assert status.filter(status["artefact"] == "magic")["parents"].to_list() == [
        [sc.cache.stamps["knn"].id]
    ]


def _residual_run(ref: dict, sc: bv.SingleCells, want: dict) -> None:
    fit = sc.cache.residual_fit
    assert isinstance(fit, pp.ResidualFit)
    assert fit.n_groups == want["n_groups"]
    assert fit.genes.tolist() == want["genes"]
    pp.find_hvg(
        sc, hvg_no=ref["residual_hvg_no"], hvg_params=HvgParams(method="residual")
    )
    assert sc.map.hvg is not None
    assert sc.map.hvg.tolist() == want["hvg"]
    np.testing.assert_allclose(
        sc.var(cols=["residual_variance"])["residual_variance"]
        .fill_null(np.nan)
        .to_numpy(),
        _arr(want["residual_variance"]),
        rtol=RTOL_F64,
    )
    pp.calculate_pca(
        sc,
        ref["no_pcs"],
        pca_params=PcaParams(normalise_variance=False, svd_solver="covariance"),
        sparse_svd=False,
        residuals=True,
    )
    _same_pca(sc, want["pca"], "residual")
    status = sc.cache_status()
    assert status["stale"].to_list() == [False, False]
    assert status["parents"].to_list()[0] == [sc.cache.stamps["residuals"].id]


def test_sctransform_matches_r(ref, sc, tmp_path):
    pp.fit_residuals(sc, method="sctransform")
    _residual_run(ref, sc, ref["sct"])

    corrected = pp.sct_corrected_counts(sc, dir_out=tmp_path / "py_corrected")
    assert isinstance(corrected, bv.SingleCells)
    r_corrected = bv.SingleCells(ref["corrected_dir"])
    assert corrected.dims == r_corrected.dims
    assert corrected.gene_ids().tolist() == r_corrected.gene_ids().tolist()
    assert corrected.cell_ids().tolist() == r_corrected.cell_ids().tolist()
    for layer in ("raw", "norm"):
        got = bv.get_counts(corrected, layer=layer).scipy_csr()
        want = bv.get_counts(r_corrected, layer=layer).scipy_csr()
        np.testing.assert_allclose(
            got.toarray(), want.toarray(), rtol=RTOL_F32, atol=ATOL_F32, err_msg=layer
        )
    assert corrected.obs().columns == r_corrected.obs().columns
    assert corrected.var().columns == r_corrected.var().columns


def test_sctransform_covariate_matches_r(ref, sc):
    pp.fit_residuals(sc, method="sctransform", covariate_columns=["nnz"])
    fit = sc.cache.residual_fit
    assert isinstance(fit, pp.ResidualFit)
    assert fit.covariate_names == ("nnz",)
    _residual_run(ref, sc, ref["sct_covariate"])


def test_analytic_pearson_grouped_matches_r(ref, sc):
    pp.fit_residuals(
        sc,
        method="analytic_pearson",
        group_column="sample_id",
        residual_params=AprParams(),
    )
    _residual_run(ref, sc, ref["apr_grouped"])


def test_r_reads_python_var(ref, sc, tmp_path):
    """R opens the Python directory and sees the HVG columns Python wrote."""
    pp.find_hvg(sc, hvg_no=ref["hvg_no"], hvg_params=HvgParams(method="dispersion"))
    out = tmp_path / "out.json"
    _run_r("read", str(sc.dir), str(out))
    got = json.loads(out.read_text())["var"]
    want = ref["hvg"]["dispersion"]["var"]
    _same_stats({k: got[k] for k in want}, want)
    assert got["gene_idx"] == list(range(1, sc.n_genes + 1))
