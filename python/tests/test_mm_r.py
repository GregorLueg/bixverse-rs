"""Multi-modal parity with the bixverse R package.

R loads seeded synthetic RNA into two directories, the second as a
`SingleCellsMultiModal` with the first cells dropped, adds the seeded
synthetic ADT counts with their rows reversed and runs every multi-modal step
on them (`dev/sc_mm_fixtures.R`). Python copies the untouched directory,
applies the same filter, regenerates the same ADT counts and reruns: CLR in
both variants, DSB in four configurations, the ADT PCA (full and randomised),
WNN with two parameter sets, the ADT kNN and graph, Leiden on the ADT and WNN
graphs, PAGA and MAGIC over the WNN and ADT kNN. WNN and everything after it
start from R's RNA and ADT PCA, so they test the routing and the fusion, not
the PCA again. Skipped when Rscript or the package is missing.

Tolerances, measured against R 0.8.4. The `float64` outputs (CLR, DSB, the
raw ADT PCA) agree to a few ulps (max relative 2e-11, on near-zero PCA
scores) and are held to `RTOL` with `ATOL` for the near-zero entries. The
`float32` outputs (the cached PCA, kNN distances, weights, graph weights,
MAGIC) agree to the 15 digits jsonlite prints them with and are held to
`RTOL_F32`. Indices, labels and graph structure are exact.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import graph, mm, pp, traj
from bixverse.params import (
    DsbParams,
    KnnGraphDefaults,
    NeighboursParams,
    SyntheticDataAdtParams,
    WnnParams,
)

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_mm_fixtures.R"

RTOL = 1e-10
ATOL = 1e-12
RTOL_F32 = 1e-6
ATOL_F32 = 1e-7

DSB_ARRAYS = (
    "protein_background_mean",
    "protein_background_sd",
    "technical_component",
    "cellwise_background_mean",
)


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


###########
# Helpers #
###########


def _close(got, want, name: str, rtol: float = RTOL, atol: float = ATOL) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(
        [np.nan if v is None else v for v in np.ravel(want)], dtype=np.float64
    ).reshape(got.shape)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=name)


def _as_list(x) -> list:
    """R's `auto_unbox` turns length-one vectors into scalars."""
    return x if isinstance(x, list) else [x]


def _same_knn(got: bv.KnnGraph, want: dict) -> None:
    assert got.dist_metric == want["metric"]
    np.testing.assert_array_equal(got.indices, np.array(want["indices"]))
    _close(got.distances, want["dist"], "knn dist", RTOL_F32, ATOL_F32)


def _same_graph(got: bv.CsrMatrix, want: dict) -> None:
    assert got.shape == (want["n"], want["n"])
    np.testing.assert_array_equal(got.indptr, want["indptr"])
    np.testing.assert_array_equal(got.indices, want["indices"])
    _close(got.data, want["data"], "graph data", RTOL_F32, ATOL_F32)


############
# Fixtures #
############


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the multi-modal steps."""
    root = tmp_path_factory.mktemp("r_mm")
    _run_r("write", str(root), str(root / "reference.json"))
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def adt():
    """R's synthetic ADT counts, rows reversed as R hands them over."""
    data = bv.io.generate_single_cell_test_data_adt(
        SyntheticDataAdtParams(n_cells=600), seed=123
    )
    return (
        np.ascontiguousarray(data.counts[::-1]),
        data.obs["cell_id"].to_numpy()[::-1],
        data.var["protein_id"].to_list(),
    )


@pytest.fixture(scope="module")
def empty():
    """R's empty droplets: other synthetic ADT counts, scaled down."""
    data = bv.io.generate_single_cell_test_data_adt(
        SyntheticDataAdtParams(n_cells=300), seed=7
    )
    return np.floor(data.counts / 10)


@pytest.fixture
def sc(ref, adt, tmp_path):
    """A copy of R's untouched directory with R's filter and CLR ADT counts."""
    directory = tmp_path / "py_mm"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCellsMultiModal(directory)
    sc.set_cells_to_keep(ref["keep"])
    counts, barcodes, proteins = adt
    mm.add_adt_counts(sc, counts, barcodes, proteins)
    return sc


@pytest.fixture
def sc_pca(ref, sc):
    """`sc` with R's RNA and ADT PCA."""
    sc.cache.pca_factors = np.array(ref["rna_pca"], dtype=np.float32)
    sc.stamp("pca")
    sc.adt_cache.pca_factors = np.array(ref["adt_pca"]["factors"], dtype=np.float32)
    sc.stamp("pca", modality="adt")
    return sc


@pytest.fixture
def cell_info(sc) -> pl.DataFrame:
    return pl.DataFrame(
        {"cell_id": sc.cell_ids(filtered=True), "cell_idx": sc.cells_to_keep}
    )


#######
# ADT #
#######


def test_add_adt_counts_matches_r(ref, sc):
    a = sc.adt
    assert a is not None
    assert a.type == "CLR"
    assert a.cell_ids.tolist() == ref["adt_rows"]
    assert (a.cell_idx == sc.cells_to_keep).all()
    _close(a.norm_counts, ref["clr"], "clr")

    var_adt = sc.var(modality="adt")
    want = ref["var_adt"]
    assert [i + 1 for i in var_adt["feature_idx"].to_list()] == want["feature_idx"]
    assert var_adt["feature_id"].to_list() == want["feature_id"]
    _close(var_adt["nnz"], want["nnz"], "nnz")

    obs = sc.obs()
    want = ref["obs_adt"]
    assert [i + 1 for i in obs["cell_idx"].to_list()] == want["cell_idx"]
    for col in ("adt_nnz", "adt_lib_size"):
        assert obs[col].dtype == pl.Float64
        _close(obs[col], want[col], col)


@pytest.mark.parametrize(
    ("run", "kwargs"),
    [
        ("clr_seurat", {"seurat_clr": True, "clean_clr_counts": False}),
        ("clr_seurat_clean", {"seurat_clr": True, "percentile": 0.05}),
    ],
)
def test_clr_variants_match_r(ref, adt, cell_info, run, kwargs):
    counts, barcodes, proteins = adt
    res = mm.new_adt_counts_clr(counts, barcodes, proteins, cell_info, **kwargs)
    _close(res.norm_counts, ref[run], run)


@pytest.mark.parametrize(
    "run", ["kmeans_iso", "kmeans_no_iso", "empty_std", "empty_ms_clip"]
)
def test_dsb_matches_r(ref, adt, empty, cell_info, run):
    kwargs: dict[str, Any] = {
        "kmeans_iso": {"isotype_names": ref["isotypes"]},
        "kmeans_no_iso": {
            "dsb_params": DsbParams(use_isotype_controls=False, pseudocount=1.0)
        },
        "empty_std": {
            "empty_drops": empty,
            "isotype_names": ref["isotypes"],
            "seed": 3,
        },
        "empty_ms_clip": {
            "empty_drops": empty,
            "dsb_params": DsbParams(
                denoise_counts=False,
                use_isotype_controls=False,
                quantile_low=0.01,
                quantile_high=0.99,
            ),
            "scale_factor": "mean_subtract",
        },
    }[run]
    counts, barcodes, proteins = adt
    res = mm.new_adt_counts_dsb(counts, barcodes, proteins, cell_info, **kwargs)
    want = ref["dsb"][run]
    assert res.type == "DSB"
    assert res.dsb is not None
    _close(res.norm_counts, want["norm"], f"{run} norm")
    for key in DSB_ARRAYS:
        _close(getattr(res.dsb, key), _as_list(want[key]), f"{run} {key}")


#######
# PCA #
#######


@pytest.mark.parametrize(
    ("run", "no_pcs", "randomised", "seed"),
    [("adt_pca", 10, False, 42), ("adt_pca_rand", 5, True, 9)],
)
def test_calculate_pca_adt_matches_r(ref, sc, run, no_pcs, randomised, seed):
    mm.calculate_pca_adt(sc, no_pcs, randomised_svd=randomised, seed=seed)
    want = ref[run]
    c = sc.adt_cache
    assert c.pca_factors is not None
    assert c.pca_factors.dtype == np.float32
    _close(c.pca_factors, want["factors"], "factors", RTOL_F32, 1e-6)
    _close(c.pca_loadings, want["loadings"], "loadings", RTOL_F32, ATOL_F32)
    _close(c.pca_singular_vals, want["sv"], "sv", RTOL_F32, ATOL_F32)
    assert sc.check_state(["pca"], modality="adt") is True


#######
# WNN #
#######


@pytest.mark.parametrize("run", ["wnn", "wnn_alt"])
def test_generate_wnn_graph_matches_r(ref, sc_pca, run):
    kwargs: dict[str, Any] = {
        "wnn": {"wnn_params": WnnParams(knn_range=100)},
        "wnn_alt": {
            "no_embd_to_use_1": 8,
            "wnn_params": WnnParams(
                k_nn=15,
                knn_range=60,
                s_nn=12,
                sd_scale=1.5,
                kernel_power=2.0,
                cross_const=1e-3,
            ),
            "graph_params": KnnGraphDefaults(knn_sym="umap"),
            "seed": 5,
        },
    }[run]
    mm.generate_wnn_graph(sc_pca, **kwargs)
    want = ref[run]
    w = sc_pca.wnn
    assert w is not None
    assert w.cache.knn is not None
    assert w.cache.connectivities is not None
    _same_knn(w.cache.knn, want["knn"])
    _same_graph(w.cache.connectivities, want["graph"])
    weights = mm.get_wnn_weights(sc_pca)
    assert weights["cell_id"].to_list() == want["weights"]["cell_id"]
    for col in ("weight_modality_1", "weight_modality_2"):
        _close(weights[col], want["weights"][col], col, RTOL_F32, ATOL_F32)

    status = sc_pca.cache_status().filter(pl.col("modality") == "wnn")
    assert status["artefact"].to_list() == ["knn", "connectivities"]
    assert status["parents"].to_list()[0] == [
        sc_pca.cache.stamps["pca"].id,
        sc_pca.adt_cache.stamps["pca"].id,
    ]


####################
# Modality routing #
####################


def test_adt_neighbours_and_clusters_match_r(ref, sc_pca):
    graph.find_neighbours(
        sc_pca,
        modality="adt",
        neighbours_params=NeighboursParams(k=15, knn_method="exhaustive"),
    )
    assert sc_pca.adt_cache.knn is not None
    assert sc_pca.adt_cache.connectivities is not None
    assert sc_pca.cache.knn is None
    _same_knn(sc_pca.adt_cache.knn, ref["adt_knn"])
    _same_graph(sc_pca.adt_cache.connectivities, ref["adt_graph"])

    mm.generate_wnn_graph(sc_pca, wnn_params=WnnParams(knn_range=100))
    graph.find_clusters(sc_pca, modality="adt", name="adt_leiden")
    graph.find_clusters(sc_pca, modality="wnn", name="wnn_leiden", res=0.5)
    obs = sc_pca.obs(filtered=True)
    for col in ("adt_leiden", "wnn_leiden"):
        assert obs[col].to_list() == ref["clusters"][col], col


def test_paga_and_magic_over_other_modalities_match_r(ref, sc_pca):
    mm.generate_wnn_graph(sc_pca, wnn_params=WnnParams(knn_range=100))
    res = traj.run_paga(sc_pca, "cell_grp", modality="wnn")
    assert res is not None
    assert res.params["modality"] == "wnn"
    _close(res.connectivities, ref["paga_wnn"]["connectivities"], "paga")
    _close(res.connectivities_tree, ref["paga_wnn"]["tree"], "paga tree")

    graph.find_neighbours(
        sc_pca,
        modality="adt",
        neighbours_params=NeighboursParams(k=15, knn_method="exhaustive"),
    )
    pp.run_magic(sc_pca, ref["magic_genes"], modality="adt")
    assert sc_pca.cache.magic is not None
    _close(sc_pca.cache.magic.data, ref["magic_adt"], "magic", RTOL_F32, ATOL_F32)
    status = sc_pca.cache_status()
    magic = status.filter(pl.col("artefact") == "magic")
    assert magic["parents"].to_list() == [[sc_pca.adt_cache.stamps["knn"].id]]


###########
# Interop #
###########


def test_r_reads_python_adt_tables(ref, sc, tmp_path):
    """R opens the Python directory as a SingleCellsMultiModal and finds
    var_adt and the obs ADT columns with the types it writes itself."""
    out = tmp_path / "out.json"
    _run_r("read", str(sc.dir), str(out))
    got = json.loads(out.read_text())
    assert got["var_adt_class"] == ref["var_adt_class"]
    assert got["var_adt"]["feature_id"] == ref["var_adt"]["feature_id"]
    assert got["var_adt"]["feature_idx"] == ref["var_adt"]["feature_idx"]
    _close(got["var_adt"]["nnz"], ref["var_adt"]["nnz"], "nnz")
    for col in ("adt_nnz", "adt_lib_size"):
        assert got["obs_class"][col] == ref["obs_adt_class"][col]
        _close(got["obs"][col], ref["obs_adt"][col], col)
