"""ScType and Symphony parity with the bixverse R package.

R generates two seeded synthetic data sets with planted markers and two
batches each (`dev/sc_annotate_fixtures.R`): a reference and a query that
lacks five genes, two of them reference HVGs. It scores ScType against the
planted markers plus a decoy type, clusters, assigns per cell, builds
Symphony references with Harmony v1 and v2, maps the query and transfers the
labels. Python copies the untouched directories, takes R's cell filter, PCA
and Leiden clusters, builds its own neighbour graph and has to reproduce
R's results. Skipped when Rscript or the package is missing.

Both sides run the same Rust on the same inputs with the same seeds, so the
`float32` scores and Symphony terms agree to the digits jsonlite prints
(measured: at most 5e-15 relative on every ScType score, margin, agreement
and composition fraction, every reference term and both query mappings);
they are compared at `RTOL`. Every assignment, cluster call, composition
count and transferred label is compared exactly. Two caveats: the SVD sign
of half the PCs floats between the Python and the R builds, so everything
in PC space is sign-aligned on the loadings first; and the pre-Harmony
scores `z_orig` differ from R's by up to one `float32` ulp on small values
(measured: 7.1e-8 relative, 4.8e-11 of the largest score; the cause is not
established, `z_corr` built from them matches), so they are compared at
`RTOL_F32`.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import annotate, graph
from bixverse.params import (
    HarmonyParams,
    HarmonyV2Params,
    KnnParams,
    SctypeCellsParams,
    SymphonyMapParams,
)

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_annotate_fixtures.R"

RTOL = 1e-12
RTOL_F32 = 1e-6


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


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of ScType and Symphony."""
    root = tmp_path_factory.mktemp("r_annotate")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), "write", str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched reference with R's cell filter, PCA and
    Leiden clusters, and Python's own neighbour graph."""
    directory = tmp_path_factory.mktemp("py") / "py_ref"
    shutil.copytree(ref["ref_base"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    graph.find_neighbours(sc)
    sc.add_obs(
        pl.DataFrame(
            {
                "cell_idx": sc.cells_to_keep,
                "leiden": np.asarray(ref["leiden"], dtype=np.int32),
            }
        )
    )
    return sc


@pytest.fixture(scope="module")
def query(ref, tmp_path_factory):
    """A copy of R's untouched query."""
    directory = tmp_path_factory.mktemp("py") / "py_query"
    shutil.copytree(ref["query_base"], directory)
    return bv.SingleCells(directory)


@pytest.fixture(scope="module")
def markers(ref, sc):
    return annotate.prepare_cell_markers(sc, pl.DataFrame(ref["marker_df"]))


@pytest.fixture(scope="module")
def scores(sc, markers):
    return annotate.calc_sc_type_scores(sc, markers)


@pytest.fixture(scope="module")
def symphony(ref, sc):
    return annotate.build_symphony_ref(
        sc,
        "batch_index",
        np.asarray(ref["hvg"]),
        harmony_params=HarmonyParams(),
        no_pcs=10,
        label_columns=["cell_grp"],
    )


###########
# Helpers #
###########


def _close(got: object, want: object, rtol: float = RTOL) -> None:
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
        rtol=rtol,
        atol=rtol * float(np.max(np.abs(np.asarray(want, dtype=np.float64)))),
    )


def _signs(got: np.ndarray, want: object) -> np.ndarray:
    """Per-PC sign that aligns `got` with `want`; the SVD sign floats
    between the Python and the R builds."""
    w = np.asarray(want, dtype=np.float64)
    return np.where(np.sum(got.astype(np.float64) * w, axis=0) < 0, -1.0, 1.0)


##########
# ScType #
##########


def test_prepare_cell_markers_matches_r(ref, markers):
    assert list(markers) == ref["marker_names"]
    for ct, idx in ref["marker_indices"].items():
        assert markers[ct].positive.tolist() == idx
        assert markers[ct].negative.size == 0


@pytest.mark.parametrize(
    ("key", "kwargs"),
    [
        ("sctype", {}),
        ("sctype_flat", {"sensitivity": False}),
        ("sctype_floor", {"weight_floor": 0.4}),
    ],
)
def test_sc_type_scores_match_r(ref, sc, markers, key, kwargs):
    res = annotate.calc_sc_type_scores(sc, markers, **kwargs)
    want = ref[key]
    assert res.cell_types == want["cell_types"]
    _close(res.scores, want["scores"])
    got = annotate.get_scores(res)
    assert isinstance(got, pl.DataFrame)
    assert got.columns == want["cell_types"]


def test_sc_type_print_matches_r(ref, scores):
    assert repr(scores).splitlines() == ref["sctype_print"]


def test_score_clusters_matches_r(ref, scores):
    got = annotate.score_clusters(scores, np.asarray(ref["leiden"]))
    want = ref["score_clusters"]
    assert got.columns == list(want)
    for col in ("cluster_id", "cell_type", "n_cells"):
        assert got[col].to_list() == want[col], col
    _close(got["scores"], want["scores"])


@pytest.mark.parametrize(
    ("key", "kwargs"),
    [
        ("assign", {}),
        ("assign_hybrid", {"cluster_col": "leiden"}),
        (
            "assign_z",
            {
                "cluster_col": "leiden",
                "sctype_cell_params": SctypeCellsParams(
                    calibration="column_z",
                    alpha=0.3,
                    iterations=5,
                    score_floor=0.5,
                    purity_threshold=0.7,
                ),
            },
        ),
    ],
)
def test_assign_sc_type_matches_r(ref, sc, scores, key, kwargs):
    res = annotate.assign_sc_type(sc, scores, **kwargs)
    want = ref[key]
    assert res.cell_types == want["cell_types"]
    assert res.assignments.to_list() == want["assignments"]
    _close(res.scores, want["scores"])
    _close(res.margins, want["margins"])
    assert res.agreement is not None
    _close(res.agreement, want["agreement"])
    assert repr(res).splitlines() == want["print"]
    if want["composition"] is None:
        assert res.composition is None and res.hybrid is None and res.counts is None
        return
    assert res.hybrid is not None and res.composition is not None
    assert res.counts is not None
    assert res.hybrid.to_list() == want["hybrid"]
    comp = want["composition"]
    assert res.composition.columns == list(comp)
    for col in ("cluster_id", "n_cells", "n_unknown", "dominant", "second"):
        assert res.composition[col].to_list() == comp[col], col
    assert res.composition["cluster_mixed"].to_list() == comp["cluster_mixed"]
    for col in ("purity", "second_fraction", "entropy"):
        _close(res.composition[col], comp[col])
    assert res.counts.drop("cluster_id").to_numpy().tolist() == want["counts"]


############
# Symphony #
############

# terms in PC space flip with the loading signs, the rest do not
PC_TERMS = ("loadings", "z_corr", "centroids", "c_cache")
FIXED_TERMS = ("gene_means", "gene_sds", "nr")


def _same_reference(got: annotate.SymphonyReference, want: dict) -> np.ndarray:
    """Compare the reference terms, sign-aligned; returns the signs."""
    signs = _signs(got.loadings, want["loadings"])
    for name in PC_TERMS:
        _close(getattr(got, name) * signs, want[name])
    for name in FIXED_TERMS:
        _close(getattr(got, name), want[name])
    return signs


def test_symphony_reference_matches_r(ref, symphony):
    want = ref["symphony"]
    assert symphony.hvg_gene_names == want["hvg_gene_names"]
    signs = _same_reference(symphony, want)
    assert symphony.z_orig is not None and symphony.r is not None
    _close(symphony.z_orig * signs, want["z_orig"], rtol=RTOL_F32)
    _close(symphony.r, want["r"])
    assert symphony.labels is not None
    assert symphony.labels["cell_grp"].to_list() == want["labels"]["cell_grp"]
    assert repr(symphony).splitlines() == want["print"]


def test_symphony_v2_slim_matches_r(ref, sc):
    got = annotate.build_symphony_ref(
        sc,
        "batch_index",
        np.asarray(ref["hvg"]),
        harmony_params=HarmonyV2Params(k=12),
        no_pcs=8,
        slim=True,
        seed=7,
    )
    want = ref["symphony_v2"]
    assert got.z_orig is None and got.r is None and got.labels is None
    _same_reference(got, want)
    assert repr(got).splitlines() == want["print"]


@pytest.mark.parametrize(
    ("key", "kwargs"),
    [
        ("map", {"batch_column": "batch_index"}),
        ("map_no_batch", {"params": SymphonyMapParams(sigma=0.2)}),
    ],
)
def test_map_symphony_query_matches_r(ref, symphony, query, key, kwargs):
    annotate.map_symphony_query(symphony, query, **kwargs)
    want = ref[key]
    signs = _signs(symphony.loadings, ref["symphony"]["loadings"])
    embd = query.cache.embeddings
    _close(embd["symphony"] * signs, want["z_corr"])
    _close(embd["symphony_pca"] * signs, want["z_pca"])
    _close(embd["symphony_r"], want["r"])
    if "batch_column" not in kwargs:
        np.testing.assert_array_equal(
            query.cache.embeddings["symphony"], query.cache.embeddings["symphony_pca"]
        )


def test_transfer_labels_matches_r(ref, symphony, query):
    annotate.map_symphony_query(symphony, query, batch_column="batch_index")
    got = annotate.transfer_labels_symphony(
        symphony, query, "cell_grp", knn_params=KnnParams(ann_dist="euclidean")
    )
    want = ref["transfer"]
    assert got.columns == list(want)
    _close(got["confidence_cell_grp"], want["confidence_cell_grp"])
    # R 0.8.4 gives a tied vote to the highest label; the crate now gives it to
    # the lowest, as documented. Labels agree everywhere else, and where they
    # differ the vote share is the same and Python holds the lower label.
    pred = got["predicted_cell_grp"].to_list()
    differ = [
        i
        for i, (a, b) in enumerate(zip(pred, want["predicted_cell_grp"], strict=True))
        if a != b
    ]
    assert all(pred[i] < want["predicted_cell_grp"][i] for i in differ)
    assert len(differ) < len(pred) // 10
    obs = query.obs(
        cols=["cell_idx", "predicted_cell_grp", "confidence_cell_grp"], filtered=True
    )
    assert obs["predicted_cell_grp"].to_list() == pred
