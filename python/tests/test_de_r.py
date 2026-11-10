"""DGE, marker, pseudobulk, NEBULA, Milo and MELD parity with the bixverse R
package.

R generates seeded synthetic data with six samples, plants a condition effect
(genes 40 to 44 tripled in the three treated samples) and an abundance shift
(most treated cell type 1 cells dropped), loads it into two directories and
runs its methods on one (`dev/sc_de_fixtures.R`), leaving the other
untouched. Python copies the untouched one, applies the same cell filter,
takes R's PCA scores for Milo and MELD, builds its own kNN on them and has to
reproduce R's results. Skipped when Rscript or the package is missing.

Both sides run the same Rust on the same inputs with the same seeds, so the
statistics agree to the digits jsonlite prints (measured: at most 8e-15
relative on every marker statistic, pseudobulk aggregate and edgeR column,
NEBULA coefficient, SE and test, Milo F test and MELD score). They are
compared at `RTOL`; gene order, group labels, worst rivals, ranks, NEBULA
convergence codes, Milo index cells, neighbourhoods and counts exactly, and
so are the FDR calls at 0.05. The one exception is Milo's k-th neighbour
distance, an f32 sum that differs from R's by up to one ulp (measured:
1.3e-7 relative; the cause, presumably a different reduction order in the R
build, is not established), so it and the k-distance spatial FDR it weights
(measured: 7.9e-9) are compared at `RTOL_F32`. Milo's `Nhood` is 0-based
here, R's 1-based.
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
from bixverse import de, graph
from bixverse.params import EdgerQlParams, MeldParams, NebulaParams

from .conftest import R_SKIP_REASON, r_available

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_de_fixtures.R"

RTOL = 1e-12
# Milo's k-th neighbour distances are f32 sums, and differ from R's by up to
# one f32 ulp; the spatial FDR inherits that.
RTOL_F32 = 1e-6
ALPHA = 0.05


pytestmark = pytest.mark.skipif(not r_available(), reason=R_SKIP_REASON)


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the DGE and differential abundance methods."""
    root = tmp_path_factory.mktemp("r_de")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), "write", str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched directory with R's cell filter, R's PCA and
    Python's own kNN on it."""
    directory = tmp_path_factory.mktemp("py") / "py_de"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    graph.find_neighbours(sc)
    return sc


###########
# Helpers #
###########


def _close(got: object, want: object, rtol: float = RTOL) -> None:
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
        rtol=rtol,
        atol=1e-300,
    )


def _same_frame(
    got: pl.DataFrame,
    want: dict,
    exact: tuple[str, ...],
    calls: tuple[str, ...] = (),
    loose: tuple[str, ...] = (),
) -> None:
    """Same columns, exact on `exact`, close on the rest (`loose` ones at
    `RTOL_F32`); the calls at `ALPHA` on `calls` exact."""
    assert got.columns == list(want)
    for col, values in want.items():
        g = got[col]
        if col in exact:
            assert g.to_list() == values, col
            continue
        w = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
        gv = g.cast(pl.Float64).fill_null(np.nan).to_numpy()
        tol = RTOL_F32 if col in loose else RTOL
        np.testing.assert_allclose(gv, w, rtol=tol, atol=1e-300, err_msg=col)
        if col in calls:
            np.testing.assert_array_equal(gv < ALPHA, w < ALPHA, err_msg=col)


###########
# Markers #
###########


def test_find_markers_matches_r(ref, sc):
    got = de.find_markers(sc, ref["cells_1"], ref["cells_2"])
    _same_frame(got, ref["markers"], exact=("gene_id",), calls=("fdr",))
    got = de.find_markers(
        sc, ref["cells_1"], ref["cells_2"], alternative="greater", min_prop=0.2
    )
    _same_frame(got, ref["markers_greater"], exact=("gene_id",), calls=("fdr",))


def test_find_all_markers_matches_r(ref, sc):
    got = de.find_all_markers(sc, "cell_grp")
    _same_frame(got, ref["all_markers"], exact=("grp", "gene_id"), calls=("fdr",))
    got = de.find_all_markers(sc, "sample_id", alternative="twosided", min_prop=0.1)
    _same_frame(
        got, ref["all_markers_twosided"], exact=("grp", "gene_id"), calls=("fdr",)
    )


@pytest.mark.parametrize(
    ("key", "kwargs"),
    [
        ("specific", {}),
        ("specific_ref", {"reference_group": "cell_type_2", "alternative": "twosided"}),
    ],
)
def test_find_specific_markers_matches_r(ref, sc, key, kwargs):
    got = de.find_specific_markers(sc, "cell_grp", **kwargs)
    assert got is not None
    want = ref[key]
    _same_frame(
        de.get_marker_summary(got),
        want["summary"],
        exact=("ref_grp", "gene_id", "worst_rival", "min_rank"),
        calls=("simes_fdr", "max_p_fdr"),
    )
    _same_frame(
        de.get_marker_comparisons(got),
        want["per_comparison"],
        exact=("ref_grp", "rival_grp", "gene_id"),
        calls=("fdr",),
    )


##############
# Pseudobulk #
##############


def _cell_list(sc: bv.SingleCells, samples: list[str]) -> dict[str, list[str]]:
    obs = sc.obs(cols=["cell_id", "sample_id"], filtered=True)
    return {
        s: obs.filter(pl.col("sample_id") == s)["cell_id"].to_list() for s in samples
    }


def test_pseudobulk_matches_r(ref, sc):
    cell_list = _cell_list(sc, ref["pb_samples"])
    raw = de.get_pseudobulked(sc, cell_list)
    assert raw.samples == tuple(ref["pb_samples"])
    np.testing.assert_array_equal(raw.counts, ref["pb_raw"])
    norm = de.get_pseudobulked(sc, cell_list, assay="norm")
    _close(norm.counts, ref["pb_norm"])
    sparse = de.get_pseudobulked(sc, cell_list, return_format="sparse")
    assert isinstance(sparse.counts, bv.CsrMatrix)
    np.testing.assert_array_equal(sparse.counts.indptr, ref["pb_sparse"]["p"])
    np.testing.assert_array_equal(sparse.counts.indices, ref["pb_sparse"]["j"])
    np.testing.assert_array_equal(sparse.counts.data, ref["pb_sparse"]["x"])


def test_pseudobulk_dge_matches_r(ref, sc):
    cell_list = _cell_list(sc, ref["pb_samples"])
    design = np.array(ref["pb_design"], dtype=np.float64)
    got = de.pseudobulk_dge(sc, cell_list, design)
    _same_frame(got, ref["pb_dge"], exact=("feature_id",), calls=("fdr",))
    got = de.pseudobulk_dge(
        sc,
        cell_list,
        design,
        edger_params=EdgerQlParams(norm_method="RLE", robust=True, legacy=True),
    )
    _same_frame(got, ref["pb_dge_robust"], exact=("feature_id",), calls=("fdr",))


##########
# NEBULA #
##########


def _same_nebula(got: de.ScNebula, want: dict) -> None:
    _same_frame(
        got.results,
        want["results"],
        exact=("gene_id", "convergence", "sigma_at_bound"),
        calls=("fdr",),
    )
    assert got.coefficients.columns[1:] == want["coef_names"]
    _close(got.coefficients.drop("gene_id").to_numpy(), want["coefficients"])
    _close(got.se.drop("gene_id").to_numpy(), want["se"])
    assert got.params["tested"] == want["tested"]


def test_nebula_matches_r(ref, sc):
    got = de.nebula(sc, "sample_id", "~condition", genes_to_use=ref["nebula_genes"])
    _same_nebula(got, ref["nebula"])


def test_nebula_covariate_matches_r(ref, sc):
    got = de.nebula(
        sc,
        "sample_id",
        "~ condition + age",
        coef="conditiontrt",
        genes_to_use=ref["nebula_genes"],
        nebula_params=NebulaParams(nebula_method="hl", shrink_dispersion=False),
    )
    _same_nebula(got, ref["nebula_age"])


def test_nebula_contrast_matches_r(ref, sc):
    got = de.nebula(
        sc,
        "sample_id",
        "~condition",
        contrast=[0.0, 1.0],
        genes_to_use=ref["nebula_genes"],
        cells_to_use=ref["nebula_cells"],
    )
    _same_nebula(got, ref["nebula_contrast"])


########
# Milo #
########


def test_knn_matches_r(ref, sc):
    assert sc.cache.knn is not None
    np.testing.assert_array_equal(sc.cache.knn.indices, ref["knn"])


@pytest.mark.parametrize("strategy", ["approximate", "bruteforce", "index"])
def test_milor_abundances_match_r(ref, sc, strategy):
    milo = de.get_milor_abundances(
        sc,
        "sample_id",
        milor_params=de.MiloRParams(
            k_refine=10,
            refinement_strategy=strategy,
            index_type="exhaustive" if strategy == "index" else "nndescent",
        ),
        seed=42,
    )
    assert milo is not None
    want = ref["milo"][strategy]
    np.testing.assert_array_equal(de.get_index_cells(milo), want["index_cell"])
    rows = np.repeat(
        np.arange(milo.nhoods.shape[0]), np.diff(milo.nhoods.indptr).astype(np.int64)
    )
    got_pairs = sorted(zip(milo.nhoods.indices.tolist(), rows.tolist(), strict=True))
    want_pairs = sorted(zip(want["nhoods_j"], want["nhoods_i"], strict=True))
    assert got_pairs == want_pairs
    np.testing.assert_array_equal(milo.sample_counts, want["sample_counts"])
    assert list(milo.samples) == want["samples"]
    _close(milo.spatial_dist, want["spatial_dist"], RTOL_F32)
    np.testing.assert_array_equal(milo.nhood_overlap, want["nhood_overlap"])


@pytest.fixture(scope="module")
def milo(sc):
    """The exhaustive-index neighbourhoods R tests."""
    milo = de.get_milor_abundances(
        sc,
        "sample_id",
        milor_params=de.MiloRParams(k_refine=10, index_type="exhaustive"),
    )
    assert milo is not None
    return milo


@pytest.fixture(scope="module")
def design_df(ref, milo):
    return pl.DataFrame(
        {"sample_id": list(milo.samples), "condition": ref["design_condition"]}
    )


def _nhoods_r(want: dict) -> dict:
    """R's table with `Nhood` 0-based."""
    return {**want, "Nhood": [n - 1 for n in want["Nhood"]]}


def test_test_nhoods_matches_r(ref, milo, design_df):
    tested = de.test_nhoods(milo, "~condition", design_df)
    got = de.get_differential_abundance_res(tested)
    assert got is not None
    _same_frame(
        got,
        _nhoods_r(ref["nhoods_info"]),
        exact=("Nhood",),
        calls=("FDR", "SpatialFDR"),
        loose=("SpatialFDR",),
    )


def test_test_nhoods_variants_match_r(ref, milo, design_df):
    got = de.test_nhoods(
        milo,
        "~condition",
        design_df,
        norm_method="logMS",
        legacy=False,
        robust=False,
        fdr_weighting="graph-overlap",
    ).nhoods_info
    assert got is not None
    _same_frame(
        got,
        _nhoods_r(ref["nhoods_info_overlap"]),
        exact=("Nhood",),
        calls=("FDR", "SpatialFDR"),
    )
    got = de.test_nhoods(
        milo, "~condition", design_df, min_mean=5, fdr_weighting="none"
    ).nhoods_info
    assert got is not None
    _same_frame(got, _nhoods_r(ref["nhoods_info_none"]), exact=("Nhood",))
    assert got["SpatialFDR"].null_count() == got.height


def test_add_nhoods_info_matches_r(ref, sc, milo, design_df):
    tested = de.test_nhoods(milo, "~condition", design_df)
    labels = sc.obs(cols=["cell_grp"], filtered=True)["cell_grp"]
    got = de.add_nhoods_info(tested, labels).nhoods_info
    assert got is not None
    _same_frame(
        got,
        _nhoods_r(ref["nhoods_annotated"]),
        exact=("Nhood", "majority_celltype"),
        loose=("SpatialFDR",),
    )


########
# MELD #
########


def test_meld_matches_r(ref, sc):
    cases: tuple[tuple[str, str, dict[str, Any]], ...] = (
        ("meld", "sample_id", {}),
        (
            "meld_lap",
            "condition",
            {
                "meld_params": MeldParams(
                    filter="laplacian",
                    lap_type="normalised",
                    beta=20,
                    normalise_indicators=False,
                ),
                "seed": 3,
            },
        ),
    )
    for key, col, kwargs in cases:
        got = de.meld(sc, col, **kwargs)
        assert got is not None
        want = ref[key]
        assert got.raw_scores.columns == ["cell_id", *want["samples"]]
        assert got.raw_scores["cell_id"].to_list() == want["cells"]
        _close(got.raw_scores.drop("cell_id").to_numpy(), want["raw"])
        _close(got.norm_scores.drop("cell_id").to_numpy(), want["norm"])
