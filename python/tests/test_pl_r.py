"""Plot data extractor parity with the bixverse R package.

R loads seeded synthetic RNA into three directories, drops the first cells on
two of them and runs the upstream steps there (`dev/sc_pl_fixtures.R`): HVGs,
PCA, neighbours, Leiden, MAGIC, PAGA, Milo and bootstrapped meta cells on a
`SingleCells`, the PCA and the seeded synthetic ADT counts (rows reversed)
with their PCA on a `SingleCellsMultiModal`. Python copies the untouched
directory twice, applies the same filter and takes R's PCA, kNN, clusters,
MAGIC layer, PAGA and Milo results, so the tests check the extraction only.
The meta cells are regenerated from R's kNN, which phase 10 holds exact.
Skipped when Rscript or the package is missing.

Tolerances, measured against R 0.8.4: expression values, means, percents
and z-scores (store and in-memory) agree to 6e-14 absolute, 3e-14 relative,
and are held to `RTOL` / `ATOL`. Coordinates on the RNA PCA agree to 5e-15;
R's ADT PCA scores are `float64` and the Python cache holds `float32`, so
those agree to 1.2e-7 absolute and are held to `RTOL_COORD` / `ATOL_COORD`.
Labels, levels, row order and graph structure are exact.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import de, metacells, mm, traj
from bixverse import pl as bvpl
from bixverse.params import BtMetacellsParams, SyntheticDataAdtParams

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_pl_fixtures.R"

RTOL = 1e-12
ATOL = 1e-12
# R's PCA scores are float64, the Python cache holds them as float32
RTOL_COORD = 1e-6
ATOL_COORD = 1e-7


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


###########
# Helpers #
###########


def _as_list(x) -> list:
    """R's `auto_unbox` turns length-one vectors into scalars."""
    return x if isinstance(x, list) else [x]


def _is_coord(col: str) -> bool:
    return col.startswith("dim_") or col in ("x", "y", "xend", "yend")


def _same_frame(got: pl.DataFrame, want: dict) -> None:
    """Columns, their order, Enum levels and values of an R table."""
    data = {k: _as_list(v) for k, v in want["data"].items()}
    assert got.columns == list(data), (got.columns, list(data))
    for col, exp in data.items():
        s = got[col]
        if isinstance(s.dtype, pl.Enum):
            assert s.dtype.categories.to_list() == _as_list(want["levels"][col]), col
            s = s.cast(pl.String)
        if s.dtype == pl.Boolean:
            assert s.to_list() == exp, col
        elif s.dtype.is_numeric() and not all(isinstance(v, str) for v in exp):
            exp_arr = np.array([np.nan if v is None else v for v in exp], dtype=float)
            tol = (RTOL_COORD, ATOL_COORD) if _is_coord(col) else (RTOL, ATOL)
            np.testing.assert_allclose(
                s.cast(pl.Float64).to_numpy(),
                exp_arr,
                rtol=tol[0],
                atol=tol[1],
                err_msg=col,
            )
        else:
            assert s.cast(pl.String).to_list() == [
                None if v is None else str(v) for v in exp
            ], col


def _sorted_edges(frame: pl.DataFrame, key: dict) -> pl.DataFrame:
    return (
        frame.with_columns(
            _a=pl.col("from").replace_strict(key), _b=pl.col("to").replace_strict(key)
        )
        .sort("_a", "_b")
        .drop("_a", "_b")
    )


def _same_graph(got: bvpl.GraphPlotData, want: dict, nhood: bool = False) -> None:
    nodes, edges = want["nodes"], want["edges"]
    if nhood:
        for d in (nodes["data"], edges["data"]):
            for col in ("Nhood", "from", "to"):
                if col in d:
                    d[col] = [v - 1 for v in _as_list(d[col])]
    _same_frame(got.nodes, nodes)
    if nhood:
        key = {v: v for v in got.nodes["Nhood"].to_list()}
    else:
        key = {v: i for i, v in enumerate(got.nodes["cluster"].cast(pl.String))}
    want_edges = pl.DataFrame(
        {k: _as_list(v) for k, v in edges["data"].items()},
        schema_overrides={
            "from": got.edges.schema["from"],
            "to": got.edges.schema["to"],
        },
    )
    _same_frame(
        _sorted_edges(got.edges, key),
        {"data": _sorted_edges(want_edges, key).to_dict(as_series=False), "levels": {}},
    )
    assert got.embedding == want["embedding"]


############
# Fixtures #
############


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the extractors."""
    root = tmp_path_factory.mktemp("r_pl")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched directory with R's filter, PCA, kNN, Leiden
    and MAGIC layer."""
    directory = tmp_path_factory.mktemp("py_pl") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    sc.cache.knn = bv.KnnGraph(
        indices=np.array(ref["knn"]["indices"], dtype=np.uint32),
        distances=np.array(ref["knn"]["dist"], dtype=np.float32),
        dist_metric=ref["knn"]["metric"],
    )
    sc.stamp("knn", parents=["pca"])
    sc.cache.magic = bv.MagicLayer(
        data=np.array(ref["magic"], dtype=np.float32),
        features=np.array(ref["magic_genes"]),
    )
    sc.stamp("magic", parents=["knn"])
    sc.add_obs(pl.DataFrame({"leiden_clustering": [int(x) for x in ref["leiden"]]}))
    return sc


@pytest.fixture(scope="module")
def mmo(ref, tmp_path_factory):
    """The multi-modal twin, with R's RNA and ADT PCA."""
    directory = tmp_path_factory.mktemp("py_pl") / "mm"
    shutil.copytree(ref["base_dir"], directory)
    obj = bv.SingleCellsMultiModal(directory)
    obj.set_cells_to_keep(ref["keep"])
    data = bv.io.generate_single_cell_test_data_adt(
        SyntheticDataAdtParams(n_cells=600), seed=123
    )
    mm.add_adt_counts(
        obj,
        np.ascontiguousarray(data.counts[::-1]),
        data.obs["cell_id"].to_numpy()[::-1],
        data.var["protein_id"].to_list(),
    )
    obj.cache.pca_factors = np.array(ref["mm_pca"], dtype=np.float32)
    obj.stamp("pca")
    obj.adt_cache.pca_factors = np.array(ref["adt_pca"], dtype=np.float32)
    obj.stamp("pca", modality="adt")
    return obj


@pytest.fixture(scope="module")
def mc(ref, sc):
    """The bootstrapped meta cells with R's `mc_grp`."""
    res = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=40, k=10)
    )
    assert res is not None
    res.add_obs(pl.DataFrame({"mc_grp": ref["mc_grp"]}))
    return res


@pytest.fixture(scope="module")
def paga(ref):
    """R's PAGA result."""
    r = ref["paga_res"]
    return traj.PagaRes(
        connectivities=np.array(r["connectivities"], dtype=np.float64),
        connectivities_tree=np.array(r["tree"], dtype=np.float64),
        clusters=tuple(r["clusters"]),
        sizes=pl.DataFrame(
            {
                "cluster": pl.Series(r["sizes"]["cluster"], dtype=pl.String),
                "n_cells": pl.Series(r["sizes"]["n_cells"], dtype=pl.UInt32),
            }
        ),
        params={"cluster_col": "leiden_clustering", "modality": "rna"},
    )


@pytest.fixture(scope="module")
def milo(ref):
    """R's tested Milo neighbourhoods, `Nhood` 0-based."""
    r = ref["milo_res"]
    i, j = np.array(r["nhoods_i"]), np.array(r["nhoods_j"])
    order = np.lexsort((j, i))
    i, j = i[order], j[order]
    indptr = np.zeros(r["n_cells"] + 1, dtype=np.uint32)
    np.cumsum(np.bincount(i, minlength=r["n_cells"]), out=indptr[1:])
    info = dict(r["nhoods_info"])
    info["Nhood"] = [v - 1 for v in info["Nhood"]]
    return de.MiloR(
        nhoods=bv.CsrMatrix(
            np.ones(i.size, dtype=np.float32),
            j.astype(np.uint32),
            indptr,
            (r["n_cells"], r["n_nhoods"]),
        ),
        sample_counts=np.zeros((r["n_nhoods"], 1)),
        samples=("s",),
        spatial_dist=np.zeros(r["n_nhoods"]),
        nhood_overlap=np.zeros(r["n_nhoods"]),
        params={"index_cell": np.array(r["index_cell"], dtype=np.int64)},
        nhoods_info=pl.DataFrame(
            {
                k: [np.nan if v is None else v for v in vals] if k != "Nhood" else vals
                for k, vals in info.items()
            }
        ),
    )


##############
# Embeddings #
##############


def test_embedding_matches_r(ref, sc, mmo):
    _same_frame(
        bvpl.extract_embedding_data(sc, "pca", obs_cols=["cell_grp", "lib_size"]),
        ref["embedding"],
    )
    _same_frame(
        bvpl.extract_embedding_data(mmo, "pca", modality="adt"), ref["embedding_adt"]
    )


##############
# Expression #
##############


def test_gene_expression_matches_r(ref, sc):
    with pytest.warns(UserWarning, match="could not be matched"):
        got = bvpl.extract_gene_expression(
            sc, [*ref["genes"], "not_a_gene"], obs_cols=["cell_grp"]
        )
    _same_frame(got, ref["expr"])
    _same_frame(
        bvpl.extract_gene_expression(sc, ref["genes"], scale=True, clip=1.5),
        ref["expr_scaled"],
    )
    magic_genes = [ref["magic_genes"][3], ref["magic_genes"][1]]
    _same_frame(
        bvpl.extract_gene_expression(sc, magic_genes, layer="magic", scale=True),
        ref["expr_magic"],
    )


def test_adt_expression_matches_r(ref, mmo):
    _same_frame(
        bvpl.extract_gene_expression(
            mmo, ref["proteins"], obs_cols=["cell_grp"], modality="adt"
        ),
        ref["expr_adt"],
    )
    _same_frame(
        bvpl.extract_gene_expression(
            mmo, ref["proteins"], scale=True, clip=2.0, modality="adt"
        ),
        ref["expr_adt_scaled"],
    )


def test_dot_plot_data_matches_r(ref, sc, mmo):
    _same_frame(bvpl.extract_dot_plot_data(sc, ref["genes"], "cell_grp"), ref["dot"])
    _same_frame(
        bvpl.extract_dot_plot_data(sc, ref["genes"], "sample_id", scale_exp=False),
        ref["dot_raw"],
    )
    _same_frame(
        bvpl.extract_dot_plot_data(mmo, ref["proteins"], "cell_grp", modality="adt"),
        ref["dot_adt"],
    )


def test_violin_feature_and_pair_match_r(ref, sc, mmo):
    _same_frame(
        bvpl.extract_gene_violin_data(sc, ref["genes"], "cell_grp", scale=True),
        ref["violin"],
    )
    _same_frame(
        bvpl.extract_feature_plot_data(sc, ref["genes"][:2], "pca"), ref["feature"]
    )
    _same_frame(
        bvpl.extract_feature_plot_data(
            mmo,
            ref["proteins"][:2],
            "pca",
            expr_modality="adt",
            embd_modality="adt",
        ),
        ref["feature_adt"],
    )
    _same_frame(
        bvpl.extract_feature_plot_data(
            mmo, [ref["genes"][0]], "pca", embd_modality="adt"
        ),
        ref["feature_cross"],
    )
    _same_frame(
        bvpl.extract_feature_pair(
            sc, ref["genes"][0], ref["genes"][1], obs_cols=["cell_grp"]
        ),
        ref["pair"],
    )
    _same_frame(
        bvpl.extract_feature_pair(
            mmo, f"{ref['genes'][0]}_rna", f"{ref['proteins'][0]}_adt", scale=True
        ),
        ref["pair_cross"],
    )


def test_subset_and_metacells_match_r(ref, sc, mc):
    sub = bv.SingleCellsSubset(sc, "cell_grp", ref["subset_group"])
    _same_frame(
        bvpl.extract_gene_expression(sub, ref["genes"], obs_cols=["sample_id"]),
        ref["subset_expr"],
    )
    _same_frame(
        bvpl.extract_dot_plot_data(sub, ref["genes"], "sample_id"), ref["subset_dot"]
    )
    _same_frame(
        bvpl.extract_gene_expression(mc, ref["genes"], obs_cols=["mc_grp"], scale=True),
        ref["mc_expr"],
    )
    _same_frame(bvpl.extract_dot_plot_data(mc, ref["genes"], "mc_grp"), ref["mc_dot"])


##########
# Graphs #
##########


def test_paga_plot_data_matches_r(ref, sc, paga):
    _same_graph(bvpl.extract_paga_plot_data(sc, paga, embedding="pca"), ref["paga"])
    _same_graph(
        bvpl.extract_paga_plot_data(
            sc,
            paga,
            embedding="pca",
            tree_only=True,
            centroid="mean",
            threshold=0.05,
            node_stat_col="lib_size",
        ),
        ref["paga_tree"],
    )
    with pytest.raises(ValueError, match="PAGA was run on"):
        bvpl.extract_paga_plot_data(sc, paga, embedding="pca", cluster_col="cell_grp")


def test_milo_plot_data_matches_r(ref, sc, milo):
    _same_graph(
        bvpl.extract_milo_plot_data(sc, milo, embedding="pca"), ref["milo"], nhood=True
    )
    _same_graph(
        bvpl.extract_milo_plot_data(sc, milo, embedding="pca", alpha=0.9, overlap=3),
        ref["milo_alt"],
        nhood=True,
    )
