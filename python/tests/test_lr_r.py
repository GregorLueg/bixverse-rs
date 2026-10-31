"""Ligand-receptor and DIALOGUE parity with the bixverse R package.

R loads seeded synthetic data, drops the first cells and runs the NicheNet
chain (influence in two parameter sets, activities, per-cluster expression,
prioritisation in four weightings) on seeded networks, CellPhoneDB in all
three methods on a synthetic database zipped as the downloader would fetch
it, and DIALOGUE on the seeded DIALOGUE fixture over `SingleCells` (with the
quality column, and with the library size and the HVGs), a
`SingleCellsSubset` and bootstrapped meta cells (`dev/sc_lr_fixtures.R`).
Python copies the untouched directories, applies the same filter, rebuilds
the same inputs and reruns. Skipped when Rscript or the package is missing.

Tolerances, measured against R 0.8.4. Both sides run the same Rust on the
same inputs, so ids, gates, ranks, kept features, verdict and signature
genes are exact, and every float agrees to the 15 digits jsonlite prints
(max relative 5e-15), held to `RTOL`. The subunit order within a complex is
not compared: R sorts it in the session locale. Checked once by hand on the
real CellPhoneDB v5 download: both flatten it to the same 2911 rows.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import dialogue, lr, metacells, pp
from bixverse.params import (
    BtMetacellsParams,
    CellphonedbParams,
    DialogueHlmParams,
    DialoguePmdParams,
    DialogueRefineParams,
    LigandTargetParams,
)

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_lr_fixtures.R"

RTOL = 1e-12
ATOL = 1e-12


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
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _close(got, want, name: str, rtol: float = RTOL, atol: float = ATOL) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(
        [np.nan if v is None else v for v in np.ravel(want)], dtype=np.float64
    ).reshape(got.shape)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=name)


def _frame(d: dict) -> pl.DataFrame:
    return pl.DataFrame({k: _as_list(v) for k, v in d.items()}, strict=False)


def _same_frame(got: pl.DataFrame, want: dict, name: str) -> None:
    """Columns and order as R's; floats close, everything else exact."""
    assert got.columns == list(want), name
    for col in got.columns:
        w = _as_list(want[col])
        g = got[col]
        if g.dtype.is_float():
            _close(g.to_numpy(), w, f"{name}.{col}")
        else:
            assert g.to_list() == w, f"{name}.{col}"


def _knn(want: dict) -> bv.KnnGraph:
    return bv.KnnGraph(
        indices=np.array(want["indices"], dtype=np.uint32),
        distances=np.array(want["dist"], dtype=np.float32),
        dist_metric=want["metric"],
    )


############
# Fixtures #
############


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the ligand-receptor and DIALOGUE steps."""
    root = tmp_path_factory.mktemp("r_lr")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads((root / "reference.json").read_text())
    out["root"] = str(root)
    return out


@pytest.fixture(scope="module")
def sc(ref, tmp_path_factory):
    """A copy of R's untouched store with R's filter."""
    directory = tmp_path_factory.mktemp("py_lr") / "sc"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    return sc


@pytest.fixture(scope="module")
def networks(ref):
    return _frame(ref["ppi"]), _frame(ref["grn"])


@pytest.fixture(scope="module")
def influences(ref, networks):
    ppi, grn = networks
    seeds = {k: _as_list(v) for k, v in ref["seeds"].items()}
    params = {
        "default": LigandTargetParams(),
        "tuned": LigandTargetParams(
            ltf_cutoff=0.0,
            lr_sig_hub=0.5,
            gr_hub=0.5,
            secondary_targets=True,
            topology_correction=True,
        ),
    }
    return {
        k: lr.generate_ligand_target_influence(seeds, ppi, grn, p)
        for k, p in params.items()
    }


@pytest.fixture(scope="module")
def dsc(ref, tmp_path_factory):
    """A copy of R's untouched DIALOGUE store."""
    directory = tmp_path_factory.mktemp("py_dlg") / "sc"
    shutil.copytree(ref["dlg_base_dir"], directory)
    return bv.SingleCells(directory)


@pytest.fixture(scope="module")
def dlg_data():
    return bv.io.generate_dialogue_test_data(seed=11)


PMD = DialoguePmdParams(k=2, n_permutations=20, abn_c=10, n_genes=12)


def _zero_based(want: dict) -> dict:
    """R numbers programmes from 1; Python from 0."""
    return {**want, "programme": [p - 1 for p in _as_list(want["programme"])]}


def _same_dialogue(got: dialogue.DialogueResult, want: dict) -> None:
    _same_frame(got.programmes, _zero_based(want["programmes"]), "programmes")
    assert got.cell_types == tuple(_as_list(want["cell_types"]))
    assert got.source_class == want["source_class"]
    assert got.shared_samples == tuple(_as_list(want["shared_samples"]))
    assert [list(t) for t in got.mcp_cell_types] == [
        _as_list(t) for t in want["mcp_cell_types"]
    ]
    assert {k: list(v) for k, v in got.kept_features.items()} == {
        k: _as_list(v) for k, v in want["kept_features"].items()
    }
    for attr in ("scores", "cca_scores"):
        for ct, w in want[attr].items():
            g = getattr(got, attr)[ct]
            assert g["cell_id"].to_list() == _as_list(w["rows"])
            _close(g.drop("cell_id").to_numpy(), w["m"], f"{attr}.{ct}")
    for ct, w in want["ws"].items():
        g = got.ws[ct]
        assert g["feature"].to_list() == _as_list(w["rows"])
        _close(g.drop("feature").to_numpy(), w["m"], f"ws.{ct}")
    _close(
        got.refit_fidelity.drop("cell_type").to_numpy(), want["refit_fidelity"], "fid"
    )
    _same_frame(got.verdicts, _zero_based(want["verdicts"]), "verdicts")
    _same_frame(got.signatures, _zero_based(want["signatures"]), "signatures")


############
# NicheNet #
############


@pytest.mark.parametrize("which", ["default", "tuned"])
def test_influence_matches_r(ref, influences, which):
    want = ref["influence"][which]
    got = influences[which]
    assert list(got.ligand_names) == want["rows"]
    assert list(got.gene_ids) == want["cols"]
    _close(lr.get_influence(got), want["m"], which)


def test_unnamed_seeds_join_with_plus(ref, networks):
    ppi, grn = networks
    seeds = [_as_list(v) for v in ref["seeds"].values()]
    got = lr.generate_ligand_target_influence([seeds[0], seeds[8]], ppi, grn)
    assert list(got.ligand_names) == ref["unnamed_rows"]


@pytest.mark.parametrize("which", ["default", "tuned"])
def test_ligand_activity_matches_r(ref, influences, which):
    got = lr.ligand_activity_scores(
        influences[which],
        {k: _as_list(v) for k, v in ref["gene_sets"].items()},
        background=ref["background"],
    )
    _same_frame(got, ref["activity"][which], which)


def test_ligand_activity_single_set_matches_r(ref, influences):
    got = lr.ligand_activity_scores(
        influences["default"], _as_list(ref["gene_sets"]["degs"])
    )
    _same_frame(got, ref["activity_single"], "single")


def test_expression_info_matches_r(ref, sc):
    genes = (
        _as_list(ref["lr_network"]["ligand"])[:10]
        + _as_list(ref["lr_network"]["receptor"])[:10]
    )
    with pytest.warns(UserWarning, match="Dropping 1 gene"):
        got = lr.compute_expression_info(sc, "cell_grp", [*genes, "NOTAGENE"])
    _same_frame(got, ref["expr_info"], "expr_info")
    got = lr.compute_expression_info(
        sc,
        "cell_grp",
        genes,
        condition_colname="batch_index",
        condition_oi="1",
    )
    _same_frame(got, ref["expr_info_cond"], "expr_info_cond")


@pytest.mark.parametrize("case", ["one_condition", "case_control", "partial", "ties"])
def test_prioritise_matches_r(ref, sc, influences, case):
    lr_net = _frame(ref["lr_network"])
    genes = lr_net["ligand"].to_list()[:10] + lr_net["receptor"].to_list()[:10]
    expression_info = lr.compute_expression_info(sc, "cell_grp", genes)
    act = lr.ligand_activity_scores(
        influences["default"],
        {k: _as_list(v) for k, v in ref["gene_sets"].items()},
        background=ref["background"],
    )
    la = act.filter(pl.col("gene_set") == "degs").select("ligand", "aupr_corrected")
    cond = _frame(ref["condition_de"]) if case in ("case_control", "partial") else None
    weights = {"partial": ref["weights_partial"], "ties": ref["weights_ties"]}.get(case)
    got = lr.prioritise_interactions(
        _frame(ref["celltype_de"]),
        expression_info,
        la,
        lr_net,
        ["cell_type_1", "cell_type_2"],
        ["cell_type_3", "cell_type_1"],
        condition_de=cond,
        weights=weights,
        scenario="one_condition" if case == "one_condition" else "case_control",
    )
    _same_frame(got, ref["prio"][case], case)


###############
# CellPhoneDB #
###############


@pytest.fixture(scope="module")
def lr_db(ref):
    return lr.get_cellphonedb_db(ref["cpdb_dir"], verbose=False)


def test_cellphonedb_db_matches_r(ref, lr_db):
    want = ref["lr_db"]
    assert lr_db.columns == list(want)
    for col in lr_db.columns:
        w = _as_list(want[col])
        if col.startswith("genes_"):
            # R sorts the subunits in the session locale, Python by code point
            assert [sorted(g) for g in lr_db[col].to_list()] == [
                sorted(_as_list(g)) for g in w
            ], col
        else:
            assert lr_db[col].to_list() == w, col


@pytest.mark.parametrize("case", ["simple", "statistical", "degs", "restricted"])
def test_cellphonedb_matches_r(ref, sc, lr_db, case):
    restricted = case == "restricted"
    with pytest.warns(UserWarning, match="Dropping"):
        got = lr.cellphonedb(
            sc,
            "cell_grp",
            lr_db,
            method="simple"
            if case == "simple"
            else "degs"
            if case == "degs"
            else "statistical",
            deg_table=_frame(ref["deg_table"]) if case == "degs" else None,
            senders=["cell_type_3", "cell_type_1"] if restricted else None,
            receivers=["cell_type_2"] if restricted else None,
            params=CellphonedbParams(n_perm=200, seed=7),
        )
    _same_frame(got, ref["cpdb"][case], case)


############
# DIALOGUE #
############


def test_dialogue_quality_matches_r(ref, dsc, dlg_data):
    got = dialogue.dialogue(
        dsc,
        "cell_grp",
        "sample_id",
        dlg_data.features,
        quality_col="cell_quality",
        gene_ids=dlg_data.var["gene_id"].to_list(),
        pmd_params=PMD,
    )
    _same_dialogue(got, ref["dialogue"]["quality"])


def test_dialogue_lib_size_and_hvg_match_r(ref, dsc, dlg_data):
    pp.find_hvg(dsc, hvg_no=150)
    assert dsc.map.hvg is not None
    assert dsc.map.hvg.tolist() == ref["dlg_hvg"]
    got = dialogue.dialogue(
        dsc,
        "cell_grp",
        "sample_id",
        dlg_data.features,
        pmd_params=PMD,
        hlm_params=DialogueHlmParams(satterthwaite=False),
    )
    _same_dialogue(got, ref["dialogue"]["lib_size"])


def test_dialogue_subset_matches_r(ref, dsc, dlg_data):
    obs = dsc.obs(cols=["cell_idx", "sample_id"])
    dsc.add_obs(
        obs.select(
            "cell_idx",
            dlg_split=pl.when(pl.col("sample_id").is_in(ref["dlg_split_samples"]))
            .then(pl.lit("in"))
            .otherwise(pl.lit("out")),
        )
    )
    sub = bv.SingleCellsSubset(dsc, "dlg_split", "in")
    got = dialogue.dialogue(
        sub,
        "cell_grp",
        "sample_id",
        dlg_data.features,
        quality_col="cell_quality",
        gene_ids=dlg_data.var["gene_id"].to_list(),
        pmd_params=PMD,
        refine_params=DialogueRefineParams(support_p=0.2),
    )
    _same_dialogue(got, ref["dialogue"]["subset"])


@pytest.fixture(scope="module")
def mc(ref, dsc):
    """Bootstrapped meta cells with R's labels, quality and features."""
    dsc.cache.pca_factors = np.array(ref["dlg_pca"], dtype=np.float32)
    dsc.stamp("pca")
    dsc.cache.knn = _knn(ref["dlg_knn"])
    dsc.stamp("knn", parents=["pca"])
    mc = metacells.generate_bt_meta_cells(
        dsc, params=BtMetacellsParams(target_no_metacells=150, k=10)
    )
    assert mc is not None
    assert mc.cell_ids().tolist() == ref["mc_ids"]
    root = Path(ref["root"])
    mc.add_obs(
        pl.DataFrame(
            {
                "cell_grp": ref["mc_cell_grp"],
                "sample_id": ref["mc_sample"],
                "cell_quality": np.fromfile(root / "mc_quality.f64", dtype="<f8"),
            }
        )
    )
    return mc


@pytest.fixture(scope="module")
def mc_features(ref, mc) -> dict[str, pl.DataFrame]:
    n, k = ref["mc_feature_dim"]
    m = np.fromfile(Path(ref["root"]) / "mc_features.f64", dtype="<f8").reshape(n, k)
    frame = pl.DataFrame(
        {
            "meta_cell_id": mc.cell_ids().tolist(),
            **{c: m[:, j] for j, c in enumerate(ref["mc_feature_names"])},
        }
    )
    labels = pl.Series(ref["mc_cell_grp"])
    return {ct: frame.filter(labels == ct) for ct in sorted(set(ref["mc_cell_grp"]))}


MC_PMD = DialoguePmdParams(k=2, n_permutations=20, abn_c=2, p_anova=1.0, n_genes=12)


@pytest.mark.parametrize("quality", [True, False])
def test_dialogue_metacells_match_r(ref, mc, mc_features, dlg_data, quality):
    got = dialogue.dialogue(
        mc,
        "cell_grp",
        "sample_id",
        mc_features,
        quality_col="cell_quality" if quality else None,
        gene_ids=dlg_data.var["gene_id"].to_list(),
        pmd_params=MC_PMD,
    )
    _same_dialogue(got, ref["dialogue"]["mc_quality" if quality else "mc_raw"])
