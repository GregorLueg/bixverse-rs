"""Ligand-receptor analysis and DIALOGUE without R: the scaling helpers,
guards, the CellPhoneDB flattening and cache, and the object plumbing on
synthetic data.

Parity with the R package is in `test_lr_r.py`.
"""

import io
import os
import urllib.request
import zipfile

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import dialogue, lr
from bixverse._core.lr import (
    WEIGHT_NAMES,
    _flatten_cellphonedb,
    _rank_scale,
    _scale_quantile_adapted,
    _scaling_zscore,
)
from bixverse.params import (
    CellphonedbParams,
    DialogueHlmParams,
    DialoguePmdParams,
    LigandTargetParams,
    MinQualityParams,
    SyntheticDataParams,
    SyntheticDialogueParams,
)

PPI = pl.DataFrame(
    {
        "from": ["L1", "SIG1", "L2", "SIG2"],
        "to": ["SIG1", "TF1", "SIG2", "TF2"],
        "weight": [1.0] * 4,
    }
)
GRN = pl.DataFrame(
    {
        "from": ["TF1"] * 3 + ["TF2"] * 3,
        "to": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "weight": [1.0] * 6,
    }
)
NO_CUTOFF = LigandTargetParams(ltf_cutoff=0.0)


###########
# Helpers #
###########


def test_rank_scale_averages_ties_and_tops_at_one():
    assert _rank_scale(np.array([1.0, 2.0, 3.0, 4.0])).tolist() == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    tied = _rank_scale(np.array([1.0, 1.0, 2.0, 2.0]))
    assert tied[0] == tied[1] < tied[2] == tied[3] == 1.0


def test_scaling_zscore_moments_and_degenerate():
    x = np.random.default_rng(1).normal(size=50)
    z = _scaling_zscore(x)
    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=1) - 1) < 1e-12
    assert _scaling_zscore(np.ones(4)).tolist() == [0.0] * 4
    assert _scaling_zscore(np.array([3.0])).tolist() == [0.0]
    with_nan = _scaling_zscore(np.array([1.0, np.nan, 3.0]))
    assert np.isnan(with_nan[1])
    np.testing.assert_allclose(with_nan[[0, 2]], [-np.sqrt(0.5), np.sqrt(0.5)])


def test_scale_quantile_adapted_clips_and_degenerates():
    y = _scale_quantile_adapted(np.arange(1.0, 101.0))
    assert y.min() == 0.0 and y.max() == 1.0
    assert (y == 0).sum() > 1 and (y == 1).sum() > 1
    assert _scale_quantile_adapted(np.ones(10)).tolist() == [0.5] * 10


def test_resolve_weights():
    assert set(lr.resolve_weights(None, "case_control", True).values()) == {1.0}
    w = lr.resolve_weights(None, "one_condition", False)
    assert list(w) == list(WEIGHT_NAMES)
    assert [w[k] for k in WEIGHT_NAMES][-2:] == [0.0, 0.0]
    with pytest.raises(ValueError, match="require `condition_de`"):
        lr.resolve_weights(None, "case_control", False)
    with pytest.raises(ValueError, match="missing"):
        lr.resolve_weights({"de_ligand": 1.0}, "case_control", True)
    bad = dict.fromkeys(WEIGHT_NAMES, 1.0) | {"de_ligand": -1.0}
    with pytest.raises(ValueError, match="non-negative"):
        lr.resolve_weights(bad, "case_control", True)


############
# NicheNet #
############


def test_influence_follows_the_network():
    inf = lr.generate_ligand_target_influence(
        {"L1": ["L1"], "L2": "L2"}, PPI, GRN, NO_CUTOFF
    )
    assert inf.shape == (2, len(inf.gene_ids))
    assert inf.gene_ids[:4] == ("L1", "SIG1", "L2", "SIG2")
    m = lr.get_influence(inf)
    col = {g: j for j, g in enumerate(inf.gene_ids)}
    assert m[0, col["G1"]] > 0 and m[0, col["G4"]] == 0
    assert m[1, col["G4"]] > 0 and m[1, col["G1"]] == 0
    assert "damping_factor=0.5" in repr(inf)


def test_influence_rejects_unknown_seeds():
    with pytest.raises(ValueError, match="not present in either network"):
        lr.generate_ligand_target_influence([["L1", "NOPE"]], PPI, GRN)
    with pytest.raises(ValueError, match="missing the column"):
        lr.generate_ligand_target_influence(["L1"], PPI.drop("weight"), GRN)


def test_as_ligand_target_influence_transposes_nichenetr():
    ltm = np.array([[0.5, 0.4], [0.0, 0.3], [0.1, 0.1]])
    inf = lr.as_ligand_target_influence(
        ltm, ["G1", "G2", "G3"], ["L1", "L2"], ligands_as_cols=True
    )
    assert inf.ligand_names == ("L1", "L2")
    assert inf.gene_ids == ("G1", "G2", "G3")
    assert inf.ligand_seeds == (("L1",), ("L2",))
    np.testing.assert_array_equal(inf.influence, ltm.T)
    assert "damping_factor=None" in repr(inf)
    with pytest.raises(ValueError, match="unique"):
        lr.as_ligand_target_influence(ltm, ["G1", "G1", "G3"], ["L1", "L2"])
    with pytest.raises(ValueError, match="do not fit"):
        lr.as_ligand_target_influence(ltm, ["G1"], ["L1", "L2"])


def test_activity_scores_rank_the_planted_ligand():
    inf = lr.generate_ligand_target_influence(
        {"L1": "L1", "L2": "L2"}, PPI, GRN, NO_CUTOFF
    )
    res = lr.ligand_activity_scores(inf, {"set_a": ["G1", "G2", "G3"]})
    assert res.columns == [
        "gene_set",
        "ligand",
        "auroc",
        "aupr",
        "aupr_corrected",
        "pearson",
        "spearman",
    ]
    by = dict(zip(res["ligand"], res["auroc"], strict=True))
    assert by["L1"] > by["L2"]
    unnamed = lr.ligand_activity_scores(inf, [["G1"], ["G4"]])
    assert unnamed["gene_set"].unique(maintain_order=True).to_list() == ["1", "2"]
    with pytest.raises(ValueError, match="No background genes"):
        lr.ligand_activity_scores(inf, ["G1"], background=["NOPE"])


##################
# Prioritisation #
##################


def _prio_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    lr_net = pl.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]})
    cl = ["sender", "receiver"]
    genes = ["L1", "L2", "R1", "R2"]
    de = pl.DataFrame(
        {
            "cluster_id": [c for c in cl for _ in genes],
            "gene": genes * 2,
            "lfc": [3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 2.5, 0.1],
            "pval": [1e-10, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-9, 0.5],
        }
    )
    expr = de.select(
        "cluster_id",
        "gene",
        avg_expr=pl.when(
            ((pl.col("cluster_id") == "sender") & (pl.col("gene") == "L1"))
            | ((pl.col("cluster_id") == "receiver") & (pl.col("gene") == "R1"))
        )
        .then(5.0)
        .otherwise(0.05),
    )
    la = pl.DataFrame({"ligand": ["L1", "L2"], "aupr_corrected": [0.75, -0.125]})
    return de, expr, la, lr_net


def test_prioritise_recovers_the_planted_pair():
    de, expr, la, lr_net = _prio_inputs()
    res = lr.prioritise_interactions(
        de, expr, la, lr_net, ["sender"], ["receiver"], scenario="one_condition"
    )
    assert res.columns[:4] == ["sender", "receiver", "ligand", "receptor"]
    assert res.row(0, named=True)["ligand"] == "L1"
    assert res["prioritisation_rank"].to_list() == [1, 2]


def test_prioritise_guards():
    de, expr, la, lr_net = _prio_inputs()
    with pytest.raises(ValueError, match="one row per ligand"):
        lr.prioritise_interactions(
            de, expr, pl.concat([la, la]), lr_net, ["sender"], ["receiver"]
        )
    with pytest.raises(ValueError, match="must lie in"):
        lr.prioritise_interactions(
            de.with_columns(pval=pl.lit(2.0)), expr, la, lr_net, ["s"], ["r"]
        )
    with pytest.warns(UserWarning, match="No interactions remain"):
        res = lr.prioritise_interactions(
            de, expr, la, lr_net, ["nobody"], ["receiver"], scenario="one_condition"
        )
    assert res.height == 0


def test_prioritise_undefined_activity_sorts_last():
    lr_net = pl.DataFrame({"ligand": ["L1", "L2", "L3"], "receptor": ["R1"] * 3})
    expr = pl.DataFrame(
        {
            "cluster_id": ["s", "s", "s", "r"],
            "gene": ["L1", "L2", "L3", "R1"],
            "avg_expr": [1.0, 2.0, 3.0, 1.0],
        }
    )
    de = expr.select("cluster_id", "gene", lfc=pl.lit(1.0), pval=pl.lit(0.01))
    la = pl.DataFrame(
        {"ligand": ["L1", "L2", "L3"], "aupr_corrected": [float("nan"), 0.1, 0.5]}
    )
    res = lr.prioritise_interactions(
        de, expr, la, lr_net, ["s"], ["r"], scenario="one_condition"
    )
    assert res["ligand"].to_list() == ["L3", "L2", "L1"]
    assert res["prioritisation_score"].null_count() == 1
    assert res["prioritisation_rank"].to_list() == [1, 2, None]


###############
# CellPhoneDB #
###############


def _cpdb_tables() -> dict[str, pl.DataFrame]:
    return {
        "gene_table": pl.DataFrame(
            {"protein_id": [1, 2, 3, 4, 2], "gene_name": ["g1", "g2", "g3", "g4", "g2"]}
        ),
        "protein_table": pl.DataFrame(
            {"id_protein": [1, 2, 3, 4], "protein_multidata_id": [1, 2, 3, 4]}
        ),
        "complex_composition_table": pl.DataFrame(
            {"complex_multidata_id": [10, 10], "protein_multidata_id": [4, 3]}
        ),
        "multidata_table": pl.DataFrame(
            {
                "id_multidata": [1, 2, 3, 4, 10],
                "name": ["P1", "P2", "P3", "P4", "cpx"],
                "is_complex": ["False", "False", "False", "False", "True"],
            }
        ),
        "interaction_table": pl.DataFrame(
            {
                "id_cp_interaction": ["CPI-2", "CPI-1"],
                "multidata_1_id": [1, 2],
                "multidata_2_id": [10, 1],
                "directionality": ["Ligand-Receptor", "Adhesion-Adhesion"],
                "classification": ["a", None],
            }
        ),
    }


def test_flatten_cellphonedb_resolves_partners():
    res = _flatten_cellphonedb(_cpdb_tables())
    assert res["interaction_id"].to_list() == ["CPI-1", "CPI-2"]
    assert res["partner_a"].to_list() == ["g2", "g1"]
    assert res["partner_b"].to_list() == ["g1", "cpx"]
    assert res["genes_b"].to_list() == [["g1"], ["g3", "g4"]]
    assert res["is_complex_b"].to_list() == [False, True]


def test_get_cellphonedb_db_reads_a_cached_zip(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError("a cached download touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with zipfile.ZipFile(tmp_path / "cellphonedb.zip", "w") as zf:
        for name, df in _cpdb_tables().items():
            buf = io.BytesIO()
            df.write_csv(buf)
            zf.writestr(f"{name}.csv", buf.getvalue())
    res = lr.get_cellphonedb_db(tmp_path, verbose=False)
    assert res.equals(_flatten_cellphonedb(_cpdb_tables()))


@pytest.mark.skipif(
    not os.environ.get("BIXVERSE_TEST_DOWNLOADS"),
    reason="set BIXVERSE_TEST_DOWNLOADS to fetch CellPhoneDB",
)
def test_get_cellphonedb_db_downloads(tmp_path):
    res = lr.get_cellphonedb_db(tmp_path)
    assert res.height > 1000
    assert res["interaction_id"].is_unique().all()
    assert (res["genes_a"].list.len() > 0).all()


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=300, n_genes=60), seed=11
    )
    return bv.io.load_csr(
        tmp_path_factory.mktemp("lr") / "sc",
        data.counts,
        data.obs,
        data.var,
        qc=MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1),
    )


def _lr_db(genes: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "interaction_id": ["I1", "I2", "I3"],
            "partner_a": [genes[0], genes[2], genes[4]],
            "partner_b": [genes[10], "cpx", "NOPE"],
            "genes_a": [[genes[0]], [genes[2]], [genes[4]]],
            "genes_b": [[genes[10]], [genes[11], genes[40]], ["NOPE"]],
        }
    )


def test_cellphonedb_methods_share_means_and_shape(sc):
    db = _lr_db(sc.gene_ids().tolist())
    params = CellphonedbParams(n_perm=50)
    with pytest.warns(UserWarning, match="Dropping 1 of 3"):
        simple = lr.cellphonedb(sc, "cell_grp", db, method="simple", params=params)
    with pytest.warns(UserWarning, match="Dropping"):
        stat = lr.cellphonedb(sc, "cell_grp", db, params=params)
    n_types = sc.obs()["cell_grp"].n_unique()
    assert simple.height == 2 * n_types**2
    assert simple["pval"].null_count() == simple.height
    assert stat["pval"].null_count() == 0
    np.testing.assert_array_equal(simple["mean"], stat["mean"])
    with pytest.warns(UserWarning, match="Dropping"):
        sub = lr.cellphonedb(
            sc, "cell_grp", db, senders=["cell_type_2"], params=params, method="simple"
        )
    assert set(sub["sender"]) == {"cell_type_2"}
    assert sub.height == 2 * n_types


def test_cellphonedb_guards(sc):
    genes = sc.gene_ids().tolist()
    db = _lr_db(genes)
    with pytest.raises(ValueError, match="needs a `deg_table`"):
        lr.cellphonedb(sc, "cell_grp", db, method="degs")
    with (
        pytest.raises(ValueError, match="senders holds clusters"),
        pytest.warns(UserWarning),
    ):
        lr.cellphonedb(sc, "cell_grp", db, senders=["nope"], method="simple")
    with pytest.raises(ValueError, match="No interaction has all"):
        lr.cellphonedb(sc, "cell_grp", db.filter(pl.col("interaction_id") == "I3"))
    with pytest.raises(ValueError, match="must be unique"):
        lr.cellphonedb(sc, "cell_grp", pl.concat([db, db]))


def test_expression_info_layout(sc):
    genes = sc.gene_ids().tolist()[:3]
    res = lr.compute_expression_info(sc, "cell_grp", genes)
    levels = sorted(sc.obs()["cell_grp"].unique().to_list())
    assert res["cluster_id"].to_list() == [c for c in levels for _ in genes]
    assert res["gene"].to_list() == genes * len(levels)
    assert ((res["frac_expr"] >= 0) & (res["frac_expr"] <= 1)).all()
    with pytest.raises(ValueError, match="both"):
        lr.compute_expression_info(sc, "cell_grp", genes, condition_colname="x")


############
# DIALOGUE #
############

DLG_PARAMS = SyntheticDialogueParams(n_samples=8, cells_per_sample=20, n_genes=120)
PMD = DialoguePmdParams(k=2, n_permutations=10, abn_c=5, n_genes=10)


@pytest.fixture(scope="module")
def dlg(tmp_path_factory):
    data = bv.io.generate_dialogue_test_data(DLG_PARAMS, seed=3)
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("dlg") / "sc",
        data.counts,
        data.obs,
        data.var,
        qc=MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1),
    )
    return sc, data


def test_dialogue_runs_and_names_everything(dlg):
    sc, data = dlg
    res = dialogue.dialogue(
        sc,
        "cell_grp",
        "sample_id",
        data.features,
        quality_col="cell_quality",
        gene_ids=data.var["gene_id"].to_list(),
        pmd_params=PMD,
        hlm_params=DialogueHlmParams(satterthwaite=False),
    )
    types = tuple(data.features)
    assert res.cell_types == types
    assert res.source_class == "SingleCells"
    assert dialogue.get_results(res).height == 2 * 3
    assert dialogue.get_params(res)["pmd_params"] == PMD
    for ct in types:
        assert res.scores[ct]["cell_id"].to_list() == (
            data.obs.filter(pl.col("cell_grp") == ct)["cell_id"].to_list()
        )
        assert res.scores[ct].columns == ["cell_id", "mcp_00", "mcp_01"]
    assert set(res.signatures["list"]) <= {"permissive", "strict"}
    assert res.refit_fidelity["cell_type"].to_list() == list(types)
    assert "DialogueResult" in repr(res)


def test_dialogue_guards(dlg):
    sc, data = dlg
    f = data.features
    one = {k: f[k] for k in list(f)[:1]}
    with pytest.raises(ValueError, match="at least two cell types"):
        dialogue.dialogue(sc, "cell_grp", "sample_id", one)
    with pytest.raises(ValueError, match="absent from"):
        dialogue.dialogue(sc, "cell_grp", "sample_id", {**f, "nope": f["cell_type_1"]})
    short = {k: v.head(5) for k, v in f.items()}
    with pytest.raises(ValueError, match="is missing"):
        dialogue.dialogue(sc, "cell_grp", "sample_id", short)
    narrow = {k: v.select("cell_id", "feature_01") for k, v in f.items()}
    with pytest.raises(ValueError, match="at least 2"):
        dialogue.dialogue(sc, "cell_grp", "sample_id", narrow)
    nan = {k: v.with_columns(feature_01=pl.lit(float("nan"))) for k, v in f.items()}
    with pytest.raises(ValueError, match="NAs"):
        dialogue.dialogue(sc, "cell_grp", "sample_id", nan)
    with pytest.raises(ValueError, match="No genes"):
        dialogue.dialogue(sc, "cell_grp", "sample_id", f)


def test_dialogue_needs_shared_samples(dlg):
    sc, data = dlg
    first = data.obs["sample_id"].unique().sort().to_list()[:4]
    sc.add_obs(
        sc.obs(cols=["cell_idx", "sample_id"]).select(
            "cell_idx",
            few=pl.when(pl.col("sample_id").is_in(first))
            .then(pl.lit("a"))
            .otherwise(pl.lit("b")),
        )
    )
    with pytest.raises(ValueError, match="present in every cell type"):
        dialogue.dialogue(
            sc,
            "cell_grp",
            "few",
            data.features,
            gene_ids=data.var["gene_id"].to_list(),
        )
