"""Cell QC: metric passes against dense truth, and the outlier call edge cases."""

import warnings

import numpy as np
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import bixverse as bv
from bixverse import qc
from bixverse._core import db

N_CELLS = 40
N_GENES = 12


@pytest.fixture
def exp(tmp_path):
    """An experiment directory built in-test, plus the dense raw counts."""
    dense = np.asarray(
        bv._bixverse._write_synthetic_store(
            str(tmp_path / "counts_cells.bin"), N_CELLS, N_GENES, 0.4, 7
        )
    )
    obs = pl.DataFrame({"cell_id": [f"cell_{i:02d}" for i in range(N_CELLS)]})
    with db.connect(db.db_path(tmp_path)) as con:
        db.populate_from_frame(con, "obs", obs)
        db.set_to_keep_column(con)
        db.populate_var_minimal(con, [f"gene_{j:02d}" for j in range(N_GENES)])
    return bv.SingleCells(tmp_path), dense.astype(np.float64)


def _metrics(n: int = 50, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lib = rng.normal(1000.0, 100.0, n)
    lib[0] = 50.0
    return {"lib_size": lib, "pct_mt": rng.uniform(0.0, 20.0, n)}


#################
# Metric passes #
#################


@pytest.mark.parametrize("streaming", [False, True])
def test_gene_set_proportions_match_dense(exp, streaming):
    sc, dense = exp
    sets = {"a": ["gene_00", "gene_03"], "b": ["gene_05"]}
    qc.gene_set_proportions(sc, sets, streaming=streaming)
    obs = sc.obs()
    total = dense.sum(axis=1)
    for name, genes in sets.items():
        cols = [int(g.split("_")[1]) for g in genes]
        want = np.where(total > 0, dense[:, cols].sum(axis=1) / total, 0.0)
        np.testing.assert_allclose(obs[name].to_numpy(), want, rtol=1e-6)
        assert obs[name].dtype == pl.Float64


@pytest.mark.parametrize("streaming", [False, True])
def test_top_genes_perc_matches_dense(exp, streaming):
    sc, dense = exp
    qc.top_genes_perc(sc, top_n_vals=[1, 3, 50], streaming=streaming)
    obs = sc.obs()
    total = dense.sum(axis=1)
    for n in (1, 3, 50):
        top = -np.sort(-dense, axis=1)[:, :n].sum(axis=1)
        want = np.where((dense > 0).sum(axis=1) <= n, 1.0, top / total)
        got = obs[f"top_{n}_genes_percentage"].to_numpy()
        np.testing.assert_allclose(got, want, rtol=1e-6)


def test_metric_pass_leaves_dropped_cells_null(exp):
    sc, _ = exp
    sc.set_cells_to_keep(np.arange(0, N_CELLS, 2))
    qc.gene_set_proportions(sc, {"a": ["gene_01"]})
    col = sc.obs()["a"]
    assert col.null_count() == N_CELLS // 2
    assert col[1::2].is_null().all()
    assert not col[::2].is_null().any()


def test_metric_pass_follows_unsorted_cell_order(exp):
    sc, dense = exp
    order = np.array([5, 1, 9, 0])
    sc.set_cells_to_keep(order)
    qc.gene_set_proportions(sc, {"a": ["gene_02"]})
    obs = sc.obs(filtered=True)
    total = dense[obs["cell_idx"].to_numpy()].sum(axis=1)
    want = dense[obs["cell_idx"].to_numpy(), 2] / total
    np.testing.assert_allclose(obs["a"].to_numpy(), want, rtol=1e-6)


def test_gene_set_with_no_known_gene_raises(exp):
    sc, _ = exp
    with pytest.raises(ValueError), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qc.gene_set_proportions(sc, {"a": ["nope"]})


def test_top_genes_perc_rejects_bad_n(exp):
    sc, _ = exp
    with pytest.raises(ValueError):
        qc.top_genes_perc(sc, top_n_vals=[0])
    with pytest.raises(ValueError):
        qc.top_genes_perc(sc, top_n_vals=[])


def test_available_features_lists_obs_then_genes(exp):
    sc, _ = exp
    feats = qc.get_available_features(sc)
    obs_cols = sc.obs().columns
    assert feats["feature_name"].to_list() == obs_cols + list(sc.gene_ids())
    assert feats["origin"].to_list() == ["obs"] * len(obs_cols) + ["counts"] * N_GENES


#######
# MAD #
#######


def test_per_cell_qc_outlier_thresholds_follow_direction():
    x = _metrics()["lib_size"]
    below = qc.per_cell_qc_outlier(x, direction="below")
    above = qc.per_cell_qc_outlier(x, direction="above")
    both = qc.per_cell_qc_outlier(x)
    assert below.upper_threshold is None and below.lower_threshold is not None
    assert above.lower_threshold is None and above.upper_threshold is not None
    assert both.upper_threshold is not None and both.lower_threshold is not None
    assert below.outlier[0] and not above.outlier[0]
    assert np.array_equal(both.outlier, below.outlier | above.outlier)


def test_per_cell_qc_outlier_rejects_bad_input():
    with pytest.raises(ValueError):
        qc.per_cell_qc_outlier(np.array([]))
    with pytest.raises(BeartypeCallHintParamViolation):
        qc.per_cell_qc_outlier([1.0, 2.0], direction="sideways")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        bv._bixverse.mad_outlier(np.array([1.0, 2.0]), 3.0, "sideways", True)


def test_raw_mad_is_narrower_than_scaled():
    x = _metrics()["lib_size"]
    scaled = qc.per_cell_qc_outlier(x, r_standard=True)
    raw = qc.per_cell_qc_outlier(x, r_standard=False)
    assert raw.upper_threshold is not None and scaled.upper_threshold is not None
    assert raw.upper_threshold < scaled.upper_threshold
    assert raw.outlier.sum() >= scaled.outlier.sum()


##########
# CellQc #
##########


def test_run_cell_qc_single_group_has_no_group_stats():
    m = _metrics()
    res = qc.run_cell_qc(m, np.arange(50))
    assert res.per_group_stats is None
    assert res.thresholds is not None
    assert res.thresholds["group"].to_list() == ["all", "all"]
    assert res.combined[0]


def test_run_cell_qc_groups_keep_first_seen_order():
    m = _metrics(60)
    groups = ["z", "a", "m"] * 20
    res = qc.run_cell_qc(m, np.arange(60), groups=groups)
    assert res.thresholds is not None
    assert res.thresholds["group"].to_list()[:3] == ["z", "a", "m"]
    assert res.per_group_stats is not None
    assert res.per_group_stats.height == 6


def test_run_cell_qc_argument_checks():
    m = _metrics()
    cells = np.arange(50)
    with pytest.raises(ValueError, match="Nothing to do"):
        qc.run_cell_qc(m, cells, mad=False)
    with pytest.raises(ValueError, match="directions"):
        qc.run_cell_qc(m, cells, directions={"lib_size": "below"})
    with pytest.raises(ValueError, match="'above' but `lower`"):
        qc.run_cell_qc(
            m,
            cells,
            directions={"lib_size": "twosided", "pct_mt": "above"},
            hard_thresholds={"pct_mt": {"lower": 1.0}},
        )
    with pytest.raises(ValueError, match="bounds"):
        qc.run_cell_qc(m, cells, hard_thresholds={"pct_mt": {"max": 1.0}})
    with pytest.raises(ValueError, match="unknown"):
        qc.run_cell_qc(m, cells, hard_thresholds={"nope": {"upper": 1.0}})
    with pytest.raises(ValueError, match="one value per cell"):
        qc.run_cell_qc({"a": np.ones(3), "b": np.ones(4)}, np.arange(3))
    with pytest.raises(ValueError, match="cells_to_keep"):
        qc.run_cell_qc(m, np.arange(10))
    with pytest.raises(ValueError, match="groups"):
        qc.run_cell_qc(m, cells, groups=["a"] * 49)


def test_contradicting_bounds_allowed_without_mad():
    m = _metrics()
    res = qc.run_cell_qc_fixed(m, np.arange(50), {"pct_mt": {"lower": 1.0}})
    assert res.thresholds is None
    assert np.array_equal(res.combined, m["pct_mt"] < 1.0)
    with pytest.raises(ValueError, match="No MAD"):
        qc.rescue_cells(res, {"pct_mt": {"lower": 0.0}})


def test_flag_cells_unions_and_resets():
    m = _metrics()
    res = qc.run_cell_qc_fixed(m, np.arange(50), {"pct_mt": {"upper": 15.0}})
    res = qc.flag_cells(res, {"pct_mt": {"upper": 18.0}, "lib_size": {"lower": 900.0}})
    assert np.array_equal(res.hard_mat["pct_mt"], m["pct_mt"] > 15.0)
    assert res.hard_thresholds == {
        "pct_mt": {"upper": 18.0},
        "lib_size": {"lower": 900.0},
    }
    res = qc.flag_cells(res, {"pct_mt": {"upper": 18.0}}, reset=True)
    assert np.array_equal(res.hard_mat["pct_mt"], m["pct_mt"] > 18.0)
    assert not res.hard_mat["lib_size"].any()
    assert res.hard_thresholds == {"pct_mt": {"upper": 18.0}}


def test_rescue_cells_clears_mad_but_not_hard():
    m = _metrics()
    res = qc.run_cell_qc(
        m, np.arange(50), directions={"lib_size": "below", "pct_mt": "above"}
    )
    assert res.outlier_mat["lib_size"][0]
    rescued = qc.rescue_cells(res, {"lib_size": {"lower": 0.0}})
    assert rescued.rescued_mat["lib_size"][0]
    assert not rescued.final("lib_size")[0]

    hard = qc.flag_cells(res, {"lib_size": {"lower": 100.0}})
    with pytest.warns(UserWarning, match="remain flagged by hard"):
        still = qc.rescue_cells(hard, {"lib_size": {"lower": 0.0}})
    assert still.final("lib_size")[0]
    assert still.combined[0]


def test_get_data_columns_follow_r():
    m = _metrics()
    res = qc.run_cell_qc(m, np.arange(50) + 3)
    frame = qc.get_data(res)
    per = [
        f"{x}_{s}"
        for x in ("lib_size", "pct_mt")
        for s in ("is_outlier", "mad_outlier", "hard_outlier", "rescued")
    ]
    assert frame.columns == [
        "lib_size",
        "pct_mt",
        "cell_idx",
        "grp",
        "global_outlier",
        *per,
    ]
    assert frame["cell_idx"].to_list() == list(range(3, 53))
    assert frame["grp"].dtype == pl.String


def test_get_data_writes_into_obs(exp):
    sc, dense = exp
    obs = sc.obs(filtered=True)
    m = {"lib": dense.sum(axis=1)}
    res = qc.run_cell_qc(m, obs["cell_idx"].to_numpy())
    sc.add_obs(qc.get_data(res))
    got = sc.obs()
    assert got["global_outlier"].to_list() == res.combined.tolist()
    assert "lib_is_outlier" in got.columns


########################
# Independent oracles #
########################


@pytest.mark.parametrize("direction", ["twosided", "below", "above"])
@pytest.mark.parametrize("r_standard", [True, False])
def test_mad_outlier_matches_numpy(direction, r_standard):
    from scipy import stats

    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(10, 2, 500), [0.5, 30.0, -4.0, 25.0]])
    res = qc.per_cell_qc_outlier(
        x, threshold=2.5, direction=direction, r_standard=r_standard
    )
    med = np.median(x)
    # the exact 1 / qnorm(0.75), not R's rounded 1.4826
    scale = 1 / stats.norm.ppf(0.75) if r_standard else 1.0
    margin = 2.5 * np.median(np.abs(x - med)) * scale
    lo, hi = x < med - margin, x > med + margin
    want = {"twosided": lo | hi, "below": lo, "above": hi}[direction]
    np.testing.assert_array_equal(res.outlier, want)
    assert res.median == pytest.approx(med)
    if direction != "below":
        assert res.upper_threshold == pytest.approx(med + margin, rel=1e-6)
    if direction != "above":
        assert res.lower_threshold == pytest.approx(med - margin, rel=1e-6)
