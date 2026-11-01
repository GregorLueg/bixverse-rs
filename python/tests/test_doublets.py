"""Doublet detection, empty droplets and CellSweep without R: result records,
grouping, re-calls, input checks and the denoised store.

Parity with the R functions is in `test_doublets_r.py`.
"""

import dataclasses

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import _bixverse, doublets
from bixverse.params import (
    BoostParams,
    CellsweepParams,
    EmptyDropletsParams,
    MinQualityParams,
    ScdblfinderParams,
    ScrubletParams,
    SyntheticCellsweepParams,
    SyntheticDataParams,
    to_rust,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
OPEN_QC = MinQualityParams(min_unique_genes=0, min_lib_size=0, min_cells=0)
SCRUBLET = ScrubletParams(no_pcs=10, min_gene_var_pctl=0.0, n_bins_histogram=20)
SUPPLIED = EmptyDropletsParams(method="supplied", is_empty_column="is_empty")


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment with two samples."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=400, n_genes=100, n_samples=2, sample_bias="even"),
        seed=5,
    )


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment."""
    return bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)


@pytest.fixture(scope="module")
def cs_data():
    """Synthetic counts with empty droplets."""
    return bv.io.generate_cellsweep_test_data(
        SyntheticCellsweepParams(n_real=150, n_empty=400, n_genes=60), seed=3
    )


@pytest.fixture
def cs(cs_data, tmp_path):
    """The CellSweep data loaded with the empty droplets kept."""
    return bv.io.load_csr(
        tmp_path / "cs", cs_data.counts, cs_data.obs, cs_data.var, qc=OPEN_QC
    )


############
# Scrublet #
############


def test_scrublet_record_and_obs_frame(sc):
    res = doublets.scrublet(sc, scrublet_params=SCRUBLET)
    n = sc.n_cells
    assert res.predicted_doublets.shape == (n,)
    assert res.predicted_doublets.dtype == bool
    assert res.doublet_scores_obs.dtype == np.float64
    assert isinstance(res.threshold, float)
    np.testing.assert_array_equal(res.cell_indices, sc.cells_to_keep)
    assert res.pca is None and res.pair_1 is None and res.group_by is None
    n_doublets = int(res.predicted_doublets.sum())
    assert f"n_cells={n}" in repr(res) and f"n_doublets={n_doublets}" in repr(res)

    frame = doublets.get_data(res)
    assert frame.columns == ["doublet", "doublet_score", "cell_idx"]
    sc.add_obs(frame)
    obs = sc.obs(cols=["doublet", "doublet_score"])
    assert obs["doublet"].to_list() == res.predicted_doublets.tolist()


def test_scrublet_optional_outputs(sc):
    res = doublets.scrublet(
        sc, scrublet_params=SCRUBLET, return_combined_pca=True, return_pairs=True
    )
    assert isinstance(res.pca, np.ndarray) and isinstance(res.pair_1, np.ndarray)
    n_sim = len(res.doublet_scores_sim)
    assert res.pca.shape == (sc.n_cells + n_sim, SCRUBLET.no_pcs)
    assert res.pair_1.shape == res.pair_2.shape == (n_sim,)
    assert res.pair_1.max() < sc.n_cells
    assert "pca=True" in repr(res) and "pairs=True" in repr(res)


def test_scrublet_is_seeded(sc):
    a = doublets.scrublet(sc, scrublet_params=SCRUBLET, seed=3)
    b = doublets.scrublet(sc, scrublet_params=SCRUBLET, seed=3)
    np.testing.assert_array_equal(a.doublet_scores_obs, b.doublet_scores_obs)


def test_scrublet_cells_to_use_by_id(sc):
    ids = sc.cell_ids(filtered=True)[::-1][:200]
    res = doublets.scrublet(sc, scrublet_params=SCRUBLET, cells_to_use=list(ids))
    np.testing.assert_array_equal(res.cell_indices, sc.cell_indices(list(ids)))
    assert len(res.doublet_scores_obs) == 200


def test_scrublet_grouped_orders_by_cell(sc):
    with pytest.warns(UserWarning, match="fewer than 500"):
        res = doublets.scrublet(sc, scrublet_params=SCRUBLET, group_by="sample_id")
    assert res.group_by == "sample_id"
    assert isinstance(res.threshold, dict)
    assert list(res.threshold) == ["sample_1", "sample_2"]
    assert isinstance(res.doublet_scores_sim, dict)
    np.testing.assert_array_equal(res.cell_indices, np.sort(sc.cells_to_keep))
    obs = sc.obs(cols=["cell_idx", "sample_id"], filtered=True)
    assert res.cell_groups is not None
    assert res.cell_groups.tolist() == obs["sample_id"].to_list()
    assert "group_by='sample_id'" in repr(res)
    assert "'sample_1'" in repr(res) and "'sample_2'" in repr(res)


#####################
# Grouping failures #
#####################


def test_group_by_missing_column(sc):
    with pytest.raises(ValueError, match="not found in the obs table"):
        doublets.scrublet(sc, scrublet_params=SCRUBLET, group_by="nope")


def test_group_by_with_nulls(sc):
    labels = ["a"] * (sc.n_cells - 1) + [None]
    sc.add_obs(pl.DataFrame({"grp": labels}, schema={"grp": pl.String}))
    with pytest.raises(ValueError, match="1 NA value"):
        doublets.scrublet(sc, scrublet_params=SCRUBLET, group_by="grp")


def test_group_too_small(sc):
    labels = ["small"] * 10 + ["big"] * (sc.n_cells - 10)
    sc.add_obs(pl.DataFrame({"grp": labels}))
    with pytest.raises(ValueError, match="'small' \\(n=10\\)"):
        doublets.scrublet(sc, scrublet_params=SCRUBLET, group_by="grp")


################
# Manual calls #
################


def test_call_doublets_manual_ungrouped(sc):
    res = doublets.scrublet(sc, scrublet_params=SCRUBLET)
    new = doublets.call_doublets_manual(res, 0.2)
    np.testing.assert_array_equal(new.predicted_doublets, res.doublet_scores_obs > 0.2)
    np.testing.assert_allclose(
        new.z_scores, (res.doublet_scores_obs - 0.2) / res.doublet_errors_obs
    )
    assert new.threshold == 0.2
    assert new.detected_doublet_rate == pytest.approx(
        new.predicted_doublets.mean(), abs=1e-15
    )
    # the original record is untouched
    assert res.predicted_doublets is not new.predicted_doublets
    np.testing.assert_array_equal(
        res.predicted_doublets, res.doublet_scores_obs > res.threshold
    )


def test_call_doublets_manual_grouped_defaults_to_first(sc):
    with pytest.warns(UserWarning):
        res = doublets.scrublet(sc, scrublet_params=SCRUBLET, group_by="sample_id")
    new = doublets.call_doublets_manual(res, 0.5)
    assert isinstance(new.threshold, dict) and isinstance(res.threshold, dict)
    assert new.threshold["sample_1"] == 0.5
    assert new.threshold["sample_2"] == res.threshold["sample_2"]
    other = res.cell_groups == "sample_2"
    np.testing.assert_array_equal(
        new.predicted_doublets[other], res.predicted_doublets[other]
    )


def test_call_doublets_manual_rejects(sc):
    res = doublets.scrublet(sc, scrublet_params=SCRUBLET)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        doublets.call_doublets_manual(res, 1.5)
    grouped = dataclasses.replace(
        res,
        cell_groups=np.full(sc.n_cells, "a"),
        group_by="x",
        doublet_scores_sim={"a": res.doublet_scores_sim},
    )
    with pytest.raises(ValueError, match="Sample 'b' not found"):
        doublets.call_doublets_manual(grouped, 0.3, for_sample="b")


#########################
# Boost and scDblFinder #
#########################


def test_boost_record(sc):
    res = doublets.doublet_detection_boost(
        sc, boost_params=BoostParams(no_pcs=10, n_iters=3, min_gene_var_pctl=0.0)
    )
    assert res.doublet.shape == res.doublet_score.shape == (sc.n_cells,)
    assert np.all((res.voting_avg >= 0) & (res.voting_avg <= 1))
    assert doublets.get_data(res).columns == [
        "doublet",
        "doublet_score",
        "voting_avg",
        "cell_idx",
    ]
    assert repr(res).startswith("BoostRes(") and f"n_cells={sc.n_cells}" in repr(res)


def test_fast_cluster_centroids_resolved():
    from bixverse._core.doublets import _with_centroids

    assert _with_centroids(BoostParams(fast_cluster=True), 400)["n_centroids"] == 80
    assert _with_centroids(BoostParams(), 400)["n_centroids"] is None
    assert (
        _with_centroids(ScdblfinderParams(fast_cluster=True, n_centroids=7), 400)[
            "n_centroids"
        ]
        == 7
    )


def test_scdblfinder_record_and_getters(sc):
    params = ScdblfinderParams(no_pcs=10, n_genes=40, cxds_genes=40)
    res = doublets.scdblfinder(sc, scdblfinder_params=params, return_features=True)
    assert res.features is not None
    assert res.features.height == sc.n_cells
    assert res.features["cell_id"].to_list() == sc.cell_ids(filtered=True).tolist()
    assert doublets.get_scores(res) is res.weighted
    assert doublets.get_scores(res, "cxds_scores") is res.cxds_scores
    assert "features=True" in repr(res)

    plain = doublets.scdblfinder(sc, scdblfinder_params=params)
    with pytest.warns(UserWarning, match="did not extract"):
        assert doublets.get_feature_mat(plain) is None


def test_scdblfinder_grouped_prefixes_clusters(sc):
    params = ScdblfinderParams(no_pcs=10, n_genes=40, cxds_genes=40)
    with pytest.warns(UserWarning):
        res = doublets.scdblfinder(sc, scdblfinder_params=params, group_by="sample_id")
    assert all(
        str(c).startswith(f"{g}_")
        for c, g in zip(res.cluster_labels, res.cell_groups, strict=True)
    )


##################
# Empty droplets #
##################


def test_infer_empty_droplets_cutoff():
    lib = pl.Series([0.0, 5.9, None, -3.0, 10.0, 50.0])
    got = doublets.infer_empty_droplets(
        lib, EmptyDropletsParams(method="umi_cutoff", umi_cutoff=10)
    )
    assert got.tolist() == [True, True, True, True, False, False]


def test_infer_empty_droplets_rejects_supplied():
    with pytest.raises(ValueError, match="nothing to infer"):
        doublets.infer_empty_droplets([1, 2], SUPPLIED)


def test_empty_droplet_spec_errors_in_rust():
    lib = np.array([1, 2, 3], dtype=np.uint32)
    with pytest.raises(ValueError, match="needs a 'umi_cutoff' entry"):
        _bixverse.empty_droplets(lib, {"method": "umi_cutoff"})
    with pytest.raises(ValueError, match="not a known empty droplet method"):
        _bixverse.empty_droplets(lib, {"method": "magic"})
    with pytest.raises(bv.BixverseError):
        _bixverse.empty_droplets(lib, to_rust(SUPPLIED))


#############
# CellSweep #
#############


def test_cellsweep_writes_denoised_store(cs, tmp_path):
    keep = cs.cells_to_keep[5:]
    cs.set_cells_to_keep(keep)
    out = doublets.cellsweep(tmp_path / "out", cs, "cell_grp", "sample_id", SUPPLIED)
    # the five unkept real barcodes and every empty droplet drop out
    obs_in = cs.obs()
    n_real = int((~obs_in["is_empty"]).sum())
    assert out.dims == (n_real - 5, cs.n_genes)
    obs = out.obs()
    assert obs["cell_id"].to_list() == obs_in["cell_id"][5:n_real].to_list()
    for col in (
        "cellsweep_alpha",
        "cellsweep_z",
        "cellsweep_beta",
        "cellsweep_ll",
        "cellsweep_converged",
    ):
        assert col in obs.columns
    assert obs["cellsweep_z"].dtype == pl.String
    assert set(obs["cellsweep_z"].to_list()) <= set(obs["cell_grp"].to_list())
    assert obs["to_keep"].all()
    var = out.var()
    assert var["cellsweep_ambient"].sum() == pytest.approx(1.0, rel=1e-5)
    assert var["gene_id"].to_list() == cs.gene_ids().tolist()

    raw_in = bv.get_counts(cs, cells=obs_in["cell_id"][5:n_real].to_list()).matrix
    raw_out = bv.get_counts(out).matrix
    assert raw_out.data.sum() < raw_in.data.sum()


def test_cellsweep_input_checks(cs, tmp_path):
    with pytest.raises(ValueError, match="different directories"):
        doublets.cellsweep(cs.dir, cs, "cell_grp", "sample_id", SUPPLIED)
    with pytest.raises(ValueError, match="'nope' is not a column"):
        doublets.cellsweep(tmp_path / "o", cs, "nope", "sample_id", SUPPLIED)
    bad = EmptyDropletsParams(method="supplied", is_empty_column="cell_grp")
    with pytest.raises(ValueError, match="must be a logical column"):
        doublets.cellsweep(tmp_path / "o", cs, "cell_grp", "sample_id", bad)
    empties = np.flatnonzero(cs.obs()["is_empty"].to_numpy())
    cs.set_cells_to_keep(empties)
    with pytest.raises(ValueError, match="No barcode is both annotated"):
        doublets.cellsweep(tmp_path / "o", cs, "cell_grp", "sample_id", SUPPLIED)


def test_cellsweep_refuses_filtered_ingest(cs_data, tmp_path):
    qc = MinQualityParams(min_unique_genes=1, min_lib_size=500, min_cells=1)
    sc = bv.io.load_csr(tmp_path / "f", cs_data.counts, cs_data.obs, cs_data.var, qc=qc)
    with pytest.raises(ValueError, match="filtered out at ingest"):
        doublets.cellsweep(tmp_path / "o", sc, "cell_grp", "sample_id", SUPPLIED)


def test_cellsweep_rejects_unfrozen_empties():
    with pytest.raises(ValueError, match="freeze_empties"):
        CellsweepParams(freeze_empties=False)
