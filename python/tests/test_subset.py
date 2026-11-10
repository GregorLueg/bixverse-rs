"""SingleCellsSubset without R: construction, index spaces, obs handling and
the write-back to the parent.

Parity with the R subset chain is in `test_subset_r.py`.
"""

import numpy as np
import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import bixverse as bv
from bixverse import graph, pp
from bixverse.params import MinQualityParams, SyntheticDataParams

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
GROUP = ("cell_grp", "cell_type_1")


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=240, n_genes=60, n_batches=2), seed=5
    )


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment with the first ten cells filtered out."""
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    sc.set_cells_to_keep(sc.cell_ids()[10:])
    return sc


def _type_1(sc: bv.SingleCells) -> np.ndarray:
    """0-based store indices of the kept cell_type_1 cells, ascending."""
    obs = sc.obs(cols=["cell_idx", "cell_grp"], filtered=True)
    return np.sort(obs.filter(pl.col("cell_grp") == GROUP[1])["cell_idx"].to_numpy())


################
# Construction #
################


def test_subset_holds_the_kept_cells_of_the_level(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    want = _type_1(sc)
    np.testing.assert_array_equal(sub.subset_to_original, want)
    np.testing.assert_array_equal(np.sort(sub.cells_to_keep), want)
    assert sub.shape == (len(want), sc.n_genes)
    assert sub.dims == sc.dims
    assert (sub.obs()["cell_grp"] == GROUP[1]).all()
    assert sub.map.hvg is None
    assert sub.cache.artefacts() == {}


def test_subset_drops_the_parent_hvg(sc):
    pp.find_hvg(sc, hvg_no=20)
    sub = bv.SingleCellsSubset(sc, *GROUP)
    assert sub.map.hvg is None
    assert sc.map.hvg is not None


def test_unknown_level_raises(sc):
    with pytest.raises(ValueError, match="No cells found for group 'nope'"):
        bv.SingleCellsSubset(sc, "cell_grp", "nope")


def test_unknown_column_raises(sc):
    with pytest.raises(KeyError, match="not_a_column"):
        bv.SingleCellsSubset(sc, "not_a_column", "x")


def test_level_with_no_kept_cell_is_empty(sc):
    sc.set_cells_to_keep(_type_1(sc))
    with pytest.raises(ValueError, match="No cells found for group 'cell_type_2'"):
        bv.SingleCellsSubset(sc, "cell_grp", "cell_type_2")


def test_level_is_cast_to_the_column_type(sc):
    """A numeric column matches the level as R does, "1" against 1."""
    obs = sc.obs(cols=["cell_idx", "batch_index"], filtered=True)
    sub = bv.SingleCellsSubset(sc, "batch_index", "1")
    want = np.sort(obs.filter(pl.col("batch_index") == 1)["cell_idx"].to_numpy())
    np.testing.assert_array_equal(sub.subset_to_original, want)


#################
# Index mapping #
#################


def test_subset_rows_map_back_to_the_parent(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    obs = sub.obs()
    np.testing.assert_array_equal(obs["cell_idx"].to_numpy(), sub.subset_to_original)
    np.testing.assert_array_equal(sub.cell_ids(), sc.cell_ids(sub.subset_to_original))
    a = bv.get_counts(sub).scipy_csr().toarray()  # type: ignore[union-attr]
    b = bv.get_counts(sc, cells=sub.cells_to_keep).scipy_csr().toarray()  # type: ignore[union-attr]
    np.testing.assert_array_equal(a, b)


def test_ids_resolve_to_store_indices_inside_the_subset(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    ids = sub.cell_ids()[:3].tolist()
    np.testing.assert_array_equal(sub.cell_indices(ids), sub.subset_to_original[:3])
    outside = sc.cell_ids(np.setdiff1d(sc.cells_to_keep, sub.subset_to_original)[:1])
    with pytest.warns(UserWarning, match="could not be matched"):
        idx = sub.cell_indices([*ids, *outside.tolist()])
    assert len(idx) == 3


def test_indices_outside_the_subset_raise(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    outside = np.setdiff1d(sc.cells_to_keep, sub.subset_to_original)[:2]
    with pytest.raises(ValueError, match="outside the subset"):
        bv.get_counts(sub, cells=outside)


def test_obs_indices_are_subset_positions(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    rows = sub.obs(indices=[0, 2])
    assert rows["cell_idx"].to_list() == sub.subset_to_original[[0, 2]].tolist()
    with pytest.raises(ValueError, match="outside"):
        sub.obs(indices=[sub.obs().height])


###################################
# Independence from the parent #
###################################


def test_parent_filter_after_subsetting_leaves_the_subset(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    keep = sub.cells_to_keep.copy()
    pp.find_hvg(sub, hvg_no=20)
    pp.calculate_pca(sub, 5)
    sc.set_cells_to_keep(sc.cells_to_keep[20:])
    np.testing.assert_array_equal(sub.cells_to_keep, keep)
    assert sub.check_state(["pca"]) is True
    assert sub.cache_status()["stale"].to_list() == [False]


def test_subset_processing_leaves_the_parent(sc):
    obs_cols, var_cols = sc.obs_columns(), sc.var().columns
    sub = bv.SingleCellsSubset(sc, *GROUP)
    pp.find_hvg(sub, hvg_no=20)
    pp.calculate_pca(sub, 5)
    graph.find_neighbours(sub)
    graph.find_clusters(sub, name="sub_leiden")
    assert sc.map.hvg is None
    assert sc.cache.artefacts() == {}
    assert sc.obs_columns() == obs_cols
    assert sc.var().columns == var_cols
    assert sub.var().columns == var_cols
    labels = sub.obs()["sub_leiden"]
    assert labels.null_count() == 0
    assert labels.len() == sub.n_cells


def test_subset_is_not_persisted(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    with pytest.raises(BeartypeCallHintParamViolation):
        bv.io.save(sub)  # ty: ignore[invalid-argument-type] # wrong type on purpose


###########
# add_obs #
###########


def test_add_obs_by_row_and_by_cell(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    n = sub.obs().height
    sub.add_obs(pl.DataFrame({"a": np.arange(n)}))
    assert sub.obs()["a"].to_list() == list(range(n))
    first = sub.subset_to_original[:2]
    sub.add_obs(pl.DataFrame({"cell_idx": first, "b": ["x", "y"]}))
    assert sub.obs()["b"].to_list() == ["x", "y", *[None] * (n - 2)]
    sub.add_obs(pl.DataFrame({"a": np.zeros(n, dtype=np.int64)}))
    assert sub.obs()["a"].sum() == 0


def test_add_obs_refuses_bad_frames(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    with pytest.raises(ValueError, match="rows"):
        sub.add_obs(pl.DataFrame({"a": [1, 2]}))
    outside = np.setdiff1d(sc.cells_to_keep, sub.subset_to_original)[:1]
    with pytest.raises(ValueError, match="outside the subset"):
        sub.add_obs(pl.DataFrame({"cell_idx": outside, "a": [1]}))
    with pytest.raises(ValueError, match="protected"):
        sub.add_obs(pl.DataFrame({"cell_id": sub.cell_ids()}))


####################
# merge_subset_obs #
####################


def test_merge_writes_new_columns_and_nulls_the_rest(sc):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    sub.add_obs(pl.DataFrame({"label": ["t"] * sub.obs().height}))
    bv.merge_subset_obs(sc, sub)
    obs = sc.obs(cols=["cell_idx", "label"])
    inside = obs["cell_idx"].is_in(sub.subset_to_original.tolist())
    assert (obs.filter(inside)["label"] == "t").all()
    assert obs.filter(~inside)["label"].null_count() == (~inside).sum()


def test_merge_several_subsets_with_prefix(sc):
    subs = [
        bv.SingleCellsSubset(sc, "cell_grp", g) for g in ("cell_type_1", "cell_type_2")
    ]
    for s in subs:
        s.add_obs(pl.DataFrame({"cl": [0] * s.obs().height}))
    with pytest.warns(UserWarning, match="more than one subset"):
        bv.merge_subset_obs(sc, subs, cols=["cl"])
    bv.merge_subset_obs(sc, subs, cols=["cl"], new_names=["cl_p"], prefix_values=True)
    obs = sc.obs(cols=["cell_idx", "cl_p"])
    got = obs.filter(pl.col("cell_idx") == int(subs[1].subset_to_original[0]))
    assert got["cl_p"].item() == "cell_type_2_0"


def test_merge_refuses_clashes_overlaps_and_foreign_subsets(sc, data, tmp_path):
    sub = bv.SingleCellsSubset(sc, *GROUP)
    with pytest.raises(ValueError, match="No new obs columns"):
        bv.merge_subset_obs(sc, sub)
    with pytest.raises(ValueError, match="already exist"):
        bv.merge_subset_obs(sc, sub, cols=["batch_index"])
    with pytest.raises(ValueError, match="join key"):
        bv.merge_subset_obs(sc, sub, cols=["cell_idx"])
    sub.add_obs(pl.DataFrame({"z": [1] * sub.obs().height}))
    with pytest.raises(ValueError, match="Duplicated cells"):
        bv.merge_subset_obs(sc, [sub, sub], cols=["z"], prefix_values=True)

    renamed = data.obs.with_columns(
        pl.format("x_{}", pl.col("cell_id")).alias("cell_id")
    )
    foreign = bv.io.load_csr(
        tmp_path / "foreign", data.counts, renamed, data.var, qc=QC
    )
    with pytest.raises(ValueError, match="does not originate"):
        bv.merge_subset_obs(sc, bv.SingleCellsSubset(foreign, *GROUP), cols=["z"])
