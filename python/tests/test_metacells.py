"""MetaCells without R: generation, aggregation, accessors, merge, metrics on
edge cases and the persistence round trip.

Parity with the R meta cell suite is in `test_metacells_r.py`.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import graph, metacells, pp
from bixverse.params import (
    BtMetacellsParams,
    MinQualityParams,
    NeighboursParams,
    SupercellParams,
    SyntheticDataParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
BT = BtMetacellsParams(target_no_metacells=12, k=8)


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    """A small synthetic experiment with a PCA and a kNN."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=240, n_genes=60, n_batches=2), seed=5
    )
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("mc") / "sc", data.counts, data.obs, data.var, qc=QC
    )
    sc.set_cells_to_keep(sc.cell_ids()[10:])
    pp.find_hvg(sc, hvg_no=30)
    pp.calculate_pca(sc, 8)
    graph.find_neighbours(sc)
    return sc


@pytest.fixture(scope="module")
def mc(sc):
    mc = metacells.generate_bt_meta_cells(sc, params=BT)
    assert mc is not None
    return mc


##############
# Generation #
##############


def test_aggregation_sums_the_member_counts(sc, mc):
    raw = mc.raw.scipy_csr().toarray()
    for i, members in enumerate(mc.obs()["original_cell_idx"].to_list()):
        counts = bv.get_counts(sc, cells=np.array(members, dtype=np.uint32))
        np.testing.assert_array_equal(raw[i], counts.scipy_csr().toarray().sum(axis=0))
    norm = mc.norm.scipy_csr().toarray()
    lib = raw.sum(axis=1, keepdims=True)
    expect = np.log1p(raw / lib * 1e5)
    np.testing.assert_allclose(norm, expect, rtol=1e-5)


def test_generation_leaves_the_source_alone(sc, mc):
    assert mc.shape == (12, sc.n_genes)
    assert mc.original_assignment.n_cells == sc.dims[0]
    np.testing.assert_array_equal(
        mc.original_assignment.cells_to_keep, sc.cells_to_keep
    )
    assert "original_cell_idx" not in sc.obs_columns()
    assert mc.obs()["meta_cell_id"].to_list()[:2] == ["meta_cell_01", "meta_cell_02"]
    members = mc.cell_assignments()
    assert members["cell_idx"].is_in(sc.cells_to_keep.tolist()).all()


def test_missing_knn_warns_and_returns_none(sc):
    sub = bv.SingleCellsSubset(sc, "cell_grp", "cell_type_1")
    with pytest.warns(UserWarning, match="No kNN data"):
        assert metacells.generate_bt_meta_cells(sub, params=BT) is None
    with pytest.warns(UserWarning, match="embedding was not found"):
        assert metacells.generate_supercells(sub, regenerate_knn=True) is None


def test_generator_on_a_subset_stays_in_the_subset(sc):
    sub = bv.SingleCellsSubset(sc, "cell_grp", "cell_type_1")
    pp.find_hvg(sub, hvg_no=30)
    pp.calculate_pca(sub, 6)
    graph.find_neighbours(sub, neighbours_params=NeighboursParams(k=5))
    mc = metacells.generate_supercells(
        sub, params=SupercellParams(graining_factor=5.0, k=5)
    )
    assert mc is not None
    cells = mc.cell_assignments()["cell_idx"].to_numpy()
    assert np.isin(cells, sub.subset_to_original).all()
    assert mc.n_cells == int(np.ceil(sub.n_cells / 5.0))


def test_stale_knn_is_refused(sc, mc):
    del mc
    sub = bv.SingleCellsSubset(sc, "cell_grp", "cell_type_2")
    pp.find_hvg(sub, hvg_no=30)
    pp.calculate_pca(sub, 6)
    graph.find_neighbours(sub, neighbours_params=NeighboursParams(k=5))
    pp.calculate_pca(sub, 5)
    with pytest.raises(ValueError, match="stale"):
        metacells.generate_bt_meta_cells(sub, params=BT)


#############
# Accessors #
#############


def test_counts_subset_by_ids_and_order(mc):
    dense = mc.raw.scipy_csr().toarray()
    ids = mc.cell_ids()[[3, 0]].tolist()
    genes = mc.var()["gene_id"].to_list()
    got = mc.counts(cells=ids, genes=[genes[5], genes[1], genes[2]])
    np.testing.assert_array_equal(
        got.scipy_csr().toarray(), dense[[3, 0]][:, [5, 1, 2]]
    )
    sub = got.scipy_csr()
    for r in range(sub.shape[0]):
        cols = sub.indices[sub.indptr[r] : sub.indptr[r + 1]]
        assert (np.diff(cols) > 0).all()


def test_add_obs_by_row_and_by_index(mc):
    m = _copy(mc)
    m.add_obs(pl.DataFrame({"score": np.arange(m.n_cells, dtype=np.float64)}))
    m.add_obs(pl.DataFrame({"meta_cell_idx": [2], "flag": [True]}))
    obs = m.obs()
    assert obs["score"].to_list() == list(range(m.n_cells))
    assert obs["flag"].to_list()[1:3] == [None, True]
    with pytest.raises(ValueError, match="protected"):
        m.add_obs(pl.DataFrame({"meta_cell_id": ["x"] * m.n_cells}))


def test_chain_on_metacells_stamps_and_clusters(mc):
    m = _copy(mc)
    pp.find_hvg(m, hvg_no=20)
    assert m.hvg is not None and len(m.hvg) == 20
    assert "var_std" in m.var().columns
    pp.calculate_pca(m, 4)
    graph.find_neighbours(m, neighbours_params=NeighboursParams(k=4))
    graph.find_clusters(m, name="cl")
    assert m.obs()["cl"].null_count() == 0
    status = m.cache_status()
    assert status.filter(pl.col("stale")).height == 0
    assert set(status["artefact"]) == {"pca", "knn", "connectivities"}


#########
# Merge #
#########


def _copy(mc: metacells.MetaCells, *, drop_genes: int = 0) -> metacells.MetaCells:
    """A fresh MetaCells over the same data, optionally without trailing
    genes."""
    if drop_genes == 0:
        return metacells.MetaCells(
            mc.obs(),
            mc.var(cols=["gene_idx", "gene_id"]),
            mc.raw,
            mc.norm,
            mc.original_assignment,
            mc.meta_cell_method,
        )
    keep = np.arange(mc.n_genes - drop_genes, dtype=np.uint32)
    return metacells.MetaCells(
        mc.obs(),
        mc.var(indices=keep, cols=["gene_idx", "gene_id"]),
        mc.counts(genes=keep, layer="raw"),
        mc.counts(genes=keep, layer="norm"),
        mc.original_assignment,
        mc.meta_cell_method,
    )


def test_merge_shared_parent_and_ids(mc):
    merged = metacells.merge_meta_cells({"a": mc, "b": _copy(mc)})
    assert merged.is_merged and merged.n_cells == 2 * mc.n_cells
    assert merged.obs()["meta_cell_id"][0] == "a__meta_cell_01"
    assert (
        merged.original_assignment.n_unassigned == mc.original_assignment.n_unassigned
    )
    np.testing.assert_array_equal(
        merged.raw.scipy_csr().toarray(),
        np.vstack([mc.raw.scipy_csr().toarray()] * 2),
    )
    with pytest.raises(ValueError, match="Duplicated"):
        metacells.merge_meta_cells([mc, _copy(mc)], prefix_ids=False)


def test_merge_feature_spaces(mc):
    short = _copy(mc, drop_genes=5)
    with pytest.warns(UserWarning, match="Dropping up to 5 genes"):
        inter = metacells.merge_meta_cells([mc, short])
    assert inter.n_genes == mc.n_genes - 5
    union = metacells.merge_meta_cells([short, mc], feature_space="union")
    assert union.n_genes == mc.n_genes
    dense = union.raw.scipy_csr().toarray()
    assert (dense[: mc.n_cells, -5:] == 0).all()
    np.testing.assert_array_equal(dense[mc.n_cells :], mc.raw.scipy_csr().toarray())


def test_merged_metacells_refuse_the_diffusion_map(sc, mc):
    merged = metacells.merge_meta_cells([mc, _copy(mc)])
    with pytest.warns(UserWarning, match="merged"):
        metacells.calc_diffusion_coordinates(merged, sc.cache.knn)
    assert "density_region" not in merged.obs_columns()


###########
# Metrics #
###########


def test_purity_single_label_and_length_check(sc, mc):
    m = _copy(mc)
    metacells.calc_meta_cell_purity(m, ["x"] * sc.dims[0], add_entropy=True)
    obs = m.obs()
    assert (obs["mc_purity"] == 1.0).all()
    assert (obs["mc_entropy"] == 0.0).all()
    with pytest.raises(ValueError, match="entries"):
        metacells.get_meta_cell_purity(m, ["x"] * 7)


def test_free_scores_on_tiny_groups():
    rng = np.random.default_rng(1)
    dcs = rng.normal(size=(12, 3)).astype(np.float32)
    one = metacells.calc_meta_cell_separation(dcs, ["a"] * 12)
    assert np.isinf(one["separation"][0])
    lone = ["a"] * 11 + ["b"]
    comp = metacells.calc_meta_cell_compactness(dcs, lone)
    assert comp["meta_cell"].to_list() == ["a", "b"]
    assert comp["compactness"][1] == 0.0
    gaps = ["a", None] * 6
    assert metacells.calc_meta_cell_compactness(dcs, gaps).height == 1


###############
# Persistence #
###############


def test_save_load_round_trip(tmp_path, sc, mc):
    m = _copy(mc)
    pp.find_hvg(m, hvg_no=20)
    pp.calculate_pca(m, 4)
    graph.find_neighbours(m, neighbours_params=NeighboursParams(k=4))
    metacells.calc_diffusion_coordinates(m, sc.cache.knn, n_dcs=4, k_density=10)
    m.other_data["table"] = pl.DataFrame({"a": [1, 2]})
    metacells.save_metacells(m, tmp_path / "mc")
    back = metacells.load_metacells(tmp_path / "mc")

    assert back.obs().equals(m.obs())
    assert back.var().equals(m.var())
    for layer in ("raw", "norm"):
        np.testing.assert_array_equal(getattr(back, layer).data, getattr(m, layer).data)
    np.testing.assert_array_equal(back.hvg, m.hvg)
    np.testing.assert_array_equal(back.cache.pca_factors, m.cache.pca_factors)
    assert back.cache.knn is not None and m.cache.knn is not None
    np.testing.assert_array_equal(back.cache.knn.indices, m.cache.knn.indices)
    assert back.cache_status().equals(m.cache_status())
    assert back.other_data["regions"] == m.other_data["regions"]
    table = back.other_data["table"]
    assert isinstance(table, pl.DataFrame)
    assert table.equals(m.other_data["table"])
    np.testing.assert_array_equal(
        back.original_assignment.cells_to_keep, m.original_assignment.cells_to_keep
    )
    assert back.meta_cell_method == m.meta_cell_method


def test_residual_fit_is_not_persisted(tmp_path, mc):
    m = _copy(mc)
    pp.fit_residuals(m, method="analytic_pearson")
    with pytest.warns(UserWarning, match="not persisted"):
        metacells.save_metacells(m, tmp_path / "mc")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = metacells.load_metacells(tmp_path / "mc")
    assert back.cache.residual_fit is None
