"""Multi-modal without R: ADT handling, modality routing, provenance across
caches and persistence.

Parity with the R methods is in `test_mm_r.py`.
"""

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import embed, graph, integrate, mm, pp, traj
from bixverse._core.db import TableNotFoundError
from bixverse.params import (
    DsbParams,
    MinQualityParams,
    NeighboursParams,
    SyntheticDataAdtParams,
    SyntheticDataParams,
    WnnParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
N_CELLS = 240
WNN = WnnParams(knn_range=60)
NEIGHBOURS = NeighboursParams(k=10, knn_method="exhaustive")


@pytest.fixture(scope="module")
def data():
    """Small synthetic RNA, two batches."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=N_CELLS, n_genes=60, n_batches=2), seed=5
    )


@pytest.fixture(scope="module")
def adt():
    """The matching ADT counts."""
    return bv.io.generate_single_cell_test_data_adt(
        SyntheticDataAdtParams(n_cells=N_CELLS), seed=5
    )


def _add(sc: bv.SingleCellsMultiModal, adt, **kwargs) -> None:
    mm.add_adt_counts(
        sc, adt.counts, adt.obs["cell_id"], adt.var["protein_id"], **kwargs
    )


@pytest.fixture
def sc(data, adt, tmp_path):
    """A multi-modal experiment with RNA and ADT PCA."""
    bv.io.load_csr(tmp_path / "mm", data.counts, data.obs, data.var, qc=QC)
    sc = bv.SingleCellsMultiModal(tmp_path / "mm")
    _add(sc, adt)
    pp.find_hvg(sc, hvg_no=40)
    pp.calculate_pca(sc, 10)
    mm.calculate_pca_adt(sc, 8)
    return sc


#######
# ADT #
#######


def test_add_adt_counts_writes_r_tables(sc, adt):
    a = sc.adt
    assert a is not None
    assert a.raw_counts.shape == (N_CELLS, 15)
    assert mm.get_adt_names(sc) == adt.var["protein_id"].to_list()
    var_adt = sc.var(modality="adt")
    assert var_adt.columns == ["feature_idx", "feature_id", "nnz"]
    assert var_adt["feature_idx"].to_list() == list(range(15))
    np.testing.assert_array_equal(
        var_adt["nnz"].to_numpy(), (adt.counts > 0).sum(axis=0)
    )
    obs = sc.obs()
    np.testing.assert_allclose(obs["adt_lib_size"].to_numpy(), adt.counts.sum(axis=1))
    assert mm.get_adt_sample_info(sc)["cell_idx"].to_list() == list(range(N_CELLS))


def test_add_adt_counts_needs_every_kept_cell(data, adt, tmp_path):
    bv.io.load_csr(tmp_path / "mm", data.counts, data.obs, data.var, qc=QC)
    sc = bv.SingleCellsMultiModal(tmp_path / "mm")
    with pytest.raises(ValueError, match="not in the barcodes"):
        mm.add_adt_counts(
            sc, adt.counts[1:], adt.obs["cell_id"][1:], adt.var["protein_id"]
        )


def test_var_adt_absent_before_adt(data, tmp_path):
    bv.io.load_csr(tmp_path / "mm", data.counts, data.obs, data.var, qc=QC)
    sc = bv.SingleCellsMultiModal(tmp_path / "mm")
    with pytest.raises(TableNotFoundError):
        sc.var(modality="adt")
    with pytest.warns(UserWarning, match="No ADT counts"):
        assert mm.get_adt_names(sc) is None


def test_get_adt_counts_selections(sc, adt):
    full = mm.get_adt_counts(sc)
    np.testing.assert_array_equal(full, adt.counts)
    sub = mm.get_adt_counts(
        sc, layer="norm", cells=sc.cell_ids([4, 1]).tolist(), features=["protein_03"]
    )
    assert sc.adt is not None
    np.testing.assert_array_equal(sub, sc.adt.norm_counts[[4, 1]][:, [2]])
    with pytest.warns(UserWarning, match="not found"):
        mm.get_adt_counts(sc, features=["protein_01", "nope"])

    counts = bv.get_counts(sc, assay="adt", cells=[3, 0], genes=[0, 4])
    np.testing.assert_array_equal(counts.cells, [3, 0])
    np.testing.assert_array_equal(counts.genes, [0, 4])
    m = counts.matrix
    dense = np.zeros(m.shape)
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    dense[rows, m.indices] = m.data
    np.testing.assert_allclose(dense, adt.counts[[3, 0]][:, [0, 4]])


def test_get_counts_adt_needs_multi_modal(data, tmp_path):
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    with pytest.raises(ValueError, match="SingleCellsMultiModal"):
        bv.get_counts(sc, assay="adt")


def test_adt_rows_follow_cells_not_store(sc):
    keep = sc.cells_to_keep[::-1][:100]
    with pytest.warns(UserWarning, match="rna:pca, adt:pca"):
        sc.set_cells_to_keep(keep)
    got = mm.get_adt_counts(sc)
    assert sc.adt is not None
    np.testing.assert_array_equal(got, sc.adt.raw_counts[np.sort(keep)])


def test_dsb_checks(adt, sc):
    info = pl.DataFrame(
        {"cell_id": sc.cell_ids(filtered=True), "cell_idx": sc.cells_to_keep}
    )
    args = (adt.counts, adt.obs["cell_id"], adt.var["protein_id"], info)
    with pytest.raises(ValueError, match="isotype_names is None"):
        mm.new_adt_counts_dsb(*args)
    with pytest.raises(ValueError, match="not in feature_ids"):
        mm.new_adt_counts_dsb(*args, isotype_names=["nope"])
    res = mm.new_adt_counts_dsb(*args, dsb_params=DsbParams(use_isotype_controls=False))
    assert res.dsb is not None
    assert res.dsb.protein_background_sd.size == 0
    assert res.dsb.technical_component.shape == (N_CELLS,)


#######
# PCA #
#######


def test_calculate_pca_adt(sc):
    c = sc.adt_cache
    assert c.pca_factors is not None
    assert c.pca_factors.shape == (N_CELLS, 8)
    assert c.pca_loadings is not None
    assert c.pca_loadings.shape == (15, 8)
    mm.calculate_pca_adt(sc, 4, features=[f"protein_{i:02d}" for i in range(1, 13)])
    assert sc.adt_cache.pca_loadings is not None
    assert sc.adt_cache.pca_loadings.shape == (12, 4)
    with pytest.raises(ValueError, match="no_pcs"):
        mm.calculate_pca_adt(sc, 16)


def test_calculate_pca_adt_refuses_cells_without_adt(data, adt, tmp_path):
    bv.io.load_csr(tmp_path / "mm", data.counts, data.obs, data.var, qc=QC)
    sc = bv.SingleCellsMultiModal(tmp_path / "mm")
    sc.set_cells_to_keep(np.arange(100))
    _add(sc, adt)
    assert sc.adt is not None
    assert sc.adt.raw_counts.shape[0] == 100
    sc.reset_cells_to_keep(force=True)
    with pytest.raises(ValueError, match="no ADT counts"), pytest.warns(UserWarning):
        mm.calculate_pca_adt(sc, 5)


####################
# Modality routing #
####################


def test_modalities_need_a_multi_modal_object(data, tmp_path):
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    pp.find_hvg(sc, hvg_no=40)
    pp.calculate_pca(sc, 10)
    with pytest.raises(ValueError, match="SingleCellsMultiModal"):
        graph.find_neighbours(sc, modality="adt")
    with pytest.raises(ValueError, match="SingleCellsMultiModal"):
        graph.find_clusters(sc, modality="wnn")
    with pytest.raises(ValueError, match="SingleCellsMultiModal"):
        traj.run_paga(sc, "cell_grp", modality="adt")


def test_wnn_modality_before_the_integration(sc):
    with pytest.raises(ValueError, match="No 'wnn' integration"):
        graph.find_clusters(sc, modality="wnn")
    with pytest.raises(ValueError, match="No 'wnn' integration"):
        mm.get_wnn_weights(sc)


def test_adt_neighbours_stay_in_the_adt_cache(sc):
    graph.find_neighbours(sc, modality="adt", neighbours_params=NEIGHBOURS)
    assert sc.cache.knn is None
    assert sc.adt_cache.knn is not None
    assert sc.adt_cache.knn.indices.shape == (N_CELLS, 10)
    graph.find_clusters(sc, modality="adt", name="adt_clusters")
    assert sc.obs()["adt_clusters"].n_unique() > 1
    graph_adt = graph.get_connectivities(sc, modality="adt")
    assert graph_adt is sc.adt_cache.connectivities


def test_wnn_graph_and_its_consumers(sc):
    mm.generate_wnn_graph(sc, wnn_params=WNN)
    w = sc.wnn
    assert w is not None
    assert w.cache.knn is not None
    assert w.cache.knn.indices.shape == (N_CELLS, WNN.k_nn)
    weights = mm.get_wnn_weights(sc)
    np.testing.assert_allclose(
        weights["weight_modality_1"] + weights["weight_modality_2"], 1.0, rtol=1e-5
    )
    graph.find_clusters(sc, modality="wnn", name="wnn_clusters")
    assert sc.obs()["wnn_clusters"].n_unique() > 1

    embed.umap(sc, modality="wnn", slot_name="wnn_umap")
    assert "wnn_umap" in w.cache.embeddings
    assert "wnn_umap" not in sc.cache.embeddings
    status = sc.cache_status()
    umap = status.filter(pl.col("artefact") == "wnn_umap")
    assert umap["modality"].to_list() == ["wnn"]
    assert umap["parents"].to_list() == [
        [sc.cache.stamps["pca"].id, w.cache.stamps["knn"].id]
    ]

    pp.run_magic(sc, sc.gene_ids([0, 1]).tolist(), modality="wnn")
    assert sc.cache.magic is not None
    res = traj.run_paga(sc, "cell_grp", modality="wnn")
    assert res is not None


def test_sigma_idx_is_honoured(sc):
    mm.generate_wnn_graph(sc, wnn_params=WNN)
    assert sc.wnn is not None
    snn = sc.wnn.weights
    mm.generate_wnn_graph(
        sc, wnn_params=WnnParams(knn_range=60, sigma_method="sigma_idx")
    )
    assert not snn.equals(mm.get_wnn_weights(sc))


def test_harmony_on_adt(sc):
    integrate.harmony(sc, "batch_index", modality="adt")
    assert "harmony" in sc.adt_cache.embeddings
    assert "harmony" not in sc.cache.embeddings
    assert sc.check_state(["harmony"], modality="adt") is True


##############
# Provenance #
##############


def test_cell_filter_stales_every_modality(sc):
    mm.generate_wnn_graph(sc, wnn_params=WNN)
    with pytest.warns(UserWarning, match="adt:pca"):
        sc.set_cells_to_keep(sc.cells_to_keep[:200])
    status = sc.cache_status()
    assert status.filter(pl.col("modality") == "wnn")["stale"].all()
    with pytest.raises(ValueError, match="stale"):
        graph.find_neighbours(sc, modality="adt", neighbours_params=NEIGHBOURS)

    mm.calculate_pca_adt(sc, 8)
    knn = sc.cache_status().filter(
        (pl.col("modality") == "wnn") & (pl.col("artefact") == "knn")
    )
    assert knn["stale"].to_list() == [True]


def test_reset_drops_every_cache(sc):
    mm.generate_wnn_graph(sc, wnn_params=WNN)
    with pytest.raises(ValueError, match="adt:pca"):
        sc.reset_cells_to_keep()
    sc.reset_cells_to_keep(force=True)
    assert sc.adt_cache.pca_factors is None
    assert sc.wnn is None
    assert sc.adt is not None


###############
# Persistence #
###############


def test_save_and_load_round_trip(sc, adt):
    _add(sc, adt, method="dsb", isotype_names=["protein_13", "protein_14"])
    mm.calculate_pca_adt(sc, 8)
    mm.generate_wnn_graph(sc, wnn_params=WNN)
    graph.find_neighbours(sc, modality="adt", neighbours_params=NEIGHBOURS)
    bv.io.save(sc)

    back = bv.io.load_existing(sc.dir)
    assert isinstance(back, bv.SingleCellsMultiModal)
    assert back.adt is not None
    assert sc.adt is not None
    assert back.adt.type == "DSB"
    assert back.adt.dsb is not None
    assert back.adt.dsb.dsb_params == DsbParams()
    assert back.adt.dsb.isotype_names == ("protein_13", "protein_14")
    np.testing.assert_array_equal(back.adt.norm_counts, sc.adt.norm_counts)
    assert back.adt.cell_ids.tolist() == sc.adt.cell_ids.tolist()
    assert back.wnn is not None
    assert sc.wnn is not None
    assert back.wnn.weights.equals(sc.wnn.weights)
    assert back.wnn.modality_2 == "adt"
    assert back.wnn.cache.knn is not None
    assert sc.wnn.cache.knn is not None
    np.testing.assert_array_equal(back.wnn.cache.knn.indices, sc.wnn.cache.knn.indices)
    assert back.cache_status().equals(sc.cache_status())


def test_load_existing_multi_modal_flag(data, tmp_path):
    bv.io.load_csr(tmp_path / "mm", data.counts, data.obs, data.var, qc=QC)
    assert type(bv.io.load_existing(tmp_path / "mm")) is bv.SingleCells
    sc = bv.io.load_existing(tmp_path / "mm", multi_modal=True)
    assert isinstance(sc, bv.SingleCellsMultiModal)
    assert sc.adt is None
