"""HALS NMF without R: shapes, getters, guards and the meta cell twins on
synthetic data.

Parity with the R package is in `test_nmf_r.py`.
"""

import numpy as np
import pytest

import bixverse as bv
from bixverse import metacells, pp
from bixverse import nmf as bvnmf
from bixverse.params import (
    BtMetacellsParams,
    MinQualityParams,
    NmfConsensusParams,
    NmfHalsParams,
    SyntheticDataParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
HALS = NmfHalsParams(max_iter=60)


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    """A small synthetic experiment with HVGs, PCA and a kNN."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=300, n_genes=60), seed=5
    )
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("nmf") / "sc", data.counts, data.obs, data.var, qc=QC
    )
    sc.set_cells_to_keep(sc.cell_ids()[5:])
    pp.find_hvg(sc, hvg_no=30)
    pp.calculate_pca(sc, 8)
    bv.graph.find_neighbours(sc, neighbours_params=bv.params.NeighboursParams(k=10))
    return sc


@pytest.fixture(scope="module")
def mc(sc):
    mc = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=40, k=8)
    )
    assert mc is not None
    pp.find_hvg(mc, hvg_no=30)
    return mc


def test_nmf_shapes_and_getters(sc):
    res = bvnmf.nmf(sc, 3)
    assert res.shape == (30, sc.n_cells, 3)
    assert (res.w >= 0).all() and (res.h >= 0).all()
    w = bvnmf.get_w(res)
    assert w.columns == ["gene_id", "comp_01", "comp_02", "comp_03"]
    obs = bvnmf.get_data(res)
    assert obs.columns == ["comp_01", "comp_02", "comp_03", "cell_idx"]
    assert obs["cell_idx"].to_list() == sc.cells_to_keep.tolist()
    sc.add_obs(obs)
    assert bvnmf.get_params(res)["k"] == 3
    assert repr(res).startswith("NmfResult(") and "k=3" in repr(res)


def test_nmf_is_seeded(sc):
    p = NmfHalsParams(nmf_init="random", max_iter=40)
    a = bvnmf.nmf(sc, 3, nmf_hals_params=p, seed=1)
    b = bvnmf.nmf(sc, 3, nmf_hals_params=p, seed=1)
    np.testing.assert_array_equal(a.w, b.w)


def test_nmf_needs_genes(tmp_path):
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=60, n_genes=30), seed=2
    )
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    with pytest.raises(ValueError, match="No HVGs"):
        bvnmf.nmf(sc, 2)
    assert bvnmf.nmf(sc, 2, gene_ids=sc.gene_ids()[:10].tolist()).shape[0] == 10


def test_stabilised_nmf_best_run(sc):
    res = bvnmf.stabilised_nmf(sc, 3, n_runs=3, nmf_hals_params=HALS)
    assert res.w_all.shape == (30, 9)
    assert len(bvnmf.get_h(res)) == 3
    best = bvnmf.get_best_run(res)
    assert best.n_iter is None
    np.testing.assert_array_equal(best.h, res.h_per_run[res.best_idx])
    assert best.final_loss == res.losses.min()
    assert f"best_run={res.best_idx}" in repr(res)


def test_consensus_nmf_frames(sc):
    res = bvnmf.consensus_nmf(sc, 3, n_runs=4, nmf_hals_params=HALS)
    assert res.clusters.height == 12
    assert res.clusters["component_id"][0] == "run_01.comp_01"
    assert res.cluster_sizes["n"].sum() == int(res.clusters["kept"].sum())
    stab = bvnmf.get_stability(res)
    assert stab["stability"] == res.stability
    assert repr(res).startswith("ConsensusNmfResult(")


def test_consensus_nmf_guards(sc):
    with pytest.raises(ValueError, match="k must be at least 2"):
        bvnmf.consensus_nmf(sc, 1)
    with pytest.raises(ValueError, match="n_runs"):
        bvnmf.consensus_nmf(sc, 2, n_runs=1)
    with pytest.raises(ValueError, match="k_range"):
        bvnmf.nmf_k_sweep(sc, [1, 2])
    with pytest.raises(ValueError, match="k_range"):
        bvnmf.nmf_k_sweep(sc, [2, 2])


def test_k_sweep_nulls_failed_stability(sc):
    # a density filter that drops everything makes every k fail
    tight = NmfConsensusParams(density_threshold=0.0)
    with pytest.warns(UserWarning, match="consensus step failed"):
        res = bvnmf.nmf_k_sweep(
            sc, [2, 3], n_runs=3, nmf_hals_params=HALS, nmf_consensus_params=tight
        )
    assert res.frame["stability"].null_count() == 2
    assert "best_k=None" in repr(res)
    with pytest.raises(ValueError, match="density_threshold"):
        bvnmf.consensus_nmf(
            sc, 2, n_runs=3, nmf_hals_params=HALS, nmf_consensus_params=tight
        )


def test_meta_cell_twins(mc):
    res = bvnmf.nmf(mc, 3)
    assert res.source_class == "MetaCells"
    assert bvnmf.get_data(res).columns[-1] == "meta_cell_idx"
    raw = bvnmf.nmf(mc, 3, use_second_layer=False)
    assert not np.allclose(raw.w, res.w)
    sweep = bvnmf.nmf_k_sweep(mc, [2, 3], n_runs=3, nmf_hals_params=HALS)
    assert sweep.frame["k"].to_list() == [2, 3]
