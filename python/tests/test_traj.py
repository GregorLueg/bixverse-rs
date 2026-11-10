"""Palantir, PAGA, gene trends and Bonsai without R: structure, guards and
the object plumbing on a planted Y-shaped kNN graph.

Parity with the R package is in `test_traj_r.py`.
"""

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import metacells, pp, traj, trees
from bixverse.params import (
    BonsaiParams,
    BtMetacellsParams,
    GeneTrendsParams,
    MinQualityParams,
    PalantirParams,
    SyntheticDataParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
PALANTIR = PalantirParams(knn=20, num_waypoints=120, n_eigs=3, knn_method="exhaustive")
K = 12


def _exact_knn(points: np.ndarray, k: int) -> bv.KnnGraph:
    """Exact Euclidean kNN without self, by brute force."""
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return bv.KnnGraph(
        indices=idx.astype(np.uint32),
        distances=np.take_along_axis(d, idx, axis=1).astype(np.float32),
        dist_metric="euclidean",
    )


def _y_manifold(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A noisy Y: a trunk then two arms. Returns the points and the 0-based
    segment of each (0 trunk, 1 and 2 the arms)."""
    arm = n // 4
    trunk = n - 2 * arm
    t_trunk = np.arange(trunk) / trunk * 5
    t_arm = np.arange(1, arm + 1) / arm * 5
    pts = np.vstack(
        [
            np.column_stack([t_trunk, rng.uniform(-0.15, 0.15, trunk)]),
            np.column_stack([5 + t_arm, 1.4 * t_arm + rng.uniform(-0.15, 0.15, arm)]),
            np.column_stack([5 + t_arm, -1.4 * t_arm + rng.uniform(-0.15, 0.15, arm)]),
        ]
    )
    seg = np.repeat([0, 1, 2], [trunk, arm, arm])
    return pts, seg


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    """A small synthetic experiment with a planted Y kNN and segment labels."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=260, n_genes=50), seed=3
    )
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("traj") / "sc", data.counts, data.obs, data.var, qc=QC
    )
    sc.set_cells_to_keep(sc.cell_ids()[10:])
    pts, seg = _y_manifold(sc.n_cells, np.random.default_rng(1))
    sc.cache.knn = _exact_knn(pts, K)
    sc.stamp("knn")
    sc.add_obs(
        pl.DataFrame(
            {
                "cell_idx": sc.cells_to_keep,
                "segment": pl.Series([f"seg_{s}" for s in seg]).cast(
                    pl.Enum(["seg_0", "seg_1", "seg_2", "seg_empty"])
                ),
                "position": pts[:, 0],
            }
        )
    )
    return sc


@pytest.fixture(scope="module")
def pal(sc):
    res = traj.run_palantir(sc, sc.cell_ids(filtered=True)[0], palantir_params=PALANTIR)
    assert res is not None
    return res


@pytest.fixture(scope="module")
def mc(sc):
    mc = metacells.generate_bt_meta_cells(
        sc, params=BtMetacellsParams(target_no_metacells=40, k=8)
    )
    assert mc is not None
    pts = np.column_stack(
        [
            np.linspace(0, 10, mc.n_cells),
            np.random.default_rng(2).uniform(-0.3, 0.3, mc.n_cells),
        ]
    )
    mc.cache.knn = _exact_knn(pts, 6)
    mc.stamp("knn")
    return mc


@pytest.fixture(scope="module")
def tree(sc):
    return trees.bonsai(sc)


############
# Palantir #
############


def test_palantir_follows_the_planted_y(sc, pal):
    cells = sc.cell_ids(filtered=True).tolist()
    assert pal.pseudotime["cell_id"].to_list() == cells
    pt = pal.pseudotime["pseudotime"].to_numpy()
    assert pt.min() >= 0 and pt.max() <= 1
    assert pal.start_cell == cells[0]
    assert pal.waypoints[0] == pal.start_cell
    position = sc.obs(cols=["position"], filtered=True)["position"].to_numpy()
    assert np.corrcoef(pt, position)[0, 1] > 0.9
    assert pal.branch_probs.columns == ["cell_id", *pal.branches]
    assert set(pal.terminal_states.values()) <= set(cells)
    assert pal.multiscale.columns[1:] == ["MS_1", "MS_2"]
    assert "PalantirRes" in repr(pal)


def test_palantir_labels_name_the_branches(sc, pal):
    labelled = {f"fate_{i}": c for i, c in enumerate(pal.terminal_states.values())}
    res = traj.run_palantir(
        sc,
        pal.start_cell,
        terminal_states=labelled,
        palantir_params=PALANTIR,
    )
    assert res is not None
    assert sorted(res.branches) == sorted(labelled)
    assert res.terminal_states == {k: labelled[k] for k in res.branches}
    plain = traj.run_palantir(
        sc,
        pal.start_cell,
        terminal_states=list(labelled.values()),
        palantir_params=PALANTIR,
    )
    assert plain is not None
    assert plain.branches == list(plain.terminal_states.values())


def test_palantir_guards(sc):
    with pytest.raises(ValueError, match="not found in the kNN graph"):
        traj.run_palantir(sc, "not_a_cell")
    with pytest.raises(ValueError, match="SingleCellsMultiModal"):
        traj.run_palantir(sc, sc.cell_ids(filtered=True)[0], modality="adt")


def test_palantir_without_knn_warns(sc):
    sub = bv.SingleCellsSubset(sc, "cell_grp", "cell_type_1")
    with pytest.warns(UserWarning, match="No kNN data"):
        assert traj.run_palantir(sub, sub.cell_ids(filtered=True)[0]) is None
    with pytest.warns(UserWarning, match="No kNN data"):
        assert traj.run_paga(sub, "cell_grp") is None


def test_palantir_metacells(mc):
    ids = mc.cell_ids().tolist()
    res = traj.run_palantir(
        mc,
        ids[0],
        terminal_states=[ids[-1]],
        palantir_params=PalantirParams(
            knn=8, num_waypoints=20, n_eigs=3, knn_method="exhaustive"
        ),
    )
    assert res is not None
    assert res.pseudotime["cell_id"].to_list() == ids
    assert res.branches == [ids[-1]]


########
# PAGA #
########


def test_paga_keeps_empty_enum_levels(sc):
    res = traj.run_paga(sc, "segment")
    assert res is not None
    assert res.clusters == ("seg_0", "seg_1", "seg_2", "seg_empty")
    assert res.sizes["n_cells"].to_list()[-1] == 0
    assert int(res.sizes["n_cells"].sum()) == sc.n_cells
    c = res.connectivities
    np.testing.assert_array_equal(c, c.T)
    assert np.all(np.diag(c) == 0)
    assert c.max() <= 1
    # the arms touch the trunk, and each other at the branch point
    assert c[0, 1] > 0 and c[0, 2] > 0 and c[1, 2] > 0
    t = res.connectivities_tree
    assert np.count_nonzero(t) / 2 == 2
    # the maximum spanning tree drops the weakest edge of the triangle
    edges = [(0, 1), (0, 2), (1, 2)]
    weakest = min(edges, key=lambda e: c[e])
    assert t[weakest] == 0
    assert all(t[e] > 0 for e in edges if e != weakest)
    assert np.all(t[t > 0] == c[t > 0])
    assert "PagaRes" in repr(res)


def test_paga_missing_column_raises(sc):
    with pytest.raises(ValueError, match="not found"):
        traj.run_paga(sc, "not_a_column")


###############
# Gene trends #
###############


def test_gene_trends_shape(sc, pal):
    genes = sc.gene_ids()[:4].tolist()
    res = traj.run_gene_trends(
        sc, pal, features=genes, gene_trend_params=GeneTrendsParams(resolution=20)
    )
    assert res.trends.height == 20 * len(genes) * len(pal.branches)
    assert (
        res.trends["gene"].cast(pl.String).unique(maintain_order=True).to_list()
        == genes
    )
    assert list(res.branches) == pal.branches
    assert np.isfinite(res.trends["expression"].to_numpy()).all()
    n_cells = res.run_info["n_cells"]
    assert isinstance(n_cells, dict)
    assert {b: len(c) for b, c in res.branch_cells.items()} == n_cells
    assert "GeneTrendsRes" in repr(res)


def test_gene_trends_guards(sc, pal, mc):
    with pytest.raises(ValueError, match="only optional"):
        traj.run_gene_trends(sc, pal)
    with pytest.raises(ValueError, match="missing from the Palantir"):
        traj.run_gene_trends(mc, pal, features=sc.gene_ids()[:2].tolist())


def test_gene_trends_magic(sc, pal):
    genes = sc.gene_ids()[:3].tolist()
    pp.run_magic(sc, genes)
    res = traj.run_gene_trends(
        sc, pal, use_magic=True, gene_trend_params=GeneTrendsParams(resolution=15)
    )
    assert (
        res.trends["gene"].cast(pl.String).unique(maintain_order=True).to_list()
        == genes
    )
    with pytest.raises(ValueError, match="not in the imputed layer"):
        traj.run_gene_trends(sc, pal, features=[sc.gene_ids()[10]], use_magic=True)


def test_gene_trends_metacells(mc):
    ids = mc.cell_ids().tolist()
    pal = traj.run_palantir(
        mc,
        ids[0],
        terminal_states=[ids[-1]],
        palantir_params=PalantirParams(
            knn=8, num_waypoints=20, n_eigs=3, knn_method="exhaustive"
        ),
    )
    assert pal is not None
    res = traj.run_gene_trends(
        mc,
        pal,
        features=mc.gene_ids()[:3].tolist(),
        gene_trend_params=GeneTrendsParams(resolution=10),
    )
    assert res.trends.height == 10 * 3
    with pytest.raises(ValueError, match="MAGIC is only available"):
        traj.run_gene_trends(mc, pal, use_magic=True)


##########
# Bonsai #
##########


def test_bonsai_tree_structure(sc, tree):
    nodes = tree.nodes
    leaves = nodes.filter(pl.col("is_leaf"))
    assert leaves["cell_id"].to_list() == sc.cell_ids(filtered=True).tolist()
    assert leaves["node"].to_list() == list(range(sc.n_cells))
    assert nodes["parent"].null_count() == 1
    child = nodes.filter(pl.col("parent").is_not_null())
    assert (child["parent"] > child["node"]).all()
    assert np.isfinite(nodes.select("x", "y").to_numpy()).all()
    assert sorted(tree.genes_used + tree.genes_dropped) == sorted(
        sc.gene_ids().tolist()
    )
    assert tree.timings["stage"].to_list() == [
        "sanity",
        "ingest",
        "bonsai",
        "layout",
        "total",
    ]
    np.testing.assert_array_equal(tree.cell_idx, sc.cells_to_keep)
    assert "BonsaiTree" in repr(tree)


def test_bonsai_candidates_and_guards(sc):
    res = trees.bonsai(sc, hvg=[5, 1, 3, 1, 20, 30, 40])
    assert set(res.genes_used) <= set(sc.gene_ids([1, 3, 5, 20, 30, 40]).tolist())
    with pytest.raises(ValueError, match="outside"):
        trees.bonsai(sc, hvg=[sc.n_genes])
    with pytest.raises(ValueError, match="fixed_variance"):
        trees.bonsai(sc, bonsai_params=BonsaiParams(variance_rule="fixed"))


def test_relayout_keeps_the_leaves(tree):
    res = trees.relayout_bonsai(tree, layout="dendrogram")
    assert res.layout == "dendrogram"
    assert tree.layout == "equal_angle"
    leaves = res.nodes.filter(pl.col("is_leaf"))
    assert (
        leaves["cell_id"].to_list()
        == tree.nodes.filter(pl.col("is_leaf"))["cell_id"].to_list()
    )
    assert res.nodes.height == tree.nodes.height
    np.testing.assert_allclose(
        float(res.nodes["branch"].sum()), float(tree.nodes["branch"].sum()), rtol=1e-12
    )


def test_bonsai_embedding(sc, tree):
    trees.set_bonsai_embedding(sc, tree, name="bonsai_test")
    emb = sc.cache.embeddings["bonsai_test"]
    assert emb.shape == (sc.n_cells, 2)
    assert sc.check_state(["bonsai_test"]) is True


def test_bonsai_embedding_refuses_other_cells(tmp_path, tree):
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=60, n_genes=20), seed=4
    )
    other = bv.io.load_csr(tmp_path / "o", data.counts, data.obs, data.var, qc=QC)
    with pytest.raises(ValueError, match="not built over the current cells"):
        trees.set_bonsai_embedding(other, tree)


def test_bonsai_metacells(mc):
    res = trees.bonsai(mc)
    leaves = res.nodes.filter(pl.col("is_leaf"))
    assert leaves["cell_id"].to_list() == mc.cell_ids().tolist()
    assert leaves["n_cells"].to_list() == mc.obs()["no_originating_cells"].to_list()
    trees.set_bonsai_embedding(mc, res)
    assert mc.cache.embeddings["bonsai"].shape == (mc.n_cells, 2)
