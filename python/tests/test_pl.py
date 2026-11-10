"""Plot data extractors and the matplotlib plots without R: guards, shapes
and what each plot draws (points, collections, labels, colours, order), on
the Agg backend. No image comparison.

Extractor parity with the R package is in `test_pl_r.py`. The plot tests skip
without matplotlib.
"""

import dataclasses
import importlib
import sys

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import de, graph, pp, qc, traj
from bixverse import pl as bvpl
from bixverse.params import MinQualityParams, SyntheticDataParams

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
GENES = ["gene_03", "gene_01", "gene_07"]


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    """A small experiment with PCA, kNN, Leiden and an Enum obs column with
    an unused level."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=300, n_genes=40, n_samples=4, sample_bias="even"),
        seed=11,
    )
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("pl") / "sc", data.counts, data.obs, data.var, qc=QC
    )
    sc.set_cells_to_keep(sc.cell_ids()[5:])
    pp.find_hvg(sc, hvg_no=30)
    pp.calculate_pca(sc, no_pcs=5)
    graph.find_neighbours(sc)
    graph.find_clusters(sc, res=0.8)
    grp = sc.obs(cols=["cell_idx", "cell_grp"], filtered=True)
    sc.add_obs(
        grp.select(
            "cell_idx",
            grp_enum=pl.col("cell_grp").cast(
                pl.Enum(["cell_type_3", "cell_type_1", "cell_type_2", "unused"])
            ),
        )
    )
    return sc


@pytest.fixture(scope="module")
def milo(sc):
    milo = de.get_milor_abundances(
        sc, "sample_id", milor_params=de.MiloRParams(k_refine=5)
    )
    assert milo is not None
    design = pl.DataFrame(
        {"sample_id": list(milo.samples), "condition": ["a", "a", "b", "b"]}
    )
    return de.test_nhoods(milo, "~ condition", design)


@pytest.fixture(scope="module")
def paga(sc):
    # samples are mixed, so the graph has edges
    res = traj.run_paga(sc, "sample_id")
    assert res is not None
    return res


##############
# Extractors #
##############


def test_extract_rows_follow_kept_order(sc):
    frame = bvpl.extract_gene_expression(sc, GENES, obs_cols=["cell_grp"])
    assert frame.columns == ["cell_id", *GENES, "cell_grp"]
    assert frame["cell_id"].to_list() == sc.cell_ids(filtered=True).tolist()
    obs = sc.obs(cols=["cell_id", "cell_grp"], filtered=True)
    assert frame.select("cell_id", "cell_grp").equals(obs)


def test_extract_dot_plot_drops_unused_levels_and_missing_groups(sc):
    dot = bvpl.extract_dot_plot_data(sc, GENES, "grp_enum")
    assert dot["group"].dtype == pl.Enum(["cell_type_3", "cell_type_1", "cell_type_2"])
    assert dot["gene"].dtype == pl.Enum(GENES)
    assert dot.height == 9
    assert dot["scaled_exp"].is_between(0, 1).all()
    assert dot["pct_exp"].is_between(0, 100).all()


def test_extract_guards(sc, milo, paga):
    with (
        pytest.warns(UserWarning, match="could not be matched"),
        pytest.raises(ValueError, match="None of the provided gene_ids"),
    ):
        bvpl.extract_gene_expression(sc, ["nope"])
    with pytest.raises(ValueError, match="only supported for SingleCellsMultiModal"):
        bvpl.extract_gene_expression(sc, GENES, modality="adt")
    with pytest.raises(ValueError, match="No imputed layer"):
        bvpl.extract_gene_expression(sc, GENES, layer="magic")
    with pytest.raises(ValueError, match="was not found"):
        bvpl.extract_embedding_data(sc, "umap")
    with pytest.raises(ValueError, match="No differential abundance"):
        bvpl.extract_milo_plot_data(sc, dataclasses.replace(milo, nhoods_info=None))
    with pytest.raises(ValueError, match="PAGA was run on"):
        bvpl.extract_paga_plot_data(sc, paga, embedding="pca", cluster_col="cell_grp")


def test_extract_feature_pair_suffixes(sc):
    pair = bvpl.extract_feature_pair(
        sc, "gene_03_rna", "gene_01", obs_cols=["cell_grp"]
    )
    expr = bvpl.extract_gene_expression(sc, ["gene_03", "gene_01"])
    np.testing.assert_array_equal(pair["feature_1"], expr["gene_03"])
    np.testing.assert_array_equal(pair["feature_2"], expr["gene_01"])
    assert pair.columns == ["cell_id", "feature_1", "feature_2", "cell_grp"]


def test_extract_milo_edges_are_shared_cells(sc, milo):
    data = bvpl.extract_milo_plot_data(sc, milo, embedding="pca", overlap=2)
    nhoods = milo.nhoods
    rows = np.repeat(
        np.arange(nhoods.shape[0]), np.diff(nhoods.indptr.astype(np.int64))
    )
    dense = np.zeros(nhoods.shape)
    dense[rows, nhoods.indices] = 1
    shared = dense.T @ dense
    for a, b, w in data.edges.select("from", "to", "weight").iter_rows():
        assert a < b and w == shared[a, b] and w >= 2
    assert data.edges.height == int((np.triu(shared, 1) >= 2).sum())


#########
# Plots #
#########


@pytest.fixture(scope="module")
def mpl():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    yield mpl
    import matplotlib.pyplot as plt

    plt.close("all")


def _points(ax) -> list:
    from matplotlib.collections import PathCollection

    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _legends(ax) -> list:
    from matplotlib.legend import Legend

    return [a for a in ax.artists if isinstance(a, Legend)]


def _top_down(ax) -> list[str]:
    """y tick labels from the top of the axes down."""
    ticks = sorted(
        zip(ax.get_yticks(), ax.get_yticklabels(), strict=True), key=lambda t: -t[0]
    )
    return [t.get_text() for _, t in ticks]


def test_missing_matplotlib_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "bixverse._core.plots", raising=False)
    with pytest.raises(ImportError, match=r"bixverse\[plots\]"):
        importlib.import_module("bixverse._core.plots")
    with pytest.raises(ImportError, match=r"bixverse\[plots\]"):
        bvpl.embedding_plot  # noqa: B018
    monkeypatch.undo()
    assert bvpl.extract_embedding_data is not None


def test_bx_colors(mpl):
    main = bvpl.bx_colors("main", n=7)
    assert bvpl.bx_colors("main", n=5) == [main[i] for i in (0, 1, 2, 3, 5)]
    eight = bvpl.bx_colors("main", n=8)
    assert len(eight) == 8 and eight[0] == main[0] and eight[-1] == main[-1]
    assert (
        bvpl.bx_colors("diverging", n=10, reverse=True)[0]
        == bvpl.bx_colors("diverging", n=10)[-1]
    )
    assert len(bvpl.bx_colors("sequential", n=12)) == 12


def test_embedding_plot_discrete(mpl, sc):
    from matplotlib.colors import to_hex

    ax = bvpl.embedding_plot(sc, "pca", "grp_enum", label_by="cell_grp")
    (pts,) = _points(ax)
    assert len(pts.get_offsets()) == sc.n_cells
    grp = bvpl.extract_embedding_data(sc, "pca", obs_cols=["grp_enum"])["grp_enum"]
    levels = ["cell_type_3", "cell_type_1", "cell_type_2"]
    colours = bvpl.bx_colors("main", n=3)
    faces = [to_hex(c) for c in pts.get_facecolors()]
    assert faces == [colours[levels.index(g)] for g in grp.to_list()]
    (legend,) = _legends(ax)
    assert legend.get_title().get_text() == "grp_enum"
    assert [t.get_text() for t in legend.get_texts()] == levels
    assert sorted(t.get_text() for t in ax.texts) == sorted(levels)
    assert ax.get_xlabel() == "pca 1" and ax.get_ylabel() == "pca 2"


def test_embedding_plot_continuous_from_frame(mpl, sc):
    frame = bvpl.extract_embedding_data(sc, "pca", obs_cols=["lib_size"])
    ax = bvpl.embedding_plot(frame, "pca", "lib_size")
    (pts,) = _points(ax)
    np.testing.assert_allclose(pts.get_array(), frame["lib_size"].to_numpy())
    assert len(ax.figure.axes) == 2  # the colour bar


def test_feature_plot_free_and_shared(mpl, sc):
    fig = bvpl.feature_plot(
        sc, GENES[:2], "pca", feature_labels={"gene_03": "A", "gene_01": "B"}
    )
    panels = [a for a in fig.axes if a.get_title()]
    assert [a.get_title() for a in panels] == ["A", "B"]
    assert len(fig.axes) == 4  # a colour bar each
    for a in panels:
        (pts,) = _points(a)
        assert len(pts.get_offsets()) == sc.n_cells
        assert np.all(np.diff(pts.get_array()) >= 0)  # low to high

    fig = bvpl.feature_plot(
        sc, GENES[:2], "pca", scale_mode="shared", highlight_features=True
    )
    panels = [a for a in fig.axes if a.get_title()]
    assert len(fig.axes) == 3
    norms = {(_points(a)[1].norm.vmin, _points(a)[1].norm.vmax) for a in panels}
    assert len(norms) == 1
    long = bvpl.extract_feature_plot_data(sc, GENES[:1], "pca")["expression"].to_numpy()
    bg, fg = _points(panels[0])
    assert len(bg.get_offsets()) == int((long <= np.quantile(long, 0.25)).sum())
    assert len(bg.get_offsets()) + len(fg.get_offsets()) == sc.n_cells


def test_dot_plot(mpl, sc):
    ax = bvpl.dot_plot(
        sc,
        GENES,
        "cell_grp",
        feature_labels={g: g.upper() for g in GENES},
        cluster_groups=False,
    )
    (pts,) = _points(ax)
    assert len(pts.get_offsets()) == 9
    top_down = _top_down(ax)
    assert top_down == [g.upper() for g in GENES]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "cell_type_1",
        "cell_type_2",
        "cell_type_3",
    ]
    dot = bvpl.extract_dot_plot_data(sc, GENES, "cell_grp")
    order = np.argsort(dot["pct_exp"].to_numpy())
    assert np.all(np.diff(pts.get_sizes()[order]) >= 0)

    ax = bvpl.dot_plot(
        dot, feature_grouping={"gene_03": "x", "gene_01": "y", "gene_07": "x"}
    )
    top_down = _top_down(ax)
    assert top_down == ["gene_03", "gene_07", "gene_01"]
    assert {t.get_text() for t in ax.texts} == {"x", "y"}
    assert sorted(t.get_text() for t in ax.get_xticklabels()) == sorted(
        dot["group"].cast(pl.String).unique().to_list()
    )


def test_hclust_order_groups_similar_rows():
    from bixverse._core.plots import _hclust_order

    mat = np.array([[0.0], [10.0], [0.1], [10.2], [5.0]])
    order = _hclust_order(mat)
    assert sorted(order) == list(range(5))
    pos = {v: i for i, v in enumerate(order)}
    assert abs(pos[0] - pos[2]) == 1 and abs(pos[1] - pos[3]) == 1


def test_stacked_violin_plot(mpl, sc):
    from matplotlib.collections import FillBetweenPolyCollection, PolyCollection

    fig = bvpl.stacked_violin_plot(
        sc,
        GENES,
        "cell_grp",
        feature_labels=dict.fromkeys(GENES, "g") | {"gene_01": "one"},
    )
    assert len(fig.axes) == 3
    assert [a.get_ylabel() for a in fig.axes] == ["g", "one", "g"]
    for a in fig.axes:
        polys = [
            c
            for c in a.collections
            if isinstance(c, PolyCollection | FillBetweenPolyCollection)
        ]
        assert len(polys) + len(a.lines) == 3  # one violin (or flat line) per group
    assert [t.get_text() for t in fig.axes[-1].get_xticklabels()] == [
        "cell_type_1",
        "cell_type_2",
        "cell_type_3",
    ]


def test_heatmap_plot(mpl, sc):
    ax = bvpl.heatmap_plot(
        sc,
        GENES,
        "grp_enum",
        feature_grouping={"gene_03": "a", "gene_01": "b", "gene_07": "a"},
        max_cells_per_group=20,
    )
    assert len(ax.images) == 3 * 2
    assert sum(np.asarray(im.get_array()).shape[1] for im in ax.images) == 2 * 60
    labels = [t.get_text() for t in ax.texts]
    assert labels[:3] == ["cell_type_3", "cell_type_1", "cell_type_2"]
    assert labels[-3:] == ["gene_03", "gene_07", "gene_01"]
    cbar = ax.figure.axes[-1]
    assert cbar.get_ylabel() == "Scaled\nexpression"
    lo, hi = np.asarray(ax.images[0].get_clim(), dtype=float)
    assert lo == -hi and hi <= 2.5


def test_feature_scatter_plot(mpl, sc):
    pair = bvpl.extract_feature_pair(sc, GENES[0], GENES[1])
    n = int(((pair["feature_1"] > 0) | (pair["feature_2"] > 0)).sum())
    ax = bvpl.feature_scatter_plot(sc, GENES[0], GENES[1])
    (pts,) = _points(ax)
    assert len(pts.get_offsets()) == n
    assert np.all(np.isfinite(pts.get_array())) and pts.get_array().min() > 0
    assert ax.get_xlabel() == GENES[0]
    ax = bvpl.feature_scatter_plot(pair, geom="hex", remove_zeros=False)
    assert np.asarray(ax.collections[0].get_array()).sum() == pair.height


def _qc(sc):
    obs = sc.obs(cols=["lib_size", "nnz", "cell_grp"], filtered=True)
    lib = obs["lib_size"].to_numpy().astype(float)
    lib[:3] = lib.max() * 20
    return qc.run_cell_qc(
        {"lib_size": lib, "nnz": obs["nnz"].to_numpy().astype(float)},
        sc.cells_to_keep,
        groups=obs["cell_grp"].to_list(),
    )


def test_qc_violin_plots(mpl, sc):
    from matplotlib.colors import to_hex

    res = _qc(sc)
    fig = bvpl.violin_plot(res)
    assert [a.get_title(loc="left") for a in fig.axes] == ["lib_size", "nnz"]
    (pts,) = _points(fig.axes[0])
    orange = sum(to_hex(c) == to_hex("orange") for c in pts.get_facecolors())
    assert orange == int(res.final("lib_size").sum()) > 0
    with pytest.raises(ValueError, match="one panel per metric"):
        bvpl.violin_plot(res, ax=fig.axes[0])

    frame = bv.qc.get_data(res)
    fig = bvpl.violin_plot(frame, "grp", "lib_size", show_outlier=False)
    (ax,) = fig.axes
    assert ax.get_ylabel() == "lib_size (log10)"
    assert len(_points(ax)[0].get_offsets()) == frame.height


def test_qc_density_and_joint_plots(mpl, sc):
    res = _qc(sc)
    frame = bv.qc.get_data(res).with_columns(
        pl.when(pl.col("grp") == "cell_type_1")
        .then(pl.col("nnz") * 50)
        .otherwise(pl.col("nnz"))
        .alias("nnz_shift")
    )
    fig = bvpl.density_plot(frame, "grp", "nnz_shift", log_scale=False)
    (ax,) = fig.axes
    assert len(ax.lines) == 3
    assert [t.get_text() for t in ax.texts] == ["cell_type_1"]
    with pytest.raises(ValueError, match="grouped"):
        bvpl.density_plot(
            qc.run_cell_qc({"m": np.arange(20.0)}, np.arange(20, dtype=np.uint32))
        )
    fig = bvpl.joint_plot(frame, log_scale=True)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_xlabel() == "Genes per cell (log10)"


def test_paga_plot(mpl, sc, paga):
    from matplotlib.collections import LineCollection

    data = bvpl.extract_paga_plot_data(sc, paga, embedding="pca")
    ax = bvpl.paga_plot(sc, paga, embedding="pca")
    cells, nodes = _points(ax)
    assert len(cells.get_offsets()) == sc.n_cells
    assert len(nodes.get_offsets()) == data.nodes.height
    sizes = nodes.get_sizes()
    assert sizes.argmax() == data.nodes["n_cells"].arg_max()
    (lines,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lines.get_segments()) == data.edges.height
    assert sorted(t.get_text() for t in ax.texts) == sorted(
        data.nodes["cluster"].cast(pl.String).to_list()
    )

    ax = bvpl.paga_plot(
        sc,
        paga,
        embedding="pca",
        node_colour_by="lib_size",
        show_cells=False,
        label=False,
    )
    (nodes,) = _points(ax)
    np.testing.assert_allclose(
        nodes.get_array(),
        bvpl.extract_paga_plot_data(sc, paga, embedding="pca", node_stat_col="lib_size")
        .nodes["stat"]
        .to_numpy(),
    )


def test_milo_nhood_plot(mpl, sc, milo):
    from matplotlib.colors import to_hex

    data = bvpl.extract_milo_plot_data(sc, milo, embedding="pca", alpha=1.0)
    ax = bvpl.milo_nhood_plot(sc, milo, embedding="pca", alpha=1.0)
    cells, blank, sig = _points(ax)
    assert len(cells.get_offsets()) == sc.n_cells
    n_sig = int(data.nodes["is_sig"].sum())
    assert len(sig.get_offsets()) == n_sig
    assert len(blank.get_offsets()) == data.nodes.height - n_sig
    assert all(to_hex(c) == "#ffffff" for c in blank.get_facecolors())
    lfc = np.abs(sig.get_array())
    assert np.all(np.diff(lfc) >= 0)  # strongest drawn last
    assert sig.norm.vmin == -sig.norm.vmax

    annotated = de.add_nhoods_info(
        milo, sc.obs(cols=["cell_grp"], filtered=True)["cell_grp"]
    )
    ax = bvpl.milo_nhood_plot(
        sc, annotated, embedding="pca", colour_by="majority_celltype"
    )
    titles = [leg.get_title().get_text() for leg in _legends(ax)]
    assert titles[0] == "Majority cell type"
    keys = [t.get_text() for t in _legends(ax)[1].get_texts()]
    assert keys[:2] == ["Nhood size", "Shared cells"]
    with pytest.raises(ValueError, match="add_nhoods_info"):
        bvpl.milo_nhood_plot(sc, milo, embedding="pca", colour_by="majority_celltype")
