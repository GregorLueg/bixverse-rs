"""DGE, markers, pseudobulk, NEBULA, Milo and MELD without R: designs,
result records, getters and input checks.

Parity with the R functions is in `test_de_r.py`.
"""

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import _bixverse, de, graph, pp
from bixverse._core.de import _model_matrix
from bixverse.params import MinQualityParams, NebulaParams, SyntheticDataParams

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
TRT = ("sample_4", "sample_5", "sample_6")


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment with six samples and a condition."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=600, n_genes=80, n_samples=6, sample_bias="even"),
        seed=5,
    )
    obs = data.obs.with_columns(
        condition=pl.when(pl.col("sample_id").is_in(TRT))
        .then(pl.lit("trt"))
        .otherwise(pl.lit("ctr"))
    )
    return data, obs


@pytest.fixture(scope="module")
def sc(data, tmp_path_factory):
    """The experiment with a PCA and the kNN on it. Nothing here writes to it."""
    data, obs = data
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("de") / "sc", data.counts, obs, data.var, qc=QC
    )
    pp.find_hvg(sc, hvg_no=50)
    pp.calculate_pca(sc, no_pcs=10)
    graph.find_neighbours(sc)
    return sc


@pytest.fixture(scope="module")
def bare(data, tmp_path_factory):
    """The experiment without a kNN."""
    data, obs = data
    return bv.io.load_csr(
        tmp_path_factory.mktemp("de") / "bare", data.counts, obs, data.var, qc=QC
    )


##########
# Design #
##########


def test_model_matrix_codes_like_r():
    frame = pl.DataFrame(
        {
            "grp": ["b", "a", "c", "a"],
            "age": [1, 2, 3, 4],
            "flag": [True, False, True, True],
        }
    )
    mm, names = _model_matrix(frame, "~ grp + age + flag")
    assert names == ["(Intercept)", "grpb", "grpc", "age", "flagTRUE"]
    np.testing.assert_array_equal(
        mm,
        [[1, 1, 0, 1, 1], [1, 0, 0, 2, 0], [1, 0, 1, 3, 1], [1, 0, 0, 4, 1]],
    )


@pytest.mark.parametrize("formula", ["grp", "y ~ grp", "~ grp:age", "~ log(age)", "~"])
def test_model_matrix_rejects_unsupported(formula):
    frame = pl.DataFrame({"grp": ["a", "b"], "age": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"formula|main effects"):
        _model_matrix(frame, formula)


###########
# Markers #
###########


def test_find_markers_ranks_planted_markers(sc):
    obs = sc.obs(cols=["cell_id", "cell_grp"], filtered=True)
    ids = {
        g: obs.filter(pl.col("cell_grp") == g)["cell_id"].to_list()
        for g in ("cell_type_1", "cell_type_2")
    }
    res = de.find_markers(
        sc, ids["cell_type_1"], ids["cell_type_2"], alternative="greater"
    )
    assert res.columns == [
        "gene_id",
        "lfc",
        "prop1",
        "prop2",
        "z_scores",
        "p_values",
        "fdr",
    ]
    top = res.sort("z_scores", descending=True)["gene_id"].head(10).to_list()
    assert set(top) == set(sc.gene_ids()[:10])


def test_find_all_markers_one_block_per_group(sc):
    res = de.find_all_markers(sc, "cell_grp")
    assert res.columns[:2] == ["grp", "gene_id"]
    groups = sc.obs(cols=["cell_grp"], filtered=True)["cell_grp"].unique(
        maintain_order=True
    )
    assert res["grp"].unique(maintain_order=True).to_list() == groups.to_list()


def test_find_specific_markers_record(sc):
    res = de.find_specific_markers(sc, "cell_grp", reference_group="cell_type_3")
    assert res is not None
    assert de.get_marker_summary(res)["ref_grp"].unique().to_list() == ["cell_type_3"]
    comp = de.get_marker_comparisons(res)
    assert set(comp["rival_grp"].unique()) == {"cell_type_1", "cell_type_2"}
    assert comp.height == 2 * de.get_marker_summary(res).height
    assert repr(res).startswith("ScSpecificMarkers(") and "n_references=1" in repr(res)


def test_find_specific_markers_checks(sc):
    with pytest.raises(ValueError, match="not one of"):
        de.find_specific_markers(sc, "cell_grp", reference_group="nope")
    with pytest.raises(ValueError, match="Not found"):
        de.find_specific_markers(sc, "nope")


##############
# Pseudobulk #
##############


def _cell_list(sc: bv.SingleCells) -> dict[str, list[str]]:
    obs = sc.obs(cols=["cell_id", "sample_id"], filtered=True)
    return {
        s: obs.filter(pl.col("sample_id") == s)["cell_id"].to_list()
        for s in sorted(obs["sample_id"].unique())
    }


def test_pseudobulk_dense_and_sparse_agree(sc):
    cell_list = _cell_list(sc)
    dense = de.get_pseudobulked(sc, cell_list)
    sparse = de.get_pseudobulked(sc, cell_list, return_format="sparse")
    assert isinstance(dense.counts, np.ndarray) and isinstance(
        sparse.counts, bv.CsrMatrix
    )
    assert dense.counts.shape == (6, sc.n_genes)
    np.testing.assert_array_equal(sparse.counts.scipy_csr().toarray(), dense.counts)
    counts = bv.get_counts(sc, cells=sc.cell_indices(cell_list["sample_1"]))
    np.testing.assert_array_equal(
        dense.counts[0], np.asarray(counts.scipy_csr().sum(axis=0)).ravel()
    )


def test_pseudobulk_dge_table(sc):
    cell_list = _cell_list(sc)
    design = pl.DataFrame(
        {"(Intercept)": [1.0] * 6, "trt": [float(s in TRT) for s in cell_list]}
    )
    res = de.pseudobulk_dge(sc, cell_list, design, coef="trt")
    assert res.columns == [
        "feature_id",
        "log_fc",
        "log_cpm",
        "f_stat",
        "p_value",
        "fdr",
    ]
    assert res.height <= sc.n_genes
    assert res["fdr"].is_between(0, 1).all()


def test_run_edger_ql_checks():
    counts = np.full((5, 4), 10.0)
    design = np.column_stack([np.ones(4), [0, 0, 1, 1]])
    ids = [f"g{i}" for i in range(5)]
    with pytest.raises(ValueError, match="two columns"):
        de.run_edger_ql(counts, design[:, :1], ids)
    with pytest.raises(ValueError, match="rows against"):
        de.run_edger_ql(counts, design[:3], ids)
    with pytest.raises(ValueError, match="negative"):
        de.run_edger_ql(-counts, design, ids)
    with pytest.raises(ValueError, match="not both"):
        de.run_edger_ql(counts, design, ids, coef=1, contrast=[0, 1])
    with pytest.raises(ValueError, match="not a design column"):
        de.run_edger_ql(counts, design, ids, coef="trt")
    with pytest.raises(ValueError, match="rows against 2 design"):
        de.run_edger_ql(counts, design, ids, contrast=[0, 1, 0])


def test_pseudobulk_dge_checks_design_rows(sc):
    with pytest.raises(ValueError, match="pseudo-bulk"):
        de.pseudobulk_dge(sc, _cell_list(sc), np.ones((3, 2)))


##########
# NEBULA #
##########


def test_nebula_record(sc):
    res = de.nebula(sc, "sample_id", "~ condition", genes_to_use=list(range(10)))
    assert res.params["tested"] == "conditiontrt"
    assert res.coefficients.columns == ["gene_id", "(Intercept)", "conditiontrt"]
    assert res.results["gene_id"].to_list() == res.coefficients["gene_id"].to_list()
    assert "cell_overdispersion_shrunk" in res.results.columns
    assert de.get_params(res)["n_subjects"] == 6
    assert "method='ln'" in repr(res) and "subject_col='sample_id'" in repr(res)
    plain = de.nebula(
        sc,
        "sample_id",
        "~ condition",
        coef=0,
        genes_to_use=list(range(10)),
        nebula_params=NebulaParams(shrink_dispersion=False),
    )
    assert plain.params["tested"] == "(Intercept)"
    assert "cell_overdispersion_shrunk" not in plain.results.columns


def test_nebula_checks(sc):
    with pytest.raises(ValueError, match="not both"):
        de.nebula(sc, "sample_id", "~ condition", coef=1, contrast=[0, 1])
    with pytest.raises(ValueError, match="Not found"):
        de.nebula(sc, "sample_id", "~ nope")
    with pytest.raises(ValueError, match="offset"):
        de.nebula(sc, "sample_id", "~ condition", offset=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite weights"):
        de.nebula(sc, "sample_id", "~ condition", contrast=[1.0])
    with pytest.raises(ValueError, match="could be matched"):
        de.nebula(sc, "sample_id", "~ condition", cells_to_use=["nope"])


########
# Milo #
########


@pytest.fixture(scope="module")
def milo(sc):
    milo = de.get_milor_abundances(
        sc,
        "sample_id",
        milor_params=de.MiloRParams(k_refine=10, index_type="exhaustive"),
    )
    assert milo is not None
    return milo


def test_milo_record(sc, milo):
    n_nhoods = milo.sample_counts.shape[0]
    assert milo.nhoods.shape == (sc.n_cells, n_nhoods)
    assert milo.samples == tuple(f"sample_{i}" for i in range(1, 7))
    # each neighbourhood counts every member once, in its sample
    per_nhood = np.bincount(milo.nhoods.indices, minlength=n_nhoods)
    np.testing.assert_array_equal(milo.sample_counts.sum(axis=1), per_nhood)
    index_cells = de.get_index_cells(milo)
    assert len(index_cells) == n_nhoods and (np.diff(index_cells) > 0).all()
    assert de.get_params(milo)["used_emb"] == "pca"


def test_milo_test_and_annotate(sc, milo):
    with pytest.warns(UserWarning, match="test_nhoods"):
        assert de.add_nhoods_info(milo, ["x"] * sc.n_cells) is milo
    with pytest.warns(UserWarning, match="No differential"):
        assert de.get_differential_abundance_res(milo) is None
    design_df = pl.DataFrame(
        {
            "sample_id": list(reversed(milo.samples)),
            "condition": ["trt"] * 3 + ["ctr"] * 3,
        }
    )
    with pytest.warns(UserWarning, match="Reordering"):
        tested = de.test_nhoods(milo, "~ condition", design_df)
    info = de.get_differential_abundance_res(tested)
    assert info is not None
    assert info.columns == [
        "Nhood",
        "logFC",
        "logCPM",
        "F",
        "PValue",
        "FDR",
        "SpatialFDR",
    ]
    assert info["Nhood"].to_list() == list(range(milo.sample_counts.shape[0]))
    labels = sc.obs(cols=["cell_grp"], filtered=True)["cell_grp"]
    annotated = de.add_nhoods_info(tested, labels).nhoods_info
    assert annotated is not None
    assert annotated["majority_prop"].is_between(0, 1).all()
    with pytest.raises(ValueError, match="Length of cell_info"):
        de.add_nhoods_info(tested, ["x"])
    with pytest.raises(ValueError, match="don't match"):
        de.test_nhoods(milo, "~ condition", design_df.head(3))


def test_milo_and_meld_need_a_knn(bare):
    bare.cache.pca_factors = np.zeros((bare.n_cells, 5), dtype=np.float32)
    with pytest.warns(UserWarning, match="No kNN"):
        assert de.get_milor_abundances(bare, "sample_id") is None
    with pytest.warns(UserWarning, match="No kNN"):
        assert de.meld(bare, "sample_id") is None


def test_milor_params_checks():
    with pytest.raises(ValueError, match="prop"):
        de.MiloRParams(prop=1.0)
    with pytest.raises(ValueError, match="index_type"):
        _bixverse.milor_nhoods(
            np.zeros((3, 2), dtype=np.float32),
            np.zeros((3, 1), dtype=np.uint32),
            np.zeros(3, dtype=np.uint32),
            1,
            {"index_type": "ivf"},
        )


########
# MELD #
########


def test_meld_rows_are_likelihoods(sc):
    res = de.meld(sc, "condition")
    assert res is not None
    assert res.norm_scores.columns == ["cell_id", "ctr", "trt"]
    sums = res.norm_scores.drop("cell_id").sum_horizontal().to_numpy()
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)
    assert res.raw_scores.height == sc.n_cells
