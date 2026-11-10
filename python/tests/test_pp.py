"""Processing without R: state handling and the edge cases of each step.

Parity with the R helpers is in `test_pp_r.py`.
"""

import math
import warnings

import numpy as np
import pytest

import bixverse as bv
from bixverse import _bixverse, pp
from bixverse.params import (
    AprParams,
    HvgParams,
    MagicParams,
    MinQualityParams,
    PcaParams,
    SctransformParams,
    SyntheticDataParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
PCA_RESIDUAL = PcaParams(normalise_variance=False)


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment with batches and samples."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(
            n_cells=200,
            n_genes=60,
            n_batches=2,
            n_samples=2,
            sample_bias="even",
        ),
        seed=3,
    )


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment per test."""
    return bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)


def _knn(sc: bv.SingleCells, k: int = 5) -> bv.KnnGraph:
    """A ring graph over the kept cells, for MAGIC."""
    n = sc.n_cells
    idx = (np.arange(n)[:, None] + np.arange(1, k + 1)[None, :]) % n
    dist = np.tile(np.arange(1, k + 1, dtype=np.float32), (n, 1))
    return bv.KnnGraph(
        indices=idx.astype(np.uint32), distances=dist, dist_metric="euclidean"
    )


#######
# HVG #
#######


def test_find_hvg_writes_var_and_map(sc):
    pp.find_hvg(sc, hvg_no=20)
    var = sc.var()
    for col in ("mean", "var", "var_exp", "var_std"):
        assert col in var.columns
    order = np.argsort(-var["var_std"].to_numpy(), kind="stable")[:20]
    assert sc.map.hvg is not None
    assert sc.map.hvg.tolist() == order.tolist()
    assert sc.map.hvg.dtype == np.uint32


def test_find_hvg_rejects_hvg_no_past_the_genes(sc):
    with pytest.raises(ValueError, match="hvg_no"):
        pp.find_hvg(sc, hvg_no=sc.n_genes + 1)
    assert sc.map.hvg is None
    assert "var_std" not in sc.var().columns


@pytest.mark.parametrize("streaming", [False, True])
def test_get_hvg_data_leaves_state_alone(sc, streaming):
    frame = pp.get_hvg_data(sc, hvg_no=10, streaming=streaming)
    assert frame["is_hvg"].sum() == 10
    assert sorted(frame["hvg_rank"].drop_nulls().to_list()) == list(range(1, 11))
    assert frame["gene_idx"].to_list() == list(range(sc.n_genes))
    assert sc.map.hvg is None
    assert "var_std" not in sc.var().columns


def test_streaming_does_not_change_hvg(sc):
    a = pp.get_hvg_data(sc, hvg_no=10, streaming=False)
    b = pp.get_hvg_data(sc, hvg_no=10, streaming=True)
    assert a.equals(b)


def test_residual_hvg_is_refused_where_it_cannot_apply(sc):
    p = HvgParams(method="residual")
    with pytest.raises(ValueError, match="residual"):
        pp.get_hvg_data(sc, hvg_params=p)
    with pytest.raises(ValueError, match="residual"):
        pp.find_hvg_batch_aware(sc, "batch_index", hvg_params=p)
    with pytest.raises(ValueError, match="No fitted residual model"):
        pp.find_hvg(sc, hvg_params=p)


def test_batch_aware_needs_the_column(sc):
    with pytest.raises(ValueError, match="not found"):
        pp.find_hvg_batch_aware(sc, "nope")


def test_batch_aware_union_covers_each_batch(sc):
    res = pp.find_hvg_batch_aware(sc, "batch_index", hvg_no=5)
    assert set(res.hvg_data["batch"].unique().to_list()) == {"1", "2"}
    assert len(res.hvg_gene_idx) == len(set(res.hvg_gene_idx.tolist())) >= 5
    inter = pp.find_hvg_batch_aware(
        sc, "batch_index", hvg_no=5, gene_comb_method="intersection"
    )
    assert set(inter.hvg_gene_idx.tolist()) <= set(res.hvg_gene_idx.tolist())


def test_rust_rejects_unknown_param_values(sc):
    with pytest.raises(ValueError, match="invalid hvg params"):
        _bixverse.hvg(
            str(sc.genes_path),
            str(sc.cells_path),
            sc.cells_to_keep,
            {"method": "bogus"},
            False,
        )
    with pytest.raises(ValueError, match="invalid pca params"):
        _bixverse.pca(
            str(sc.genes_path),
            str(sc.cells_path),
            sc.cells_to_keep,
            np.arange(5, dtype=np.uint32),
            2,
            {"svd_solver": "bogus"},
            True,
            1,
        )


#######
# PCA #
#######


def test_pca_without_hvg_warns_and_returns(sc):
    with pytest.warns(UserWarning, match="No HVGs"):
        pp.calculate_pca(sc, 5)
    assert sc.cache.pca_factors is None


def test_pca_takes_hvg_ids_and_sets_them(sc):
    genes = sc.gene_ids()[[3, 1, 7, 9, 11, 13]].tolist()
    pp.calculate_pca(sc, 3, hvg=genes)
    assert sc.map.hvg is not None
    assert sc.map.hvg.tolist() == [3, 1, 7, 9, 11, 13]
    assert sc.cache.pca_factors.shape == (sc.n_cells, 3)
    assert sc.cache.pca_loadings.shape == (6, 3)
    assert sc.cache.pca_singular_vals.shape == (3,)


def test_dense_and_sparse_pca_agree(sc):
    pp.find_hvg(sc, hvg_no=30)
    p = PcaParams(svd_solver="exact")
    pp.calculate_pca(sc, 4, pca_params=p, sparse_svd=False)
    dense = sc.cache.pca_singular_vals.copy()
    pp.calculate_pca(sc, 4, pca_params=p, sparse_svd=True)
    np.testing.assert_allclose(sc.cache.pca_singular_vals, dense, rtol=1e-4)


def test_cell_filter_makes_the_pca_stale(sc):
    pp.find_hvg(sc, hvg_no=30)
    pp.calculate_pca(sc, 4)
    assert sc.check_state(["pca"]) is True
    with pytest.warns(UserWarning, match="pca"):
        sc.set_cells_to_keep(np.arange(100))
    status = sc.cache_status()
    assert status["stale"].to_list() == [True]
    pp.calculate_pca(sc, 4)
    assert sc.check_state(["pca"]) is True
    assert sc.cache.pca_factors.shape == (100, 4)


#############
# Residuals #
#############


def test_residual_fit_is_stamped_and_bound_to_its_cells(sc):
    pp.fit_residuals(sc, method="analytic_pearson")
    fit = sc.cache.residual_fit
    assert isinstance(fit, pp.ResidualFit)
    assert fit.method == "analytic_pearson"
    assert fit.cell_indices.tolist() == sc.cells_to_keep.tolist()
    assert sc.check_state(["residuals"]) is True

    pp.find_hvg(sc, hvg_no=10, hvg_params=HvgParams(method="residual"))
    assert sc.map.hvg is not None and len(sc.map.hvg) == 10

    with pytest.warns(UserWarning, match="residuals"):
        sc.set_cells_to_keep(np.arange(150))
    with pytest.raises(ValueError, match="filter moved"):
        pp.find_hvg(sc, hvg_no=10, hvg_params=HvgParams(method="residual"))
    with pytest.raises(ValueError, match="filter moved"):
        pp.calculate_pca(sc, 3, pca_params=PCA_RESIDUAL, residuals=True)


def test_residual_pca_refuses_what_it_cannot_honour(sc):
    pp.fit_residuals(sc, method="analytic_pearson")
    pp.find_hvg(sc, hvg_no=10, hvg_params=HvgParams(method="residual"))
    with pytest.raises(ValueError, match="normalise the variance"):
        pp.calculate_pca(sc, 3, residuals=True)
    with pytest.raises(ValueError, match="PFlogPF"):
        pp.calculate_pca(
            sc,
            3,
            pca_params=PcaParams(clr=True, normalise_variance=False),
            residuals=True,
        )
    with pytest.raises(ValueError, match="no sparse solver"):
        pp.calculate_pca(
            sc, 3, pca_params=PCA_RESIDUAL, sparse_svd=True, residuals=True
        )

    pp.calculate_pca(sc, 3, pca_params=PCA_RESIDUAL, residuals=True)
    stamps = sc.cache.stamps
    assert stamps["pca"].parents == (stamps["residuals"].id,)


def test_residual_pca_drops_uncovered_genes(sc):
    pp.fit_residuals(sc, method="analytic_pearson")
    fit = sc.cache.residual_fit
    assert isinstance(fit, pp.ResidualFit)
    covered = fit.genes[:5]
    pp.calculate_pca(
        sc, 2, pca_params=PCA_RESIDUAL, hvg=covered, residuals=True, verbose=True
    )
    assert sc.cache.pca_loadings.shape == (5, 2)


def test_residual_variance_is_null_outside_the_model(sc):
    p = AprParams(min_cells=150)
    pp.fit_residuals(sc, method="analytic_pearson", residual_params=p)
    fit = sc.cache.residual_fit
    assert isinstance(fit, pp.ResidualFit)
    pp.find_hvg(sc, hvg_no=3, hvg_params=HvgParams(method="residual"))
    rv = sc.var(cols=["residual_variance"])["residual_variance"]
    assert rv.null_count() == sc.n_genes - len(fit.genes)


def test_fit_residuals_input_checks(sc):
    with pytest.raises(TypeError, match="SctransformParams"):
        pp.fit_residuals(sc, residual_params=AprParams())
    with pytest.raises(ValueError, match="only supported"):
        pp.fit_residuals(sc, method="analytic_pearson", covariate_columns=["nnz"])
    with pytest.raises(ValueError, match="not numeric"):
        pp.fit_residuals(sc, covariate_columns=["cell_grp"])
    with pytest.raises(ValueError, match="not found"):
        pp.fit_residuals(sc, covariate_columns=["nope"])
    with pytest.raises(ValueError, match="not found"):
        pp.fit_residuals(sc, group_column="nope")
    assert sc.cache.residual_fit is None


def test_poisson_limit_and_one_sided_clip_deserialise(sc):
    pp.fit_residuals(
        sc, method="analytic_pearson", residual_params=AprParams(theta=math.inf)
    )
    assert sc.cache.residual_fit is not None
    pp.fit_residuals(sc, residual_params=SctransformParams(clip_min=-3.0, clip_max=3.0))
    assert sc.cache.residual_fit is not None
    # the dataclass refuses half a range; Rust ignores one, as R's reader does
    fit = _bixverse.fit_residuals(
        str(sc.genes_path),
        str(sc.cells_path),
        "sctransform",
        sc.cells_to_keep,
        None,
        [],
        {"clip_max": 3.0},
    )
    assert fit.n_groups == 1
    with pytest.raises(ValueError, match="clipping range"):
        _bixverse.fit_residuals(
            str(sc.genes_path),
            str(sc.cells_path),
            "sctransform",
            sc.cells_to_keep,
            None,
            [],
            {"clip_min": 3.0, "clip_max": -3.0},
        )


def test_grouped_fit_has_one_model_per_group(sc):
    pp.fit_residuals(sc, method="analytic_pearson", group_column="sample_id")
    fit = sc.cache.residual_fit
    assert isinstance(fit, pp.ResidualFit)
    assert fit.n_groups == 2
    assert fit.group_column == "sample_id"


def test_corrected_counts(sc, tmp_path):
    pp.fit_residuals(sc, method="analytic_pearson")
    with pytest.raises(ValueError, match="scTransform fit"):
        pp.sct_corrected_counts(sc, dir_out=tmp_path / "x")

    pp.fit_residuals(sc)
    out = pp.sct_corrected_counts(sc, dir_out=tmp_path / "corr", build_cell_store=False)
    assert isinstance(out, pp.CorrectedCounts)
    assert out.f_path.is_file()
    assert out.n_cells == sc.n_cells
    with pytest.raises(FileExistsError):
        pp.sct_corrected_counts(sc, dir_out=tmp_path / "corr")

    new = pp.sct_corrected_counts(sc, dir_out=tmp_path / "corr", overwrite=True)
    assert isinstance(new, bv.SingleCells)
    assert new.shape == (sc.n_cells, out.n_genes)
    assert new.gene_ids().tolist() == sc.gene_ids(out.genes).tolist()
    assert new.cell_ids().tolist() == sc.cell_ids(filtered=True).tolist()
    counts = bv.get_counts(new).scipy_csr()
    assert counts.data.size and np.all(counts.data == np.round(counts.data))


#########
# MAGIC #
#########


def test_magic_without_knn_warns(sc):
    with pytest.warns(UserWarning, match="No kNN"):
        pp.run_magic(sc, ["gene_01"])
    assert sc.cache.magic is None


def test_magic_zero_steps_is_the_input(sc):
    sc.cache.knn = _knn(sc)
    genes = sc.gene_ids()[[0, 5, 9]].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pp.run_magic(sc, genes, magic_params=MagicParams(n_steps=0, clip_threshold=0.0))
    want = bv.get_counts(sc, genes=genes, layer="norm").scipy_csr().toarray()
    assert sc.cache.magic is not None
    np.testing.assert_allclose(sc.cache.magic.data, want, rtol=1e-6)
    assert sc.cache.magic.features.tolist() == genes


def test_magic_refuses_a_stale_or_misaligned_knn(sc):
    sc.cache.knn = _knn(sc)
    sc.stamp("knn")
    with pytest.warns(UserWarning, match="knn"):
        sc.set_cells_to_keep(np.arange(50))
    with pytest.raises(ValueError, match="stale"):
        pp.run_magic(sc, ["gene_01"])

    sc.cache.knn = bv.KnnGraph(
        indices=np.zeros((10, 3), dtype=np.uint32),
        distances=np.ones((10, 3), dtype=np.float32),
        dist_metric="euclidean",
    )
    with pytest.raises(ValueError, match="covers 10 cells"):
        pp.run_magic(sc, ["gene_01"])


######################
# Review regressions #
######################


@pytest.mark.parametrize("sparse_svd", [True, False])
def test_pca_asking_for_more_pcs_than_cells_errors(sc, sparse_svd):
    # every solver clamps to the rank; the output used to be sized to no_pcs
    # anyway and panicked on the slice
    pp.find_hvg(sc, hvg_no=40)
    sc.set_cells_to_keep(np.arange(25))
    with pytest.raises(bv.BixverseError, match="only supports"):
        pp.calculate_pca(sc, 30, sparse_svd=sparse_svd)


def test_residual_pca_survives_save_and_load(sc):
    # the residual model is not saved; its children must not load as stale
    pp.fit_residuals(sc, method="analytic_pearson")
    pp.find_hvg(sc, hvg_no=30)
    pp.calculate_pca(sc, 5, residuals=True, pca_params=PCA_RESIDUAL)
    bv.graph.find_neighbours(sc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bv.io.save(sc)
    back = bv.io.load_existing(sc.dir)
    assert back.check_state(["pca", "knn"]) is True
    bv.graph.find_neighbours(back)


def test_pca_remembers_its_genes(sc):
    pp.find_hvg(sc, hvg_no=30)
    hvg = sc.map.hvg.copy()
    pp.calculate_pca(sc, 5)
    np.testing.assert_array_equal(sc.cache.pca_genes, hvg)
    pp.fit_residuals(sc, method="analytic_pearson")
    pp.calculate_pca(sc, 5, residuals=True, pca_params=PCA_RESIDUAL)
    # the residual path sorts the genes and the loadings rows follow
    assert np.all(np.diff(sc.cache.pca_genes.astype(np.int64)) > 0)
    assert set(sc.cache.pca_genes) <= set(hvg)
