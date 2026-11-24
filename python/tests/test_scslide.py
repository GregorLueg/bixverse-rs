"""scSLIDE: shapes, count bookkeeping, cross-fitting, phenotype guards and the
custom-embedding entry point on synthetic multi-sample data.
"""

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import _bixverse, pp, traj
from bixverse.params import MinQualityParams, SyntheticDataParams, to_rust

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
N_SAMPLES = 12
PARAMS = traj.ScSlideParams(
    n_cells_per_donor=60,
    n_landmarks=120,
    n_comp=4,
    cor_min_cells=20,
    knn_range=40,
    leverage_n_sketch=300,
)


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    """Twelve samples, half of them cases, with a numeric covariate."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(
            n_cells=1800, n_genes=120, n_samples=N_SAMPLES, sample_bias="even"
        ),
        seed=5,
    )
    sample_no = data.obs["sample_id"].str.extract(r"(\d+)").cast(pl.Int64)
    obs = data.obs.with_columns(
        pl.when(sample_no % 2 == 0)
        .then(pl.lit("case"))
        .otherwise(pl.lit("control"))
        .alias("condition"),
        (sample_no * 1.5).alias("age"),
    )
    sc = bv.io.load_csr(
        tmp_path_factory.mktemp("scslide") / "sc", data.counts, obs, data.var, qc=QC
    )
    pp.find_hvg(sc, hvg_no=60, verbose=False)
    pp.calculate_pca(sc, no_pcs=8, verbose=False)
    return sc


@pytest.fixture(scope="module")
def res(sc):
    return traj.run_scslide(
        sc,
        "sample_id",
        "condition",
        cell_type_col="cell_grp",
        scslide_params=PARAMS,
        verbose=False,
    )


def test_scslide_shapes_and_bookkeeping(sc, res):
    assert res.counts.shape == (PARAMS.n_landmarks, N_SAMPLES)
    assert res.cell_landmarks.shape == (sc.n_cells, PARAMS.k_count)
    assert res.pls_embedding.shape == (sc.n_cells, PARAMS.n_comp)
    # every non-landmark cell counts k_count times
    assert res.counts.sum() == (sc.n_cells - PARAMS.n_landmarks) * PARAMS.k_count
    assert len(set(res.landmark_ids)) == PARAMS.n_landmarks
    assert set(res.landmark_ids) <= set(res.training_cell_ids)
    assert res.landmark_labels is not None
    assert len(res.landmark_labels) == PARAMS.n_landmarks
    assert res.sample_meta.columns == ["sample", "condition"]
    assert tuple(res.sample_meta["sample"]) == res.samples


def test_scslide_cross_fitting(sc):
    params = traj.ScSlideParams(**{**to_rust(PARAMS), "n_folds": 3})
    res = traj.run_scslide(
        sc, "sample_id", "condition", scslide_params=params, verbose=False
    )
    assert len(res.pls_genes) == 3
    assert sorted(set(res.sample_folds.tolist())) == [0, 1, 2]
    assert res.counts.sum() == (sc.n_cells - PARAMS.n_landmarks) * PARAMS.k_count


def test_scslide_numeric_phenotype(sc):
    res = traj.run_scslide(sc, "sample_id", "age", scslide_params=PARAMS, verbose=False)
    assert res.sample_meta["age"].dtype == pl.Float64


def test_scslide_refuses_phenotype_varying_within_sample(sc):
    with pytest.raises(ValueError, match="constant within a sample"):
        traj.run_scslide(
            sc, "sample_id", "cell_grp", scslide_params=PARAMS, verbose=False
        )


def test_landmark_counts_reproduces_run(sc, res):
    """The custom-embedding entry point on the run's own embeddings gives the
    run's counts back."""
    pca = np.ascontiguousarray(sc.cache.pca_factors, dtype=np.float32)
    pos = {c: i for i, c in enumerate(res.cell_ids)}
    landmarks = np.array([pos[c] for c in res.landmark_ids], dtype=np.uint32)
    donors = np.array(
        [res.samples.index(s) for s in sc.obs(filtered=True)["sample_id"].to_list()],
        dtype=np.uint32,
    )
    out = _bixverse.scslide_landmark_counts(
        pca,
        np.ascontiguousarray(res.pls_embedding),
        landmarks,
        donors,
        N_SAMPLES,
        to_rust(PARAMS),
        42,
    )
    np.testing.assert_array_equal(out["counts"], res.counts)


def test_sample_diffusion_map(res):
    dm = traj.sample_diffusion_map(res, dm_params=traj.SampleDmParams(n_dcs=3, k=4))
    assert dm.dcs.height == N_SAMPLES
    assert dm.dcs.columns == ["sample", "condition", "DC_1", "DC_2", "DC_3"]
    assert dm.residuals.shape == res.counts.shape
    assert np.isfinite(dm.dcs.select("DC_1", "DC_2", "DC_3").to_numpy()).all()


def test_sample_diffusion_map_lowers_k(res, caplog):
    dm = traj.sample_diffusion_map(res, dm_params=traj.SampleDmParams(n_dcs=2, k=50))
    assert dm.params["k"] == N_SAMPLES - 2
