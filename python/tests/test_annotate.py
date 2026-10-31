"""ScType and Symphony without R: records, persistence, gene matching and
input checks.

Parity with the R functions is in `test_annotate_r.py`.
"""

import dataclasses

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse import _bixverse, annotate, graph, pp
from bixverse.params import (
    HarmonyParams,
    MinQualityParams,
    SctypeCellsParams,
    SyntheticDataParams,
    to_rust,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
N_PCS = 8
# the planted markers are genes 0 to 29; the HVG pick on this data keeps
# few of them, so the reference uses these directly
REF_GENES = np.arange(30)


def _load(data, directory, var=None):
    return bv.io.load_csr(
        directory, data.counts, data.obs, data.var if var is None else var, qc=QC
    )


@pytest.fixture(scope="module")
def data():
    """A synthetic experiment with planted markers and two batches."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=500, n_genes=80, n_batches=2), seed=11
    )


@pytest.fixture(scope="module")
def sc(data, tmp_path_factory):
    """The experiment with HVGs, a PCA and the neighbour graph."""
    sc = _load(data, tmp_path_factory.mktemp("annotate") / "sc")
    pp.find_hvg(sc, hvg_no=40)
    pp.calculate_pca(sc, no_pcs=10)
    graph.find_neighbours(sc)
    return sc


@pytest.fixture(scope="module")
def bare(data, tmp_path_factory):
    """The experiment without a neighbour graph."""
    return _load(data, tmp_path_factory.mktemp("annotate") / "bare")


@pytest.fixture(scope="module")
def markers(sc):
    return annotate.prepare_cell_markers(
        sc,
        pl.DataFrame(
            {
                "cell_type": [f"cell_type_{i}" for i in (1, 2, 3) for _ in range(10)],
                "gene_id": sc.gene_ids(np.arange(30)).tolist(),
            }
        ),
    )


@pytest.fixture(scope="module")
def scores(sc, markers):
    return annotate.calc_sc_type_scores(sc, markers)


@pytest.fixture(scope="module")
def reference(sc):
    return annotate.build_symphony_ref(
        sc,
        "batch_index",
        REF_GENES,
        harmony_params=HarmonyParams(k=10),
        no_pcs=N_PCS,
        label_columns=["cell_grp"],
    )


def _same_reference(a: annotate.SymphonyReference, b: annotate.SymphonyReference):
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert isinstance(y, np.ndarray) and x.dtype == y.dtype, f.name
            np.testing.assert_array_equal(x, y, err_msg=f.name)
        elif isinstance(x, pl.DataFrame):
            assert isinstance(y, pl.DataFrame) and x.equals(y), f.name
        else:
            assert x == y, f.name


##########
# ScType #
##########


def test_prepare_cell_markers_drops_unknown_genes(sc):
    genes = sc.gene_ids(np.arange(3)).tolist()
    got = annotate.prepare_cell_markers(
        sc,
        pl.DataFrame(
            {
                "cell_type": ["b", "a", "b", "c"],
                "gene_id": [genes[0], genes[1], genes[2], "absent"],
            }
        ),
    )
    assert list(got) == ["b", "a"]
    assert got["b"].positive.tolist() == [0, 2]
    with pytest.raises(ValueError, match="gene_id"):
        annotate.prepare_cell_markers(sc, pl.DataFrame({"cell_type": ["a"]}))


def test_sc_type_finds_planted_types(sc, scores):
    truth = sc.obs(cols=["cell_idx", "cell_grp"], filtered=True)["cell_grp"]
    called = np.asarray(scores.cell_types)[scores.scores.argmax(axis=1)]
    assert np.mean(called == truth.to_numpy()) > 0.9
    assert scores.n_cells == sc.n_cells and scores.n_cell_types == 3


def test_sc_type_rejects_bad_markers(sc):
    with pytest.raises(ValueError, match="no cell types"):
        annotate.calc_sc_type_scores(sc, {})
    bad = {"a": annotate.CellMarkers(positive=np.asarray([sc.n_genes]))}
    with pytest.raises(ValueError, match="out of range"):
        annotate.calc_sc_type_scores(sc, bad)


def test_score_clusters_checks_length(scores):
    with pytest.raises(ValueError, match="entries"):
        annotate.score_clusters(scores, np.zeros(3, dtype=np.int64))


def test_assign_without_graph_warns_and_takes_raw_argmax(bare, markers, scores):
    with pytest.warns(UserWarning, match="without smoothing"):
        res = annotate.assign_sc_type(bare, scores)
    assert res.agreement is None and res.composition is None
    best = scores.scores.max(axis=1)
    want = np.where(
        best >= SctypeCellsParams().score_floor,
        np.asarray(scores.cell_types, dtype=object)[scores.scores.argmax(axis=1)],
        None,
    )
    assert res.assignments.to_list() == want.tolist()
    np.testing.assert_allclose(res.scores, best, rtol=1e-6)


def test_assign_rejects_mismatched_graph(sc, scores):
    short = annotate.ScTypeResults(
        cell_types=scores.cell_types, scores=scores.scores[:-1]
    )
    with pytest.raises(ValueError, match="neighbour graph"):
        annotate.assign_sc_type(sc, short)


def test_assign_hybrid_composition(sc, scores):
    graph.find_clusters(sc, name="leiden")
    res = annotate.assign_sc_type(sc, scores, cluster_col="leiden")
    assert res.composition is not None and res.counts is not None
    assert res.hybrid is not None and res.hybrid.len() == sc.n_cells
    assert res.counts.columns == ["cluster_id", *scores.cell_types]
    sizes = (
        res.counts.drop("cluster_id").sum_horizontal() + res.composition["n_unknown"]
    )
    assert sizes.to_list() == res.composition["n_cells"].to_list()
    with pytest.raises(ValueError, match="integer cluster"):
        annotate.assign_sc_type(sc, scores, cluster_col="cell_grp")


def test_invalid_calibration_rejected_in_rust(scores):
    params = to_rust(SctypeCellsParams()) | {"calibration": "quantile"}
    with pytest.raises(ValueError, match="calibration"):
        _bixverse.sc_type_assign_cells(scores.cell_types, scores.scores, params)


############
# Symphony #
############


def test_reference_shapes(sc, reference):
    k, n_hvg = 10, REF_GENES.size
    assert reference.loadings.shape == (n_hvg, N_PCS)
    assert reference.z_corr.shape == (sc.n_cells, N_PCS)
    assert reference.centroids.shape == reference.c_cache.shape == (k, N_PCS)
    assert reference.r is not None and reference.r.shape == (k, sc.n_cells)
    np.testing.assert_allclose(reference.nr, reference.r.sum(axis=1), rtol=1e-4)
    assert "Labels: cell_grp" in repr(reference)


def test_save_load_round_trip(reference, tmp_path):
    path = tmp_path / "ref.npz"
    annotate.save_symphony_ref(reference, path)
    loaded = annotate.load_symphony_ref(path)
    _same_reference(reference, loaded)
    assert repr(loaded) == repr(reference)


def test_save_load_slim_without_labels(sc, tmp_path):
    slim = annotate.build_symphony_ref(
        sc,
        "batch_index",
        sc.map.hvg,
        harmony_params=HarmonyParams(k=5),
        no_pcs=4,
        slim=True,
    )
    assert slim.z_orig is None and slim.r is None and slim.labels is None
    path = tmp_path / "slim.npz"
    annotate.save_symphony_ref(slim, path)
    _same_reference(slim, annotate.load_symphony_ref(path))


def test_load_rejects_foreign_file(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, x=np.zeros(3))
    with pytest.raises(ValueError, match="not a saved Symphony reference"):
        annotate.load_symphony_ref(path)


def test_build_rejects_bad_hvg(sc):
    with pytest.raises(ValueError, match="out-of-range"):
        annotate.build_symphony_ref(sc, "batch_index", [0, sc.n_genes])


def test_gene_mismatch_maps_as_zero_column(data, sc, reference, tmp_path):
    """A query lacking HVGs maps as if their loading rows were zero."""
    full = _load(data, tmp_path / "full")
    absent = set(reference.hvg_gene_names[:3])
    renamed = data.var.with_columns(
        pl.when(pl.col("gene_id").is_in(absent))
        .then(pl.lit("renamed_") + pl.col("gene_id"))
        .otherwise(pl.col("gene_id"))
        .alias("gene_id")
    )
    partial = _load(data, tmp_path / "partial", var=renamed)

    annotate.map_symphony_query(reference, partial, batch_column="batch_index")
    zeroed = reference.loadings.copy()
    zeroed[:3] = 0.0
    annotate.map_symphony_query(
        dataclasses.replace(reference, loadings=zeroed),
        full,
        batch_column="batch_index",
    )
    for name in ("symphony_pca", "symphony", "symphony_r"):
        np.testing.assert_allclose(
            partial.cache.embeddings[name],
            full.cache.embeddings[name],
            rtol=1e-5,
            atol=1e-6,
            err_msg=name,
        )
    assert partial.cache.embeddings["symphony_r"].shape == (partial.n_cells, 10)


def test_map_without_batch_skips_correction(data, reference, tmp_path):
    query = _load(data, tmp_path / "q")
    annotate.map_symphony_query(reference, query)
    np.testing.assert_array_equal(
        query.cache.embeddings["symphony"], query.cache.embeddings["symphony_pca"]
    )


def test_add_symphony_labels(sc, reference):
    with pytest.raises(ValueError, match="already present"):
        annotate.add_symphony_labels(reference, sc, ["cell_grp"])
    new = annotate.add_symphony_labels(
        reference, sc, ["cell_grp", "batch_index"], overwrite=True
    )
    assert new.labels is not None and new.labels.columns == ["cell_grp", "batch_index"]
    assert reference.labels is not None and reference.labels.columns == ["cell_grp"]
    with pytest.raises(ValueError, match="not found"):
        annotate.add_symphony_labels(reference, sc, ["nope"])


def test_transfer_labels(data, sc, reference, tmp_path):
    query = _load(data, tmp_path / "q")
    with pytest.raises(ValueError, match="map_symphony_query"):
        annotate.transfer_labels_symphony(reference, query, "cell_grp")
    with pytest.raises(ValueError, match="no stored labels"):
        annotate.transfer_labels_symphony(
            dataclasses.replace(reference, labels=None), query, "cell_grp"
        )
    annotate.map_symphony_query(reference, query, batch_column="batch_index")
    # the reference's own cells project onto its pre-Harmony scores
    assert reference.z_orig is not None
    np.testing.assert_allclose(
        query.cache.embeddings["symphony_pca"], reference.z_orig, atol=1e-4
    )
    got = annotate.transfer_labels_symphony(reference, query, "cell_grp")
    assert got.columns == ["predicted_cell_grp", "confidence_cell_grp"]
    truth = query.obs(cols=["cell_idx", "cell_grp"], filtered=True)["cell_grp"]
    assert (got["predicted_cell_grp"] == truth).mean() > 0.8
    obs = query.obs(cols=["cell_idx", "predicted_cell_grp"], filtered=True)
    assert obs["predicted_cell_grp"].to_list() == got["predicted_cell_grp"].to_list()


def test_transfer_missing_label_is_own_class(data, sc, reference, tmp_path):
    """R's `factor(exclude = NULL)`: a missing reference label can win."""
    labels = pl.DataFrame({"lab": pl.Series([None] * sc.n_cells, dtype=pl.String)})
    query = _load(data, tmp_path / "q")
    annotate.map_symphony_query(reference, query)
    got = annotate.transfer_labels_symphony(
        dataclasses.replace(reference, labels=labels), query, "lab"
    )
    assert got["predicted_lab"].null_count() == query.n_cells
    np.testing.assert_array_equal(got["confidence_lab"].to_numpy(), 1.0)
