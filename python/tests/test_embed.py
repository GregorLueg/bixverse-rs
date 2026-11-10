"""2D embeddings without R: the missing extra, inputs, stamps, staleness and
persistence.

Parity with the R helpers is in `test_embed_r.py`.
"""

import importlib
import sys
import warnings

import numpy as np
import pytest

import bixverse as bv
from bixverse import graph, pp
from bixverse.params import MinQualityParams, SyntheticDataParams

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)


def test_missing_extra_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "manifolds_rs", None)
    monkeypatch.delitem(sys.modules, "bixverse.embed", raising=False)
    monkeypatch.delitem(sys.modules, "bixverse._core.embed", raising=False)
    with pytest.raises(ImportError, match=r"bixverse\[embed\]"):
        importlib.import_module("bixverse.embed")


embed = pytest.importorskip("bixverse.embed", exc_type=ImportError)


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=200, n_genes=60, n_batches=1), seed=5
    )


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment with HVGs, a PCA and the neighbours."""
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    pp.find_hvg(sc, hvg_no=40)
    pp.calculate_pca(sc, 10)
    graph.find_neighbours(sc)
    return sc


def _fast_umap():
    return embed.umap_defaults().set_params(n_epochs=50)


##########
# Inputs #
##########


def test_umap_writes_and_stamps(sc):
    embed.umap(sc, params=_fast_umap())
    out = sc.cache.embeddings["umap"]
    assert out.shape == (sc.n_cells, 2)
    assert np.isfinite(out).all()
    status = sc.cache_status()
    row = status.filter(status["artefact"] == "umap").row(0, named=True)
    assert row["stamped"] and not row["stale"]
    knn_id = sc.cache.stamps["knn"].id
    assert sc.cache.stamps["umap"].parents == (sc.cache.stamps["pca"].id, knn_id)


def test_without_knn_only_the_embedding_is_a_parent(sc):
    embed.umap(sc, use_knn=False, params=_fast_umap())
    assert sc.cache.stamps["umap"].parents == (sc.cache.stamps["pca"].id,)


def test_cached_knn_is_reused(sc):
    # the cached kNN against a fresh search of the same width: identical,
    # because the default search is exact
    embed.umap(sc, params=_fast_umap())
    cached = sc.cache.embeddings["umap"]
    embed.umap(sc, use_knn=False, params=_fast_umap())
    np.testing.assert_array_equal(sc.cache.embeddings["umap"], cached)


def test_planted_knn_changes_the_embedding(sc):
    # a ring graph in the cache: a run that reads it must differ from a search
    k = sc.cache.knn.indices.shape[1]
    n = sc.n_cells
    ring = (np.arange(n)[:, None] + np.arange(1, k + 1)[None, :]) % n
    sc.cache.knn = bv.KnnGraph(
        indices=ring.astype(np.uint32),
        distances=np.tile(np.arange(1, k + 1, dtype=np.float32), (n, 1)),
        dist_metric="euclidean",
    )
    sc.stamp("knn", parents=["pca"])
    embed.umap(sc, params=_fast_umap())
    planted = sc.cache.embeddings["umap"]
    embed.umap(sc, use_knn=False, params=_fast_umap())
    assert not np.allclose(sc.cache.embeddings["umap"], planted)


def test_missing_knn_warns_and_searches(sc):
    sc.cache.knn = None
    with pytest.warns(UserWarning, match="No kNN data found"):
        embed.umap(sc, params=_fast_umap())
    assert sc.cache.embeddings["umap"].shape == (sc.n_cells, 2)
    assert sc.cache.stamps["umap"].parents == (sc.cache.stamps["pca"].id,)


def test_missing_embedding_raises(sc):
    with pytest.raises(ValueError, match="not found"):
        embed.tsne(sc, embd_to_use="harmony")


def test_reserved_slot_raises(sc):
    with pytest.raises(ValueError, match="reserved"):
        embed.umap(sc, slot_name="knn", params=_fast_umap())
    assert "knn" not in sc.cache.embeddings


def test_slot_name_and_other_embedding(sc):
    embed.umap(sc, slot_name="umap_a", params=_fast_umap())
    embed.forceatlas2(
        sc,
        use_knn=False,
        embd_to_use="umap_a",
        slot_name="fa2_on_umap",
        params=embed.forceatlas2_defaults().set_params(n_epochs=50),
    )
    stamps = sc.cache.stamps
    assert stamps["fa2_on_umap"].parents == (stamps["umap_a"].id,)


def test_no_embd_to_use_truncates(sc):
    p = embed.phate_defaults()
    embed.phate(sc, use_knn=False, no_embd_to_use=3, params=p)
    three = sc.cache.embeddings["phate"]
    sc.cache.pca_factors = np.ascontiguousarray(sc.cache.pca_factors[:, :3])
    embed.phate(sc, use_knn=False, params=p)
    np.testing.assert_array_equal(sc.cache.embeddings["phate"], three)


def test_every_function_runs(sc):
    embed.densmap(sc, params=embed.densmap_defaults().set_params(n_epochs=50))
    embed.tsne(sc, params=embed.tsne_defaults().set_params(n_epochs=100))
    embed.densne(sc, params=embed.densne_defaults().set_params(n_epochs=100))
    embed.forceatlas2(sc, params=embed.forceatlas2_defaults().set_params(n_epochs=50))
    for slot in ("densmap", "tsne", "densne", "fa2"):
        assert sc.cache.embeddings[slot].shape == (sc.n_cells, 2), slot


def test_defaults_are_fresh_objects():
    assert embed.umap_defaults() is not embed.umap_defaults()


def test_umap_sgd_parallel_runs(sc):
    p = embed.umap_defaults().set_params(optimiser="sgd_parallel", n_epochs=50)
    embed.umap(sc, slot_name="umap_sgd", params=p)
    got = sc.cache.embeddings["umap_sgd"]
    assert got.shape == (sc.n_cells, 2)
    assert np.isfinite(got).all()


def test_phate_searches_afresh_by_default(sc):
    embed.phate(sc)
    assert sc.cache.embeddings["phate"].shape == (sc.n_cells, 2)
    assert sc.cache.stamps["phate"].parents == (sc.cache.stamps["pca"].id,)


#############
# Staleness #
#############


def test_filter_change_makes_the_embedding_stale(sc):
    embed.umap(sc, params=_fast_umap())
    with pytest.warns(UserWarning, match="umap"):
        sc.set_cells_to_keep(sc.cell_ids(filtered=True)[10:].tolist())
    assert "umap" in sc.cache_status().filter("stale")["artefact"].to_list()


def test_recomputed_knn_makes_the_embedding_stale(sc):
    embed.umap(sc, params=_fast_umap())
    graph.find_neighbours(sc)
    status = sc.cache_status()
    row = status.filter(status["artefact"] == "umap").row(0, named=True)
    assert row["stale"]


def test_stale_input_raises(sc):
    pp.calculate_pca(sc, 8)
    with pytest.raises(ValueError, match="knn"):
        embed.umap(sc, params=_fast_umap())


def test_replaced_embedding_drops_its_stamp(sc):
    embed.umap(sc, params=_fast_umap())
    sc.cache.embeddings["umap"] = sc.cache.embeddings["umap"].copy()
    assert sc.cache.artefacts()["umap"] is None


###############
# Persistence #
###############


def test_embeddings_survive_save_and_load(sc):
    embed.umap(sc, params=_fast_umap())
    embed.phate(sc)
    want = {k: v.copy() for k, v in sc.cache.embeddings.items()}
    stamps = {k: sc.cache.stamps[k] for k in want}
    bv.io.save(sc)

    back = bv.io.load_existing(sc.dir)
    assert set(back.cache.embeddings) == set(want)
    for k, v in want.items():
        np.testing.assert_array_equal(back.cache.embeddings[k], v)
        assert back.cache.stamps[k] == stamps[k]
    assert back.check_state(["umap", "phate"]) is True


def test_no_warnings_on_the_happy_path(sc):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        embed.umap(sc, params=_fast_umap())
        embed.phate(sc)
