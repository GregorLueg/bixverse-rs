"""Batching, shuffling and the CSR contract."""

import numpy as np
import pytest

import bixverse as bv

from .conftest import N_CELLS, N_GENES


def test_batch_count_and_shapes(reader):
    loader = bv.CellLoader(reader, batch_size=64, shuffle=False)
    assert len(loader) == 4  # 200 cells -> 64, 64, 64, 8

    batches = list(loader)
    assert len(batches) == 4
    assert [b.shape[0] for b in batches] == [64, 64, 64, 8]
    assert all(b.shape[1] == N_GENES for b in batches)


def test_drop_last_trims_the_partial_batch(reader):
    loader = bv.CellLoader(reader, batch_size=64, shuffle=False, drop_last=True)
    assert len(loader) == 3
    assert [b.shape[0] for b in loader] == [64, 64, 64]


def test_sparse_is_the_default(reader):
    batch = next(iter(bv.CellLoader(reader, batch_size=8, shuffle=False)))
    assert batch.is_sparse
    assert batch.indptr is not None
    assert batch.indices.dtype == np.uint32
    assert batch.indptr.dtype == np.uint32
    assert batch.norm.dtype == np.float32


def test_indptr_is_a_valid_csr_row_pointer(reader):
    batch = next(iter(bv.CellLoader(reader, batch_size=32, shuffle=False)))
    assert len(batch.indptr) == batch.shape[0] + 1
    assert batch.indptr[0] == 0
    assert batch.indptr[-1] == len(batch.indices)
    assert np.all(np.diff(batch.indptr) >= 0)


def test_raw_layer_matches_the_dense_truth(reader, truth):
    loader = bv.CellLoader(reader, batch_size=50, shuffle=False, layer="raw")
    rows = [b.scipy_csr().toarray() for b in loader]
    np.testing.assert_array_equal(np.vstack(rows), truth)


def test_dense_layout_matches_the_dense_truth(reader, truth):
    loader = bv.CellLoader(
        reader, batch_size=50, shuffle=False, layer="raw", sparse=False
    )
    batches = list(loader)
    assert not batches[0].is_sparse
    assert batches[0].raw.shape == (50, N_GENES)
    np.testing.assert_array_equal(np.vstack([b.raw for b in batches]), truth)


def test_both_layers_share_a_sparsity_pattern(reader):
    batch = next(
        iter(bv.CellLoader(reader, batch_size=16, shuffle=False, layer="both"))
    )
    assert batch.raw is not None and batch.norm is not None
    assert len(batch.raw) == len(batch.norm) == len(batch.indices)
    with pytest.raises(ValueError, match="both layers"):
        batch.values()


def test_norm_layer_is_log1p_of_a_scaled_library(reader, truth):
    batch = next(iter(bv.CellLoader(reader, batch_size=4, shuffle=False, layer="both")))
    csr_raw = batch.scipy_csr("raw").toarray()
    csr_norm = batch.scipy_csr("norm").toarray()
    expected = np.log1p(csr_raw / csr_raw.sum(axis=1, keepdims=True) * 1e4)
    # data_norm is stored as f16, so the tolerance is the storage, not the maths.
    np.testing.assert_allclose(csr_norm, expected, rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(csr_raw, truth[:4])


def test_unshuffled_epoch_visits_every_cell_once(reader, truth):
    loader = bv.CellLoader(reader, batch_size=33, shuffle=False, layer="raw")
    seen = np.vstack([b.scipy_csr().toarray() for b in loader])
    assert seen.shape == (N_CELLS, N_GENES)
    np.testing.assert_array_equal(seen, truth)


def test_shuffled_epoch_is_a_permutation(reader, truth):
    loader = bv.CellLoader(reader, batch_size=33, shuffle=True, seed=7, layer="raw")
    seen = np.vstack([b.scipy_csr().toarray() for b in loader])
    assert sorted(map(tuple, seen)) == sorted(map(tuple, truth))


def test_same_seed_gives_the_same_order(reader):
    def first_row(seed):
        loader = bv.CellLoader(reader, batch_size=8, seed=seed, layer="raw")
        return next(iter(loader)).scipy_csr().toarray()[0]

    np.testing.assert_array_equal(first_row(11), first_row(11))


def test_successive_epochs_reshuffle(reader):
    loader = bv.CellLoader(reader, batch_size=8, seed=11, layer="raw")
    first = next(iter(loader)).scipy_csr().toarray()
    second = next(iter(loader)).scipy_csr().toarray()
    assert not np.array_equal(first, second)


def test_two_iterators_do_not_share_state(reader):
    loader = bv.CellLoader(reader, batch_size=8, shuffle=False, layer="raw")
    a, b = iter(loader), iter(loader)
    np.testing.assert_array_equal(
        next(a).scipy_csr().toarray(), next(b).scipy_csr().toarray()
    )


def test_subset_of_cells_is_respected(reader, truth):
    wanted = np.array([3, 1, 7, 100], dtype=np.uint32)
    loader = bv.CellLoader(reader, wanted, batch_size=4, shuffle=False, layer="raw")
    batch = next(iter(loader))
    assert batch.shape == (4, N_GENES)
    np.testing.assert_array_equal(batch.scipy_csr().toarray(), truth[wanted])


def test_abandoned_iterator_shuts_the_feeder_down(reader):
    loader = bv.CellLoader(reader, batch_size=1, shuffle=False, prefetch=1)
    it = iter(loader)
    next(it)
    del it  # the feeder is parked on a full channel; dropping must unblock it


def test_dense_batch_refuses_scipy_csr(reader):
    batch = next(iter(bv.CellLoader(reader, batch_size=4, sparse=False)))
    with pytest.raises(ValueError, match="dense"):
        batch.scipy_csr()
