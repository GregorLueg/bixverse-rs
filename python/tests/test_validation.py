"""Argument checks at the boundary."""

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import bixverse as bv


def test_rejects_float_indices(reader):
    with pytest.raises(TypeError, match="integers"):
        bv.CellLoader(reader, np.array([1.0, 2.0]))


def test_rejects_boolean_indices(reader):
    with pytest.raises(TypeError, match="integers"):
        bv.CellLoader(reader, np.array([True, False]))


def test_rejects_negative_indices(reader):
    with pytest.raises(ValueError, match="non-negative"):
        bv.CellLoader(reader, np.array([-1, 2]))


def test_rejects_out_of_range_indices(reader):
    with pytest.raises(ValueError, match="out of range"):
        bv.CellLoader(reader, np.array([0, 10_000]))


def test_rejects_two_dimensional_indices(reader):
    with pytest.raises(ValueError, match="1-D"):
        bv.CellLoader(reader, np.zeros((2, 2), dtype=np.uint32))


def test_rejects_empty_indices(reader):
    with pytest.raises(ValueError, match="non-empty"):
        bv.CellLoader(reader, np.array([], dtype=np.uint32))


def test_rejects_bad_batch_size(reader):
    with pytest.raises(ValueError, match="batch_size"):
        bv.CellLoader(reader, batch_size=0)


def test_rejects_bad_prefetch(reader):
    with pytest.raises(ValueError, match="prefetch"):
        bv.CellLoader(reader, prefetch=0)


def test_rejects_unknown_layer(reader):
    # `layer` is a Literal, so beartype rejects it before Rust sees it.
    with pytest.raises(BeartypeCallHintParamViolation):
        bv.CellLoader(reader, layer="nonsense")  # ty: ignore[invalid-argument-type]


def test_rust_also_rejects_unknown_layer(reader):
    idx = np.arange(8, dtype=np.uint32)
    with pytest.raises(ValueError, match="unknown layer"):
        bv._bixverse.CellLoader(reader, idx, 4, False, None, "nonsense", True, 1, False)


def test_defaults_to_every_cell(reader):
    loader = bv.CellLoader(reader)
    assert len(loader.indices) == reader.total_cells
