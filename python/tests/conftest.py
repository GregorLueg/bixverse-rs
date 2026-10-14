"""Shared fixtures. Every store is generated, never checked in."""

import numpy as np
import pytest

import bixverse as bv

N_CELLS = 200
N_GENES = 50
DENSITY = 0.3
SEED = 42


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """A synthetic cell-major store plus its dense raw counts."""
    path = tmp_path_factory.mktemp("bixverse") / "counts.bin"
    dense = bv._bixverse._write_synthetic_store(
        str(path), N_CELLS, N_GENES, DENSITY, SEED
    )
    return str(path), np.asarray(dense)


@pytest.fixture(scope="session")
def reader(store):
    """An open reader over the synthetic store."""
    path, _ = store
    return bv.CellReader(path)


@pytest.fixture(scope="session")
def truth(store):
    """The dense raw counts the store was written from."""
    _, dense = store
    return dense
