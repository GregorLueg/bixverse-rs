"""Shared fixtures. Every store is generated, never checked in."""

import functools
import shutil
import subprocess

import numpy as np
import pytest

import bixverse as bv

N_CELLS = 200
N_GENES = 50
DENSITY = 0.3
SEED = 42

R_BIXVERSE = "0.9.0"
R_SKIP_REASON = f"needs Rscript and bixverse >= {R_BIXVERSE}"


@functools.cache
def r_available() -> bool:
    """Whether the R package the parity fixtures are written for is installed."""
    if shutil.which("Rscript") is None:
        return False
    check = (
        'quit(status = !(requireNamespace("bixverse", quietly = TRUE) && '
        f'packageVersion("bixverse") >= "{R_BIXVERSE}"))'
    )
    return subprocess.run(["Rscript", "-e", check], capture_output=True).returncode == 0


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
