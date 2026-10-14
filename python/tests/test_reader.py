"""The reader handle."""

import pytest

import bixverse as bv

from .conftest import N_CELLS, N_GENES


def test_reports_shape(reader):
    assert reader.total_cells == N_CELLS
    assert reader.total_genes == N_GENES


def test_is_cell_major(reader):
    assert reader.is_cell_based is True


def test_target_size_round_trips(reader):
    assert reader.target_size == pytest.approx(1e4)


def test_repr_names_the_shape(reader):
    assert f"total_cells={N_CELLS}" in repr(reader)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        bv.CellReader(str(tmp_path / "absent.bin"))


def test_foreign_file_raises_sparse_file_error(tmp_path):
    path = tmp_path / "foreign.bin"
    path.write_bytes(b"not a bixverse store" * 8)
    with pytest.raises(bv.SparseFileError):
        bv.CellReader(str(path))


def test_sparse_file_error_is_a_bixverse_error():
    assert issubclass(bv.SparseFileError, bv.BixverseError)


def test_versions_are_exposed():
    assert bv.__version__
    assert bv.__core_version__
