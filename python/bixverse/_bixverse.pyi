"""Type stubs for the compiled extension module."""

from typing import Any

import numpy as np

__version__: str
__core_version__: str

class BixverseError(Exception): ...
class SparseFileError(BixverseError): ...

class CellReader:
    def __init__(self, path: str) -> None: ...
    @property
    def total_cells(self) -> int: ...
    @property
    def total_genes(self) -> int: ...
    @property
    def is_cell_based(self) -> bool: ...
    @property
    def target_size(self) -> float | None: ...

class CellLoaderIter:
    def __iter__(self) -> CellLoaderIter: ...
    def __next__(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        tuple[int, int],
    ]: ...

class CellLoader:
    def __init__(
        self,
        reader: CellReader,
        indices: Any,
        batch_size: int = ...,
        shuffle: bool = ...,
        seed: int | None = ...,
        layer: str = ...,
        sparse: bool = ...,
        prefetch: int = ...,
        drop_last: bool = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> CellLoaderIter: ...

def _write_synthetic_store(
    path: str,
    n_cells: int,
    n_genes: int,
    density: float = ...,
    seed: int = ...,
) -> np.ndarray: ...
