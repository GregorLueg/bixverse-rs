"""Single-cell streaming for Python, in Rust.

Opens a bixverse binary sparse store and yields minibatches fast enough to keep
a GPU training loop fed. The decode never leaves Rust: a feeder thread
decompresses and materialises batches ahead of the consumer, releasing the GIL
while it works.

    >>> import bixverse as bv
    >>> reader = bv.CellReader("counts.bin")
    >>> loader = bv.CellLoader(reader, batch_size=256, layer="norm")
    >>> for batch in loader:
    ...     x = batch.scipy_csr()

Batches are sparse CSR by default. Pass ``sparse=False`` for a dense
``(batch_size, n_genes)`` array, and expect it to cost a zero-fill and a
scatter over every gene whether or not it holds a count.
"""

from . import _bixverse, de, doublets, graph, integrate, io, pp, qc
from ._bixverse import (
    BixverseError,
    CellReader,
    SparseFileError,
    __core_version__,
    __version__,
)
from ._core.db import TableNotFoundError
from ._core.single_cells import Counts, SingleCells, get_counts
from ._core.state import CsrMatrix, KnnGraph, MagicLayer, ScCache, ScMap, Stamp
from .loader import Batch, CellLoader

__all__ = [
    "Batch",
    "BixverseError",
    "CellLoader",
    "CellReader",
    "Counts",
    "CsrMatrix",
    "KnnGraph",
    "MagicLayer",
    "ScCache",
    "ScMap",
    "SingleCells",
    "SparseFileError",
    "Stamp",
    "TableNotFoundError",
    "__core_version__",
    "__version__",
    "_bixverse",
    "de",
    "doublets",
    "get_counts",
    "graph",
    "integrate",
    "io",
    "pp",
    "qc",
]
