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

from . import _bixverse
from ._bixverse import (
    BixverseError,
    CellReader,
    SparseFileError,
    __core_version__,
    __version__,
)
from .loader import Batch, CellLoader

__all__ = [
    "Batch",
    "BixverseError",
    "CellLoader",
    "CellReader",
    "SparseFileError",
    "__core_version__",
    "__version__",
    "_bixverse",
]
