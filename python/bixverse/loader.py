"""Minibatch iteration over a cell-major store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from beartype import beartype

from . import _bixverse
from ._validate import check_indices

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scipy.sparse import csr_matrix

Layer = Literal["raw", "norm", "both"]


class Batch(NamedTuple):
    """One minibatch of cells.

    Sparse batches carry `indices` and `indptr` and CSR-ordered `raw`/`norm`.
    Dense batches carry `(n_cells, n_genes)` buffers and leave both `None`.
    Whichever layer was not requested is `None`.
    """

    raw: np.ndarray | None
    norm: np.ndarray | None
    indices: np.ndarray | None
    indptr: np.ndarray | None
    shape: tuple[int, int]

    @property
    def is_sparse(self) -> bool:
        """Whether this batch holds CSR components rather than a dense buffer."""
        return self.indptr is not None

    def values(self) -> np.ndarray:
        """The single requested layer.

        Returns:
            `raw` or `norm`, whichever was asked for.

        Raises:
            ValueError: If the batch carries both layers, which is ambiguous.
        """
        if self.raw is not None and self.norm is not None:
            raise ValueError("batch holds both layers; use .raw or .norm")
        found = self.raw if self.raw is not None else self.norm
        if found is None:  # pragma: no cover - the loader always fills one
            raise ValueError("batch holds no data layer")
        return found

    def scipy_csr(self, layer: Literal["raw", "norm"] | None = None) -> csr_matrix:
        """Rebuild this batch as a `scipy.sparse.csr_matrix`.

        Args:
            layer: Which layer to use when the batch carries both.

        Returns:
            A CSR matrix of shape `shape`, sharing the batch's buffers.

        Raises:
            ValueError: If the batch is dense, or `layer` is needed and absent.
        """
        from scipy.sparse import csr_matrix

        if not self.is_sparse:
            raise ValueError("batch is dense; construct the array directly")
        data = self.values() if layer is None else getattr(self, layer)
        if data is None:
            raise ValueError(f"batch does not carry the '{layer}' layer")
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)


class CellLoader:
    """Iterate a cell-major store in minibatches, decoding ahead of the consumer.

    Each pass spawns one feeder thread that decompresses and materialises
    batches into a bounded queue, so decode overlaps the training step. The
    queue depth is the backpressure knob, not a parallelism knob: decode is
    already rayon-parallel inside a single batch.

        >>> import bixverse as bv
        >>> reader = bv.CellReader("counts.bin")
        >>> loader = bv.CellLoader(reader, batch_size=256)
        >>> for batch in loader:
        ...     x = batch.scipy_csr()

    Sparse is the default. A dense batch costs a zero-fill and a scatter over
    `batch_size * n_genes` floats no matter how few non-zeros it holds, which
    at droplet sparsity is roughly twenty times the memory traffic.
    """

    @beartype
    def __init__(
        self,
        reader: _bixverse.CellReader,
        indices: Any = None,
        *,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: int | None = None,
        layer: Layer = "norm",
        sparse: bool = True,
        prefetch: int = 4,
        drop_last: bool = False,
    ) -> None:
        """Build a loader.

        Args:
            reader: An open cell-major `CellReader`.
            indices: Cells to iterate. `None` uses every cell in the store.
            batch_size: Cells per batch.
            shuffle: Reshuffle at the start of every epoch.
            seed: Base seed for the shuffle. `None` draws one at random.
            layer: `"raw"`, `"norm"` or `"both"`.
            sparse: CSR components when true, a dense buffer when false.
            prefetch: Batches held in flight by the feeder thread.
            drop_last: Drop a trailing partial batch.

        Raises:
            ValueError: For a gene-major store, an unusable parameter, or an
                out-of-range cell index.
        """
        if not reader.is_cell_based:
            raise ValueError(
                "CellLoader needs a cell-major store; this one is gene-major"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if prefetch <= 0:
            raise ValueError(f"prefetch must be positive, got {prefetch}")

        if indices is None:
            resolved = np.arange(reader.total_cells, dtype=np.uint32)
        else:
            resolved = check_indices(indices, n_cells=reader.total_cells)

        self._reader = reader
        self._indices = resolved
        self._inner = _bixverse.CellLoader(
            reader,
            resolved,
            batch_size,
            shuffle,
            seed,
            layer,
            sparse,
            prefetch,
            drop_last,
        )

    @property
    def reader(self) -> _bixverse.CellReader:
        """The store this loader reads from."""
        return self._reader

    @property
    def indices(self) -> np.ndarray:
        """The cells this loader iterates, in the order they were given."""
        return self._indices

    def __len__(self) -> int:
        """Number of batches in one epoch."""
        return len(self._inner)

    def __iter__(self):
        """Start an epoch.

        Yields:
            One `Batch` per step. Each call reshuffles and spawns its own
            feeder, so two concurrent passes do not share state.
        """
        for raw, norm, indices, indptr, shape in self._inner:
            yield Batch(raw, norm, indices, indptr, shape)

    def __repr__(self) -> str:
        return f"CellLoader(n_cells={len(self._indices)}, n_batches={len(self)})"
