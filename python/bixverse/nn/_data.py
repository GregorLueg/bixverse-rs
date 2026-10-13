"""Backend-agnostic minibatches for the models: dense raw counts plus batch codes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from beartype import beartype

from .. import _bixverse
from .._validate import check_indices
from ..loader import CellLoader


@beartype
def encode_batches(
    batch: Any, *, n_cells: int, categories: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Turn per-cell batch labels into integer codes.

    Args:
        batch: Labels aligned with the cells, any 1-D array-like. `None` puts
            every cell in one batch.
        n_cells: Expected length.
        categories: Known categories, e.g. from a fitted model. `None` takes
            the sorted unique labels.

    Returns:
        `(codes, categories)`, codes as `int64` indexing into `categories`.

    Raises:
        ValueError: On a length mismatch or a label outside `categories`.
    """
    if batch is None:
        labels = np.zeros(n_cells, dtype=np.int64)
    else:
        labels = np.asarray(batch)
        if labels.ndim != 1 or labels.shape[0] != n_cells:
            raise ValueError(
                f"batch must be 1-D with one label per cell ({n_cells}), "
                f"got shape {labels.shape}"
            )
    if categories is None:
        categories = np.unique(labels)
    codes = np.searchsorted(categories, labels)
    codes = np.clip(codes, 0, len(categories) - 1)
    if not np.array_equal(categories[codes], labels):
        raise ValueError("batch holds labels the model was not trained on")
    return codes.astype(np.int64), categories


@beartype
def resolve_cells(reader: _bixverse.CellReader, cells: Any) -> np.ndarray:
    """Validate cell indices, defaulting to every cell in the store.

    Args:
        reader: The store.
        cells: Store cell indices, or `None` for all of them.

    Returns:
        A `uint32` index array.
    """
    if cells is None:
        return np.arange(reader.total_cells, dtype=np.uint32)
    return check_indices(cells, n_cells=reader.total_cells, name="cells")


@beartype
def resolve_genes(reader: _bixverse.CellReader, genes: Any) -> np.ndarray:
    """Validate gene indices, defaulting to every gene in the store.

    Args:
        reader: The store.
        genes: Store gene indices, or `None` for all of them.

    Returns:
        A `uint32` index array.
    """
    if genes is None:
        return np.arange(reader.total_genes, dtype=np.uint32)
    return check_indices(genes, n_cells=reader.total_genes, name="genes")


@beartype
def iterate_counts(
    reader: _bixverse.CellReader,
    cells: np.ndarray,
    genes: np.ndarray,
    codes: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One pass over the cells as dense raw-count minibatches.

    The Rust feeder subsets the genes and densifies, so each batch arrives as
    a `(n, n_genes)` float32 array ready for a zero-copy tensor wrap.

    Args:
        reader: The store.
        cells: Store cell indices.
        genes: Store gene indices, in column order.
        codes: Batch code per entry of `cells`.
        batch_size: Cells per minibatch.
        shuffle: Shuffle the cell order.
        seed: Shuffle seed.

    Yields:
        `(x, batch_codes, cell_ids)`: counts, the `int64` batch code of each
        row and its store cell index.

    Raises:
        ValueError: If a cell has no counts over `genes`, which would make the
            observed library size `log(0)`.
    """
    lookup = np.full(reader.total_cells, -1, dtype=np.int64)
    lookup[cells] = codes
    loader = CellLoader(
        reader,
        cells,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        layer="raw",
        sparse=False,
        genes=genes,
    )
    for batch in loader:
        x = batch.raw
        if x is None:  # pragma: no cover - layer="raw" always fills it
            raise ValueError("loader returned no raw layer")
        empty = np.flatnonzero(x.sum(axis=1) == 0)
        if empty.size:
            raise ValueError(
                f"cell {int(batch.cell_ids[empty[0]])} has no counts over the "
                "selected genes; filter it out before training"
            )
        yield x, lookup[batch.cell_ids], batch.cell_ids
