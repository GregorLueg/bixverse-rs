"""Array checks at the FFI boundary."""

from typing import Any

import numpy as np
from beartype import beartype


@beartype
def check_indices(indices: Any, *, n_cells: int, name: str = "indices") -> np.ndarray:
    """Coerce cell indices into the contiguous ``uint32`` array Rust borrows.

    Integer arrays pass through after a range check. Floats are rejected rather
    than truncated, because a rounded index is a silently wrong cell.

    Args:
        indices: Array-like of cell indices.
        n_cells: Total cells in the store, for the upper bound.
        name: Argument name, used in error messages.

    Returns:
        A C-contiguous 1-D ``uint32`` array.

    Raises:
        TypeError: If the input does not hold integers.
        ValueError: If it is not 1-D, is empty, or holds an index outside
            ``[0, n_cells)``.
    """
    arr = np.asarray(indices)

    if arr.dtype.kind == "b" or arr.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers, got dtype {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")

    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0:
        raise ValueError(f"{name} must be non-negative, got {lo}")
    if hi >= n_cells:
        raise ValueError(
            f"{name} holds index {hi}, out of range for a store of {n_cells} cells"
        )

    return np.ascontiguousarray(arr, dtype=np.uint32)
