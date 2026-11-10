"""Value checks the generated param bundles call from `__post_init__`."""

from __future__ import annotations

from beartype import beartype


@beartype
def check_len(name: str, value: tuple[object, ...] | None) -> None:
    """Reject an empty vector where at least one element is needed.

    Args:
        name: Field name, for the message.
        value: The field value. `None` passes; nullability is the type's job.

    Raises:
        ValueError: If `value` is an empty tuple.
    """
    if value is not None and len(value) == 0:
        raise ValueError(f"`{name}` needs at least one element.")


@beartype
def check_range(
    name: str,
    value: int | float | tuple[int | float, ...] | None,
    lo: int | float | None,
    hi: int | float | None,
    lo_open: bool,
    hi_open: bool,
) -> None:
    """Check a scalar, or every element of a vector, against a range.

    Args:
        name: Field name, for the message.
        value: The field value. `None` passes; nullability is the type's job.
        lo: Lower bound, `None` for unbounded.
        hi: Upper bound, `None` for unbounded.
        lo_open: Whether the lower bound is excluded.
        hi_open: Whether the upper bound is excluded.

    Raises:
        ValueError: If any element falls outside the range.
    """
    if value is None:
        return
    for v in value if isinstance(value, tuple) else (value,):
        below = lo is not None and (v <= lo if lo_open else v < lo)
        above = hi is not None and (v >= hi if hi_open else v > hi)
        if below or above:
            left = "(" if lo_open else "["
            right = ")" if hi_open else "]"
            lo_s = "" if lo is None else lo
            hi_s = "" if hi is None else hi
            raise ValueError(f"`{name}` = {v} is outside {left}{lo_s}, {hi_s}{right}.")
