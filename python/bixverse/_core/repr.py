"""The one repr format of the result classes."""

from __future__ import annotations

MAX_LINE: int = 88


def compact_repr(name: str, /, **fields: object) -> str:
    """`Name(a=1, b=2)` on one line if it fits, else one field per line.

    Args:
        name: The class name.
        **fields: Summary values, counts and shapes rather than arrays.

    Returns:
        The repr.
    """
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    line = f"{name}({', '.join(parts)})"
    if len(line) <= MAX_LINE:
        return line
    body = "".join(f"  {part},\n" for part in parts)
    return f"{name}(\n{body})"
