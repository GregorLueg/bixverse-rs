"""Parameter bundles for the single cell suite.

One frozen dataclass per bixverse R `params_*()` constructor, flattened the way
R flattens them before Rust sees the list: `ScrubletParams(no_pcs=10, k=0)`
rather than nested blocks. Field names, defaults and ranges are generated from
the R specs, so both front ends hand Rust the same values.
"""

from __future__ import annotations

import dataclasses

from beartype import beartype

from bixverse.params._generated import *  # noqa: F403
from bixverse.params._generated import RUST_NAMES, __all__  # noqa: F401


@beartype
def to_rust(p: object) -> dict[str, object]:
    """The flat dict Rust deserialises a bundle from.

    Args:
        p: A param bundle from this module.

    Returns:
        Field name to value, with keyword-clashing names (`lambda_`) mapped
        back to the R / Rust key.

    Raises:
        TypeError: If `p` is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(p) or isinstance(p, type):
        raise TypeError(f"Expected a param bundle, got {type(p).__name__}.")
    renames = RUST_NAMES.get(type(p).__name__, {})
    return {
        renames.get(f.name, f.name): getattr(p, f.name) for f in dataclasses.fields(p)
    }
