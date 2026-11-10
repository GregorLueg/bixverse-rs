"""Parameter bundles for the single cell suite.

One frozen dataclass per parameter bundle, flat as Rust sees them:
`ScrubletParams(no_pcs=10, k=0)` rather than nested blocks. Field names,
defaults and ranges are generated from the parameter specs. `ScenicParams` is
written by hand, see `bixverse.params._scenic`.
"""

from __future__ import annotations

import dataclasses
import sys

from beartype import beartype

from bixverse.params import _generated
from bixverse.params._generated import *  # noqa: F403
from bixverse.params._scenic import ScenicParams

__all__ = [*_generated.__all__, "ScenicParams"]


@beartype
def to_rust(p: object) -> dict[str, object]:
    """The flat dict Rust deserialises a bundle from.

    A field holding another bundle (the learner block of `ScenicParams`) is
    spliced in at its position.

    Args:
        p: A param bundle from this module or `bixverse_gpu.params`.

    Returns:
        Field name to value, with keyword-clashing names (`lambda_`) mapped
        back to the Rust key.

    Raises:
        TypeError: If `p` is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(p) or isinstance(p, type):
        raise TypeError(f"Expected a param bundle, got {type(p).__name__}.")
    # each generated module carries the renames of its own bundles
    names = getattr(sys.modules[type(p).__module__], "RUST_NAMES", {})
    renames = names.get(type(p).__name__, {})
    out: dict[str, object] = {}
    for f in dataclasses.fields(p):
        v = getattr(p, f.name)
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            out.update(to_rust(v))
        else:
            out[renames.get(f.name, f.name)] = v
    return out
