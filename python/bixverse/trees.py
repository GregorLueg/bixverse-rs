"""Bonsai trees, the port of the bixverse R `bonsai_sc`, `relayout_bonsai`
and `set_bonsai_embedding`.

Sanity then Bonsai over a `SingleCells` or `MetaCells`, every cell or meta
cell a leaf. See `bixverse._core.trees` for the workflow.
"""

from bixverse._core.trees import (
    BonsaiTree,
    bonsai,
    relayout_bonsai,
    set_bonsai_embedding,
)

__all__ = ["BonsaiTree", "bonsai", "relayout_bonsai", "set_bonsai_embedding"]
