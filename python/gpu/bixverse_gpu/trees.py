"""GPU Bonsai.

Returns the `bixverse.trees.BonsaiTree` the CPU `bonsai` returns. Only Sanity
moves to the GPU, gene chunk by gene chunk; the tree search and the layout
stay on the CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from beartype import beartype

from bixverse._core.metacells import MetaCells
from bixverse._core.single_cells import SingleCells
from bixverse._core.trees import BonsaiTree, _bonsai
from bixverse._validate import Verbose
from bixverse.params import BonsaiParams
from bixverse_gpu import _bixverse_gpu

__all__ = [
    "bonsai",
]


@beartype
def bonsai(
    sc: SingleCells | MetaCells,
    *,
    hvg: np.ndarray | Sequence[int] | None = None,
    bonsai_params: BonsaiParams | None = None,
    verbose: Verbose = True,
) -> BonsaiTree:
    """A Bonsai tree over the cells or meta cells with Sanity on the GPU.

    Args:
        sc: The experiment or meta cells.
        hvg: 0-based candidate genes. `None` takes every gene.
        bonsai_params: The parameters. `None` uses the defaults.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The `BonsaiTree`, leaves in the kept cell or meta cell order.

    Raises:
        ValueError: On a gene index out of range, or if no gene survives the
            signal-to-noise filter.
        BixverseError: Without a GPU adapter that runs plane operations.

    References:
        de Groot, et al., Nat. Biotechnol., 2026; Breda, et al., Nat.
        Biotechnol., 2021.
    """
    return _bonsai(
        sc, hvg=hvg, bonsai_params=bonsai_params, verbose=verbose, ext=_bixverse_gpu
    )
