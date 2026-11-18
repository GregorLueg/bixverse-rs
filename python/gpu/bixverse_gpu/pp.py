"""GPU sparse PCA.

Writes what `bixverse.pp.calculate_pca` writes: scores, loadings and the first
`no_pcs` singular values to `sc.cache`, stamped `"pca"`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
from beartype import beartype

from bixverse._core.pp import _set_pca
from bixverse._core.single_cells import SingleCells
from bixverse._validate import Verbose
from bixverse.params import PcaParams, to_rust
from bixverse_gpu._backend import Backend, pick

__all__ = [
    "calculate_pca",
]

logger = logging.getLogger(__name__)


@beartype
def calculate_pca(
    sc: SingleCells,
    no_pcs: int,
    *,
    pca_params: PcaParams | None = None,
    hvg: np.ndarray | Sequence[int] | Sequence[str] | None = None,
    seed: int = 42,
    verbose: Verbose = True,
    backend: Backend = "wgpu",
) -> None:
    """Sparse PCA over the kept cells and the HVGs on the GPU.

    Always sparse, so the data are never scaled into a dense matrix.
    `svd_solver = "randomised"` runs the randomised SVD, every other solver
    the exact block Lanczos; there is no GPU covariance path. No residual
    path either: use `bixverse.pp.calculate_pca(..., residuals=True)`.

    Args:
        sc: The experiment.
        no_pcs: Number of components.
        pca_params: The PCA parameters. `None` uses the defaults.
        hvg: 0-based gene indices or gene ids to use instead of `sc.map.hvg`;
            they replace it.
        seed: Seed for the sketch or the Lanczos starting block.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.
        backend: `"wgpu"` (any GPU), or `"mlx"` (Apple Silicon, needs a
            source build with the `mlx` feature; see `mlx_available()`).

    Raises:
        BixverseError: Without a GPU adapter, or with `backend="mlx"` on a
            build without the MLX twins.
    """
    params = pca_params or PcaParams()
    if sc.map.hvg is None and hvg is None:
        warnings.warn(
            "No HVGs identified in the object nor provided. Please run find_hvg() "
            "or provide the indices of the HVG. Returning object as is.",
            stacklevel=2,
        )
        return
    if hvg is not None:
        if verbose:
            logger.info("HVGs provided. Setting the internal HVG to them.")
        sc.map.hvg = sc._resolve_genes(hvg)
    selected = sc.map.hvg
    assert selected is not None
    if verbose:
        logger.info(
            "Using GPU-accelerated sparse SVD (%s, %s) with %i HVG.",
            "randomised" if params.svd_solver == "randomised" else "block Lanczos",
            backend,
            selected.size,
        )
    scores, loadings, sv = pick("pca", backend)(
        str(sc.genes_path),
        str(sc.cells_path),
        sc.cells_to_keep,
        selected,
        no_pcs,
        to_rust(params),
        True,
        seed,
        verbose,
    )
    _set_pca(sc, scores, loadings, sv[:no_pcs], selected)
