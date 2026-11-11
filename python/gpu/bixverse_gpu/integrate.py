"""GPU batch correction: Harmony v2, BBKNN and fastMNN.

Harmony and fastMNN write their corrected embedding as `"harmony_gpu"` and
`"mnn_gpu"`, stamped as derived from `"pca"`; BBKNN writes
the kNN and the connectivities as `bixverse.integrate.bbknn` does.

    >>> harmony(sc, "batch")
    >>> bixverse_gpu.graph.find_neighbours(sc, embd_to_use="harmony_gpu")

Unlike `bixverse.integrate.fast_mnn`, the GPU fastMNN corrects the PCA in the
object rather than computing one on batch-aware HVGs.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from beartype import beartype

from bixverse._core.integrate import _bbknn, _hard_state, _harmony, _labels
from bixverse._core.multimodal import CacheModality
from bixverse._core.subset import ScOrScSubset
from bixverse._validate import Verbose
from bixverse.params import to_rust
from bixverse_gpu import _bixverse_gpu
from bixverse_gpu.params import BbknnGpuParams, FastmnnGpuParams, HarmonyV2GpuParams

__all__ = [
    "bbknn",
    "fast_mnn",
    "harmony",
]

logger = logging.getLogger(__name__)


@beartype
def harmony(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    modality: CacheModality = "rna",
    harmony_params: HarmonyV2GpuParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> None:
    """Harmony v2 on the PCA on the GPU.

    One batch covariate only. The corrected embedding goes to the modality's
    `embeddings["harmony_gpu"]`, stamped as derived from `"pca"`.

    Args:
        sc: The experiment, with a PCA.
        batch_column: obs column with the batch.
        modality: The PCA to correct and the cache to write: `"rna"`, or
            `"adt"` on a `SingleCellsMultiModal`.
        harmony_params: The parameters. `None` uses the defaults; a `None`
            `k` becomes `min(round(n_cells / 30), 100)`.
        seed: Seed.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Raises:
        ValueError: If the PCA is stale, or the batch column is missing or
            incomplete.
        BixverseError: Without a GPU adapter.

    References:
        Patikas, et al., bioRxiv, 2026.
    """
    _harmony(
        sc,
        batch_column,
        None,
        harmony_params or HarmonyV2GpuParams(),
        "harmony_gpu",
        _bixverse_gpu.harmony_v2,
        modality,
        seed,
        verbose,
    )


@beartype
def bbknn(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    no_neighbours_to_keep: int = 5,
    no_embd_to_use: int | None = None,
    bbknn_params: BbknnGpuParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> None:
    """Batch-balanced kNN with the per-batch searches on the GPU.

    Writes what `bixverse.integrate.bbknn` writes: the kNN (the
    `no_neighbours_to_keep` nearest per row of the BBKNN distance matrix,
    balanced across batches) to `sc.cache.knn` and the max-symmetrised
    connectivities to `sc.cache.connectivities`, on the PCA.

    Args:
        sc: The experiment, with a PCA.
        batch_column: obs column with the batch.
        no_neighbours_to_keep: Neighbours per cell in the stored kNN; capped
            at `n_batches * neighbours_within_batch`.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        bbknn_params: The parameters. `None` uses the defaults.
        seed: Seed.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Raises:
        ValueError: If the PCA is stale, or the batch column is missing or
            incomplete.
        BixverseError: Without a GPU adapter.

    References:
        Polański, et al., Bioinformatics, 2020.
    """
    _bbknn(
        sc,
        batch_column,
        no_neighbours_to_keep=no_neighbours_to_keep,
        embd_to_use="pca",
        no_embd_to_use=no_embd_to_use,
        params=to_rust(bbknn_params or BbknnGpuParams()),
        seed=seed,
        verbose=verbose,
        run=_bixverse_gpu.bbknn_graph,
    )


@beartype
def fast_mnn(
    sc: ScOrScSubset,
    batch_column: str,
    *,
    no_embd_to_use: int | None = None,
    fastmnn_params: FastmnnGpuParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> None:
    """fastMNN on the PCA in the object, neighbour searches on the GPU.

    The corrected embedding goes to `sc.cache.embeddings["mnn_gpu"]`, stamped
    as derived from `"pca"`. Run the PCA on batch-aware HVGs first if that is
    what you want.

    Args:
        sc: The experiment, with a PCA.
        batch_column: obs column with the batch.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        fastmnn_params: The parameters. `None` uses the defaults.
        seed: Seed.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Raises:
        ValueError: If the PCA is stale, or the batch column is missing,
            incomplete or does not cover the PCA.
        BixverseError: Without a GPU adapter.

    References:
        Haghverdi, et al., Nat Biotechnol, 2018.
    """
    _hard_state(sc, ["pca"])
    pca = sc.cache.pca_factors
    if pca is None:
        warnings.warn(
            "No PCA embeddings found in the object. Returning class as is.",
            stacklevel=2,
        )
        return
    if no_embd_to_use is not None:
        pca = pca[:, : min(no_embd_to_use, pca.shape[1])]
    codes, levels = _labels(sc, batch_column)
    if len(codes) != pca.shape[0]:
        raise ValueError("The batch column does not cover the rows of the PCA.")
    if len(levels) < 2:
        warnings.warn(
            "The batch column only has one batch. Returning object as is.",
            stacklevel=2,
        )
        return
    if verbose:
        logger.info("Running fastMNN on the GPU.")
    sc.cache.embeddings["mnn_gpu"] = _bixverse_gpu.fast_mnn(
        np.ascontiguousarray(pca, dtype=np.float32),
        codes,
        to_rust(fastmnn_params or FastmnnGpuParams()),
        seed,
        verbose,
    )
    sc.stamp("mnn_gpu", parents=("pca",))
