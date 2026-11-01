"""GPU SEACells.

Returns the `bixverse.metacells.MetaCells` the CPU `generate_seacells`
returns. Both Frank-Wolfe solves run on the GPU; kernel construction,
archetype initialisation, the kNN and the aggregation are the CPU steps.
"""

from __future__ import annotations

from collections.abc import Sequence

from beartype import beartype

from bixverse._core.metacells import MetaCells
from bixverse._core.metacells_generate import _generate_seacells
from bixverse._core.subset import ScOrScSubset
from bixverse.params import SeacellsParams
from bixverse_gpu import _bixverse_gpu

__all__ = [
    "generate_seacells",
]


@beartype
def generate_seacells(
    sc: ScOrScSubset,
    seacell_params: SeacellsParams,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    cells_to_use: Sequence[str] | None = None,
    regenerate_knn: bool = False,
    target_size: float = 1e5,
    seed: int = 42,
    verbose: bool = False,
) -> MetaCells:
    """SEACells with the fit on the GPU.

    Inputs, kNN reuse and the result are those of
    `bixverse.metacells.generate_seacells`.

    Args:
        sc: The experiment or a subset of it.
        seacell_params: The SEACells and kNN parameters; `n_sea_cells` has no
            default.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        cells_to_use: Cell ids to restrict the meta cells to.
        regenerate_knn: Build the kNN on the embedding.
        target_size: Library size to normalise the meta cells to.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the embedding is missing, or it or the kNN is stale.
        BixverseError: Without a GPU adapter.

    References:
        Persad, et al., Nat Biotechnol, 2023.
    """
    return _generate_seacells(
        sc,
        seacell_params,
        embd_to_use=embd_to_use,
        no_embd_to_use=no_embd_to_use,
        cells_to_use=cells_to_use,
        regenerate_knn=regenerate_knn,
        target_size=target_size,
        seed=seed,
        verbose=verbose,
        run=_bixverse_gpu.seacells,
    )
