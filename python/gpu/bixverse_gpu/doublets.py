"""GPU Scrublet, the port of R's `scrublet_gpu_sc`.

Returns the `bixverse.doublets.ScrubletRes` the CPU `scrublet` returns, so
`call_doublets_manual` and `get_data` work unchanged, `group_by` included.
"""

from __future__ import annotations

from beartype import beartype

from bixverse._core.doublets import ScrubletRes, Selection, _scrublet
from bixverse._core.single_cells import SingleCells
from bixverse.params import to_rust
from bixverse_gpu import _bixverse_gpu
from bixverse_gpu.params import ScrubletGpuParams


@beartype
def scrublet(
    sc: SingleCells,
    *,
    scrublet_params: ScrubletGpuParams | None = None,
    seed: int = 42,
    streaming: bool | None = None,
    cells_to_use: Selection | None = None,
    group_by: str | None = None,
    return_combined_pca: bool = False,
    return_pairs: bool = False,
    verbose: bool = False,
) -> ScrubletRes:
    """Scrublet doublet detection with the SVD, the doublet projection and
    (by default) the kNN on the GPU, as R's `scrublet_gpu_sc`.

    HVG selection, doublet simulation, the kNN classifier and the Otsu
    threshold are the CPU steps of `bixverse.doublets.scrublet`.

    Args:
        sc: The experiment.
        scrublet_params: The parameters. `None` uses the defaults.
        seed: Seed.
        streaming: Stream the HVG pass. `None` streams on large data.
        cells_to_use: 0-based cell indices or ids to run on. `None` uses the
            kept cells.
        group_by: obs column to run per group on, e.g. the sample.
        return_combined_pca: Also return the PCA of observed and simulated
            cells.
        return_pairs: Also return the parents of the simulated doublets.
        verbose: Controls verbosity.

    Returns:
        The `ScrubletRes`.

    Raises:
        ValueError: If `group_by` is missing or incomplete, or a group is too
            small.
        BixverseError: Without a GPU adapter.

    References:
        Wolock, et al., Cell Syst, 2019.
    """
    return _scrublet(
        sc,
        to_rust(scrublet_params or ScrubletGpuParams()),
        _bixverse_gpu.scrublet,
        seed=seed,
        streaming=streaming,
        cells_to_use=cells_to_use,
        group_by=group_by,
        return_combined_pca=return_combined_pca,
        return_pairs=return_pairs,
        verbose=verbose,
    )
