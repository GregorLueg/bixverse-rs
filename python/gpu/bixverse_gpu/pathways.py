"""GPU SCENIC GRN inference, the port of R's `scenic_grn_sc_gpu`.

Returns the `bixverse.pathways.ScenicGrn` the CPU `scenic_grn` returns. The
tree ensembles run on the GPU; the target gene filter is the CPU one. Only
the tree learners (`"randomforest"`, `"extratrees"`) have a GPU path.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace

from beartype import beartype

from bixverse._core.scenic import ScenicGrn, _scenic_grn
from bixverse._core.subset import ScOrMc
from bixverse.params import ScenicParams
from bixverse_gpu import _bixverse_gpu

__all__ = [
    "scenic_grn",
]

###########
# Globals #
###########

#: R's default VRAM budget per wave of trees, 4 GiB.
WAVE_BYTE_BUDGET: int = 4 * 1024**3


@beartype
def scenic_grn(
    sc: ScOrMc,
    tf_ids: Sequence[str],
    *,
    scenic_params: ScenicParams | None = None,
    wave_byte_budget: int = WAVE_BYTE_BUDGET,
    genes_to_take: Sequence[str] | None = None,
    cells_to_take: Sequence[str] | None = None,
    streaming: bool | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> ScenicGrn:
    """GRN inference with the tree ensembles on the GPU, as R's
    `scenic_grn_sc_gpu`.

    Args:
        sc: The experiment, a subset or meta cells.
        tf_ids: The TF gene ids; ids not in the object warn and drop.
        scenic_params: The parameters, a tree learner. `None` uses the
            defaults.
        wave_byte_budget: VRAM budget per wave of trees, in bytes.
        genes_to_take: Target gene ids. `None` runs the CPU gene filter.
        cells_to_take: Cell (meta cell) ids. `None` for the kept cells.
        streaming: Stream the targets. `None` decides by size; unused for
            meta cells.
        seed: Seed for the learners.
        verbose: Controls verbosity.

    Returns:
        The `ScenicGrn`.

    Raises:
        ValueError: With the gradient boosting learner, or if no TF is in the
            object or none passes as a target.
        BixverseError: Without a GPU adapter.

    References:
        Aibar, et al., Nat. Methods, 2017.
    """
    params = scenic_params or ScenicParams()
    if params.learner_type == "grnboost2":
        raise ValueError(
            "GRNBoost2 (gradient boosting) is not supported on the GPU. Use "
            "bixverse.pathways.scenic_grn for the CPU implementation."
        )
    ext = SimpleNamespace(
        scenic_grn=partial(_bixverse_gpu.scenic_grn, wave_byte_budget=wave_byte_budget),
        mc_scenic_grn=partial(
            _bixverse_gpu.mc_scenic_grn, wave_byte_budget=wave_byte_budget
        ),
    )
    return _scenic_grn(
        sc,
        tf_ids,
        scenic_params=params,
        genes_to_take=genes_to_take,
        cells_to_take=cells_to_take,
        streaming=streaming,
        seed=seed,
        verbose=verbose,
        ext=ext,
    )
