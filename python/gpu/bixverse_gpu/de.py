"""GPU NEBULA, the port of R's `nebula_gpu_sc`.

Returns the `bixverse.de.ScNebula` the CPU `nebula` returns. Only stage two,
the per-gene penalised fits, runs on the GPU, in `f32`, so the estimates sit
close to the CPU ones rather than on them. No `reml` on the GPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from beartype import BeartypeConf, beartype

from bixverse._core.de import ScNebula, _nebula
from bixverse._core.single_cells import Selection
from bixverse._core.subset import ScOrScSubset
from bixverse_gpu import _bixverse_gpu
from bixverse_gpu.params import NebulaGpuParams

# `float` sequences accept ints, as in `bixverse.de`.
_CONF = BeartypeConf(is_pep484_tower=True)


@beartype(conf=_CONF)
def nebula(
    sc: ScOrScSubset,
    subject_col: str,
    design: str,
    *,
    coef: int | str | None = None,
    contrast: np.ndarray | Sequence[float] | None = None,
    genes_to_use: Selection | None = None,
    cells_to_use: Sequence[str] | None = None,
    offset: np.ndarray | Sequence[float] | None = None,
    nebula_params: NebulaGpuParams | None = None,
    verbose: bool = False,
) -> ScNebula:
    """NEBULA with the per-gene fits on the GPU, as R's `nebula_gpu_sc`.

    Inputs and the result are those of `bixverse.de.nebula`.

    Args:
        sc: The experiment.
        subject_col: obs column with the subject (donor), the random effect.
        design: Formula over obs, e.g. `"~ condition + age"`.
        coef: The coefficient the Wald test reports, a 0-based design column
            or its name. `None` takes the last column.
        contrast: One weight per design column. Exclusive with `coef`.
        genes_to_use: 0-based gene indices or ids. `None` fits every gene.
        cells_to_use: Cell ids to fit. `None` takes the kept cells.
        offset: Strictly positive scaling per cell. `None` uses the library
            sizes.
        nebula_params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The `ScNebula`.

    Raises:
        ValueError: On missing columns, no cells left, both `coef` and
            `contrast`, or a bad offset.
        BixverseError: Without a GPU adapter that runs plane operations.

    References:
        He, et al., Commun Biol, 2021.
    """
    return _nebula(
        sc,
        subject_col,
        design,
        coef=coef,
        contrast=contrast,
        genes_to_use=genes_to_use,
        cells_to_use=cells_to_use,
        offset=offset,
        nebula_params=nebula_params or NebulaGpuParams(),
        verbose=verbose,
        ext=_bixverse_gpu,
    )
