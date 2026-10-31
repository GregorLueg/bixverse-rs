"""Parameter bundles for the GPU twins.

One frozen dataclass per bixverse.gpu R `params_*()` constructor the twins
take, flattened as R flattens them before Rust sees the list. Generated from
the R specs, except `ScrubletGpuParams`, whose R constructor is hand-written.
Twins with no GPU-specific knobs (PCA, SEACells, Bonsai, NMF, SCENIC) take
the CPU bundles from `bixverse.params`; `bixverse.params.to_rust` serialises
both.
"""

from __future__ import annotations

from bixverse_gpu.params import _generated
from bixverse_gpu.params._generated import *  # noqa: F403
from bixverse_gpu.params._scrublet import ScrubletGpuParams

__all__ = [*_generated.__all__, "ScrubletGpuParams"]
