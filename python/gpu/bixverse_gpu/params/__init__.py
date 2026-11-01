"""Parameter bundles for the GPU twins.

One frozen dataclass per parameter bundle the twins take, flat as Rust sees
them. Generated from the parameter specs, except `ScrubletGpuParams`, which is
hand-written.
Twins with no GPU-specific knobs (PCA, SEACells, Bonsai, NMF, SCENIC) take
the CPU bundles from `bixverse.params`; `bixverse.params.to_rust` serialises
both.
"""

from __future__ import annotations

from bixverse_gpu.params import _generated
from bixverse_gpu.params._generated import *  # noqa: F403
from bixverse_gpu.params._scrublet import ScrubletGpuParams

__all__ = [*_generated.__all__, "ScrubletGpuParams"]
