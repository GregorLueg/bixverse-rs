"""Parameter bundles for the GPU twins.

One frozen dataclass per parameter bundle the twins take, flat as Rust sees
them; `ScrubletGpuParams` sits in its own module.
Twins with no GPU-specific knobs (PCA, SEACells, Bonsai, NMF, SCENIC) take
the CPU bundles from `bixverse.params`; `bixverse.params.to_rust` serialises
both.
"""

from __future__ import annotations

from bixverse_gpu.params import _specs
from bixverse_gpu.params._scrublet import ScrubletGpuParams
from bixverse_gpu.params._specs import *  # noqa: F403

__all__ = [*_specs.__all__, "ScrubletGpuParams"]
