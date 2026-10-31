"""GPU twins of the bixverse single cell methods, on wgpu.

Each module mirrors its `bixverse` namesake (`bixverse_gpu.pp` for
`bixverse.pp`, and so on) and its functions take the same experiment objects,
write the same cache slots and obs columns and return the same result types as
the CPU functions, so a pipeline swaps one import for the other:

    >>> from bixverse import pp
    >>> from bixverse_gpu import pp as pp_gpu
    >>> pp.find_hvg(sc, hvg_no=2000)
    >>> pp_gpu.calculate_pca(sc, 30)

Where R names the GPU result differently (`harmony_gpu`, `mnn_gpu`) so does
this. The GPU-specific knobs live in `bixverse_gpu.params`; everything else
takes the `bixverse.params` bundles. Probe the device with `gpu_available()`;
the twins raise `bixverse.BixverseError` without one.
"""

from bixverse_gpu import (
    de,
    doublets,
    embed,
    graph,
    integrate,
    metacells,
    nmf,
    params,
    pathways,
    pp,
    trees,
)
from bixverse_gpu._bixverse_gpu import __version__, gpu_available, gpu_plane_ops

__all__ = [
    "__version__",
    "de",
    "doublets",
    "embed",
    "gpu_available",
    "gpu_plane_ops",
    "graph",
    "integrate",
    "metacells",
    "nmf",
    "params",
    "pathways",
    "pp",
    "trees",
]
