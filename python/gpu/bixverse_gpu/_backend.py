"""Pick the wgpu or the MLX version of a compiled GPU function.

The MLX twins (`*_mlx`) are only compiled into a source build with the `mlx`
feature, so `backend="mlx"` on a wheel from PyPI raises with the build
instructions instead of an `AttributeError`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from beartype import beartype

from bixverse import BixverseError
from bixverse_gpu import _bixverse_gpu

###########
# Globals #
###########

Backend = Literal["wgpu", "mlx"]

MLX_MISSING = (
    "backend='mlx' needs a bixverse-gpu build with the MLX twins, Apple Silicon "
    "only. Install mlx-c (`brew install mlx-c`), then rebuild from `python/gpu`: "
    "`uv run --no-sync maturin develop --release --features mlx`. Check with "
    "`bixverse_gpu.mlx_available()`."
)

########
# Main #
########


@beartype
def pick(name: str, backend: Backend) -> Callable[..., Any]:
    """Return the compiled function for a backend.

    Args:
        name: The wgpu function's name in `_bixverse_gpu`, e.g. `"pca"`.
        backend: `"wgpu"`, or `"mlx"` for its `*_mlx` twin.

    Returns:
        The compiled function.

    Raises:
        BixverseError: If `backend="mlx"` and the MLX twins are not compiled
            in.

    Examples:
        >>> pick("pca", "wgpu") is _bixverse_gpu.pca
        True
    """
    if backend == "wgpu":
        return getattr(_bixverse_gpu, name)
    if not _bixverse_gpu.mlx_available():
        raise BixverseError(MLX_MISSING)
    return getattr(_bixverse_gpu, f"{name}_mlx")
