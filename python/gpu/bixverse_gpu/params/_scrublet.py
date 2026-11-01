"""GPU Scrublet parameters, written by hand.

The kNN block is the GPU or the CPU one depending on `knn_backend`, so the
bundle is not generated. Both blocks share five key names, so the block has to
match the backend; a CPU-only knob would otherwise do nothing on the GPU arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from beartype import BeartypeConf, beartype

from bixverse.params import HvgDefaults, KnnDefaults, NormDoubletsDefaults
from bixverse.params._checks import check_range
from bixverse_gpu.params._generated import KnnGpuDefaults

_CONF = BeartypeConf(is_pep484_tower=True)


@beartype(conf=_CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class ScrubletGpuParams:
    """GPU Scrublet doublet detection parameters.

    The randomised SVD and the projection of the simulated doublets always run
    on the GPU; `knn_backend` picks where the kNN runs. There is no
    `svd_solver`: the GPU SVD is always randomised.

    Attributes:
        sim_doublet_ratio: Number of doublets to simulate relative to the
            observed cells.
        expected_doublet_rate: Expected doublet rate.
        stdev_doublet_rate: Uncertainty in the expected doublet rate.
        n_bins_histogram: Number of histogram bins for the Otsu threshold.
        manual_threshold: Manual doublet score threshold; `None` uses Otsu.
        no_pcs: Number of principal components.
        normalisation: The normalisation block.
        hvg: The HVG block.
        knn_backend: `"gpu"` or `"cpu"`.
        knn: The kNN block, `KnnGpuDefaults` for `"gpu"` and `KnnDefaults` for
            `"cpu"`. `None` is the backend's block with `k = 0`, which hands
            the choice of k to Rust.
    """

    sim_doublet_ratio: float = 1.5
    expected_doublet_rate: float = 0.1
    stdev_doublet_rate: float = 0.02
    n_bins_histogram: int = 100
    manual_threshold: float | None = None
    no_pcs: int = 30
    normalisation: NormDoubletsDefaults = field(default_factory=NormDoubletsDefaults)
    hvg: HvgDefaults = field(default_factory=HvgDefaults)
    knn_backend: Literal["gpu", "cpu"] = "gpu"
    knn: KnnGpuDefaults | KnnDefaults | None = None

    def __post_init__(self) -> None:
        check_range("sim_doublet_ratio", self.sim_doublet_ratio, 0, None, True, True)
        check_range(
            "expected_doublet_rate", self.expected_doublet_rate, 0, 1, False, False
        )
        check_range("stdev_doublet_rate", self.stdev_doublet_rate, 0, 1, False, False)
        check_range("n_bins_histogram", self.n_bins_histogram, 10, None, False, True)
        check_range("manual_threshold", self.manual_threshold, 0, None, False, True)
        check_range("no_pcs", self.no_pcs, 1, None, False, True)
        match self.knn_backend, self.knn:
            case "gpu", None:
                object.__setattr__(self, "knn", KnnGpuDefaults(k=0))
            case "cpu", None:
                object.__setattr__(self, "knn", KnnDefaults(k=0))
            case ("gpu", KnnGpuDefaults()) | ("cpu", KnnDefaults()):
                pass
            case backend, knn:
                raise ValueError(
                    f"knn_backend = '{backend}' needs a "
                    f"{'KnnGpuDefaults' if backend == 'gpu' else 'KnnDefaults'} kNN "
                    f"block, got {type(knn).__name__}."
                )
