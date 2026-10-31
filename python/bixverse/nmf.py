"""HALS NMF, the port of the bixverse R `nmf_sc`, `stabilised_nmf_sc`,
`consensus_nmf_sc` and `nmf_k_sweep_sc`.

Over a `SingleCells`, `SingleCellsSubset` or `MetaCells`, genes defaulting
to the HVGs. R keeps these in `R/methods_sc_pathways.R`; they get their own
module here, as they share nothing with the pathway scores. See
`bixverse._core.nmf`.
"""

from bixverse._core.nmf import (
    ConsensusNmfResult,
    NmfKSweepResult,
    NmfResult,
    StabilisedNmfResult,
    consensus_nmf,
    get_best_run,
    get_data,
    get_h,
    get_params,
    get_stability,
    get_w,
    nmf,
    nmf_k_sweep,
    stabilised_nmf,
)

__all__ = [
    "ConsensusNmfResult",
    "NmfKSweepResult",
    "NmfResult",
    "StabilisedNmfResult",
    "consensus_nmf",
    "get_best_run",
    "get_data",
    "get_h",
    "get_params",
    "get_stability",
    "get_w",
    "nmf",
    "nmf_k_sweep",
    "stabilised_nmf",
]
