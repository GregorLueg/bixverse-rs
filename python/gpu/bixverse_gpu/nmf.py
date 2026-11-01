"""GPU HALS NMF: one run, stabilised restarts, consensus NMF and the k sweep.

The port of R's `nmf_gpu_sc`, `stabilised_nmf_gpu_sc`, `consensus_nmf_gpu_sc`
and `nmf_k_sweep_gpu_sc`. Same inputs and result classes as `bixverse.nmf`;
only the HALS solver runs on the GPU, which caps the rank at
`NMF_GPU_MAX_RANK`. The consensus step stays on the CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

from beartype import beartype

from bixverse._core.nmf import (
    ConsensusNmfResult,
    NmfKSweepResult,
    NmfResult,
    Preprocessing,
    StabilisedNmfResult,
    _consensus_nmf,
    _nmf,
    _nmf_k_sweep,
    _stabilised_nmf,
)
from bixverse._core.subset import ScOrMc
from bixverse.params import NmfConsensusParams, NmfHalsParams
from bixverse_gpu import _bixverse_gpu

__all__ = [
    "consensus_nmf",
    "nmf",
    "nmf_k_sweep",
    "stabilised_nmf",
]

###########
# Globals #
###########

#: R's `NMF_GPU_MAX_RANK`: the largest k the GPU HALS solver takes.
NMF_GPU_MAX_RANK: int = 128


@beartype
def _check_rank(k: int) -> None:
    """R's `.assert_nmf_gpu_rank`.

    Raises:
        ValueError: Above `NMF_GPU_MAX_RANK`.
    """
    if k > NMF_GPU_MAX_RANK:
        raise ValueError(
            f"The GPU NMF solver caps the rank at {NMF_GPU_MAX_RANK}, but {k} was "
            "requested. Lower k, or use the CPU version in bixverse.nmf, which has "
            "no such cap."
        )


@beartype
def nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> NmfResult:
    """One HALS NMF run on the GPU, as R's `nmf_gpu_sc`.

    Args:
        sc: The experiment, a subset or meta cells.
        k: Number of components, at most `NMF_GPU_MAX_RANK`.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver. `None` uses the defaults (NNDSVD).
        seed: Seed for a random initialisation.
        verbose: Controls verbosity.

    Returns:
        The `bixverse.nmf.NmfResult`.

    Raises:
        ValueError: On `k` out of range or without genes.
        BixverseError: Without a GPU adapter.
    """
    _check_rank(k)
    return _nmf(
        sc,
        k,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        seed=seed,
        verbose=verbose,
        ext=_bixverse_gpu,
    )


@beartype
def stabilised_nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: bool = False,
) -> StabilisedNmfResult:
    """Randomly initialised HALS NMF restarts on the GPU, as R's
    `stabilised_nmf_gpu_sc`.

    Args:
        sc: The experiment, a subset or meta cells.
        k: Number of components, at most `NMF_GPU_MAX_RANK`.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver; `nmf_init` is ignored.
        n_runs: Number of restarts.
        seed: Base seed of the restarts.
        verbose: Controls verbosity.

    Returns:
        The `bixverse.nmf.StabilisedNmfResult`.

    Raises:
        ValueError: On `k` or `n_runs` out of range, or without genes.
        BixverseError: Without a GPU adapter.
    """
    _check_rank(k)
    return _stabilised_nmf(
        sc,
        k,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        n_runs=n_runs,
        seed=seed,
        verbose=verbose,
        ext=_bixverse_gpu,
    )


@beartype
def consensus_nmf(
    sc: ScOrMc,
    k: int,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    nmf_consensus_params: NmfConsensusParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: bool = False,
) -> ConsensusNmfResult:
    """Consensus HALS NMF (cNMF) with the restarts on the GPU, as R's
    `consensus_nmf_gpu_sc`.

    Args:
        sc: The experiment, a subset or meta cells.
        k: Number of components, 2 to `NMF_GPU_MAX_RANK`.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver.
        nmf_consensus_params: The consensus step. `None` uses the defaults.
        n_runs: Number of restarts, at least 2.
        seed: Base seed of the restarts and the consensus k-means.
        verbose: Controls verbosity.

    Returns:
        The `bixverse.nmf.ConsensusNmfResult`.

    Raises:
        ValueError: On `k` or `n_runs` out of range, without genes, or when
            the consensus step fails.
        BixverseError: Without a GPU adapter.

    References:
        Kotliar, et al., eLife, 2019.
    """
    _check_rank(k)
    return _consensus_nmf(
        sc,
        k,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        nmf_consensus_params=nmf_consensus_params,
        n_runs=n_runs,
        seed=seed,
        verbose=verbose,
        ext=_bixverse_gpu,
    )


@beartype
def nmf_k_sweep(
    sc: ScOrMc,
    k_range: Sequence[int] | range,
    *,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    preprocessing: Preprocessing = "none",
    use_second_layer: bool = True,
    nmf_hals_params: NmfHalsParams | None = None,
    nmf_consensus_params: NmfConsensusParams | None = None,
    n_runs: int = 30,
    seed: int = 42,
    verbose: bool = False,
) -> NmfKSweepResult:
    """Consensus NMF over a range of k with the restarts on the GPU, as R's
    `nmf_k_sweep_gpu_sc`.

    Args:
        sc: The experiment, a subset or meta cells.
        k_range: The ranks, unique, 2 to `NMF_GPU_MAX_RANK`.
        cell_ids: Cell (meta cell) ids. `None` for the kept cells.
        gene_ids: Gene ids. `None` for the HVGs.
        preprocessing: `"none"`, per gene `"sd"` or `"sqrt_sd"` scaling.
        use_second_layer: Factorise the normalised rather than the raw counts.
        nmf_hals_params: The solver.
        nmf_consensus_params: The consensus step. `None` uses the defaults.
        n_runs: Restarts per k, at least 2.
        seed: Base seed of the restarts and the consensus k-means.
        verbose: Controls verbosity.

    Returns:
        The `bixverse.nmf.NmfKSweepResult`.

    Raises:
        ValueError: On an invalid `k_range` or `n_runs`, or without genes.
        BixverseError: Without a GPU adapter.

    References:
        Kotliar, et al., eLife, 2019.
    """
    if len(k_range):
        _check_rank(max(k_range))
    return _nmf_k_sweep(
        sc,
        k_range,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        preprocessing=preprocessing,
        use_second_layer=use_second_layer,
        nmf_hals_params=nmf_hals_params,
        nmf_consensus_params=nmf_consensus_params,
        n_runs=n_runs,
        seed=seed,
        verbose=verbose,
        ext=_bixverse_gpu,
    )
