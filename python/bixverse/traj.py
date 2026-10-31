"""Trajectory inference, the port of the bixverse R Palantir, PAGA and gene
trend methods.

All three take a `SingleCells`, `SingleCellsSubset` or `MetaCells` with a
kNN graph and return frozen records keyed by cell id; nothing is written to
obs, as in R. Bonsai trees live in `bixverse.trees`, as R keeps them apart.
See `bixverse._core.traj` for the workflow.
"""

from bixverse._core.traj import (
    GeneTrendsRes,
    PagaRes,
    PalantirRes,
    run_gene_trends,
    run_paga,
    run_palantir,
)

__all__ = [
    "GeneTrendsRes",
    "PagaRes",
    "PalantirRes",
    "run_gene_trends",
    "run_paga",
    "run_palantir",
]
