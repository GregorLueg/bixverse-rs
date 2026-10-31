"""Differential expression, markers, NEBULA, pseudobulk DE, Milo and MELD,
the port of the bixverse R DGE and differential abundance methods.

Marker tests return polars frames; the specific markers, NEBULA, Milo and
MELD return records with R's getters. The markers, NEBULA and the
pseudobulk functions also take a `SingleCellsSubset`, as in R. See
`bixverse._core.de` for the workflow.
"""

from bixverse._core.de import (
    MeldRes,
    MiloR,
    MiloRParams,
    PseudoBulked,
    ScNebula,
    ScSpecificMarkers,
    add_nhoods_info,
    find_all_markers,
    find_markers,
    find_specific_markers,
    get_differential_abundance_res,
    get_index_cells,
    get_marker_comparisons,
    get_marker_summary,
    get_milor_abundances,
    get_params,
    get_pseudobulked,
    meld,
    nebula,
    pseudobulk_dge,
    run_edger_ql,
    test_nhoods,
)

__all__ = [
    "MeldRes",
    "MiloR",
    "MiloRParams",
    "PseudoBulked",
    "ScNebula",
    "ScSpecificMarkers",
    "add_nhoods_info",
    "find_all_markers",
    "find_markers",
    "find_specific_markers",
    "get_differential_abundance_res",
    "get_index_cells",
    "get_marker_comparisons",
    "get_marker_summary",
    "get_milor_abundances",
    "get_params",
    "get_pseudobulked",
    "meld",
    "nebula",
    "pseudobulk_dge",
    "run_edger_ql",
    "test_nhoods",
]
