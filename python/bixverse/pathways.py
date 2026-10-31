"""Pathway activity, HotSpot and SCENIC, the port of the bixverse R module
scores, AUCell, VISION, ULM / MLM, HotSpot and SCENIC methods.

Everything takes a `SingleCells`, `SingleCellsSubset` or `MetaCells` where R
dispatches on them (module scores: no meta cells). Scores come back as
`ScMatrixRes`; `get_data` turns one into an obs frame. Nothing is written to
obs, as in R. See `bixverse._core.pathways` and `bixverse._core.scenic`.
"""

from bixverse._core.pathways import (
    Hotspot,
    LmActivityRes,
    ScMatrixRes,
    VisionAutocorRes,
    aucell,
    generate_hotspot_membership,
    generate_null_perm_gs,
    get_data,
    get_hotspot_membership,
    hotspot_autocor,
    hotspot_gene_cor,
    lm_activity,
    module_scores,
    prep_lm_network,
    vision,
    vision_w_autocor,
)
from bixverse._core.scenic import (
    RegulonBinarisation,
    ScenicGrn,
    binarise_regulon_activity,
    build_regulons,
    get_cistarget_res,
    get_params,
    get_tf_to_gene,
    identify_tf_to_genes,
    run_cistarget,
    scenic_gene_filter,
    scenic_grn,
    tf_to_genes_correlations,
    tf_to_genes_motif_enrichment,
)

__all__ = [
    "Hotspot",
    "LmActivityRes",
    "RegulonBinarisation",
    "ScMatrixRes",
    "ScenicGrn",
    "VisionAutocorRes",
    "aucell",
    "binarise_regulon_activity",
    "build_regulons",
    "generate_hotspot_membership",
    "generate_null_perm_gs",
    "get_cistarget_res",
    "get_data",
    "get_hotspot_membership",
    "get_params",
    "get_tf_to_gene",
    "hotspot_autocor",
    "hotspot_gene_cor",
    "identify_tf_to_genes",
    "lm_activity",
    "module_scores",
    "prep_lm_network",
    "run_cistarget",
    "scenic_gene_filter",
    "scenic_grn",
    "tf_to_genes_correlations",
    "tf_to_genes_motif_enrichment",
    "vision",
    "vision_w_autocor",
]
