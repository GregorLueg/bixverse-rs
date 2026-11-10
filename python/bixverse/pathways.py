"""Gene set scores, HotSpot gene modules and SCENIC regulons.

Three jobs. Score every cell against gene sets (module scores, AUCell, VISION,
ULM / MLM), find gene modules that vary smoothly over the kNN graph (HotSpot),
and infer TF regulons and score those (SCENIC).

Start with `aucell` for gene sets; `vision` takes signed signatures,
`lm_activity` a weighted TF-target network. For modules call `hotspot_autocor`,
then `hotspot_gene_cor` on the genes that score, then
`generate_hotspot_membership`. For regulons run `scenic_grn`,
`identify_tf_to_genes`, `tf_to_genes_correlations`, optionally
`tf_to_genes_motif_enrichment`, then `build_regulons` and score them with
`aucell`. Each SCENIC step takes the `ScenicGrn` of the last and returns an
updated copy; the input is never modified.

**What lands where.** Nothing here writes to obs, var, `sc.map` or `sc.cache`.
Scores come back as an `ScMatrixRes` (cells by gene sets) and `get_data` turns
one into an obs frame for `sc.add_obs`. The HotSpot, VISION autocorrelation
and SCENIC functions return their own result objects and data frames. All of
them take a `SingleCells`, a `SingleCellsSubset` or a `MetaCells` (the meta
cells compute on their in-memory counts), except `module_scores`, which takes
no meta cells.

```python
genes = sc.gene_ids().tolist()
gene_sets = {"set_a": genes[:20], "set_b": genes[20:50]}
res = bv.pathways.aucell(sc, gene_sets)
sc.add_obs(bv.pathways.get_data(res))
```

HotSpot reads the PCA and, by default, the cached kNN graph:

```python
bv.pp.find_hvg(sc, hvg_no=500)
bv.pp.calculate_pca(sc, no_pcs=20)
bv.graph.find_neighbours(sc)
autocor = bv.pathways.hotspot_autocor(sc)
top = autocor.sort("z_score", descending=True)["gene_id"].head(100).to_list()
hs = bv.pathways.hotspot_gene_cor(sc, genes_to_take=top)
hs = bv.pathways.generate_hotspot_membership(hs, min_size=5)
modules = bv.pathways.get_hotspot_membership(hs)
```

The SCENIC chain, without the motif step:

```python
tf_ids = genes[:30]  # your TF list, as gene ids
grn = bv.pathways.scenic_grn(sc, tf_ids)
grn = bv.pathways.identify_tf_to_genes(grn, n_sd=1.5)
grn = bv.pathways.tf_to_genes_correlations(grn, sc)
regulons = bv.pathways.build_regulons(grn, use_leading_edge=False, min_genes=5)
auc = bv.pathways.aucell(sc, regulons)
on_off = bv.pathways.binarise_regulon_activity(auc)
```

**Gotchas.** Gene sets, TFs and networks use the gene ids of var, not symbols.
Unknown ids drop; for the TFs in `scenic_grn` you only hear about it with
`verbose` on. `hotspot_autocor` and `hotspot_gene_cor` raise if the
embedding or the cached kNN is stale after a change of the cell filter, and the
gene by gene output of `hotspot_gene_cor` grows with the square of
`genes_to_take`, so cut it down by autocorrelation first. `cells_to_take` drops
the cached kNN and searches again. The motif ranking database and the
motif-to-TF annotation are yours to supply: `run_cistarget` and
`tf_to_genes_motif_enrichment` take the ranks as a genes by motifs integer
matrix and the annotation as a frame with `motif`, `TF` and
`annotationSource`, and `build_regulons` keeps only links with motif support
unless you pass `use_leading_edge=False`. `build_regulons` and `lm_activity`
return `None` with a warning when nothing survives their size filters.
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
