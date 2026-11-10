# Gene sets and regulons

::: bixverse.pathways
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Gene set analysis and GRN inference on PBMCs](../vignettes/bag_of_genes_single_cells.ipynb).

## Gene set scores

::: bixverse.pathways
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - aucell
        - module_scores
        - ScMatrixRes
        - get_data

## VISION signatures

::: bixverse.pathways
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - vision
        - vision_w_autocor
        - VisionAutocorRes
        - generate_null_perm_gs

## TF activity from a network

::: bixverse.pathways
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - lm_activity
        - LmActivityRes
        - prep_lm_network

## HotSpot gene modules

::: bixverse.pathways
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - hotspot_autocor
        - hotspot_gene_cor
        - Hotspot
        - generate_hotspot_membership
        - get_hotspot_membership

## SCENIC network inference

::: bixverse.pathways
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - scenic_gene_filter
        - scenic_grn
        - ScenicGrn
        - get_params
        - identify_tf_to_genes
        - tf_to_genes_correlations
        - get_tf_to_gene

## Motif enrichment

::: bixverse.pathways
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - run_cistarget
        - tf_to_genes_motif_enrichment
        - get_cistarget_res

## Regulons

::: bixverse.pathways
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - build_regulons
        - binarise_regulon_activity
        - RegulonBinarisation
