# Annotation

::: bixverse.annotate
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Clustering, cell type annotation and reference mapping](../vignettes/cell_type_annotation.ipynb).

## ScType

::: bixverse.annotate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - prepare_cell_markers
        - CellMarkers
        - calc_sc_type_scores
        - ScTypeResults
        - get_scores
        - score_clusters
        - assign_sc_type
        - ScTypeCellResults

## Symphony reference

::: bixverse.annotate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - build_symphony_ref
        - SymphonyReference
        - add_symphony_labels
        - get_symphony_loadings
        - get_symphony_z_corr
        - get_symphony_hvg_names
        - get_symphony_labels

## Mapping a query

::: bixverse.annotate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - map_symphony_query
        - transfer_labels_symphony

## Saving and loading

::: bixverse.annotate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - save_symphony_ref
        - load_symphony_ref
