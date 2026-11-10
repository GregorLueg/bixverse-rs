# Batch correction

::: bixverse.integrate
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Batch correction with bixverse](../vignettes/single_cell_batch_corrections.ipynb).

## Embedding corrections

::: bixverse.integrate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - harmony
        - harmony_v1
        - fast_mnn
        - seurat_cca
        - seurat_rpca

## Batch-balanced neighbours

::: bixverse.integrate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - bbknn

## All metrics in one table

::: bixverse.integrate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - calculate_integration_metrics

## Batch mixing

::: bixverse.integrate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - calculate_kbet
        - KbetScores
        - calculate_lisi
        - LisiScores
        - calculate_batch_asw
        - BatchSilhouetteScores
        - calculate_pcr
        - PcrScores

## Biological conservation

::: bixverse.integrate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - calculate_cell_type_asw
        - CellTypeAswScores
        - calculate_graph_connectivity
        - GraphConnectivityScores
