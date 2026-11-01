# Quality control

::: bixverse.qc
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Analysing PBMCs with bixverse](../vignettes/pbmc_single_cell.ipynb).

## Per-cell metrics

::: bixverse.qc
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - gene_set_proportions
        - top_genes_perc
        - get_available_features

## Outlier calls

::: bixverse.qc
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - run_cell_qc
        - run_cell_qc_fixed
        - CellQc
        - get_data
        - per_cell_qc_outlier
        - MadOutlier

## Adjusting the calls

::: bixverse.qc
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - flag_cells
        - rescue_cells
