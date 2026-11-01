# Preprocessing

::: bixverse.pp
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Analysing PBMCs with bixverse](../vignettes/pbmc_single_cell.ipynb) and [Pearson residuals and scTransform](../vignettes/single_cell_residuals.ipynb).

## Highly variable genes

::: bixverse.pp
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - find_hvg
        - get_hvg_data
        - find_hvg_batch_aware
        - BatchHvg

## PCA

::: bixverse.pp
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - calculate_pca

## Residuals and scTransform

::: bixverse.pp
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - fit_residuals
        - ResidualFit
        - sct_corrected_counts
        - CorrectedCounts

## MAGIC imputation

::: bixverse.pp
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - run_magic
