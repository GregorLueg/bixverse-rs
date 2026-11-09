# Pipelines

::: bixverse.pipeline
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Sub-clustering and per-group pipelines](../vignettes/single_cell_subsets.ipynb).

## Applying a pipeline

::: bixverse.pipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - Pipeline
        - Step
        - apply_pipeline
        - apply_pipeline_per_group
        - validate_pipeline

## Preprocessing steps

::: bixverse.pipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - step_hvg
        - step_pca
        - step_neighbours
        - step_clusters

## Integration steps

::: bixverse.pipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - step_harmony
        - step_harmony_v1
        - step_bbknn
        - step_fast_mnn

## Meta cells

::: bixverse.pipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - step_metacells
        - meta_cells_per_group
