# Meta cells

::: bixverse.metacells
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Meta cells with bixverse](../vignettes/meta_cells.ipynb).

## Generating meta cells

::: bixverse.metacells
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - generate_bt_meta_cells
        - generate_seacells
        - generate_supercells
        - generate_metacells2

## The MetaCells object

::: bixverse.metacells
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - MetaCells
        - MetaCellAssignment

## Label purity

::: bixverse.metacells
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - calc_meta_cell_purity
        - get_meta_cell_purity

## Diffusion map and manifold metrics

::: bixverse.metacells
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - calc_diffusion_coordinates
        - calc_manifold_metrics
        - calc_meta_cell_compactness
        - calc_meta_cell_separation

## Merging and saving

::: bixverse.metacells
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - meta_cells_per_group
        - merge_meta_cells
        - save_metacells
        - load_metacells
