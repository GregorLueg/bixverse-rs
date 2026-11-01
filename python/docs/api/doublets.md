# Doublets and ambient RNA

::: bixverse.doublets
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Doublet detection with bixverse](../vignettes/doublet_detection.ipynb) and [Ambient RNA removal with CellSweep](../vignettes/cell_sweep.ipynb).

## Scrublet

::: bixverse.doublets
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - scrublet
        - ScrubletRes
        - call_doublets_manual

## Boosted classification

::: bixverse.doublets
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - doublet_detection_boost
        - BoostRes

## scDblFinder

::: bixverse.doublets
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - scdblfinder
        - ScDblFinderRes
        - get_feature_mat
        - get_scores

## Results into obs

::: bixverse.doublets
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - get_data

## Empty droplets and CellSweep

::: bixverse.doublets
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - infer_empty_droplets
        - cellsweep
