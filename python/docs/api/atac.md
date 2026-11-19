# ATAC

::: bixverse.atac
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Joint RNA and ATAC](../vignettes/atac_multiome.ipynb).

## Adding the counts

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - add_atac_tenx
        - add_atac_fragments
        - AtacInfo

## QC and multiplets

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - atac_fragment_qc
        - amulet

## Peak calling

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - call_peaks

## TF-IDF and LSI

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - run_tfidf
        - find_top_features
        - calculate_lsi
        - get_lsi_depth_cor

## Gene activity

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - gene_activity

## Mapping and integration

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - project_lsi
        - run_rlsi
