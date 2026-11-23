# ATAC

::: bixverse.atac
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Joint RNA and ATAC](../vignettes/atac_multiome.ipynb) and
[ATAC without RNA](../vignettes/atac_only.ipynb). The ATAC-only loaders,
`bv.io.load_atac_fragments` and `bv.io.load_atac_tenx`, are on the
[I/O page](io.md); the `SingleCellsAtac` class on the [core page](core.md).

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

## Barcodes and fragment files

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - barcode_fragment_counts
        - read_cellranger_barcodes
        - merge_fragments

## QC and multiplets

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - fragment_qc
        - fragment_profiles
        - amulet

## Peaks

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - call_peaks
        - remove_blacklisted
        - annotate_peaks

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

## Gene activity and links

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - gene_activity
        - get_gene_activity
        - peak_to_gene

## Mapping and integration

::: bixverse.atac
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - project_lsi
        - run_rlsi
