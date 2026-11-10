# Input and output

::: bixverse.io
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Single-cell data I/O in bixverse](../vignettes/single_cell_io.ipynb) and [Scaling to millions of cells with bixverse](../vignettes/single_cell_big_data.ipynb).

## Loading one file

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - load_csr
        - load_h5ad
        - load_h5ad_norm
        - stream_h5ad
        - load_tenx_h5
        - load_mtx
        - get_cell_ranger_params

## Loading several files

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - prescan_h5ad_files
        - prescan_tenx_h5_files
        - prescan_mtx_dirs
        - Prescan
        - load_multi_h5ad
        - load_multi_tenx_h5
        - load_multi_mtx
        - merge_sc_experiments

## Looking at a file first

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - detect_raw_count_slot
        - get_h5ad_dimensions
        - H5adDims
        - read_h5ad_metadata
        - get_tenx_h5_metadata
        - TenxH5Meta
        - read_tenx_h5_metadata

## Saving and reopening

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - save
        - load_existing

## Exporting

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - save_h5ad
        - write_h5ad_sc
        - write_tenx_h5_sc
        - write_cellranger_output

## Store maintenance

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - archive_sc_exp
        - upgrade_sc_store

## Synthetic data

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - generate_single_cell_test_data
        - SyntheticData
        - generate_single_cell_test_data_adt
        - SyntheticAdtData
        - generate_cellsweep_test_data
        - SyntheticCellsweepData
        - generate_dialogue_test_data
        - SyntheticDialogueData

## Example data sets

::: bixverse.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - download_pbmc3k
        - download_pbmc8k
        - download_pbmc_1k_5p
        - download_pbmc_batches
        - download_pbmc_totalseq_data
        - download_kang_pbmc
        - download_demuxlet_pbmc
        - download_cd34_data
        - download_marrow_cd34
        - download_dialogue_uc
        - download_thymus_ageing
