# NMF

::: bixverse.nmf
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Worked through in [Gene set analysis and GRN inference on PBMCs](../vignettes/bag_of_genes_single_cells.ipynb).

## Choosing k

::: bixverse.nmf
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - nmf_k_sweep
        - NmfKSweepResult

## Consensus NMF

::: bixverse.nmf
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - consensus_nmf
        - ConsensusNmfResult
        - get_stability

## Single and repeated runs

::: bixverse.nmf
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - nmf
        - NmfResult
        - stabilised_nmf
        - StabilisedNmfResult
        - get_best_run

## Reading results

::: bixverse.nmf
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - get_w
        - get_h
        - get_data
        - get_params
