# bixverse (Python) API index

Every public entry point, grouped by module. Generated from each module's
`__all__` and docstrings by `dev/gen_skill_api_index.py`. Do not edit by hand.

Grep this file to check whether a function exists before calling it, then
`help(bv.<module>.<name>)` or `inspect.signature()` for how to call it.

Version: 0.4.0.

## bv

Single-cell streaming for Python, in Rust.

- `Batch`: One minibatch of cells.
- `BixverseError`: Base class for every error raised by the bixverse Rust core.
- `CellLoader`: Iterate a cell-major store in minibatches, decoding ahead of the consumer.
- `CellReader`: Handle on a bixverse binary sparse store.
- `Counts`: A cells x genes count matrix in CSR, with what its rows and columns are.
- `CsrMatrix`: A CSR sparse matrix as its three component vectors.
- `KnnGraph`: A kNN result. Both arrays are held in C order (copied if needed).
- `MagicLayer`: The MAGIC imputed layer.
- `MetaCells`: Meta cells in memory.
- `ScCache`: Analysis results held in memory.
- `ScMap`: Ids and index sets of an experiment.
- `SingleCells`: A single cell experiment on disk.
- `SingleCellsAtac`: A single cell ATAC experiment on disk: peaks or tiles, no RNA.
- `SingleCellsMultiModal`: A single cell experiment with RNA plus ADT and / or ATAC.
- `SingleCellsSubset`: The cells of one level of an obs column.
- `SparseFileError`: A sparse store is missing, truncated, corrupt, or of the wrong format version or layout.
- `Stamp`: Provenance of one cached artefact.
- `TableNotFoundError`: A table the call needs is missing from the DuckDB.
- `__core_version__`
- `__version__`
- `get_counts`: Read counts for a selection of cells and genes.
- `install_agent_skill`: Install the bixverse agent skill where your coding agent looks for it.
- `merge_subset_obs`: Write obs columns of one or more subsets into the parent's obs.

## bv.io

Getting counts into an experiment directory, and back out.

- `H5adDims`: Dimensions and storage of an h5ad file, as `get_h5ad_dimensions()`.
- `Prescan`: The gene universe and per-file tasks of a multi-file load.
- `SyntheticAdtData`: Synthetic ADT counts.
- `SyntheticCellsweepData`: Synthetic counts with a planted ambient profile.
- `SyntheticData`: Synthetic counts with their obs and var tables.
- `SyntheticDialogueData`: Synthetic counts with a planted multicellular programme.
- `TenxH5Meta`: Layout and dimensions of a 10x h5 file, as `get_tenx_h5_metadata()`.
- `archive_sc_exp`: Archive the counts for cold storage.
- `detect_raw_count_slot`: Detect which slot of an h5ad file or Zarr store holds raw integer counts.
- `download_cd34_data`: The CD34 data set of the SEACells paper.
- `download_demuxlet_pbmc`: PBMCs with demuxlet doublet calls, for doublet detection.
- `download_dialogue_uc`: The ulcerative colitis subset of the DIALOGUE paper.
- `download_kang_pbmc`: The Kang et al. IFN-beta stimulated PBMCs (Nat Biotechnol, 2018).
- `download_marrow_cd34`: The bone marrow CD34 data of the Palantir paper.
- `download_pbmc3k`: The 10x PBMC 3k filtered matrices.
- `download_pbmc8k`: The 10x PBMC 8k data.
- `download_pbmc_1k_5p`: The unfiltered 10x PBMC 1k 5' matrix.
- `download_pbmc_batches`: Two PBMC h5ad files, for batch correction.
- `download_pbmc_multiome`: The 10x PBMC unsorted 3k multiome (Cell Ranger ARC 2.0).
- `download_pbmc_totalseq_data`: The 10k PBMC TotalSeq-B data, with ADT counts.
- `download_thymus_ageing`: The Baran-Gale et al. ageing thymus data (Development, 2020).
- `generate_cellsweep_test_data`: Synthetic counts with a planted ambient profile, for CellSweep.
- `generate_dialogue_test_data`: Synthetic counts with a planted multicellular programme, for DIALOGUE.
- `generate_single_cell_test_data`: Synthetic RNA counts with planted cell types.
- `generate_single_cell_test_data_adt`: Synthetic ADT counts, cell for cell with `generate_single_cell_test_data`.
- `get_cell_ranger_params`: The mtx parameters for a Cell Ranger output directory.
- `get_h5ad_dimensions`: Dimensions and storage layout of an h5ad file or anndata Zarr store.
- `get_tenx_h5_metadata`: Detect the layout of a 10x h5 file and read its dimensions.
- `load_atac_fragments`: Load an ATAC-only experiment from a fragment file.
- `load_atac_tenx`: Load an ATAC-only experiment from a 10x peak matrix h5.
- `load_csr`: Load an in-memory cells x genes count matrix.
- `load_existing`: Open an experiment directory and restore any state Python saved.
- `load_h5ad`: Load an h5ad file or anndata Zarr store.
- `load_h5ad_norm`: Load an h5ad file holding only normalised counts.
- `load_mtx`: Load an mtx file with its barcodes and features.
- `load_multi_h5ad`: Load several h5ad files into one experiment.
- `load_multi_mtx`: Load several mtx directories into one experiment.
- `load_multi_tenx_h5`: Load several 10x h5 files into one experiment.
- `load_tenx_h5`: Load a 10x CellRanger h5 file.
- `merge_sc_experiments`: Merge several experiments into a new one.
- `prescan_h5ad_files`: Prescan several h5ad files for a multi-file load.
- `prescan_mtx_dirs`: Prescan several CellRanger style mtx directories for a multi-file load.
- `prescan_tenx_h5_files`: Prescan several 10x h5 files for a multi-file load.
- `read_h5ad_metadata`: Read the obs and var tables and the layout of an h5ad file or Zarr store.
- `read_tenx_h5_metadata`: Read the barcode and feature tables and the layout of a 10x h5 file.
- `save`: Write the map's index sets and the cache next to the stores.
- `save_h5ad`: Write an experiment to a spec compliant h5ad file.
- `save_zarr`: Write an experiment to an AnnData Zarr (v3) store.
- `stream_h5ad`: `load_h5ad` with `h5ad_streaming=True`.
- `upgrade_sc_store`: Convert a v3 or v4 store to the v5 format in place.
- `write_cellranger_output`: Write counts as a CellRanger style trio.
- `write_h5ad_sc`: Write in-memory counts to a sparse h5ad file.
- `write_tenx_h5_sc`: Write in-memory counts to a 10x h5 file.

## bv.qc

Cell QC: per-cell metrics, then MAD or fixed-threshold outlier calls.

- `N_CELLS_STREAMING_THRESHOLD`
- `CellQc`: Outlier calls over several per-cell metrics.
- `MadOutlier`: MAD outlier calls on one metric.
- `flag_cells`: Add hard-threshold flags.
- `gene_set_proportions`: Share of each kept cell's reads in each gene set.
- `get_available_features`: Every obs column and gene id.
- `get_data`: The calls as an obs frame for `SingleCells.add_obs`.
- `per_cell_qc_outlier`: MAD outlier detection on a per-cell QC metric.
- `rescue_cells`: Un-flag MAD outliers that fall within safe bounds.
- `run_cell_qc`: Outlier detection over several per-cell QC metrics.
- `run_cell_qc_fixed`: Fixed-threshold cell QC: `run_cell_qc` with MAD off.
- `top_genes_perc`: Share of each kept cell's reads in its top N genes.

## bv.doublets

Doublet callers, empty-droplet calls and CellSweep ambient RNA removal.

- `BoostRes`: Boosted doublet classification result.
- `ScDblFinderRes`: scDblFinder result.
- `ScrubletRes`: Scrublet result.
- `call_doublets_manual`: Re-call Scrublet doublets at a hand-picked threshold.
- `cellsweep`: Remove ambient and bulk contamination with CellSweep.
- `doublet_detection_boost`: Boosted doublet classification.
- `get_data`: The per-cell calls as an obs frame for `SingleCells.add_obs`.
- `get_feature_mat`: The scDblFinder classifier features.
- `get_scores`: The weighted or cxds scores.
- `infer_empty_droplets`: Call empty droplets from the library sizes.
- `scdblfinder`: Cluster-aware doublet detection.
- `scrublet`: Scrublet doublet detection.

## bv.pp

Highly variable genes, PCA, residual models and MAGIC.

- `BatchHvg`: Batch-aware HVGs.
- `CorrectedCounts`: A corrected gene-major store written without its cell-major twin.
- `ResidualFit`: A fitted residual model.
- `calculate_pca`: PCA over the kept cells and the HVGs.
- `find_hvg`: Select highly variable genes over the kept cells.
- `find_hvg_batch_aware`: HVGs per batch, combined across batches.
- `fit_residuals`: Fit a scTransform (v2) or analytic Pearson residual model over the
- `get_hvg_data`: HVG statistics without touching the experiment.
- `run_magic`: Impute genes with MAGIC over the cached kNN graph.
- `sct_corrected_counts`: scTransform corrected counts as a new experiment.

## bv.graph

kNN search, the neighbour graph and clustering.

- `FastClusters`: Fast clustering result.
- `KnnMetrics`: A kNN compared against a reference.
- `calc_knn_metrics`: Recall at k and distance ratio of a kNN against a reference.
- `fast_cluster`: k-means coarsening, a centroid graph and Louvain / Leiden on it.
- `find_clusters`: Leiden or Louvain on the neighbour graph.
- `find_neighbours`: The kNN and the neighbour graph over an embedding.
- `generate_knn`: A kNN over an embedding, without touching the experiment.
- `get_connectivities`: The neighbour graph.

## bv.embed

2D embeddings: UMAP, densMAP, t-SNE, den-SNE, PHATE and ForceAtlas2.

- `densmap`: densMAP, UMAP with a density-preserving term.
- `densmap_defaults`: densMAP set to the package defaults.
- `densne`: den-SNE, t-SNE with a density-preserving term.
- `densne_defaults`: den-SNE set to the package defaults.
- `forceatlas2`: ForceAtlas2 on the fuzzy kNN graph.
- `forceatlas2_defaults`: ForceAtlas2 set to the package defaults.
- `phate`: PHATE.
- `phate_defaults`: PHATE set to the package defaults.
- `tsne`: t-SNE.
- `tsne_defaults`: t-SNE set to the package defaults.
- `umap`: UMAP.
- `umap_defaults`: UMAP set to the package defaults.

## bv.integrate

Batch correction and integration metrics.

- `BatchSilhouetteScores`: Batch silhouette width.
- `CellTypeAswScores`: Cell type silhouette width rescaled to `[0, 1]`.
- `GraphConnectivityScores`: Graph connectivity per cell type.
- `KbetScores`: kBET result.
- `LisiScores`: iLISI or cLISI.
- `PcrScores`: Principal component regression on batch.
- `bbknn`: Batch-balanced kNN.
- `calculate_batch_asw`: Batch average silhouette width on an embedding.
- `calculate_cell_type_asw`: Cell type silhouette width on an embedding, rescaled to `[0, 1]`.
- `calculate_graph_connectivity`: Graph connectivity per cell type on the cached kNN.
- `calculate_integration_metrics`: The batch mixing and biological conservation metrics in one row.
- `calculate_kbet`: kBET on the cached kNN.
- `calculate_lisi`: LISI on the cached kNN.
- `calculate_pcr`: Principal component regression on batch.
- `fast_mnn`: fastMNN on a PCA of the batch-aware HVGs.
- `harmony`: Harmony (v2) on the PCA or another embedding.
- `harmony_v1`: The original Harmony (v1) on the PCA or another embedding.
- `seurat_cca`: Seurat CCA anchor integration.
- `seurat_rpca`: Seurat reciprocal PCA anchor integration.

## bv.de

Marker genes, donor-aware differential expression and differential abundance.

- `MeldRes`: MELD densities.
- `MiloR`: Milo neighbourhoods and their test.
- `MiloRParams`: Parameters for the Milo neighbourhoods.
- `PseudoBulked`: Pseudobulked counts.
- `ScNebula`: NEBULA fits.
- `ScSpecificMarkers`: One-vs-many specific markers.
- `add_nhoods_info`: The majority cell type of each neighbourhood.
- `find_all_markers`: One-vs-rest Wilcoxon or logistic regression DGE for every group.
- `find_markers`: Wilcoxon or logistic regression DGE between two groups of cells.
- `find_specific_markers`: Markers that hold up against every rival group separately.
- `get_differential_abundance_res`: The differential abundance table; warns and returns `None` if untested.
- `get_index_cells`: The neighbourhood index cells, 0-based positions among the kept cells.
- `get_marker_comparisons`: The per-rival statistics.
- `get_marker_summary`: The per-gene summaries across every rival.
- `get_milor_abundances`: Milo neighbourhoods and their cell counts per sample.
- `get_params`: The parameters of the run.
- `get_pseudobulked`: Pseudobulk groups of cells.
- `meld`: MELD sample densities over the kNN graph.
- `nebula`: NEBULA's negative binomial gamma mixed model.
- `pseudobulk_dge`: edgeR quasi-likelihood DGE on pseudobulked cells.
- `run_edger_ql`: edgeR's quasi-likelihood chain.
- `test_nhoods`: Differential abundance of the neighbourhoods.

## bv.annotate

Cell type annotation with ScType, reference mapping with Symphony and Seurat.

- `CellMarkers`: The marker genes of one cell type, one entry of the
- `ScTypeCellResults`: Per-cell ScType calls.
- `ScTypeResults`: Per-cell ScType scores.
- `SymphonyReference`: A Symphony reference.
- `add_symphony_labels`: Attach obs columns to a reference.
- `assign_sc_type`: Per-cell ScType calls.
- `build_symphony_ref`: Build a Symphony reference.
- `calc_sc_type_scores`: ScType scores per cell.
- `get_scores`: The scores of ScType results.
- `get_symphony_hvg_names`: The HVG gene ids in loading order.
- `get_symphony_labels`: The stored labels in `z_corr` row order.
- `get_symphony_loadings`: The PCA gene loadings, `(n_hvgs, d)`.
- `get_symphony_z_corr`: The corrected reference embedding, `(N, d)`.
- `load_symphony_ref`: Read a reference written by `save_symphony_ref`.
- `map_symphony_query`: Map a query onto a reference.
- `prepare_cell_markers`: Marker genes per cell type.
- `save_symphony_ref`: Write a reference to one `.npz`, readable without pickling.
- `score_clusters`: One ScType call per cluster.
- `transfer_labels_seurat`: Seurat label transfer: CCA anchors, then `TransferData`.
- `transfer_labels_symphony`: kNN majority vote of reference labels onto a mapped query.

## bv.pathways

Gene set scores, HotSpot gene modules and SCENIC regulons.

- `Hotspot`: HotSpot gene-gene local correlations.
- `LmActivityRes`: ULM / MLM scores with p-values.
- `RegulonBinarisation`: On/off regulon calls.
- `ScMatrixRes`: Cells x gene sets scores, for cells and meta cells alike.
- `ScenicGrn`: SCENIC GRN inference results.
- `VisionAutocorRes`: VISION scores with their autocorrelation.
- `aucell`: AUCell scores.
- `binarise_regulon_activity`: On/off calls per regulon.
- `build_regulons`: Regulons from the TF to gene links.
- `generate_hotspot_membership`: Gene modules from the pair Z-scores.
- `generate_null_perm_gs`: Random signatures for the VISION autocorrelation null.
- `get_cistarget_res`: The motif enrichment; empty with a warning
- `get_data`: The scores as an obs frame for `add_obs`.
- `get_hotspot_membership`: The module membership.
- `get_params`: The parameters of the run.
- `get_tf_to_gene`: The TF to gene links; empty with a warning
- `hotspot_autocor`: HotSpot gene autocorrelation.
- `hotspot_gene_cor`: HotSpot gene-gene local correlations.
- `identify_tf_to_genes`: TF to gene links from the importances.
- `lm_activity`: ULM / MLM activity scores (decoupler style).
- `module_scores`: Module activity scores.
- `prep_lm_network`: A source-target network as weighted gene sets.
- `run_cistarget`: CisTarget motif enrichment of gene sets.
- `scenic_gene_filter`: Genes with enough counts and enough expressing cells.
- `scenic_grn`: GRN inference, the GENIE3 / GRNBoost2 step of SCENIC.
- `tf_to_genes_correlations`: Sign the TF to gene links by correlation.
- `tf_to_genes_motif_enrichment`: Motif support for the TF to gene links.
- `vision`: VISION signature scores.
- `vision_w_autocor`: VISION scores with their local autocorrelation.

## bv.lr

Ligand-receptor analysis, two ways: CellPhoneDB on the expression in your

- `LigandTargetInfluence`: The ligands x genes influence matrix.
- `as_ligand_target_influence`: Wrap an existing ligand-target matrix.
- `cellphonedb`: CellPhoneDB ligand-receptor analysis.
- `compute_expression_info`: Per-cluster mean expression and expressing fraction.
- `generate_ligand_target_influence`: The NicheNet-style ligand to target influence matrix.
- `get_cellphonedb_db`: The CellPhoneDB v5.0.0 ligand-receptor database.
- `get_influence`: The ligands x genes influence; the names are
- `ligand_activity_scores`: Ligand activities against gene sets.
- `prioritise_interactions`: Prioritise sender-ligand-receiver-receptor interactions.
- `resolve_weights`: The prioritisation weights.

## bv.nmf

Non-negative matrix factorisation of the counts into gene programmes.

- `ConsensusNmfResult`: Consensus HALS NMF.
- `NmfKSweepResult`: A consensus NMF k sweep.
- `NmfResult`: A single HALS NMF run.
- `StabilisedNmfResult`: Randomly initialised HALS NMF restarts.
- `consensus_nmf`: Consensus HALS NMF (cNMF).
- `get_best_run`: The lowest loss run as an `NmfResult`.
- `get_data`: The cell usages as an obs frame for `add_obs`: one
- `get_h`: The cell usages: `(k, n_cells)`, one per run for
- `get_params`: The parameters of the run.
- `get_stability`: The consensus diagnostics.
- `get_w`: The gene loadings: `gene_id` and one column per
- `nmf`: One HALS NMF run.
- `nmf_k_sweep`: Consensus NMF over a range of k.
- `stabilised_nmf`: Randomly initialised HALS NMF restarts.

## bv.dialogue

DIALOGUE: multicellular programmes, genes that are cell type specific but whose

- `DialogueResult`: A DIALOGUE run.
- `dialogue`: Multicellular programmes.
- `get_params`: The parameters of the run.
- `get_results`: The programme table.

## bv.traj

Palantir pseudotime, PAGA graph abstraction and gene trends.

- `GeneTrendsRes`: Gene trends along the Palantir branches.
- `PagaRes`: A PAGA graph abstraction.
- `PalantirRes`: A Palantir run.
- `run_gene_trends`: Gene trends over the Palantir pseudotime.
- `run_paga`: PAGA graph abstraction of the cached kNN graph.
- `run_palantir`: Palantir trajectory inference over the cached kNN graph.

## bv.trees

Bonsai trees over cells or meta cells.

- `BonsaiTree`: A Bonsai tree with its layout.
- `bonsai`: A Bonsai tree over the cells.
- `relayout_bonsai`: The tree laid out again without searching.
- `set_bonsai_embedding`: Store the leaf coordinates as an embedding.

## bv.metacells

Meta cell generators, quality metrics, merging and persistence.

- `MetaCellAssignment`: Where the meta cells came from.
- `MetaCells`: Meta cells in memory.
- `calc_diffusion_coordinates`: Diffusion map of the source cells and a density region per meta cell.
- `calc_manifold_metrics`: Compactness and separation of each meta cell in the diffusion map.
- `calc_meta_cell_compactness`: Compactness of any meta cell assignment in a diffusion map.
- `calc_meta_cell_purity`: Label purity of each meta cell, written to obs.
- `calc_meta_cell_separation`: Separation of any meta cell assignment in a diffusion map. A single
- `generate_bt_meta_cells`: Bootstrapped meta cells after hdWGCNA.
- `generate_metacells2`: MetaCells2 (divide and conquer) on the raw counts.
- `generate_seacells`: SEACells, kernel archetypal analysis.
- `generate_supercells`: SuperCells, walktrap on the kNN graph.
- `get_meta_cell_purity`: `calc_meta_cell_purity` without touching the object.
- `load_metacells`: Read a `MetaCells` written by `save_metacells`.
- `merge_meta_cells`: Row-bind several `MetaCells` into one.
- `meta_cells_per_group`: Source-pure meta cells: one generator run per group, merged.
- `save_metacells`: Write a `MetaCells` to a directory: Parquet, npz and JSON, no pickle.

## bv.mm

Multi-modal single cell: ADT counts, their PCA and the WNN graph.

- `AdtCounts`: ADT counts with their normalisation.
- `DsbInfo`: What a DSB normalisation estimated on the way.
- `WnnSlot`: The weighted nearest neighbour integration.
- `add_adt_counts`: Add ADT counts for the kept cells.
- `calculate_pca_adt`: PCA over the scaled, normalised ADT counts.
- `generate_wnn_graph`: Weighted nearest neighbour graph over two modalities.
- `get_adt_counts`: The dense ADT counts.
- `get_adt_feature_info`: Cells expressing each protein.
- `get_adt_names`: The ADT feature ids.
- `get_adt_sample_info`: ADT capture per cell.
- `get_wnn_weights`: The per-cell modality weights of the WNN integration.
- `new_adt_counts_clr`: CLR-normalised ADT counts.
- `new_adt_counts_dsb`: DSB-normalised ADT counts.

## bv.atac

Single cell ATAC: peak or tile counts, QC, TF-IDF, LSI, links to genes.

- `AtacInfo`: What the ATAC store holds and how far it was processed.
- `add_atac_fragments`: Count a fragment file into peaks or genome tiles as the ATAC store.
- `add_atac_tenx`: Add the `Peaks` of a 10x multiome h5 as the ATAC store.
- `amulet`: Flag multiplets from fragment overlaps (AMULET), written to obs.
- `annotate_peaks`: The closest gene of every ATAC feature (Signac `ClosestFeature`).
- `barcode_fragment_counts`: Fragments per barcode, every barcode in the file, for cell calling.
- `calculate_lsi`: LSI over the TF-IDF of the kept cells (Signac `RunSVD`).
- `call_peaks`: Call peaks on the kept cells, pooled or per group (MACS3 `callpeak`).
- `find_top_features`: The most accessible features (Signac `FindTopFeatures`).
- `fragment_profiles`: TSS insertion profile and fragment length histogram of the kept cells.
- `fragment_qc`: Per-cell QC from the fragment file, written to obs.
- `gene_activity`: Count fragments over gene body plus promoter into a gene activity store.
- `get_gene_activity`: Open an experiment's gene activity store, its cell filter brought in line.
- `get_lsi_depth_cor`: Correlation of each LSI component with the library size.
- `merge_fragments`: Merge per-sample fragment files into one, as `cellranger-arc aggr` does.
- `peak_to_gene`: Link peaks to nearby genes by correlation over groups of cells.
- `project_lsi`: Project the query's kept cells onto the reference LSI.
- `read_cellranger_barcodes`: The cell barcodes Cell Ranger ATAC called.
- `remove_blacklisted`: Drop the peaks that overlap a blacklist region.
- `run_rlsi`: rLSI integration, Signac's `FindIntegrationAnchors(reduction = "rlsi")`.
- `run_tfidf`: Switch the ATAC store to TF-IDF over the kept cells (Signac method 1).

## bv.pipeline

Pipelines: the preprocessing calls you repeat on every subset, written once.

- `Pipeline`: A linear chain of steps. Inert until applied.
- `Step`: One pipeline step.
- `apply_pipeline`: Run each step in order.
- `apply_pipeline_per_group`: Run a pipeline independently on each group.
- `meta_cells_per_group`: Source-pure meta cells: one generator run per group, merged.
- `step_bbknn`: Pipeline step for `bixverse.integrate.bbknn`.
- `step_clusters`: Pipeline step for `bixverse.graph.find_clusters`.
- `step_fast_mnn`: Pipeline step for `bixverse.integrate.fast_mnn`.
- `step_harmony`: Pipeline step for `bixverse.integrate.harmony`.
- `step_harmony_v1`: Pipeline step for `bixverse.integrate.harmony_v1`.
- `step_hvg`: Pipeline step for `bixverse.pp.find_hvg`.
- `step_metacells`: Pipeline step for a meta cell generator.
- `step_neighbours`: Pipeline step for `bixverse.graph.find_neighbours`.
- `step_pca`: Pipeline step for `bixverse.pp.calculate_pca`.
- `validate_pipeline`: Check that a pipeline can run on an object of class `cls`.

## bv.pl

Plot data extractors and matplotlib plots for single cell results.

- `GraphPlotData`: A graph laid out on an embedding: the PAGA or Milo node-link plot.
- `bx_colors`: The bixverse palettes.
- `density_plot`: QC densities per group with the outlier groups named.
- `dot_plot`: Marker expression across groups, colour the scaled mean and size the
- `embedding_plot`: An embedding coloured by an obs column.
- `extract_dot_plot_data`: Per-group mean expression and percent expressing.
- `extract_embedding_data`: Embedding coordinates for plotting.
- `extract_feature_pair`: Two features side by side for a scatter.
- `extract_feature_plot_data`: Per-cell expression on an embedding, long.
- `extract_gene_expression`: Per-cell expression, one column per feature.
- `extract_gene_violin_data`: Per-cell expression with its group, long.
- `extract_milo_plot_data`: The Milo neighbourhood graph with every neighbourhood at its index
- `extract_paga_plot_data`: The PAGA graph with every cluster at the centroid of its cells.
- `feature_plot`: Feature expression over an embedding, one panel per feature.
- `feature_scatter_plot`: Two features against each other, coloured by point density or binned
- `fragment_length_plot`: Fragment length distribution, one line per group (Signac
- `heatmap_plot`: Genes x cells, cells blocked by group with a colour strip per group.
- `joint_plot`: Genes per cell against UMIs per cell, hexbinned with marginal
- `lsi_depth_plot`: Correlation of each LSI component with the library size (Signac
- `milo_nhood_plot`: The Milo neighbourhood graph over an embedding.
- `paga_plot`: The PAGA graph over an embedding, every cluster at the centroid of
- `stacked_violin_plot`: Expression across groups, one row of violins per feature.
- `tss_profile_plot`: Tn5 insertions around the TSSs, one line per group (Signac `TSSPlot`).
- `violin_plot`: QC violins.

## bv.loader

Minibatch iteration over a cell-major store.

- `Batch`: One minibatch of cells.
- `CellLoader`: Iterate a cell-major store in minibatches, decoding ahead of the consumer.

## bv.nn

Neural network models trained straight off a bixverse store.

- `ScviFit`: A trained scVI model plus what is needed to use it again.
- `ScviHistory`: Per-epoch training means, one entry per epoch.
- `ScviParams`: scVI hyperparameters.
- `scvi_elbo`: Mean per-cell negative ELBO and its two terms, at KL weight 1.
- `scvi_latent`: The scVI embedding: the posterior mean of z, as `get_latent_representation`.
- `train_scvi`: Train scVI on raw counts streamed from a bixverse store.

## bv.params

Parameter bundles for the single cell suite.

- `AprParams`: Wrapper function for analytic Pearson residual parameters
- `AucellParams`: Wrapper function for parameters for AUCell
- `BbknnParams`: Wrapper function for the BBKNN parameters
- `BonsaiParams`: Wrapper function for Bonsai parameters
- `BoostParams`: Wrapper function for Boost parameters
- `BranchSelectionParams`: Wrapper function for the branch cell selection parameters
- `BtMetacellsParams`: Wrapper function for parameters for bootstrapped meta cell generation
- `CellphonedbParams`: Parameters for the CellPhoneDB analysis
- `CellsweepParams`: Default parameters for CellSweep denoising
- `CistargetParams`: Wrapper function to CisTarget parameters
- `DialogueHlmParams`: Wrapper function for the DIALOGUE mixed model parameters
- `DialoguePmdParams`: Wrapper function for the DIALOGUE decomposition parameters
- `DialogueRefineParams`: Wrapper function for the DIALOGUE refinement parameters
- `DsbParams`: Default parameters for DSB ADT normalisation
- `EdgerQlParams`: Wrapper function for parameters for the edgeR quasi-likelihood workflow
- `EmptyDropletsParams`: Parameters for identifying empty droplets
- `FastClusterParams`: Fast single cell clustering parameters
- `FastmnnParams`: Wrapper function for the fastMNN parameters
- `GeneTrendsParams`: Wrapper function for gene trend parameters
- `HarmonyParams`: Default parameters for Harmony batch correction
- `HarmonyV2Params`: Default parameters for Harmony v2 batch correction
- `HotspotParams`: Wrapper function for parameters for HotSpot
- `HvgDefaults`: Helper function to generate HVG defaults
- `HvgParams`: Wrapper function for HVG detection parameters.
- `HvgScranDefaults`: Helper function to generate the scran HVG trend defaults
- `KnnDefaults`: Helper function to generate kNN defaults
- `KnnGraphDefaults`: Helper function to generate the kNN graph defaults
- `KnnParams`: Parameters for single cell kNN searches
- `LigandTargetParams`: Parameters for ligand to target influence computation
- `LmActivityParams`: Wrapper function to generate ULM / MLM activity parameters
- `MagicParams`: Wrapper function for MAGIC imputation parameters
- `MeldParams`: Constructor for MELD parameters
- `Metacells2DacParams`: Wrapper function for the MetaCells2 divide and conquer parameters
- `Metacells2Params`: Wrapper function for the MetaCells2 per-pile parameters
- `MinQualityParams`: Wrapper function to generate QC metric params for single cell
- `MtxIoParams`: Wrapper function to provide data for mtx-based loading
- `NebulaParams`: Wrapper function for parameters for NEBULA
- `NeighboursParams`: Wrapper function for parameters for neighbour identification in single cell
- `NmfConsensusParams`: Wrapper function for consensus NMF parameters
- `NmfHalsParams`: Wrapper function for NMF (HALS) parameters
- `NormDoubletsDefaults`: Helper function to generate normalisation defaults for doublet detection.
- `PalantirParams`: Wrapper function for Palantir parameters
- `PcaDefaults`: Helper function to generate default parameters for PCA
- `PcaParams`: Wrapper for PCA specifically designed for single cells
- `ScdblfinderParams`: Wrapper function for scDblFinder doublet detection parameters
- `ScenicBinariseParams`: Wrapper function for parameters for the SCENIC binarisation
- `ScenicExtraTreesDefaults`: Default parameters for the SCENIC ExtraTrees regression learner
- `ScenicGradientBoostingDefaults`: Default parameters for the SCENIC GradientBoosting (GRNBoost2) regression learner
- `ScenicRandomForestDefaults`: Default parameters for the SCENIC RandomForest regression learner
- `ScrubletParams`: Wrapper function for Scrublet doublet detection parameters
- `SctransformParams`: Wrapper function for scTransform (v2) parameters
- `SctypeCellsParams`: Parameters for the per-cell ScType assignment
- `SeacellsParams`: Wrapper function for the SEACells parameters
- `SeuratCcaParams`: Wrapper function for the Seurat CCA parameters
- `SeuratRpcaParams`: Wrapper function for the Seurat rPCA parameters
- `SupercellParams`: Wrapper function for parameters for SuperCell generation
- `SymphonyMapParams`: Default parameters for Symphony query mapping
- `SyntheticCellsweepParams`: Default parameters for generation of synthetic CellSweep data
- `SyntheticDataAdtParams`: Default parameters for generation of synthetic single cell data (ADT)
- `SyntheticDataParams`: Default parameters for generation of synthetic single cell data (RNA)
- `SyntheticDialogueParams`: Default parameters for generation of synthetic DIALOGUE data
- `VisionParams`: Wrapper function for parameters for VISION with auto-correlation
- `WnnParams`: Wrapper function for WNN parameters
- `ScenicParams`: Constructor for SCENIC parameters.
