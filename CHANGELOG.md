# News

## 0.10.0

### Breaking

- `ann-search-rs` 0.12. Its `Verbosity` is now the one verbosity enum: the
  preludes re-export `ann_search_rs::Verbosity` and `parse_verbosity_level`
  maps to `parse_annsearch_verbosity`. `bixverse_sparse::verbosity` is gone.
  `verbosity_from_flag()` maps a `verbose: bool` onto it.
- `extract_knn` is gone from `KnnParams` and `KnnParamsGpu`, and
  `generate_knn_nndescent()` lost the argument. `knn_method = "nndescent"` is
  graph-only on every backend now (see Features), and the beam search on the GPU
  and MLX is opt-in through `"cagra"`. `warn_unsupported_extract()` is gone with it.
- `bbknn_first_neighbours()` is now `bbknn_nearest_neighbours()`: each
  batch-balanced row is sorted by distance (ties by index) before taking k, so
  the stored kNN holds the k nearest rather than the k lowest-index entries. The
  connectivity graph is unchanged; kBET and LISI computed after BBKNN change.
- The sparse Lanczos SVD on the CPU (`sparse_svd_lanczos()`) and in `bixverse-gpu`
  applies `SvdResults::flip_signs` (largest-magnitude loading positive), so runs
  are reproducible. Stored PCA factors can change sign against 0.9.4; distances,
  kNN and clusters don't.
- Symphony: `knn_majority_vote()` gives ties to the lowest label, as documented
  (was the highest). `build_symphony_reference()`, `symphony_map_query()` and
  `symphony_map_query_parts()` return errors on mismatched batch or gene map
  lengths instead of panicking.
- Plot extractors z-score by the sample SD on the store path too
  (`extract_norm_counts()`, `extract_norm_counts_multi()`), as the in-memory paths
  do. Scaled values shift by `sqrt(n/(n-1)) - 1`. Fewer than two values stay
  unscaled.
- `bixverse-r`: scDblFinder reads `gbm_n_trees` for the GBM rounds, not the Annoy
  `n_trees`, so the effective default goes from 50 rounds to the spec's 200.
  `parse_sigma_method()` accepts `snn_farthest` / `sigma_idx` and an unknown value
  errors; `sigma_idx` used to fall back silently to the SNN bandwidth, so WNN
  weights change for anyone who set it. DIALOGUE reads an integer `seed`; every run
  used 1234 before, so any non-default seed changes results.
- `parse_leiden_iterations()`, `parse_community_algorithm()`, `parse_magic_layer()`,
  the Bonsai parsers and the MetaCells2 similarity parsers moved out of `bixverse-r`
  into the crates that own the params.

### Features

- `"nndescent"` on the CPU, the GPU and MLX is the NN-Descent kNN graph built
  graph-only through `build_knn_graph_nndescent()` /
  `build_knn_graph_nndescent_gpu()`, with no index kept and no search run.
  The GPU one goes through the clustered build, so datasets past the
  per-binding limit work. Self-kNN only; the cross-queries (BBKNN, Symphony,
  fastMNN, `batch_knn_search()`) run it as NN-Descent / CAGRA.
  `generate_knn_nndescent_graph()` and `knn_graph_to_rows()` are the CPU
  entry points.
- `"cagra"` kNN method on the GPU and MLX, the graph with a beam search over it.
- MLX kNN dispatch: `dispatch_knn_mlx()`, `knn_with_dist_mlx()`, `knn_query_mlx()`.
- `SingleCellReading::read_genes_sparse_csr()`: genes from a gene-major store
  straight into CSR, without holding the CSC next to its transpose. Default
  method, so every reader has it.
- Logic lifted out of the R shell so every front end calls the same function:
  `hvg_on_store()`, `pca_on_store()`, `fit_residuals_on_store()` and the residual
  twins, the meta cell generators, `build_symphony_reference_on_store()`,
  `make_milor_nhoods()`, `fast_mnn_on_store()`, `seurat_cca_on_store()`,
  `seurat_rpca_on_store()`, `compare_knn()`, `pad_knn_rows()`, `run_wnn()`,
  `adt_clr()`, `cellphonedb_run()`, `vision_with_autocorrelation()`,
  `max_symmetrise_to_sparse_graph()`, `seacells_on_store_with()`,
  `knn_with_dist_gpu()`. `bixverse_sc_data::h5_scan` reads h5ad dimensions and
  detects the raw count slot, `h5_10x_io::write_tenx_h5()` writes 10x h5.
- serde `Deserialize` on the single cell params structs, with renames matching the
  R keys, so a front end needs one key map. This is what the Python wheels use.

### Fixes

- `meld` checked labels with `l > n_groups`, so a label equal to `n_groups`
  panicked. Now `>=`.
- `extract_grouped_gene_stats()` returns `InvalidArgument` on a length mismatch or
  an out-of-range group id instead of panicking.
- CPU Scrublet returns `InvalidArgument` when `no_pcs` is 0 or exceeds
  `min(genes, cells)` instead of a faer assertion.
- Hotspot's `Default` model is `"danb"`, matching the R spec.

- IVF in `dispatch_knn()` and `batch_knn_search()` passed `n_list` as
  `n_probe`, so it probed every list. It uses `n_probe` now.
- `CompressedSparseData2::transform()` panicked on a matrix without its first
  layer, i.e. anything read with `DataLayerReturn::Norm`. The empty layer now
  stays empty.

### Dependencies

- `ann-search-rs` 0.12, `bonsai-rs` 0.3.1, `edge-rs` 0.3.2.

### Performance

- `sparse_svd_lanczos()` holds one f32 copy of the matrix instead of a CSR, a
  CSC and two f64 casts: lower peak memory. Results differ from 0.9.4 at
  round-off.
- `compute_largest_eigenpairs_lanczos()` no longer clones its input or holds an
  f64 copy: lower peak memory.
- `sparse_covariance_svd()` builds one shared `m x m` accumulator instead of
  one per thread, drops the input CSC after the transpose and writes U
  directly: lower peak memory and faster. Results differ from 0.9.4 at
  round-off.
- The sparse PCA (`pca_on_sc_sparse()` and the `_stats` twin) reads the HVGs
  straight into CSR through `read_genes_sparse_csr()` when there are at least
  as many cells as genes, so none of its solvers transposes any more. Lower
  peak memory on large data, on macOS and Linux. Column statistics may differ
  from 0.9.4 at round-off.
- `read_genes_sparse()` and `read_cells_sparse()` free each per-thread
  fragment as it is copied into the output instead of holding them all to the
  end: lower peak memory on every sparse read, including the input of the
  GPU and MLX sparse PCA.
- The MLX sparse SVDs (`sparse_svd_lanczos_mlx()`,
  `randomised_sparse_svd_mlx()`) consume the value layer instead of copying
  it: lower peak memory.

## 0.9.4

### Breaking

- `GENE_FILE_BYTES_PER_NNZ` is 8 (was 10); budgets derived from it hold 25%
  more non-zeros per phase.
- `MtxReader::parse_mtx_quality()` returns `Result`; the streaming write
  rejects a non-ascending `cells_to_keep`; bucket files go to the system temp
  directory (`MtxReader::with_temp_dir()` to change it).
- Non-count MTX values error with `MtxNotACount`, `complex`, `array` and
  non-`general` files with `MtxUnsupported`.
- The h5ad and 10x quality passes error when a dataset fails to open.

### Feature

- `parquet` feature: h5ad obs / var and 10x barcodes / features to Parquet
  (`h5ad_frame_to_parquet()`, `tenx_frame_to_parquet()`), no arrow.
- `to_bytes()` / `from_bytes()` on `CompressedSparseData2` and `SparseGraph`;
  `SparseGraph::to_edge_pairs()` and `get_edge_number()`.
- Gzipped MTX input; `real` and `pattern` fields.
- `SingleCellReading::read_gene_parallel_raw()` for raw counts only.

### Performance

- h5ad and 10x h5 reads 5x faster; CSR to CSC, HVG and MTX loading 2x.

### Fix

- `real` MTX values were truncated and `pattern` files loaded empty.
- Fixed-length HDF5 strings over 63 bytes were truncated or failed.
- Multi-MTX scans allocated a cell-sized vector per chunk.

## 0.9.3

### Breaking

- `mad_outlier()` takes an `r_standard` flag; `true` scales the MAD like R's
  `mad()`, `false` keeps the raw MAD.

## 0.9.2

### Breaking

- MetaCells2 parameters follow metacells 0.9.5 (`SimilarityMethod`,
  `DeviantsParams`, no `MetacellsParams::random_seed`).
- `select_features()`, `choose_seeds()`, `compute_candidate_metacells()` and
  `find_deviant_cells()` return `Result`; `dissolve_metacells()` takes a mass
  vector.

### Feature

- MetaCells2 divide and conquer with rare gene modules
  (`compute_dac_metacells()`).
- Hierarchical clustering matching SciPy's `linkage`.
- decoupler ULM and MLM activity scores, matching decoupler 2.2.1.
- `bixverse-r`: `FromRList` for the new params.

### Fix

- The MetaCells2 pipeline matches metacells 0.9.5 stage by stage; deviants no
  longer build dense matrices.

## 0.9.1

### Feature

- New crate `bixverse-sparse` with the compressed sparse structures, re-exported
  by `bixverse-core`. `bixverse-sc-data` depends on it instead of core; with
  its `analysis` feature off the tree drops from 145 to 59 crates.

### Breaking

- `from_dense_matrix()` is now the `SparseFromDense` trait method (in the
  prelude). Sparse functions return `SparseErrors`, which the upper error
  enums convert. Plotting and synthetic data in `bixverse-sc-data` need its
  `analysis` feature.

## 0.9.0

### Breaking

- Single cell files are version 4 (`SC_FILE_VERSION`). Readers open version 3
  too; 0.8.x cannot read version 4. `new_v3()` writes the old layout.
- `compress_cell_row()` takes a `ChunkFormat`; version 4 payloads come from
  `ChunkFormat::encode_cell()` / `encode_gene()`.
- `faer` without default features (drops `npy` and `stacker`); `ann-search-rs`
  0.11.2, `edge-rs` 0.3.1.

### Feature

- Version 4 format: delta bit-packed indices and counts, the normalised layer
  recomputed on read, a choice of lz4 or zstd per writer.
- `upgrade_store_to_v4()` and `convert_*_to_v4()` for existing stores,
  `peek_file_version()` to check one.
- Reads prefetch into the page cache (unix).

## 0.8.1

### Feature

- `PcaSolver::Auto`, and exact sparse PCA on the GPU and MLX via a block
  Lanczos (`block_lanczos_thick_restart()`). GPU and MLX PCA honour
  `svd_solver`, so they are exact by default.

### Fix

- `sparse_svd_lanczos()` is thick-restarted, so trailing components converge.

## 0.8.0

### Breaking

- The crate is a workspace (`bixverse-core`, `-methods`, `-sc-data`, `-sc`,
  `-gpu`, `-sc-gpu`, `-mlx`, `-r`); `bixverse-rs` re-exports them under the old
  paths.
- One error enum per crate replaces `BixverseErrors`.
- R conversions moved to `bixverse-r` (`FromRList`, `FromRMap`, `ToRList`);
  the `r` feature is gone.
- `single-cell` no longer pulls in HDF5 (new `h5` feature); `gpu` is bulk only,
  single cell GPU needs `sc-gpu`.
- sNN builders replaced by `graph::knn_graph::build_knn_graph()`; the sNN prune
  default is 1/16.
- Fast clustering rewritten as `fast_clusters()`; `LeidenParams::n_iterations`
  is optional and defaults to 5.
- petgraph is gone; build PageRank graphs with `PageRankGraph::from_strings()`
  or `from_indices()`.

### Feature

- Each layer works on its own, e.g. the data layer without HDF5 or analysis code.
- `build_knn_graph()` with sNN, UMAP fuzzy simplicial set or plain union.
- `fast_clusters()` with CPU, GPU or MLX k-means coarsening and Louvain or
  Leiden on the centroid graph.

## 0.7.0

### Breaking

- `cubecl` 0.11, `cubek` 0.3, `faer` 0.24; the GPU API takes a
  `cubecl::Device` and is no longer generic over the runtime.
- `louvain_sparse_graph()` runs on the Leiden engine, so partitions may differ.
- `SingleCellReading` gains `read_cells_sparse()` and `read_genes_sparse()`.

### Feature

- Leiden community detection.
- Fused sparse batch readers on the cell store.
- Python bindings for the streaming engine, first used by scVI.
- Experimental MLX backend for Apple Silicon.
- Small cell subsets skip the gene-major sweep.

### Performance

- Faster SEACells GPU B update, sparse addition, transposes, filtered reads and
  exact covariance PCA on few cells.

## 0.6.4

### Breaking

- `write_compressed_cell_chunks()` takes `&[CellPayload]`.

### Feature

- Cell files carry a library-size index.

## 0.6.3

### Fix

- Personalised PageRank converges properly (L1 change, tolerance 1e-12) and
  handles dangling nodes consistently. `tied_diffusion_parallel()` takes the
  damping, iterations and tolerance.

## 0.6.2

- `edge-rs` 0.2.1; fixed a panic on small h5ad.

## 0.6.1

- `edge-rs` 0.2.0, with minimum sample filtering for NEBULA.

## 0.6.0

- GPU mNN, faster GPU randomised SVD, store archiving, CellPhoneDB.
- sNN pruning without isolated communities.
- Performance sweep across the crate; some numbers shift slightly.

## 0.5.10

- `ann-search-rs` bump for faster Annoy and NNDescent.

## 0.5.9

- Fix: reverse edges for the miloR kNN graph.

## 0.5.8

- Faster one-vs-many Wilcox and AUROC DGE.

## 0.5.7

- Bonsai trees (`bonsai` feature), CPU and GPU.
- Apple Accelerate GEMM on macOS (`accelerate`, default).
- Fix: `hdf5-metno` 0.15 for `ndarray` 0.17.

## 0.5.6

- Faster mtx, h5 and h5ad readers.

## 0.5.5

- Faster single cell PCA solvers; Bayesian and probabilistic PCA.
- No singletons in sNN graphs.
- Fix: drift in the GPU SVD.

## 0.5.4

- Fix: GPU NEBULA with any number of lanes.

## 0.5.3

- Fix: no snappy in h5ad Blosc, to avoid C++ linkage with R.

## 0.5.2

- scTransform v2, CellSweep, GPU NEBULA.
- More informative h5ad compression errors.

## 0.5.1

- Cosine distance, limma-voom, batch metrics (PCR, cell-type ASW, graph
  connectivity), faster constrained PageRank.

## 0.5.0

### Breaking

- `single-cell` no longer builds HDF5 from source (`hdf5-static`).
- kNN graphs return Euclidean, not squared Euclidean, distances.
- `ann-search-rs` 0.8.1 trait bounds.

### Feature

- NEBULA, bulk DGE, exhaustive search for miloR, CAGRA.

## 0.4.7

- DIALOGUE, better Hotspot membership, LDA.

## 0.4.6

- Vision scores for metacells.

## 0.4.5

- Consensus NMF; GPU NMF solvers.

## 0.4.4

- GPU Scrublet.
- Breaking: GPU k-means moved to `ann-search-rs` 0.5.2.

## 0.4.3

- Faster metacell HVG, Hotspot for metacells, better SCENIC binariser.
- Test suite refactor and small fixes.

## 0.4.2

- Fix: GPU k-means edge case.

## 0.4.1

- Faster HVG, GPU fast clusters, Palantir, PAGA.

## 0.4.0

### Breaking

- `CellGeneSparseWriter::new()` takes a `target_size`, stored in the header.
- Writer functions return `Result`.

### Fix

- Chunk parsers validate lengths; raw counts no longer saturate at `u16::MAX`;
  little-endian throughout; CLR checks the stored size factor.

## 0.3.12

- GPU SEACells, faster CPU SEACells, runtime-dispatched AVX2 / AVX-512.

## 0.3.11

- Streaming engine refactor.

## 0.3.10

- ScType with kNN smoothing.

## 0.3.9

- Faster GPU SCENIC, k-means, randomised SVD and correlations; AUCell variants.

## 0.3.8

- Bulk synthetic data, GPU SCENIC attempt, Seurat rPCA and CCA integration.

## 0.3.7

- Fix: IVF always returns k neighbours (`ann-search-rs` 0.4.4).

## 0.3.6

- Faster fgsea multi-level, plus a panic fix.

## 0.3.5

- singscore.

## 0.3.4

- E-distance for perturbation experiments.

## 0.3.3

- Fix: oversampling guard in the GPU sparse SVD.

## 0.3.2

- Dense h5ad reading.

## 0.3.1

- NMF, faster Harmony, multi-file 10x h5, PFlogPF, NicheNet.
- GPU sparse SVD, Harmony v2 and correlations.
- Fix: f64 overflow in mitch.

## 0.3.0

- Multi-file MTX, MELD, store merging, `multi-modal` (WNN, DSB), ScType,
  faster Harmony and Hotspot, memory-efficient SEACells and SuperCells.
- First GPU method: k-means (`gpu`).

## 0.2.0

- scDblFinder, Harmony v2, metacell methods, Walktrap, SuperCell 2.0, faster
  SEACells, mini-batch k-means, proper error handling.

## 0.1.7

- `ann-search-rs` bump.

## 0.1.6

- Single cell plotting helpers, sparse SVD in doublet detection, pure fastMNN,
  Harmony and SVD fixes.

## 0.1.5

- Faster Hotspot and SEACells, IVF index, files beyond `u16::MAX` genes and
  counts (v2 to v3 migration).

## 0.1.4

- Sparse SVDs, Harmony, SCENIC-like GRNs, multi-file and normalised-only h5ad.

## 0.1.3

- Fix: Scrublet auto detection.

## 0.1.2

- Label propagation with max hops, weights and directed graphs.

## 0.1.1

- `ann-search-rs` bump.

## 0.1.0

- Ported from [bixverse](https://github.com/GregorLueg/bixverse) as an
  independent crate, generic over the float type.
