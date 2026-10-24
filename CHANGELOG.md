# News

## 0.9.2

### Breaking

- MetaCells2 parameters follow upstream metacells 0.9.5: `SimilarityMethod` is
  `AbsPearson` (default) | `Pearson` plus a `log_data` flag, the kNN factors
  are `f64`, `DeviantsParams` drops `max_gene_fraction` and gains the noisy-gene
  fields, and `MetacellsParams::random_seed` is gone (the seed is an argument).
- `select_features()`, `choose_seeds()`, `compute_candidate_metacells()` and
  `find_deviant_cells()` return `Result`; `dissolve_metacells()` takes a mass
  vector; `Pile` gains `mass_per_cell` and `Pile::from_raw()`.

### Feature

- MetaCells2 divide and conquer (`compute_dac_metacells()`, `DacParams`),
  including rare gene modules (`RareParams`, on by default), streaming piles
  from the cell store.
- Hierarchical clustering matching SciPy's `linkage` (ward, complete, average,
  weighted) in `ml::clustering::hierarchical`.
- decoupler ULM and MLM activity scores in `enrichment::lm_activity`, dense and
  on the cell store (`calculate_lm_activity()`). Matches decoupler 2.2.1.
- `bixverse-r`: `FromRList` for the MetaCells2, divide-and-conquer, rare-module
  and `LmActivityParams` params, and `r_list_to_feature_set_weights()`.

### Fix

- The direct MetaCells2 pipeline now matches metacells 0.9.5 (parity tests on
  every deterministic stage). Among others: min-cut strength, similarity
  transform, kNN ranks and degree retry, deviant gap skip, top-3 statistic,
  relative-variance window and the candidates loop. Deviants no longer build
  dense cells x genes matrices.

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
