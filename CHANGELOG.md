# News

## 0.4.0

Breaking changes in the streaming engine.

### Breaking changes

- `CellGeneSparseWriter::new` takes a fifth `target_size: f32`. It is stored in
  the file header and exposed via `SingleCellReading::target_size`, so CLR
  transforms and bin merges can verify the normalisation instead of assuming it.
  Carved out of reserved bytes, so existing files still open and report `None`.
- `write_cell_chunk`, `write_gene_chunk`, `finalise`, `write_r_counts` and
  `write_r_counts_csr` return `Result<_, BixverseErrors>`. Writer orientation
  asserts and the remaining h5ad panics are now typed errors.

### Fixes

- Fixed a panic in `CscGeneChunk::read_from_buffer`, which guarded on 32 bytes
  and then sliced 36.
- Fixed `migrate_v2_to_v3`, which compared against `SC_FILE_VERSION` (3) instead
  of 2.
- Chunk parsers now validate the total payload against the buffer before
  slicing. The three length fields are untrusted and previously fed two `unsafe`
  out-of-bounds reads.
- Raw counts no longer saturate at `u16::MAX`. `from_gene_chunks` /
  `from_cell_chunks` are generic over `FromPrimitive` and return
  `RawCountOverflow` instead of clipping; mtx ingest and the Scrublet doublet
  simulation keep full u32 width.
- The binary format is consistently little-endian.

## 0.3.12

### Features

- GPU-accelerated SEACells
- Speed improvements on the CPU SEACells version
- AVX-2 and AVX-512 SIMD arms now actually run on x86. They were gated behind
  `cfg(target_feature)`, which is compile-time, so stock builds fell all the
  way back to SSE2; they are runtime-dispatched via `target_feature` now.

## 0.3.11

### Features

- Large refactor for the streaming engine for single cell.

## 0.3.10

### Features

- ScType with kNN smoothing.

## 0.3.9

### Features

- **GPU improvements**:
  * An actually fast GPU-accelerated SCENIC version that beats CPU.
  * Improved k-means clustering on the GPU.
  * Faster randomised sparse SVD on the GPU.
  * Custom kernels for the correlations which beat `cubek` (at least on Apple
    Silicon).
- Different AUCell methods with true alternatives.

## 0.3.8

### Features

- Updated synthetic data set generation for bulk RNAseq.
- Attempt at a GPU-accelerated SCENIC version. Individual tree-building is
  faster; however, parallelised CPU smokes GPU (at least on Apple Silicon).
- Faster GPU correlations.
- Implementation of two additional batch correction methods:
  * Seurat rPCA batch correction.
  * Seurat CCA batch correction.

## 0.3.7

### Fix

- Bumped to `ann-search-rs = "0.4.4"` to ensure the IVF kNN search always
  returns k neighbours.

## 0.3.6

### Features

- Speed improvements to the fgsea multi-level implementation removing
  unnecessary computations and allocations.

### Fix

- Fixed an edge case in the multi-level implementation that caused panics.

## 0.3.5

### Features

- Implemented singscore from Foroutan et al., BMC Bioinform., 2018

## 0.3.4

### Features

- Added E-Distance calculations for perturbation experiments

## 0.3.3

### Fix

- Guard against too much oversampling in the single cell randomised, sparse SVD
  with GPU-acceleration.

## 0.3.2

### Features

#### Single cells

- H5Ad reading in for DENSE formats in single cell

## 0.3.1

Another big one

### Features

#### General

- NMF implementation that is exposed to both dense (for example RNAseq) and
  sparse data sets (for example single cell transcriptomics).

#### Single cells

- Improved verbosity on Harmony and better performance (version 1 and 2).
- Multi file support for h5 reading from 10x genomics
- PFlogPF normalisation prior to PCAs enabled for single cells, see
  [Booeshaghi et al.](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v3)
- NicheNet, see [Browaeys et al.](https://www.nature.com/articles/s41592-019-0667-5).

#### GPU-accelerated methods:

- Sparse, randomised SVD for single cell
- Harmony (version 2) on GPU.
- GPU-accelerated covariance and correlations (column-wise).

#### Fixes

- `f64` overflow problem with mitch when using large data sets. This is now
  fixed and regression tests are in place.

## 0.3.0

Major update / release

### Features

#### General

- Better error handling

#### Single cells:

- Multi file support for MTX reading (several 10x experiments to ingest).
- MELD implementation from Burkhardt, et al., Nat Biotechnol, 2021.
- Merging single cell binary files
- New feature gate `"multi-modal"`:
  * Weighted nearest neighbour graph method behind a new feature gate.
  * DSB normalisation for CITE Seq ADT
- First annotation helpers... ScType implemented
- Improved Harmony (version 1 and 2) and HotSpot with increased speed
- h5ad ingest automatically recognises where raw counts are being stored
- Updates to various package editions
- SEACells and SuperCells that are actually memory efficient. Run on a million
  cells in a ~20 GB memory pressure range.
- Improved batching for SCENIC

#### GPU-accelerated methods:

- First implementation of GPU-accelerated method (behind the feature gate
  `"gpu"`):
  * k-means clustering -> to be integrated in the future in some of the methods

## 0.2.0

Major update / release

### Features

#### Single cell methods

- scDblFinder implemented
- Refactor out of shared tree structure between scDblFinder and SCENIC
- Implementation of Harmony v2
- First methods (HVG selection, PCA, AUCell and SCENIC) for meta cells directly
- Made the kNN validation optional and improved the distance calculation with
  SIMD.
- Better loading in of MTX files that do not blow up during counting of cells
  per gene.
- Faster, better Louvain clustering for the doublet detection methods.
- Fast cluster version for Louvain for single cell that allows sweeping over a
  a set of resolutions.
- Actual proper implementation of WalkTrap community detection and not just
  pseudo hierarchical clustering.
- SuperCell 2.0 approach for SuperCells.
- Reduced memory pressure for the sparse SVDs that are streaming in from binary
  files due to discarding unneeded raw counts.
- Faster SEACells with better cache locality.

#### General

- Proper Rust-based error handling across the crate. Less `.unwrap()`, less
  panics!
- k-means implementation (mini batch) + clustering quality methods ported over
- Faster correlations, covariances and Euclidean distance calculations
- Version bump to extendr `0.9.0` and other crates.
- Some restructuring of modules: the meta cell generations live in their own
  module now.

## 0.1.7

### Features

- Version bump for `ann-search-rs`.

## 0.1.6

### Features

- Various plotting helpers for single cell (gene-based).
- Leverage sparse SVD for doublet detection methods.
- Version bump on `ann-search-rs` for KnnValidation error.
- Updates to the doublet detection parts: improved thresholding for Scrublet
  and refactor of the code to extract shared elements.
- Improved fastMNN code. Prior version was a mixture of mnnCorrect and fastMNN ->
  refactor to pure fastMNN.
- Fix: bug fix in the Harmony code.
- Fix: fix numerical stability issue in the SVD code for large data sets.
- Moved SIMD code over, so `ann-search-rs` dependency only pops up with
  single-cell features.

## 0.1.5

### Features

- Improved Hotspot performance
- Improved SEACells performance and reduced memory pressure
- Thresholded GRNBoost2 pending the data set size
- IVF index added after improvements in ann-search-rs
- Updated the files to be able to have more than u16::MAX features and deal with
  situations where the raw counts are larger than u16::MAX. Additionally, a
  function has been added to transform old `v2` files to `v3`.

### Fixes

- Bug in the Hotspot streaming version
- R can actually call GRNBoost2

## 0.1.4

### Features

- Sparse SVD and sparse randomised SVD were added for single cell.
- SIMD instructions for the single cell Hotspot method.
- Harmony batch correction for single cell.
- SCENIC-like approach for gene regulatory network (GRN) generation.
- Multi-file support for reading in several h5ad files at once.
- h5ad reader for situations where the file only has normalised counts.

### Fixes

*NA*

## 0.1.3

### Features

*NA*

### Fixes

- Hot fix in the auto detection in Scrublet which was broken.

## 0.1.2

### Features

- Modified graph label propagation algorithm to support max_hops, weights and
  directed graphs.

### Fixes

*NA*

## 0.1.1

### Features

`NA`

### Fixes

- Bumped the version of `ann-search-rs` to ensure consistency across packages/
  crates.

## 0.1.0 (official release)

This version ported over all of the Rust code from
[bixverse](https://github.com/GregorLueg/bixverse) as an independent crate.

### Features

- The hardcoded f32/f64 types were replaced with generic types.
- Redundancy in the codebase was removed.
- Restructuring of the codebase for better organisation.
- Wrong copy/paste documentation was properly updated.

### Fixes

*NA*
