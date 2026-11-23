//! Ingestion from HDF5 based formats (h5ad, 10x CellRanger h5) and anndata
//! Zarr stores into the bixverse binary store.

pub mod ann_store;
pub mod chunked;
pub mod filters;
pub mod h5ad;
pub mod h5ad_frame;
pub mod h5ad_multifile;
pub mod scan;
pub mod tenx;
pub mod tenx_adt;
pub mod tenx_multifile;

/// Rows (cells, or genes on a CSC layout) requested per HDF5 read on the
/// paths that keep what they read: the in-memory loaders and the dense
/// writer.
///
/// This is a slice height for the h5 layer, not a cell batch size: it bounds
/// how much of a dataset is materialised at once during ingest. Deliberately
/// far smaller than [`crate::store::CELL_BATCH_SIZE`], which sizes reads
/// against the bixverse binary format.
pub(crate) const H5_SLICE_SIZE: usize = 1_000;

/// Rows per read in the passes that only count: the QC scans and the dense
/// modality extract.
///
/// They hold one slice of a sparse dataset (or of a dense one, `n_genes` wide)
/// plus small per-row tallies, so the slice can be ten times
/// [`H5_SLICE_SIZE`] and pay the per-read overhead of HDF5 and Zarr less
/// often. Not tuned.
pub(crate) const H5_SCAN_CHUNK: usize = 10_000;

/// Genes per read on the CSC passes that keep per-cell state for the whole
/// file while they go: the CSC to CSR writers and the multi-file CSC scan.
///
/// Between the other two: a gene slice spans every cell, so it is wider than
/// a cell slice of the same height. Not tuned.
pub(crate) const H5_GENE_CHUNK: usize = 5_000;
