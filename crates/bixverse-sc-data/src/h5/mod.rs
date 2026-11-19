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

/// Number of cell rows requested per HDF5 hyperslab read.
///
/// This is a slice height for the h5 layer, not a cell batch size: it bounds
/// how much of a dataset is materialised at once during ingest. Deliberately
/// far smaller than [`crate::store::CELL_BATCH_SIZE`], which sizes reads
/// against the bixverse binary format.
pub(crate) const H5_CELL_SLICE_SIZE: usize = 1_000;
