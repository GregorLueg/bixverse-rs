//! The single cell data layer: the bixverse binary sparse format, its
//! streaming readers and writers, and ingestion from h5ad, 10x h5 (feature
//! `h5`) and MTX.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

/// Version of this crate.
///
/// Exposed so a dependent can report which version of the numerics it was
/// built against. The Python bindings version independently and vendor this
/// crate through a path dependency, so their own version number says nothing
/// about what is inside the wheel; this does.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod archive_io;
pub mod bin_merge_io;
pub mod data_io;
pub mod errors;
pub mod gene_file_io;
#[cfg(feature = "h5")]
pub mod h5_10x_adt_io;
#[cfg(feature = "h5")]
pub mod h5_10x_io;
#[cfg(feature = "h5")]
pub mod h5_10x_multifile_io;
#[cfg(feature = "h5")]
pub mod h5_filters;
#[cfg(feature = "h5")]
pub mod h5ad_io;
#[cfg(feature = "h5")]
pub mod h5ad_multifile_io;
pub mod in_memory_io;
pub mod mtx_io;
pub mod mtx_multifile_io;
pub mod plotting;
pub mod prelude;
pub mod r_obj_io;
pub mod sc_synthetic_data;
pub mod sc_traits;

/// How many cells to read in per given batch - global constant
pub const CELL_BATCH_SIZE: usize = 50_000;

/// Version of the single cell files
pub const SC_FILE_VERSION: u32 = 3;

/// Number of cell rows requested per HDF5 hyperslab read.
///
/// This is a slice height for the h5 layer, not a cell batch size: it bounds
/// how much of a dataset is materialised at once during ingest. Deliberately
/// far smaller than [`CELL_BATCH_SIZE`], which sizes reads against the
/// bixverse binary format.
#[cfg(feature = "h5")]
pub(crate) const H5_CELL_SLICE_SIZE: usize = 1_000;
