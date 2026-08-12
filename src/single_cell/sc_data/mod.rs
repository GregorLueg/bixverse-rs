//! This module contains all of the single cell streamining-related
//! functionalities, structures and readers.

pub mod bin_merge_io;
pub mod data_io;
pub mod depracated_conversion;
pub mod h5_10x_io;
pub mod h5_10x_multifile_io;
pub mod h5ad_io;
pub mod h5ad_multifile_io;
pub mod in_memory_io;
pub mod mtx_io;
pub mod mtx_multifile_io;
pub mod plotting;
pub mod r_obj_io;
pub mod sc_synthetic_data;

/// Number of cell rows requested per HDF5 hyperslab read.
///
/// This is a slice height for the h5 layer, not a cell batch size: it bounds
/// how much of a dataset is materialised at once during ingest. Deliberately
/// far smaller than [`crate::prelude::CELL_BATCH_SIZE`], which sizes reads
/// against the bixverse binary format.
pub(crate) const H5_CELL_SLICE_SIZE: usize = 1_000;
