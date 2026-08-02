//! Spatial-only ingest.
//!
//! Reads the parts of a file the single cell readers have no notion of:
//! `obsm/spatial` and the `uns/spatial` group. The counts path is untouched,
//! because `load_h5ad` already handles that and the single cell reader should
//! stay a single cell reader.
//!
//! ## Layout
//!
//! * `orientation.rs` decides which column of `obsm/spatial` is `x`. Pure
//!   functions over slices, no HDF5.
//! * `h5ad_spatial.rs` does the reading.
//!
//! ## Why the orientation gets its own file
//!
//! It is the one thing here that can be wrong without anything downstream
//! noticing. A transposed tissue builds a graph, runs Moran's I and returns
//! numbers that look fine.

pub mod h5ad_spatial;
pub mod orientation;

pub use h5ad_spatial::{SpatialH5adData, SpatialH5adParams, SpatialImageEntry, read_spatial_h5ad};
pub use orientation::{
    OrientationCall, OrientationEvidence, SpatialOrientation, orientation_from_image_frame,
    orientation_from_obs_pixels, orientation_from_tissue_mask, tissue_mask,
};
