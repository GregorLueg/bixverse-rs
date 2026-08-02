//! Bixverse-specific spatial transcriptomics methods.
//!
//! This extends the single cell analysis suite.

pub mod sp_analysis;
pub mod sp_data;
pub mod sp_graph;
#[cfg(feature = "spatial-image")]
pub mod sp_image;
pub mod sp_processing;
pub mod sp_r_wrappers;
pub mod sp_validate;
