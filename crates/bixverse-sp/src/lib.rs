//! Spatial transcriptomics on top of `bixverse-sc`: spatial graphs, spatially
//! variable genes (Moran's I, SPARK-X), neighbourhood enrichment, the h5ad
//! spatial reader and per-spot histology image features.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod errors;
pub mod prelude;
pub mod sp_analysis;
pub mod sp_data;
pub mod sp_graph;
#[cfg(feature = "spatial-image")]
pub mod sp_image;
pub mod sp_processing;
pub mod sp_validate;
