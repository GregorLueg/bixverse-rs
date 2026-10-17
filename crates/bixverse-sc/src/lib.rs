//! Bixverse specific single cell and meta cell-specific 'omics module. Contains
//! all of the methods to run large scale single cell experiments locally while
//! reducing memory pressure.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod errors;
pub mod mc_analysis;
pub mod mc_generation;
pub mod mc_processing;
pub mod prelude;
pub mod sc_analysis;
pub mod sc_annotation;
pub mod sc_batch_correction;
pub mod sc_processing;
pub mod sc_trajectory;
pub mod sc_utils;

#[cfg(feature = "multi-modal")]
pub mod multi_modal;
