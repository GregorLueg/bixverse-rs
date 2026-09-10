//! This crate contains highly optimised code for computational biology. It
//! has various statistical and computational biology methods, written with
//! performance in mind. Additionally, it now also has a first take at a single
//! cell suite of methods and functions.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

/// Version of this crate.
///
/// Exposed so a dependent can report which version of the numerics it was
/// built against. The Python bindings version independently of this crate and
/// vendor its source through a path dependency, so their own version number
/// says nothing about what is inside the wheel; this does.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod core;
pub mod enrichment;
pub mod errors;
pub mod graph;
pub mod methods;
pub mod ml;
pub mod ontology;
pub mod prelude;
pub mod utils;

#[cfg(feature = "single-cell")]
pub mod single_cell;

#[cfg(feature = "gpu")]
pub mod gpu;
