//! Core numerics for the bixverse crates: sparse structures, linear algebra,
//! correlations, statistics, SIMD helpers, clustering and Gaussian processes.
//! Everything else in the workspace builds on this crate.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

/// Version of this crate.
///
/// Exposed so a dependent can report which version of the numerics it was
/// built against.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod core;
pub mod errors;
pub mod ml;
pub mod prelude;
pub mod utils;
