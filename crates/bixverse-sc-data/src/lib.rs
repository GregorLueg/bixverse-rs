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

pub mod errors;
#[cfg(feature = "h5")]
pub mod h5;
pub mod mtx;
#[cfg(feature = "parquet")]
pub mod obs_var;
#[cfg(feature = "analysis")]
pub mod plotting;
pub mod prelude;
pub mod r_obj;
pub mod sc_traits;
pub mod store;
#[cfg(feature = "analysis")]
pub mod synthetic;
