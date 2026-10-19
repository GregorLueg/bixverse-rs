//! The R layer of bixverse: extendr conversions for the `bixverse-*` crates.
//!
//! Re-exports `bixverse-rs` in full, so an R package needs this one dependency.
//! Every `extendr-api` type in the workspace lives here, which keeps libR off
//! the link line of every other crate.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub use bixverse_rs::*;

pub mod core_r_wrappers;
pub mod enrichment_r_wrapper;
pub mod errors;
#[cfg(feature = "sc-gpu")]
pub mod gpu_r_wrappers;
pub mod graph_r_wrapper;
pub mod methods_r_wrapper;
pub mod ml_r_wrappers;
#[cfg(feature = "multi-modal")]
pub mod mm_r_wrappers;
pub mod ontology_r_wrappers;
pub mod prelude;
pub mod r_rust_interface;
#[cfg(feature = "single-cell")]
pub mod sc_r_wrappers;

/// Builds a parameter struct from an R list.
///
/// One trait for every params type in the workspace. The types live in other
/// crates, so an inherent `from_r_list` would break the orphan rule.
pub trait FromRList: Sized {
    /// Parses the R list into `Self`, falling back to `Self::default()` for
    /// absent fields where the type has defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list holding the parameters
    ///
    /// ### Returns
    ///
    /// The parsed structure, or an extendr error naming the bad field.
    fn from_r_list(r_list: extendr_api::List) -> extendr_api::Result<Self>;
}

/// Builds a parameter struct from an already flattened R list.
///
/// For sub-configs that are read out of the parameter list of whichever
/// method owns them, so the list is only flattened once.
pub trait FromRMap: Sized {
    /// Parses the flattened R list into `Self`.
    ///
    /// ### Params
    ///
    /// * `params` - The R list contents, already flattened to a HashMap
    ///
    /// ### Returns
    ///
    /// The parsed structure, or an extendr error naming the bad field.
    fn from_r_map(
        params: &std::collections::HashMap<&str, extendr_api::Robj>,
    ) -> extendr_api::Result<Self>;
}

/// Serialises a fitted structure into an R list that [`FromRList`] can read
/// back.
pub trait ToRList {
    /// Converts `self` into an R list.
    ///
    /// ### Returns
    ///
    /// The R list holding every field needed to rebuild `self`.
    fn to_r_list(&self) -> extendr_api::List;
}
