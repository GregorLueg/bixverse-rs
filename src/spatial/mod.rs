//! Spatial transcriptomics methods.
//!
//! All methods in this module operate per-sample: the R-side wrapper iterates
//! across samples and the Rust functions take a single sample's spots and
//! spatial graph at a time. Spot-to-gene-chunk mapping follows the same
//! convention as `single_cell::sc_analysis::hotspot`: the graph is indexed
//! `0..n_local` and a separate `spots_to_keep` slice carries the mapping back
//! to global gene-chunk indices.

pub mod nhood_enrichment;
pub mod svg;
