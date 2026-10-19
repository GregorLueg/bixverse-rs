//! Bulk omics methods on top of `bixverse-core`: gene set enrichment, graphs
//! and community detection, ontologies, matrix factorisation, co-expression
//! and (feature `dge`) the edgeR / limma differential expression chain.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod enrichment;
pub mod errors;
pub mod graph;
pub mod methods;
pub mod ontology;
pub mod prelude;
