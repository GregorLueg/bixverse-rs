#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod core;
pub mod data;
pub mod enrichment;
pub mod graph;
pub mod methods;
pub mod ontology;
pub mod prelude;
pub mod utils;

#[cfg(feature = "single-cell")]
pub mod single_cell;
