//! This crate contains highly optimised code for computational biology. It
//! has various statistical and computational biology methods, written with
//! performance in mind. Additionally, it now also has a first take at a single
//! cell suite of methods and functions.

#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod core;
pub mod enrichment;
pub mod graph;
pub mod methods;
pub mod ontology;
pub mod prelude;
pub mod utils;

#[cfg(feature = "single-cell")]
pub mod single_cell;
