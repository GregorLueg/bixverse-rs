#![allow(clippy::needless_range_loop)]

pub mod core;
pub mod enrichment;
pub mod prelude;
pub mod utils;

#[cfg(feature = "single-cell")]
pub mod single_cell;
