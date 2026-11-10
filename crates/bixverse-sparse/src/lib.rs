//! Compressed sparse structures for the bixverse crates: CSR and CSC storage,
//! transposition, format conversion and sparse arithmetic. Deliberately free
//! of dense linear algebra (`faer`), SIMD and nearest neighbour dependencies,
//! so readers of the single cell stores stay light.
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod errors;
pub mod prelude;
pub mod serialise;
pub mod sparse;
pub mod traits;

pub use sparse::thread_chunks;
