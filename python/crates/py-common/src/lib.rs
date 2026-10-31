//! What the `bixverse` and `bixverse-gpu` extensions share.
//!
//! Error mapping, NumPy and parameter conversions, and the result dicts a CPU
//! method and its GPU twin both hand back. Compiled into each extension; the
//! two never exchange Rust objects at run time.

pub mod convert;
pub mod error;
pub mod results;
