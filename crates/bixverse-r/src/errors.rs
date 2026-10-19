//! Errors raised by `bixverse-r`.

use thiserror::Error;

/// Error variants raised by `bixverse-r`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum RErrors {
    // -- extendr --
    /// List to HashMap parsing error occured
    #[error("R list parsing failed: {0}")]
    RListParse(&'static str),
}
