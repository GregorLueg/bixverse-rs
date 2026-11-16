//! Errors of the export crate.

use bixverse_sc_data::errors::ScDataErrors;
use thiserror::Error;

/// Errors raised while exporting a store.
#[derive(Debug, Error)]
pub enum ExportErrors {
    /// Reading the store failed.
    #[error(transparent)]
    ScData(#[from] ScDataErrors),
    /// The scx-core writer failed.
    #[error("scx: {0}")]
    Scx(#[from] scx_core::error::ScxError),
    /// Inputs that do not fit together.
    #[error("{0}")]
    InvalidArgument(String),
}
