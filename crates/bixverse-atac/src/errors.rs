//! Errors raised by `bixverse-atac`.

use thiserror::Error;

/// Error variants raised by `bixverse-atac`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum AtacErrors {
    // -- lower crates --
    /// Propagates an error from `bixverse-sc`.
    #[error(transparent)]
    Sc(#[from] bixverse_sc::errors::ScErrors),

    /// Propagates an error from `bixverse-sc-data`.
    #[error(transparent)]
    ScData(#[from] bixverse_sc_data::errors::ScDataErrors),

    /// I/O on a fragment, BED or GTF file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // -- input --
    /// A genomic interval string that does not parse.
    #[error("cannot parse the interval '{interval}' (expected chrom:start-end)")]
    IntervalParse {
        /// The offending string
        interval: String,
    },

    /// Cell barcodes that the input does not contain.
    #[error("{n_missing} of {n_requested} barcodes are not in the input, e.g. '{example}'")]
    BarcodesNotFound {
        /// Barcodes not found
        n_missing: usize,
        /// Barcodes requested
        n_requested: usize,
        /// One of the missing barcodes
        example: String,
    },

    /// Inputs that do not fit together.
    #[error("{0}")]
    InvalidArgument(String),

    // -- dimension reduction --
    /// LSI was asked for on a store that is not TF-IDF normalised.
    #[error("LSI needs a TF-IDF normalised store; run set_tf_idf_on_store first")]
    NotTfIdf,
}
