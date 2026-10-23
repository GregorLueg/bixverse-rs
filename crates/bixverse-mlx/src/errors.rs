//! Errors raised by `bixverse-mlx`.

use thiserror::Error;

/// Error variants raised by `bixverse-mlx`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum MlxErrors {
    // -- external crates --
    /// Propagates an error from `ann-search-rs`.
    #[error("Error from the ann-search-rs crate: {0}")]
    AnnSearchRsError(#[from] ann_search_rs::errors::AnnSearchErrors),

    // -- lower crates --
    /// Propagates an error from `bixverse-core`.
    #[error(transparent)]
    Core(#[from] bixverse_core::errors::CoreErrors),

    /// Propagates an error from `bixverse-methods`.
    #[error(transparent)]
    Methods(#[from] bixverse_methods::errors::MethodsErrors),

    /// Propagates an error from `bixverse-sc-data`.
    #[error(transparent)]
    ScData(#[from] bixverse_sc_data::errors::ScDataErrors),

    /// Propagates an error from `bixverse-sc`.
    #[error(transparent)]
    Sc(#[from] bixverse_sc::errors::ScErrors),

    // -- mlx --
    /// An mlx-c call returned a non-zero code. mlx-c prints the reason to
    /// stderr through the handler installed by `install_error_handler`.
    #[error("MLX: {op} failed with code {code}")]
    MlxError {
        /// Name of the failing mlx-c call
        op: &'static str,
        /// Return code (or the offending dtype for a dtype mismatch)
        code: i32,
    },
    /// The requested NMF rank is outside what the MLX HALS sweep stages in
    /// threadgroup memory.
    #[error(
        "NMF MLX: rank k = {k} is outside the supported range 1..={max}; use the CPU entry point for ranks this large"
    )]
    MlxNmfRankTooLarge {
        /// Requested number of components.
        k: usize,
        /// Largest rank the sweep tiers cover.
        max: usize,
    },
    /// MLX Harmony only supports one co-variate (arrowhead ridge solve), as
    /// the cubecl path.
    #[error("Harmony MLX: Only a single co-variate is supported for the MLX path")]
    MlxHarmonySupportsSingleCovariateOnly,
}

/// Sparse structure errors surface through the core variant, as before the
/// sparse structures moved to `bixverse-sparse`.
impl From<bixverse_core::errors::SparseErrors> for MlxErrors {
    fn from(err: bixverse_core::errors::SparseErrors) -> Self {
        MlxErrors::Core(err.into())
    }
}
