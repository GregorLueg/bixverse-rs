//! Errors raised by `bixverse-sc-gpu`.

use thiserror::Error;

/// Error variants raised by `bixverse-sc-gpu`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum ScGpuErrors {
    // -- external crates --
    /// Propagates an error from `ann-search-rs`.
    #[error("Error from the ann-search-rs crate: {0}")]
    AnnSearchRsError(#[from] ann_search_rs::errors::AnnSearchErrors),

    /// Propagates an error from `edge-rs`.
    #[error("Error from the edge-rs crate: {0}")]
    EdgeRsError(#[from] edge_rs::errors::EdgeErrors),

    /// Propagates an error from `cubecl-utils-rs`.
    #[error("GPU: {0}")]
    CubeclUtils(#[from] cubecl_utils_rs::CubeclUtilsErrors),

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

    /// Propagates an error from `bixverse-gpu`.
    #[error(transparent)]
    Gpu(#[from] bixverse_gpu::errors::GpuErrors),

    // -- gpu / single cell --
    /// GPU Harmony only supports one co-variate for now (based on Arrowhead)
    #[error("Harmony GPU: Only a single co-variate is supported for the GPU path")]
    GpuHarmonySupportsSingleCovariateOnly,
    /// Selected SCENIC regression learner has no GPU implementation. Only the
    /// tree-based learners (ExtraTrees, RandomForest) are ported; GRNBoost2 /
    /// gradient boosting stays on CPU by design.
    #[error(
        "SCENIC GPU: the {learner} regression learner has no GPU path; use the CPU entry point (run_scenic_grn / run_scenic_grn_streaming / run_scenic_grn_in_memory)"
    )]
    GpuNotSupportedForLearner {
        /// Name of the requested regression learner.
        learner: &'static str,
    },
    /// The device cannot give `build_score_rf_fused` the threadgroup memory it
    /// asks for, so SCENIC's RandomForest learner has no GPU path on it.
    ///
    /// The kernel's ask is a compile-time constant checked against the tightest
    /// shipping backend floor, so reaching this means a device well outside
    /// what the crate targets rather than an unlucky shape.
    #[error(
        "SCENIC GPU: the fused RandomForest kernel needs {bytes} bytes of shared memory but this device offers {limit}; use the CPU entry point (run_scenic_grn / run_scenic_grn_streaming / run_scenic_grn_in_memory) or switch the learner to ExtraTrees"
    )]
    GpuScenicFusedRfUnavailable {
        /// Shared-memory bytes the fused kernel allocates per workgroup.
        bytes: usize,
        /// Shared-memory bytes the device reports.
        limit: usize,
    },
}
