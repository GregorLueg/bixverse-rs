//! Errors raised by `bixverse-gpu`.

use thiserror::Error;

/// Error variants raised by `bixverse-gpu`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum GpuErrors {
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

    // -- gpu --
    /// A device-limit or runtime error from the `cubecl-utils-rs` primitives.
    ///
    /// Covers the cube-count, grid, binding-size and shared-memory guards, plus
    /// buffer reads that fail on the CubeCL server.
    #[error("GPU: {0}")]
    CubeclUtils(#[from] cubecl_utils_rs::CubeclUtilsErrors),
    /// A GPU cubecl matrix multiplication error from the cubek crate
    #[error("GPU: A matrix multiplication occurred: {0}")]
    GpuMatmul(String),
    /// A single named buffer exceeds the device's per-binding size limit.
    ///
    /// Over-sized bindings are rejected without an error surfacing: the kernel
    /// does no work and returns zeros, so the condition is caught on the host
    /// before the upload instead.
    ///
    /// `GpuTensor`'s own constructors already guard their allocations and
    /// report `CubeclUtils`. This variant is for the host-side pre-checks that
    /// walk a whole named buffer set before a dispatch, where the buffer name
    /// is worth more than a bare byte count.
    #[error(
        "GPU: buffer '{buffer}' needs {bytes} bytes, device per-binding limit is {limit} bytes"
    )]
    GpuBindingTooLarge {
        /// Name of the buffer that does not fit.
        buffer: &'static str,
        /// Bytes the buffer requires.
        bytes: usize,
        /// Per-binding limit reported by the device.
        limit: usize,
    },
    /// The requested NMF rank is above the largest comptime register tier the
    /// GPU HALS sweeps are compiled for.
    ///
    /// The sweeps hold a whole row of `W` or column of `H` in a register array
    /// whose length must be known at compilation, so the dispatch picks the
    /// smallest tier that fits `k`. Past the largest tier the array would spill
    /// to global memory and the kernel would be slower than the CPU path, so
    /// this errors rather than silently degrading.
    #[error(
        "NMF GPU: rank k = {k} is above the largest supported rank of {max}; use the CPU entry point for ranks this large"
    )]
    GpuNmfRankTooLarge {
        /// Requested number of components.
        k: usize,
        /// Largest rank the compiled tiers cover.
        max: usize,
    },
}
