//! Runtime knobs for the GPU SCENIC path. Kept separate from the CPU-side
//! [`ScenicParams`](crate::single_cell::sc_analysis::scenic::ScenicParams) so
//! GPU-only tuning does not leak into the CPU config surface.

#![cfg(all(feature = "single-cell", feature = "gpu"))]

/// Parameters for the GPU multi-tree SCENIC driver.
pub struct ScenicGpuParams {
    /// VRAM ceiling (bytes) for the per-wave histogram + cumulative tensors.
    /// The wave scheduler halves the wave size from 8 until its byte cost
    /// fits under this budget; an error is returned only when a single-tree
    /// wave still busts it.
    ///
    /// Default: 4 GiB. Shrink on 8 GB adapters that host other workloads;
    /// raise on 16 GB+ adapters to keep the wave at 8.
    pub wave_byte_budget: usize,
}

/// Default implementation.
impl Default for ScenicGpuParams {
    fn default() -> Self {
        Self {
            wave_byte_budget: 4 * 1024 * 1024 * 1024,
        }
    }
}
