//! Contains GPU-accelerated methods via cubecl and burn.

pub mod gpu_r_wrappers;
pub mod linalg;
pub mod ml;
#[cfg(feature = "single-cell")]
pub mod sc_gpu;
