//! Contains GPU-accelerated methods via cubecl and burn.

pub mod gpu_r_wrappers;
pub mod linalg;
pub mod ml;
#[cfg(feature = "single-cell")]
pub mod sc_gpu;

////////////
// Consts //
////////////

/// Smaller work group version with 32
pub const WORKGROUP_32: u32 = 32;

/// Medium work group version with 32
pub const WORKGROUP_64: u32 = 64;

/// Larger work group version with 128
pub const WORKGROUP_128: u32 = 128;

/// Very large work group version with 512
pub const WORKGROUP_512: u32 = 512;
