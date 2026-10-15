//! GPU-accelerated linear algebra and bulk methods via cubecl and cubek.
//! The single cell kernels live in `bixverse-sc-gpu`.

#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod errors;
pub mod linalg;
pub mod methods_gpu;
pub mod prelude;

////////////
// Consts //
////////////

/// Smaller work group version with 32
pub const WORKGROUP_32: u32 = 32;

/// Medium work group version with 32
pub const WORKGROUP_64: u32 = 64;

/// Larger work group version with 128
pub const WORKGROUP_128: u32 = 128;

/// Even larger work group version with 256
pub const WORKGROUP_256: u32 = 256;

/// Very large work group version with 512
pub const WORKGROUP_512: u32 = 512;
