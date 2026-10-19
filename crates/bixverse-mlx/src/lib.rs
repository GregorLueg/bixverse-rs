//! Experimental MLX backend (Apple Silicon) for the GEMM-heavy kernels, via
//! mlx-c. f32 only: MLX has no f64 on the GPU.

#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod errors;
#[cfg(mlx_available)]
mod ffi;
#[cfg(mlx_available)]
pub mod linalg;
#[cfg(mlx_available)]
pub mod methods;
pub mod prelude;
#[cfg(mlx_available)]
pub mod sc;

/// Whether the MLX backend was compiled in: the build script found
/// `libmlxc`.
pub const MLX_AVAILABLE: bool = cfg!(mlx_available);
