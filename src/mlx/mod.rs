//! Experimental MLX backend (Apple Silicon) for the GEMM-heavy kernels, via
//! mlx-c. f32 only: MLX has no f64 on the GPU.

mod ffi;
pub mod methods;
