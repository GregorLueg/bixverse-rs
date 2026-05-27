//! Contains GPU-accelerated methods via cubecl and burn.

use half::{bf16, f16};
use num_traits::{Float, ToPrimitive};

pub mod gpu_r_wrappers;
pub mod ml;

/// Force float quantisation for `f16` or `bf16`
pub enum Quantisation {
    /// Brain floating point -> large range, very low precision
    BF16,
    /// F16 -> smaller range, but better precision compared to BF16
    F16,
}

pub fn quantise_data() {}
