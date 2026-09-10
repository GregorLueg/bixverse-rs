//! Various machine learning implementations that can be useful for
//! bioinformatics, such as clustering methods, etc.

pub mod clustering;
pub mod gp;
#[cfg(feature = "r")]
pub mod ml_r_wrappers;
