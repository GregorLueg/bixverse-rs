//! This module contains various mathematical functions and utilities.

pub mod linear_algebra;
pub mod matrix_helpers;
pub mod pca_svd;
pub mod rbf;
pub mod sparse;
pub mod stats;
pub mod vector_helpers;

////////////
// Consts //
////////////

/// The default oversampling for randomised SVD
pub const DEFAULT_OVERSAMPLING_RAND_SVD: usize = 10;

/// The default power iterations for randomised SVD
pub const DEFAULT_N_POWER_ITERS_RAND_SVD: usize = 2;
