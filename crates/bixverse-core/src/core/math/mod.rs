//! This module contains various mathematical functions and utilities.

pub mod distributions;
pub mod linear_algebra;
pub mod matrix_helpers;
pub mod mixed_model;
pub mod optimise;
pub mod pca_missing;
pub mod pca_svd;
pub mod rbf;
pub mod sparse;
pub mod special;
pub mod stats;
pub mod vector_helpers;

////////////
// Consts //
////////////

/// The default oversampling for randomised SVD
pub const DEFAULT_OVERSAMPLING_RAND_SVD: usize = 10;

/// Oversampling for randomised SVD on single-cell data.
///
/// Paired with [N_POWER_ITERS_SINGLE_CELL]. Single-cell embeddings have a
/// slowly decaying spectrum below the first few components, so the sketch
/// has to separate components whose singular values are close.
///
/// Clamp it against the rank of the input before use: on a small matrix
/// `no_pcs + 20` can exceed `min(rows, cols)`.
pub const MAX_OVERSAMPLING_SINGLE_CELL: usize = 20;

/// Power iterations for randomised SVD on single-cell data.
pub const N_POWER_ITERS_SINGLE_CELL: usize = 4;

/// The default power iterations for randomised SVD
pub const DEFAULT_N_POWER_ITERS_RAND_SVD: usize = 2;

/// MAD scaling constant. R's `mad()` applies this factor by default.
pub const MAD_SCALE: f64 = 1.482_602_218_505_602;
