//! This module contains various mathematical functions and utilities.

pub mod distributions;
pub mod linear_algebra;
pub mod matrix_helpers;
pub mod mixed_model;
pub mod optimise;
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

/// Oversampling for randomised SVD on single-cell data.
///
/// Ten times [DEFAULT_OVERSAMPLING_RAND_SVD]. Single-cell embeddings have a
/// long flat noise floor below the first few components, and a narrow sketch
/// under-estimates every singular value in it. Every CPU and GPU single-cell
/// PCA path uses this, so the two stay comparable; a mismatch between them
/// shows up as a silent accuracy gap rather than an error.
///
/// Clamp it against the rank of the input before use: on a small matrix
/// `no_pcs + 100` can exceed `min(rows, cols)`.
pub const MAX_OVERSAMPLING_SINGLE_CELL: usize = 100;

/// The default power iterations for randomised SVD
pub const DEFAULT_N_POWER_ITERS_RAND_SVD: usize = 2;
