//! Gaussian process regression.
//!
//! One model so far: a landmark (subset-of-regressors) posterior mean over 1-D
//! inputs with many response columns. It backs the Palantir gene trends, but it
//! knows nothing about single cells and is not feature-gated.

pub mod kernels;
pub mod landmark;

pub use kernels::{fill_matern52_cross_1d, matern52};
pub use landmark::{LandmarkGpFit, LandmarkGpParams, fit_landmark_gp};
