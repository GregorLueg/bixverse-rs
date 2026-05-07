//! Implementation of the MELD method from Burkhardt, et al., Nat. Biotechnol.,
//! 2021. For large scale data sets, the landmark method is available. Estimates
//! per-condition cell densities by spectrally low-pass filtering one-hot sample
//! indicator vectors over a cell-cell kNN graph using a Chebyshev polynomial
//! approximation of the filter.

use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;
use thousands::Separable;

use crate::core::math::sparse::*;
use crate::core::math::sparse::*;
use crate::prelude::*;
use crate::single_cell::mc_generation::seacells::{
    build_data_to_landmark_transitions, compute_diffusion_kernel, landmark_knn,
    select_density_landmarks,
};
