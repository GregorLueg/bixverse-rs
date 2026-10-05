//! Large-n parity between the GPU and CPU column-pairwise correlation paths.
//!
//! The inline tests in `gpu/linalg/corr.rs` run at n = 80, d = 6, so this file
//! is the only coverage of the GPU accumulation at sizes where f32 drift could
//! actually show. Pearson, covariance and Spearman are each checked against
//! [`column_pairwise_cor`] / [`column_pairwise_cov`] on Gaussian data.

#![cfg(all(feature = "gpu", feature = "large-test"))]

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;
use rand::prelude::*;
use rand_distr::Normal;

use bixverse_rs::core::base::cors_similarity::{column_pairwise_cor, column_pairwise_cov};
use bixverse_rs::gpu::linalg::corr::{GpuCorCov, column_pairwise_cor_gpu};

///////////////
// Constants //
///////////////

/// Seed for every input matrix.
const SEED: u64 = 42;
/// Relative tolerance against the CPU maximum. f32 accumulation over up to
/// 2000 rows, so anything tighter trips on summation order alone.
const REL_TOL: f32 = 1e-3;

/////////////
// Helpers //
/////////////

/// Standard normal matrix.
///
/// ### Params
///
/// * `n_rows` - Number of rows (samples)
/// * `n_cols` - Number of columns (features)
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// An `n_rows x n_cols` matrix of N(0, 1) draws.
fn make_gaussian(n_rows: usize, n_cols: usize, seed: u64) -> Mat<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::<f32>::new(0.0, 1.0).expect("valid normal");
    Mat::from_fn(n_rows, n_cols, |_, _| normal.sample(&mut rng))
}

/// Runs one GPU correlation and asserts it against the CPU reference.
///
/// Checks the GPU output is not all zero first, since a launch that busts a
/// device limit fails silently and writes nothing.
///
/// ### Params
///
/// * `n` - Number of rows
/// * `d` - Number of columns, so the output is `d x d`
/// * `cor_type` - Which statistic to compute
/// * `label` - Name used in failure messages
fn assert_gpu_matches_cpu(n: usize, d: usize, cor_type: GpuCorCov, label: &str) {
    let data = make_gaussian(n, d, SEED);
    let device = WgpuDevice::DefaultDevice;

    let gpu = column_pairwise_cor_gpu::<f32, WgpuRuntime>(data.as_ref(), cor_type, device, false)
        .expect("GPU correlation failed");

    let cpu = match cor_type {
        GpuCorCov::Covariance => column_pairwise_cov(&data.as_ref()),
        GpuCorCov::Pearson => column_pairwise_cor(&data.as_ref(), false),
        GpuCorCov::Spearman => column_pairwise_cor(&data.as_ref(), true),
    };

    let mut max_diff = 0.0f32;
    let mut gpu_max_abs = 0.0f32;
    let mut cpu_max_abs = 0.0f32;
    for i in 0..d {
        for j in 0..d {
            gpu_max_abs = gpu_max_abs.max(gpu[(i, j)].abs());
            cpu_max_abs = cpu_max_abs.max(cpu[(i, j)].abs());
            max_diff = max_diff.max((gpu[(i, j)] - cpu[(i, j)]).abs());
        }
    }

    assert!(gpu_max_abs > 0.0, "{label} {n}x{d}: GPU output is all zero");
    assert!(
        max_diff <= REL_TOL * cpu_max_abs.max(1.0),
        "{label} {n}x{d}: GPU and CPU disagree by {max_diff:.3e} against a CPU \
         maximum of {cpu_max_abs:.3e}"
    );
}

///////////
// Tests //
///////////

/// Pearson at 500 to 2000 rows and columns, square and not.
#[test]
fn test_gpu_pearson_matches_cpu_sweep() {
    for &(n, d) in &[
        (500, 500),
        (1000, 500),
        (500, 1000),
        (1000, 1000),
        (2000, 1000),
        (1000, 2000),
        (2000, 2000),
    ] {
        assert_gpu_matches_cpu(n, d, GpuCorCov::Pearson, "pearson");
    }
}

/// Covariance at large n, where entries are not bounded by one.
#[test]
fn test_gpu_covariance_matches_cpu_sweep() {
    for &(n, d) in &[(500, 500), (1000, 1000), (2000, 2000)] {
        assert_gpu_matches_cpu(n, d, GpuCorCov::Covariance, "cov");
    }
}

/// Spearman at large n, ranking thousands of rows rather than tens.
#[test]
fn test_gpu_spearman_matches_cpu_sweep() {
    for &(n, d) in &[(500, 500), (1000, 1000), (2000, 2000)] {
        assert_gpu_matches_cpu(n, d, GpuCorCov::Spearman, "spearman");
    }
}
