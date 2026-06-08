#![allow(clippy::needless_range_loop)]
#![cfg(all(feature = "gpu", feature = "large_scale_diagnostics"))]

use bixverse_rs::core::base::cors_similarity::{column_pairwise_cor, column_pairwise_cov};
use bixverse_rs::gpu::linalg::corr::{GpuCorCov, column_pairwise_cor_gpu};

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;
use rand::prelude::*;
use rand_distr::Normal;

fn make_gaussian(n_rows: usize, n_cols: usize, seed: u64) -> Mat<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    Mat::from_fn(n_rows, n_cols, |_, _| normal.sample(&mut rng))
}

fn diagnose(n: usize, d: usize, cor_type: GpuCorCov, label: &str) {
    let data = make_gaussian(n, d, 42);
    let device = WgpuDevice::DefaultDevice;

    let gpu =
        column_pairwise_cor_gpu::<f32, WgpuRuntime, f32>(data.as_ref(), cor_type, device, false)
            .unwrap();

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
            let g = gpu[(i, j)].abs();
            let c = cpu[(i, j)].abs();
            if g > gpu_max_abs {
                gpu_max_abs = g;
            }
            if c > cpu_max_abs {
                cpu_max_abs = c;
            }
            let dd = (gpu[(i, j)] - cpu[(i, j)]).abs();
            if dd > max_diff {
                max_diff = dd;
            }
        }
    }
    let diag_mean: f32 = (0..d).map(|i| gpu[(i, i)]).sum::<f32>() / d as f32;

    eprintln!(
        "{label} {n}x{d}: max_diff={max_diff:.3e}  gpu_max_abs={gpu_max_abs:.3e}  cpu_max_abs={cpu_max_abs:.3e}  gpu_diag_mean={diag_mean:.4}"
    );

    assert!(gpu_max_abs > 0.0, "{label} {n}x{d}: GPU output is all zero",);
}

#[test]
fn diag_pearson_sweep() {
    for &(n, d) in &[
        (500, 500),
        (1000, 500),
        (500, 1000),
        (1000, 1000),
        (2000, 1000),
        (1000, 2000),
        (2000, 2000),
    ] {
        diagnose(n, d, GpuCorCov::Pearson, "pearson");
    }
}

#[test]
fn diag_covariance_sweep() {
    for &(n, d) in &[(500, 500), (1000, 1000), (2000, 2000)] {
        diagnose(n, d, GpuCorCov::Covariance, "cov");
    }
}

#[test]
fn diag_spearman_sweep() {
    for &(n, d) in &[(500, 500), (1000, 1000), (2000, 2000)] {
        diagnose(n, d, GpuCorCov::Spearman, "spearman");
    }
}
