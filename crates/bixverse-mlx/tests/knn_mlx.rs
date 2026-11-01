//! MLX kNN dispatch against the CPU exhaustive reference.

#![cfg(mlx_available)]

use faer::Mat;
use rand::prelude::*;
use rustc_hash::FxHashSet;

use bixverse_mlx::sc::knn::{dispatch_knn_mlx, knn_query_mlx, knn_with_dist_mlx};
use bixverse_sc::sc_processing::knn::{KnnParams, generate_knn_with_dist};
use bixverse_sc::sc_processing::knn_device::KnnParamsGpu;

/// Gaussian-ish blobs, `n_blobs` centres in `dim` dimensions.
///
/// ### Params
///
/// * `n` - Points
/// * `dim` - Dimensions
/// * `n_blobs` - Number of centres
/// * `seed` - RNG seed
///
/// ### Returns
///
/// The data, points x dimensions.
fn blobs(n: usize, dim: usize, n_blobs: usize, seed: u64) -> Mat<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres: Vec<f32> = (0..n_blobs * dim)
        .map(|_| rng.random_range(-10.0..10.0))
        .collect();
    Mat::from_fn(n, dim, |i, j| {
        centres[(i % n_blobs) * dim + j] + rng.random_range(-1.0..1.0)
    })
}

/// Exact kNN on the CPU, self excluded.
///
/// ### Params
///
/// * `data` - Points x dimensions
/// * `k` - Neighbours per point
///
/// ### Returns
///
/// The neighbour rows.
fn cpu_exhaustive(data: &Mat<f32>, k: usize) -> Vec<Vec<usize>> {
    let params = KnnParams {
        knn_method: "exhaustive".to_string(),
        ann_dist: "euclidean".to_string(),
        k,
        ..Default::default()
    };
    generate_knn_with_dist(data.as_ref(), &params, false, false, 42, false)
        .expect("CPU exhaustive must succeed")
        .0
}

/// Share of the reference neighbours recovered.
///
/// ### Params
///
/// * `got` - Rows under test
/// * `want` - Reference rows
///
/// ### Returns
///
/// Recall in `[0, 1]`.
fn recall(got: &[Vec<usize>], want: &[Vec<usize>]) -> f32 {
    let (hit, total) = got
        .iter()
        .zip(want)
        .fold((0usize, 0usize), |(hit, total), (g, w)| {
            let w: FxHashSet<usize> = w.iter().copied().collect();
            (
                hit + g.iter().filter(|x| w.contains(x)).count(),
                total + w.len(),
            )
        });
    hit as f32 / total as f32
}

/// Every method fills every row to `k`, never returns self, and recovers the
/// exact neighbours. `"nndescent"` is graph-only, `"cagra"` beam searches.
#[test]
fn test_knn_mlx_methods_match_cpu_exhaustive() {
    let (n, dim, k) = (2000usize, 24usize, 15usize);
    let data = blobs(n, dim, 10, 7);
    let want = cpu_exhaustive(&data, k);

    for (method, floor) in [
        ("exhaustive", 0.99f32),
        ("ivf", 0.90),
        ("cagra", 0.85),
        ("nndescent", 0.85),
    ] {
        let params = KnnParamsGpu {
            knn_method: method.to_string(),
            k,
            refine_knn: Some(1),
            ..Default::default()
        };
        let got = dispatch_knn_mlx(data.as_ref(), k, &params, 42, 0).unwrap();

        assert_eq!(got.len(), n, "{}: wrong row count", method);
        for (i, row) in got.iter().enumerate() {
            assert_eq!(row.len(), k, "{}: row {} holds {}", method, i, row.len());
            assert!(!row.contains(&i), "{}: row {} contains itself", method, i);
        }

        let r = recall(&got, &want);
        assert!(r > floor, "{}: recall {:.4} below {}", method, r, floor);
    }
}

/// The with-distances path returns true, ascending distances alongside the
/// rows, for the graph-only method too.
#[test]
fn test_knn_with_dist_mlx_graph_distances_ascending() {
    let (n, k) = (1000usize, 10usize);
    let data = blobs(n, 16, 8, 3);
    let params = KnnParamsGpu {
        knn_method: "nndescent".to_string(),
        k,
        ..Default::default()
    };

    let (idx, dist) = knn_with_dist_mlx(data.as_ref(), k, &params, 42, 0).unwrap();

    for (i, (row, d)) in idx.iter().zip(&dist).enumerate() {
        assert_eq!(row.len(), k, "row {}", i);
        assert_eq!(d.len(), k, "row {}", i);
        assert!(d.windows(2).all(|w| w[0] <= w[1]), "row {} not sorted", i);
        assert!(d.iter().all(|&x| x.is_finite() && x >= 0.0), "row {}", i);
    }
}

/// A cross-query on `"nndescent"` falls back to CAGRA and still finds the
/// query points in the reference they were copied from.
#[test]
fn test_knn_query_mlx_nndescent_falls_back_to_cagra() {
    let (n, k) = (1500usize, 10usize);
    let reference = blobs(n, 16, 6, 11);
    let query = Mat::from_fn(200, 16, |i, j| reference[(i * 7, j)]);
    let params = KnnParamsGpu {
        knn_method: "nndescent".to_string(),
        k,
        ..Default::default()
    };

    let (idx, _) = knn_query_mlx(
        reference.as_ref(),
        query.as_ref(),
        k,
        &params,
        42,
        false,
        false,
    )
    .unwrap();

    assert_eq!(idx.len(), 200);
    let found = idx
        .iter()
        .enumerate()
        .filter(|(i, row)| row.contains(&(i * 7)))
        .count();
    assert!(
        found >= 190,
        "only {} of 200 query points found themselves",
        found
    );
}
