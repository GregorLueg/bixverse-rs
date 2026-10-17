//! Parity between the CPU and MLX coarseners for fast clustering.
//!
//! Both k-means start from the same host initialisation at the same seed, so
//! the inertia should agree closely; labels are compared by ARI because
//! near-ties in the GEMM assignment can move single cells.

#![cfg(mlx_available)]

use faer::Mat;
use rand::prelude::*;

use bixverse_core::ml::clustering::clustering_metrics::adjusted_rand_index;
use bixverse_core::ml::clustering::k_means::KMeansParamsWrappers;
use bixverse_mlx::sc::fast_clusters::MlxKMeans;
use bixverse_sc::sc_analysis::fast_clusters::*;

/// Label agreement between the MLX and CPU pipelines. Provisional: the
/// proposal leaves the bar to be set.
const MIN_ARI: f64 = 0.9;

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
/// `(data, blob id per point)`
fn blobs(n: usize, dim: usize, n_blobs: usize, seed: u64) -> (Mat<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres: Vec<f32> = (0..n_blobs * dim)
        .map(|_| rng.random_range(-10.0..10.0))
        .collect();
    let truth: Vec<usize> = (0..n).map(|i| i % n_blobs).collect();
    let data = Mat::from_fn(n, dim, |i, j| {
        centres[truth[i] * dim + j] + rng.random_range(-1.0..1.0)
    });
    (data, truth)
}

/// Sum of squared distances to the assigned centroid.
///
/// ### Params
///
/// * `data` - Points
/// * `centroids` - Centroids
/// * `assign` - Centroid per point
///
/// ### Returns
///
/// The inertia
fn inertia(data: &Mat<f32>, centroids: &Mat<f32>, assign: &[usize]) -> f64 {
    (0..data.nrows())
        .map(|i| {
            (0..data.ncols())
                .map(|j| (data[(i, j)] - centroids[(assign[i], j)]) as f64)
                .map(|d| d * d)
                .sum::<f64>()
        })
        .sum()
}

/// MLX and CPU k-means agree on the inertia, and the fast-clustering labels
/// agree by ARI, on the kNN and the cell-mediated centroid graphs.
#[test]
fn test_fast_clusters_mlx_matches_cpu() {
    let (data, _) = blobs(6_000, 16, 8, 3);
    let cpu = CpuKMeans::new(CpuKMeansKind::Standard, KMeansParamsWrappers::default());
    let mlx = MlxKMeans::default();

    for graph in [
        CentroidGraph::default(),
        CentroidGraph::CellMediated { n_anchors: 5 },
    ] {
        let params = FastClusterParams {
            n_centroids: 100,
            graph,
            ..FastClusterParams::default()
        };
        let a = fast_clusters(data.as_ref(), &cpu, &params, &[1.0], 1, 11, 0).unwrap();
        let b = fast_clusters(data.as_ref(), &mlx, &params, &[1.0], 1, 11, 0).unwrap();

        let (ia, ib) = (
            inertia(&data, &a.centroids, &a.assignments),
            inertia(&data, &b.centroids, &b.assignments),
        );
        assert!(
            (ia - ib).abs() / ia < 0.02,
            "{graph:?}: cpu {ia} vs mlx {ib}"
        );

        let ari = adjusted_rand_index(&a.per_resolution[0].labels, &b.per_resolution[0].labels);
        assert!(ari > MIN_ARI, "{graph:?}: ARI {ari}");
    }
}

/// The MLX top-m returns the same centroid sets as the CPU scan, up to f32
/// ties at the m-th boundary.
#[test]
fn test_mlx_top_m_matches_cpu() {
    let (data, _) = blobs(2_000, 8, 5, 1);
    let centroids = Mat::from_fn(40, 8, |i, j| data[(i * 50, j)] + 0.01);
    let mlx = MlxKMeans::default()
        .top_m(data.as_ref(), centroids.as_ref(), 5, "euclidean")
        .unwrap();
    let cpu = top_m_cpu(data.as_ref(), centroids.as_ref(), 5, "euclidean");
    let mismatch = mlx
        .chunks_exact(5)
        .zip(cpu.chunks_exact(5))
        .filter(|(a, b)| {
            let (mut a, mut b) = (a.to_vec(), b.to_vec());
            a.sort_unstable();
            b.sort_unstable();
            a != b
        })
        .count();
    assert!(mismatch <= 5, "{mismatch} rows differ");
}
