//! A fast approach to generate clustering based on k-means clustering first on
//! the data, followed by kNN search on the centroids, (for now only) Louvain
//! clustering on the resulting kNN graph with subsequent propagation of the
//! module membership based on the original nearest centroid.

use ann_search_rs::prelude::*;
use ann_search_rs::utils::dist::parse_ann_dist;
use ann_search_rs::utils::k_means_utils::*;
use ann_search_rs::utils::matrix_to_flat;
use faer::{Mat, MatRef};

use crate::graph::community_detections::*;
use crate::graph::graph_structures::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::utils_doublets::dispatch_knn;

/////////////
// k-means //
/////////////

/// Generate k-means clusters for subsequent kNN search
///
/// ### Params
///
/// * `data` - The data for which to run the k-means clustering
/// * `dist` - The distance metric to use. One of `"euclidean"` or `"cosine"`.
///   Weird strings will default to `"euclidean"` (under the hood squared
///   euclidean)
/// * `n_centroids` - The number of centroids to identify
/// * `n_iters` - Number of iterations for the k-means clustering
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls the verbosity of the function
///
/// ### Returns
///
/// A tuple of `(centroid matrix -> shape n_centoids x dim, assignments of
/// points to cells)`
fn k_means_clusters<T>(
    data: MatRef<T>,
    dist: &str,
    n_centroids: usize,
    n_iters: usize,
    seed: usize,
    verbose: bool,
) -> (Mat<T>, Vec<usize>)
where
    T: BixverseFloat + AnnSearchFloat,
{
    let (data_flat, n, dim) = matrix_to_flat(data);
    let dist = parse_ann_dist(dist).unwrap_or_default();

    let norms = if dist == Dist::Cosine {
        (0..n)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                T::calculate_l2_norm(&data_flat[start..end])
            })
            .collect()
    } else {
        Vec::new()
    };

    let centroids = train_centroids(
        &data_flat,
        dim,
        n,
        n_centroids,
        &dist,
        n_iters,
        seed,
        verbose,
    );

    let centroids_norm = if dist == Dist::Cosine {
        (0..n_centroids)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                T::calculate_l2_norm(&centroids[start..end])
            })
            .collect()
    } else {
        Vec::new()
    };

    let assignments = assign_all_parallel(
        &data_flat,
        &norms,
        dim,
        n,
        &centroids,
        &centroids_norm,
        n_centroids,
        &dist,
    );

    let centroid_mat = Mat::from_fn(n_centroids, dim, |i, j| centroids[i + j * n_centroids]);

    (centroid_mat, assignments)
}

//////////////////
// Fast Louvain //
//////////////////

/// Parameters for fast Louvain clustering via k-means + kNN.
#[derive(Clone, Debug)]
pub struct FastLouvainParams {
    /// Number of k-means centroids.
    pub n_centroids: usize,
    /// Number of k-means iterations.
    pub kmeans_iters: usize,
    /// kNN search parameters applied to the centroids. `ann_dist` also drives
    /// the k-means distance and `k` is the number of neighbours per centroid.
    pub knn_params: KnnParams,
    /// Louvain resolution.
    pub resolution: f32,
    /// Number of Louvain iterations.
    pub louvain_iters: usize,
}

impl FastLouvainParams {
    /// Generate a version of FastLouvainParams with sensible base parameters
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            n_centroids: 1000,
            kmeans_iters: 50,
            knn_params: KnnParams::default(),
            resolution: 1.0,
            louvain_iters: 10,
        }
    }
}

/// Default implementation for KnnParams
impl Default for FastLouvainParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast Louvain clustering on large data via k-means coarsening.
///
/// Runs k-means to obtain centroids, builds a kNN graph on the centroids,
/// applies Louvain to that graph, then propagates each centroid's community
/// label back to the points assigned to it.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix.
/// * `params` - Pipeline parameters.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// Per-sample community labels (length n_samples).
pub fn fast_louvain_clusters(
    data: MatRef<f32>,
    params: &FastLouvainParams,
    seed: usize,
    verbose: bool,
) -> Vec<usize> {
    let (centroids, assignments) = k_means_clusters(
        data,
        &params.knn_params.ann_dist,
        params.n_centroids,
        params.kmeans_iters,
        seed,
        verbose,
    );

    let knn = dispatch_knn(
        centroids.as_ref(),
        params.knn_params.k,
        &params.knn_params,
        seed,
        verbose,
    );

    let graph = knn_to_sparse_graph(&knn);
    let centroid_communities =
        louvain_sparse_graph(&graph, params.resolution, params.louvain_iters, seed);

    assignments
        .iter()
        .map(|&c| centroid_communities[c])
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use rand::prelude::*;

    fn make_two_blobs(n_per: usize, dim: usize, separation: f32, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut buf = vec![0.0f32; 2 * n_per * dim];
        for i in 0..(2 * n_per) {
            let centre = if i < n_per { 0.0 } else { separation };
            for j in 0..dim {
                buf[i * dim + j] = centre + rng.random_range(-0.3..0.3);
            }
        }
        Mat::from_fn(2 * n_per, dim, |i, j| buf[i * dim + j])
    }

    fn default_params() -> FastLouvainParams {
        let mut knn = KnnParams::new();
        knn.knn_method = "exhaustive".to_string();
        knn.ann_dist = "euclidean".to_string();
        knn.k = 10;
        FastLouvainParams {
            n_centroids: 10,
            kmeans_iters: 20,
            knn_params: knn,
            resolution: 0.5,
            louvain_iters: 2,
        }
    }

    #[test]
    fn output_length_matches_input() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let labels = fast_louvain_clusters(data.as_ref(), &default_params(), 0, false);
        assert_eq!(labels.len(), 200);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = default_params();
        let a = fast_louvain_clusters(data.as_ref(), &params, 7, false);
        let b = fast_louvain_clusters(data.as_ref(), &params, 7, false);
        assert_eq!(a, b);
    }

    #[test]
    fn produces_multiple_communities_on_separated_blobs() {
        let data = make_two_blobs(150, 8, 10.0, 42);
        let labels = fast_louvain_clusters(data.as_ref(), &default_params(), 0, false);

        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert!(
            distinct.len() >= 2,
            "expected >= 2 communities, got {}",
            distinct.len()
        );
        assert!(distinct.len() <= default_params().n_centroids);
    }
}
