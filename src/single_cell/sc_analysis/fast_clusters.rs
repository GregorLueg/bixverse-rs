//! A fast approach to generate clustering based on k-means clustering first on
//! the data, followed by kNN search on the centroids, (for now only) Louvain
//! clustering on the resulting kNN graph with subsequent propagation of the
//! module membership based on the original nearest centroid.

use faer::MatRef;

use crate::graph::community_detections::*;
use crate::graph::graph_structures::*;
use crate::ml::clustering::k_means::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::utils_doublets::dispatch_knn;

//////////////////
// Fast Louvain //
//////////////////

/// Parameters for fast Louvain clustering via k-means + kNN.
#[derive(Clone, Debug)]
pub struct FastLouvainParams<T> {
    // -- k means --
    /// Number of k-means centroids.
    pub n_centroids: usize,
    /// Number of k-means iterations.
    pub kmeans_iters: usize,
    /// Batch size for mini-batch k-means
    pub batch_size: usize,
    /// Drift threshold for mini batch k-means
    pub drift_threshold: T,
    /// Learning rate exponent for mini batch k-means:
    /// `eta = m / count[c]^lr_alpha`
    pub lr_alpha: T,

    // -- knn --
    /// kNN search parameters applied to the centroids. `ann_dist` also drives
    /// the k-means distance and `k` is the number of neighbours per centroid.
    pub knn_params: KnnParams,

    // -- louvain --
    /// Number of Louvain iterations.
    pub louvain_iters: usize,
}

impl<T> FastLouvainParams<T>
where
    T: BixverseFloat,
{
    /// Generate a version of FastLouvainParams with sensible base parameters
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            n_centroids: 1000,
            kmeans_iters: 100,
            batch_size: 4096,
            drift_threshold: T::from_f64(1e-4).unwrap(),
            lr_alpha: T::from_f64(1.0).unwrap(),
            knn_params: KnnParams::default(),
            louvain_iters: 10,
        }
    }
}

/// Default implementation for KnnParams
impl<T> Default for FastLouvainParams<T>
where
    T: BixverseFloat,
{
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
/// * `km_type` - String. Which type of k-means clustering to use. `"standard"`
///   or `"minibatch"`.
/// * `resolutions` - Slice of resolutions to iterate over.
/// * `params` - Pipeline parameters.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// Per-sample community labels (length n_samples).
pub fn fast_louvain_clusters(
    data: MatRef<f32>,
    km_type: &str,
    resolutions: &[f32],
    params: &FastLouvainParams<f32>,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<usize>> {
    let km_type = parse_k_means(km_type).unwrap_or_default();

    let (centroids, assignments) = match km_type {
        KMeansType::StandardKMeans => k_means_clusters(
            data,
            &params.knn_params.ann_dist,
            params.n_centroids,
            params.kmeans_iters,
            seed,
            verbose,
        ),
        KMeansType::MiniBatchKMeans => train_centroids_minibatch(
            data,
            &params.knn_params.ann_dist,
            params.n_centroids,
            params.kmeans_iters,
            params.batch_size,
            params.drift_threshold,
            params.lr_alpha,
            seed,
            verbose,
        ),
    };

    let knn = dispatch_knn(
        centroids.as_ref(),
        params.knn_params.k,
        &params.knn_params,
        seed,
        verbose,
    );

    let graph = knn_to_sparse_graph(&knn);

    let mut results: Vec<Vec<usize>> = Vec::with_capacity(resolutions.len());

    for &res in resolutions {
        let centroid_communities = louvain_sparse_graph(&graph, res, params.louvain_iters, seed);

        let membership = assignments
            .iter()
            .map(|&c| centroid_communities[c])
            .collect();

        results.push(membership)
    }

    results
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

    fn default_params() -> FastLouvainParams<f32> {
        let mut knn = KnnParams::new();
        knn.knn_method = "exhaustive".to_string();
        knn.ann_dist = "euclidean".to_string();
        knn.k = 10;
        FastLouvainParams {
            batch_size: 100,
            drift_threshold: 1e-4,
            lr_alpha: 1.0,
            n_centroids: 10,
            kmeans_iters: 20,
            knn_params: knn,
            louvain_iters: 2,
        }
    }

    // #[test]
    // fn output_length_matches_input() {
    //     let data = make_two_blobs(100, 5, 5.0, 42);
    //     let labels = fast_louvain_clusters(data.as_ref(), &default_params(), 0, false);
    //     assert_eq!(labels.len(), 200);
    // }

    // #[test]
    // fn deterministic_with_same_seed() {
    //     let data = make_two_blobs(100, 5, 5.0, 42);
    //     let params = default_params();
    //     let a = fast_louvain_clusters(data.as_ref(), &params, 7, false);
    //     let b = fast_louvain_clusters(data.as_ref(), &params, 7, false);
    //     assert_eq!(a, b);
    // }

    // #[test]
    // fn produces_multiple_communities_on_separated_blobs() {
    //     let data = make_two_blobs(150, 8, 10.0, 42);
    //     let labels = fast_louvain_clusters(data.as_ref(), &default_params(), 0, false);

    //     let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
    //     assert!(
    //         distinct.len() >= 2,
    //         "expected >= 2 communities, got {}",
    //         distinct.len()
    //     );
    //     assert!(distinct.len() <= default_params().n_centroids);
    // }
}
