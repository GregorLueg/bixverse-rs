//! A fast approach to generate clustering based on k-means clustering first on
//! the data, followed by kNN search on the centroids, (for now only) Louvain
//! clustering on the resulting kNN graph with subsequent propagation of the
//! module membership based on the original nearest centroid.

use faer::MatRef;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use crate::core::math::vector_helpers::{mean_nan, median};
use crate::graph::community_detections::*;
use crate::graph::graph_metrics::conductance_undirected;
use crate::graph::graph_structures::*;
use crate::ml::clustering::clustering_metrics::adjusted_rand_index;
use crate::ml::clustering::k_means::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::snn::*;
use crate::single_cell::sc_processing::utils_doublets::dispatch_knn;

//////////////////
// Fast Louvain //
//////////////////

/////////////
// Helpers //
/////////////

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
    /// [KnnParams] parameters applied to the centroids. `ann_dist` also drives
    /// the k-means distance and `k` is the number of neighbours per centroid.
    pub knn_params: KnnParams,

    // -- snn ---
    /// Shall the full sNN graph be generated (also from non-connected
    /// neighbours)
    pub full_snn: bool,
    /// Optional pruning threshold. If not provided, will default to
    /// `1 / ceil(0.8 * k)`.
    /// neighbours.
    pub pruning: Option<T>,
    /// String for the type of sNN similarity to use. One of `"jaccard"` or
    /// `"rank"`.
    pub snn_similarity: String,

    // -- louvain --
    /// Number of Louvain iterations.
    pub louvain_iters: usize,
}

/// Per-resolution stability and quality metrics from the grid search.
#[derive(Clone, Debug)]
pub struct FastLouvainGridResult {
    /// Resolution this entry corresponds to.
    pub resolution: f32,
    /// Mean pairwise ARI across all seed pairs.
    pub mean_ari: f32,
    /// Median pairwise ARI across all seed pairs.
    pub median_ari: f32,
    /// Mean conductance across communities, averaged across seeds.
    /// NaN community values are skipped per seed.
    pub mean_conductance: f32,
    /// Median conductance across communities, averaged across seeds.
    pub median_conductance: f32,
    /// Number of communities, averaged across seeds.
    pub mean_n_communities: f32,
    /// Labels from the seed with the lowest mean conductance.
    pub best_labels: Vec<usize>,
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
            // kmeans
            n_centroids: 1000,
            kmeans_iters: 100,
            batch_size: 4096,
            drift_threshold: T::from_f64(1e-4).unwrap(),
            lr_alpha: T::from_f64(1.0).unwrap(),
            // snn
            full_snn: false,
            pruning: None,
            snn_similarity: "jaccard".to_string(),
            // knn - louvain
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

/// Flatten a `Vec<Vec<T>>` of per-sample neighbour lists into column-major
/// layout: `flat[neighbour_idx * n_samples + i]`. This matches the layout R
/// matrices use (and what `generate_snn_*` expect).
///
/// ### Params
///
/// * `knn` - The row major kNN data
/// * `k` - The number of neighbours
///
/// ### Returns
///
/// Column major version for sNN
fn flatten_knn_column_major<T: Copy>(knn: &[Vec<T>], k: usize) -> Vec<T> {
    let n = knn.len();
    let mut out = Vec::with_capacity(n * k);
    // SAFETY: we fill exactly n * k entries below, in any order
    out.resize(n * k, knn[0][0]);
    for (i, row) in knn.iter().enumerate() {
        debug_assert_eq!(row.len(), k);
        for (j, &v) in row.iter().enumerate() {
            out[j * n + i] = v;
        }
    }
    out
}

////////////////////
// Main functions //
////////////////////

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
/// * `run_snn` - Shall a shared nearest neighbour generation be run.
/// * `same_weight` - Shall the weights of the kNN graph be set to 1.0 or shall
///   reverse edges get higher weights.
/// * `multi_level` - If `true`, run the full multi-level Louvain (phase 1 +
///   phase 2 aggregation, repeated). If `false`, run only one phase 1 pass
///   (matches Phenograph / the original doubletdetection behaviour).
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// Per-sample community labels (length n_samples).
#[allow(clippy::too_many_arguments)]
pub fn fast_louvain_clusters(
    data: MatRef<f32>,
    km_type: &str,
    resolutions: &[f32],
    params: &FastLouvainParams<f32>,
    run_snn: bool,
    same_weight: bool,
    multi_level_louvain: bool,
    seed: usize,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let km_type = parse_k_means(km_type).unwrap_or_default();
    let n_centroids = params.n_centroids.min(data.nrows() - 1);

    let (centroids, assignments) = match km_type {
        KMeansType::StandardKMeans => k_means_clusters(
            data,
            &params.knn_params.ann_dist,
            n_centroids,
            params.kmeans_iters,
            seed,
            verbose,
        ),
        KMeansType::MiniBatchKMeans => train_centroids_minibatch(
            data,
            &params.knn_params.ann_dist,
            n_centroids,
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

    let graph = if run_snn {
        let pruning = params
            .pruning
            .unwrap_or(1.0 / (params.knn_params.k as f32 * 0.8).ceil());

        let start = Instant::now();

        if verbose {
            println!(" Running the sNN graph as {}...", &params.snn_similarity)
        }

        let knn_flat = flatten_knn_column_major(&knn, params.knn_params.k);

        let snn_method = parse_snn_similiarity_method(&params.snn_similarity).unwrap_or_default();

        let (snn_edges, snn_weights) = if params.full_snn {
            generate_snn_full(
                &knn_flat,
                params.knn_params.k,
                knn.len(),
                pruning,
                snn_method,
                verbose,
            )
        } else {
            generate_snn_limited(
                &knn_flat,
                params.knn_params.k,
                knn.len(),
                pruning,
                snn_method,
                verbose,
            )
        };

        let end = start.elapsed();

        if verbose {
            println!(" Finished the sNN generation in {:.2?}...", end)
        }

        snn_edges_to_sparse_graph(&snn_edges, &snn_weights, knn.len())
    } else {
        knn_to_sparse_graph(&knn, same_weight)
    };

    let mut results: Vec<Vec<usize>> = Vec::with_capacity(resolutions.len());

    for &res in resolutions {
        let centroid_communities =
            louvain_sparse_graph(&graph, res, params.louvain_iters, multi_level_louvain, seed)?;

        let membership = assignments
            .iter()
            .map(|&c| centroid_communities[c])
            .collect();

        results.push(membership)
    }

    Ok(results)
}

/// Grid-search version of [`fast_louvain_clusters`] with stability metrics.
///
/// Builds the k-means -> kNN/sNN graph once, then runs Louvain across
/// `louvain_seeds` for each resolution. Reports mean/median pairwise ARI
/// (Louvain stability), mean/median conductance (community quality), and
/// the labels from the seed achieving the best (lowest) mean conductance.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix.
/// * `km_type` - String. Which type of k-means clustering to use. `"standard"`
///   or `"minibatch"`.
/// * `resolutions` - Slice of resolutions to iterate over.
/// * `params` - Pipeline parameters.
/// * `run_snn` - Shall a shared nearest neighbour generation be run.
/// * `multi_level_louvain` - Multi-level Louvain or single phase 1 pass.
/// * `seed` - Seed for k-means and kNN (the shared graph).
/// * `louvain_seeds` - Seeds to vary Louvain over. Must contain at least 2
///   entries to produce ARI.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// One [`FastLouvainGridResult`] per resolution, in the same order.
#[allow(clippy::too_many_arguments)]
pub fn fast_louvain_clusters_grid(
    data: MatRef<f32>,
    km_type: &str,
    resolutions: &[f32],
    params: &FastLouvainParams<f32>,
    run_snn: bool,
    multi_level_louvain: bool,
    seed: usize,
    n_louvain_seeds: usize,
    verbose: bool,
) -> Result<Vec<FastLouvainGridResult>, BixverseErrors> {
    if n_louvain_seeds < 2 {
        return Err(BixverseErrors::InvalidArgument(
            "n_louvain_seeds must be at least 2".to_string(),
        ));
    }

    let km_type = parse_k_means(km_type).unwrap_or_default();
    let n_centroids = params.n_centroids.min(data.nrows() - 1);

    let (centroids, assignments) = match km_type {
        KMeansType::StandardKMeans => k_means_clusters(
            data,
            &params.knn_params.ann_dist,
            n_centroids,
            params.kmeans_iters,
            seed,
            verbose,
        ),
        KMeansType::MiniBatchKMeans => train_centroids_minibatch(
            data,
            &params.knn_params.ann_dist,
            n_centroids,
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

    let graph = if run_snn {
        let pruning = params
            .pruning
            .unwrap_or(1.0 / (params.knn_params.k as f32 * 0.8).ceil());

        let knn_flat = flatten_knn_column_major(&knn, params.knn_params.k);
        let snn_method = parse_snn_similiarity_method(&params.snn_similarity).unwrap_or_default();

        let (snn_edges, snn_weights) = if params.full_snn {
            generate_snn_full(
                &knn_flat,
                params.knn_params.k,
                knn.len(),
                pruning,
                snn_method,
                verbose,
            )
        } else {
            generate_snn_limited(
                &knn_flat,
                params.knn_params.k,
                knn.len(),
                pruning,
                snn_method,
                verbose,
            )
        };

        snn_edges_to_sparse_graph(&snn_edges, &snn_weights, knn.len())
    } else {
        knn_to_sparse_graph(&knn, true)
    };

    let mut results = Vec::with_capacity(resolutions.len());

    let mut seed_rng = StdRng::seed_from_u64(seed as u64);
    let louvain_seeds: Vec<usize> = (0..n_louvain_seeds)
        .map(|_| seed_rng.random::<u64>() as usize)
        .collect();

    for &res in resolutions {
        // run Louvain across all seeds at the centroid level
        let centroid_partitions: Vec<Vec<usize>> = louvain_seeds
            .iter()
            .map(|&s| {
                louvain_sparse_graph(&graph, res, params.louvain_iters, multi_level_louvain, s)
            })
            .collect::<Result<_, _>>()?;

        // propagate to per-sample labels
        let sample_partitions: Vec<Vec<usize>> = centroid_partitions
            .iter()
            .map(|cp| assignments.iter().map(|&c| cp[c]).collect())
            .collect();

        // pairwise ARI on centroid-level partitions (cheaper, equivalent ranking)
        let s = louvain_seeds.len();
        let mut aris: Vec<f64> = Vec::with_capacity(s * (s - 1) / 2);
        for i in 0..s {
            for j in (i + 1)..s {
                aris.push(adjusted_rand_index(
                    &centroid_partitions[i],
                    &centroid_partitions[j],
                ));
            }
        }
        let mean_ari = aris.iter().sum::<f64>() / aris.len() as f64;
        let median_ari = median(&aris).unwrap();

        // per-seed mean conductance on the centroid graph
        let mut per_seed_mean_cond = Vec::with_capacity(s);
        let mut per_seed_median_cond = Vec::with_capacity(s);
        for cp in &centroid_partitions {
            let cond = conductance_undirected(&graph, cp)?;
            let finite: Vec<f32> = cond.into_iter().filter(|x| x.is_finite()).collect();
            if finite.is_empty() {
                per_seed_mean_cond.push(f32::NAN);
                per_seed_median_cond.push(f32::NAN);
            } else {
                let mean = finite.iter().sum::<f32>() / finite.len() as f32;
                let mut sorted = finite.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if sorted.len().is_multiple_of(2) {
                    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                } else {
                    sorted[sorted.len() / 2]
                };
                per_seed_mean_cond.push(mean);
                per_seed_median_cond.push(median);
            }
        }

        let mean_conductance = mean_nan(&per_seed_mean_cond);
        let median_conductance = mean_nan(&per_seed_median_cond);

        let mean_n_communities = centroid_partitions
            .iter()
            .map(|cp| cp.iter().max().copied().unwrap_or(0) + 1)
            .sum::<usize>() as f32
            / s as f32;

        // pick best seed by mean conductance (NaN treated as worst)
        let best_idx = per_seed_mean_cond
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let a = if a.is_finite() { **a } else { f32::INFINITY };
                let b = if b.is_finite() { **b } else { f32::INFINITY };
                a.partial_cmp(&b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        results.push(FastLouvainGridResult {
            resolution: res,
            mean_ari: mean_ari as f32,
            median_ari: median_ari as f32,
            mean_conductance,
            median_conductance,
            mean_n_communities,
            best_labels: sample_partitions[best_idx].clone(),
        });
    }

    Ok(results)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use std::collections::HashMap;

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
            n_centroids: 20,
            kmeans_iters: 100,
            knn_params: knn,
            louvain_iters: 10,
            full_snn: false,
            pruning: None,
            snn_similarity: "jaccard".to_string(),
        }
    }

    fn majority(xs: &[usize]) -> usize {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &x in xs {
            *counts.entry(x).or_insert(0) += 1;
        }
        *counts.iter().max_by_key(|(_, c)| **c).unwrap().0
    }

    #[test]
    fn output_length_matches_input() {
        let data = make_two_blobs(100, 5, 5.0, 42);

        let resolutions: Vec<f32> = vec![1.0];

        // without snn
        let labels = fast_louvain_clusters(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            false,
            true,
            0,
            false,
        )
        .unwrap();
        assert_eq!(labels.len(), resolutions.len());
        assert_eq!(labels[0].len(), 200);

        // with snn
        let labels_snn = fast_louvain_clusters(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            false,
            true,
            0,
            false,
        )
        .unwrap();
        assert_eq!(labels_snn.len(), resolutions.len());
        assert_eq!(labels_snn[0].len(), 200);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = default_params();
        let resolutions: Vec<f32> = vec![1.0];

        let a = fast_louvain_clusters(
            data.as_ref(),
            "standard",
            &resolutions,
            &params,
            false,
            false,
            true,
            7,
            false,
        )
        .unwrap();
        let b = fast_louvain_clusters(
            data.as_ref(),
            "standard",
            &resolutions,
            &params,
            false,
            false,
            true,
            7,
            false,
        )
        .unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn produces_multiple_communities_on_separated_blobs() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let params = default_params();
        let resolutions: Vec<f32> = vec![1.0];

        let labels = fast_louvain_clusters(
            data.as_ref(),
            "standard",
            &resolutions,
            &params,
            false,
            false,
            true,
            0,
            false,
        )
        .unwrap();

        let (blob_a, blob_b) = labels[0].split_at(n_per);

        let maj_a = majority(blob_a);
        let maj_b = majority(blob_b);
        assert_ne!(maj_a, maj_b);

        let purity_a = blob_a.iter().filter(|&&l| l == maj_a).count() as f32 / n_per as f32;
        let purity_b = blob_b.iter().filter(|&&l| l == maj_b).count() as f32 / n_per as f32;

        assert!(purity_a > 0.9, "blob A purity {}", purity_a);
        assert!(purity_b > 0.9, "blob B purity {}", purity_b);
    }

    #[test]
    fn produces_multiple_communities_on_separated_blobs_snn() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let params = default_params();
        let resolutions: Vec<f32> = vec![1.0];

        let labels = fast_louvain_clusters(
            data.as_ref(),
            "standard",
            &resolutions,
            &params,
            true,
            false,
            true,
            0,
            false,
        )
        .unwrap();

        let (blob_a, blob_b) = labels[0].split_at(n_per);

        let maj_a = majority(blob_a);
        let maj_b = majority(blob_b);
        assert_ne!(maj_a, maj_b);

        let purity_a = blob_a.iter().filter(|&&l| l == maj_a).count() as f32 / n_per as f32;
        let purity_b = blob_b.iter().filter(|&&l| l == maj_b).count() as f32 / n_per as f32;

        assert!(purity_a > 0.9, "blob A purity {}", purity_a);
        assert!(purity_b > 0.9, "blob B purity {}", purity_b);
    }

    #[test]
    fn grid_output_shape_matches_resolutions() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let resolutions: Vec<f32> = vec![0.5, 1.0, 2.0];

        let results = fast_louvain_clusters_grid(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            true,
            0,
            5,
            false,
        )
        .unwrap();

        assert_eq!(results.len(), resolutions.len());
        for (r, res) in results.iter().zip(resolutions.iter()) {
            assert_eq!(r.resolution, *res);
            assert_eq!(r.best_labels.len(), 200);
        }
    }

    #[test]
    fn grid_rejects_too_few_seeds() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let result = fast_louvain_clusters_grid(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            true,
            0,
            1,
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn grid_deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let a = fast_louvain_clusters_grid(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            true,
            7,
            4,
            false,
        )
        .unwrap();
        let b = fast_louvain_clusters_grid(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            true,
            7,
            4,
            false,
        )
        .unwrap();

        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert!((ra.mean_ari - rb.mean_ari).abs() < 1e-6);
            assert!((ra.median_ari - rb.median_ari).abs() < 1e-6);
            assert!((ra.mean_conductance - rb.mean_conductance).abs() < 1e-6);
            assert!((ra.median_conductance - rb.median_conductance).abs() < 1e-6);
            assert_eq!(ra.best_labels, rb.best_labels);
        }
    }

    #[test]
    fn grid_metrics_in_valid_ranges() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let results = fast_louvain_clusters_grid(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            true,
            0,
            4,
            false,
        )
        .unwrap();

        let r = &results[0];

        assert!(
            r.mean_ari >= -1.0 && r.mean_ari <= 1.0,
            "mean_ari {}",
            r.mean_ari
        );
        assert!(
            r.median_ari >= -1.0 && r.median_ari <= 1.0,
            "median_ari {}",
            r.median_ari
        );
        assert!(
            r.mean_conductance >= 0.0,
            "mean_conductance {}",
            r.mean_conductance
        );
        assert!(
            r.median_conductance >= 0.0,
            "median_conductance {}",
            r.median_conductance
        );
        assert!(r.mean_n_communities >= 1.0);
    }

    #[test]
    fn grid_well_separated_blobs_are_stable() {
        // On obviously separable data Louvain should agree across seeds, giving
        // ARI close to 1.0 and recovering the two blobs.
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let results = fast_louvain_clusters_grid(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            false,
            true,
            0,
            5,
            false,
        )
        .unwrap();

        let r = &results[0];
        assert!(r.mean_ari > 0.95, "mean_ari {}", r.mean_ari);

        let (blob_a, blob_b) = r.best_labels.split_at(n_per);
        let maj_a = majority(blob_a);
        let maj_b = majority(blob_b);
        assert_ne!(maj_a, maj_b);
    }

    #[test]
    fn grid_runs_with_snn() {
        let data = make_two_blobs(200, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let results = fast_louvain_clusters_grid(
            data.as_ref(),
            "standard",
            &resolutions,
            &default_params(),
            true,
            true,
            0,
            4,
            false,
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].best_labels.len(), 400);
    }
}
