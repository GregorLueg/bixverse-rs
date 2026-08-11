//! A fast approach to generate clustering based on k-means clustering first on
//! the data, followed by kNN search on the centroids, (for now only) Louvain
//! clustering on the resulting kNN graph with subsequent propagation of the
//! module membership based on the original nearest centroid.
//!
//! The pipeline is split into three stages ([`fast_cluster_kmeans`],
//! [`build_centroid_graph`] and [`louvain_over_resolutions`] /
//! [`louvain_grid_over_resolutions`]) so that the GPU variant in
//! `crate::gpu::sc_gpu::fast_clusters_gpu` can swap out stage one and reuse the
//! rest untouched.

use either::Either;
use faer::{Mat, MatRef};
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

///////////////////
// Enums / types //
///////////////////

/// Enum that defines the results pending type of fast clustering
pub enum FastLouvainResults {
    /// Single version with only Louvain membership
    Single {
        /// Membership assignments across the resolutions
        assignments: Vec<Vec<usize>>,
        /// Optional k-means cluster assignment
        clusters: Option<Vec<usize>>,
        /// Optional k-means centroids
        centroids: Option<Mat<f32>>,
    },
    /// Grid version across several seeds
    GridRes {
        /// [FastLouvainGridResult] per resolution
        assignments: Vec<FastLouvainGridResult>,
        /// Optional k-means cluster assignment
        clusters: Option<Vec<usize>>,
        /// Optional k-means centroids
        centroids: Option<Mat<f32>>,
    },
}

impl FastLouvainResults {
    /// Build the [`FastLouvainResults::Single`] variant.
    ///
    /// ### Params
    ///
    /// * `assignments` - Per-resolution membership vectors
    /// * `centroids` - The k-means centroids
    /// * `clusters` - The k-means assignment per sample
    /// * `keep_kmeans` - If `false`, `centroids` and `clusters` are dropped
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn single(
        assignments: Vec<Vec<usize>>,
        centroids: Mat<f32>,
        clusters: Vec<usize>,
        keep_kmeans: bool,
    ) -> Self {
        let (centroids, clusters) = if keep_kmeans {
            (Some(centroids), Some(clusters))
        } else {
            (None, None)
        };
        Self::Single {
            assignments,
            clusters,
            centroids,
        }
    }

    /// Build the [`FastLouvainResults::GridRes`] variant.
    ///
    /// ### Params
    ///
    /// * `assignments` - Per-resolution [FastLouvainGridResult]
    /// * `centroids` - The k-means centroids
    /// * `clusters` - The k-means assignment per sample
    /// * `keep_kmeans` - If `false`, `centroids` and `clusters` are dropped
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn grid(
        assignments: Vec<FastLouvainGridResult>,
        centroids: Mat<f32>,
        clusters: Vec<usize>,
        keep_kmeans: bool,
    ) -> Self {
        let (centroids, clusters) = if keep_kmeans {
            (Some(centroids), Some(clusters))
        } else {
            (None, None)
        };
        Self::GridRes {
            assignments,
            clusters,
            centroids,
        }
    }

    /// Helper to get the assignments
    ///
    /// ### Returns
    ///
    /// Union type of either `Vec<Vec<usize>>` or a vector of
    /// [FastLouvainGridResult].
    #[inline]
    pub fn get_assignments(&self) -> Either<Vec<Vec<usize>>, Vec<FastLouvainGridResult>> {
        match self {
            Self::Single { assignments, .. } => Either::Left(assignments.clone()),
            Self::GridRes { assignments, .. } => Either::Right(assignments.clone()),
        }
    }

    /// Helper to return the centroids
    ///
    /// ### Returns
    ///
    /// Returns the centroid matrix if found; Error otherwise
    #[inline]
    pub fn get_centroids(&self) -> Result<Mat<f32>, BixverseErrors> {
        let centroids = match self {
            Self::Single { centroids, .. } | Self::GridRes { centroids, .. } => centroids,
        };
        centroids
            .to_owned()
            .ok_or(BixverseErrors::FastClusterNoCentroids)
    }

    /// Helper to return the k-means cluster assignments
    ///
    /// ### Returns
    ///
    /// Returns the k-means cluster assignment; Error otherwise
    #[inline]
    pub fn get_k_mean_clusters(&self) -> Result<Vec<usize>, BixverseErrors> {
        let clusters = match self {
            Self::Single { clusters, .. } | Self::GridRes { clusters, .. } => clusters,
        };
        clusters
            .clone()
            .ok_or(BixverseErrors::FastClusterNoKmeansAssignments)
    }
}

////////////
// Params //
////////////

/// Graph-construction settings shared by the CPU and GPU fast-clustering paths.
///
/// Both `FastLouvainParams` and its GPU sibling produce one of these via their
/// `graph_params` method, so [`build_centroid_graph`] does not need to know
/// which k-means backend produced the centroids.
#[derive(Clone, Debug)]
pub struct CentroidGraphParams {
    /// [KnnParams] applied to the centroids. `ann_dist` also drives the k-means
    /// distance and `k` is the number of neighbours per centroid.
    pub knn_params: KnnParams,
    /// Shall the weight of the kNN be set to 1.0 across or edges that have
    /// reverse edges be double counted.
    pub same_weight: bool,
    /// Shall the full sNN graph be generated (also from non-connected
    /// neighbours)
    pub full_snn: bool,
    /// Optional pruning threshold. If not provided, will default to
    /// `1 / ceil(0.8 * k)`.
    pub pruning: Option<f32>,
    /// String for the type of sNN similarity to use. One of `"jaccard"` or
    /// `"rank"`.
    pub snn_similarity: String,
}

/// Parameters for fast Louvain clustering via k-means + kNN.
#[derive(Clone, Debug)]
pub struct FastLouvainParams<T> {
    // -- k means --
    /// Number of k-means centroids.
    pub n_centroids: usize,
    /// Batch size for mini-batch k-means
    pub batch_size: usize,
    /// Drift threshold for mini batch k-means
    pub drift_threshold: T,
    /// Learning rate exponent for mini batch k-means:
    /// `eta = m / count[c]^lr_alpha`
    pub lr_alpha: T,
    /// k-means parameters, see [KMeansParamsWrappers]
    pub kmeans_params: KMeansParamsWrappers,

    // -- knn --
    /// [KnnParams] parameters applied to the centroids. `ann_dist` also drives
    /// the k-means distance and `k` is the number of neighbours per centroid.
    pub knn_params: KnnParams,
    /// Shall the weight of the kNN be set to 1.0 across or edges that have
    /// reverse edges be double counted.
    pub same_weight: bool,

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
    /// Shall multi-level Louvain be applied
    pub multi_level_louvain: bool,
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
            batch_size: 4096,
            drift_threshold: T::from_f64(1e-4).unwrap(),
            lr_alpha: T::from_f64(1.0).unwrap(),
            kmeans_params: KMeansParamsWrappers::default(),
            // snn
            full_snn: false,
            pruning: None,
            snn_similarity: "jaccard".to_string(),
            // knn - louvain
            knn_params: KnnParams::default(),
            same_weight: false,
            // louvain
            louvain_iters: 10,
            multi_level_louvain: true,
        }
    }

    /// Returns the stored k-means parameters
    ///
    /// Helper function to extract the parameters
    ///
    /// ### Returns
    ///
    /// The [KMeansParamsWrappers]
    pub fn get_kmeans_params(&self) -> KMeansParamsWrappers {
        self.kmeans_params.clone()
    }
}

impl FastLouvainParams<f32> {
    /// Extract the graph-construction settings for [`build_centroid_graph`].
    ///
    /// ### Returns
    ///
    /// The [CentroidGraphParams].
    pub fn graph_params(&self) -> CentroidGraphParams {
        CentroidGraphParams {
            knn_params: self.knn_params.clone(),
            same_weight: self.same_weight,
            full_snn: self.full_snn,
            pruning: self.pruning,
            snn_similarity: self.snn_similarity.clone(),
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

/////////////
// Results //
/////////////

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

/////////////
// Helpers //
/////////////

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

////////////
// Stages //
////////////

/// Stage one: k-means coarsening of the data on the CPU.
///
/// The requested centroid count is clamped to `n_samples - 1`. The distance
/// metric is taken from `params.knn_params.ann_dist`, so the coarsening and the
/// centroid kNN always agree on the geometry.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix.
/// * `km_type` - Which type of k-means clustering to use. `"standard"` or
///   `"minibatch"`; anything unparseable falls back to `"standard"`.
/// * `params` - Pipeline parameters.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// `(centroids of shape n_centroids x n_features, per-sample centroid index)`.
pub fn fast_cluster_kmeans(
    data: MatRef<f32>,
    km_type: &str,
    params: &FastLouvainParams<f32>,
    seed: usize,
    verbose: usize,
) -> Result<(Mat<f32>, Vec<usize>), BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let km_type = parse_k_means(km_type).unwrap_or_default();
    let n_centroids = params.n_centroids.min(data.nrows() - 1);

    let res = match km_type {
        KMeansType::StandardKMeans => k_means_clusters(
            data,
            &params.knn_params.ann_dist,
            n_centroids,
            Some(params.get_kmeans_params()),
            seed,
            verbosity.detailed_verbosity(),
        )?,
        KMeansType::MiniBatchKMeans => train_centroids_minibatch(
            data,
            &params.knn_params.ann_dist,
            n_centroids,
            params.kmeans_params.clone().get_data().iters,
            params.batch_size,
            params.drift_threshold,
            params.lr_alpha,
            seed,
            verbosity.detailed_verbosity(),
        ),
    };

    Ok(res)
}

/// Stage two: build the neighbour graph over the k-means centroids.
///
/// Runs a kNN search on the centroids and turns it into a [SparseGraph], either
/// directly or via a shared-nearest-neighbour pass. The sNN pruning threshold
/// defaults to `1 / ceil(0.8 * k)` when `params.pruning` is `None`.
///
/// ### Params
///
/// * `centroids` - n_centroids x n_features matrix from stage one.
/// * `params` - Graph-construction settings.
/// * `run_snn` - Shall a shared nearest neighbour graph be generated.
/// * `seed` - Seed for reproducibility of the kNN search.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The undirected centroid graph.
pub fn build_centroid_graph(
    centroids: MatRef<f32>,
    params: &CentroidGraphParams,
    run_snn: bool,
    seed: usize,
    verbose: usize,
) -> Result<SparseGraph<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let knn = dispatch_knn(
        centroids,
        params.knn_params.k,
        &params.knn_params,
        seed,
        verbosity.detailed_verbosity(),
    )?;

    if !run_snn {
        return Ok(knn_to_sparse_graph(&knn, params.same_weight));
    }

    let pruning = params
        .pruning
        .unwrap_or(1.0 / (params.knn_params.k as f32 * 0.8).ceil());

    let start = Instant::now();

    if verbosity.normal_verbosity() {
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

    if verbosity.normal_verbosity() {
        println!(" Finished the sNN generation in {:.2?}...", start.elapsed())
    }

    Ok(snn_edges_to_sparse_graph(
        &snn_edges,
        &snn_weights,
        knn.len(),
    ))
}

/// Stage three: Louvain per resolution, propagated back to the samples.
///
/// ### Params
///
/// * `graph` - The centroid graph from stage two.
/// * `assignments` - Per-sample centroid index from stage one.
/// * `resolutions` - Slice of resolutions to iterate over.
/// * `louvain_iters` - Number of Louvain iterations.
/// * `multi_level` - If `true`, run the full multi-level Louvain (phase 1 +
///   phase 2 aggregation, repeated). If `false`, run only one phase 1 pass
///   (matches Phenograph / the original doubletdetection behaviour).
/// * `seed` - Seed for reproducibility.
///
/// ### Returns
///
/// One per-sample membership vector per resolution, in the same order.
pub fn louvain_over_resolutions(
    graph: &SparseGraph<f32>,
    assignments: &[usize],
    resolutions: &[f32],
    louvain_iters: usize,
    multi_level: bool,
    seed: usize,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    resolutions
        .iter()
        .map(|&res| {
            let centroid_communities =
                louvain_sparse_graph(graph, res, louvain_iters, multi_level, seed)?;
            Ok(assignments
                .iter()
                .map(|&c| centroid_communities[c])
                .collect())
        })
        .collect()
}

/// Stage three, grid variant: Louvain across several seeds per resolution.
///
/// Reports mean/median pairwise ARI (Louvain stability), mean/median
/// conductance (community quality), and the labels from the seed achieving the
/// best (lowest) mean conductance. Both ARI and conductance are computed on the
/// centroid-level partitions, which is cheaper than the per-sample ones and
/// ranks the resolutions identically.
///
/// ### Params
///
/// * `graph` - The centroid graph from stage two.
/// * `assignments` - Per-sample centroid index from stage one.
/// * `resolutions` - Slice of resolutions to iterate over.
/// * `louvain_iters` - Number of Louvain iterations.
/// * `multi_level` - Multi-level Louvain or a single phase 1 pass.
/// * `seed` - Seed from which the per-run Louvain seeds are drawn.
/// * `n_louvain_seeds` - Number of Louvain seeds. Must be at least 2 to produce
///   an ARI.
///
/// ### Returns
///
/// One [FastLouvainGridResult] per resolution, in the same order.
pub fn louvain_grid_over_resolutions(
    graph: &SparseGraph<f32>,
    assignments: &[usize],
    resolutions: &[f32],
    louvain_iters: usize,
    multi_level: bool,
    seed: usize,
    n_louvain_seeds: usize,
) -> Result<Vec<FastLouvainGridResult>, BixverseErrors> {
    if n_louvain_seeds < 2 {
        return Err(BixverseErrors::InvalidArgument(
            "n_louvain_seeds must be at least 2".to_string(),
        ));
    }

    let mut seed_rng = StdRng::seed_from_u64(seed as u64);
    let louvain_seeds: Vec<usize> = (0..n_louvain_seeds)
        .map(|_| seed_rng.random::<u64>() as usize)
        .collect();

    let mut results = Vec::with_capacity(resolutions.len());

    for &res in resolutions {
        // run Louvain across all seeds at the centroid level
        let centroid_partitions: Vec<Vec<usize>> = louvain_seeds
            .iter()
            .map(|&s| louvain_sparse_graph(graph, res, louvain_iters, multi_level, s))
            .collect::<Result<_, _>>()?;

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
            let cond = conductance_undirected(graph, cp)?;
            let finite: Vec<f32> = cond.into_iter().filter(|x| x.is_finite()).collect();
            per_seed_mean_cond.push(mean_nan(&finite));
            per_seed_median_cond.push(median(&finite).unwrap_or(f32::NAN));
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

        // propagate the winning centroid partition to per-sample labels
        let best_labels = assignments
            .iter()
            .map(|&c| centroid_partitions[best_idx][c])
            .collect();

        results.push(FastLouvainGridResult {
            resolution: res,
            mean_ari: mean_ari as f32,
            median_ari: median_ari as f32,
            mean_conductance,
            median_conductance,
            mean_n_communities,
            best_labels,
        });
    }

    Ok(results)
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
/// * `return_k_mean` - Shall the k-means centroids and assignments be returned
///   alongside the Louvain memberships.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [FastLouvainResults], `Single` variant.
#[allow(clippy::too_many_arguments)]
pub fn fast_louvain_clusters(
    data: MatRef<f32>,
    km_type: &str,
    resolutions: &[f32],
    params: &FastLouvainParams<f32>,
    run_snn: bool,
    return_k_mean: bool,
    seed: usize,
    verbose: usize,
) -> Result<FastLouvainResults, BixverseErrors> {
    let (centroids, assignments) = fast_cluster_kmeans(data, km_type, params, seed, verbose)?;

    let graph = build_centroid_graph(
        centroids.as_ref(),
        &params.graph_params(),
        run_snn,
        seed,
        verbose,
    )?;

    let results = louvain_over_resolutions(
        &graph,
        &assignments,
        resolutions,
        params.louvain_iters,
        params.multi_level_louvain,
        seed,
    )?;

    Ok(FastLouvainResults::single(
        results,
        centroids,
        assignments,
        return_k_mean,
    ))
}

/// Grid-search version of [`fast_louvain_clusters`] with stability metrics.
///
/// Builds the k-means -> kNN/sNN graph once, then runs Louvain across
/// `n_louvain_seeds` seeds for each resolution. Reports mean/median pairwise
/// ARI (Louvain stability), mean/median conductance (community quality), and
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
/// * `return_k_mean` - Shall the k-means centroids and assignments be returned
///   alongside the grid results.
/// * `seed` - Seed for k-means and kNN (the shared graph).
/// * `n_louvain_seeds` - Number of seeds to vary Louvain over. Must be at least
///   2 to produce ARI.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [FastLouvainResults], `GridRes` variant, with one
/// [FastLouvainGridResult] per resolution in the same order.
#[allow(clippy::too_many_arguments)]
pub fn fast_louvain_clusters_grid(
    data: MatRef<f32>,
    km_type: &str,
    resolutions: &[f32],
    params: &FastLouvainParams<f32>,
    run_snn: bool,
    return_k_mean: bool,
    seed: usize,
    n_louvain_seeds: usize,
    verbose: usize,
) -> Result<FastLouvainResults, BixverseErrors> {
    let (centroids, assignments) = fast_cluster_kmeans(data, km_type, params, seed, verbose)?;

    let graph = build_centroid_graph(
        centroids.as_ref(),
        &params.graph_params(),
        run_snn,
        seed,
        verbose,
    )?;

    let results = louvain_grid_over_resolutions(
        &graph,
        &assignments,
        resolutions,
        params.louvain_iters,
        params.multi_level_louvain,
        seed,
        n_louvain_seeds,
    )?;

    Ok(FastLouvainResults::grid(
        results,
        centroids,
        assignments,
        return_k_mean,
    ))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use either::Either;
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
        knn.k = 5;
        FastLouvainParams {
            batch_size: 100,
            drift_threshold: 1e-4,
            lr_alpha: 1.0,
            n_centroids: 30,
            kmeans_params: KMeansParamsWrappers::default(),
            knn_params: knn,
            louvain_iters: 10,
            full_snn: false,
            pruning: None,
            same_weight: false,
            multi_level_louvain: true,
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

    fn unwrap_single(r: FastLouvainResults) -> Vec<Vec<usize>> {
        match r.get_assignments() {
            Either::Left(v) => v,
            Either::Right(_) => panic!("expected Single variant"),
        }
    }

    fn unwrap_grid(r: FastLouvainResults) -> Vec<FastLouvainGridResult> {
        match r.get_assignments() {
            Either::Right(v) => v,
            Either::Left(_) => panic!("expected GridRes variant"),
        }
    }

    #[test]
    fn centroid_graph_matches_centroid_count() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = default_params();
        let (centroids, _) = fast_cluster_kmeans(data.as_ref(), "standard", &params, 0, 0).unwrap();

        for run_snn in [false, true] {
            let graph =
                build_centroid_graph(centroids.as_ref(), &params.graph_params(), run_snn, 0, 0)
                    .unwrap();
            assert_eq!(graph.get_node_number(), centroids.nrows());
            assert!(!graph.is_directed());
        }
    }

    #[test]
    fn louvain_over_resolutions_propagates_to_samples() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = default_params();
        let resolutions: Vec<f32> = vec![1.0, 0.5];

        let (centroids, assignments) =
            fast_cluster_kmeans(data.as_ref(), "standard", &params, 0, 0).unwrap();
        let graph =
            build_centroid_graph(centroids.as_ref(), &params.graph_params(), false, 0, 0).unwrap();

        let labels = louvain_over_resolutions(
            &graph,
            &assignments,
            &resolutions,
            params.louvain_iters,
            params.multi_level_louvain,
            0,
        )
        .unwrap();

        assert_eq!(labels.len(), resolutions.len());
        for l in &labels {
            assert_eq!(l.len(), assignments.len());
        }
    }

    #[test]
    fn output_length_matches_input() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let resolutions: Vec<f32> = vec![1.0, 0.5];

        // without snn
        let labels = unwrap_single(
            fast_louvain_clusters(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                false,
                false,
                0,
                0,
            )
            .unwrap(),
        );
        assert_eq!(labels.len(), resolutions.len());
        assert_eq!(labels[0].len(), 200);

        // with snn
        let labels_snn = unwrap_single(
            fast_louvain_clusters(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                true,
                false,
                0,
                0,
            )
            .unwrap(),
        );
        assert_eq!(labels_snn.len(), resolutions.len());
        assert_eq!(labels_snn[0].len(), 200);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = default_params();
        let resolutions: Vec<f32> = vec![1.0];

        let a = unwrap_single(
            fast_louvain_clusters(
                data.as_ref(),
                "standard",
                &resolutions,
                &params,
                false,
                false,
                7,
                0,
            )
            .unwrap(),
        );
        let b = unwrap_single(
            fast_louvain_clusters(
                data.as_ref(),
                "standard",
                &resolutions,
                &params,
                false,
                false,
                7,
                0,
            )
            .unwrap(),
        );

        assert_eq!(a, b);
    }

    #[test]
    fn produces_multiple_communities_on_separated_blobs() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let params = default_params();
        let resolutions: Vec<f32> = vec![1.0];

        let labels = unwrap_single(
            fast_louvain_clusters(
                data.as_ref(),
                "standard",
                &resolutions,
                &params,
                false,
                false,
                0,
                0,
            )
            .unwrap(),
        );

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

        let labels = unwrap_single(
            fast_louvain_clusters(
                data.as_ref(),
                "standard",
                &resolutions,
                &params,
                true,
                false,
                0,
                0,
            )
            .unwrap(),
        );

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

        let results = unwrap_grid(
            fast_louvain_clusters_grid(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                false,
                false,
                0,
                5,
                0,
            )
            .unwrap(),
        );

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
            false,
            0,
            1,
            0,
        );

        assert!(result.is_err());
    }

    #[test]
    fn grid_deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let a = unwrap_grid(
            fast_louvain_clusters_grid(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                false,
                false,
                7,
                4,
                0,
            )
            .unwrap(),
        );
        let b = unwrap_grid(
            fast_louvain_clusters_grid(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                false,
                false,
                7,
                4,
                0,
            )
            .unwrap(),
        );

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

        let results = unwrap_grid(
            fast_louvain_clusters_grid(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                false,
                false,
                0,
                4,
                0,
            )
            .unwrap(),
        );

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
        let n_per = 250;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0, 0.5, 0.25];

        let results = unwrap_grid(
            fast_louvain_clusters_grid(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                false,
                false,
                42,
                5,
                0,
            )
            .unwrap(),
        );

        for (i, r) in results.iter().enumerate() {
            assert!(
                r.mean_ari > 0.9,
                "mean_ari {} for seed no {}",
                r.mean_ari,
                i
            );

            let (blob_a, blob_b) = r.best_labels.split_at(n_per);
            let maj_a = majority(blob_a);
            let maj_b = majority(blob_b);
            assert_ne!(maj_a, maj_b, "not the same majority for seed no {}", i);
        }
    }

    #[test]
    fn grid_runs_with_snn() {
        let data = make_two_blobs(200, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let results = unwrap_grid(
            fast_louvain_clusters_grid(
                data.as_ref(),
                "standard",
                &resolutions,
                &default_params(),
                true,
                false,
                0,
                4,
                0,
            )
            .unwrap(),
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].best_labels.len(), 400);
    }
}
