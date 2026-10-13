//! A fast approach to generate clustering based on k-means clustering first on
//! the data, followed by a graph over the centroids, (for now only) Louvain
//! clustering on that graph with subsequent propagation of the module
//! membership based on the original nearest centroid.
//!
//! The centroid graph is either a kNN/sNN graph on the centroid coordinates or
//! the cell-mediated graph from Secuer, where centroids are linked by the cells
//! that sit near both.
//!
//! The pipeline is split into three stages ([`fast_cluster_kmeans`],
//! [`build_centroid_graph`] and [`louvain_over_resolutions`] /
//! [`louvain_grid_over_resolutions`]) so that the GPU variant in
//! `crate::gpu::sc_gpu::fast_clusters_gpu` can swap out stage one and reuse the
//! rest untouched.

use ann_search_rs::utils::dist::{Dist, SimdDistance, parse_ann_dist};
use ann_search_rs::utils::k_means_utils::assign_all_parallel_top_m;
use ann_search_rs::utils::matrix_to_flat;
use either::Either;
use faer::{Mat, MatRef};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
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
    /// Build the cell-mediated centroid graph instead of the kNN/sNN one. If
    /// `true`, `run_snn` and the sNN settings are ignored.
    pub cell_mediated: bool,
    /// Number of nearest centroids each cell links to in the cell-mediated
    /// graph.
    pub n_anchors: usize,
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

    // -- cell-mediated graph --
    /// Build the cell-mediated centroid graph instead of the kNN/sNN one. If
    /// `true`, `run_snn` and the sNN settings are ignored.
    pub cell_mediated: bool,
    /// Number of nearest centroids each cell links to in the cell-mediated
    /// graph.
    pub n_anchors: usize,

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
            // cell-mediated graph
            cell_mediated: false,
            n_anchors: 5,
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
        self.kmeans_params
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
            cell_mediated: self.cell_mediated,
            n_anchors: self.n_anchors,
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

/// Cells per accumulation block in [`cell_mediated_graph`]. Blocks are summed
/// in a fixed order so the edge weights do not depend on the thread count;
/// 16k cells keeps the per-block maps small while giving rayon enough blocks
/// at atlas scale.
const CELL_MEDIATED_BLOCK: usize = 16_384;

/// Build the centroid graph through the cells.
///
/// Every cell links to its `n_anchors` nearest centroids with a locally scaled
/// Gaussian weight `w_ij = exp(-d_ij^2 / (2 sigma_i^2))`, `sigma_i` being the
/// mean distance to those centroids. Centroids are then joined by
/// `W = B^T D_x^-1 B`, i.e. by the cells that sit near both, so two centroids
/// that are close in space but have no cells between them stay apart. The
/// diagonal of `W` is dropped, so no self-loops reach the community detection.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix the centroids were fitted on.
/// * `centroids` - n_centroids x n_features matrix.
/// * `n_anchors` - Centroids per cell; clamped to `1..=n_centroids`.
/// * `ann_dist` - Distance metric, `"euclidean"` or `"cosine"`.
///
/// ### Returns
///
/// The undirected centroid graph.
///
/// ### References
///
/// Wei et al., PLoS Comput Biol, 2022
fn cell_mediated_graph(
    data: MatRef<f32>,
    centroids: MatRef<f32>,
    n_anchors: usize,
    ann_dist: &str,
) -> SparseGraph<f32> {
    let (data_flat, n, dim) = matrix_to_flat(data);
    let (cent_flat, k, _) = matrix_to_flat(centroids);
    let dist = parse_ann_dist(ann_dist).unwrap_or_default();
    let (data_norms, cent_norms) = if dist == Dist::Cosine {
        (
            compute_data_norms(&data_flat, dim, n, &dist),
            recompute_centroid_norms(&cent_flat, dim, k, &dist),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // same clamp as assign_all_parallel_top_m, which sets the row stride
    let m = n_anchors.min(k).max(1);
    let anchors = assign_all_parallel_top_m(
        &data_flat,
        &data_norms,
        dim,
        n,
        &cent_flat,
        &cent_norms,
        k,
        m,
        &dist,
    );

    let blocks: Vec<FxHashMap<(u32, u32), f32>> = anchors
        .par_chunks(m * CELL_MEDIATED_BLOCK)
        .enumerate()
        .map(|(b, block)| {
            let mut acc: FxHashMap<(u32, u32), f32> = FxHashMap::default();
            let mut w = vec![0.0f32; m];
            for (r, row) in block.chunks_exact(m).enumerate() {
                let i = b * CELL_MEDIATED_BLOCK + r;
                let x = &data_flat[i * dim..(i + 1) * dim];
                for (wj, &c) in w.iter_mut().zip(row) {
                    let c = c as usize;
                    let cent = &cent_flat[c * dim..(c + 1) * dim];
                    *wj = match dist {
                        Dist::Cosine => {
                            let denom = data_norms[i] * cent_norms[c];
                            if denom > 0.0 {
                                (1.0 - f32::dot_simd(x, cent) / denom).max(0.0)
                            } else {
                                1.0
                            }
                        }
                        _ => f32::euclidean_simd(x, cent).sqrt(),
                    };
                }
                let sigma = w.iter().sum::<f32>() / m as f32;
                for wj in w.iter_mut() {
                    // a cell on top of all its anchors gets equal weights
                    *wj = if sigma > 0.0 {
                        (-(*wj * *wj) / (2.0 * sigma * sigma)).exp()
                    } else {
                        1.0
                    };
                }
                let dx: f32 = w.iter().sum();
                for a in 0..m {
                    for c in (a + 1)..m {
                        let key = (row[a].min(row[c]), row[a].max(row[c]));
                        *acc.entry(key).or_insert(0.0) += w[a] * w[c] / dx;
                    }
                }
            }
            acc
        })
        .collect();

    let mut total: FxHashMap<(u32, u32), f32> = FxHashMap::default();
    for block in blocks {
        for (key, v) in block {
            *total.entry(key).or_insert(0.0) += v;
        }
    }
    let mut pairs: Vec<((u32, u32), f32)> = total.into_iter().collect();
    pairs.sort_unstable_by_key(|p| p.0);

    let mut edges = Vec::with_capacity(2 * pairs.len());
    let mut weights = Vec::with_capacity(pairs.len());
    for ((u, v), wt) in pairs {
        edges.push(u as usize);
        edges.push(v as usize);
        weights.push(wt);
    }

    edge_pairs_to_sparse_graph(&edges, &weights, k)
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
            params.kmeans_params.get_data().iters,
            params.batch_size,
            params.drift_threshold,
            params.lr_alpha,
            seed,
            verbosity.detailed_verbosity(),
        ),
    };

    Ok(res)
}

/// Stage two: build the graph over the k-means centroids.
///
/// With `params.cell_mediated`, centroids are linked by the cells that sit near
/// both (see [`cell_mediated_graph`]). Otherwise runs a kNN search on the
/// centroids and turns it into a [SparseGraph], either directly or via a
/// shared-nearest-neighbour pass. The sNN pruning threshold defaults to
/// `1 / ceil(0.8 * k)` when `params.pruning` is `None`.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix the centroids were fitted on. Only
///   read for the cell-mediated graph.
/// * `centroids` - n_centroids x n_features matrix from stage one.
/// * `params` - Graph-construction settings.
/// * `run_snn` - Shall a shared nearest neighbour graph be generated. Ignored
///   for the cell-mediated graph.
/// * `seed` - Seed for reproducibility of the kNN search.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The undirected centroid graph.
pub fn build_centroid_graph(
    data: MatRef<f32>,
    centroids: MatRef<f32>,
    params: &CentroidGraphParams,
    run_snn: bool,
    seed: usize,
    verbose: usize,
) -> Result<SparseGraph<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    if params.cell_mediated {
        let start = Instant::now();
        let graph = cell_mediated_graph(
            data,
            centroids,
            params.n_anchors,
            &params.knn_params.ann_dist,
        );
        if verbosity.normal_verbosity() {
            println!(
                " Built the cell-mediated centroid graph in {:.2?}...",
                start.elapsed()
            );
        }
        return Ok(graph);
    }

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
/// Runs k-means to obtain centroids, builds a graph on the centroids,
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
        data,
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
/// Builds the k-means -> centroid graph once, then runs Louvain across
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
        data,
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
            cell_mediated: false,
            n_anchors: 5,
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

    /// The same seed must give identical assignments, so nothing downstream sees run-to-run drift.
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

    /// Two well-separated blobs never share a community, on the kNN, sNN and
    /// cell-mediated centroid graphs. A blob may split into several
    /// communities: on the SNN centroid graph that is the higher-modularity
    /// answer (igraph splits it too).
    #[test]
    fn produces_multiple_communities_on_separated_blobs() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        for (run_snn, cell_mediated) in [(false, false), (true, false), (false, true)] {
            let mut params = default_params();
            params.cell_mediated = cell_mediated;
            let labels = unwrap_single(
                fast_louvain_clusters(
                    data.as_ref(),
                    "standard",
                    &resolutions,
                    &params,
                    run_snn,
                    false,
                    0,
                    0,
                )
                .unwrap(),
            );

            assert_eq!(labels.len(), resolutions.len());
            assert_eq!(labels[0].len(), 2 * n_per);

            let (blob_a, blob_b) = labels[0].split_at(n_per);
            assert_ne!(
                majority(blob_a),
                majority(blob_b),
                "run_snn {} cell_mediated {}",
                run_snn,
                cell_mediated
            );

            // every community is at least 90% one blob
            let n_labels = labels[0].iter().max().unwrap() + 1;
            for l in 0..n_labels {
                let in_a = blob_a.iter().filter(|&&x| x == l).count();
                let in_b = blob_b.iter().filter(|&&x| x == l).count();
                let purity = in_a.max(in_b) as f32 / (in_a + in_b).max(1) as f32;
                assert!(
                    purity > 0.9,
                    "community {} purity {} (snn {}, cell_mediated {})",
                    l,
                    purity,
                    run_snn,
                    cell_mediated
                );
            }
        }
    }

    /// Stability metrics need seeds to compare, so a single Louvain seed is rejected up front.
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

        assert!(matches!(result, Err(BixverseErrors::InvalidArgument(_))));
    }

    /// The grid search is reproducible down to the ARI, conductance and best labels it reports.
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

    /// On easy data every resolution reports a high mean ARI and still splits the two blobs.
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

    /// Adjacency of a graph as sorted `(u, v, w)` triplets, for exact comparison.
    fn edge_list(g: &SparseGraph<f32>) -> Vec<(usize, usize, f32)> {
        let mut out = Vec::new();
        for u in 0..g.get_node_number() {
            let (nbrs, w) = g.get_neighbours(u);
            out.extend(nbrs.into_iter().zip(w).map(|(v, &x)| (u, v, x)));
        }
        out.sort_by_key(|e| (e.0, e.1));
        out
    }

    /// One cell between two anchors gives a single edge of weight
    /// `w1 w2 / (w1 + w2)`; the far third centroid stays isolated.
    #[test]
    fn cell_mediated_edge_weight_matches_hand_computation() {
        let data = Mat::from_fn(1, 1, |_, _| 0.0f32);
        let centroids = Mat::from_fn(3, 1, |i, _| [1.0f32, 3.0, 10.0][i]);

        let g = cell_mediated_graph(data.as_ref(), centroids.as_ref(), 2, "euclidean");

        // sigma = (1 + 3) / 2 = 2
        let w1 = (-1.0f32 / 8.0).exp();
        let w2 = (-9.0f32 / 8.0).exp();
        let expected = w1 * w2 / (w1 + w2);

        let edges = edge_list(&g);
        assert_eq!(edges.len(), 2, "one undirected edge, stored both ways");
        assert_eq!((edges[0].0, edges[0].1), (0, 1));
        assert_eq!((edges[1].0, edges[1].1), (1, 0));
        approx::assert_relative_eq!(edges[0].2, expected, max_relative = 1e-6);
        assert_eq!(g.get_node_degree(2), 0);
    }

    /// Centroids are only linked through cells that sit near both, so two
    /// blobs with no cells in between give no edge across the gap.
    #[test]
    fn cell_mediated_graph_has_no_edges_across_an_empty_gap() {
        let n_per = 200;
        let dim = 4;
        let data = make_two_blobs(n_per, dim, 15.0, 42);
        // two centroids per blob, nudged apart along the first axis
        let centroids = Mat::from_fn(4, dim, |i, j| {
            let centre = if i < 2 { 0.0 } else { 15.0 };
            let nudge = if j == 0 {
                [-0.1, 0.1, -0.1, 0.1][i]
            } else {
                0.0
            };
            centre + nudge
        });

        let g = cell_mediated_graph(data.as_ref(), centroids.as_ref(), 2, "euclidean");

        for (u, v, w) in edge_list(&g) {
            assert_eq!(u < 2, v < 2, "edge {u}-{v} crosses the gap");
            assert!(w > 0.0);
        }
        // and within each blob the two centroids are linked
        assert_eq!(g.get_neighbours(0).0, vec![1]);
        assert_eq!(g.get_neighbours(2).0, vec![3]);
    }

    /// Several accumulation blocks summed on 1 and on 4 threads give the
    /// bit-identical graph, so results do not depend on the thread count.
    #[test]
    fn cell_mediated_graph_independent_of_thread_count() {
        let n = 2 * CELL_MEDIATED_BLOCK + 1000;
        let dim = 4;
        let mut rng = StdRng::seed_from_u64(7);
        let data = Mat::from_fn(n, dim, |_, _| rng.random_range(-1.0f32..1.0));
        let centroids = Mat::from_fn(20, dim, |i, j| data[(i * 997, j)]);

        let build = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| cell_mediated_graph(data.as_ref(), centroids.as_ref(), 5, "euclidean"))
        };

        let one = edge_list(&build(1));
        let four = edge_list(&build(4));
        assert!(!one.is_empty());
        assert_eq!(one, four);
    }
}
