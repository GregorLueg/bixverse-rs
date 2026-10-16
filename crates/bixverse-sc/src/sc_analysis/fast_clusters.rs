//! Fast clustering on large data: k-means coarsening, a graph over the
//! centroids, Leiden or Louvain on that graph, and every cell inherits its
//! centroid's community (Lun's two-step clustering, as in bluster).
//!
//! The centroid graph is either a symmetrised kNN on the centroid coordinates
//! or a cell-mediated graph, where centroids are linked by the cells that sit
//! near both (`W = B^T D_x^-1 B`, the anchor graph of Wei et al.). Neither is
//! a spectral method: only the centroids are clustered.
//!
//! Stage one is behind the [`Coarsener`] trait, so the GPU and MLX crates swap
//! in their own k-means (and nearest-centroid search) and reuse the rest.

use ann_search_rs::utils::dist::{Dist, SimdDistance, parse_ann_dist};
use ann_search_rs::utils::k_means_utils::assign_all_parallel_top_m;
use ann_search_rs::utils::matrix_to_flat;
use faer::{Mat, MatRef};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::time::Instant;

use crate::prelude::*;
use crate::sc_processing::knn::generate_knn_with_dist;
use bixverse_core::core::math::vector_helpers::{mean_nan, median};
use bixverse_core::ml::clustering::clustering_metrics::adjusted_rand_index;
use bixverse_core::ml::clustering::k_means::*;
use bixverse_methods::graph::graph_metrics::conductance_undirected;
use bixverse_methods::graph::graph_structures::*;
use bixverse_methods::graph::knn_graph::*;
use bixverse_methods::graph::leiden::{LeidenParams, community_detection};

///////////
// Const //
///////////

/// Cells per accumulation block in [`cell_mediated_graph`]. Blocks are summed
/// in a fixed order so the edge weights do not depend on the thread count;
/// 16k cells keeps the per-block maps small while giving rayon enough blocks
/// at atlas scale.
const CELL_MEDIATED_BLOCK: usize = 16_384;

////////////////
// Coarsening //
////////////////

/// Stage one of fast clustering: k-means and the nearest-centroid search.
///
/// Implemented by [`CpuKMeans`] here and by the GPU and MLX k-means in their
/// own crates. The error type follows the backend seam pattern, so a backend
/// can surface its own failures without a variant in [`ScErrors`].
pub trait Coarsener {
    /// Error returned by the backend
    type Error: From<ScErrors> + From<CoreErrors> + Debug;

    /// k-means on the cells.
    ///
    /// ### Params
    ///
    /// * `data` - n_cells x n_features matrix
    /// * `n_centroids` - Number of centroids, already clamped below `n_cells`
    /// * `dist` - `"euclidean"` or `"cosine"`
    /// * `seed` - Seed for reproducibility
    /// * `verbose` - `0` silent, `1` normal, `2` detailed
    ///
    /// ### Returns
    ///
    /// `(centroids, n_centroids x n_features; centroid index per cell)`
    fn coarsen(
        &self,
        data: MatRef<f32>,
        n_centroids: usize,
        dist: &str,
        seed: usize,
        verbose: usize,
    ) -> Result<(Mat<f32>, Vec<usize>), Self::Error>;

    /// The `m` nearest centroids of every cell, for [`CentroidGraph::CellMediated`].
    ///
    /// The default is the exact CPU scan.
    ///
    /// ### Params
    ///
    /// * `data` - n_cells x n_features matrix
    /// * `centroids` - n_centroids x n_features matrix
    /// * `m` - Centroids per cell, at most `n_centroids`
    /// * `dist` - `"euclidean"` or `"cosine"`
    ///
    /// ### Returns
    ///
    /// `n_cells * m` centroid ids, row-major, ascending by distance
    fn top_m(
        &self,
        data: MatRef<f32>,
        centroids: MatRef<f32>,
        m: usize,
        dist: &str,
    ) -> Result<Vec<u32>, Self::Error> {
        Ok(top_m_cpu(data, centroids, m, dist))
    }
}

/// Exact CPU top-m nearest centroids per cell.
///
/// ### Params
///
/// * `data` - n_cells x n_features matrix
/// * `centroids` - n_centroids x n_features matrix
/// * `m` - Centroids per cell, at most `n_centroids`
/// * `dist` - `"euclidean"` or `"cosine"`
///
/// ### Returns
///
/// `n_cells * m` centroid ids, row-major, ascending by distance
pub fn top_m_cpu(data: MatRef<f32>, centroids: MatRef<f32>, m: usize, dist: &str) -> Vec<u32> {
    let (data_flat, n, dim) = matrix_to_flat(data);
    let (cent_flat, k, _) = matrix_to_flat(centroids);
    let metric = parse_ann_dist(dist).unwrap_or_default();
    let data_norms = compute_data_norms(&data_flat, dim, n, &metric);
    let cent_norms = if metric == Dist::Cosine {
        recompute_centroid_norms(&cent_flat, dim, k, &metric)
    } else {
        Vec::new()
    };
    assign_all_parallel_top_m(
        &data_flat,
        &data_norms,
        dim,
        n,
        &cent_flat,
        &cent_norms,
        k,
        m,
        &metric,
    )
}

/// Which CPU k-means to run.
#[derive(Clone, Copy, Debug, Default)]
pub enum CpuKMeansKind {
    /// Lloyd's / Hamerly k-means on all cells
    #[default]
    Standard,
    /// Mini-batch k-means
    MiniBatch {
        /// Cells per batch
        batch_size: usize,
        /// Stop once the centroids move less than this
        drift_threshold: f32,
        /// Learning rate exponent: `eta = m / count[c]^lr_alpha`
        lr_alpha: f32,
    },
}

/// CPU k-means coarsener.
#[derive(Clone, Debug, Default)]
pub struct CpuKMeans {
    /// Standard or mini-batch
    pub kind: CpuKMeansKind,
    /// k-means parameters; for mini-batch only `iters` is read
    pub params: KMeansParamsWrappers,
}

impl CpuKMeans {
    /// Generate a new CPU k-means coarsener.
    ///
    /// ### Params
    ///
    /// * `kind` - Standard or mini-batch
    /// * `params` - k-means parameters
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(kind: CpuKMeansKind, params: KMeansParamsWrappers) -> Self {
        Self { kind, params }
    }
}

impl Coarsener for CpuKMeans {
    type Error = ScErrors;

    fn coarsen(
        &self,
        data: MatRef<f32>,
        n_centroids: usize,
        dist: &str,
        seed: usize,
        verbose: usize,
    ) -> Result<(Mat<f32>, Vec<usize>), ScErrors> {
        let detailed = parse_verbosity_level(verbose).detailed_verbosity();
        Ok(match self.kind {
            CpuKMeansKind::Standard => {
                k_means_clusters(data, dist, n_centroids, Some(self.params), seed, detailed)?
            }
            CpuKMeansKind::MiniBatch {
                batch_size,
                drift_threshold,
                lr_alpha,
            } => train_centroids_minibatch(
                data,
                dist,
                n_centroids,
                self.params.get_data().iters,
                batch_size,
                drift_threshold,
                lr_alpha,
                seed,
                detailed,
            ),
        })
    }
}

////////////
// Params //
////////////

/// The graph built over the centroids.
#[derive(Clone, Copy, Debug)]
pub enum CentroidGraph {
    /// Exact kNN on the centroid coordinates, then symmetrised
    Knn {
        /// Neighbours per centroid
        k: usize,
        /// How the kNN becomes undirected
        sym: KnnSymmetrisation,
    },
    /// Centroids linked through the cells near both, see
    /// [`cell_mediated_graph`]
    CellMediated {
        /// Nearest centroids per cell
        n_anchors: usize,
    },
}

/// Default: 15-NN on the centroids with the default [`KnnSymmetrisation`]
impl Default for CentroidGraph {
    fn default() -> Self {
        CentroidGraph::Knn {
            k: 15,
            sym: KnnSymmetrisation::default(),
        }
    }
}

/// Parameters for fast clustering. The coarsening backend is passed
/// separately, see [`Coarsener`].
#[derive(Clone, Debug)]
pub struct FastClusterParams {
    /// Number of k-means centroids, clamped below the number of cells
    pub n_centroids: usize,
    /// Distance for the k-means and the centroid graph: `"euclidean"` or
    /// `"cosine"`
    pub dist: String,
    /// The centroid graph
    pub graph: CentroidGraph,
    /// Community detection. `refine = false` gives Louvain. `resolution` and
    /// `seed` are set per run by [`fast_clusters`].
    pub community: LeidenParams,
    /// Full multi-level run; `false` stops after the first local-moving phase
    pub multi_level: bool,
    /// Cap on node evaluations per local-moving phase, in units of the level's
    /// node count
    pub max_evals: usize,
}

/// Defaults: 1000 centroids, Euclidean, the default [`CentroidGraph`] and
/// multi-level Louvain with one iteration
impl Default for FastClusterParams {
    fn default() -> Self {
        Self {
            n_centroids: 1000,
            dist: "euclidean".to_string(),
            graph: CentroidGraph::default(),
            community: LeidenParams {
                n_iterations: Some(1),
                refine: false,
                ..LeidenParams::default()
            },
            multi_level: true,
            max_evals: 10,
        }
    }
}

/////////////
// Results //
/////////////

/// Stability and quality of one resolution across community-detection seeds.
#[derive(Clone, Debug)]
pub struct GridStats {
    /// Mean pairwise ARI across all seed pairs
    pub mean_ari: f32,
    /// Median pairwise ARI across all seed pairs
    pub median_ari: f32,
    /// Mean conductance across communities, averaged across seeds. NaN
    /// community values are skipped per seed.
    pub mean_conductance: f32,
    /// Median conductance across communities, averaged across seeds
    pub median_conductance: f32,
    /// Number of communities, averaged across seeds
    pub mean_n_communities: f32,
}

/// Fast clustering output for one resolution.
#[derive(Clone, Debug)]
pub struct ResolutionResult {
    /// The resolution
    pub resolution: f32,
    /// Community per cell. With several seeds, those of the seed with the
    /// lowest mean conductance.
    pub labels: Vec<usize>,
    /// `Some` when more than one seed was run
    pub stability: Option<GridStats>,
}

/// Fast clustering output.
#[derive(Clone, Debug)]
pub struct FastClusterResult {
    /// One entry per resolution, in the order given
    pub per_resolution: Vec<ResolutionResult>,
    /// The k-means centroids, n_centroids x n_features
    pub centroids: Mat<f32>,
    /// Centroid index per cell
    pub assignments: Vec<usize>,
}

////////////////////
// Centroid graph //
////////////////////

/// Build the centroid graph through the cells.
///
/// Every cell links to its `m` nearest centroids with a locally scaled
/// Gaussian weight `w_ij = exp(-d_ij^2 / (2 sigma_i^2))`, `sigma_i` being the
/// mean distance to those centroids. Centroids are then joined by
/// `W = B^T D_x^-1 B`, i.e. by the cells that sit near both, so two centroids
/// that are close in space but have no cells between them stay apart. The
/// diagonal of `W` is dropped, so no self-loops reach the community detection.
///
/// ### Params
///
/// * `data` - n_cells x n_features matrix the centroids were fitted on
/// * `centroids` - n_centroids x n_features matrix
/// * `anchors` - `n_cells * m` nearest centroid ids, row-major
/// * `m` - Anchors per cell
/// * `dist` - `"euclidean"` or `"cosine"`
///
/// ### Returns
///
/// The undirected centroid graph
///
/// ### References
///
/// Wei et al., PLoS Comput Biol, 2022
fn cell_mediated_graph(
    data: MatRef<f32>,
    centroids: MatRef<f32>,
    anchors: &[u32],
    m: usize,
    dist: &str,
) -> SparseGraph<f32> {
    let (data_flat, n, dim) = matrix_to_flat(data);
    let (cent_flat, k, _) = matrix_to_flat(centroids);
    let metric = parse_ann_dist(dist).unwrap_or_default();
    let (data_norms, cent_norms) = if metric == Dist::Cosine {
        (
            compute_data_norms(&data_flat, dim, n, &metric),
            recompute_centroid_norms(&cent_flat, dim, k, &metric),
        )
    } else {
        (Vec::new(), Vec::new())
    };

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
                    *wj = match metric {
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

/// Exact kNN graph over the centroids.
///
/// The centroids are few, so the search is always exhaustive.
///
/// ### Params
///
/// * `centroids` - n_centroids x n_features matrix
/// * `k` - Neighbours per centroid, clamped below `n_centroids`
/// * `sym` - How the kNN becomes undirected
/// * `dist` - `"euclidean"` or `"cosine"`
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The undirected centroid graph
fn centroid_knn_graph(
    centroids: MatRef<f32>,
    k: usize,
    sym: &KnnSymmetrisation,
    dist: &str,
    verbose: usize,
) -> Result<SparseGraph<f32>, ScErrors> {
    let n = centroids.nrows();
    let knn_params = KnnParams {
        knn_method: "exhaustive".to_string(),
        ann_dist: dist.to_string(),
        k: k.min(n.saturating_sub(1)),
        ..KnnParams::default()
    };
    let needs_dist = matches!(sym, KnnSymmetrisation::Umap { .. });
    let (knn, knn_dist) =
        generate_knn_with_dist(centroids, &knn_params, needs_dist, false, 0, false)?;
    let (idx, k) = flatten_knn(&knn);
    let flat_dist: Option<Vec<f32>> = knn_dist.map(|d| d.into_iter().flatten().collect());
    Ok(build_knn_graph(
        &idx,
        flat_dist.as_deref(),
        k,
        n,
        sym,
        verbose,
    )?)
}

/////////////////////////
// Community detection //
/////////////////////////

/// Community detection on the centroid graph at several seeds, with the
/// stability and quality of the partitions.
///
/// ARI and conductance are taken on the centroid partitions, which is cheaper
/// than the per-cell ones and ranks the resolutions identically. The winning
/// partition is the one with the lowest mean conductance.
///
/// ### Params
///
/// * `graph` - The centroid graph
/// * `params` - Community parameters with the resolution already set
/// * `max_evals` - Node-evaluation cap, see [`FastClusterParams`]
/// * `multi_level` - Full multi-level run or one local-moving phase
/// * `seeds` - One run per seed, at least two
///
/// ### Returns
///
/// `(winning centroid partition, stats)`
fn seed_grid(
    graph: &SparseGraph<f32>,
    params: &LeidenParams,
    max_evals: usize,
    multi_level: bool,
    seeds: &[u64],
) -> Result<(Vec<usize>, GridStats), ScErrors> {
    let mut parts: Vec<Vec<usize>> = seeds
        .iter()
        .map(|&s| {
            community_detection(
                graph,
                &LeidenParams { seed: s, ..*params },
                max_evals,
                multi_level,
            )
        })
        .collect::<Result<_, _>>()?;

    let s = parts.len();
    let mut aris: Vec<f64> = Vec::with_capacity(s * (s - 1) / 2);
    for i in 0..s {
        for j in (i + 1)..s {
            aris.push(adjusted_rand_index(&parts[i], &parts[j]));
        }
    }
    let mean_ari = aris.iter().sum::<f64>() / aris.len() as f64;
    let median_ari = median(&aris).unwrap_or(f64::NAN);

    let mut mean_cond = Vec::with_capacity(s);
    let mut median_cond = Vec::with_capacity(s);
    for p in &parts {
        let finite: Vec<f32> = conductance_undirected(graph, p)?
            .into_iter()
            .filter(|x| x.is_finite())
            .collect();
        mean_cond.push(mean_nan(&finite));
        median_cond.push(median(&finite).unwrap_or(f32::NAN));
    }

    let mean_n_communities = parts
        .iter()
        .map(|p| p.iter().max().copied().unwrap_or(0) + 1)
        .sum::<usize>() as f32
        / s as f32;

    // NaN counts as worst
    let best = mean_cond
        .iter()
        .map(|&c| if c.is_finite() { c } else { f32::INFINITY })
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i);

    let stats = GridStats {
        mean_ari: mean_ari as f32,
        median_ari: median_ari as f32,
        mean_conductance: mean_nan(&mean_cond),
        median_conductance: mean_nan(&median_cond),
        mean_n_communities,
    };
    Ok((parts.swap_remove(best), stats))
}

//////////
// Main //
//////////

/// Fast clustering via k-means coarsening.
///
/// k-means through `coarsener`, a graph over the centroids, then Leiden or
/// Louvain per resolution on that graph; every cell takes its centroid's
/// community. With `n_seeds > 1` each resolution runs once per seed and
/// reports [`GridStats`]; the seeds are drawn from `seed`.
///
/// ### Params
///
/// * `data` - n_cells x n_features matrix
/// * `coarsener` - The stage-one backend
/// * `params` - The [`FastClusterParams`]
/// * `resolutions` - Resolutions to run
/// * `n_seeds` - Community-detection seeds per resolution, at least 1
/// * `seed` - Seed for the k-means, and for the community detection when
///   `n_seeds == 1`
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`FastClusterResult`]
///
/// ### References
///
/// Lun, bluster (`clusterRows` with `TwoStepParam`), Bioconductor
pub fn fast_clusters<C: Coarsener>(
    data: MatRef<f32>,
    coarsener: &C,
    params: &FastClusterParams,
    resolutions: &[f32],
    n_seeds: usize,
    seed: usize,
    verbose: usize,
) -> Result<FastClusterResult, C::Error> {
    if n_seeds == 0 {
        return Err(CoreErrors::InvalidArgument("n_seeds must be at least 1".to_string()).into());
    }
    let verbosity = parse_verbosity_level(verbose);
    let n_centroids = params
        .n_centroids
        .min(data.nrows().saturating_sub(1))
        .max(1);

    let start = Instant::now();
    let (centroids, assignments) =
        coarsener.coarsen(data, n_centroids, &params.dist, seed, verbose)?;
    if verbosity.normal_verbosity() {
        println!(
            " k-means with {n_centroids} centroids: {:.2?}",
            start.elapsed()
        );
    }

    let start = Instant::now();
    let graph = match params.graph {
        CentroidGraph::Knn { k, sym } => {
            centroid_knn_graph(centroids.as_ref(), k, &sym, &params.dist, verbose)?
        }
        CentroidGraph::CellMediated { n_anchors } => {
            let m = n_anchors.clamp(1, centroids.nrows());
            let anchors = coarsener.top_m(data, centroids.as_ref(), m, &params.dist)?;
            cell_mediated_graph(data, centroids.as_ref(), &anchors, m, &params.dist)
        }
    };
    if verbosity.normal_verbosity() {
        println!(" Centroid graph: {:.2?}", start.elapsed());
    }

    let start = Instant::now();
    let seeds: Vec<u64> = if n_seeds == 1 {
        vec![seed as u64]
    } else {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        (0..n_seeds).map(|_| rng.random::<u64>()).collect()
    };
    let per_resolution = resolutions
        .iter()
        .map(|&res| {
            let p = LeidenParams {
                resolution: res as f64,
                ..params.community
            };
            let (part, stability) = if n_seeds == 1 {
                let part = community_detection(
                    &graph,
                    &LeidenParams {
                        seed: seeds[0],
                        ..p
                    },
                    params.max_evals,
                    params.multi_level,
                )
                .map_err(ScErrors::from)?;
                (part, None)
            } else {
                let (part, stats) =
                    seed_grid(&graph, &p, params.max_evals, params.multi_level, &seeds)?;
                (part, Some(stats))
            };
            Ok(ResolutionResult {
                resolution: res,
                labels: assignments.iter().map(|&c| part[c]).collect(),
                stability,
            })
        })
        .collect::<Result<Vec<_>, ScErrors>>()?;
    if verbosity.normal_verbosity() {
        println!(" Community detection: {:.2?}", start.elapsed());
    }

    Ok(FastClusterResult {
        per_resolution,
        centroids,
        assignments,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
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

    fn params_with(graph: CentroidGraph) -> FastClusterParams {
        FastClusterParams {
            n_centroids: 30,
            graph,
            ..FastClusterParams::default()
        }
    }

    fn knn_graph(sym: KnnSymmetrisation) -> CentroidGraph {
        CentroidGraph::Knn { k: 5, sym }
    }

    fn majority(xs: &[usize]) -> usize {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &x in xs {
            *counts.entry(x).or_insert(0) += 1;
        }
        *counts.iter().max_by_key(|(_, c)| **c).unwrap().0
    }

    fn run(
        data: &Mat<f32>,
        coarsener: &CpuKMeans,
        params: &FastClusterParams,
        resolutions: &[f32],
        n_seeds: usize,
        seed: usize,
    ) -> FastClusterResult {
        fast_clusters(
            data.as_ref(),
            coarsener,
            params,
            resolutions,
            n_seeds,
            seed,
            0,
        )
        .unwrap()
    }

    /// The same seed gives identical labels, so nothing downstream sees
    /// run-to-run drift.
    #[test]
    fn test_fast_clusters_deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = params_with(knn_graph(KnnSymmetrisation::Knn { binary: false }));
        let a = run(&data, &CpuKMeans::default(), &params, &[1.0], 1, 7);
        let b = run(&data, &CpuKMeans::default(), &params, &[1.0], 1, 7);
        assert_eq!(a.per_resolution[0].labels, b.per_resolution[0].labels);
        assert_eq!(a.assignments, b.assignments);
        assert!(a.per_resolution[0].stability.is_none());
    }

    /// Two well-separated blobs never share a community, for every centroid
    /// graph and for Louvain and Leiden. A blob may split into several
    /// communities: on the sNN centroid graph that is the higher-modularity
    /// answer (igraph splits it too).
    #[test]
    fn test_fast_clusters_separates_blobs_on_every_graph() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let graphs = [
            ("knn", knn_graph(KnnSymmetrisation::Knn { binary: false })),
            ("snn", knn_graph(KnnSymmetrisation::default())),
            (
                "umap",
                knn_graph(KnnSymmetrisation::Umap {
                    local_connectivity: 1.0,
                    mix_ratio: 1.0,
                }),
            ),
            (
                "cell_mediated",
                CentroidGraph::CellMediated { n_anchors: 5 },
            ),
        ];

        for (name, graph) in graphs {
            for refine in [false, true] {
                let mut params = params_with(graph);
                params.community.refine = refine;
                let res = run(&data, &CpuKMeans::default(), &params, &[1.0], 1, 0);
                let labels = &res.per_resolution[0].labels;
                assert_eq!(labels.len(), 2 * n_per);

                let (blob_a, blob_b) = labels.split_at(n_per);
                assert_ne!(majority(blob_a), majority(blob_b), "{name} refine {refine}");

                // every community is at least 90% one blob
                let n_labels = labels.iter().max().unwrap() + 1;
                for l in 0..n_labels {
                    let in_a = blob_a.iter().filter(|&&x| x == l).count();
                    let in_b = blob_b.iter().filter(|&&x| x == l).count();
                    let purity = in_a.max(in_b) as f32 / (in_a + in_b).max(1) as f32;
                    assert!(
                        purity > 0.9,
                        "{name} refine {refine}: community {l} purity {purity}"
                    );
                }
            }
        }
    }

    /// The mini-batch k-means path also separates the blobs.
    #[test]
    fn test_fast_clusters_minibatch_kmeans() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let coarsener = CpuKMeans::new(
            CpuKMeansKind::MiniBatch {
                batch_size: 100,
                drift_threshold: 1e-4,
                lr_alpha: 1.0,
            },
            KMeansParamsWrappers::default(),
        );
        let params = params_with(knn_graph(KnnSymmetrisation::Knn { binary: false }));
        let res = run(&data, &coarsener, &params, &[1.0], 1, 3);
        let (blob_a, blob_b) = res.per_resolution[0].labels.split_at(n_per);
        assert_ne!(majority(blob_a), majority(blob_b));
    }

    /// Zero seeds is an argument error, not a panic.
    #[test]
    fn test_fast_clusters_rejects_zero_seeds() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let result = fast_clusters(
            data.as_ref(),
            &CpuKMeans::default(),
            &FastClusterParams::default(),
            &[1.0],
            0,
            0,
            0,
        );
        assert!(matches!(
            result,
            Err(ScErrors::Core(CoreErrors::InvalidArgument(_)))
        ));
    }

    /// The seed grid is reproducible down to the ARI, conductance and labels
    /// it reports.
    #[test]
    fn test_fast_clusters_grid_deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = params_with(knn_graph(KnnSymmetrisation::Knn { binary: false }));
        let a = run(&data, &CpuKMeans::default(), &params, &[1.0, 0.5], 4, 7);
        let b = run(&data, &CpuKMeans::default(), &params, &[1.0, 0.5], 4, 7);
        for (ra, rb) in a.per_resolution.iter().zip(&b.per_resolution) {
            let (sa, sb) = (
                ra.stability.as_ref().unwrap(),
                rb.stability.as_ref().unwrap(),
            );
            assert_eq!(sa.mean_ari, sb.mean_ari);
            assert_eq!(sa.median_ari, sb.median_ari);
            assert_eq!(sa.mean_conductance, sb.mean_conductance);
            assert_eq!(sa.median_conductance, sb.median_conductance);
            assert_eq!(ra.labels, rb.labels);
        }
    }

    /// On easy data every resolution reports a high mean ARI and still splits
    /// the two blobs.
    #[test]
    fn test_fast_clusters_grid_well_separated_blobs_are_stable() {
        let n_per = 250;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let params = params_with(knn_graph(KnnSymmetrisation::Knn { binary: false }));
        let res = run(
            &data,
            &CpuKMeans::default(),
            &params,
            &[1.0, 0.5, 0.25],
            5,
            42,
        );
        for r in &res.per_resolution {
            let s = r.stability.as_ref().unwrap();
            assert!(
                s.mean_ari > 0.9,
                "mean_ari {} at {}",
                s.mean_ari,
                r.resolution
            );
            let (blob_a, blob_b) = r.labels.split_at(n_per);
            assert_ne!(majority(blob_a), majority(blob_b));
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

    /// The cell-mediated graph with its anchors from the CPU top-m.
    fn cell_mediated(data: MatRef<f32>, centroids: MatRef<f32>, m: usize) -> SparseGraph<f32> {
        let anchors = top_m_cpu(data, centroids, m, "euclidean");
        cell_mediated_graph(data, centroids, &anchors, m, "euclidean")
    }

    /// One cell between two anchors gives a single edge of weight
    /// `w1 w2 / (w1 + w2)`; the far third centroid stays isolated.
    #[test]
    fn test_cell_mediated_edge_weight_matches_hand_computation() {
        let data = Mat::from_fn(1, 1, |_, _| 0.0f32);
        let centroids = Mat::from_fn(3, 1, |i, _| [1.0f32, 3.0, 10.0][i]);

        let g = cell_mediated(data.as_ref(), centroids.as_ref(), 2);

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
    fn test_cell_mediated_graph_has_no_edges_across_an_empty_gap() {
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

        let g = cell_mediated(data.as_ref(), centroids.as_ref(), 2);

        for (u, v, w) in edge_list(&g) {
            assert_eq!(u < 2, v < 2, "edge {u}-{v} crosses the gap");
            assert!(w > 0.0);
        }
        assert_eq!(g.get_neighbours(0).0, vec![1]);
        assert_eq!(g.get_neighbours(2).0, vec![3]);
    }

    /// Several accumulation blocks summed on 1 and on 4 threads give the
    /// bit-identical graph, so results do not depend on the thread count.
    #[test]
    fn test_cell_mediated_graph_independent_of_thread_count() {
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
                .install(|| cell_mediated(data.as_ref(), centroids.as_ref(), 5))
        };

        let one = edge_list(&build(1));
        let four = edge_list(&build(4));
        assert!(!one.is_empty());
        assert_eq!(one, four);
    }

    /// The whole pipeline gives the same labels on 1 and on 4 threads.
    #[test]
    fn test_fast_clusters_independent_of_thread_count() {
        let data = make_two_blobs(300, 6, 4.0, 9);
        for graph in [
            knn_graph(KnnSymmetrisation::default()),
            CentroidGraph::CellMediated { n_anchors: 4 },
        ] {
            let mut params = params_with(graph);
            params.community.refine = true;
            params.community.n_iterations = Some(2);
            let build = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| run(&data, &CpuKMeans::default(), &params, &[0.8], 1, 5))
            };
            assert_eq!(
                build(1).per_resolution[0].labels,
                build(4).per_resolution[0].labels
            );
        }
    }
}
