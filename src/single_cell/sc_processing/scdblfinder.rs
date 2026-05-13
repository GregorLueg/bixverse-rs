//! The Rust implementation of a scDblFinder-like approach, see Germain et al.,
//! F1000Res, 2021. Under the hood this uses the internal implementation of
//! lightGBM.

use either::Either;
use faer::{Mat, MatRef};
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::core::math::pca_svd::{compute_pc_scores, randomised_svd};
use crate::graph::community_detections::*;
use crate::graph::graph_structures::*;
use crate::prelude::*;
use crate::single_cell::sc_analysis::fast_clusters::*;
use crate::single_cell::sc_processing::knn::generate_knn_with_dist;
use crate::single_cell::sc_processing::utils_doublets::*;
use crate::single_cell::sc_utils::{cxds::*, logistic_gbm::*, utils_tree::*};
use crate::utils::vec_utils::min_max_scale;

////////////////////////
// Params and results //
////////////////////////

/// Parameters for scDblFinder-style doublet detection.
///
/// Controls preprocessing, simulation, clustering, classification and
/// iterative refinement.
#[derive(Clone, Debug)]
pub struct ScDblFinderParams {
    // -- Preprocessing --
    /// Whether to log-transform counts after normalisation.
    pub log_transform: bool,
    /// Whether to mean-centre genes before PCA.
    pub mean_center: bool,
    /// Whether to scale genes to unit variance before PCA.
    pub normalise_variance: bool,
    /// Optional target library size. Defaults to the mean selected-gene library
    /// size.
    pub target_size: Option<f32>,
    /// Number of top-expressed genes to use as features. Matches R
    /// scDblFinder's `nfeatures=1352`.
    pub n_genes: usize,

    // -- Simulation --
    /// Ratio of simulated doublets to observed cells.
    pub doublet_ratio: f32,
    /// Fraction of pairs forced to be from different clusters (0.0-1.0).
    pub heterotypic_bias: f32,
    /// Parameters for the doublet simulation
    pub sim_params: ScDblSimParams,

    // -- PCA --
    /// Number of principal components.
    pub no_pcs: usize,
    /// Whether to use randomised SVD.
    pub random_svd: bool,

    // -- Clustering --
    /// Resolution for Louvain clustering.
    pub cluster_resolution: f32,
    /// Number of Louvain iterations per clustering step.
    pub cluster_iters: usize,
    /// Use fast clustering - useful on larger data sets.
    pub fast_cluster: bool,
    /// Which k-means clustering to use - standard or mini-batch
    pub km_type: String,
    /// [FastLouvainParams] for the fast clustering path. Gives control over
    /// the k-means methods, batch size, etc.
    pub fast_cluster_params: FastLouvainParams<f32>,

    // -- kNN --
    /// [KnnParams] for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,

    // -- Iteration --
    /// Number of refinement iterations (typically 2-3).
    pub n_iterations: usize,

    // -- Classification --
    /// Maximum number of boosting rounds.
    pub n_trees: usize,
    /// Maximum tree depth (shallow trees, 3-5, work best).
    pub max_depth: usize,
    /// Shrinkage applied to each tree's predictions.
    pub learning_rate: f32,
    /// Minimum training samples per leaf node.
    pub min_samples_leaf: usize,
    /// Fraction of samples used for training each tree.
    pub subsample_rate: f32,
    /// Number of CV folds for boosting round selection.
    pub cv_folds: usize,
    /// Early stopping patience per CV fold.
    pub cv_early_stop: usize,
    /// Multiplier on std for the SE rule (0.25 matches R's
    /// `nrounds=0.25`).
    pub se_fraction: f32,

    // -- Feature engineering --
    /// Number of leading PCs to include as classifier features.
    pub include_pcs: usize,
    /// Expected doublet rate
    pub expected_doublet_rate: Option<f32>,
    /// Return the per-cell feature matrix for inspection/debugging.
    pub return_features: bool,
    /// Number of cxds genes to take
    pub cxds_genes: Option<usize>,

    // -- Thresholding --
    /// Optional manual threshold. If `None`, cost-based optimisation is used.
    pub manual_threshold: Option<f32>,
}

impl Default for ScDblFinderParams {
    fn default() -> Self {
        Self {
            log_transform: true,
            mean_center: true,
            normalise_variance: true,
            target_size: None,
            n_genes: 1352,
            no_pcs: 30,
            random_svd: true,
            doublet_ratio: 1.0,
            heterotypic_bias: 0.8,
            sim_params: ScDblSimParams::default(),
            cluster_resolution: 1.0,
            cluster_iters: 10,
            fast_cluster: false,
            km_type: "minibatch".to_string(),
            fast_cluster_params: FastLouvainParams::default(),
            knn_params: KnnParams::default(),
            n_iterations: 3,
            n_trees: 200,
            max_depth: 4,
            learning_rate: 0.3,
            min_samples_leaf: 20,
            subsample_rate: 0.75,
            cv_folds: 5,
            cv_early_stop: 2,
            se_fraction: 0.25,
            include_pcs: 19,
            expected_doublet_rate: None,
            return_features: false,
            cxds_genes: None,
            manual_threshold: None,
        }
    }
}

///////////////////////
// Helper structures //
///////////////////////

/// Results from scDblFinder.
#[derive(Clone, Debug)]
pub struct ScDblFinderResult {
    /// Per-cell doublet predictions (`true` = doublet).
    pub predicted_doublets: Vec<bool>,
    /// Classifier probability per observed cell (0 = singlet, 1 = doublet).
    pub doublet_scores: Vec<f32>,
    /// The weighted scores
    pub weighted: Vec<f32>,
    /// CXDS scores
    pub cxds: Vec<f32>,
    /// Threshold used for doublet calling.
    pub threshold: f32,
    /// Cluster labels from the final iteration.
    pub cluster_labels: Vec<usize>,
    /// Fraction of observed cells called as doublets.
    pub detected_doublet_rate: f32,
    /// Indices of the selected genes for ScDblFinder.
    pub selected_genes: Vec<usize>,
    /// Features
    pub features: Option<FeatureTable>,
}

/// Per-cell kNN-derived intermediate values.
struct CellKnnFeatures {
    k_ratios: Vec<f32>,
    weighted: f32,
    dist_real: f32,
    origin: Option<(usize, usize)>,
}

/// Per-cell intermediate scores for diagnostics and comparison
/// against R's scDblFinder output.
///
/// All vectors are length `n_obs` (observed cells only).
#[derive(Clone, Debug)]
pub struct ScDblFinderDiagnostics {
    /// Distance-weighted kNN doublet proportion.
    pub weighted: Vec<f32>,
    /// Raw kNN ratio at the largest k value.
    pub ratio: Vec<f32>,
    /// Co-expression doublet score.
    pub cxds_score: Vec<f32>,
    /// Distance to the nearest real neighbour.
    pub dist_to_real: Vec<f32>,
    /// Identification difficulty (1 - mean class-weighted score for this cell's
    /// most likely doublet origin; 1.0 if no origin assigned).
    pub difficulty: Vec<f32>,
    /// Most likely parent cluster pair (canonical ordering); None for cells
    /// with no heterotypic sim neighbours.
    pub most_likely_origin: Vec<Option<(usize, usize)>>,
}

/// Classifier features for the observed samples.
#[derive(Clone, Debug)]
pub struct FeatureTable {
    /// Name of the features
    pub feature_names: Vec<String>,
    /// Table of the features of shape n_obs x n_features
    pub values: Mat<f32>,
    /// Included in training flag
    pub include_in_training: Vec<bool>,
}

///////////
// Types //
///////////

/// Tuple of (pairs, parent_cluster_labels) where pairs index into the
/// cell population and parent_cluster_labels record each pair's cluster
/// origins.
type ClusterAwarePairs = (Vec<(usize, usize)>, Vec<(usize, usize)>);

/////////////
// Helpers //
/////////////

/// Resolve the expected doublet rate.
///
/// If explicitly set, returns that value (clamped to [0.001, 0.5]).
/// Otherwise estimates from cell count using the 10x linear model.
///
/// ### Params
///
/// * `explicit` - Optional user-provided doublet rate.
/// * `n_cells` - Number of observed cells, used for the 10x linear estimate
///   when `explicit` is `None`.
///
/// ### Returns
///
/// Doublet rate based on user-provided data or the 10x linear model.
fn resolve_doublet_rate(explicit: Option<f32>, n_cells: usize) -> f32 {
    match explicit {
        Some(r) => r.clamp(0.001, 0.5),
        None => (0.008 * n_cells as f32 / 1000.0).clamp(0.001, 0.5),
    }
}

/// Compute per-origin mean classifier score over non-excluded sims.
///
/// ### Params
///
/// * `sim_scores` - Per-sim classifier (or proxy) score, length `n_sim`.
/// * `parent_clusters` - Parent cluster pair for each sim, length `n_sim`.
/// * `exclusion_mask` - If `true`, the sim is skipped, length `n_sim`.
///
/// ### Returns
///
/// Tuple of `(class_weighted_per_origin, mean_over_origins)`. The mean is
/// used as the fallback for cells with no assignable origin.
fn compute_class_weighted(
    sim_scores: &[f32],
    parent_clusters: &[(usize, usize)],
    exclusion_mask: &[bool],
) -> (FxHashMap<(usize, usize), f32>, f32) {
    let mut sums: FxHashMap<(usize, usize), (f64, u32)> = FxHashMap::default();
    for (si, &(ca, cb)) in parent_clusters.iter().enumerate() {
        if exclusion_mask[si] {
            continue;
        }
        if let Some(canon) = canonical_pair(ca, cb) {
            let e = sums.entry(canon).or_insert((0.0, 0));
            e.0 += sim_scores[si] as f64;
            e.1 += 1;
        }
    }
    let class_weighted: FxHashMap<(usize, usize), f32> = sums
        .into_iter()
        .map(|(k, (s, n))| (k, (s / n as f64) as f32))
        .collect();
    let mean = if class_weighted.is_empty() {
        0.5
    } else {
        class_weighted.values().copied().sum::<f32>() / class_weighted.len() as f32
    };
    (class_weighted, mean)
}

/// Compute per-cell difficulty for all cells (obs + sim).
///
/// R's formula: `difficulty = 1 - class_weighted[origin]`. Cells with no
/// assigned origin fall back to `1 - mean_class_weighted`.
///
/// ### Params
///
/// * `n_obs` - Number of observed cells.
/// * `n_total` - Total cells (`n_obs + n_sim`).
/// * `obs_origins` - Most-likely origin per observed cell, length `n_obs`.
/// * `parent_clusters` - Parent cluster pair per sim, length `n_sim`.
/// * `class_weighted` - Per-origin mean classifier score.
/// * `fallback_mean` - Mean class-weighted score; used when an origin has no
///   entry in `class_weighted`.
///
/// ### Returns
///
/// Per-cell difficulty, length `n_total` (observed rows first, then sims).
fn compute_difficulties(
    n_obs: usize,
    n_total: usize,
    obs_origins: &[Option<(usize, usize)>],
    parent_clusters: &[(usize, usize)],
    class_weighted: &FxHashMap<(usize, usize), f32>,
    fallback_mean: f32,
) -> Vec<f32> {
    let fallback = 1.0 - fallback_mean;
    (0..n_total)
        .map(|i| {
            let origin = if i < n_obs {
                obs_origins[i]
            } else {
                let (ca, cb) = parent_clusters[i - n_obs];
                canonical_pair(ca, cb)
            };
            match origin {
                Some(o) => class_weighted.get(&o).map_or(fallback, |&cd| 1.0 - cd),
                None => fallback,
            }
        })
        .collect()
}

/// Write a difficulty vector into the feature matrix in-place.
///
/// ### Params
///
/// * `features` - Mutable feature matrix of shape `(n_total, n_features)`.
/// * `difficulty_col` - Column index of the difficulty feature.
/// * `difficulties` - Per-cell difficulty values, length `n_total`.
fn update_difficulty_column(features: &mut Mat<f32>, difficulty_col: usize, difficulties: &[f32]) {
    for (i, &d) in difficulties.iter().enumerate() {
        *features.get_mut(i, difficulty_col) = d;
    }
}

/// Build feature name strings in the order they appear in the matrix.
///
/// ### Params
///
/// * `k_values` - The k values used for multi-scale kNN ratios.
/// * `n_pcs` - Number of leading PCs included as features.
///
/// ### Returns
///
/// Feature names in column order: `ratio.k{k}` per k, the seven fixed
/// features, then `PC{1..=n_pcs}`.
fn feature_names(k_values: &[usize], n_pcs: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(k_values.len() + 7 + n_pcs);
    for &k in k_values {
        names.push(format!("ratio.k{}", k));
    }
    names.extend(
        [
            "weighted",
            "distanceToNearestReal",
            "difficulty",
            "lsizes",
            "nfeatures",
            "nAbove2",
            "cxds_score",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    for pc in 0..n_pcs {
        names.push(format!("PC{}", pc + 1));
    }
    names
}

/// Extract the observed-cell slice of a combined (obs + sim) feature matrix
/// into a new owned `Mat`.
///
/// ### Params
///
/// * `features` - Combined feature matrix of shape `(n_total, n_feat)`.
/// * `n_obs` - Number of leading rows (observed cells) to copy.
///
/// ### Returns
///
/// Owned matrix containing the first `n_obs` rows of `features`.
fn slice_obs_rows(features: MatRef<f32>, n_obs: usize) -> Mat<f32> {
    let n_feat = features.ncols();
    Mat::<f32>::from_fn(n_obs, n_feat, |i, j| *features.get(i, j))
}

//////////////
// Features //
//////////////

/// Compute per-cell gene complexity statistics.
///
/// Returns `(n_features, n_above2)` where `n_features` is the number of genes
/// with non-zero expression and `n_above2` is the number of genes with
/// count > 2. These are computed over selected genes only.
///
/// ### Params
///
/// * `f_path_cell` - Path to the cell-based binary file (CSR format).
/// * `cells_to_keep` - Indices of the cells to keep
/// * `selected_genes` - The genes to include in this analysis
///
/// ### Returns
///
/// Tuple of `(non_zero, count_above_2)`
fn compute_cell_complexity(
    f_path_cell: &str,
    cells_to_keep: &[usize],
    selected_genes: &[usize],
) -> Result<(Vec<u32>, Vec<u32>), BixverseErrors> {
    let selected_gene_set: FxHashSet<usize> = selected_genes.iter().copied().collect();
    let reader = ParallelSparseReader::new(f_path_cell)?;
    let results: Vec<(u32, u32)> = cells_to_keep
        .par_iter()
        .map(|&cell_idx| -> Result<(u32, u32), BixverseErrors> {
            let chunk = reader.read_cell(cell_idx)?;
            let mut n_feat = 0u32;
            let mut n_above2 = 0u32;
            for (i, &gene_idx) in chunk.indices.iter().enumerate() {
                if selected_gene_set.contains(&(gene_idx as usize)) {
                    let count = chunk.data_raw.get(i);
                    if count > 0 {
                        n_feat += 1;
                    }
                    if count > 2 {
                        n_above2 += 1;
                    }
                }
            }
            Ok((n_feat, n_above2))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n_features: Vec<u32> = results.iter().map(|&(nf, _)| nf).collect();
    let n_above2: Vec<u32> = results.iter().map(|&(_, na)| na).collect();

    Ok((n_features, n_above2))
}

/// Compute gene complexity for simulated doublet chunks.
///
/// Since chunks already have HVG-remapped indices, we just count non-zero and
/// >2 entries directly.
///
/// ### Params
///
/// * `sim_chunks` - Slice of simulated `CsrCellChunks`
///
/// ### Returns
///
/// Tuple of `(non_zero, count_above_2)`
fn compute_sim_complexity(sim_chunks: &[CsrCellChunk]) -> (Vec<u32>, Vec<u32>) {
    let results: Vec<(u32, u32)> = sim_chunks
        .par_iter()
        .map(|chunk| {
            let mut n_feat = 0u32;
            let mut n_above2 = 0u32;
            for i in 0..chunk.indices.len() {
                let count = chunk.data_raw.get(i);
                if count > 0 {
                    n_feat += 1;
                }
                if count > 2 {
                    n_above2 += 1;
                }
            }
            (n_feat, n_above2)
        })
        .collect();

    let n_features: Vec<u32> = results.iter().map(|&(nf, _)| nf).collect();
    let n_above2: Vec<u32> = results.iter().map(|&(_, na)| na).collect();
    (n_features, n_above2)
}

/// Canonicalise a parent-cluster pair.
///
/// ### Params
///
/// * `c_a` - Cluster identifier `a`
/// * `c_b` - Cluster identifier `b`
///
/// ### Returns
///
/// Returns `None` for homotypic pairs (same cluster), matching R's behaviour
/// where homotypic doublets get NA origins.
#[inline]
fn canonical_origin(c_a: usize, c_b: usize) -> Option<(usize, usize)> {
    if c_a == c_b {
        None
    } else {
        Some((c_a.min(c_b), c_a.max(c_b)))
    }
}

/// Compute the most likely origin for a single cell from its kNN.
///
/// Mirrors R's `.getMostLikelyOrigins`:
///   1. Tabulate origins of artificial-doublet neighbours only.
///   2. If there's a unique most-frequent origin, return it.
///   3. If tied on count, break ties by minimum neighbour distance
///      within that origin group.
///
/// Returns `None` if no artificial heterotypic neighbours exist.
///
/// ### Params
///
/// * `neighbours` - Indices of the neighbours
/// * `distances` - Distances to their respective neighbours
/// * `n_obs` - Total number of observations
/// * `parent_clusters` - Slice of the parent clusters
///
/// ### Returns
///
/// The canonical `(c_a, c_b)` parent-cluster pair of the most common
/// artificial-doublet neighbour, with ties broken by minimum neighbour distance
/// (then by tuple ordering for determinism). `None` if the cell has no
/// heterotypic sim neighbours.
fn cell_most_likely_origin(
    neighbours: &[usize],
    distances: &[f32],
    n_obs: usize,
    parent_clusters: &[(usize, usize)],
) -> Option<(usize, usize)> {
    let mut counts: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    let mut min_dists: FxHashMap<(usize, usize), f32> = FxHashMap::default();

    for (&neigh, &dist) in neighbours.iter().zip(distances.iter()) {
        if neigh < n_obs {
            continue;
        }
        let (c_a, c_b) = parent_clusters[neigh - n_obs];
        if let Some(origin) = canonical_origin(c_a, c_b) {
            *counts.entry(origin).or_insert(0) += 1;
            let entry = min_dists.entry(origin).or_insert(f32::INFINITY);
            if dist < *entry {
                *entry = dist;
            }
        }
    }

    if counts.is_empty() {
        return None;
    }

    let max_count = *counts.values().max().unwrap();
    let mut candidates: Vec<(usize, usize)> = counts
        .iter()
        .filter(|&(_, &c)| c == max_count)
        .map(|(k, _)| *k)
        .collect();

    if candidates.len() == 1 {
        return Some(candidates[0]);
    }

    // Tiebreak by minimum distance, then by tuple (for determinism)
    candidates.sort_unstable_by(|a, b| {
        let da = min_dists[a];
        let db = min_dists[b];
        da.partial_cmp(&db)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    Some(candidates[0])
}

/// Distance from a cell to its nearest real-cell neighbour.
///
/// Falls back to `2 * fallback_max_dist` when no real neighbours exist in the
/// k-nearest set, matching R's `2*md` convention.
///
/// ### Params
///
/// * `neighbours` - Slice of neighbours
/// * `distances` - Slice of the distances to the neighbours
/// * `n_obs` - Number of observations
/// * `fallback_max_dist` - The fallback maximum distance
///
/// ### Returns
///
/// The distance to the nearest (real) cell neighbour
fn cell_distance_to_nearest_real(
    neighbours: &[usize],
    distances: &[f32],
    n_obs: usize,
    fallback_max_dist: f32,
) -> f32 {
    for (&neigh, &dist) in neighbours.iter().zip(distances.iter()) {
        if neigh < n_obs {
            return dist;
        }
    }
    2.0 * fallback_max_dist
}

/// Count cells assigned to each origin.
///
/// UNUSED FOR NOW!
///
/// ### Params
///
/// * `origins` - Most-likely origin per cell; `None` entries are skipped.
///
/// ### Returns
///
/// Map from canonical parent-cluster pair to its count.
#[allow(dead_code)]
fn aggregate_origin_counts(origins: &[Option<(usize, usize)>]) -> FxHashMap<(usize, usize), u32> {
    let mut counts: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    for o in origins.iter().flatten() {
        *counts.entry(*o).or_insert(0) += 1;
    }
    counts
}

/// PCA on the combined (observed + simulated) matrix.
///
/// Densifies both observed and simulated data into a single matrix,
/// computes column-wise mean/std, centres and scales, then runs
/// truncated SVD. Returns PC scores for all cells.
///
/// This matches R's scDblFinder which runs PCA on `cbind(counts, ad)`.
#[allow(clippy::too_many_arguments)]
pub fn pca_combined(
    f_path_cell: &str,
    cells_to_keep: &[usize],
    hvg_genes: &[usize],
    hvg_library_sizes: &[usize],
    target_size: f32,
    sim_chunks: &[CsrCellChunk],
    log_transform: bool,
    mean_center: bool,
    normalise_variance: bool,
    no_pcs: usize,
    seed: usize,
) -> Result<Mat<f32>, BixverseErrors> {
    let n_obs = cells_to_keep.len();
    let n_sim = sim_chunks.len();
    let n_total = n_obs + n_sim;
    let n_genes = hvg_genes.len();

    let gene_to_hvg: FxHashMap<usize, usize> = hvg_genes
        .iter()
        .enumerate()
        .map(|(pos, &g)| (g, pos))
        .collect();

    // build dense combined matrix
    let mut combined = Mat::<f64>::zeros(n_total, n_genes);

    // fill observed rows
    let reader = ParallelSparseReader::new(f_path_cell)?;
    for (row, &cell_idx) in cells_to_keep.iter().enumerate() {
        let chunk = reader.read_cell(cell_idx)?;
        let lib = hvg_library_sizes[row] as f64;
        for (i, &gene_idx) in chunk.indices.iter().enumerate() {
            let gi = gene_idx as usize;
            if let Some(&hvg_pos) = gene_to_hvg.get(&gi) {
                let raw = chunk.data_raw.get(i) as f64;
                let val = if log_transform {
                    ((raw / lib) * target_size as f64).ln_1p()
                } else {
                    (raw / lib) * target_size as f64
                };
                *combined.get_mut(row, hvg_pos) = val;
            }
        }
    }

    // fill simulated rows (already normalised and HVG-remapped)
    for (si, chunk) in sim_chunks.iter().enumerate() {
        let row = n_obs + si;
        for i in 0..chunk.indices.len() {
            let gene = chunk.indices[i] as usize;
            let val = chunk.data_norm[i].to_f32() as f64;
            *combined.get_mut(row, gene) = val;
        }
    }

    // Column means and stds
    let mut means = vec![0.0f64; n_genes];
    let mut vars = vec![0.0f64; n_genes];
    let nf = n_total as f64;

    for j in 0..n_genes {
        let mut sum = 0.0f64;
        for i in 0..n_total {
            sum += *combined.get(i, j);
        }
        means[j] = sum / nf;
    }

    for j in 0..n_genes {
        let mu = means[j];
        let mut ss = 0.0f64;
        for i in 0..n_total {
            let d = *combined.get(i, j) - mu;
            ss += d * d;
        }
        vars[j] = ss / (nf - 1.0);
    }

    // centre and scale in place
    for j in 0..n_genes {
        let mu = if mean_center { means[j] } else { 0.0 };
        let sd = if normalise_variance {
            let s = vars[j].sqrt();
            if s > 1e-10 { s } else { 1.0 }
        } else {
            1.0
        };
        for i in 0..n_total {
            let v = combined.get_mut(i, j);
            *v = (*v - mu) / sd;
        }
    }

    let svd_res = randomised_svd(combined.as_ref(), no_pcs, seed, None, None)?;

    let scores = compute_pc_scores(&svd_res);

    Ok(Mat::from_fn(scores.nrows(), scores.ncols(), |i, j| {
        scores[(i, j)] as f32
    }))
}

/// Compute default k values for multi-scale kNN doublet scoring.
///
/// Mirrors the R scDblFinder `.defaultKnnKs(k=NULL, n)` logic:
/// `kmax = max(ceil(sqrt(n/2)), 25)`, then
/// `unique(c(3,10,15,20,25,50,kmax)[<=kmax])`.
///
/// ### Params
///
/// * `n_obs` - Number of observations in the data set
///
/// ### Returns
///
/// The vectors of k-neighbours to consider in the classifier
fn default_knn_ks(n_obs: usize) -> Vec<usize> {
    let kmax = ((n_obs as f32 / 2.0).sqrt().ceil() as usize).max(25);
    let candidates = [3, 10, 15, 20, 25, 50, kmax];
    let mut ks: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&k| k <= kmax)
        .map(|k| k.min(MAX_DOUBLET_K_NEIGHBOURS))
        .collect();
    ks.sort_unstable();
    ks.dedup();
    ks
}

/// Build the feature matrix for the GBM classifier.
///
/// Features per cell (matching R scDblFinder's `.defTrainFeatures`):
///
/// 1. Multi-scale kNN ratios (one per k value)
/// 2. Distance-weighted doublet proportion
/// 3. Distance to nearest real cell
/// 4. Identification difficulty
/// 5. Raw library size
/// 6. Number of expressed genes
/// 7. Number of genes with count > 2
/// 8. Co-expression doublet score
/// 9. Principal components (n_pcs columns)
///
/// ### Params
///
/// * `knn_indices` - Per-cell neighbour indices into the combined (obs + sim)
///   population.
/// * `knn_distances` - Per-cell neighbour distances, aligned with
///   `knn_indices`.
/// * `n_obs` - Number of observed cells (sims start at index `n_obs`).
/// * `parent_clusters` - Parent cluster pair per sim.
/// * `k_values` - k values for the multi-scale ratio features.
/// * `obs_n_features` - Non-zero gene count per observed cell.
/// * `obs_n_above2` - Genes with count > 2 per observed cell.
/// * `sim_n_features` - Non-zero gene count per simulated doublet.
/// * `sim_n_above2` - Genes with count > 2 per simulated doublet.
/// * `library_sizes` - Per-observed-cell library size over selected genes.
/// * `sim_combined_lib_sizes` - Per-sim library size after combination.
/// * `obs_cxds` - CXDS score per observed cell.
/// * `sim_cxds` - CXDS score per simulated doublet.
/// * `combined_pca` - PC scores of shape `(n_total, n_pcs_total)`.
/// * `n_pcs` - Number of leading PCs to include as features.
///
/// ### Returns
///
/// Tuple `(features, knn_intermediates, difficulty_col)` where `features`
/// is the dense `(n_total, n_feat)` matrix, `knn_intermediates` holds
/// per-cell kNN-derived values for downstream use, and `difficulty_col` is
/// the column index of the difficulty feature.
#[allow(clippy::too_many_arguments)]
fn build_feature_matrix(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    n_obs: usize,
    parent_clusters: &[(usize, usize)],
    k_values: &[usize],
    obs_n_features: &[u32],
    obs_n_above2: &[u32],
    sim_n_features: &[u32],
    sim_n_above2: &[u32],
    library_sizes: &[usize],
    sim_combined_lib_sizes: &[usize],
    obs_cxds: &[f32],
    sim_cxds: &[f32],
    combined_pca: MatRef<f32>,
    n_pcs: usize,
) -> (Mat<f32>, Vec<CellKnnFeatures>, usize) {
    let n_total = knn_indices.len();
    let n_k = k_values.len();
    let n_feat = n_k + 7 + n_pcs;
    let difficulty_col = n_k + 2; // after k_ratios, weighted, dist_real

    let fallback_max_dist = knn_distances
        .iter()
        .map(|d| d.first().copied().unwrap_or(0.0))
        .fold(0.0f32, f32::max);

    let stage1: Vec<CellKnnFeatures> = (0..n_total)
        .into_par_iter()
        .map(|i| {
            let neighbours = &knn_indices[i];
            let distances = &knn_distances[i];
            let k_max = neighbours.len();

            // multi-scale ratios
            let mut k_ratios = Vec::with_capacity(n_k);
            for &k in k_values {
                let k_use = k.min(k_max);
                let n_sim_nb = neighbours[..k_use]
                    .iter()
                    .filter(|&&idx| idx >= n_obs)
                    .count();
                k_ratios.push(n_sim_nb as f32 / k_use as f32);
            }

            // distance-weighted doublet proportion
            let k_f = k_max as f32;
            let mut w_sum = 0.0f32;
            let mut w_dbl = 0.0f32;
            for (rank, (&neigh, &dist)) in neighbours.iter().zip(distances.iter()).enumerate() {
                let d = dist.max(1e-10);
                let w = (k_f - rank as f32 - 1.0).max(0.0).sqrt() / d;
                w_sum += w;
                if neigh >= n_obs {
                    w_dbl += w;
                }
            }
            let weighted = if w_sum > 0.0 { w_dbl / w_sum } else { 0.0 };

            let dist_real =
                cell_distance_to_nearest_real(neighbours, distances, n_obs, fallback_max_dist);

            let origin = cell_most_likely_origin(neighbours, distances, n_obs, parent_clusters);

            CellKnnFeatures {
                k_ratios,
                weighted,
                dist_real,
                origin,
            }
        })
        .collect();

    // Initial difficulty: use `weighted` as the score proxy (no classifier yet).
    // No permanent sim exclusions on the first build.
    let weighted_arr: Vec<f32> = stage1.iter().map(|c| c.weighted).collect();
    let obs_origins: Vec<Option<(usize, usize)>> =
        stage1.iter().take(n_obs).map(|c| c.origin).collect();

    let no_exclusions = vec![false; parent_clusters.len()];
    let (class_weighted, mean_cw) =
        compute_class_weighted(&weighted_arr[n_obs..], parent_clusters, &no_exclusions);
    let difficulties = compute_difficulties(
        n_obs,
        n_total,
        &obs_origins,
        parent_clusters,
        &class_weighted,
        mean_cw,
    );

    let rows: Vec<Vec<f32>> = (0..n_total)
        .into_par_iter()
        .map(|i| {
            let int = &stage1[i];
            let mut feats = Vec::with_capacity(n_feat);

            // k ratios
            feats.extend_from_slice(&int.k_ratios);

            // weighted
            feats.push(int.weighted);

            // distance to nearest real cell
            feats.push(int.dist_real);

            // difficulty
            feats.push(difficulties[i]);

            // raw library size (matching R's lsizes = colSums(e))
            let lib = if i < n_obs {
                library_sizes[i] as f32
            } else {
                sim_combined_lib_sizes[i - n_obs] as f32
            };
            feats.push(lib);

            // nfeatures, nAbove2
            if i < n_obs {
                feats.push(obs_n_features[i] as f32);
                feats.push(obs_n_above2[i] as f32);
            } else {
                let si = i - n_obs;
                feats.push(sim_n_features[si] as f32);
                feats.push(sim_n_above2[si] as f32);
            }

            // cxds
            if i < n_obs {
                feats.push(obs_cxds[i]);
            } else {
                feats.push(sim_cxds[i - n_obs]);
            }

            // PCs
            for pc in 0..n_pcs {
                feats.push(*combined_pca.get(i, pc));
            }

            feats
        })
        .collect();

    let mut features = Mat::<f32>::zeros(n_total, n_feat);
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            *features.get_mut(i, j) = val;
        }
    }

    (features, stage1, difficulty_col)
}

/// Generate cell pairs with cluster-aware heterotypic bias.
///
/// A fraction `heterotypic_bias` of pairs are forced to come from different
/// clusters; the remainder are drawn uniformly at random.
///
/// ### Params
///
/// * `cluster_labels` - Cluster assignment per observed cell.
/// * `n_sim` - Number of pairs to generate.
/// * `heterotypic_bias` - Fraction of pairs forced to be heterotypic
///   (0.0-1.0).
/// * `seed` - Seed for reproducibility.
///
/// ### Returns
///
/// The `ClusterAwarePairs`
fn cluster_aware_pairs(
    cluster_labels: &[usize],
    n_sim: usize,
    heterotypic_bias: f32,
    seed: usize,
) -> ClusterAwarePairs {
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let n_cells = cluster_labels.len();

    let mut cluster_to_cells: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (pos, &cl) in cluster_labels.iter().enumerate() {
        cluster_to_cells.entry(cl).or_default().push(pos);
    }
    let cluster_ids: Vec<usize> = cluster_to_cells.keys().copied().collect();
    let have_multiple_clusters = cluster_ids.len() >= 2;

    let n_heterotypic = (n_sim as f32 * heterotypic_bias) as usize;
    let n_random = n_sim - n_heterotypic;

    let mut pairs = Vec::with_capacity(n_sim);
    let mut parent_clusters = Vec::with_capacity(n_sim);

    // Heterotypic: enumerate unordered cluster pairs, weight by s_A * s_B
    if have_multiple_clusters && n_heterotypic > 0 {
        let mut cluster_pairs: Vec<(usize, usize)> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for i in 0..cluster_ids.len() {
            for j in (i + 1)..cluster_ids.len() {
                let ca = cluster_ids[i];
                let cb = cluster_ids[j];
                let sa = cluster_to_cells[&ca].len() as f64;
                let sb = cluster_to_cells[&cb].len() as f64;
                cluster_pairs.push((ca, cb));
                weights.push(sa * sb);
            }
        }
        let dist = WeightedIndex::new(&weights).expect("invalid cluster-pair weights");

        for _ in 0..n_heterotypic {
            let (ca, cb) = cluster_pairs[dist.sample(&mut rng)];
            let cells_a = &cluster_to_cells[&ca];
            let cells_b = &cluster_to_cells[&cb];
            let i = cells_a[rng.random_range(0..cells_a.len())];
            let j = cells_b[rng.random_range(0..cells_b.len())];
            pairs.push((i, j));
            parent_clusters.push((ca, cb));
        }
    } else {
        for _ in 0..n_heterotypic {
            let i = rng.random_range(0..n_cells);
            let j = rng.random_range(0..n_cells);
            pairs.push((i, j));
            parent_clusters.push((cluster_labels[i], cluster_labels[j]));
        }
    }

    for _ in 0..n_random {
        let i = rng.random_range(0..n_cells);
        let j = rng.random_range(0..n_cells);
        pairs.push((i, j));
        parent_clusters.push((cluster_labels[i], cluster_labels[j]));
    }

    (pairs, parent_clusters)
}

////////////////////
// Classification //
////////////////////

/// Train the logistic GBM classifier and return per-cell predicted
/// doublet probabilities for all samples.
///
/// Wraps `fit_logistic_gbm` with the appropriate label and exclusion
/// setup for the scDblFinder pipeline.
///
/// ### Params
///
/// * `features` - Dense feature matrix `(n_total, n_features)`.
/// * `labels` - Per-sample boolean labels; `true` = simulated
///   doublet, `false` = observed cell.
/// * `exclude` - Per-sample exclusion mask; `true` means excluded
///   from training but still scored.
/// * `config` - Logistic GBM configuration.
/// * `seed` - Random seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
///
/// ### Returns
///
/// Per-sample predicted probabilities, length `n_total`.
fn classify_doublets(
    features: MatRef<f32>,
    labels: &[bool],
    exclude: &[bool],
    config: &LogisticGbmConfig,
    seed: u64,
    verbose: bool,
) -> Vec<f32> {
    let store = QuantisedStore::from_mat(features);
    fit_logistic_gbm(&store, labels, exclude, config, seed, verbose)
}

/// Cost-based threshold optimisation matching R's `doubletThresholding`.
///
/// Minimises over `[0, 1]`:
///   cost = deviation² + 2*(1-stringency)*FNR + 2*stringency*FPR
///
/// where deviation is zero within the uncertainty band
/// `[dbr - dbr_sd, dbr + dbr_sd]`, and FNR/FPR are computed on simulated /
/// observed cells respectively.
///
/// ### Params
///
/// * `obs_scores` - Classifier probabilities for observed cells.
/// * `sim_scores` - Classifier probabilities for simulated doublets.
/// * `expected_dbr` - Expected doublet rate; sets the tolerance band.
/// * `stringency` - Trade-off in `[0, 1]`: higher penalises FPR more, lower
///   penalises FNR more.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
///
/// ### Returns
///
/// Threshold in `[0, 1]` minimising the cost function over up to 500
/// thinned candidate cuts.
fn find_threshold_optimised(
    obs_scores: &[f32],
    sim_scores: &[f32],
    expected_dbr: f32,
    stringency: f32,
    verbose: usize,
) -> f32 {
    let verbosity = parse_verbosity_level(verbose);

    let n_obs = obs_scores.len();
    let n_sim = sim_scores.len();

    let dbr_sd = 0.3 * expected_dbr + 0.025;
    let expected_lo = ((expected_dbr - dbr_sd).max(0.0) * n_obs as f32) + 1.0;
    let expected_hi = ((expected_dbr + dbr_sd).min(1.0) * n_obs as f32) + 1.0;

    let mut candidates: Vec<f32> = Vec::with_capacity(n_obs + n_sim);
    candidates.extend_from_slice(obs_scores);
    candidates.extend_from_slice(sim_scores);
    candidates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    candidates.dedup();

    // thin to at most 500 candidates for speed
    let step = (candidates.len() / 500).max(1);
    let candidates: Vec<f32> = candidates.into_iter().step_by(step).collect();

    let mut best_threshold = 0.5f32;
    let mut best_cost = f32::INFINITY;

    for &th in &candidates {
        // observed cells above threshold (called as doublets)
        let n_obs_above = obs_scores.iter().filter(|&&s| s >= th).count() as f32;

        // deviation from expected count (zero within tolerance band)
        let dev = if n_obs_above >= expected_lo && n_obs_above <= expected_hi {
            0.0
        } else {
            let mid = (expected_lo + expected_hi) / 2.0;
            ((n_obs_above - mid).abs() / mid.max(1.0)).powi(2)
        };

        // FNR: fraction of simulated doublets scored below threshold
        let n_sim_below = sim_scores.iter().filter(|&&s| s < th).count();
        let fnr = n_sim_below as f32 / n_sim.max(1) as f32;

        // FPR: fraction of observed cells scored above threshold
        let fpr = n_obs_above / n_obs.max(1) as f32;

        let cost = dev + 2.0 * (1.0 - stringency) * fnr + 2.0 * stringency * fpr;

        if cost < best_cost {
            best_cost = cost;
            best_threshold = th;
        }
    }

    if verbosity.normal_verbosity() {
        // diagnostic messages
        println!("Threshold optimiser:");
        println!(
            "  expected band: [{:.0}, {:.0}] doublets",
            expected_lo - 1.0,
            expected_hi - 1.0
        );
        println!("  selected threshold: {:.4}", best_threshold);
        println!("  best cost: {:.4}", best_cost);
    }

    best_threshold
}

////////////////////
// Main structure //
////////////////////

/// scDblFinder-style doublet detection.
///
/// Combines cluster-aware doublet simulation, engineered features, and
/// gradient-boosted classification with iterative refinement of cluster
/// assignments.
#[derive(Clone, Debug)]
pub struct ScDblFinder {
    /// Path to the gene-based binary file (CSC format).
    f_path_gene: String,
    /// Path to the cell-based binary file (CSR format).
    f_path_cell: String,
    /// Algorithm parameters.
    params: ScDblFinderParams,
    /// Number of observed cells.
    n_cells: usize,
    /// Number of simulated doublets per iteration.
    n_cells_sim: usize,
    /// Cell indices included in this analysis.
    cells_to_keep: Vec<usize>,
    /// Per-cell library sizes computed over selected genes only.
    red_library_sizes: Vec<usize>,
}

impl ScDblFinder {
    /// Create a new ScDblFinder instance.
    ///
    /// ### Params
    ///
    /// * `f_path_gene` - Path to the gene-based binary file (CSC).
    /// * `f_path_cell` - Path to the cell-based binary file (CSR).
    /// * `params` - ScDblFinder parameters.
    /// * `cell_indices` - Cell indices to include in the analysis.
    ///
    /// ### Returns
    ///
    /// Initialised `ScDblFinder`.
    pub fn new(
        f_path_gene: &str,
        f_path_cell: &str,
        params: ScDblFinderParams,
        cell_indices: &[usize],
    ) -> Self {
        Self {
            f_path_gene: f_path_gene.to_string(),
            f_path_cell: f_path_cell.to_string(),
            params,
            n_cells: cell_indices.len(),
            n_cells_sim: 0,
            cells_to_keep: cell_indices.to_vec(),
            red_library_sizes: Vec::new(),
        }
    }

    /// Run the full scDblFinder pipeline.
    ///
    /// ### Algorithm
    ///
    /// 1. Select top expressed genes -> library size calculations on these.
    /// 2. Run PCA on observed cells.
    /// 3. Build kNN on observed cells, cluster via Louvain.
    /// 4. **Iteration loop** (typically 2-3 rounds):
    ///    a. Simulate doublets with cluster-aware pairing.
    ///    b. Project simulated doublets into PC space.
    ///    c. Build kNN on combined (obs + sim) cells.
    ///    d. Assign cluster labels to simulated cells via majority vote.
    ///    e. Engineer feature matrix from kNN + cluster info.
    ///    f. Train GBM classifier (label: 0 = observed, 1 = simulated).
    ///    g. Score observed cells; apply sigmoid for probabilities.
    ///    h. Re-cluster observed cells for the next iteration.
    /// 5. Threshold final scores via Otsu (or manual).
    ///
    /// ### Params
    ///
    /// * `seed` - Seed for reproducibility.
    /// * `streaming` - Boolean if the genes shall be streamed in.
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// `ScDblFinderResult` with predictions, scores and cluster labels.
    pub fn run(
        &mut self,
        seed: usize,
        streaming: bool,
        verbose: usize,
    ) -> Result<ScDblFinderResult, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let start_all = Instant::now();

        if verbosity.normal_verbosity() {
            println!("Running scDblFinder in Rust");
            println!(" Selecting top-expressed genes...");
        }
        let start_sel = Instant::now();

        let selected_genes = if streaming {
            select_top_genes_streaming(
                &self.f_path_gene,
                &self.cells_to_keep,
                None,
                self.params.n_genes,
                verbosity.detailed_verbosity(),
            )?
        } else {
            select_top_genes(
                &self.f_path_gene,
                &self.cells_to_keep,
                None,
                self.params.n_genes,
            )?
        };

        if verbosity.normal_verbosity() {
            println!(
                " Using {} top-expressed genes. Done in {:.2?}",
                selected_genes.len(),
                start_sel.elapsed()
            );
        }

        // recycling function from scrublet/doublet detection
        self.red_library_sizes =
            compute_hvg_library_sizes(&self.f_path_cell, &self.cells_to_keep, &selected_genes)?;
        let target_size = resolve_target_size(self.params.target_size, &self.red_library_sizes);
        self.n_cells_sim = (self.n_cells as f32 * self.params.doublet_ratio) as usize;

        if verbosity.normal_verbosity() {
            println!(" Computing cell complexity features...");
        }
        let (obs_n_features, obs_n_above2) =
            compute_cell_complexity(&self.f_path_cell, &self.cells_to_keep, &selected_genes)?;

        let n_cxds = self.params.cxds_genes.unwrap_or(CXDS_NTOP);
        if verbosity.normal_verbosity() {
            println!(
                " Building cxds co-expression model with {} genes...",
                n_cxds
            );
        }

        let start_cxds = Instant::now();
        let (cxds_model, obs_cxds_gene_sets) = CxdsModel::fit(
            &self.f_path_cell,
            &self.cells_to_keep,
            &selected_genes,
            n_cxds,
        )?;
        let obs_cxds_scores = cxds_model.score(&obs_cxds_gene_sets);

        if verbosity.normal_verbosity() {
            let mean_cxds: f32 = obs_cxds_scores.iter().sum::<f32>() / obs_cxds_scores.len() as f32;
            println!(
                "  cxds model built in {:.2?} (mean score: {:.2})",
                start_cxds.elapsed(),
                mean_cxds,
            );
        }

        if verbosity.normal_verbosity() {
            println!(" Running PCA on observed cells...");
        }

        let (obs_pca, _, _, _) = pca_observed(
            &self.f_path_gene,
            &self.cells_to_keep,
            &selected_genes,
            &self.red_library_sizes,
            target_size,
            self.params.log_transform,
            self.params.mean_center,
            self.params.normalise_variance,
            self.params.no_pcs,
            self.params.random_svd,
            seed,
            verbose,
        )?;

        if verbosity.normal_verbosity() {
            println!(" Initial clustering of observed cells...");
        }
        let start_clustering = Instant::now();

        let use_fast = self.params.fast_cluster && self.n_cells >= FAST_CLUSTER_MIN_CELLS;

        let cluster_labels = if use_fast {
            let n_centroids = self.params.fast_cluster_params.n_centroids;
            let centroid_k = ((n_centroids as f32).sqrt() * 0.5).round() as usize;

            let mut centroid_knn_params = self.params.knn_params.clone();
            centroid_knn_params.k = centroid_k;

            // extract some (but not all...)
            let fast_params = FastLouvainParams {
                n_centroids,
                knn_params: centroid_knn_params,
                louvain_iters: self.params.cluster_iters,
                batch_size: self.params.fast_cluster_params.batch_size,
                kmeans_iters: self.params.fast_cluster_params.kmeans_iters,
                ..Default::default()
            };
            if verbosity.normal_verbosity() {
                println!(
                    "  Using fast clustering with {} centroids, k={} on {} cells.",
                    n_centroids, centroid_k, self.n_cells
                );
            }

            let res = vec![self.params.cluster_resolution];

            let res = fast_louvain_clusters(
                obs_pca.as_ref(),
                &self.params.km_type,
                &res,
                &fast_params,
                false,
                false,
                seed,
                verbose,
            )?;

            // quick helper to extract the assignments
            let get_assignments = |x: FastLouvainResults| -> Vec<Vec<usize>> {
                match x.get_assignments() {
                    Either::Left(v) => v,
                    Either::Right(_) => panic!("expected Single variant"),
                }
            };

            get_assignments(res)[0].clone()
        } else {
            let obs_k = if self.params.knn_params.k == 0 {
                (((self.n_cells as f32).sqrt() * 0.5).round() as usize)
                    .max(MAX_DOUBLET_K_NEIGHBOURS)
            } else {
                self.params.knn_params.k
            };

            let obs_knn = dispatch_knn(
                obs_pca.as_ref(),
                obs_k,
                &self.params.knn_params,
                seed,
                verbosity.detailed_verbosity(),
            );

            let obs_graph = knn_to_sparse_graph(&obs_knn, true);
            louvain_sparse_graph(
                &obs_graph,
                self.params.cluster_resolution,
                self.params.cluster_iters,
                true,
                seed,
            )?
        };

        let end_clustering = start_clustering.elapsed();

        if verbosity.normal_verbosity() {
            println!(
                "  Finished initial kNN generation and clustering in {:.2?}",
                end_clustering
            );
        }

        let k_values = default_knn_ks(self.n_cells);
        let k_max = *k_values.last().unwrap();
        let n_k = k_values.len();
        let n_pcs_include = self.params.include_pcs.min(self.params.no_pcs);

        if verbosity.normal_verbosity() {
            println!(
                " Using k values {:?} (max {}), including {} PCs as features",
                k_values, k_max, n_pcs_include
            );
        }

        let expected_dbr = resolve_doublet_rate(self.params.expected_doublet_rate, self.n_cells);

        if verbosity.normal_verbosity() {
            println!(
                " Simulating cluster-aware doublets with a heterotypic bias of {:.2}...",
                self.params.heterotypic_bias
            );
        }

        let start_sim = Instant::now();

        let (pairs, parent_clusters) = cluster_aware_pairs(
            &cluster_labels,
            self.n_cells_sim,
            self.params.heterotypic_bias,
            seed,
        );

        let (sim_chunks, sim_combined_lib_sizes) = simulate_doublets_scdbl(
            &pairs,
            &self.cells_to_keep,
            &self.red_library_sizes,
            &cluster_labels,
            &selected_genes,
            &self.f_path_cell,
            target_size,
            self.params.log_transform,
            &self.params.sim_params,
            seed,
        )?;

        let (sim_n_features, sim_n_above2) = compute_sim_complexity(&sim_chunks);

        let selected_genes_to_cxds = cxds_model.hvg_position_map(&selected_genes);
        let sim_cxds_scores = cxds_model.score_simulated_from_chunks(
            &sim_chunks,
            &selected_genes_to_cxds,
            1, // bin_thresh: match what fit() uses (raw count > 0)
        );

        if verbosity.normal_verbosity() {
            println!(" Running PCA across all observed and synthetic cells...");
        }
        let combined_pca = pca_combined(
            &self.f_path_cell,
            &self.cells_to_keep,
            &selected_genes,
            &self.red_library_sizes,
            target_size,
            &sim_chunks,
            self.params.log_transform,
            self.params.mean_center,
            self.params.normalise_variance,
            self.params.no_pcs,
            seed,
        )?;

        if verbosity.normal_verbosity() {
            println!(" Building combined kNN graph (k_max={})...", k_max);
        }

        let mut knn_params = self.params.knn_params.clone();
        knn_params.k = k_max;

        let (combined_knn, combined_dists) = generate_knn_with_dist(
            combined_pca.as_ref(),
            &knn_params,
            true,
            false,
            seed,
            verbosity.detailed_verbosity(),
        );

        let mut combined_dists = combined_dists.unwrap();

        // ann-search-rs returns squared Euclideans
        if knn_params.ann_dist == "euclidean" {
            combined_dists.par_iter_mut().for_each(|row| {
                for d in row.iter_mut() {
                    *d = d.max(0.0).sqrt();
                }
            });
        }

        let end_sim = start_sim.elapsed();

        // build the feature matrix
        let n_feat = n_k + 7 + n_pcs_include;
        if verbosity.normal_verbosity() {
            println!(
                "  Finished generation of simulated doublets, projection on PCA space and kNN generation in {:.2?}",
                end_sim
            );
            println!(" Building feature matrix ({} features)...", n_feat);
        }

        let (mut features, knn_intermediates, difficulty_col) = build_feature_matrix(
            &combined_knn,
            &combined_dists,
            self.n_cells,
            &parent_clusters,
            &k_values,
            &obs_n_features,
            &obs_n_above2,
            &sim_n_features,
            &sim_n_above2,
            &self.red_library_sizes,
            &sim_combined_lib_sizes,
            &obs_cxds_scores,
            &sim_cxds_scores,
            combined_pca.as_ref(),
            n_pcs_include,
        );

        let obs_intermediates = &knn_intermediates[..self.n_cells];
        let n_total = self.n_cells + self.n_cells_sim;

        // labels: false = observed (singlet), true = simulated (doublet)
        let labels: Vec<bool> = (0..n_total).map(|i| i >= self.n_cells).collect();

        // -- trainings loop --

        // initial scores: (cxds_norm + ratio_norm) / 2, matching R here
        let max_k_col = n_k - 1;

        let max_ratio = (0..n_total)
            .map(|i| *features.get(i, max_k_col))
            .fold(0.0f32, f32::max)
            .max(1e-10);

        let max_cxds = obs_cxds_scores
            .iter()
            .chain(sim_cxds_scores.iter())
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-10);

        let mut final_scores: Vec<f32> = (0..self.n_cells)
            .map(|i| {
                let ratio_norm = *features.get(i, max_k_col) / max_ratio;
                let cxds_norm = obs_cxds_scores[i] / max_cxds;
                (ratio_norm + cxds_norm) / 2.0
            })
            .collect();

        let mut sim_scores: Vec<f32> = (0..self.n_cells_sim)
            .map(|si| {
                let ratio_norm = *features.get(self.n_cells + si, max_k_col) / max_ratio;
                let cxds_norm = sim_cxds_scores[si] / max_cxds;
                (ratio_norm + cxds_norm) / 2.0
            })
            .collect();

        // classifier config
        let gbm_config = LogisticGbmConfig {
            max_rounds: self.params.n_trees,
            learning_rate: self.params.learning_rate,
            max_depth: self.params.max_depth,
            min_samples_leaf: self.params.min_samples_leaf,
            subsample_rate: self.params.subsample_rate,
            n_folds: self.params.cv_folds,
            cv_early_stop: self.params.cv_early_stop,
            se_fraction: self.params.se_fraction,
            ..Default::default()
        };

        // permanent sim exclusions from .filterUnrecognizableDoublets -> set
        // once after iteration 0 and carried through all subsequent iterations
        let mut permanent_sim_exclusion = vec![false; self.n_cells_sim];

        for iter in 0..self.params.n_iterations {
            if verbosity.normal_verbosity() {
                println!(
                    " == Iteration {} of {} ==",
                    iter + 1,
                    self.params.n_iterations
                );
            }
            let start_iter = Instant::now();
            let iter_seed = seed + iter * 1000;

            // build exclusion mask: permanent sim exclusions + per-iter additions
            let mut exclude_mask = vec![false; n_total];
            for (si, &perm) in permanent_sim_exclusion.iter().enumerate() {
                if perm {
                    exclude_mask[self.n_cells + si] = true;
                }
            }

            let mut n_excluded_obs = 0usize;
            if iter > 0 {
                // Exclude top-scoring real cells (suspected doublets) from
                // training. Quantile cut at (1 - expected_dbr) with a 20% cap
                // to match R's behaviour of not excluding more than a third of
                // real cells.
                let mut obs_sorted = final_scores.clone();
                obs_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                let quantile_idx =
                    ((1.0 - expected_dbr) * (self.n_cells - 1) as f32).round() as usize;
                let exclusion_threshold = obs_sorted[quantile_idx.min(self.n_cells - 1)];
                let max_exclude_obs = self.n_cells / 5;

                for i in 0..self.n_cells {
                    if n_excluded_obs >= max_exclude_obs {
                        break;
                    }
                    if final_scores[i] > exclusion_threshold {
                        exclude_mask[i] = true;
                        n_excluded_obs += 1;
                    }
                }

                if verbosity.detailed_verbosity() {
                    let n_perm = permanent_sim_exclusion.iter().filter(|&&x| x).count();
                    println!(
                        "Excluding {} suspected real doublets (threshold={:.4}) \
                         and {} permanently-flagged sims",
                        n_excluded_obs, exclusion_threshold, n_perm
                    );
                }
            }

            if verbosity.normal_verbosity() {
                println!("Training logistic GBM classifier...");
            }
            let probabilities = classify_doublets(
                features.as_ref(),
                &labels,
                &exclude_mask,
                &gbm_config,
                (iter_seed + 100) as u64,
                verbosity.detailed_verbosity(),
            );

            let (class_weighted, mean_cw) =
                compute_class_weighted(&sim_scores, &parent_clusters, &permanent_sim_exclusion);
            let obs_origins: Vec<Option<(usize, usize)>> = knn_intermediates
                .iter()
                .take(self.n_cells)
                .map(|c| c.origin)
                .collect();
            let difficulties = compute_difficulties(
                self.n_cells,
                n_total,
                &obs_origins,
                &parent_clusters,
                &class_weighted,
                mean_cw,
            );
            update_difficulty_column(&mut features, difficulty_col, &difficulties);

            final_scores.copy_from_slice(&probabilities[..self.n_cells]);
            sim_scores.copy_from_slice(&probabilities[self.n_cells..]);

            // R's .filterUnrecognizableDoublets: run once, after iter 0
            if iter == 0 {
                let flagged = identify_unrecognisable_origins(
                    &final_scores,
                    &cluster_labels,
                    &sim_scores,
                    &parent_clusters,
                    &UnrecognisableFilterParams::default(),
                );
                let sim_mask = mark_sims_from_flagged_origins(&parent_clusters, &flagged);
                for (si, &flag) in sim_mask.iter().enumerate() {
                    if flag {
                        permanent_sim_exclusion[si] = true;
                    }
                }
                if verbosity.detailed_verbosity() {
                    let n_excl = permanent_sim_exclusion.iter().filter(|&&x| x).count();
                    println!(
                        "Unrecognisable filter: flagged {} origin(s), \
                         {} sims permanently excluded",
                        flagged.len(),
                        n_excl,
                    );
                }
            }

            if verbosity.normal_verbosity() {
                let mean = final_scores.iter().sum::<f32>() / final_scores.len() as f32;
                let (mn, mx) = final_scores
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                        (a.min(v), b.max(v))
                    });
                println!(
                    "Iteration {} done in {:.2?} (obs scores: mean={:.4}, min={:.4}, max={:.4})",
                    iter + 1,
                    start_iter.elapsed(),
                    mean,
                    mn,
                    mx,
                );
            }
        }

        // Clamp final scores to [0, 1]
        for s in final_scores.iter_mut() {
            *s = s.clamp(0.0, 1.0);
        }

        let feature_table = if self.params.return_features {
            let mut include_in_training = vec![true; self.n_cells];
            let mut obs_sorted = final_scores.clone();
            obs_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let quantile_idx = ((1.0 - expected_dbr) * (self.n_cells - 1) as f32).round() as usize;
            let exclusion_threshold = obs_sorted[quantile_idx.min(self.n_cells - 1)];
            let max_exclude_obs = self.n_cells / 5;
            let mut n_excluded = 0usize;
            for i in 0..self.n_cells {
                if n_excluded >= max_exclude_obs {
                    break;
                }
                if final_scores[i] > exclusion_threshold {
                    include_in_training[i] = false;
                    n_excluded += 1;
                }
            }

            Some(FeatureTable {
                feature_names: feature_names(&k_values, n_pcs_include),
                values: slice_obs_rows(features.as_ref(), self.n_cells),
                include_in_training,
            })
        } else {
            None
        };

        let threshold = self.params.manual_threshold.unwrap_or_else(|| {
            let t =
                find_threshold_optimised(&final_scores, &sim_scores, expected_dbr, 0.5, verbose);
            if verbosity.normal_verbosity() {
                println!("Threshold set at score = {:.4}", t);
            }
            t
        });

        let predicted_doublets: Vec<bool> = final_scores.iter().map(|&s| s > threshold).collect();
        let n_doublets = predicted_doublets.iter().filter(|&&d| d).count();
        let detected_doublet_rate = n_doublets as f32 / self.n_cells as f32;

        if verbosity.normal_verbosity() {
            println!(
                "Detected {} doublets ({:.1}%)",
                n_doublets,
                100.0 * detected_doublet_rate
            );
            println!("Total runtime: {:.2?}", start_all.elapsed());
        }

        Ok(ScDblFinderResult {
            predicted_doublets,
            doublet_scores: final_scores,
            cxds: min_max_scale(&obs_cxds_scores),
            weighted: obs_intermediates
                .iter()
                .map(|c: &CellKnnFeatures| c.weighted)
                .collect(),
            selected_genes,
            threshold,
            cluster_labels,
            detected_doublet_rate,
            features: feature_table,
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_origin_homotypic_is_none() {
        assert_eq!(canonical_origin(3, 3), None);
    }

    #[test]
    fn test_canonical_origin_sorts_tuple() {
        assert_eq!(canonical_origin(5, 2), Some((2, 5)));
        assert_eq!(canonical_origin(2, 5), Some((2, 5)));
    }

    #[test]
    fn test_origin_no_sim_neighbours() {
        // All neighbours are real cells (indices < n_obs)
        let neighbours = vec![0, 1, 2, 3];
        let distances = vec![0.1, 0.2, 0.3, 0.4];
        let parent_clusters: Vec<(usize, usize)> = vec![];
        let n_obs = 10;
        assert_eq!(
            cell_most_likely_origin(&neighbours, &distances, n_obs, &parent_clusters),
            None
        );
    }

    #[test]
    fn test_origin_unique_majority() {
        // n_obs = 5, neighbours 5,6,7 are sim doublets with clusters
        // (0,1), (0,1), (1,2)
        let neighbours = vec![5, 6, 7, 0];
        let distances = vec![0.1, 0.2, 0.3, 0.4];
        let parent_clusters = vec![(0, 1), (0, 1), (1, 2)];
        let n_obs = 5;
        assert_eq!(
            cell_most_likely_origin(&neighbours, &distances, n_obs, &parent_clusters),
            Some((0, 1))
        );
    }

    #[test]
    fn test_origin_tiebreak_by_min_distance() {
        // Two origins tied 2:2, (0,1) has closer min distance
        let neighbours = vec![5, 6, 7, 8];
        let distances = vec![0.05, 0.50, 0.10, 0.60]; // (0,1) at 0.05, (1,2) at 0.10
        let parent_clusters = vec![(0, 1), (1, 2), (0, 1), (1, 2)];
        let n_obs = 5;
        assert_eq!(
            cell_most_likely_origin(&neighbours, &distances, n_obs, &parent_clusters),
            Some((0, 1))
        );
    }

    #[test]
    fn test_origin_skips_homotypic_neighbours() {
        // Neighbour 5 has homotypic origin (should be ignored)
        let neighbours = vec![5, 6];
        let distances = vec![0.1, 0.9];
        let parent_clusters = vec![(2, 2), (0, 1)];
        let n_obs = 5;
        assert_eq!(
            cell_most_likely_origin(&neighbours, &distances, n_obs, &parent_clusters),
            Some((0, 1))
        );
    }

    #[test]
    fn test_dist_to_real_finds_first_real() {
        let neighbours = vec![10, 11, 2, 3]; // 2 is first real (n_obs=5)
        let distances = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(
            cell_distance_to_nearest_real(&neighbours, &distances, 5, 1.0),
            0.3
        );
    }

    #[test]
    fn test_dist_to_real_fallback() {
        // All neighbours are simulated
        let neighbours = vec![10, 11, 12];
        let distances = vec![0.1, 0.2, 0.3];
        assert_eq!(
            cell_distance_to_nearest_real(&neighbours, &distances, 5, 0.75),
            1.5 // 2 * 0.75
        );
    }

    #[test]
    fn test_aggregate_origin_counts() {
        let origins = vec![Some((0, 1)), Some((0, 1)), None, Some((1, 2)), Some((0, 1))];
        let counts = aggregate_origin_counts(&origins);
        assert_eq!(counts[&(0, 1)], 3);
        assert_eq!(counts[&(1, 2)], 1);
        assert_eq!(counts.len(), 2);
    }
}
