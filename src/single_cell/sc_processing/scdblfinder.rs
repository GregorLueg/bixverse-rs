//! Work-in-progress not behaving as desired...

use faer::{Mat, MatRef, concat};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::graph::community_detections::*;
use crate::graph::graph_structures::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::utils_doublets::*;

// GBM infrastructure from SCENIC
use crate::single_cell::sc_analysis::scenic::{
    DenseQuantisedStore, GbmScratch, GradientBoostingConfig, HistogramPool, build_gbm_node,
};

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
    /// Optional target library size. Defaults to the mean HVG library size.
    pub target_size: Option<f32>,
    /// Percentile threshold for HVG selection.
    pub min_gene_var_pctl: f32,
    /// HVG method: `"vst"`, `"mvb"`, or `"dispersion"`.
    pub hvg_method: String,
    /// Loess span for VST fitting.
    pub loess_span: f64,
    /// Optional clip max for variance stabilisation.
    pub clip_max: Option<f32>,

    // -- PCA --
    /// Number of principal components.
    pub no_pcs: usize,
    /// Whether to use randomised SVD.
    pub random_svd: bool,

    // -- Simulation --
    /// Ratio of simulated doublets to observed cells.
    pub doublet_ratio: f32,
    /// Fraction of pairs forced to be from different clusters (0.0-1.0).
    pub heterotypic_bias: f32,

    // -- Clustering --
    /// Resolution for Louvain clustering.
    pub cluster_resolution: f32,
    /// Number of Louvain iterations per clustering step.
    pub cluster_iters: usize,

    // -- kNN --
    /// Parameters for kNN construction.
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
    /// Number of recent OOB improvements to average for early stopping.
    pub early_stop_window: usize,
    /// Fraction of samples used for training each tree.
    pub subsample_rate: f32,

    // -- Feature engineering --
    /// Number of leading PCs to include as classifier features.
    pub include_pcs: usize,
    /// Expected doublet rate
    pub dbr_per_1k: f32,

    // -- Thresholding --
    /// Optional manual threshold. If `None`, Otsu's method is used.
    pub manual_threshold: Option<f32>,
    /// Number of histogram bins for Otsu threshold detection.
    pub n_bins: usize,
}

impl Default for ScDblFinderParams {
    fn default() -> Self {
        Self {
            log_transform: true,
            mean_center: true,
            normalise_variance: true,
            target_size: None,
            min_gene_var_pctl: 0.85,
            hvg_method: "vst".to_string(),
            loess_span: 0.3,
            clip_max: None,
            no_pcs: 30,
            random_svd: true,
            doublet_ratio: 1.0,
            heterotypic_bias: 0.8,
            cluster_resolution: 1.0,
            cluster_iters: 10,
            knn_params: KnnParams::default(),
            n_iterations: 3,
            // -- Aligned with R's XGBoost defaults --
            n_trees: 200,
            max_depth: 4,
            learning_rate: 0.3,
            min_samples_leaf: 20,
            early_stop_window: 3,
            subsample_rate: 0.75,
            include_pcs: 19,
            manual_threshold: None,
            n_bins: 100,
            // -- Expected doublet rate --
            dbr_per_1k: 0.008,
        }
    }
}

/// Results from scDblFinder.
#[derive(Clone, Debug)]
pub struct ScDblFinderResult {
    /// Per-cell doublet predictions (`true` = doublet).
    pub predicted_doublets: Vec<bool>,
    /// Classifier probability per observed cell (0 = singlet, 1 = doublet).
    pub doublet_scores: Vec<f32>,
    /// Threshold used for doublet calling.
    pub threshold: f32,
    /// Cluster labels from the final iteration.
    pub cluster_labels: Vec<usize>,
    /// Fraction of observed cells called as doublets.
    pub detected_doublet_rate: f32,
}

///////////
// Types //
///////////

/// Tuple of (pairs, parent_cluster_labels) where pairs index into the
/// cell population and parent_cluster_labels record each pair's cluster
/// origins.
type ClusterAwarePairs = (Vec<(usize, usize)>, Vec<(usize, usize)>);

//////////////
// refactor //
//////////////

/// Build kNN returning both indices and distances.
///
/// Wraps `dispatch_knn` and a parallel distance computation from the
/// PCA embedding.
fn dispatch_knn_with_distances(
    embd: MatRef<f32>,
    k: usize,
    knn_params: &KnnParams,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let indices = dispatch_knn(embd, k, knn_params, seed, verbose);

    let n_dims = embd.ncols();
    let distances: Vec<Vec<f32>> = indices
        .par_iter()
        .enumerate()
        .map(|(i, neighbours)| {
            neighbours
                .iter()
                .map(|&j| {
                    let mut d2 = 0.0f32;
                    for dim in 0..n_dims {
                        let diff = *embd.get(i, dim) - *embd.get(j, dim);
                        d2 += diff * diff;
                    }
                    d2.sqrt()
                })
                .collect()
        })
        .collect();

    (indices, distances)
}

/// Compute per-cell gene complexity statistics.
///
/// Returns `(n_features, n_above2)` where `n_features` is the number of
/// genes with non-zero expression and `n_above2` is the number of genes
/// with count > 2. These are computed over HVG genes only.
fn compute_cell_complexity(
    f_path_cell: &str,
    cells_to_keep: &[usize],
    hvg_genes: &[usize],
) -> (Vec<u32>, Vec<u32>) {
    let hvg_set: FxHashSet<usize> = hvg_genes.iter().copied().collect();
    let reader = ParallelSparseReader::new(f_path_cell).unwrap();

    let results: Vec<(u32, u32)> = cells_to_keep
        .par_iter()
        .map(|&cell_idx| {
            let chunk = reader.read_cell(cell_idx);
            let mut n_feat = 0u32;
            let mut n_above2 = 0u32;
            for (i, &gene_idx) in chunk.indices.iter().enumerate() {
                if hvg_set.contains(&(gene_idx as usize)) {
                    let count = chunk.data_raw.get(i);
                    if count > 0 {
                        n_feat += 1;
                    }
                    if count > 2 {
                        n_above2 += 1;
                    }
                }
            }
            (n_feat, n_above2)
        })
        .collect();

    let n_features: Vec<u32> = results.iter().map(|&(nf, _)| nf).collect();
    let n_above2: Vec<u32> = results.iter().map(|&(_, na)| na).collect();
    (n_features, n_above2)
}

/// Compute gene complexity for simulated doublet chunks.
///
/// Since chunks already have HVG-remapped indices, we just count
/// non-zero and >2 entries directly.
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

//////////////////////////////////////
// Feature engineering | pair logic //
//////////////////////////////////////

/// Find the doublet score threshold targeting the expected doublet rate.
///
/// Sorts observed cell scores descending and places the threshold so that
/// approximately `expected_dbr * n_obs` cells are called as doublets,
/// with a tolerance band of +/- 40% around the expected rate (matching
/// R's `dbr.sd` default).
///
/// Falls back to Otsu if the expected rate produces a degenerate threshold.
fn find_threshold_expected_rate(
    scores_obs: &[f32],
    n_obs: usize,
    dbr_per_1k: f32,
    n_bins: usize,
) -> f32 {
    let expected_dbr = dbr_per_1k * (n_obs as f32 / 1000.0);
    let expected_dbr = expected_dbr.min(0.5); // sanity cap

    let expected_n = (expected_dbr * n_obs as f32).round() as usize;
    if expected_n == 0 || expected_n >= n_obs {
        return find_threshold_otsu(scores_obs, n_bins);
    }

    let mut sorted: Vec<f32> = scores_obs.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

    // Threshold sits between the expected_n-th and (expected_n+1)-th
    // highest score
    let idx = expected_n.min(sorted.len() - 1);
    let threshold = if idx > 0 {
        (sorted[idx - 1] + sorted[idx]) / 2.0
    } else {
        sorted[0] - 1e-6
    };

    // Sanity check: if threshold is degenerate (e.g. all scores identical),
    // fall back to Otsu
    let n_above = sorted.iter().filter(|&&s| s > threshold).count();
    if n_above == 0 || n_above == n_obs {
        return find_threshold_otsu(scores_obs, n_bins);
    }

    threshold
}

/// Compute default k values for multi-scale kNN doublet scoring.
///
/// Mirrors the R scDblFinder `.defaultKnnKs(k=NULL, n)` logic:
/// `kmax = max(ceil(sqrt(n/2)), 25)`, then `unique(c(3,10,15,20,25,50,kmax)[<=kmax])`.
fn default_knn_ks(n_obs: usize) -> Vec<usize> {
    let kmax = ((n_obs as f32 / 2.0).sqrt().ceil() as usize).max(25);
    let candidates = [3, 10, 15, 20, 25, 50, kmax];
    let mut ks: Vec<usize> = candidates.iter().copied().filter(|&k| k <= kmax).collect();
    ks.sort_unstable();
    ks.dedup();
    ks
}

/// Build the feature matrix for the GBM classifier.
///
/// Features per cell (matching R scDblFinder's `.defTrainFeatures`):
/// 1. Multi-scale kNN ratios (one per k value)
/// 2. Distance-weighted doublet proportion ("weighted")
/// 3. Shannon entropy of neighbour cluster distribution
/// 4. Library size ratio
/// 5. Number of expressed genes ("nfeatures")
/// 6. Number of genes with count > 2 ("nAbove2")
/// 7. Principal components (n_pcs columns)
///
/// Notably absent vs R: `cxds_score`, `difficulty`. Cluster proportions
/// are intentionally excluded (R excludes `cluster` from training).
#[allow(clippy::too_many_arguments)]
fn build_feature_matrix(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    n_obs: usize,
    k_values: &[usize],
    obs_n_features: &[u32],
    obs_n_above2: &[u32],
    sim_n_features: &[u32],
    sim_n_above2: &[u32],
    library_sizes: &[usize],
    median_lib_size: f32,
    sim_combined_lib_sizes: &[usize],
    combined_pca: MatRef<f32>,
    n_pcs: usize,
) -> Mat<f32> {
    let n_total = knn_indices.len();
    let n_k = k_values.len();
    // k_ratios + weighted + entropy + lib_ratio + nfeatures + nAbove2 + PCs
    let n_feat = n_k + 5 + n_pcs;

    let rows: Vec<Vec<f32>> = (0..n_total)
        .into_par_iter()
        .map(|i| {
            let neighbours = &knn_indices[i];
            let distances = &knn_distances[i];
            let k_max = neighbours.len();

            // Multi-scale doublet ratios
            let mut feats = Vec::with_capacity(n_feat);
            for &k in k_values {
                let k_use = k.min(k_max);
                let n_sim = neighbours[..k_use]
                    .iter()
                    .filter(|&&idx| idx >= n_obs)
                    .count();
                feats.push(n_sim as f32 / k_use as f32);
            }

            // Distance-weighted doublet proportion
            let k_f = k_max as f32;
            let mut w_sum = 0.0f32;
            let mut w_dbl = 0.0f32;
            for (rank, (&neigh, &dist)) in neighbours.iter().zip(distances.iter()).enumerate() {
                let d = dist.max(1e-10);
                let w = (k_f - rank as f32).sqrt() / d;
                w_sum += w;
                if neigh >= n_obs {
                    w_dbl += w;
                }
            }
            feats.push(if w_sum > 0.0 { w_dbl / w_sum } else { 0.0 });

            // Neighbour cluster entropy -- computed from the type labels
            // (real vs sim) at multiple scales is already captured by the
            // k_ratios. Instead compute entropy over the identity of
            // neighbours themselves (how spread out are they).
            // Actually: use ratio variance across k values as a stability signal
            let mean_ratio = feats[..n_k].iter().sum::<f32>() / n_k as f32;
            let ratio_var: f32 = feats[..n_k]
                .iter()
                .map(|&r| (r - mean_ratio).powi(2))
                .sum::<f32>()
                / n_k as f32;
            feats.push(ratio_var);

            // Library size ratio
            let lib_ratio = if i < n_obs {
                library_sizes[i] as f32 / median_lib_size
            } else {
                sim_combined_lib_sizes[i - n_obs] as f32 / median_lib_size
            };
            feats.push(lib_ratio);

            // nfeatures, nAbove2
            if i < n_obs {
                feats.push(obs_n_features[i] as f32);
                feats.push(obs_n_above2[i] as f32);
            } else {
                let si = i - n_obs;
                feats.push(sim_n_features[si] as f32);
                feats.push(sim_n_above2[si] as f32);
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

    features
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

    // build cluster -> cell positions
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

    // heterotypic: pick two different clusters, one cell from each
    for _ in 0..n_heterotypic {
        if !have_multiple_clusters {
            // degenerate case: single cluster, fall back to random
            let i = rng.random_range(0..n_cells);
            let j = rng.random_range(0..n_cells);
            pairs.push((i, j));
            parent_clusters.push((cluster_labels[i], cluster_labels[j]));
            continue;
        }

        let ca_idx = rng.random_range(0..cluster_ids.len());
        let mut cb_idx = rng.random_range(0..cluster_ids.len());
        while cb_idx == ca_idx {
            cb_idx = rng.random_range(0..cluster_ids.len());
        }

        let ca = cluster_ids[ca_idx];
        let cb = cluster_ids[cb_idx];
        let cells_a = &cluster_to_cells[&ca];
        let cells_b = &cluster_to_cells[&cb];

        let i = cells_a[rng.random_range(0..cells_a.len())];
        let j = cells_b[rng.random_range(0..cells_b.len())];

        pairs.push((i, j));
        parent_clusters.push((ca, cb));
    }

    // random pairs
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

/// Train GBM classifier with optional sample exclusions.
///
/// Excluded samples are omitted from both training and OOB sets but still
/// receive predictions. This implements the R scDblFinder iterative
/// refinement where suspected doublets among observed cells are removed
/// from training to give the classifier a cleaner signal.
fn fit_gbm_classifier_with_exclusions(
    features: MatRef<f32>,
    labels: &[f32],
    exclude: &[bool],
    config: &GradientBoostingConfig,
    seed: usize,
) -> Vec<f32> {
    let n_samples = features.nrows();
    let n_features = features.ncols();
    let store = quantise_dense_features(features);
    let eligible: Vec<u32> = (0..n_samples as u32)
        .filter(|&i| !exclude[i as usize])
        .collect();
    let n_eligible = eligible.len();
    let mut residuals = labels.to_vec();
    let n_train = ((n_eligible as f32 * config.subsample_rate).round() as usize)
        .max(2 * config.min_samples_leaf);
    let n_oob = n_eligible.saturating_sub(n_train);
    let mut eligible_buf = eligible.clone();
    let mut importances = vec![0.0f32; n_features];
    let mut improvement_ring = Vec::with_capacity(config.early_stop_window);
    let mut scratch = GbmScratch::new(n_features, n_samples);
    let pool_capacity = 2 * config.max_depth + 3;
    let mut pool = HistogramPool::new(pool_capacity, n_features);
    for tree_idx in 0..config.n_trees_max {
        let mut rng = SmallRng::seed_from_u64(
            seed.wrapping_add(tree_idx.wrapping_mul(6364136223846793005)) as u64,
        );
        eligible_buf.copy_from_slice(&eligible);
        for i in 0..n_train.min(n_eligible) {
            let j = rng.random_range(i..n_eligible);
            eligible_buf.swap(i, j);
        }
        let (train, oob) = eligible_buf.split_at_mut(n_train.min(n_eligible));
        let root_idx = pool.acquire();
        pool.histograms[root_idx].build_from_samples(&store, train, &residuals);
        let mut y_sum = 0.0f32;
        let mut y_sum_sq = 0.0f32;
        for &s in train.iter() {
            let r = residuals[s as usize];
            y_sum += r;
            y_sum_sq += r * r;
        }
        let mut oob_improvement = 0.0f32;
        build_gbm_node(
            &mut pool,
            root_idx,
            &store,
            &mut residuals,
            train,
            oob,
            y_sum,
            y_sum_sq,
            train.len(),
            config,
            0,
            config.learning_rate,
            &mut importances,
            &mut oob_improvement,
            &mut scratch,
            &mut rng,
        );
        let oob_mse_improvement = if n_oob > 0 {
            oob_improvement / n_oob as f32
        } else {
            0.0
        };
        if improvement_ring.len() >= config.early_stop_window {
            improvement_ring.remove(0);
        }
        improvement_ring.push(oob_mse_improvement);
        if improvement_ring.len() >= config.early_stop_window {
            let avg: f32 = improvement_ring.iter().sum::<f32>() / improvement_ring.len() as f32;
            if avg <= 0.0 {
                break;
            }
        }
    }
    labels
        .iter()
        .zip(residuals.iter())
        .map(|(&l, &r)| l - r)
        .collect()
}

/// Quantise a dense f32 feature matrix into a `DenseQuantisedStore`.
///
/// Each column is independently scaled to `[0, 255]`, matching the SCENIC
/// quantisation scheme so the histogram-based tree infrastructure can be
/// reused directly.
///
/// ### Params
///
/// * `features` - Dense feature matrix, shape `(n_samples, n_features)`.
///
/// ### Returns
///
/// A `DenseQuantisedStore` ready for histogram-based tree building.
fn quantise_dense_features(features: MatRef<f32>) -> DenseQuantisedStore {
    let n_samples = features.nrows();
    let n_features = features.ncols();
    let mut data = vec![0u8; n_features * n_samples];
    let mut feature_min = Vec::with_capacity(n_features);
    let mut feature_range = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        for i in 0..n_samples {
            let v = *features.get(i, j);
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }

        let range = max_v - min_v;
        feature_min.push(min_v);
        feature_range.push(range);

        let offset = j * n_samples;
        if range > 1e-10 {
            let scale = 255.0 / range;
            for i in 0..n_samples {
                let v = *features.get(i, j);
                data[offset + i] = ((v - min_v) * scale).round() as u8;
            }
        }
    }

    DenseQuantisedStore {
        data,
        n_cells: n_samples,
        n_features,
        feature_min,
        feature_range,
    }
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
    /// Per-cell library sizes computed over HVG genes only.
    hvg_library_sizes: Vec<usize>,
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
            hvg_library_sizes: Vec::new(),
        }
    }

    /// Run the full scDblFinder pipeline.
    ///
    /// ### Algorithm
    ///
    /// 1. Select HVGs, compute HVG library sizes.
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
    /// * `streaming` - Stream HVG computation to reduce memory pressure.
    /// * `seed` - Seed for reproducibility.
    /// * `verbose` - Controls verbosity.
    ///
    /// ### Returns
    ///
    /// `ScDblFinderResult` with predictions, scores and cluster labels.
    pub fn run(&mut self, streaming: bool, seed: usize, verbose: bool) -> ScDblFinderResult {
        let start_all = Instant::now();

        let hvg_opts = HvgOpts {
            method: self.params.hvg_method.clone(),
            loess_span: self.params.loess_span as f32,
            clip_max: self.params.clip_max,
            min_gene_var_pctl: self.params.min_gene_var_pctl,
        };

        // -- Step 1: HVGs and library sizes --
        if verbose {
            println!("Identifying highly variable genes...");
        }
        let start_hvg = Instant::now();

        let hvg_genes = select_hvg(
            &self.f_path_gene,
            &self.cells_to_keep,
            &hvg_opts,
            streaming,
            verbose,
        );

        if verbose {
            println!(
                "Using {} highly variable genes. Done in {:.2?}",
                hvg_genes.len(),
                start_hvg.elapsed()
            );
        }

        self.hvg_library_sizes =
            compute_hvg_library_sizes(&self.f_path_cell, &self.cells_to_keep, &hvg_genes);
        let target_size = resolve_target_size(self.params.target_size, &self.hvg_library_sizes);
        self.n_cells_sim = (self.n_cells as f32 * self.params.doublet_ratio) as usize;

        let median_lib_size = {
            let mut sorted = self.hvg_library_sizes.clone();
            sorted.sort_unstable();
            sorted[sorted.len() / 2] as f32
        };

        // -- Step 1b: Cell complexity features --
        if verbose {
            println!("Computing cell complexity features...");
        }
        let (obs_n_features, obs_n_above2) =
            compute_cell_complexity(&self.f_path_cell, &self.cells_to_keep, &hvg_genes);

        // -- Step 2: PCA on observed cells --
        if verbose {
            println!("Running PCA on observed cells...");
        }
        let start_pca = Instant::now();

        let pca_res = pca_observed(
            &self.f_path_gene,
            &self.cells_to_keep,
            &hvg_genes,
            &self.hvg_library_sizes,
            target_size,
            self.params.log_transform,
            self.params.mean_center,
            self.params.normalise_variance,
            self.params.no_pcs,
            self.params.random_svd,
            seed,
            verbose,
        );

        let obs_pca = &pca_res.0;
        let loadings = &pca_res.1;
        let gene_means = &pca_res.2;
        let gene_stds = &pca_res.3;

        if verbose {
            println!("Done with PCA in {:.2?}", start_pca.elapsed());
        }

        // -- Step 3: Initial clustering --
        if verbose {
            println!("Initial clustering of observed cells...");
        }

        let obs_k = if self.params.knn_params.k == 0 {
            ((self.n_cells as f32).sqrt() * 0.5).round() as usize
        } else {
            self.params.knn_params.k
        };

        let obs_knn = dispatch_knn(
            obs_pca.as_ref(),
            obs_k,
            &self.params.knn_params,
            seed,
            verbose,
        );

        let obs_graph = knn_to_sparse_graph(&obs_knn);
        let cluster_labels = louvain_sparse_graph(
            &obs_graph,
            self.params.cluster_resolution,
            self.params.cluster_iters,
            seed,
        );

        // -- Multi-scale k values --
        let k_values = default_knn_ks(self.n_cells);
        let k_max = *k_values.last().unwrap();
        let n_pcs_include = self.params.include_pcs.min(self.params.no_pcs);

        if verbose {
            println!(
                "Using k values {:?} (max {}), including {} PCs as features",
                k_values, k_max, n_pcs_include
            );
        }

        let gbm_config = GradientBoostingConfig {
            n_trees_max: self.params.n_trees,
            learning_rate: self.params.learning_rate,
            max_depth: self.params.max_depth,
            min_samples_leaf: self.params.min_samples_leaf,
            early_stop_window: self.params.early_stop_window,
            subsample_rate: self.params.subsample_rate,
            n_features_split: 0,
        };

        // -- Step 4: Simulate doublets ONCE --
        if verbose {
            println!("Simulating cluster-aware doublets...");
        }

        let (pairs, _parent_clusters) = cluster_aware_pairs(
            &cluster_labels,
            self.n_cells_sim,
            self.params.heterotypic_bias,
            seed,
        );

        let sim_combined_lib_sizes: Vec<usize> = pairs
            .iter()
            .map(|&(a, b)| self.hvg_library_sizes[a] + self.hvg_library_sizes[b])
            .collect();

        let sim_chunks = simulate_from_pairs(
            &pairs,
            &self.cells_to_keep,
            &self.hvg_library_sizes,
            &hvg_genes,
            &self.f_path_cell,
            target_size,
            self.params.log_transform,
        );

        // Complexity features for simulated doublets
        let (sim_n_features, sim_n_above2) = compute_sim_complexity(&sim_chunks);

        // -- Step 5: Project simulated into PC space --
        if verbose {
            println!("Projecting simulated doublets...");
        }
        let scaled_sim = scale_cell_chunks_with_stats(
            &sim_chunks,
            gene_means,
            gene_stds,
            self.params.mean_center,
            self.params.normalise_variance,
            hvg_genes.len(),
        );
        let sim_pca = &scaled_sim * loadings;
        let combined_pca = concat![[obs_pca], [sim_pca]];

        // -- Step 6: kNN on combined ONCE --
        if verbose {
            println!("Building combined kNN graph (k={})...", k_max);
        }
        let (combined_knn, combined_dists) = dispatch_knn_with_distances(
            combined_pca.as_ref(),
            k_max,
            &self.params.knn_params,
            seed,
            verbose,
        );

        // -- Step 7: Build feature matrix ONCE --
        let n_feat = k_values.len() + 5 + n_pcs_include;
        if verbose {
            println!("Building feature matrix ({} features)...", n_feat);
        }

        let features = build_feature_matrix(
            &combined_knn,
            &combined_dists,
            self.n_cells,
            &k_values,
            &obs_n_features,
            &obs_n_above2,
            &sim_n_features,
            &sim_n_above2,
            &self.hvg_library_sizes,
            median_lib_size,
            &sim_combined_lib_sizes,
            combined_pca.as_ref(),
            n_pcs_include,
        );

        let n_total = self.n_cells + self.n_cells_sim;
        let labels: Vec<f32> = (0..n_total)
            .map(|i| if i < self.n_cells { 0.0 } else { 1.0 })
            .collect();

        // -- Step 8: Iterative training refinement --
        let mut final_scores = vec![0.0f32; self.n_cells];

        // Initial score: use the max-k ratio as a rough doublet estimate
        let ratio_col = k_values.len() - 1; // last k ratio column
        for i in 0..self.n_cells {
            final_scores[i] = *features.get(i, ratio_col);
        }

        for iter in 0..self.params.n_iterations {
            if verbose {
                println!(
                    " == Iteration {} of {} ==",
                    iter + 1,
                    self.params.n_iterations
                );
            }
            let start_iter = Instant::now();
            let iter_seed = seed + iter * 1000;

            // Build exclusion mask from previous scores
            let mut exclude_mask = vec![false; n_total];
            if iter > 0 {
                let n_exclude_max = self.n_cells / 5;
                let mut obs_scored: Vec<(usize, f32)> = final_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                obs_scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut n_excluded = 0;
                for &(cell_idx, score) in &obs_scored {
                    if score <= 0.3 || n_excluded >= n_exclude_max {
                        break;
                    }
                    exclude_mask[cell_idx] = true;
                    n_excluded += 1;
                }

                // Also exclude unidentifiable artificial doublets (low scores)
                let sim_scores: Vec<f32> = (self.n_cells..n_total)
                    .map(|i| labels[i] - features.get(i, ratio_col))
                    .collect();
                let n_sim_exclude_max = self.n_cells_sim / 4;
                let mut sim_scored: Vec<(usize, f32)> = sim_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                sim_scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let mut n_sim_excluded = 0;
                for &(sim_idx, _score) in &sim_scored {
                    // Use raw score from previous iteration for simulated
                    let global_idx = self.n_cells + sim_idx;
                    let prev_raw = labels[global_idx]
                        - (labels[global_idx] - final_scores.get(0).copied().unwrap_or(0.0));
                    if n_sim_excluded >= n_sim_exclude_max {
                        break;
                    }
                    // Exclude simulated doublets that scored below 0.2
                    // (the classifier couldn't identify them)
                    let sim_ratio = *features.get(global_idx, ratio_col);
                    if sim_ratio < 0.2 {
                        exclude_mask[global_idx] = true;
                        n_sim_excluded += 1;
                    }
                }

                if verbose {
                    println!(
                        "Excluding {} suspected real doublets and {} unidentifiable \
                         artificial doublets from training",
                        n_excluded, n_sim_excluded
                    );
                }
            }

            // Train GBM
            if verbose {
                println!("Training gradient-boosted classifier...");
            }
            let raw_scores = fit_gbm_classifier_with_exclusions(
                features.as_ref(),
                &labels,
                &exclude_mask,
                &gbm_config,
                iter_seed + 100,
            );

            final_scores = raw_scores[..self.n_cells].to_vec();

            if verbose {
                let mean_score: f32 = final_scores.iter().sum::<f32>() / final_scores.len() as f32;
                let max_score = final_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let min_score = final_scores.iter().cloned().fold(f32::INFINITY, f32::min);
                println!(
                    "Iteration {} done in {:.2?} (obs scores: mean={:.4}, min={:.4}, max={:.4})",
                    iter + 1,
                    start_iter.elapsed(),
                    mean_score,
                    min_score,
                    max_score,
                );
            }
        }

        // -- Step 9: Clamp and threshold --
        for s in final_scores.iter_mut() {
            *s = s.clamp(0.0, 1.0);
        }

        let threshold = self.params.manual_threshold.unwrap_or_else(|| {
            let t = find_threshold_expected_rate(
                &final_scores,
                self.n_cells,
                self.params.dbr_per_1k,
                self.params.n_bins,
            );
            if verbose {
                println!("Threshold set at score = {:.4}", t);
            }
            t
        });

        let predicted_doublets: Vec<bool> = final_scores.iter().map(|&s| s > threshold).collect();
        let n_doublets = predicted_doublets.iter().filter(|&&d| d).count();
        let detected_doublet_rate = n_doublets as f32 / self.n_cells as f32;

        if verbose {
            println!(
                "Detected {} doublets ({:.1}%)",
                n_doublets,
                100.0 * detected_doublet_rate
            );
            println!("Total runtime: {:.2?}", start_all.elapsed());
        }

        ScDblFinderResult {
            predicted_doublets,
            doublet_scores: final_scores,
            threshold,
            cluster_labels,
            detected_doublet_rate,
        }
    }
}

///////
// R //
///////

impl ScDblFinderParams {
    /// Generate ScDblFinderParams from an R list.
    ///
    /// Values not found in the list fall back to the `Default` implementation.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with scDblFinder parameters.
    ///
    /// ### Returns
    ///
    /// `ScDblFinderParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());
        let map = r_list.into_hashmap();
        let defaults = Self::default();

        Self {
            // Normalisation
            log_transform: map
                .get("log_transform")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.log_transform),
            mean_center: map
                .get("mean_center")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.mean_center),
            normalise_variance: map
                .get("normalise_variance")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.normalise_variance),
            target_size: map
                .get("target_size")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            // HVG
            min_gene_var_pctl: map
                .get("min_gene_var_pctl")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.min_gene_var_pctl as f64) as f32,
            hvg_method: String::from(
                map.get("hvg_method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("vst"),
            ),
            loess_span: map
                .get("loess_span")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.loess_span),
            clip_max: map
                .get("clip_max")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            // PCA
            no_pcs: map
                .get("no_pcs")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.no_pcs as i32) as usize,
            random_svd: map
                .get("random_svd")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.random_svd),
            // Simulation
            doublet_ratio: map
                .get("doublet_ratio")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.doublet_ratio as f64) as f32,
            heterotypic_bias: map
                .get("heterotypic_bias")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.heterotypic_bias as f64) as f32,
            // Clustering
            cluster_resolution: map
                .get("cluster_resolution")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.cluster_resolution as f64)
                as f32,
            cluster_iters: map
                .get("cluster_iters")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.cluster_iters as i32) as usize,
            // kNN
            knn_params,
            // Iteration
            n_iterations: map
                .get("n_iterations")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_iterations as i32) as usize,
            // Classification
            n_trees: map
                .get("n_trees")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_trees as i32) as usize,
            max_depth: map
                .get("max_depth")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.max_depth as i32) as usize,
            learning_rate: map
                .get("learning_rate")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.learning_rate as f64) as f32,
            min_samples_leaf: map
                .get("min_samples_leaf")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.min_samples_leaf as i32) as usize,
            early_stop_window: map
                .get("early_stop_window")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.early_stop_window as i32)
                as usize,
            subsample_rate: map
                .get("subsample_rate")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.subsample_rate as f64) as f32,
            // Feature
            include_pcs: map
                .get("include_pcs")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.include_pcs as i32) as usize,
            // Thresholding
            manual_threshold: map
                .get("manual_threshold")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            n_bins: map
                .get("n_bins")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_bins as i32) as usize,
            // Expected doublet rate
            dbr_per_1k: map
                .get("dbr_per_1k")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.dbr_per_1k as f64) as f32,
        }
    }
}
