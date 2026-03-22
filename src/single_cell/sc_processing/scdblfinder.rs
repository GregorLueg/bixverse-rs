//! scDblFinder-inspired doublet detection for single cell data.
//!
//! Implements a cluster-aware doublet detection method inspired by Germain
//! et al., F1000Research, 2022. Key differences from Scrublet and Boost:
//!
//! 1. **Cluster-aware simulation**: doublets are preferentially generated
//!    from cells in different clusters (heterotypic), producing more
//!    realistic synthetic doublets.
//! 2. **Feature engineering**: instead of working purely in embedding space,
//!    a feature matrix is constructed per cell with cluster-neighbourhood
//!    proportions, kNN-based doublet proportion, Shannon entropy, library
//!    size ratio, and origin pair information.
//! 3. **Gradient-boosted classifier**: trained on engineered features to
//!    distinguish observed cells from synthetic doublets.
//! 4. **Iterative refinement**: cluster assignments are updated after each
//!    round of classification, typically converging in 2-3 iterations.

use faer::{Mat, MatRef, concat};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
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
            n_trees: 500,
            max_depth: 4,
            learning_rate: 0.05,
            min_samples_leaf: 20,
            early_stop_window: 15,
            subsample_rate: 0.8,
            include_pcs: 19,
            manual_threshold: None,
            n_bins: 100,
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

//////////////////////////////////////
// Feature engineering | pair logic //
//////////////////////////////////////

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
/// For each cell (observed or simulated), computes:
///
/// 1. **Multi-scale kNN ratios**: proportion of neighbours that are
///    simulated, evaluated at each k in `k_values`. Captures doublet
///    signal at different neighbourhood scales.
/// 2. **Cluster neighbour proportions**: fraction of neighbours in each
///    cluster (at max k). Produces `n_clusters` columns.
/// 3. **Cluster entropy**: Shannon entropy of the neighbour cluster
///    distribution.
/// 4. **Library size ratio**: cell's HVG library size divided by the
///    median.
/// 5. **Principal components**: the first `n_pcs` columns of the combined
///    PCA embedding, giving the classifier direct access to expression
///    space coordinates.
#[allow(clippy::too_many_arguments)]
fn build_feature_matrix(
    knn_indices: &[Vec<usize>],
    cluster_labels: &[usize],
    n_obs: usize,
    n_clusters: usize,
    library_sizes: &[usize],
    median_lib_size: f32,
    sim_combined_lib_sizes: &[usize],
    combined_pca: MatRef<f32>,
    k_values: &[usize],
    n_pcs: usize,
) -> Mat<f32> {
    let n_total = knn_indices.len();
    let n_k = k_values.len();
    // k_ratios + cluster_props + entropy + lib_ratio + PCs
    let n_features = n_k + n_clusters + 2 + n_pcs;

    let rows: Vec<Vec<f32>> = (0..n_total)
        .into_par_iter()
        .map(|i| {
            let neighbours = &knn_indices[i];
            let k_max = neighbours.len();

            // Multi-scale doublet ratios
            let mut k_ratios = Vec::with_capacity(n_k);
            for &k in k_values {
                let k_use = k.min(k_max);
                let n_sim = neighbours[..k_use]
                    .iter()
                    .filter(|&&idx| idx >= n_obs)
                    .count();
                k_ratios.push(n_sim as f32 / k_use as f32);
            }

            // Cluster neighbour proportions (at max k)
            let mut cluster_counts = vec![0.0f32; n_clusters];
            for &neigh in neighbours {
                let cl = cluster_labels[neigh];
                if cl < n_clusters {
                    cluster_counts[cl] += 1.0;
                }
            }
            let k_f = k_max as f32;
            for c in cluster_counts.iter_mut() {
                *c /= k_f;
            }

            // Shannon entropy
            let entropy: f32 = cluster_counts
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();

            // Library size ratio
            let lib_ratio = if i < n_obs {
                library_sizes[i] as f32 / median_lib_size
            } else {
                let sim_idx = i - n_obs;
                sim_combined_lib_sizes[sim_idx] as f32 / median_lib_size
            };

            // Assemble row
            let mut row = Vec::with_capacity(n_features);
            row.extend_from_slice(&k_ratios);
            row.extend_from_slice(&cluster_counts);
            row.push(entropy);
            row.push(lib_ratio);

            // PCs
            for pc in 0..n_pcs {
                row.push(*combined_pca.get(i, pc));
            }

            row
        })
        .collect();

    let mut features = Mat::<f32>::zeros(n_total, n_features);
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

/// Train a gradient-boosted binary classifier and return per-sample scores.
///
/// Uses the SCENIC histogram-based GBM infrastructure but tracks
/// accumulated predictions rather than feature importances. The training
/// loop mirrors `fit_grnboost2_full_hist` with the addition of a
/// prediction accumulator.
///
/// Labels should be 0.0 (observed/singlet) or 1.0 (simulated/doublet).
/// Returned scores are raw accumulated predictions (not sigmoid-transformed);
/// the caller applies sigmoid post-hoc.
///
/// ### Params
///
/// * `features` - Feature matrix, shape `(n_samples, n_features)`.
/// * `labels` - Binary labels per sample.
/// * `config` - GBM configuration.
/// * `seed` - Seed for reproducibility.
///
/// ### Returns
///
/// Raw prediction scores per sample. Higher values indicate higher doublet
/// probability.
fn fit_gbm_classifier(
    features: MatRef<f32>,
    labels: &[f32],
    config: &GradientBoostingConfig,
    seed: usize,
) -> Vec<f32> {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    let store = quantise_dense_features(features);

    // accumulated raw predictions in logit space
    let mut raw_preds = vec![0.0f32; n_samples];
    // working residuals = negative gradient of logistic loss = y - sigmoid(f)
    let mut residuals = vec![0.0f32; n_samples];

    let n_train = ((n_samples as f32 * config.subsample_rate).round() as usize)
        .max(2 * config.min_samples_leaf);

    let mut sample_indices: Vec<u32> = (0..n_samples as u32).collect();
    let mut importances = vec![0.0f32; n_features];
    let mut improvement_ring = Vec::with_capacity(config.early_stop_window);
    let mut scratch = GbmScratch::new(n_features, n_samples);
    let pool_capacity = 2 * config.max_depth + 3;
    let mut pool = HistogramPool::new(pool_capacity, n_features);

    for tree_idx in 0..config.n_trees_max {
        let mut rng = SmallRng::seed_from_u64(
            seed.wrapping_add(tree_idx.wrapping_mul(6364136223846793005)) as u64,
        );

        // recompute pseudo-residuals (logistic negative gradient)
        for i in 0..n_samples {
            let p = 1.0 / (1.0 + (-raw_preds[i]).exp());
            residuals[i] = labels[i] - p;
        }

        // train/OOB split
        for i in 0..n_samples {
            sample_indices[i] = i as u32;
        }
        for i in 0..n_train {
            let j = rng.random_range(i..n_samples);
            sample_indices.swap(i, j);
        }

        let (train, oob) = sample_indices.split_at_mut(n_train);

        let root_idx = pool.acquire();
        pool.histograms[root_idx].build_from_samples(&store, train, &residuals);

        let mut y_sum = 0.0f32;
        let mut y_sum_sq = 0.0f32;
        for &s in train.iter() {
            let r = residuals[s as usize];
            y_sum += r;
            y_sum_sq += r * r;
        }

        // snapshot residuals before tree modifies them so we can extract what
        // the tree actually predicted
        let pre_residuals: Vec<f32> = residuals.clone();

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
            n_train,
            config,
            0,
            config.learning_rate,
            &mut importances,
            &mut oob_improvement,
            &mut scratch,
            &mut rng,
        );

        // the tree subtracted lr*leaf_pred from residuals.
        // so the tree's contribution is: pre_residuals[i] - residuals[i]
        // accumulate into raw predictions.
        for i in 0..n_samples {
            raw_preds[i] += pre_residuals[i] - residuals[i];
        }

        // early stopping
        let n_oob = n_samples - n_train;
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

    raw_preds
}

/// Apply sigmoid to convert raw GBM scores to probabilities.
///
/// ### Params
///
/// * `raw` - Raw accumulated predictions from `fit_gbm_classifier`.
///
/// ### Returns
///
/// Probabilities in `[0, 1]`.
fn sigmoid(raw: &[f32]) -> Vec<f32> {
    raw.iter().map(|&s| 1.0 / (1.0 + (-s).exp())).collect()
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
        let mut cluster_labels = louvain_sparse_graph(
            &obs_graph,
            self.params.cluster_resolution,
            self.params.cluster_iters,
            seed,
        );

        // -- Multi-scale k values for doublet scoring --
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

        // -- Step 4: Iterative refinement --
        let mut final_scores = vec![0.0f32; self.n_cells];

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

            let n_clusters = *cluster_labels.iter().max().unwrap_or(&0) + 1;

            // 4a. Cluster-aware simulation
            if verbose {
                println!("Simulating cluster-aware doublets...");
            }
            let (pairs, _parent_clusters) = cluster_aware_pairs(
                &cluster_labels,
                self.n_cells_sim,
                self.params.heterotypic_bias,
                iter_seed,
            );

            // Track combined library sizes for simulated doublets
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

            // 4b. Project simulated into PC space
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

            // 4c. kNN on combined at max(k_values)
            if verbose {
                println!("Building combined kNN graph (k={})...", k_max);
            }
            let combined_knn = dispatch_knn(
                combined_pca.as_ref(),
                k_max,
                &self.params.knn_params,
                iter_seed,
                verbose,
            );

            // 4d. Assign cluster labels to simulated cells via majority vote
            let mut combined_clusters = cluster_labels.clone();
            for sim_idx in 0..self.n_cells_sim {
                let global_idx = self.n_cells + sim_idx;
                let neighbours = &combined_knn[global_idx];
                let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
                for &neigh in neighbours {
                    if neigh < self.n_cells {
                        *counts.entry(cluster_labels[neigh]).or_insert(0) += 1;
                    }
                }
                let majority = counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(cl, _)| cl)
                    .unwrap_or(0);
                combined_clusters.push(majority);
            }

            // 4e. Feature engineering
            let n_features_total = k_values.len() + n_clusters + 2 + n_pcs_include;
            if verbose {
                println!("Building feature matrix ({} features)...", n_features_total);
            }
            let features = build_feature_matrix(
                &combined_knn,
                &combined_clusters,
                self.n_cells,
                n_clusters,
                &self.hvg_library_sizes,
                median_lib_size,
                &sim_combined_lib_sizes,
                combined_pca.as_ref(),
                &k_values,
                n_pcs_include,
            );

            // 4f. Train GBM classifier
            if verbose {
                println!("Training gradient-boosted classifier...");
            }
            let n_total = self.n_cells + self.n_cells_sim;
            let labels: Vec<f32> = (0..n_total)
                .map(|i| if i < self.n_cells { 0.0 } else { 1.0 })
                .collect();

            let raw_scores =
                fit_gbm_classifier(features.as_ref(), &labels, &gbm_config, iter_seed + 100);

            // 4g. Extract observed cell scores directly (no sigmoid -- L2 boosting
            //     on 0/1 labels already produces approximate probabilities)
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

            // 4h. Re-cluster for next iteration
            if iter < self.params.n_iterations - 1 {
                let obs_graph = knn_to_sparse_graph(&obs_knn);
                cluster_labels = louvain_sparse_graph(
                    &obs_graph,
                    self.params.cluster_resolution,
                    self.params.cluster_iters,
                    iter_seed + 500,
                );
            }
        }

        // -- Step 5: Threshold --
        // Clamp scores to [0, 1] since L2 boosting can slightly overshoot
        for s in final_scores.iter_mut() {
            *s = s.clamp(0.0, 1.0);
        }

        let threshold = self.params.manual_threshold.unwrap_or_else(|| {
            let t = find_threshold_otsu(&final_scores, self.params.n_bins);
            if verbose {
                println!("Automatically set threshold at score = {:.4}", t);
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
