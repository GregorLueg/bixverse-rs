//! Work-in-progress not behaving as desired...

use faer::{Mat, MatRef, concat};
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::graph::community_detections::*;
use crate::graph::graph_structures::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::utils_doublets::*;
use crate::single_cell::sc_utils::{cxds::*, logistic_gbm::*, utils_tree::*};

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
            early_stop_window: 3, // now maps to early_stop_rounds
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

    // threshold sits between the expected_n-th and (expected_n+1)-th
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
///
/// 1. Multi-scale kNN ratios (one per k value)
/// 2. Distance-weighted doublet proportion ("weighted")
/// 3. Ratio variance across k values (stability signal)
/// 4. Library size ratio
/// 5. Number of expressed genes ("nfeatures")
/// 6. Number of genes with count > 2 ("nAbove2")
/// 7. Co-expression doublet score ("cxds_score")
/// 8. Principal components (n_pcs columns)
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
    obs_cxds: &[f32],
    sim_cxds: &[f32],
    combined_pca: MatRef<f32>,
    n_pcs: usize,
) -> Mat<f32> {
    let n_total = knn_indices.len();
    let n_k = k_values.len();
    // k_ratios + weighted + ratio_var + lib_ratio + nfeatures + nAbove2 + cxds + PCs
    let n_feat = n_k + 6 + n_pcs;

    let rows: Vec<Vec<f32>> = (0..n_total)
        .into_par_iter()
        .map(|i| {
            let neighbours = &knn_indices[i];
            let distances = &knn_distances[i];
            let k_max = neighbours.len();

            let mut feats = Vec::with_capacity(n_feat);

            // multi-scale doublet ratios
            for &k in k_values {
                let k_use = k.min(k_max);
                let n_sim = neighbours[..k_use]
                    .iter()
                    .filter(|&&idx| idx >= n_obs)
                    .count();
                feats.push(n_sim as f32 / k_use as f32);
            }

            // distance-weighted doublet proportion
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

            // ratio variance across k values
            let mean_ratio = feats[..n_k].iter().sum::<f32>() / n_k as f32;
            let ratio_var: f32 = feats[..n_k]
                .iter()
                .map(|&r| (r - mean_ratio).powi(2))
                .sum::<f32>()
                / n_k as f32;
            feats.push(ratio_var);

            // library size ratio
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

            // cxds score
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
) -> Vec<f32> {
    let store = QuantisedStore::from_mat(features);
    fit_logistic_gbm(&store, labels, exclude, config, seed)
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

        // -- Step 1c: cxds model --
        if verbose {
            println!("Building cxds co-expression model...");
        }
        let start_cxds = Instant::now();
        let (cxds_model, obs_cxds_gene_sets) = CxdsModel::fit(
            &self.f_path_cell,
            &self.cells_to_keep,
            &hvg_genes,
            CXDS_NTOP,
        );
        let obs_cxds_scores = cxds_model.score(&obs_cxds_gene_sets);

        if verbose {
            let mean_cxds: f32 = obs_cxds_scores.iter().sum::<f32>() / obs_cxds_scores.len() as f32;
            println!(
                "cxds model built in {:.2?} (mean score: {:.2})",
                start_cxds.elapsed(),
                mean_cxds,
            );
        }

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
        let n_k = k_values.len();
        let n_pcs_include = self.params.include_pcs.min(self.params.no_pcs);

        if verbose {
            println!(
                "Using k values {:?} (max {}), including {} PCs as features",
                k_values, k_max, n_pcs_include
            );
        }

        let gbm_config = LogisticGbmConfig {
            max_rounds: self.params.n_trees,
            learning_rate: self.params.learning_rate,
            max_depth: self.params.max_depth,
            min_samples_leaf: self.params.min_samples_leaf,
            early_stop_rounds: self.params.early_stop_window,
            subsample_rate: self.params.subsample_rate,
            ..Default::default()
        };

        // Expected doublet rate for exclusion logic
        // let expected_dbr = self.params.dbr_per_1k * (self.n_cells as f32 / 1000.0);
        let expected_dbr = self.params.dbr_per_1k; // it's already a rate

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

        let (sim_n_features, sim_n_above2) = compute_sim_complexity(&sim_chunks);

        // -- Step 4b: cxds scores for simulated doublets --
        let sim_cxds_scores = cxds_model.score_simulated(&pairs, &obs_cxds_gene_sets);

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
        // k_ratios + weighted + ratio_var + lib_ratio + nfeatures + nAbove2 + cxds + PCs
        let n_feat = n_k + 6 + n_pcs_include;
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
            &obs_cxds_scores,
            &sim_cxds_scores,
            combined_pca.as_ref(),
            n_pcs_include,
        );

        let n_total = self.n_cells + self.n_cells_sim;

        // Labels: false = observed (singlet), true = simulated (doublet)
        let labels: Vec<bool> = (0..n_total).map(|i| i >= self.n_cells).collect();

        // -- Step 8: Iterative training refinement --
        let mut final_scores = vec![0.0f32; self.n_cells];

        // Initial score: use the max-k ratio as a rough doublet estimate
        let max_k_col = n_k - 1;
        for i in 0..self.n_cells {
            final_scores[i] = *features.get(i, max_k_col);
        }

        // We also need per-round scores for simulated doublets to
        // decide which ones to exclude. Initialise from the same
        // kNN ratio column.
        let mut sim_scores = vec![0.0f32; self.n_cells_sim];
        for si in 0..self.n_cells_sim {
            sim_scores[si] = *features.get(self.n_cells + si, max_k_col);
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

            // Build exclusion mask
            let mut exclude_mask = vec![false; n_total];

            if iter > 0 {
                // --- Exclude suspected real doublets ---
                // The top-scoring observed cells from the previous
                // iteration are likely real doublets. Removing them
                // from training gives the classifier a cleaner
                // singlet class.
                let n_exclude_obs = (expected_dbr * self.n_cells as f32).ceil() as usize;
                let n_exclude_obs = n_exclude_obs.min(self.n_cells / 5);

                let mut obs_ranked: Vec<(usize, f32)> = final_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                obs_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut n_excluded = 0usize;
                for &(obs_idx, score) in &obs_ranked {
                    if n_excluded >= n_exclude_obs {
                        break;
                    }
                    // Only exclude cells that actually look like
                    // doublets (probability > 0.5)
                    if score > 0.5 {
                        exclude_mask[obs_idx] = true;
                        n_excluded += 1;
                    } else {
                        break;
                    }
                }

                // --- Exclude unidentifiable artificial doublets ---
                // Simulated doublets that the previous round scored
                // below 0.2 are likely homotypic or otherwise
                // indistinguishable from singlets. Keeping them in
                // training adds noise.
                let n_sim_exclude_max = self.n_cells_sim / 4;
                let mut n_sim_excluded = 0usize;
                for si in 0..self.n_cells_sim {
                    if n_sim_excluded >= n_sim_exclude_max {
                        break;
                    }
                    if sim_scores[si] < 0.2 {
                        exclude_mask[self.n_cells + si] = true;
                        n_sim_excluded += 1;
                    }
                }

                if verbose {
                    println!(
                        "Excluding {} suspected real doublets and {} \
                             unidentifiable artificial doublets from training",
                        n_excluded, n_sim_excluded
                    );
                }
            }

            // Train logistic GBM classifier
            if verbose {
                println!("Training logistic GBM classifier...");
            }
            let probabilities = classify_doublets(
                features.as_ref(),
                &labels,
                &exclude_mask,
                &gbm_config,
                (iter_seed + 100) as u64,
            );

            // Extract observed and simulated scores for next
            // iteration's exclusion logic
            final_scores.copy_from_slice(&probabilities[..self.n_cells]);
            sim_scores.copy_from_slice(&probabilities[self.n_cells..]);

            if verbose {
                let mean_score: f32 = final_scores.iter().sum::<f32>() / final_scores.len() as f32;
                let max_score = final_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let min_score = final_scores.iter().cloned().fold(f32::INFINITY, f32::min);
                println!(
                    "Iteration {} done in {:.2?} (obs scores: \
                         mean={:.4}, min={:.4}, max={:.4})",
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
