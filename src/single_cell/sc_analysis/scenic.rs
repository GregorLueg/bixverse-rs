use faer::Mat;
use indexmap::IndexSet;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;

/// How many genes to test for in one go
const SCENIC_GENE_CHUNK_SIZE: usize = 1000;

/// How many target genes to batch into a single multi-output tree ensemble.
/// 64 targets * 256 bins * 16 bytes ≈ 256KB per feature histogram, fits L2
/// comfortably.
const MULTI_OUTPUT_BATCH: usize = 64;

///////////
// Enums //
///////////

/// Which regression learner to use
#[derive(Clone, Debug)]
pub enum RegressionLearner {
    /// ExtraTree regression learner (RF's chaotic cousin)
    ExtraTrees(ExtraTreesConfig),
    /// RandomForest regression learner
    RandomForest(RandomForestConfig),
}

impl Default for RegressionLearner {
    fn default() -> Self {
        RegressionLearner::ExtraTrees(ExtraTreesConfig::default())
    }
}

/// Parse a regression learner from a string
///
/// ### Params
///
/// * `s` - String to parse. Accepts `"extratrees"`, `"rf"`, or
///   `"randomforest"` (case-insensitive).
///
/// ### Returns
///
/// `Some(RegressionLearner)` with default parameters, or `None` if unrecognised.
pub fn parse_regression_learner(s: &str) -> Option<RegressionLearner> {
    match s.to_lowercase().as_str() {
        "extratrees" => Some(RegressionLearner::ExtraTrees(ExtraTreesConfig::default())),
        "rf" | "randomforest" => Some(RegressionLearner::RandomForest(
            RandomForestConfig::default(),
        )),
        _ => None,
    }
}

//////////////////
// Shared trait //
//////////////////

/// Shared configuration interface for tree-based regressors
trait TreeRegressorConfig: Sync {
    /// Number of trees in the ensemble
    fn n_trees(&self) -> usize;

    /// Minimum number of samples required in a leaf node
    fn min_samples_leaf(&self) -> usize;

    /// Number of features to consider at each split.
    /// A value of `0` means use `sqrt(n_features)`.
    fn n_features_split(&self) -> usize;

    /// Whether to use random thresholds (ExtraTrees) rather than optimal
    /// thresholds (RF)
    fn random_threshold(&self) -> bool;

    /// Fraction of samples to use per tree
    ///
    /// ### Returns
    ///
    /// Proportion in `[0, 1]`. Defaults to `1.0` (use all samples).
    fn subsample_rate(&self) -> f32 {
        1.0
    }

    /// Whether to sample with replacement (bootstrapping)
    fn bootstrap(&self) -> bool {
        false
    }

    /// Maximum tree depth. `None` means grow until other stopping criteria
    /// are met.
    fn max_depth(&self) -> Option<usize> {
        None
    }

    /// Minimum node variance below which no split is attempted
    fn min_variance(&self) -> f64 {
        1e-10
    }

    /// Number of random thresholds to evaluate per feature per node
    /// (ExtraTrees only)
    fn n_thresholds(&self) -> usize {
        1
    }

    /// If `Some(frac)`, subsample this fraction of cells per tree.
    /// Overrides `subsample_rate` when set.
    fn subsample_frac(&self) -> Option<f32> {
        None
    }
}

////////////
// Params //
////////////

/// Configuration for an ExtraTrees ensemble
///
/// ### Fields
///
/// * `n_trees` - Number of trees to build.
/// * `min_samples_leaf` - Minimum samples per leaf node.
/// * `n_features_split` - Features considered per split; `0` means
///   `sqrt(n_features)`.
/// * `n_thresholds` - Random thresholds to test per feature per node.
/// * `max_depth` - Maximum tree depth.
/// * `subsample_frac` - Optional fraction of cells to subsample per tree.
#[derive(Clone, Debug)]
pub struct ExtraTreesConfig {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub n_features_split: usize,
    pub n_thresholds: usize,
    pub max_depth: Option<usize>,
    pub subsample_frac: Option<f32>,
}

/// Default implementation for `ExtraTreesConfig`
impl Default for ExtraTreesConfig {
    fn default() -> Self {
        Self {
            n_trees: 500,
            min_samples_leaf: 50,
            n_features_split: 0,
            n_thresholds: 1,
            max_depth: Some(10),
            subsample_frac: None,
        }
    }
}

/// Implementation of TreeRegressorConfig for `ExtraTreesConfig`
impl TreeRegressorConfig for ExtraTreesConfig {
    fn n_trees(&self) -> usize {
        self.n_trees
    }
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }
    fn n_features_split(&self) -> usize {
        self.n_features_split
    }
    fn random_threshold(&self) -> bool {
        true
    }
    fn n_thresholds(&self) -> usize {
        self.n_thresholds
    }
    fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }
    fn subsample_frac(&self) -> Option<f32> {
        self.subsample_frac
    }
}

/// Configuration for a RandomForest ensemble
///
/// ### Fields
///
/// * `n_trees` - Number of trees to build.
/// * `min_samples_leaf` - Minimum samples per leaf node.
/// * `n_features_split` - Features considered per split; `0` means
///   `sqrt(n_features)`.
/// * `subsample_rate` - Fraction of samples to draw per tree (without
///   replacement unless `bootstrap` is set).
/// * `bootstrap` - Whether to sample with replacement.
/// * `max_depth` - Maximum tree depth.
/// * `subsample_frac` - Optional fraction of cells to subsample per tree
///   (overrides `subsample_rate`).
#[derive(Clone, Debug)]
pub struct RandomForestConfig {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub n_features_split: usize,
    pub subsample_rate: f32,
    pub bootstrap: bool,
    pub max_depth: Option<usize>,
    pub subsample_frac: Option<f32>,
}

/// Default implementation for `RandomForestConfig`
impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 200,
            min_samples_leaf: 50,
            n_features_split: 0,
            subsample_rate: 0.632,
            bootstrap: false,
            max_depth: Some(10),
            subsample_frac: None,
        }
    }
}

/// Implementation of TreeRegressorConfig for `RandomForestConfig`
impl TreeRegressorConfig for RandomForestConfig {
    fn n_trees(&self) -> usize {
        self.n_trees
    }
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }
    fn n_features_split(&self) -> usize {
        self.n_features_split
    }
    fn random_threshold(&self) -> bool {
        false
    }
    fn subsample_rate(&self) -> f32 {
        self.subsample_rate
    }
    fn bootstrap(&self) -> bool {
        self.bootstrap
    }
    fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }
    fn subsample_frac(&self) -> Option<f32> {
        self.subsample_frac
    }
}

/////////////////////
// Storage helpers //
/////////////////////

/// Dense column-major store of quantised (u8) feature values
///
/// Stores one byte per cell per feature, with per-feature min/range metadata
/// for reconstructing original values if needed.
///
/// ### Fields
///
/// * `data` - Quantised data
/// * `n_cells` - Number of cells
/// * `n_features` - Number of features
/// * `feature_min` - Minimum value for the feature for reconstruction (not
///   in use atm).
/// * `feature_range` - Feature range for the feature (not in use atm).
#[allow(dead_code)]
pub struct DenseQuantisedStore {
    data: Vec<u8>,
    n_cells: usize,
    pub n_features: usize,
    feature_min: Vec<f32>,
    feature_range: Vec<f32>,
}

impl DenseQuantisedStore {
    /// Build a `DenseQuantisedStore` from a CSC sparse matrix
    ///
    /// ### Params
    ///
    /// * `mat` - Sparse CSC feature matrix (features as columns).
    /// * `n_cells` - Total number of cells (rows) in the matrix.
    ///
    /// ### Returns
    ///
    /// A fully populated `DenseQuantisedStore`.
    ///
    /// ### Implementation details
    ///
    /// Each feature column is independently scaled to `[0, 255]` using its
    /// observed min and max. Zero-valued cells (absent from the sparse
    /// structure) map to bin 0 after subtraction of `min_v`, which is
    /// initialised to `0.0`, so implicit zeros are handled correctly provided
    /// all values are non-negative. Features with range ≤ 1e-10 (effectively
    /// constant) are left at zero.
    pub fn from_csc(mat: &CompressedSparseData<u16, f32>, n_cells: usize) -> Self {
        let n_features = mat.indptr.len() - 1;
        let mut data = vec![0u8; n_features * n_cells];
        let mut mins = Vec::with_capacity(n_features);
        let mut ranges = Vec::with_capacity(n_features);

        let vals = mat.data_2.as_ref().unwrap();

        for j in 0..n_features {
            let s = mat.indptr[j];
            let e = mat.indptr[j + 1];
            let col_indices = &mat.indices[s..e];
            let col_vals = &vals[s..e];

            let mut min_v = 0_f32;
            let mut max_v = 0_f32;
            for &v in col_vals {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            let range = max_v - min_v;
            mins.push(min_v);
            ranges.push(range);

            let offset = j * n_cells;

            if range > 1e-10 {
                let scale = 255.0 / range;
                for i in 0..col_indices.len() {
                    let cell_idx = col_indices[i];
                    let val = col_vals[i];
                    let q_val = ((val - min_v) * scale).round() as u8;
                    data[offset + cell_idx] = q_val;
                }
            }
        }

        Self {
            data,
            n_cells,
            n_features,
            feature_min: mins,
            feature_range: ranges,
        }
    }

    /// Return the quantised values for a single feature (TF) column
    ///
    /// ### Params
    ///
    /// * `tf_idx` - Feature (TF) index.
    ///
    /// ### Returns
    ///
    /// Slice of length `n_cells` containing u8-quantised values for that
    /// feature.
    #[inline(always)]
    pub fn get_col(&self, tf_idx: usize) -> &[u8] {
        let start = tf_idx * self.n_cells;
        &self.data[start..start + self.n_cells]
    }
}

///////////////////////////////
// Multi-output tree buffers //
///////////////////////////////

/// Reusable scratch buffers for building multi-output trees
///
/// All allocations are done once at construction and reused across trees and
/// nodes to avoid repeated heap allocation in the hot path.
///
/// ### Fields
///
/// * `feat_buf` - Feature index permutation buffer for partial Fisher-Yates
///   shuffle.
/// * `left_buf` - Temporary sample indices for the left child during
///   partitioning.
/// * `right_buf` - Temporary sample indices for the right child during
///   partitioning.
/// * `left_y_buf` - Temporary interleaved Y values for the left child; layout
///   `[sample * n_targets + target]`.
/// * `right_y_buf` - Temporary interleaved Y values for the right child; same
///   layout as `left_y_buf`.
/// * `counts` - Per-bin sample counts; layout `counts[bin]`.
/// * `y_sums` - Per-bin, per-target Y sums; layout
///   `y_sums[bin * n_targets + target]`.
/// * `y_sum_sqs` - Per-bin, per-target Y sum-of-squares; same layout as
///   `y_sums`.
/// * `cum_counts` - Prefix-sum of `counts` over bins.
/// * `cum_y_sums` - Prefix-sum of `y_sums` over bins; same layout as `y_sums`.
/// * `cum_y_sum_sqs` - Prefix-sum of `y_sum_sqs` over bins; same layout as
///   `y_sums`.
/// * `best_y_sums_l` - Left-child Y sums captured at the best split.
/// * `best_y_sum_sqs_l` - Left-child Y sum-of-squares captured at the best
///   split.
/// * `parent_vars` - Per-target parent variance scratch space.
struct TreeBuffers {
    feat_buf: Vec<usize>,
    left_buf: Vec<u32>,
    right_buf: Vec<u32>,
    left_y_buf: Vec<f32>,
    right_y_buf: Vec<f32>,
    counts: [usize; 256],
    y_sums: Vec<f64>,
    y_sum_sqs: Vec<f64>,
    cum_counts: [usize; 256],
    cum_y_sums: Vec<f64>,
    cum_y_sum_sqs: Vec<f64>,
    best_y_sums_l: Vec<f64>,
    best_y_sum_sqs_l: Vec<f64>,
    parent_vars: Vec<f64>,
}

impl TreeBuffers {
    /// Allocate all scratch buffers for the given problem dimensions
    ///
    /// ### Params
    ///
    /// * `n_features` - Total number of features (TFs).
    /// * `n_samples` - Maximum number of samples per tree.
    /// * `n_targets` - Number of target genes in the current batch.
    fn new(n_features: usize, n_samples: usize, n_targets: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            left_buf: vec![0; n_samples],
            right_buf: vec![0; n_samples],
            left_y_buf: vec![0.0; n_samples * n_targets],
            right_y_buf: vec![0.0; n_samples * n_targets],
            counts: [0usize; 256],
            y_sums: vec![0.0f64; 256 * n_targets],
            y_sum_sqs: vec![0.0f64; 256 * n_targets],
            cum_counts: [0usize; 256],
            cum_y_sums: vec![0.0f64; 256 * n_targets],
            cum_y_sum_sqs: vec![0.0f64; 256 * n_targets],
            best_y_sums_l: vec![0.0f64; n_targets],
            best_y_sum_sqs_l: vec![0.0f64; n_targets],
            parent_vars: vec![0.0f64; n_targets],
        }
    }

    /// Build bin histograms and their prefix-sum (cumulative) counterparts
    ///
    /// ### Params
    ///
    /// * `tf_col` - Quantised feature column for all cells.
    /// * `sample_slice` - Indices of the active samples in this node.
    /// * `y_slice` - Interleaved target values; layout
    ///   `[sample * n_targets + target]`.
    /// * `n_targets` - Number of targets in the current batch.
    ///
    /// ### Implementation details
    ///
    /// Iterates over `sample_slice` and accumulates each sample into its bin
    /// (`tf_col[sample_idx]`). After the pass, a single forward scan builds the
    /// cumulative counts, sums, and sum-of-squares needed to evaluate any split
    /// threshold in O(1) during the split search.
    #[inline]
    fn build_histograms(
        &mut self,
        tf_col: &[u8],
        sample_slice: &[u32],
        y_slice: &[f32],
        n_targets: usize,
    ) {
        self.counts.fill(0);
        let hist_len = 256 * n_targets;
        self.y_sums[..hist_len].fill(0.0);
        self.y_sum_sqs[..hist_len].fill(0.0);

        for i in 0..sample_slice.len() {
            let bin = tf_col[sample_slice[i] as usize] as usize;
            self.counts[bin] += 1;
            let y_base = i * n_targets;
            let h_base = bin * n_targets;
            for k in 0..n_targets {
                let y = y_slice[y_base + k] as f64;
                self.y_sums[h_base + k] += y;
                self.y_sum_sqs[h_base + k] += y * y;
            }
        }

        self.cum_counts[0] = self.counts[0];
        self.cum_y_sums[..n_targets].copy_from_slice(&self.y_sums[..n_targets]);
        self.cum_y_sum_sqs[..n_targets].copy_from_slice(&self.y_sum_sqs[..n_targets]);

        for b in 1..256 {
            self.cum_counts[b] = self.cum_counts[b - 1] + self.counts[b];
            let prev = (b - 1) * n_targets;
            let curr = b * n_targets;
            for k in 0..n_targets {
                self.cum_y_sums[curr + k] = self.cum_y_sums[prev + k] + self.y_sums[curr + k];
                self.cum_y_sum_sqs[curr + k] =
                    self.cum_y_sum_sqs[prev + k] + self.y_sum_sqs[curr + k];
            }
        }
    }
}

//////////////////
// Tree helpers //
//////////////////

/// Compute the variance of a node from sufficient statistics
///
/// ### Params
///
/// * `sum` - Sum of values in the node.
/// * `sum_sq` - Sum of squared values in the node.
/// * `n` - Number of samples in the node.
///
/// ### Returns
///
/// The variance, clamped to `0.0`. Returns `0.0` for nodes with fewer than 2
/// samples.
#[inline]
fn node_variance_f64(sum: f64, sum_sq: f64, n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    f64::max(0.0, sum_sq / nf - (sum / nf) * (sum / nf))
}

///////////////////
// Tree building //
///////////////////

/// Evaluate one candidate split threshold across all targets and update the
/// best-split state
///
/// ### Params
///
/// * `threshold` - Bin index to split at (samples with bin ≤ threshold go left).
/// * `feat` - Feature index being evaluated.
/// * `parent_vars` - Per-target parent variance.
/// * `n` - Total samples in this node.
/// * `y_sums_total` - Per-target sum of Y in this node.
/// * `y_sum_sqs_total` - Per-target sum of Y² in this node.
/// * `cum_counts` - Cumulative bin sample counts.
/// * `cum_y_sums` - Cumulative per-target Y sums over bins.
/// * `cum_y_sum_sqs` - Cumulative per-target Y² sums over bins.
/// * `n_targets` - Number of targets in the current batch.
/// * `min_samples_leaf` - Minimum samples required per child.
/// * `best_score` - Current best aggregate variance reduction (updated in
///   place).
/// * `best_feature` - Feature index of the current best split (updated in
///   place).
/// * `best_threshold_u8` - Threshold of the current best split (updated in
///   place).
/// * `best_n_left` - Left child size at the current best split (updated in
///   place).
/// * `best_y_sums_l` - Left-child Y sums at the current best split (updated in
///   place).
/// * `best_y_sum_sqs_l` - Left-child Y² sums at the current best split (updated
///   in place).
///
/// ### Implementation details
///
/// The split score is the sum of variance reductions across all targets,
/// weighted by node size relative to the root. Splits that would produce a
/// child smaller than `min_samples_leaf` are rejected immediately. The
/// left-child sufficient statistics are read directly from the cumulative
/// histograms; right-child statistics are derived by subtraction.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_multi(
    threshold: usize,
    feat: usize,
    parent_vars: &[f64],
    n: usize,
    y_sums_total: &[f64],
    y_sum_sqs_total: &[f64],
    cum_counts: &[usize; 256],
    cum_y_sums: &[f64],
    cum_y_sum_sqs: &[f64],
    n_targets: usize,
    min_samples_leaf: usize,
    best_score: &mut f64,
    best_feature: &mut usize,
    best_threshold_u8: &mut u8,
    best_n_left: &mut usize,
    best_y_sums_l: &mut [f64],
    best_y_sum_sqs_l: &mut [f64],
) {
    let n_left = cum_counts[threshold];
    let n_right = n - n_left;

    if n_left < min_samples_leaf || n_right < min_samples_leaf {
        return;
    }

    let nl = n_left as f64;
    let nr = n_right as f64;
    let nf = n as f64;
    let h_base = threshold * n_targets;

    let mut score = 0.0f64;
    for k in 0..n_targets {
        let y_sum_l = cum_y_sums[h_base + k];
        let y_sum_sq_l = cum_y_sum_sqs[h_base + k];
        let y_sum_r = y_sums_total[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;

        let var_l = f64::max(0.0, y_sum_sq_l / nl - (y_sum_l / nl).powi(2));
        let var_r = f64::max(0.0, y_sum_sq_r / nr - (y_sum_r / nr).powi(2));

        score += parent_vars[k] - (nl / nf) * var_l - (nr / nf) * var_r;
    }

    if score > *best_score {
        *best_score = score;
        *best_feature = feat;
        *best_threshold_u8 = threshold as u8;
        *best_n_left = n_left;
        for k in 0..n_targets {
            best_y_sums_l[k] = cum_y_sums[h_base + k];
            best_y_sum_sqs_l[k] = cum_y_sum_sqs[h_base + k];
        }
    }
}

/// Recursively build a single tree node, accumulating feature importance in
/// place
///
/// No tree structure is stored -- importance is the sole output.
///
/// ### Params
///
/// * `y_slice` - Interleaved target values for the active samples; layout
///   `[sample * n_targets + target]`.
/// * `x` - Quantised feature store.
/// * `sample_slice` - Indices of the active samples; partitioned in place
///   around the best split.
/// * `y_sums` - Per-target Y sums for this node.
/// * `y_sum_sqs` - Per-target Y² sums for this node.
/// * `n_total` - Total samples at the tree root (used to weight importance
///   contributions).
/// * `n_targets` - Number of targets in the current batch.
/// * `n_features_split` - Number of features to sample at each split.
/// * `config` - Learner configuration.
/// * `depth` - Current depth in the tree.
/// * `importances` - Accumulated importance array; layout
///   `[feature * n_targets + target]`.
/// * `bufs` - Reusable scratch buffers.
/// * `rng` - Per-tree RNG.
///
/// ### Implementation details
///
/// Stopping criteria are checked first (min samples, min variance, max depth).
/// Features are selected via a partial Fisher-Yates shuffle of `bufs.feat_buf`.
/// For each selected feature, bin histograms are built once and then all
/// candidate thresholds are evaluated in a single pass over the 256 bins. For
/// ExtraTrees, only `n_thresholds` random thresholds are tested per feature
/// instead of all bins.
///
/// After the best split is found, its variance reduction is accumulated into
/// `importances` weighted by `n / n_total`. Samples and `y_slice` are
/// partitioned in place using `bufs.left_buf` / `bufs.right_buf` as temporary
/// storage, and the function recurses into both children.
#[allow(clippy::too_many_arguments)]
fn build_node_multi(
    y_slice: &mut [f32],
    x: &DenseQuantisedStore,
    sample_slice: &mut [u32],
    y_sums: &[f64],
    y_sum_sqs: &[f64],
    n_total: usize,
    n_targets: usize,
    n_features_split: usize,
    config: &dyn TreeRegressorConfig,
    depth: usize,
    importances: &mut [f64],
    bufs: &mut TreeBuffers,
    rng: &mut SmallRng,
) {
    let n = sample_slice.len();

    let mut total_parent_var = 0.0f64;
    for k in 0..n_targets {
        let v = node_variance_f64(y_sums[k], y_sum_sqs[k], n);
        bufs.parent_vars[k] = v;
        total_parent_var += v;
    }

    let max_depth_reached = config.max_depth().map_or(false, |d| depth >= d);

    if n < 2 * config.min_samples_leaf()
        || total_parent_var < config.min_variance()
        || max_depth_reached
    {
        return;
    }

    let n_features = x.n_features;
    let k_feats = n_features_split.min(n_features);

    for i in 0..k_feats {
        let j = rng.random_range(i..n_features);
        bufs.feat_buf.swap(i, j);
    }

    let mut best_score = 0.0f64;
    let mut best_feature = usize::MAX;
    let mut best_threshold_u8 = 0u8;
    let mut best_n_left = 0usize;

    for fi_idx in 0..k_feats {
        let feat = bufs.feat_buf[fi_idx];
        let tf_col = x.get_col(feat);

        bufs.build_histograms(tf_col, sample_slice, y_slice, n_targets);

        let min_bin = bufs.counts.iter().position(|&c| c > 0).unwrap_or(0);
        let max_bin = bufs.counts.iter().rposition(|&c| c > 0).unwrap_or(255);

        if min_bin == max_bin {
            continue;
        }

        if config.random_threshold() {
            for _ in 0..config.n_thresholds() {
                let threshold = rng.random_range(min_bin..max_bin);
                evaluate_split_multi(
                    threshold,
                    feat,
                    &bufs.parent_vars,
                    n,
                    y_sums,
                    y_sum_sqs,
                    &bufs.cum_counts,
                    &bufs.cum_y_sums,
                    &bufs.cum_y_sum_sqs,
                    n_targets,
                    config.min_samples_leaf(),
                    &mut best_score,
                    &mut best_feature,
                    &mut best_threshold_u8,
                    &mut best_n_left,
                    &mut bufs.best_y_sums_l,
                    &mut bufs.best_y_sum_sqs_l,
                );
            }
        } else {
            for threshold in min_bin..max_bin {
                evaluate_split_multi(
                    threshold,
                    feat,
                    &bufs.parent_vars,
                    n,
                    y_sums,
                    y_sum_sqs,
                    &bufs.cum_counts,
                    &bufs.cum_y_sums,
                    &bufs.cum_y_sum_sqs,
                    n_targets,
                    config.min_samples_leaf(),
                    &mut best_score,
                    &mut best_feature,
                    &mut best_threshold_u8,
                    &mut best_n_left,
                    &mut bufs.best_y_sums_l,
                    &mut bufs.best_y_sum_sqs_l,
                );
            }
        }
    }

    if best_feature == usize::MAX {
        return;
    }

    let weight = n as f64 / n_total as f64;
    let nl = best_n_left as f64;
    let nr = (n - best_n_left) as f64;
    let nf = n as f64;
    let imp_base = best_feature * n_targets;

    for k in 0..n_targets {
        let y_sum_l = bufs.best_y_sums_l[k];
        let y_sum_sq_l = bufs.best_y_sum_sqs_l[k];
        let y_sum_r = y_sums[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs[k] - y_sum_sq_l;

        let var_l = f64::max(0.0, y_sum_sq_l / nl - (y_sum_l / nl).powi(2));
        let var_r = f64::max(0.0, y_sum_sq_r / nr - (y_sum_r / nr).powi(2));
        let reduction = bufs.parent_vars[k] - (nl / nf) * var_l - (nr / nf) * var_r;

        importances[imp_base + k] += weight * f64::max(0.0, reduction);
    }

    let mut left_y_sums = vec![0.0f64; n_targets];
    let mut left_y_sum_sqs = vec![0.0f64; n_targets];
    for k in 0..n_targets {
        left_y_sums[k] = bufs.best_y_sums_l[k];
        left_y_sum_sqs[k] = bufs.best_y_sum_sqs_l[k];
    }

    let tf_col = x.get_col(best_feature);
    let mut l_idx = 0usize;
    let mut r_idx = 0usize;

    for i in 0..n {
        let s = sample_slice[i];
        let val = tf_col[s as usize];
        let src_base = i * n_targets;

        if val <= best_threshold_u8 {
            bufs.left_buf[l_idx] = s;
            let dst = l_idx * n_targets;
            bufs.left_y_buf[dst..dst + n_targets]
                .copy_from_slice(&y_slice[src_base..src_base + n_targets]);
            l_idx += 1;
        } else {
            bufs.right_buf[r_idx] = s;
            let dst = r_idx * n_targets;
            bufs.right_y_buf[dst..dst + n_targets]
                .copy_from_slice(&y_slice[src_base..src_base + n_targets]);
            r_idx += 1;
        }
    }

    sample_slice[..l_idx].copy_from_slice(&bufs.left_buf[..l_idx]);
    sample_slice[l_idx..].copy_from_slice(&bufs.right_buf[..r_idx]);

    let y_left_len = l_idx * n_targets;
    let y_right_len = r_idx * n_targets;
    y_slice[..y_left_len].copy_from_slice(&bufs.left_y_buf[..y_left_len]);
    y_slice[y_left_len..y_left_len + y_right_len].copy_from_slice(&bufs.right_y_buf[..y_right_len]);

    let mut right_y_sums = vec![0.0f64; n_targets];
    let mut right_y_sum_sqs = vec![0.0f64; n_targets];
    for k in 0..n_targets {
        right_y_sums[k] = y_sums[k] - left_y_sums[k];
        right_y_sum_sqs[k] = y_sum_sqs[k] - left_y_sum_sqs[k];
    }

    let (left_samples, right_samples) = sample_slice.split_at_mut(l_idx);
    let (left_y, right_y) = y_slice.split_at_mut(y_left_len);

    build_node_multi(
        left_y,
        x,
        left_samples,
        &left_y_sums,
        &left_y_sum_sqs,
        n_total,
        n_targets,
        n_features_split,
        config,
        depth + 1,
        importances,
        bufs,
        rng,
    );
    build_node_multi(
        right_y,
        x,
        right_samples,
        &right_y_sums,
        &right_y_sum_sqs,
        n_total,
        n_targets,
        n_features_split,
        config,
        depth + 1,
        importances,
        bufs,
        rng,
    );
}

/////////////////////
// Dense Y builder //
/////////////////////

/// Build an interleaved dense Y matrix from multiple sparse target columns
///
/// ### Params
///
/// * `targets` - Sparse target gene expression columns.
/// * `n_samples` - Total number of cells.
///
/// ### Returns
///
/// Dense vector of length `n_samples * n_targets` with layout
/// `y[sample * n_targets + target_idx]`. Absent entries default to `0.0`.
fn build_y_dense_multi(targets: &[SparseAxis<u16, f32>], n_samples: usize) -> Vec<f32> {
    let n_targets = targets.len();
    let mut y_dense = vec![0.0f32; n_samples * n_targets];
    for (k, target) in targets.iter().enumerate() {
        let (y_indices, y_data) = target.get_indices_data_2();
        for (i, &idx) in y_indices.iter().enumerate() {
            y_dense[idx * n_targets + k] = y_data[i];
        }
    }
    y_dense
}

//////////////////
// Core fitting //
//////////////////

/// Fit a tree ensemble for a batch of target genes simultaneously
///
/// ### Params
///
/// * `targets` - Sparse target gene expression columns for this batch.
/// * `feature_matrix` - Quantised TF feature store shared across all targets.
/// * `n_samples` - Number of cells.
/// * `config` - Learner configuration.
/// * `seed` - Base seed for reproducibility; each tree gets a derived seed.
///
/// ### Returns
///
/// One importance vector per target: `result[target_idx][feature_idx]`.
/// Importance values are normalised to sum to 1.0 per target.
///
/// ### Implementation details
///
/// All targets in the batch share the same tree structure (same feature splits
/// and thresholds), while each target contributes independently to the split
/// score via summed variance reductions. This amortises feature histogram
/// construction cost across targets. Subsampling follows the learner config:
/// `subsample_frac` takes priority, then `subsample_rate`; bootstrap sampling
/// is supported for RandomForest. Trees are built sequentially within a batch
/// (parallelism happens at the batch level in `run_scenic_grn`).
fn fit_multi_trees(
    targets: &[SparseAxis<u16, f32>],
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    seed: usize,
) -> Vec<Vec<f32>> {
    let n_features = feature_matrix.n_features;
    let n_targets = targets.len();
    let n_features_split = if config.n_features_split() == 0 {
        ((n_features as f64).sqrt() as usize).max(1)
    } else {
        config.n_features_split()
    };

    let n_sub = if let Some(frac) = config.subsample_frac() {
        ((n_samples as f32 * frac).round() as usize).max(2 * config.min_samples_leaf())
    } else if config.subsample_rate() >= 1.0 {
        n_samples
    } else {
        ((n_samples as f32 * config.subsample_rate()).round() as usize)
            .max(2 * config.min_samples_leaf())
    };

    let y_dense = build_y_dense_multi(targets, n_samples);

    let mut sample_indices: Vec<u32> = vec![0; n_samples];
    let mut root_y_buf: Vec<f32> = vec![0.0; n_samples * n_targets];
    let mut bufs = TreeBuffers::new(n_features, n_samples, n_targets);
    let mut importances = vec![0.0f64; n_features * n_targets];
    let mut y_sums_root = vec![0.0f64; n_targets];
    let mut y_sum_sqs_root = vec![0.0f64; n_targets];

    for tree_idx in 0..config.n_trees() {
        let mut rng =
            SmallRng::seed_from_u64(seed.wrapping_add(tree_idx * 6364136223846793005) as u64);

        let active_len = if n_sub < n_samples {
            if config.bootstrap() {
                for i in 0..n_sub {
                    sample_indices[i] = rng.random_range(0..n_samples as u32);
                }
                n_sub
            } else {
                sample_indices
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v = i as u32);
                for i in 0..n_sub {
                    let j = rng.random_range(i..n_samples);
                    sample_indices.swap(i, j);
                }
                n_sub
            }
        } else {
            sample_indices
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = i as u32);
            n_samples
        };

        let active = &mut sample_indices[..active_len];
        let root_y = &mut root_y_buf[..active_len * n_targets];

        y_sums_root.fill(0.0);
        y_sum_sqs_root.fill(0.0);

        for i in 0..active_len {
            let s = active[i] as usize;
            let src_base = s * n_targets;
            let dst_base = i * n_targets;
            root_y[dst_base..dst_base + n_targets]
                .copy_from_slice(&y_dense[src_base..src_base + n_targets]);
            for k in 0..n_targets {
                let y = root_y[dst_base + k] as f64;
                y_sums_root[k] += y;
                y_sum_sqs_root[k] += y * y;
            }
        }

        build_node_multi(
            root_y,
            feature_matrix,
            active,
            &y_sums_root,
            &y_sum_sqs_root,
            active_len,
            n_targets,
            n_features_split,
            config,
            0,
            &mut importances,
            &mut bufs,
            &mut rng,
        );
    }

    let mut result = Vec::with_capacity(n_targets);
    for k in 0..n_targets {
        let mut target_imp = Vec::with_capacity(n_features);
        let mut total = 0.0f64;
        for f in 0..n_features {
            let v = importances[f * n_targets + k];
            total += v;
            target_imp.push(v as f32);
        }
        if total > 0.0 {
            let inv = 1.0 / total as f32;
            target_imp.iter_mut().for_each(|v| *v *= inv);
        }
        result.push(target_imp);
    }

    result
}

//////////
// Main //
//////////

/// Filter genes by minimum total counts and minimum expressed-cell fraction
///
/// ### Params
///
/// * `f_path` - Path to the sparse gene expression file.
/// * `cell_indices` - Indices of the cells to restrict to.
/// * `min_counts` - Minimum total UMI count across selected cells.
/// * `min_cells` - Minimum fraction of selected cells in which the gene must
///   be detected.
/// * `verbose` -
///
/// ### Returns
///
/// Indices of genes passing both filters, in their original order.
pub fn scenic_gene_filter(
    f_path: &str,
    cell_indices: &[usize],
    min_counts: usize,
    min_cells: f32,
    verbose: bool,
) -> Vec<usize> {
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let total_genes = reader.get_header().total_genes;
    let all_gene_indices: Vec<usize> = (0..total_genes).collect();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_indices.len();

    let mut passing = Vec::new();

    for (iter, chunk) in all_gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        let mut gene_chunks = reader.read_gene_parallel(chunk);
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(&cell_set);
        });

        for gene in &gene_chunks {
            let total_counts: u32 = gene.data_raw.iter().map(|&x| x as u32).sum();
            let expressed_fraction = gene.nnz as f32 / n_cells as f32;

            if total_counts >= min_counts as u32 && expressed_fraction >= min_cells {
                passing.push(gene.original_index);
            }
        }
        if verbose {
            println!(
                "Processed chunk {} out of {} for SCENIC inclusion criteria.",
                iter + 1,
                total_genes.div_ceil(SCENIC_GENE_CHUNK_SIZE)
            );
        }
    }

    passing
}

/// Run SCENIC GRN inference and return a TF-by-gene importance matrix
///
/// ### Params
///
/// * `f_path` - Path to the sparse gene expression file.
/// * `cell_indices` - Indices of cells to use.
/// * `gene_indices` - Target gene indices.
/// * `tf_indices` - Transcription factor gene indices (predictors).
/// * `learner` - Regression learner and its configuration.
/// * `seed` - Base random seed for reproducibility.
/// * `verbose` - Print progress and timing to stdout.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` where entry `[i, j]` is the
/// normalised importance of TF `j` for target gene `i`.
///
/// ### Implementation details
///
/// TF expression data is loaded once, filtered to the selected cells, and
/// quantised into a `DenseQuantisedStore`. Target genes are then processed in
/// chunks of `SCENIC_GENE_CHUNK_SIZE`. Within each chunk, targets are further
/// grouped into batches of `MULTI_OUTPUT_BATCH` and fitted as multi-output
/// ensembles via `fit_multi_trees`. Batches within a chunk are parallelised
/// across threads with Rayon.
pub fn run_scenic_grn(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    learner: &RegressionLearner,
    n_parallel_genes: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let start_total = Instant::now();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let start_reading = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let n_parallel_genes = n_parallel_genes.unwrap_or(MULTI_OUTPUT_BATCH);

    let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(tf_indices);
    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let end_reading = start_reading.elapsed();
    let tf_data: CompressedSparseData<u16, f32> =
        from_gene_chunks::<u16>(&gene_chunks, cell_set.len());
    let tf_data = DenseQuantisedStore::from_csc(&tf_data, cell_set.len());

    if verbose {
        println!(
            "Loaded, filtered and quantised TF data (n: {}) to cells of interest in: {:.2?}",
            tf_data.n_features.separate_with_underscores(),
            end_reading
        );
    }

    let n_genes = gene_indices.len();
    let n_tfs = tf_data.n_features;
    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];

    if verbose {
        println!(
            "Processing {} genes per batched Ensembl learner",
            n_parallel_genes
        );
    }

    for (chunk_idx, chunk) in gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        if verbose {
            println!(
                "Processing gene chunk {}/{} ({} genes)",
                chunk_idx + 1,
                n_genes.div_ceil(SCENIC_GENE_CHUNK_SIZE),
                chunk.len()
            );
        }

        let start_chunk = Instant::now();
        let mut gene_chunks_target: Vec<CscGeneChunk> = reader.read_gene_parallel(chunk);
        gene_chunks_target.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(&cell_set);
        });

        let sparse_columns: Vec<SparseAxis<u16, f32>> = gene_chunks_target
            .iter()
            .map(|c| c.to_sparse_axis(cell_set.len()))
            .collect();

        let sub_batches: Vec<&[SparseAxis<u16, f32>]> =
            sparse_columns.chunks(n_parallel_genes).collect();

        let config: &dyn TreeRegressorConfig = match learner {
            RegressionLearner::ExtraTrees(cfg) => cfg,
            RegressionLearner::RandomForest(cfg) => cfg,
        };

        let batch_results: Vec<Vec<Vec<f32>>> = sub_batches
            .par_iter()
            .enumerate()
            .map(|(batch_idx, batch)| {
                let batch_seed = seed.wrapping_add((chunk_idx * 1000 + batch_idx) * 2654435761);
                fit_multi_trees(batch, &tf_data, cell_set.len(), config, batch_seed)
            })
            .collect();

        let base = chunk_idx * SCENIC_GENE_CHUNK_SIZE;
        for (batch_idx, batch_result) in batch_results.into_iter().enumerate() {
            for (local_idx, imp) in batch_result.into_iter().enumerate() {
                let global_idx = base + batch_idx * n_parallel_genes + local_idx;
                importance_scores[global_idx] = imp;
            }
        }

        if verbose {
            println!("  Chunk done in {:.2?}", start_chunk.elapsed());
        }
    }

    if verbose {
        println!(
            "SCENIC GRN inference complete in {:.2?}",
            start_total.elapsed()
        );
    }

    Mat::from_fn(n_genes, n_tfs, |i, j| {
        if j < importance_scores[i].len() {
            importance_scores[i][j]
        } else {
            0.0
        }
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_variance_basic() {
        // values: 1, 2, 3 -> mean=2, var=2/3
        let (sum, sum_sq) = (6.0f64, 14.0f64);
        let v = node_variance_f64(sum, sum_sq, 3);
        assert!((v - 2.0 / 3.0).abs() < 1e-10, "got {v}");
    }

    #[test]
    fn node_variance_uniform() {
        // values: 3, 3, 3 -> var=0
        let (sum, sum_sq) = (9.0f64, 27.0f64);
        let v = node_variance_f64(sum, sum_sq, 3);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn partition_logic_single_target() {
        let tf_col: Vec<u8> = vec![10, 50, 200, 30, 250, 100];
        let sample_slice: Vec<u32> = vec![0, 1, 2, 4];
        let y_slice: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let n_targets = 1;

        let mut left_buf = vec![0u32; 4];
        let mut right_buf = vec![0u32; 4];
        let mut left_y_buf = vec![0.0f32; 4];
        let mut right_y_buf = vec![0.0f32; 4];

        let threshold = 100u8;
        let mut l_idx = 0;
        let mut r_idx = 0;

        for i in 0..sample_slice.len() {
            let s = sample_slice[i];
            let val = tf_col[s as usize];
            let src_base = i * n_targets;

            if val <= threshold {
                left_buf[l_idx] = s;
                let dst = l_idx * n_targets;
                left_y_buf[dst..dst + n_targets]
                    .copy_from_slice(&y_slice[src_base..src_base + n_targets]);
                l_idx += 1;
            } else {
                right_buf[r_idx] = s;
                let dst = r_idx * n_targets;
                right_y_buf[dst..dst + n_targets]
                    .copy_from_slice(&y_slice[src_base..src_base + n_targets]);
                r_idx += 1;
            }
        }

        assert_eq!(l_idx, 2);
        assert_eq!(r_idx, 2);
        assert_eq!(&left_buf[..l_idx], &[0, 1]);
        assert_eq!(&right_buf[..r_idx], &[2, 4]);
        assert_eq!(&left_y_buf[..l_idx], &[1.0, 2.0]);
        assert_eq!(&right_y_buf[..r_idx], &[3.0, 4.0]);
    }

    #[test]
    fn partition_logic_multi_target() {
        let tf_col: Vec<u8> = vec![10, 200, 50];
        let sample_slice: Vec<u32> = vec![0, 1, 2];
        // 2 targets per sample, interleaved: sample0=[1,10], sample1=[2,20], sample2=[3,30]
        let y_slice: Vec<f32> = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let n_targets = 2;

        let mut left_buf = vec![0u32; 3];
        let mut right_buf = vec![0u32; 3];
        let mut left_y_buf = vec![0.0f32; 6];
        let mut right_y_buf = vec![0.0f32; 6];

        let threshold = 100u8;
        let mut l_idx = 0;
        let mut r_idx = 0;

        for i in 0..sample_slice.len() {
            let s = sample_slice[i];
            let val = tf_col[s as usize];
            let src_base = i * n_targets;

            if val <= threshold {
                left_buf[l_idx] = s;
                let dst = l_idx * n_targets;
                left_y_buf[dst..dst + n_targets]
                    .copy_from_slice(&y_slice[src_base..src_base + n_targets]);
                l_idx += 1;
            } else {
                right_buf[r_idx] = s;
                let dst = r_idx * n_targets;
                right_y_buf[dst..dst + n_targets]
                    .copy_from_slice(&y_slice[src_base..src_base + n_targets]);
                r_idx += 1;
            }
        }

        assert_eq!(l_idx, 2);
        assert_eq!(r_idx, 1);
        assert_eq!(&left_buf[..l_idx], &[0, 2]);
        assert_eq!(&right_buf[..r_idx], &[1]);
        assert_eq!(&left_y_buf[..l_idx * n_targets], &[1.0, 10.0, 3.0, 30.0]);
        assert_eq!(&right_y_buf[..r_idx * n_targets], &[2.0, 20.0]);
    }

    #[test]
    fn histogram_build_multi_target() {
        let n_targets = 2;
        let n_features = 1;
        let n_samples = 4;
        let mut bufs = TreeBuffers::new(n_features, n_samples, n_targets);

        let tf_col: Vec<u8> = vec![0, 0, 0, 10, 10, 0];
        let sample_slice: Vec<u32> = vec![0, 1, 3, 4];
        // Interleaved y: sample0=[1,2], sample1=[3,4], sample2=[5,6], sample3=[7,8]
        let y_slice: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        bufs.build_histograms(&tf_col, &sample_slice, &y_slice, n_targets);

        // Bin 0: samples 0,1 -> y=[1,2],[3,4] -> sums=[4,6], sum_sq=[10,20]
        assert_eq!(bufs.counts[0], 2);
        assert!((bufs.y_sums[0 * n_targets + 0] - 4.0).abs() < 1e-10);
        assert!((bufs.y_sums[0 * n_targets + 1] - 6.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[0 * n_targets + 0] - 10.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[0 * n_targets + 1] - 20.0).abs() < 1e-10);

        // Bin 10: samples 3,4 -> y=[5,6],[7,8] -> sums=[12,14], sum_sq=[74,100]
        assert_eq!(bufs.counts[10], 2);
        assert!((bufs.y_sums[10 * n_targets + 0] - 12.0).abs() < 1e-10);
        assert!((bufs.y_sums[10 * n_targets + 1] - 14.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[10 * n_targets + 0] - 74.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[10 * n_targets + 1] - 100.0).abs() < 1e-10);

        // Cumulative at bin 10 should include bins 0..=10
        assert_eq!(bufs.cum_counts[10], 4);
        assert!((bufs.cum_y_sums[10 * n_targets + 0] - 16.0).abs() < 1e-10);
        assert!((bufs.cum_y_sums[10 * n_targets + 1] - 20.0).abs() < 1e-10);
    }
}
