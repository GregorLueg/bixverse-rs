//! Contains the SCENIC implementation from Aibar, et al., Nat Methods, 2017.
//! Several modifications were however implemented:
//!
//! a.) Usage of quantisation and histogram-based splitting. This reduces the
//!     size of the predictor variables substantially.
//! b.) Multi-output batching. The original version would create one regression
//!     learner per given gene with the TF expression as predictors. In this
//!     implementation genes are batched together to reduce number of learners
//!     to be trained. This applies to the ExtraTrees and RandomForest learners
//!     where independent trees make shared structure across targets a
//!     reasonable approximation.
//! c.) To ensure that sensible genes are batched together, the module provides
//!     two methods to batch genes together. Completely random to avoid biases
//!     of the gene index order generally speaking (fast, but potentially not
//!     optimal). And SVD on a subset of cells (for very large data sets) with
//!     k-means clustering on the gene loadings to put similar genes together.
//! d.) GRNBoost2-style gradient boosted tree ensembles (Moerman, et al.,
//!     Bioinformatics, 2019). Unlike the RF/ET learners, GBM trees are built
//!     sequentially per target (each fitting residuals from the prior
//!     ensemble), so multi-output batching is not used. Instead, parallelism
//!     is exploited across targets. The implementation uses full-feature
//!     histogram construction with parent-child subtraction (smaller child
//!     built from scratch, larger derived by difference) to minimise
//!     histogram cost in shallow trees. Early stopping via out-of-bag
//!     improvement estimates prevents overfitting and aborts regressions
//!     with little signal early, which is the primary source of speedup
//!     over RF/ET for large gene sets.

use ann_search_rs::prelude::*;
use ann_search_rs::utils::k_means_utils::{assign_all_parallel, train_centroids};
use faer::Mat;
use indexmap::IndexSet;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_processing::pca::pca_on_sc_streaming;
use crate::single_cell::sc_simd::*;

/// How many genes to test for in one go
const SCENIC_GENE_CHUNK_SIZE: usize = 1024;

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
    /// GradientBoosted learner
    GradientBoosting(GradientBoostingConfig),
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
        "boosted" => Some(RegressionLearner::GradientBoosting(
            GradientBoostingConfig::default(),
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
    fn min_variance(&self) -> f32 {
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
    /// Number of trees to build.
    pub n_trees: usize,
    /// Minimum samples per leaf node.
    pub min_samples_leaf: usize,
    /// Features considered per split; `0` means `sqrt(n_features)`.
    pub n_features_split: usize,
    /// Random thresholds to test per feature per node.
    pub n_thresholds: usize,
    /// Maximum tree depth.
    pub max_depth: Option<usize>,
    /// Optional fraction of cells to subsample per tree.
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
    /// Number of trees to build.
    pub n_trees: usize,
    /// Minimum samples per leaf node.
    pub min_samples_leaf: usize,
    /// Features considered per split; `0` means `sqrt(n_features)`.
    pub n_features_split: usize,
    /// Fraction of samples to draw per tree (without replacement unless
    /// `bootstrap` is set).
    pub subsample_rate: f32,
    /// Whether to sample with replacement.
    pub bootstrap: bool,
    /// Maximum tree depth.
    pub max_depth: Option<usize>,
    /// Optional fraction of cells to subsample per tree (overrides
    /// `subsample_rate`).
    pub subsample_frac: Option<f32>,
}

/// Default implementation for `RandomForestConfig`
impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 250,
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

/// Configuration for a GRNBoost2-style gradient boosted tree ensemble.
///
/// Single-target only. Trees are built sequentially, each fitting the
/// residuals from the prior ensemble. Early stopping via OOB improvement
/// prevents overfitting and is the main source of speedup over RF/ET.
#[derive(Clone, Debug)]
pub struct GradientBoostingConfig {
    /// Maximum number of boosting rounds (early stopping usually triggers
    /// well before this).
    pub n_trees_max: usize,
    /// Shrinkage applied to each tree's predictions.
    pub learning_rate: f32,
    /// Maximum tree depth. Shallow trees (3-5) work best for GBM.
    pub max_depth: usize,
    /// Minimum training samples per leaf node.
    pub min_samples_leaf: usize,
    /// Number of recent OOB improvements to average for the early stopping
    /// criterion. Stops when the rolling average drops to zero or below.
    pub early_stop_window: usize,
    /// Fraction of samples used for training each tree. The complement
    /// forms the OOB set used for early stopping evaluation.
    pub subsample_rate: f32,
    /// Features to evaluate per split; `0` means all features (recommended
    /// with histogram subtraction).
    pub n_features_split: usize,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_trees_max: 1000,
            learning_rate: 0.01,
            max_depth: 3,
            min_samples_leaf: 50,
            early_stop_window: 25,
            subsample_rate: 0.9,
            n_features_split: 0,
        }
    }
}

/////////////////////
// Storage helpers //
/////////////////////

/// Dense column-major store of quantised (u8) feature values
///
/// Stores one byte per cell per feature, with per-feature min/range metadata
/// for reconstructing original values if needed.
#[allow(dead_code)]
pub struct DenseQuantisedStore {
    /// Quantised data
    pub data: Vec<u8>,
    /// Number of cells
    pub n_cells: usize,
    /// Number of features
    pub n_features: usize,
    /// Minimum value for the feature for reconstruction (not in use atm).
    pub feature_min: Vec<f32>,
    /// Feature range for the feature (not in use atm).
    pub feature_range: Vec<f32>,
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
    pub fn from_csc(mat: &CompressedSparseData2<u16, f32>, n_cells: usize) -> Self {
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

//////////////////////
// Sparse Y builder //
//////////////////////

/// Static sparse Y representation for a batch of targets.
///
/// Indexed by original cell ID - remains constant across all trees and nodes.
/// Only sample index arrays are partitioned during tree building.
pub struct SparseYBatch {
    /// Start offset into `target_indices` and `values` for each cell. Length:
    /// n_cells + 1. Cell `c` has entries at `offsets[c]..offsets[c + 1]`.
    pub offsets: Vec<u32>,
    /// Which target within the batch is non-zero (0..n_targets). Since
    /// n_targets <= 64, u8 suffices.
    pub target_indices: Vec<u8>,
    /// Corresponding expression values.
    pub values: Vec<f32>,
}

impl SparseYBatch {
    /// Build from the sparse target columns for one batch.
    ///
    /// ### Params
    ///
    /// * `targets` - Sparse columns for each target in this batch.
    /// * `n_cells` - Total number of cells (determines offsets length).
    ///
    /// ### Returns
    ///
    /// A `SparseYBatch` where entries are sorted by cell, then by target
    /// index within each cell.
    fn from_targets(targets: &[SparseAxis<u16, f32>], n_cells: usize) -> Self {
        // count non-zeros per cell across all targets
        let mut counts_per_cell = vec![0u32; n_cells];
        for target in targets {
            let (indices, _) = target.get_indices_data_2();
            for &idx in indices {
                counts_per_cell[idx] += 1;
            }
        }

        // build offsets via prefix sum
        let mut offsets = Vec::with_capacity(n_cells + 1);
        offsets.push(0u32);
        let mut running = 0u32;
        for &c in &counts_per_cell {
            running += c;
            offsets.push(running);
        }
        let total_nnz = running as usize;

        let mut target_indices = vec![0u8; total_nnz];
        let mut values = vec![0.0f32; total_nnz];

        // fill using a write cursor per cell
        let mut cursor = vec![0u32; n_cells];
        for (k, target) in targets.iter().enumerate() {
            let (indices, data) = target.get_indices_data_2();
            for (i, &cell) in indices.iter().enumerate() {
                let pos = (offsets[cell] + cursor[cell]) as usize;
                target_indices[pos] = k as u8;
                values[pos] = data[i];
                cursor[cell] += 1;
            }
        }

        Self {
            offsets,
            target_indices,
            values,
        }
    }

    /// Iterate non-zero entries for a given cell.
    ///
    /// ### Params
    ///
    /// * `cell` - Cell index
    ///
    /// ### Returns
    ///
    /// Returns the target indices and values
    #[inline(always)]
    fn cell_entries(&self, cell: usize) -> (&[u8], &[f32]) {
        let s = self.offsets[cell] as usize;
        let e = self.offsets[cell + 1] as usize;
        (&self.target_indices[s..e], &self.values[s..e])
    }
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
#[allow(dead_code)]
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

///////////////////////////////
// Multi-output tree buffers //
///////////////////////////////

/// Reusable scratch buffers for building multi-output trees
///
/// All allocations are done once at construction and reused across trees and
/// nodes to avoid repeated heap allocation in the hot path.
#[allow(dead_code)]
struct TreeBuffers {
    /// Feature index permutation buffer for partial Fisher-Yates shuffle.
    feat_buf: Vec<usize>,
    /// Temporary sample indices for the left child during partitioning.
    left_buf: Vec<u32>,
    /// Temporary sample indices for the right child during partitioning.
    right_buf: Vec<u32>,
    /// Temporary interleaved Y values for the left child; layout
    /// `[sample * n_targets + target]`.
    left_y_buf: Vec<f32>,
    /// Temporary interleaved Y values for the right child; same layout as
    /// `left_y_buf`.
    right_y_buf: Vec<f32>,
    /// Per-bin sample counts; layout `counts[bin]`.
    counts: [u32; 256],
    /// Per-bin, per-target Y sums; layout
    y_sums: Vec<f32>,
    /// Per-bin, per-target Y sum-of-squares; same layout as `y_sums`.
    y_sum_sqs: Vec<f32>,
    /// Prefix-sum of `counts` over bins.
    cum_counts: [u32; 256],
    /// Prefix-sum of `y_sums` over bins; same layout as `y_sums`.
    cum_y_sums: Vec<f32>,
    /// Prefix-sum of `y_sum_sqs` over bins; same layout as `y_sums`.
    cum_y_sum_sqs: Vec<f32>,
    /// Left-child Y sums captured at the best split.
    best_y_sums_l: Vec<f32>,
    /// Left-child Y sum-of-squares captured at the best split.
    best_y_sum_sqs_l: Vec<f32>,
    /// Per-target parent variance scratch space.
    parent_vars: Vec<f32>,
}

impl TreeBuffers {
    /// Allocate all scratch buffers for the given problem dimensions
    ///
    /// ### Params
    ///
    /// * `n_features` - Total number of features (TFs).
    /// * `n_samples` - Maximum number of samples per tree.
    /// * `n_targets` - Number of target genes in the current batch.
    #[allow(dead_code)]
    fn new(n_features: usize, n_samples: usize, n_targets: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            left_buf: vec![0; n_samples],
            right_buf: vec![0; n_samples],
            left_y_buf: vec![0.0; n_samples * n_targets],
            right_y_buf: vec![0.0; n_samples * n_targets],
            counts: [0; 256],
            y_sums: vec![0_f32; 256 * n_targets],
            y_sum_sqs: vec![0_f32; 256 * n_targets],
            cum_counts: [0; 256],
            cum_y_sums: vec![0_f32; 256 * n_targets],
            cum_y_sum_sqs: vec![0_f32; 256 * n_targets],
            best_y_sums_l: vec![0_f32; n_targets],
            best_y_sum_sqs_l: vec![0_f32; n_targets],
            parent_vars: vec![0_f32; n_targets],
        }
    }

    /// Generate a tree buffer for SparseYBatch
    ///
    /// ### Params
    ///
    /// * `n_features` - Number of features
    /// * `n_samples` - Number of samples
    /// * `n_targets` - Number of target variables (do not go over 64 here!)
    ///
    /// ### Returns
    ///
    /// Initialised self
    fn new_sparse(n_features: usize, n_samples: usize, n_targets: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            left_buf: vec![0; n_samples],
            right_buf: vec![0; n_samples],
            left_y_buf: Vec::new(),
            right_y_buf: Vec::new(),
            counts: [0; 256],
            y_sums: vec![0_f32; 256 * n_targets],
            y_sum_sqs: vec![0_f32; 256 * n_targets],
            cum_counts: [0; 256],
            cum_y_sums: vec![0_f32; 256 * n_targets],
            cum_y_sum_sqs: vec![0_f32; 256 * n_targets],
            best_y_sums_l: vec![0_f32; n_targets],
            best_y_sum_sqs_l: vec![0_f32; n_targets],
            parent_vars: vec![0_f32; n_targets],
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
    #[allow(dead_code)]
    fn build_histograms(
        &mut self,
        tf_col: &[u8],
        sample_slice: &[u32],
        y_slice: &[f32],
        n_targets: usize,
    ) -> (usize, usize) {
        if sample_slice.is_empty() {
            return (0, 0);
        }

        let mut min_bin = 255u8;
        let mut max_bin = 0u8;
        for &s in sample_slice {
            let bin = tf_col[s as usize];
            if bin < min_bin {
                min_bin = bin;
            }
            if bin > max_bin {
                max_bin = bin;
            }
        }

        let min_b = min_bin as usize;
        let max_b = max_bin as usize;

        self.counts[min_b..=max_b].fill(0);
        let start_idx = min_b * n_targets;
        let end_idx = (max_b + 1) * n_targets;
        self.y_sums[start_idx..end_idx].fill(0.0);
        self.y_sum_sqs[start_idx..end_idx].fill(0.0);

        for i in 0..sample_slice.len() {
            let bin = tf_col[sample_slice[i] as usize] as usize;
            self.counts[bin] += 1;
            let y_base = i * n_targets;
            let h_base = bin * n_targets;
            for k in 0..n_targets {
                let y = y_slice[y_base + k];
                self.y_sums[h_base + k] += y;
                self.y_sum_sqs[h_base + k] += y * y;
            }
        }

        self.cum_counts[min_b] = self.counts[min_b];
        let start_h = min_b * n_targets;
        let next_h = (min_b + 1) * n_targets;
        self.cum_y_sums[start_h..next_h].copy_from_slice(&self.y_sums[start_h..next_h]);
        self.cum_y_sum_sqs[start_h..next_h].copy_from_slice(&self.y_sum_sqs[start_h..next_h]);

        for b in (min_b + 1)..=max_b {
            self.cum_counts[b] = self.cum_counts[b - 1] + self.counts[b];
            let prev = (b - 1) * n_targets;
            let curr = b * n_targets;
            for k in 0..n_targets {
                self.cum_y_sums[curr + k] = self.cum_y_sums[prev + k] + self.y_sums[curr + k];
                self.cum_y_sum_sqs[curr + k] =
                    self.cum_y_sum_sqs[prev + k] + self.y_sum_sqs[curr + k];
            }
        }

        (min_b, max_b)
    }

    /// Build histograms from sparse Y -- only touches non-zero target entries.
    ///
    /// Counts are accumulated for all samples (target-independent). Y sums and
    /// sum-of-squares are accumulated only for non-zero entries; zero targets
    /// contribute nothing and are skipped.
    ///
    /// * `tf_col` - Quantised feature column for all cells.
    /// * `sample_slice` - Indices of the active samples in this node.
    /// * `sparse_y` - Sparse representation of the targets
    /// * `n_targets` - Number of targets in the current batch.
    #[inline]
    fn build_histograms_sparse(
        &mut self,
        tf_col: &[u8],
        sample_slice: &[u32],
        sparse_y: &SparseYBatch,
        n_targets: usize,
    ) -> (usize, usize) {
        if sample_slice.is_empty() {
            return (0, 0);
        }

        // 1. Blazingly fast O(N) pre-pass to find the active bin range
        let mut min_bin = 255u8;
        let mut max_bin = 0u8;
        for &s in sample_slice {
            let bin = tf_col[s as usize];
            if bin < min_bin {
                min_bin = bin;
            }
            if bin > max_bin {
                max_bin = bin;
            }
        }

        let min_b = min_bin as usize;
        let max_b = max_bin as usize;

        // 2. Only zero out the dirtied slice (The Bandwidth Fix)
        self.counts[min_b..=max_b].fill(0);
        let start_idx = min_b * n_targets;
        let end_idx = (max_b + 1) * n_targets;
        self.y_sums[start_idx..end_idx].fill(0.0);
        self.y_sum_sqs[start_idx..end_idx].fill(0.0);

        // 3. Accumulate only within the active bounds
        for &s in sample_slice {
            let bin = tf_col[s as usize] as usize;
            self.counts[bin] += 1;

            let (tgt_indices, tgt_values) = sparse_y.cell_entries(s as usize);
            let h_base = bin * n_targets;
            for i in 0..tgt_indices.len() {
                let k = tgt_indices[i] as usize;
                let y = tgt_values[i];
                self.y_sums[h_base + k] += y;
                self.y_sum_sqs[h_base + k] += y * y;
            }
        }

        // 4. Prefix sums strictly over the active range
        self.cum_counts[min_b] = self.counts[min_b];
        let start_h = min_b * n_targets;
        let next_h = (min_b + 1) * n_targets;
        self.cum_y_sums[start_h..next_h].copy_from_slice(&self.y_sums[start_h..next_h]);
        self.cum_y_sum_sqs[start_h..next_h].copy_from_slice(&self.y_sum_sqs[start_h..next_h]);

        for b in (min_b + 1)..=max_b {
            self.cum_counts[b] = self.cum_counts[b - 1] + self.counts[b];
            let prev = (b - 1) * n_targets;
            let curr = b * n_targets;

            // copy current bin values into cumulative position
            self.cum_y_sums[curr..curr + n_targets]
                .copy_from_slice(&self.y_sums[curr..curr + n_targets]);
            self.cum_y_sum_sqs[curr..curr + n_targets]
                .copy_from_slice(&self.y_sum_sqs[curr..curr + n_targets]);

            // for n_targets <= 64 this fits on the stack comfortably.
            let mut prev_sums = [0_f32; MULTI_OUTPUT_BATCH];
            let mut prev_sum_sqs = [0_f32; MULTI_OUTPUT_BATCH];
            prev_sums[..n_targets].copy_from_slice(&self.cum_y_sums[prev..prev + n_targets]);
            prev_sum_sqs[..n_targets].copy_from_slice(&self.cum_y_sum_sqs[prev..prev + n_targets]);

            accumulate_f32_simd(
                &mut self.cum_y_sums[curr..curr + n_targets],
                &prev_sums[..n_targets],
                n_targets,
            );
            accumulate_f32_simd(
                &mut self.cum_y_sum_sqs[curr..curr + n_targets],
                &prev_sum_sqs[..n_targets],
                n_targets,
            );
        }

        (min_b, max_b)
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
fn node_variance_f32(sum: f32, sum_sq: f32, n: usize) -> f32 {
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    f32::max(0.0, sum_sq / nf - (sum / nf) * (sum / nf))
}

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
    parent_vars: &[f32],
    n: usize,
    y_sums_total: &[f32],
    y_sum_sqs_total: &[f32],
    cum_counts: &[u32; 256],
    cum_y_sums: &[f32],
    cum_y_sum_sqs: &[f32],
    n_targets: usize,
    min_samples_leaf: u32,
    best_score: &mut f32,
    best_feature: &mut usize,
    best_threshold_u8: &mut u8,
    best_n_left: &mut u32,
    best_y_sums_l: &mut [f32],
    best_y_sum_sqs_l: &mut [f32],
) {
    let n_left = cum_counts[threshold];
    let n_right = n as u32 - n_left;

    if n_left < min_samples_leaf || n_right < min_samples_leaf {
        return;
    }

    let nl = n_left as f32;
    let nr = n_right as f32;
    let nf = n as f32;
    let h_base = threshold * n_targets;

    let score = evaluate_split_score_f32_simd(
        parent_vars,
        y_sums_total,
        y_sum_sqs_total,
        &cum_y_sums[h_base..h_base + n_targets],
        &cum_y_sum_sqs[h_base..h_base + n_targets],
        n_targets,
        1.0 / nl,
        1.0 / nr,
        nl / nf,
        nr / nf,
    );

    if score > *best_score {
        *best_score = score;
        *best_feature = feat;
        *best_threshold_u8 = threshold as u8;
        *best_n_left = n_left;
        best_y_sums_l.copy_from_slice(&cum_y_sums[h_base..h_base + n_targets]);
        best_y_sum_sqs_l.copy_from_slice(&cum_y_sum_sqs[h_base..h_base + n_targets]);
    }
}

///////////////////
// Tree building //
///////////////////

/// Recursively build a single tree node using sparse Y.
///
/// Only sample_slice is partitioned in place. Y lookups go through the static
/// SparseYBatch. Histograms are built once per candidate feature; both the
/// ExtraTrees (random threshold) and RF (exhaustive) paths evaluate splits via
/// the cumulative histogram in O(1) per threshold.
///
/// ### Params
///
/// * `sparse_y` - SparseYBatch structure
/// * `x` - Quantised feature store.
/// * `sample_slice` - Indices of the active samples; partitioned in place
///   around the best split.
/// * `y_sums` - Per-target Y sums for this node.
/// * `y_sum_sqs` - Per-target Y squared sums for this node.
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
#[allow(clippy::too_many_arguments)]
fn build_node_multi_sparse(
    sparse_y: &SparseYBatch,
    x: &DenseQuantisedStore,
    sample_slice: &mut [u32],
    y_sums: &[f32],
    y_sum_sqs: &[f32],
    n_total: usize,
    n_targets: usize,
    n_features_split: usize,
    config: &dyn TreeRegressorConfig,
    depth: usize,
    importances: &mut [f32],
    bufs: &mut TreeBuffers,
    rng: &mut SmallRng,
) {
    let n = sample_slice.len();

    let mut total_parent_var = 0.0f32;
    for k in 0..n_targets {
        let v = node_variance_f32(y_sums[k], y_sum_sqs[k], n);
        bufs.parent_vars[k] = v;
        total_parent_var += v;
    }

    let max_depth_reached = config.max_depth().is_some_and(|d| depth >= d);
    if n < 2 * config.min_samples_leaf()
        || (total_parent_var) < config.min_variance()
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

    let mut best_score = 0.0f32;
    let mut best_feature = usize::MAX;
    let mut best_threshold_u8 = 0u8;
    let mut best_n_left = 0u32;

    for fi_idx in 0..k_feats {
        let feat = bufs.feat_buf[fi_idx];
        let tf_col = x.get_col(feat);

        let (min_bin, max_bin) =
            bufs.build_histograms_sparse(tf_col, sample_slice, sparse_y, n_targets);

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
                    config.min_samples_leaf() as u32,
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
                    config.min_samples_leaf() as u32,
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

    // Accumulate importance
    let nl = best_n_left as f32;
    let nr = (n as u32 - best_n_left) as f32;
    let nf = n as f32;
    let weight = nf / n_total as f32;
    let imp_base = best_feature * n_targets;

    for k in 0..n_targets {
        let y_sum_l = bufs.best_y_sums_l[k];
        let y_sum_sq_l = bufs.best_y_sum_sqs_l[k];
        let y_sum_r = y_sums[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs[k] - y_sum_sq_l;
        let mean_l = y_sum_l / nl;
        let var_l = f32::max(0.0, y_sum_sq_l / nl - mean_l * mean_l);
        let mean_r = y_sum_r / nr;
        let var_r = f32::max(0.0, y_sum_sq_r / nr - mean_r * mean_r);
        let reduction = bufs.parent_vars[k] - (nl / nf) * var_l - (nr / nf) * var_r;
        importances[imp_base + k] += weight * f32::max(0.0, reduction);
    }

    // Partition
    let mut left_y_sums = [0.0f32; MULTI_OUTPUT_BATCH];
    let mut left_y_sum_sqs = [0.0f32; MULTI_OUTPUT_BATCH];
    left_y_sums[..n_targets].copy_from_slice(&bufs.best_y_sums_l[..n_targets]);
    left_y_sum_sqs[..n_targets].copy_from_slice(&bufs.best_y_sum_sqs_l[..n_targets]);

    let mut right_y_sums = [0.0f32; MULTI_OUTPUT_BATCH];
    let mut right_y_sum_sqs = [0.0f32; MULTI_OUTPUT_BATCH];
    for k in 0..n_targets {
        right_y_sums[k] = y_sums[k] - left_y_sums[k];
        right_y_sum_sqs[k] = y_sum_sqs[k] - left_y_sum_sqs[k];
    }

    let tf_col = x.get_col(best_feature);
    let mut l_idx = 0usize;
    let mut r_idx = 0usize;

    for i in 0..n {
        let s = sample_slice[i];
        if tf_col[s as usize] <= best_threshold_u8 {
            bufs.left_buf[l_idx] = s;
            l_idx += 1;
        } else {
            bufs.right_buf[r_idx] = s;
            r_idx += 1;
        }
    }

    sample_slice[..l_idx].copy_from_slice(&bufs.left_buf[..l_idx]);
    sample_slice[l_idx..].copy_from_slice(&bufs.right_buf[..r_idx]);

    let (left_samples, right_samples) = sample_slice.split_at_mut(l_idx);

    build_node_multi_sparse(
        sparse_y,
        x,
        left_samples,
        &left_y_sums[..n_targets],
        &left_y_sum_sqs[..n_targets],
        n_total,
        n_targets,
        n_features_split,
        config,
        depth + 1,
        importances,
        bufs,
        rng,
    );
    build_node_multi_sparse(
        sparse_y,
        x,
        right_samples,
        &right_y_sums[..n_targets],
        &right_y_sum_sqs[..n_targets],
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
#[allow(dead_code)]
fn build_node_multi(
    y_slice: &mut [f32],
    x: &DenseQuantisedStore,
    sample_slice: &mut [u32],
    y_sums: &[f32],
    y_sum_sqs: &[f32],
    n_total: usize,
    n_targets: usize,
    n_features_split: usize,
    config: &dyn TreeRegressorConfig,
    depth: usize,
    importances: &mut [f32],
    bufs: &mut TreeBuffers,
    rng: &mut SmallRng,
) {
    let n = sample_slice.len();

    let mut total_parent_var = 0.0f32;
    for k in 0..n_targets {
        let v = node_variance_f32(y_sums[k], y_sum_sqs[k], n);
        bufs.parent_vars[k] = v;
        total_parent_var += v;
    }

    let max_depth_reached = config.max_depth().is_some_and(|d| depth >= d);

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

    let mut best_score = 0.0f32;
    let mut best_feature = usize::MAX;
    let mut best_threshold_u8 = 0u8;
    let mut best_n_left = 0u32;

    for fi_idx in 0..k_feats {
        let feat = bufs.feat_buf[fi_idx];
        let tf_col = x.get_col(feat);

        let (min_bin, max_bin) = bufs.build_histograms(tf_col, sample_slice, y_slice, n_targets);

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
                    config.min_samples_leaf() as u32,
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
                    config.min_samples_leaf() as u32,
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

    let nl = best_n_left as f32;
    let nr = (n as u32 - best_n_left) as f32;
    let nf = n as f32;
    let weight = nf / n_total as f32;
    let imp_base = best_feature * n_targets;

    for k in 0..n_targets {
        let y_sum_l = bufs.best_y_sums_l[k];
        let y_sum_sq_l = bufs.best_y_sum_sqs_l[k];
        let y_sum_r = y_sums[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs[k] - y_sum_sq_l;
        let mean_l = y_sum_l / nl;
        let var_l = f32::max(0.0, y_sum_sq_l / nl - mean_l * mean_l);
        let mean_r = y_sum_r / nr;
        let var_r = f32::max(0.0, y_sum_sq_r / nr - mean_r * mean_r);
        let reduction = bufs.parent_vars[k] - (nl / nf) * var_l - (nr / nf) * var_r;
        importances[imp_base + k] += weight * f32::max(0.0, reduction);
    }

    let mut left_y_sums = [0.0f32; MULTI_OUTPUT_BATCH];
    let mut left_y_sum_sqs = [0.0f32; MULTI_OUTPUT_BATCH];
    let mut right_y_sums = [0.0f32; MULTI_OUTPUT_BATCH];
    let mut right_y_sum_sqs = [0.0f32; MULTI_OUTPUT_BATCH];

    left_y_sums[..n_targets].copy_from_slice(&bufs.best_y_sums_l[..n_targets]);
    left_y_sum_sqs[..n_targets].copy_from_slice(&bufs.best_y_sum_sqs_l[..n_targets]);
    for k in 0..n_targets {
        right_y_sums[k] = y_sums[k] - left_y_sums[k];
        right_y_sum_sqs[k] = y_sum_sqs[k] - left_y_sum_sqs[k];
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

    let (left_samples, right_samples) = sample_slice.split_at_mut(l_idx);
    let (left_y, right_y) = y_slice.split_at_mut(y_left_len);

    build_node_multi(
        left_y,
        x,
        left_samples,
        &left_y_sums[..n_targets],
        &left_y_sum_sqs[..n_targets],
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
        &right_y_sums[..n_targets],
        &right_y_sum_sqs[..n_targets],
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
#[allow(dead_code)]
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
        ((n_features as f32).sqrt() as usize).max(1)
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
    let mut importances = vec![0.0f32; n_features * n_targets];
    let mut y_sums_root = vec![0.0f32; n_targets];
    let mut y_sum_sqs_root = vec![0.0f32; n_targets];

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
                let y = root_y[dst_base + k];
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
        let mut total = 0.0f32;
        for f in 0..n_features {
            let v = importances[f * n_targets + k];
            total += v;
            target_imp.push(v);
        }
        if total > 0.0 {
            let inv = 1.0 / total;
            target_imp.iter_mut().for_each(|v| *v *= inv);
        }
        result.push(target_imp);
    }
    result
}

/// Fit a tree ensemble for a batch of target genes simultaneously (sparse)
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
fn fit_multi_trees_sparse(
    targets: &[SparseAxis<u16, f32>],
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    seed: usize,
) -> Vec<Vec<f32>> {
    let n_features = feature_matrix.n_features;
    let n_targets = targets.len();
    let n_features_split = if config.n_features_split() == 0 {
        ((n_features as f32).sqrt() as usize).max(1)
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

    let sparse_y = SparseYBatch::from_targets(targets, n_samples);

    let mut sample_indices: Vec<u32> = vec![0; n_samples];
    let mut bufs = TreeBuffers::new_sparse(n_features, n_samples, n_targets);
    let mut importances = vec![0.0f32; n_features * n_targets];
    let mut y_sums_root = vec![0.0f32; n_targets];
    let mut y_sum_sqs_root = vec![0.0f32; n_targets];

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

        y_sums_root.fill(0.0);
        y_sum_sqs_root.fill(0.0);
        for &s in active.iter() {
            let (tgt_idx, tgt_val) = sparse_y.cell_entries(s as usize);
            for i in 0..tgt_idx.len() {
                let k = tgt_idx[i] as usize;
                let y = tgt_val[i];
                y_sums_root[k] += y;
                y_sum_sqs_root[k] += y * y;
            }
        }

        build_node_multi_sparse(
            &sparse_y,
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
        let mut total = 0.0f32;
        for f in 0..n_features {
            let v = importances[f * n_targets + k];
            total += v;
            target_imp.push(v);
        }
        if total > 0.0 {
            let inv = 1.0 / total;
            target_imp.iter_mut().for_each(|v| *v *= inv);
        }
        result.push(target_imp);
    }
    result
}

/////////
// GBM //
/////////

//////////////////////
// GBM Split Result //
//////////////////////

/// Information about a split found in a node's histograms.
#[derive(Clone, Debug)]
pub struct GbmSplitInfo {
    /// Feature index of the best split.
    pub feature: usize,
    /// Bin threshold; samples with bin <= threshold go left.
    pub threshold: u8,
    /// Weighted variance reduction score.
    pub score: f32,
    /// Number of training samples in the left child.
    pub n_left: u32,
    /// Sum of residuals in the left child.
    pub y_sum_left: f32,
    /// Sum of squared residuals in the left child.
    pub y_sum_sq_left: f32,
}

/////////////////////
// Node Histograms //
/////////////////////

/// Full-feature bin histograms for a single GBM tree node (single target).
///
/// Stores per-bin sample counts, residual sums, and residual sum-of-squares
/// for every feature simultaneously. This enables the histogram subtraction
/// trick: build the smaller child's histogram from scratch, derive the larger
/// child as `parent - smaller` in O(n_features * 256) rather than scanning
/// the larger child's samples.
///
/// All arrays use layout `[feature * 256 + bin]`.
pub struct NodeHistograms {
    /// Bin counts.
    pub counts: Vec<u32>,
    /// Residual sums per bin.
    pub y_sums: Vec<f32>,
    /// Residual sum-of-squares per bin.
    pub y_sum_sqs: Vec<f32>,
    /// Number of features.
    pub n_features: usize,
}

impl NodeHistograms {
    /// Allocate zeroed histograms for the given number of features.
    ///
    /// ### Params
    ///
    /// * `n_features` - Number of features
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(n_features: usize) -> Self {
        let n = n_features * 256;
        Self {
            counts: vec![0u32; n],
            y_sums: vec![0.0f32; n],
            y_sum_sqs: vec![0.0f32; n],
            n_features,
        }
    }

    /// Zero all bins.
    pub fn reset(&mut self) {
        let n = self.n_features * 256;
        self.counts[..n].fill(0);
        self.y_sums[..n].fill(0.0);
        self.y_sum_sqs[..n].fill(0.0);
    }

    /// Build histograms from a set of sample indices.
    ///
    /// Iterates feature-major for cache-friendly access to the quantised
    /// feature columns. The histogram accumulation region (256 bins * 12
    /// bytes) fits comfortably in L1.
    ///
    /// ### Params
    ///
    /// * `x` - Quantised feature store.
    /// * `samples` - Indices of the samples to include.
    /// * `residuals` - Dense residual array indexed by cell id.
    pub fn build_from_samples(
        &mut self,
        x: &DenseQuantisedStore,
        samples: &[u32],
        residuals: &[f32],
    ) {
        let n = self.n_features * 256;
        self.counts[..n].fill(0);
        self.y_sums[..n].fill(0.0);
        self.y_sum_sqs[..n].fill(0.0);

        for f in 0..self.n_features {
            let col = x.get_col(f);
            let base = f * 256;
            for &s in samples {
                let bin = col[s as usize] as usize;
                let idx = base + bin;
                self.counts[idx] += 1;
                let r = residuals[s as usize];
                self.y_sums[idx] += r;
                self.y_sum_sqs[idx] += r * r;
            }
        }
    }

    /// Find the best split across all (or a random subset of) features.
    ///
    /// For each candidate feature, computes prefix sums over the 256 bins
    /// and evaluates every valid threshold. Returns `None` if no improving
    /// split exists.
    ///
    /// ### Params
    ///
    /// * `total_sum` - Sum of residuals in this node.
    /// * `total_sum_sq` - Sum of squared residuals in this node.
    /// * `n_samples` - Number of training samples in this node.
    /// * `min_samples_leaf` - Minimum samples per child.
    /// * `n_features_split` - Features to evaluate; `0` means all.
    /// * `feat_buf` - Scratch buffer for feature permutation (length >=
    ///   n_features).
    /// * `rng` - RNG for feature subsampling.
    ///
    /// ### Returns
    ///
    /// Option of the GbmSplitInfo
    #[allow(clippy::too_many_arguments)]
    pub fn find_best_split(
        &self,
        total_sum: f32,
        total_sum_sq: f32,
        n_samples: u32,
        min_samples_leaf: u32,
        n_features_split: usize,
        feat_buf: &mut [usize],
        rng: &mut SmallRng,
    ) -> Option<GbmSplitInfo> {
        let n_features = self.n_features;
        let k_feats = if n_features_split == 0 || n_features_split >= n_features {
            n_features
        } else {
            n_features_split
        };

        // partial Fisher-Yates to select k_feats features
        for i in 0..k_feats {
            let j = rng.random_range(i..n_features);
            feat_buf.swap(i, j);
        }

        let parent_var = node_variance_f32(total_sum, total_sum_sq, n_samples as usize);
        if parent_var < 1e-10 {
            return None;
        }

        let nf = n_samples as f32;
        let mut best = GbmSplitInfo {
            feature: usize::MAX,
            threshold: 0,
            score: 0.0,
            n_left: 0,
            y_sum_left: 0.0,
            y_sum_sq_left: 0.0,
        };

        for fi in 0..k_feats {
            let feat = feat_buf[fi];
            let base = feat * 256;

            let mut cum_count = 0u32;
            let mut cum_sum = 0.0f32;
            let mut cum_sum_sq = 0.0f32;

            // scan bins 0..254; threshold = bin means val <= bin goes left
            for bin in 0..255usize {
                cum_count += self.counts[base + bin];
                cum_sum += self.y_sums[base + bin];
                cum_sum_sq += self.y_sum_sqs[base + bin];

                let nl = cum_count;
                let nr = n_samples - nl;
                if nl < min_samples_leaf || nr < min_samples_leaf {
                    continue;
                }

                let nlf = nl as f32;
                let nrf = nr as f32;
                let mean_l = cum_sum / nlf;
                let var_l = f32::max(0.0, cum_sum_sq / nlf - mean_l * mean_l);
                let sum_r = total_sum - cum_sum;
                let sum_sq_r = total_sum_sq - cum_sum_sq;
                let mean_r = sum_r / nrf;
                let var_r = f32::max(0.0, sum_sq_r / nrf - mean_r * mean_r);

                let score = parent_var - (nlf / nf) * var_l - (nrf / nf) * var_r;
                if score > best.score {
                    best.score = score;
                    best.feature = feat;
                    best.threshold = bin as u8;
                    best.n_left = nl;
                    best.y_sum_left = cum_sum;
                    best.y_sum_sq_left = cum_sum_sq;
                }
            }
        }

        if best.feature == usize::MAX {
            None
        } else {
            Some(best)
        }
    }
}

/////////////////////
// Histogram Pool  //
/////////////////////

/// Pre-allocated pool of `NodeHistograms` with acquire/release semantics.
///
/// Avoids repeated heap allocation during tree building. The pool is sized
/// for the maximum recursion depth at construction time.
struct HistogramPool {
    /// Vector of NodeHistograms
    histograms: Vec<NodeHistograms>,
    /// Free pools
    free: Vec<usize>,
}

impl HistogramPool {
    /// Create a pool with `capacity` histograms, each sized for number of
    /// features
    ///
    /// ### Params
    ///
    /// * `capacity` - The capacity of the histogram pool
    /// * `n_features` - Number of features
    ///
    /// ### Returns
    ///
    /// Initialised self
    fn new(capacity: usize, n_features: usize) -> Self {
        let histograms = (0..capacity)
            .map(|_| NodeHistograms::new(n_features))
            .collect();
        let free = (0..capacity).rev().collect();
        Self { histograms, free }
    }

    /// Acquire a histogram index from the pool. Panics if the pool is exhausted.
    ///
    /// ### Returns
    ///
    /// Index into `self.histograms` for the acquired histogram.
    fn acquire(&mut self) -> usize {
        self.free.pop().expect("histogram pool exhausted")
    }

    /// Return a histogram to the pool.
    ///
    /// ### Params
    ///
    /// * `idx` - Releases the pool
    fn release(&mut self, idx: usize) {
        self.free.push(idx);
    }

    /// Compute `out = parent - child` element-wise for all bins.
    ///
    /// All three indices must be distinct.
    ///
    /// ### Params
    ///
    /// * `parent` - Index of the parent node histogram.
    /// * `child` - Index of the smaller child histogram (built from scratch).
    /// * `out` - Index of the histogram to write the result into.
    fn subtract(&mut self, parent: usize, child: usize, out: usize) {
        debug_assert_ne!(parent, out);
        debug_assert_ne!(child, out);
        let n = self.histograms[0].n_features * 256;
        for i in 0..n {
            let pc = self.histograms[parent].counts[i];
            let cc = self.histograms[child].counts[i];
            let py = self.histograms[parent].y_sums[i];
            let cy = self.histograms[child].y_sums[i];
            let pys = self.histograms[parent].y_sum_sqs[i];
            let cys = self.histograms[child].y_sum_sqs[i];
            self.histograms[out].counts[i] = pc - cc;
            self.histograms[out].y_sums[i] = py - cy;
            self.histograms[out].y_sum_sqs[i] = pys - cys;
        }
    }
}

/////////////////
// GBM scratch //
/////////////////

/// Reusable scratch buffers for GBM tree building (non-histogram state).
struct GbmScratch {
    /// Feature permutation buffer for partial Fisher-Yates.
    feat_buf: Vec<usize>,
    /// Partition scratch for left training samples.
    train_left: Vec<u32>,
    /// Partition scratch for right training samples.
    train_right: Vec<u32>,
    /// Partition scratch for left OOB samples.
    oob_left: Vec<u32>,
    /// Partition scratch for right OOB samples.
    oob_right: Vec<u32>,
}

impl GbmScratch {
    /// Generate a new GBM scratch space
    ///
    /// ### Params
    ///
    /// * `n_features` - Number of features
    /// * `n_samples` - Number of samples
    ///
    /// ### Returns
    ///
    /// Initialised self
    fn new(n_features: usize, n_samples: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            train_left: vec![0u32; n_samples],
            train_right: vec![0u32; n_samples],
            oob_left: vec![0u32; n_samples],
            oob_right: vec![0u32; n_samples],
        }
    }
}

////////////////////
// GBM Node Build //
////////////////////

/// Apply a leaf prediction: accumulate OOB improvement then update all
/// residuals.
///
/// OOB improvement is computed before residual updates so it reflects the
/// pre-update state. The per-sample improvement is:
/// `2 * lr * pred * r[s] - lr^2 * pred^2`
/// which equals `r[s]^2 - (r[s] - lr * pred)^2`.
///
/// ### Params
///
/// * `residuals` - Dense residual array indexed by cell id (updated in place).
/// * `train_samples` - Training sample indices for this leaf.
/// * `oob_samples` - OOB sample indices for this leaf.
/// * `y_sum_train` - Sum of training residuals in this leaf.
/// * `n_train` - Number of training samples in this leaf.
/// * `learning_rate` - Shrinkage factor applied to the leaf prediction.
/// * `oob_improvement` - Accumulated OOB squared-error improvement (updated
///   in place).
fn apply_leaf(
    residuals: &mut [f32],
    train_samples: &[u32],
    oob_samples: &[u32],
    y_sum_train: f32,
    n_train: usize,
    learning_rate: f32,
    oob_improvement: &mut f32,
) {
    if n_train == 0 {
        return;
    }
    let pred = y_sum_train / n_train as f32;
    let lr_pred = learning_rate * pred;
    let lr_pred_sq = lr_pred * lr_pred;

    // OOB improvement (before updating OOB residuals)
    for &s in oob_samples {
        let r = residuals[s as usize];
        *oob_improvement += 2.0 * lr_pred * r - lr_pred_sq;
    }

    // update residuals for ALL samples (train + OOB)
    for &s in train_samples {
        residuals[s as usize] -= lr_pred;
    }
    for &s in oob_samples {
        residuals[s as usize] -= lr_pred;
    }
}

/// Recursively build a single GBM tree node using histogram subtraction.
///
/// At each internal node: find the best split from the precomputed
/// histogram, partition both training and OOB samples, build the smaller
/// child's histogram from scratch, derive the larger via subtraction, and
/// recurse. At leaves: compute the mean residual as the leaf prediction,
/// accumulate OOB improvement, then update residuals for all samples.
///
/// ### Params
///
/// * `pool` - Histogram pool for allocation and subtraction.
/// * `node_hist_idx` - Index of this node's histogram in the pool.
/// * `x` - Quantised feature store.
/// * `residuals` - Dense residual array (updated in place at leaves).
/// * `train_samples` - Training sample indices for this node (partitioned
///   in place).
/// * `oob_samples` - OOB sample indices for this node (partitioned in
///   place).
/// * `y_sum` - Sum of training residuals in this node.
/// * `y_sum_sq` - Sum of squared training residuals in this node.
/// * `n_total_train` - Total training samples at the tree root (for
///   importance weighting).
/// * `config` - GBM configuration.
/// * `depth` - Current tree depth.
/// * `importances` - Per-feature importance accumulator.
/// * `oob_improvement` - Accumulated OOB squared-error improvement.
/// * `scratch` - Reusable non-histogram buffers.
/// * `rng` - Per-tree RNG.
#[allow(clippy::too_many_arguments)]
fn build_gbm_node(
    pool: &mut HistogramPool,
    node_hist_idx: usize,
    x: &DenseQuantisedStore,
    residuals: &mut [f32],
    train_samples: &mut [u32],
    oob_samples: &mut [u32],
    y_sum: f32,
    y_sum_sq: f32,
    n_total_train: usize,
    config: &GradientBoostingConfig,
    depth: usize,
    learning_rate: f32,
    importances: &mut [f32],
    oob_improvement: &mut f32,
    scratch: &mut GbmScratch,
    rng: &mut SmallRng,
) {
    let n_train = train_samples.len();

    // stopping criteria
    if n_train < 2 * config.min_samples_leaf || depth >= config.max_depth {
        // leaf: predict, accumulate OOB improvement, update residuals
        apply_leaf(
            residuals,
            train_samples,
            oob_samples,
            y_sum,
            n_train,
            learning_rate,
            oob_improvement,
        );
        pool.release(node_hist_idx);
        return;
    }

    let parent_var = node_variance_f32(y_sum, y_sum_sq, n_train);
    if parent_var < 1e-10 {
        apply_leaf(
            residuals,
            train_samples,
            oob_samples,
            y_sum,
            n_train,
            learning_rate,
            oob_improvement,
        );
        pool.release(node_hist_idx);
        return;
    }

    // find best split
    let split = {
        let hist = &pool.histograms[node_hist_idx];
        hist.find_best_split(
            y_sum,
            y_sum_sq,
            n_train as u32,
            config.min_samples_leaf as u32,
            config.n_features_split,
            &mut scratch.feat_buf,
            rng,
        )
    };

    let split = match split {
        Some(s) => s,
        None => {
            apply_leaf(
                residuals,
                train_samples,
                oob_samples,
                y_sum,
                n_train,
                learning_rate,
                oob_improvement,
            );
            pool.release(node_hist_idx);
            return;
        }
    };

    // accumulate importance
    let nl = split.n_left as f32;
    let nr = (n_train as u32 - split.n_left) as f32;
    let nf = n_train as f32;
    let weight = nf / n_total_train as f32;

    let mean_l = split.y_sum_left / nl;
    let var_l = f32::max(0.0, split.y_sum_sq_left / nl - mean_l * mean_l);
    let y_sum_right = y_sum - split.y_sum_left;
    let y_sum_sq_right = y_sum_sq - split.y_sum_sq_left;
    let mean_r = y_sum_right / nr;
    let var_r = f32::max(0.0, y_sum_sq_right / nr - mean_r * mean_r);
    let reduction = parent_var - (nl / nf) * var_l - (nr / nf) * var_r;
    importances[split.feature] += weight * f32::max(0.0, reduction);

    // partition the trainings examples
    let tf_col = x.get_col(split.feature);
    let mut tl = 0usize;
    let mut tr = 0usize;
    for i in 0..n_train {
        let s = train_samples[i];
        if tf_col[s as usize] <= split.threshold {
            scratch.train_left[tl] = s;
            tl += 1;
        } else {
            scratch.train_right[tr] = s;
            tr += 1;
        }
    }
    train_samples[..tl].copy_from_slice(&scratch.train_left[..tl]);
    train_samples[tl..].copy_from_slice(&scratch.train_right[..tr]);
    let (left_train, right_train) = train_samples.split_at_mut(tl);

    // partion OOB samples
    let n_oob = oob_samples.len();
    let mut ol = 0usize;
    let mut or_ = 0usize;
    for i in 0..n_oob {
        let s = oob_samples[i];
        if tf_col[s as usize] <= split.threshold {
            scratch.oob_left[ol] = s;
            ol += 1;
        } else {
            scratch.oob_right[or_] = s;
            or_ += 1;
        }
    }
    oob_samples[..ol].copy_from_slice(&scratch.oob_left[..ol]);
    oob_samples[ol..].copy_from_slice(&scratch.oob_right[..or_]);
    let (left_oob, right_oob) = oob_samples.split_at_mut(ol);

    // histogram subtraction -> build smaller child, derive larger
    // neat little trick...
    let left_is_smaller = tl <= tr;
    let smaller_idx = pool.acquire();
    let larger_idx = pool.acquire();

    if left_is_smaller {
        pool.histograms[smaller_idx].build_from_samples(x, left_train, residuals);
        pool.subtract(node_hist_idx, smaller_idx, larger_idx);
    } else {
        pool.histograms[smaller_idx].build_from_samples(x, right_train, residuals);
        pool.subtract(node_hist_idx, smaller_idx, larger_idx);
    }

    // parent histogram no longer needed
    pool.release(node_hist_idx);

    // assign histogram indices to left/right
    let (left_hist_idx, right_hist_idx) = if left_is_smaller {
        (smaller_idx, larger_idx)
    } else {
        (larger_idx, smaller_idx)
    };

    // recursion
    build_gbm_node(
        pool,
        left_hist_idx,
        x,
        residuals,
        left_train,
        left_oob,
        split.y_sum_left,
        split.y_sum_sq_left,
        n_total_train,
        config,
        depth + 1,
        learning_rate,
        importances,
        oob_improvement,
        scratch,
        rng,
    );
    build_gbm_node(
        pool,
        right_hist_idx,
        x,
        residuals,
        right_train,
        right_oob,
        y_sum_right,
        y_sum_sq_right,
        n_total_train,
        config,
        depth + 1,
        learning_rate,
        importances,
        oob_improvement,
        scratch,
        rng,
    );
}

//////////////////
// Core fitting //
//////////////////

/// Fit a GRNBoost2-style gradient boosted ensemble for a single target gene.
///
/// Builds trees sequentially, each fitting the current residuals. Each tree
/// uses a random 90/10 train/OOB split (controlled by
/// `config.subsample_rate`). Early stopping triggers when the rolling
/// average of OOB improvements over the last `config.early_stop_window`
/// trees drops to zero or below.
///
/// ### Params
///
/// * `target` - Sparse target gene expression column.
/// * `feature_matrix` - Quantised TF feature store.
/// * `n_samples` - Number of cells.
/// * `config` - GBM configuration.
/// * `seed` - Base seed for reproducibility.
///
/// ### Returns
///
/// Normalised importance vector of length `n_features`, summing to 1.0.
pub fn fit_grnboost2_sparse(
    target: &SparseAxis<u16, f32>,
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &GradientBoostingConfig,
    seed: usize,
) -> Vec<f32> {
    let n_features = feature_matrix.n_features;
    let pool_capacity = 2 * config.max_depth + 3;

    // dense residuals initialised from sparse target
    let mut residuals = vec![0.0f32; n_samples];
    let (indices, values) = target.get_indices_data_2();
    for (i, &idx) in indices.iter().enumerate() {
        residuals[idx] = values[i];
    }

    let n_train = ((n_samples as f32 * config.subsample_rate).round() as usize)
        .max(2 * config.min_samples_leaf);
    let n_oob = n_samples - n_train;

    let mut sample_indices: Vec<u32> = (0..n_samples as u32).collect();
    let mut importances = vec![0.0f32; n_features];
    let mut improvement_ring = Vec::with_capacity(config.early_stop_window);
    let mut scratch = GbmScratch::new(n_features, n_samples);
    let mut pool = HistogramPool::new(pool_capacity, n_features);

    for tree_idx in 0..config.n_trees_max {
        let mut rng = SmallRng::seed_from_u64(
            seed.wrapping_add(tree_idx.wrapping_mul(6364136223846793005)) as u64,
        );

        // train/OOB split via partial Fisher-Yates
        for i in 0..n_samples {
            sample_indices[i] = i as u32;
        }
        for i in 0..n_train {
            let j = rng.random_range(i..n_samples);
            sample_indices.swap(i, j);
        }

        let (train, oob) = sample_indices.split_at_mut(n_train);

        // root histogram
        let root_idx = pool.acquire();
        pool.histograms[root_idx].build_from_samples(feature_matrix, train, &residuals);

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
            feature_matrix,
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

        // early stopping
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

    // normalise importances
    let total: f32 = importances.iter().sum();
    if total > 0.0 {
        let inv = 1.0 / total;
        importances.iter_mut().for_each(|v| *v *= inv);
    }
    importances
}

///////////////////
// Gene batching //
///////////////////

/// Strategy for grouping target genes into multi-output batches
#[derive(Clone, Debug, Default)]
pub enum GeneBatchStrategy {
    /// Shuffle genes randomly before chunking
    #[default]
    Random,
    /// Group genes by expression correlation via SVD + k-means
    /// Fields: (n_svd_components, n_subsampled_cells)
    Correlated {
        /// Number of n_svd
        n_comp: usize,
        /// Number of cells to subsample
        n_cells_subsample: usize,
    },
}

/// Parse the gene batching strategy
///
/// ### Params
///
/// * `s` - String defining the gene batch strategy
/// * `n_comp` - Number of PCs to use for the (randomised) SVD
/// * `n_cells_subsample` - Number of cells to subsample for very large data
///   sets
///
/// ### Returns
///
/// The GeneBatchStrategy option
pub fn parse_gene_batch_strategy(
    s: &str,
    n_comp: usize,
    n_cells_subsample: usize,
) -> Option<GeneBatchStrategy> {
    match s.to_lowercase().as_str() {
        "random" => Some(GeneBatchStrategy::Random),
        "correlated" => Some(GeneBatchStrategy::Correlated {
            n_comp,
            n_cells_subsample,
        }),
        _ => None,
    }
}

/// Batch genes randomly
///
/// ### Params
///
/// * `gene_indices` - The genes to include
/// * `seed` - Random seed
///
/// ### Returns
///
/// Returns the shuffled indices
fn batch_genes_random(gene_indices: &[usize], seed: usize) -> Vec<usize> {
    let mut indices = gene_indices.to_vec();
    let mut rng = SmallRng::seed_from_u64(seed as u64);
    // Fisher-Yates
    for i in (1..indices.len()).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }
    indices
}

/// Subsampling of cells
///
/// ### Params
///
/// * `cell_indices` - The indices to take forward
/// * `n_target` - Number of cells to return
/// * `seed` - For reproducibility
///
/// ### Returns
///
/// Indices of cells to keep here
fn subsample_cells(cell_indices: &[usize], n_target: usize, seed: usize) -> Vec<usize> {
    if cell_indices.len() <= n_target {
        return cell_indices.to_vec();
    }
    let mut indices = cell_indices.to_vec();
    let mut rng = SmallRng::seed_from_u64(seed as u64);
    // Partial Fisher-Yates for n_target elements
    for i in 0..n_target {
        let j = rng.random_range(i..indices.len());
        indices.swap(i, j);
    }
    indices.truncate(n_target);
    indices
}

/// Batch genes by correlation structure
///
/// Runs randomised SVD on a subset of the cells, uses the gene loadings
/// to cluster the genes together into sensible batches of genes.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based file
/// * `gene_indices` - The genes to include
/// * `batch_size` - Size of the batches
/// * `n_components` - Number of components to use for the clustering
/// * `n_cells_subsample` - Number of cells to subsample
/// * `seed` - Seed for reproducibility
/// * `verbose` Controls the verbosity
///
/// ### Returns
///
/// Returns gene indices reordered so that co-expressed genes are contiguous for
/// subsequent batching
#[allow(clippy::too_many_arguments)]
fn batch_genes_correlated(
    f_path: &str,
    gene_indices: &[usize],
    cell_indices: &[usize],
    batch_size: usize,
    n_components: usize,
    n_cells_subsample: usize,
    seed: usize,
    verbose: bool,
) -> Vec<usize> {
    let n_genes = gene_indices.len();
    let n_centroids = n_genes.div_ceil(batch_size);

    let sub_cells = subsample_cells(cell_indices, n_cells_subsample, seed);

    if verbose {
        println!(
            "Computing gene loadings: {} genes, {} subsampled cells, {} components",
            n_genes,
            sub_cells.len(),
            n_components
        );
    }

    // loadings is (n_genes, n_components)
    // using a streaming version here to avoid memory blowing up
    let (_, loadings, _, _) = pca_on_sc_streaming(
        f_path,
        &sub_cells,
        gene_indices,
        n_components,
        true,
        seed,
        false,
        SCENIC_GENE_CHUNK_SIZE,
        verbose,
    );

    // flatten to row-major for k-means: gene g, component c -> [g * dim + c]
    let dim = loadings.ncols();
    let mut gene_loadings = vec![0.0f32; n_genes * dim];
    for g in 0..n_genes {
        for c in 0..dim {
            gene_loadings[g * dim + c] = loadings[(g, c)];
        }
    }

    if verbose {
        println!("Clustering {} genes into {} groups", n_genes, n_centroids);
    }

    let centroids = train_centroids(
        &gene_loadings,
        dim,
        n_genes,
        n_centroids,
        &Dist::Euclidean,
        50,
        seed,
        verbose,
    );

    let centroid_norms: Vec<f32> = (0..n_centroids)
        .map(|i| {
            let c = &centroids[i * dim..(i + 1) * dim];
            f32::dot_simd(c, c)
        })
        .collect();

    let data_norms: Vec<f32> = (0..n_genes)
        .map(|i| {
            let v = &gene_loadings[i * dim..(i + 1) * dim];
            f32::dot_simd(v, v)
        })
        .collect();

    let assignments = assign_all_parallel(
        &gene_loadings,
        &data_norms,
        dim,
        n_genes,
        &centroids,
        &centroid_norms,
        n_centroids,
        &Dist::Euclidean,
    );

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_centroids];
    for (i, &cluster_id) in assignments.iter().enumerate() {
        clusters[cluster_id].push(gene_indices[i]);
    }

    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1) as u64);
    let mut result = Vec::with_capacity(n_genes);
    for cluster in &mut clusters {
        for i in (1..cluster.len()).rev() {
            let j = rng.random_range(0..=i);
            cluster.swap(i, j);
        }
        result.extend_from_slice(cluster);
    }

    result
}

/// Reorder gene indices into batches of `batch_size` according to the chosen
/// strategy.
///
/// ### Params
///
/// * `f_path` - Path to the sparse expression file.
/// * `gene_indices` - Target gene indices to batch.
/// * `cell_indices` - Cell indices in use.
/// * `batch_size` - Target batch size (MULTI_OUTPUT_BATCH).
/// * `strategy` - Batching strategy.
/// * `seed` - RNG seed.
/// * `verbose` - Print progress.
///
/// ### Returns
///
/// Gene indices reordered so that consecutive chunks of `batch_size` form
/// sensible multi-output groups.
pub fn batch_genes(
    f_path: &str,
    gene_indices: &[usize],
    cell_indices: &[usize],
    batch_size: usize,
    strategy: &GeneBatchStrategy,
    seed: usize,
    verbose: bool,
) -> Vec<usize> {
    match strategy {
        GeneBatchStrategy::Random => batch_genes_random(gene_indices, seed),
        GeneBatchStrategy::Correlated {
            n_comp,
            n_cells_subsample,
        } => {
            // Fewer genes than a single batch -- correlation grouping is
            // pointless, just shuffle.
            if gene_indices.len() <= batch_size {
                return batch_genes_random(gene_indices, seed);
            }
            batch_genes_correlated(
                f_path,
                gene_indices,
                cell_indices,
                batch_size,
                *n_comp,
                // subsample_cells already handles the case where
                // cell_indices.len() <= n_cells_subsample, but be explicit
                (*n_cells_subsample).min(cell_indices.len()),
                seed,
                verbose,
            )
        }
    }
}

/////////////////////
// Fit type choice //
/////////////////////

/// Multi-output RF/ET path: genes are batched and fitted with shared tree
/// structure across targets within each batch.
///
/// ### Params
///
/// * `f_path` - Path to the sparse expression file (forwarded to
///   `batch_genes` for correlated batching).
/// * `reader` - Parallel sparse reader for target gene I/O.
/// * `cell_set` - Set of active cell IDs for filtering.
/// * `cell_indices` - Cell indices as a slice (forwarded to `batch_genes`).
/// * `gene_indices` - Target gene indices.
/// * `tf_data` - Quantised TF feature store.
/// * `n_cells` - Number of active cells.
/// * `n_tfs` - Number of TFs (features).
/// * `n_genes` - Number of target genes.
/// * `scenic_params` - SCENIC configuration.
/// * `seed` - Base random seed.
/// * `verbose` - Print progress to stdout.
/// * `start_total` - Timer from the top-level call for elapsed reporting.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` containing normalised
/// per-target feature importances.
#[allow(clippy::too_many_arguments)]
fn run_scenic_multi_output(
    f_path: &str,
    reader: &ParallelSparseReader,
    cell_set: &IndexSet<u32>,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_data: &DenseQuantisedStore,
    n_cells: usize,
    n_tfs: usize,
    n_genes: usize,
    scenic_params: &ScenicParams,
    seed: usize,
    verbose: bool,
    start_total: Instant,
) -> Mat<f32> {
    let n_multi_output = scenic_params
        .gene_batch_size
        .unwrap_or(MULTI_OUTPUT_BATCH)
        .min(MULTI_OUTPUT_BATCH);

    let strategy = parse_gene_batch_strategy(
        &scenic_params.gene_batch_strategy,
        scenic_params.n_pcs,
        scenic_params.n_subsample,
    )
    .unwrap_or(GeneBatchStrategy::Random);

    let ordered_genes = batch_genes(
        f_path,
        gene_indices,
        cell_indices,
        n_multi_output,
        &strategy,
        seed,
        verbose,
    );

    let gene_id_to_pos: FxHashMap<usize, usize> = gene_indices
        .iter()
        .enumerate()
        .map(|(pos, &gid)| (gid, pos))
        .collect();

    let start_gene_read = Instant::now();
    let mut all_gene_ids: Vec<usize> = Vec::with_capacity(n_genes);
    let mut all_sparse_cols: Vec<SparseAxis<u16, f32>> = Vec::with_capacity(n_genes);

    for (iter, chunk) in ordered_genes.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(chunk);
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(cell_set);
        });

        for (i, gc) in gene_chunks.iter().enumerate() {
            all_gene_ids.push(chunk[i]);
            all_sparse_cols.push(gc.to_sparse_axis(n_cells));
        }

        if verbose {
            println!(
                "  Read gene chunk {}/{} ({} genes)",
                iter + 1,
                ordered_genes.len().div_ceil(SCENIC_GENE_CHUNK_SIZE),
                all_gene_ids.len(),
            );
        }
    }

    if verbose {
        println!(
            "Read and filtered {} target genes in {:.2?}",
            n_genes,
            start_gene_read.elapsed()
        );
    }

    let id_batches: Vec<&[usize]> = all_gene_ids.chunks(n_multi_output).collect();
    let col_batches: Vec<&[SparseAxis<u16, f32>]> =
        all_sparse_cols.chunks(n_multi_output).collect();
    let total_batches = col_batches.len();

    let config: &dyn TreeRegressorConfig = match &scenic_params.regression_learner {
        RegressionLearner::ExtraTrees(cfg) => cfg,
        RegressionLearner::RandomForest(cfg) => cfg,
        RegressionLearner::GradientBoosting(_) => unreachable!(),
    };

    if verbose {
        println!(
            "Running SCENIC on {} genes ({} TFs, {} cells, {} batches of up to {})",
            n_genes, n_tfs, n_cells, total_batches, n_multi_output,
        );
    }

    let start_fit = Instant::now();
    let batches_done = AtomicUsize::new(0);

    let batch_results: Vec<(usize, Vec<Vec<f32>>)> = (0..total_batches)
        .into_par_iter()
        .map(|batch_idx| {
            let batch_seed = seed.wrapping_add(batch_idx.wrapping_mul(2654435761));
            let imp = fit_multi_trees_sparse(
                col_batches[batch_idx],
                tf_data,
                n_cells,
                config,
                batch_seed,
            );

            if verbose {
                let done = batches_done.fetch_add(1, Ordering::Relaxed) + 1;
                let pct = done * 100 / total_batches;
                let prev_pct = (done - 1) * 100 / total_batches;
                if pct / 10 > prev_pct / 10 || done == total_batches {
                    println!(
                        "  Progress: {}% ({}/{} batches, {:.2?} elapsed)",
                        pct,
                        done,
                        total_batches,
                        start_fit.elapsed()
                    );
                }
            }

            (batch_idx, imp)
        })
        .collect();

    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];

    for (batch_idx, imp_vecs) in batch_results {
        let batch_gene_ids = id_batches[batch_idx];
        for (local_idx, imp) in imp_vecs.into_iter().enumerate() {
            let gene_id = batch_gene_ids[local_idx];
            let original_pos = gene_id_to_pos[&gene_id];
            importance_scores[original_pos] = imp;
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

/// GBM path: single-target, parallelised across individual genes.
///
/// No gene batching strategy is applied -- each gene is an independent
/// regression. All target genes are read into memory, then fitted via a
/// single `par_iter` over the full gene set.
///
/// ### Params
///
/// * `reader` - Parallel sparse reader for target gene I/O.
/// * `cell_set` - Set of active cell IDs for filtering.
/// * `gene_indices` - Target gene indices.
/// * `tf_data` - Quantised TF feature store.
/// * `n_cells` - Number of active cells.
/// * `n_tfs` - Number of TFs (features).
/// * `n_genes` - Number of target genes.
/// * `config` - GBM configuration.
/// * `seed` - Base random seed.
/// * `verbose` - Print progress to stdout.
/// * `start_total` - Timer from the top-level call for elapsed reporting.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` containing normalised
/// per-target feature importances.
#[allow(clippy::too_many_arguments)]
fn run_scenic_gbm(
    reader: &ParallelSparseReader,
    cell_set: &IndexSet<u32>,
    gene_indices: &[usize],
    tf_data: &DenseQuantisedStore,
    n_cells: usize,
    n_tfs: usize,
    n_genes: usize,
    config: &GradientBoostingConfig,
    seed: usize,
    verbose: bool,
    start_total: Instant,
) -> Mat<f32> {
    let start_gene_read = Instant::now();
    let mut all_sparse_cols: Vec<SparseAxis<u16, f32>> = Vec::with_capacity(n_genes);

    for (iter, chunk) in gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(chunk);
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(cell_set);
        });

        for gc in &gene_chunks {
            all_sparse_cols.push(gc.to_sparse_axis(n_cells));
        }

        if verbose {
            println!(
                "  Read gene chunk {}/{} ({} genes)",
                iter + 1,
                gene_indices.len().div_ceil(SCENIC_GENE_CHUNK_SIZE),
                all_sparse_cols.len(),
            );
        }
    }

    if verbose {
        println!(
            "Read and filtered {} target genes in {:.2?}",
            n_genes,
            start_gene_read.elapsed()
        );
        println!(
            "Running GRNBoost2 on {} genes ({} TFs, {} cells)",
            n_genes, n_tfs, n_cells,
        );
    }

    let start_fit = Instant::now();
    let genes_done = AtomicUsize::new(0);

    let importance_scores: Vec<Vec<f32>> = all_sparse_cols
        .par_iter()
        .enumerate()
        .map(|(gene_idx, target)| {
            let gene_seed = seed.wrapping_add(gene_idx.wrapping_mul(2654435761));
            let imp = fit_grnboost2_sparse(target, tf_data, n_cells, config, gene_seed);

            if verbose {
                let done = genes_done.fetch_add(1, Ordering::Relaxed) + 1;
                let pct = done * 100 / n_genes;
                let prev_pct = (done - 1) * 100 / n_genes;
                if pct / 10 > prev_pct / 10 || done == n_genes {
                    println!(
                        "  Progress: {}% ({}/{} genes, {:.2?} elapsed)",
                        pct,
                        done,
                        n_genes,
                        start_fit.elapsed()
                    );
                }
            }

            imp
        })
        .collect();

    if verbose {
        println!(
            "GRNBoost2 GRN inference complete in {:.2?}",
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

/// Multi-output RF/ET streaming path: read genes in I/O chunks, slice
/// into multi-output batches within each chunk, fit in parallel, drop
/// before next chunk.
///
/// ### Params
///
/// * `f_path` - Path to the sparse expression file (forwarded to
///   `batch_genes` for correlated batching).
/// * `reader` - Parallel sparse reader for target gene I/O.
/// * `cell_set` - Set of active cell IDs for filtering.
/// * `cell_indices` - Cell indices as a slice (forwarded to `batch_genes`).
/// * `gene_indices` - Target gene indices.
/// * `tf_data` - Quantised TF feature store.
/// * `n_cells` - Number of active cells.
/// * `n_tfs` - Number of TFs (features).
/// * `n_genes` - Number of target genes.
/// * `scenic_params` - SCENIC configuration.
/// * `seed` - Base random seed.
/// * `verbose` - Print progress to stdout.
/// * `start_total` - Timer from the top-level call for elapsed reporting.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` containing normalised
/// per-target feature importances.
#[allow(clippy::too_many_arguments)]
fn run_scenic_multi_output_streaming(
    f_path: &str,
    reader: &ParallelSparseReader,
    cell_set: &IndexSet<u32>,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_data: &DenseQuantisedStore,
    n_cells: usize,
    n_tfs: usize,
    n_genes: usize,
    scenic_params: &ScenicParams,
    seed: usize,
    verbose: bool,
    start_total: Instant,
) -> Mat<f32> {
    let n_multi_output = scenic_params
        .gene_batch_size
        .unwrap_or(MULTI_OUTPUT_BATCH)
        .min(MULTI_OUTPUT_BATCH);

    let strategy = parse_gene_batch_strategy(
        &scenic_params.gene_batch_strategy,
        scenic_params.n_pcs,
        scenic_params.n_subsample,
    )
    .unwrap_or(GeneBatchStrategy::Random);

    let ordered_genes = batch_genes(
        f_path,
        gene_indices,
        cell_indices,
        n_multi_output,
        &strategy,
        seed,
        verbose,
    );

    let gene_id_to_pos: FxHashMap<usize, usize> = gene_indices
        .iter()
        .enumerate()
        .map(|(pos, &gid)| (gid, pos))
        .collect();

    let config: &dyn TreeRegressorConfig = match &scenic_params.regression_learner {
        RegressionLearner::ExtraTrees(cfg) => cfg,
        RegressionLearner::RandomForest(cfg) => cfg,
        RegressionLearner::GradientBoosting(_) => unreachable!(),
    };

    let total_io_chunks = ordered_genes.len().div_ceil(SCENIC_GENE_CHUNK_SIZE);
    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];
    let mut global_batch_offset: usize = 0;

    if verbose {
        println!(
            "Running SCENIC (streaming) on {} genes ({} TFs, {} cells, {} I/O chunks, batches of {})",
            n_genes.separate_with_underscores(),
            n_tfs.separate_with_underscores(),
            n_cells.separate_with_underscores(),
            total_io_chunks,
            n_multi_output,
        );
    }

    for (chunk_idx, io_chunk) in ordered_genes.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        let start_chunk = Instant::now();

        // read and filter chunk
        let start_io = Instant::now();
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(io_chunk);
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(cell_set);
        });

        let sparse_columns: Vec<SparseAxis<u16, f32>> = gene_chunks
            .iter()
            .map(|c| c.to_sparse_axis(n_cells))
            .collect();
        drop(gene_chunks);

        if verbose {
            println!(
                "  Chunk {}/{}: loaded and filtered {} genes in {:.2?}",
                chunk_idx + 1,
                total_io_chunks,
                io_chunk.len(),
                start_io.elapsed()
            );
        }

        let id_batches: Vec<&[usize]> = io_chunk.chunks(n_multi_output).collect();
        let col_batches: Vec<&[SparseAxis<u16, f32>]> =
            sparse_columns.chunks(n_multi_output).collect();
        let n_batches_this_chunk = col_batches.len();

        let start_fit = Instant::now();
        let batches_done = AtomicUsize::new(0);

        let batch_results: Vec<(usize, Vec<Vec<f32>>)> = (0..n_batches_this_chunk)
            .into_par_iter()
            .map(|local_batch_idx| {
                let batch_seed = seed
                    .wrapping_add((global_batch_offset + local_batch_idx).wrapping_mul(2654435761));
                let imp = fit_multi_trees_sparse(
                    col_batches[local_batch_idx],
                    tf_data,
                    n_cells,
                    config,
                    batch_seed,
                );

                if verbose && n_batches_this_chunk >= 4 {
                    let done = batches_done.fetch_add(1, Ordering::Relaxed) + 1;
                    let pct = done * 100 / n_batches_this_chunk;
                    let prev_pct = (done - 1) * 100 / n_batches_this_chunk;
                    if [25, 50, 75, 100].iter().any(|&q| prev_pct < q && pct >= q) {
                        println!(
                            "    Chunk {}: ~{}% of batches done ({}/{}, {:.2?} elapsed)",
                            chunk_idx + 1,
                            pct,
                            done,
                            n_batches_this_chunk,
                            start_fit.elapsed()
                        );
                    }
                }

                (local_batch_idx, imp)
            })
            .collect();

        for (local_batch_idx, imp_vecs) in batch_results {
            let batch_gene_ids = id_batches[local_batch_idx];
            for (local_idx, imp) in imp_vecs.into_iter().enumerate() {
                let gene_id = batch_gene_ids[local_idx];
                let original_pos = gene_id_to_pos[&gene_id];
                importance_scores[original_pos] = imp;
            }
        }

        global_batch_offset += n_batches_this_chunk;

        if verbose {
            let genes_done = ((chunk_idx + 1) * SCENIC_GENE_CHUNK_SIZE).min(n_genes);
            println!(
                "  Chunk {}/{}: {}/{} genes done in {:.2?} (fit: {:.2?})",
                chunk_idx + 1,
                total_io_chunks,
                genes_done,
                n_genes,
                start_chunk.elapsed(),
                start_fit.elapsed()
            );
        }
        // sparse_columns dropped here
    }

    if verbose {
        println!(
            "SCENIC GRN inference (streaming) complete in {:.2?}",
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

/// GBM streaming path: read genes in I/O chunks, fit each gene
/// individually within the chunk, drop chunk data before reading next.
///
/// Memory is bounded to one I/O chunk of sparse columns at a time. Each
/// gene within a chunk is an independent work unit for rayon.
///
/// ### Params
///
/// * `reader` - Parallel sparse reader for target gene I/O.
/// * `cell_set` - Set of active cell IDs for filtering.
/// * `gene_indices` - Target gene indices.
/// * `tf_data` - Quantised TF feature store.
/// * `n_cells` - Number of active cells.
/// * `n_tfs` - Number of TFs (features).
/// * `n_genes` - Number of target genes.
/// * `config` - GBM configuration.
/// * `seed` - Base random seed.
/// * `verbose` - Print progress to stdout.
/// * `start_total` - Timer from the top-level call for elapsed reporting.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` containing normalised
/// per-target feature importances.
#[allow(clippy::too_many_arguments)]
fn run_scenic_gbm_streaming(
    reader: &ParallelSparseReader,
    cell_set: &IndexSet<u32>,
    gene_indices: &[usize],
    tf_data: &DenseQuantisedStore,
    n_cells: usize,
    n_tfs: usize,
    n_genes: usize,
    config: &GradientBoostingConfig,
    seed: usize,
    verbose: bool,
    start_total: Instant,
) -> Mat<f32> {
    let total_io_chunks = gene_indices.len().div_ceil(SCENIC_GENE_CHUNK_SIZE);
    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];
    let mut global_gene_offset: usize = 0;

    if verbose {
        println!(
            "Running GRNBoost2 (streaming) on {} genes ({} TFs, {} cells, {} I/O chunks)",
            n_genes.separate_with_underscores(),
            n_tfs.separate_with_underscores(),
            n_cells.separate_with_underscores(),
            total_io_chunks,
        );
    }

    for (chunk_idx, io_chunk) in gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        let start_chunk = Instant::now();

        let start_io = Instant::now();
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(io_chunk);
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(cell_set);
        });

        let sparse_columns: Vec<SparseAxis<u16, f32>> = gene_chunks
            .iter()
            .map(|c| c.to_sparse_axis(n_cells))
            .collect();
        drop(gene_chunks);

        if verbose {
            println!(
                "  Chunk {}/{}: loaded and filtered {} genes in {:.2?}",
                chunk_idx + 1,
                total_io_chunks,
                io_chunk.len(),
                start_io.elapsed()
            );
        }

        let start_fit = Instant::now();
        let genes_done = AtomicUsize::new(0);
        let n_genes_this_chunk = sparse_columns.len();

        let chunk_results: Vec<Vec<f32>> = sparse_columns
            .par_iter()
            .enumerate()
            .map(|(local_idx, target)| {
                let gene_seed =
                    seed.wrapping_add((global_gene_offset + local_idx).wrapping_mul(2654435761));
                let imp = fit_grnboost2_sparse(target, tf_data, n_cells, config, gene_seed);

                if verbose && n_genes_this_chunk >= 4 {
                    let done = genes_done.fetch_add(1, Ordering::Relaxed) + 1;
                    let pct = done * 100 / n_genes_this_chunk;
                    let prev_pct = (done - 1) * 100 / n_genes_this_chunk;
                    if [25, 50, 75, 100].iter().any(|&q| prev_pct < q && pct >= q) {
                        println!(
                            "    Chunk {}: ~{}% of genes done ({}/{}, {:.2?} elapsed)",
                            chunk_idx + 1,
                            pct,
                            done,
                            n_genes_this_chunk,
                            start_fit.elapsed()
                        );
                    }
                }

                imp
            })
            .collect();

        for (local_idx, imp) in chunk_results.into_iter().enumerate() {
            importance_scores[global_gene_offset + local_idx] = imp;
        }

        global_gene_offset += n_genes_this_chunk;

        if verbose {
            println!(
                "  Chunk {}/{}: {}/{} genes done in {:.2?} (fit: {:.2?})",
                chunk_idx + 1,
                total_io_chunks,
                global_gene_offset,
                n_genes,
                start_chunk.elapsed(),
                start_fit.elapsed()
            );
        }
        // sparse_columns dropped here
    }

    if verbose {
        println!(
            "GRNBoost2 GRN inference (streaming) complete in {:.2?}",
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

////////////
// Params //
////////////

/// Defines the parameters to run SCENIC within the bixverse-rs implementation
pub struct ScenicParams {
    /// Min total counts that a gene needs to reach to be included
    pub min_counts: usize,
    /// Min proportion of cells (between 0 and 1) that need to have a gene
    /// expressed to be considered for the analysis
    pub min_cells: f32,
    /// Regression learner - enum defining which regression learner to use
    pub regression_learner: RegressionLearner,
    /// Strategy for gene batching to use
    pub gene_batch_strategy: String,
    /// Optional gene batch size
    pub gene_batch_size: Option<usize>,
    /// Number of PCs to use for correlated gene batch
    pub n_pcs: usize,
    /// Cell subsampling threshold. If n ≥ n_subsample, n_subsample cells will
    /// be randomly selected prior to running randomised SVD for the correlated
    /// gene batch strategy
    pub n_subsample: usize,
}

/// Default implementations of the Scenic parameters
impl Default for ScenicParams {
    fn default() -> Self {
        Self {
            min_counts: 50,
            min_cells: 0.03,
            regression_learner: RegressionLearner::RandomForest(RandomForestConfig::default()),
            gene_batch_strategy: "correlated".to_string(),
            gene_batch_size: None,
            n_pcs: 50,
            n_subsample: 100_000,
        }
    }
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
/// * `scenic_params` - Reference to the SCENIC parameters indicating minimum
///   couts per gene and minimum proportions of cells expressing a gene to
///   be included.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Indices of genes passing both filters, in their original order.
pub fn scenic_gene_filter(
    f_path: &str,
    cell_indices: &[usize],
    scenic_params: &ScenicParams,
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

            if total_counts >= scenic_params.min_counts as u32
                && expressed_fraction >= scenic_params.min_cells
            {
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

/// Run SCENIC GRN inference and return a TF-by-gene importance matrix.
///
/// ### Params
///
/// * `f_path` - Path to the sparse gene expression file.
/// * `cell_indices` - Indices of cells to use.
/// * `gene_indices` - Target gene indices.
/// * `tf_indices` - Transcription factor gene indices (predictors).
/// * `scenic_params` - Reference to the SCENIC parameters.
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
/// TF expression data is loaded once, filtered and quantised into a
/// `DenseQuantisedStore`. Target genes are read in I/O chunks of
/// `SCENIC_GENE_CHUNK_SIZE`, filtered and converted to sparse columns,
/// then collected into a single flat buffer. This buffer is sliced into
/// multi-output batches of `n_multi_output` targets each. All batches are
/// fitted in a single `par_iter` pass, giving rayon full work-stealing
/// flexibility across all batches rather than being constrained to the ~16
/// batches within a single I/O chunk.
///
/// Memory note: all target gene sparse columns are held simultaneously. At
/// 100k cells and 10% sparsity this is roughly 60KB per gene (~1.2GB for
/// 20k genes). For datasets where this is prohibitive, a wave-based
/// approach processing a few thousand genes at a time can be substituted.
pub fn run_scenic_grn(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    scenic_params: &ScenicParams,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let start_total = Instant::now();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_set.len();

    // load and quantise TFs
    let start_reading = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();

    let mut tf_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(tf_indices);
    tf_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let tf_csc: CompressedSparseData2<u16, f32> = from_gene_chunks::<u16>(&tf_chunks, n_cells);
    let tf_data = DenseQuantisedStore::from_csc(&tf_csc, n_cells);
    drop(tf_chunks);
    drop(tf_csc);

    let n_tfs = tf_data.n_features;
    let n_genes = gene_indices.len();

    if verbose {
        println!(
            "Loaded, filtered and quantised TF data (n: {}) in: {:.2?}",
            n_tfs.separate_with_underscores(),
            start_reading.elapsed()
        );
    }

    match &scenic_params.regression_learner {
        RegressionLearner::GradientBoosting(gbm_config) => run_scenic_gbm(
            &reader,
            &cell_set,
            gene_indices,
            &tf_data,
            n_cells,
            n_tfs,
            n_genes,
            gbm_config,
            seed,
            verbose,
            start_total,
        ),
        _ => run_scenic_multi_output(
            f_path,
            &reader,
            &cell_set,
            cell_indices,
            gene_indices,
            &tf_data,
            n_cells,
            n_tfs,
            n_genes,
            scenic_params,
            seed,
            verbose,
            start_total,
        ),
    }
}

/// Run SCENIC GRN inference in streaming mode, processing target genes in
/// waves to bound memory usage.
///
/// ### Params
///
/// * `f_path` - Path to the sparse gene expression file.
/// * `cell_indices` - Indices of cells to use.
/// * `gene_indices` - Target gene indices.
/// * `tf_indices` - Transcription factor gene indices (predictors).
/// * `scenic_params` - Reference to the SCENIC parameters.
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
/// Unlike `run_scenic_grn` which loads all target gene data upfront, this
/// version reads target genes one I/O chunk at a time
/// (`SCENIC_GENE_CHUNK_SIZE`), slices each chunk into multi-output batches,
/// fits all batches within the chunk in parallel, stores the results, then
/// drops the chunk data before reading the next. Peak memory for target gene
/// data is bounded to one chunk (~1024 genes) rather than all targets.
///
/// There is a minor parallelism penalty at chunk boundaries: the last wave
/// of batches in a chunk may not fully saturate all cores. For most
/// configurations (1024 genes / 64 per batch = 16 batches) this is
/// acceptable. Use `run_scenic_grn` when memory permits for maximum
/// throughput.
pub fn run_scenic_grn_streaming(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    scenic_params: &ScenicParams,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let start_total = Instant::now();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_set.len();

    // load and quantise the TFs
    let start_reading = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();

    let mut tf_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(tf_indices);
    tf_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let tf_csc: CompressedSparseData2<u16, f32> = from_gene_chunks::<u16>(&tf_chunks, n_cells);
    let tf_data = DenseQuantisedStore::from_csc(&tf_csc, n_cells);
    drop(tf_chunks);
    drop(tf_csc);

    let n_tfs = tf_data.n_features;
    let n_genes = gene_indices.len();

    if verbose {
        println!(
            "Loaded, filtered and quantised TF data (n: {}) in: {:.2?}",
            n_tfs.separate_with_underscores(),
            start_reading.elapsed()
        );
    }

    match &scenic_params.regression_learner {
        RegressionLearner::GradientBoosting(gbm_config) => run_scenic_gbm_streaming(
            &reader,
            &cell_set,
            gene_indices,
            &tf_data,
            n_cells,
            n_tfs,
            n_genes,
            gbm_config,
            seed,
            verbose,
            start_total,
        ),
        _ => run_scenic_multi_output_streaming(
            f_path,
            &reader,
            &cell_set,
            cell_indices,
            gene_indices,
            &tf_data,
            n_cells,
            n_tfs,
            n_genes,
            scenic_params,
            seed,
            verbose,
            start_total,
        ),
    }
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
        let (sum, sum_sq) = (6.0f32, 14.0f32);
        let v = node_variance_f32(sum, sum_sq, 3);
        assert!((v - 2.0 / 3.0).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn node_variance_uniform() {
        // values: 3, 3, 3 -> var=0
        let (sum, sum_sq) = (9.0f32, 27.0f32);
        let v = node_variance_f32(sum, sum_sq, 3);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn partition_logic_single_target() {
        let tf_col: Vec<u8> = vec![10, 50, 200, 30, 250, 100];
        let sample_slice: Vec<u32> = vec![0, 1, 2, 4];
        let y_slice: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let n_targets = 1;

        let mut left_buf = [0u32; 4];
        let mut right_buf = [0u32; 4];
        let mut left_y_buf = [0.0f32; 4];
        let mut right_y_buf = [0.0f32; 4];

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

        let mut left_buf = [0u32; 3];
        let mut right_buf = [0u32; 3];
        let mut left_y_buf = [0.0f32; 6];
        let mut right_y_buf = [0.0f32; 6];

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

        assert_eq!(bufs.counts[0], 2);
        assert!((bufs.y_sums[0] - 4.0).abs() < 1e-10);
        assert!((bufs.y_sums[1] - 6.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[0] - 10.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[1] - 20.0).abs() < 1e-10);

        assert_eq!(bufs.counts[10], 2);
        assert!((bufs.y_sums[10 * n_targets] - 12.0).abs() < 1e-10);
        assert!((bufs.y_sums[10 * n_targets + 1] - 14.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[10 * n_targets] - 74.0).abs() < 1e-10);
        assert!((bufs.y_sum_sqs[10 * n_targets + 1] - 100.0).abs() < 1e-10);

        assert_eq!(bufs.cum_counts[10], 4);
        assert!((bufs.cum_y_sums[10 * n_targets] - 16.0).abs() < 1e-10);
        assert!((bufs.cum_y_sums[10 * n_targets + 1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn sparse_y_batch_construction() {
        // 4 cells, 2 targets
        // target 0: cell 0 = 1.0, cell 2 = 3.0
        // target 1: cell 1 = 2.0, cell 2 = 4.0

        // Simulate SparseAxis with just indices and values
        // We can't construct SparseAxis directly here, so test via
        // SparseYBatch manually
        let offsets = vec![0u32, 1, 2, 4, 4]; // cell0: 1 entry, cell1: 1, cell2: 2, cell3: 0
        let target_indices = vec![0u8, 1, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0];

        let sy = SparseYBatch {
            offsets,
            target_indices,
            values,
        };

        // cell 0: target 0 = 1.0
        let (ti, tv) = sy.cell_entries(0);
        assert_eq!(ti, &[0]);
        assert_eq!(tv, &[1.0]);

        // cell 1: target 1 = 2.0
        let (ti, tv) = sy.cell_entries(1);
        assert_eq!(ti, &[1]);
        assert_eq!(tv, &[2.0]);

        // cell 2: both targets
        let (ti, tv) = sy.cell_entries(2);
        assert_eq!(ti, &[0, 1]);
        assert_eq!(tv, &[3.0, 4.0]);

        // cell 3: empty
        let (ti, tv) = sy.cell_entries(3);
        assert!(ti.is_empty());
        assert!(tv.is_empty());
    }

    #[test]
    fn sparse_histogram_matches_dense() {
        // Build a scenario where we can compare dense and sparse histogram
        // outputs directly.
        // 6 cells, 2 targets. Bins: [0, 0, 10, 10, 0, 10]
        let n_targets = 2;
        let n_features = 1;
        let n_samples = 6;
        let tf_col: Vec<u8> = vec![0, 0, 10, 10, 0, 10];
        let sample_slice: Vec<u32> = vec![0, 1, 2, 3, 4, 5];

        // Dense y: interleaved [s0t0, s0t1, s1t0, s1t1, ...]
        // cell0=[1,2], cell1=[0,0], cell2=[3,4], cell3=[0,0], cell4=[5,6], cell5=[7,8]
        let y_dense: Vec<f32> = vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0];

        // Equivalent sparse representation
        let sparse_y = SparseYBatch {
            offsets: vec![0, 2, 2, 4, 4, 6, 8],
            target_indices: vec![0, 1, 0, 1, 0, 1, 0, 1],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };

        let mut bufs_dense = TreeBuffers::new(n_features, n_samples, n_targets);
        let mut bufs_sparse = TreeBuffers::new_sparse(n_features, n_samples, n_targets);

        let (min_d, max_d) =
            bufs_dense.build_histograms(&tf_col, &sample_slice, &y_dense, n_targets);
        let (min_s, max_s) =
            bufs_sparse.build_histograms_sparse(&tf_col, &sample_slice, &sparse_y, n_targets);

        assert_eq!(min_d, min_s);
        assert_eq!(max_d, max_s);

        // Counts must match exactly within the active bounds
        assert_eq!(
            &bufs_dense.counts[min_d..=max_d],
            &bufs_sparse.counts[min_s..=max_s]
        );
        assert_eq!(
            &bufs_dense.cum_counts[min_d..=max_d],
            &bufs_sparse.cum_counts[min_s..=max_s]
        );

        // Y sums and sum_sqs must match within the active bounds
        let start_idx = min_d * n_targets;
        let end_idx = (max_d + 1) * n_targets;

        for i in start_idx..end_idx {
            assert!(
                (bufs_dense.y_sums[i] - bufs_sparse.y_sums[i]).abs() < 1e-10,
                "y_sums mismatch at {i}: dense={} sparse={}",
                bufs_dense.y_sums[i],
                bufs_sparse.y_sums[i]
            );
            assert!(
                (bufs_dense.y_sum_sqs[i] - bufs_sparse.y_sum_sqs[i]).abs() < 1e-10,
                "y_sum_sqs mismatch at {i}"
            );
            assert!(
                (bufs_dense.cum_y_sums[i] - bufs_sparse.cum_y_sums[i]).abs() < 1e-10,
                "cum_y_sums mismatch at {i}"
            );
            assert!(
                (bufs_dense.cum_y_sum_sqs[i] - bufs_sparse.cum_y_sum_sqs[i]).abs() < 1e-10,
                "cum_y_sum_sqs mismatch at {i}"
            );
        }
    }

    #[test]
    fn evaluate_split_basic() {
        // 4 samples, 1 target. Bins: [0, 0, 10, 10]
        // y: [1, 3, 10, 12]
        // Split at threshold 0: left=[1,3], right=[10,12]
        // parent var = var([1,3,10,12]) = 19.1875
        // var_l = var([1,3]) = 1.0, var_r = var([10,12]) = 1.0
        // reduction = 19.1875 - 0.5*1.0 - 0.5*1.0 = 18.1875
        let n_targets = 1;
        let mut cum_counts = [0u32; 256];
        let mut cum_y_sums = vec![0.0f32; 256];
        let mut cum_y_sum_sqs = vec![0.0f32; 256];

        // bin 0: count=2, sum=4, sumsq=10
        // bin 10: count=2, sum=22, sumsq=244
        cum_counts[0] = 2;
        cum_y_sums[0] = 4.0;
        cum_y_sum_sqs[0] = 10.0;
        for b in 1..256 {
            cum_counts[b] = cum_counts[b - 1];
            cum_y_sums[b] = cum_y_sums[b - 1];
            cum_y_sum_sqs[b] = cum_y_sum_sqs[b - 1];
            if b == 10 {
                cum_counts[b] += 2;
                cum_y_sums[b] += 22.0;
                cum_y_sum_sqs[b] += 244.0;
            }
        }

        let parent_vars = vec![node_variance_f32(26.0, 254.0, 4)];
        let y_sums_total = vec![26.0f32];
        let y_sum_sqs_total = vec![254.0f32];

        let mut best_score = 0.0f32;
        let mut best_feature = usize::MAX;
        let mut best_threshold = 0u8;
        let mut best_n_left = 0u32;
        let mut best_ys_l = vec![0.0f32; 1];
        let mut best_yss_l = vec![0.0f32; 1];

        evaluate_split_multi(
            0, // threshold: bin <= 0 goes left
            0, // feature
            &parent_vars,
            4,
            &y_sums_total,
            &y_sum_sqs_total,
            &cum_counts,
            &cum_y_sums,
            &cum_y_sum_sqs,
            n_targets,
            1, // min_samples_leaf
            &mut best_score,
            &mut best_feature,
            &mut best_threshold,
            &mut best_n_left,
            &mut best_ys_l,
            &mut best_yss_l,
        );

        assert_eq!(best_feature, 0);
        assert_eq!(best_threshold, 0);
        assert_eq!(best_n_left, 2);
        // parent_var = 254/4 - (26/4)^2 = 63.5 - 42.25 = 21.25
        // var_l = 10/2 - (4/2)^2 = 5 - 4 = 1
        // var_r = 244/2 - (22/2)^2 = 122 - 121 = 1
        // reduction = 21.25 - 0.5*1 - 0.5*1 = 20.25
        assert!((best_score - 20.25).abs() < 1e-10, "got {}", best_score);
    }

    #[test]
    fn batch_genes_random_is_permutation() {
        let genes: Vec<usize> = (0..100).collect();
        let shuffled = batch_genes_random(&genes, 42);
        assert_eq!(shuffled.len(), genes.len());

        let mut sorted = shuffled.clone();
        sorted.sort();
        assert_eq!(sorted, genes);

        // Should actually shuffle (not identity)
        assert_ne!(shuffled, genes);
    }

    #[test]
    fn batch_genes_random_deterministic() {
        let genes: Vec<usize> = (0..100).collect();
        let a = batch_genes_random(&genes, 42);
        let b = batch_genes_random(&genes, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn subsample_cells_all_when_small() {
        let cells: Vec<usize> = (0..50).collect();
        let sub = subsample_cells(&cells, 100, 0);
        assert_eq!(sub.len(), 50);
        // Should contain all original indices
        let mut sorted = sub.clone();
        sorted.sort();
        assert_eq!(sorted, cells);
    }

    #[test]
    fn subsample_cells_correct_count() {
        let cells: Vec<usize> = (0..1000).collect();
        let sub = subsample_cells(&cells, 100, 42);
        assert_eq!(sub.len(), 100);
        // All elements should be valid cell indices
        for &c in &sub {
            assert!(c < 1000);
        }
        // No duplicates (Fisher-Yates guarantees this)
        let mut sorted = sub.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 100);
    }

    /// Helper to construct a DenseQuantisedStore from raw u8 data for tests.
    fn make_store(data: Vec<u8>, n_cells: usize, n_features: usize) -> DenseQuantisedStore {
        assert_eq!(data.len(), n_cells * n_features);
        DenseQuantisedStore {
            data,
            n_cells,
            n_features,
            feature_min: vec![0.0; n_features],
            feature_range: vec![255.0; n_features],
        }
    }

    // ---- NodeHistograms ----

    #[test]
    fn node_hist_build_from_samples() {
        // 4 cells, 2 features
        // Feature 0: [0, 10, 10, 20]
        // Feature 1: [5, 5, 15, 15]
        let data = vec![
            0, 10, 10, 20, // feature 0, cells 0..3
            5, 5, 15, 15, // feature 1, cells 0..3
        ];
        let store = make_store(data, 4, 2);
        let residuals = vec![1.0, 2.0, 3.0, 4.0];
        let samples: Vec<u32> = vec![0, 1, 2, 3];

        let mut hist = NodeHistograms::new(2);
        hist.build_from_samples(&store, &samples, &residuals);

        // Feature 0: bin 0 -> cell 0 (r=1), bin 10 -> cells 1,2 (r=2,3),
        //            bin 20 -> cell 3 (r=4)
        assert_eq!(hist.counts[0], 1);
        assert_eq!(hist.counts[10], 2);
        assert_eq!(hist.counts[20], 1);
        assert!((hist.y_sums[0] - 1.0).abs() < 1e-10);
        assert!((hist.y_sums[10] - 5.0).abs() < 1e-10);
        assert!((hist.y_sums[20] - 4.0).abs() < 1e-10);
        assert!((hist.y_sum_sqs[10] - 13.0).abs() < 1e-10); // 4+9

        // Feature 1: bin 5 -> cells 0,1 (r=1,2), bin 15 -> cells 2,3 (r=3,4)
        assert_eq!(hist.counts[256 + 5], 2);
        assert_eq!(hist.counts[256 + 15], 2);
        assert!((hist.y_sums[256 + 5] - 3.0).abs() < 1e-10);
        assert!((hist.y_sums[256 + 15] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn node_hist_build_subset() {
        // Same store, only use cells 0 and 2
        let data = vec![0, 10, 10, 20, 5, 5, 15, 15];
        let store = make_store(data, 4, 2);
        let residuals = vec![1.0, 2.0, 3.0, 4.0];
        let samples: Vec<u32> = vec![0, 2];

        let mut hist = NodeHistograms::new(2);
        hist.build_from_samples(&store, &samples, &residuals);

        // Feature 0: bin 0 -> cell 0 (r=1), bin 10 -> cell 2 (r=3)
        assert_eq!(hist.counts[0], 1);
        assert_eq!(hist.counts[10], 1);
        assert_eq!(hist.counts[20], 0);
        assert!((hist.y_sums[0] - 1.0).abs() < 1e-10);
        assert!((hist.y_sums[10] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn histogram_subtraction_matches_complement() {
        // Build parent from all 4 samples, child from {0,2},
        // subtraction should match building from {1,3}
        let data = vec![0, 10, 10, 20, 5, 5, 15, 15];
        let store = make_store(data, 4, 2);
        let residuals = vec![1.0, 2.0, 3.0, 4.0];

        let mut parent = NodeHistograms::new(2);
        parent.build_from_samples(&store, &[0, 1, 2, 3], &residuals);

        let mut child = NodeHistograms::new(2);
        child.build_from_samples(&store, &[0, 2], &residuals);

        let mut complement_direct = NodeHistograms::new(2);
        complement_direct.build_from_samples(&store, &[1, 3], &residuals);

        // Compute subtraction via pool
        let mut pool = HistogramPool::new(3, 2);
        // Copy parent into slot 0
        pool.histograms[0].counts.copy_from_slice(&parent.counts);
        pool.histograms[0].y_sums.copy_from_slice(&parent.y_sums);
        pool.histograms[0]
            .y_sum_sqs
            .copy_from_slice(&parent.y_sum_sqs);
        // Copy child into slot 1
        pool.histograms[1].counts.copy_from_slice(&child.counts);
        pool.histograms[1].y_sums.copy_from_slice(&child.y_sums);
        pool.histograms[1]
            .y_sum_sqs
            .copy_from_slice(&child.y_sum_sqs);

        pool.subtract(0, 1, 2);

        let result = &pool.histograms[2];
        let n = 2 * 256;
        for i in 0..n {
            assert_eq!(
                result.counts[i], complement_direct.counts[i],
                "counts mismatch at {i}"
            );
            assert!(
                (result.y_sums[i] - complement_direct.y_sums[i]).abs() < 1e-6,
                "y_sums mismatch at {i}: {} vs {}",
                result.y_sums[i],
                complement_direct.y_sums[i]
            );
            assert!(
                (result.y_sum_sqs[i] - complement_direct.y_sum_sqs[i]).abs() < 1e-6,
                "y_sum_sqs mismatch at {i}"
            );
        }
    }

    #[test]
    fn find_split_obvious_partition() {
        // 8 cells, 1 feature. Feature bins: [0,0,0,0, 100,100,100,100]
        // Residuals: [1,1,1,1, 10,10,10,10]
        // Obvious split at threshold ~50: left=[1,1,1,1], right=[10,10,10,10]
        let data: Vec<u8> = vec![0, 0, 0, 0, 100, 100, 100, 100];
        let store = make_store(data, 8, 1);
        let residuals = vec![1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        let samples: Vec<u32> = (0..8).collect();

        let mut hist = NodeHistograms::new(1);
        hist.build_from_samples(&store, &samples, &residuals);

        let total_sum: f32 = residuals.iter().sum();
        let total_sum_sq: f32 = residuals.iter().map(|&r| r * r).sum();
        let mut feat_buf: Vec<usize> = vec![0];
        let mut rng = SmallRng::seed_from_u64(42);

        let split = hist
            .find_best_split(total_sum, total_sum_sq, 8, 1, 0, &mut feat_buf, &mut rng)
            .expect("should find a split");

        assert_eq!(split.feature, 0);
        // Any threshold in [0, 99] gives the same partition
        assert!(split.threshold < 100, "threshold={}", split.threshold);
        assert_eq!(split.n_left, 4);
        assert!((split.y_sum_left - 4.0).abs() < 1e-6);
    }

    #[test]
    fn find_split_respects_min_samples_leaf() {
        // 4 cells, 1 feature. Bins: [0, 0, 0, 100]
        // With min_samples_leaf=3, only threshold that puts 3 left / 1 right
        // violates the right side. Should return None.
        let data: Vec<u8> = vec![0, 0, 0, 100];
        let store = make_store(data, 4, 1);
        let residuals = vec![1.0, 2.0, 3.0, 100.0];
        let samples: Vec<u32> = vec![0, 1, 2, 3];

        let mut hist = NodeHistograms::new(1);
        hist.build_from_samples(&store, &samples, &residuals);

        let total_sum: f32 = residuals.iter().sum();
        let total_sum_sq: f32 = residuals.iter().map(|&r| r * r).sum();
        let mut feat_buf: Vec<usize> = vec![0];
        let mut rng = SmallRng::seed_from_u64(0);

        let split = hist.find_best_split(total_sum, total_sum_sq, 4, 3, 0, &mut feat_buf, &mut rng);
        assert!(split.is_none(), "should not find a valid split");
    }

    #[test]
    fn find_split_no_variance() {
        // All residuals identical => no useful split
        let data: Vec<u8> = vec![0, 50, 100, 200];
        let store = make_store(data, 4, 1);
        let residuals = vec![5.0, 5.0, 5.0, 5.0];
        let samples: Vec<u32> = vec![0, 1, 2, 3];

        let mut hist = NodeHistograms::new(1);
        hist.build_from_samples(&store, &samples, &residuals);

        let mut feat_buf: Vec<usize> = vec![0];
        let mut rng = SmallRng::seed_from_u64(0);

        let split = hist.find_best_split(20.0, 100.0, 4, 1, 0, &mut feat_buf, &mut rng);
        assert!(split.is_none());
    }

    #[test]
    fn apply_leaf_updates_residuals() {
        let mut residuals = vec![10.0, 20.0, 30.0, 40.0];
        let train: Vec<u32> = vec![0, 1];
        let oob: Vec<u32> = vec![2, 3];
        // y_sum = 10 + 20 = 30, n_train = 2, pred = 15
        // lr_pred = 0.1 * 15 = 1.5
        let mut improvement = 0.0f32;

        apply_leaf(&mut residuals, &train, &oob, 30.0, 2, 0.1, &mut improvement);

        // All residuals reduced by lr * pred = 1.5
        assert!((residuals[0] - 8.5).abs() < 1e-6);
        assert!((residuals[1] - 18.5).abs() < 1e-6);
        assert!((residuals[2] - 28.5).abs() < 1e-6);
        assert!((residuals[3] - 38.5).abs() < 1e-6);

        // OOB improvement for cell 2: 2*1.5*30 - 1.5^2 = 90 - 2.25 = 87.75
        // OOB improvement for cell 3: 2*1.5*40 - 1.5^2 = 120 - 2.25 = 117.75
        // Total = 205.5
        assert!((improvement - 205.5).abs() < 1e-4, "got {}", improvement);
    }

    // ---- fit_grnboost2_sparse (integration) ----

    #[test]
    fn fit_grnboost2_strong_signal() {
        // 200 cells, 3 features (TFs). Target = 2.0 * TF0 + noise.
        // TF0 should dominate importance.
        let n_cells = 200;
        let n_features = 3;
        let mut rng = SmallRng::seed_from_u64(123);

        // Build quantised feature store directly
        let mut data = vec![0u8; n_features * n_cells];
        for f in 0..n_features {
            for c in 0..n_cells {
                data[f * n_cells + c] = rng.random_range(0..=255u8);
            }
        }
        let store = make_store(data, n_cells, n_features);

        // Target: 2 * (TF0 quantised value) + small noise
        // Build as a sparse axis
        let tf0_col = store.get_col(0);
        let mut target_indices: Vec<u16> = Vec::new();
        let mut target_values: Vec<f32> = Vec::new();
        for c in 0..n_cells {
            let val = 2.0 * tf0_col[c] as f32 + rng.random_range(-5.0..5.0f32);
            if val.abs() > 0.01 {
                target_indices.push(c as u16);
                target_values.push(val);
            }
        }

        let target = SparseAxis::from_vecs_to_csc(target_indices, target_values);

        let config = GradientBoostingConfig {
            n_trees_max: 200,
            learning_rate: 0.05,
            max_depth: 3,
            min_samples_leaf: 5,
            early_stop_window: 20,
            subsample_rate: 0.9,
            n_features_split: 0,
        };

        let imp = fit_grnboost2_sparse(&target, &store, n_cells, &config, 42);

        assert_eq!(imp.len(), n_features);
        // TF0 should have the highest importance
        assert!(
            imp[0] > imp[1] && imp[0] > imp[2],
            "TF0 importance ({}) should dominate: {:?}",
            imp[0],
            imp
        );
        // Normalised to sum to 1
        let sum: f32 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
    }

    #[test]
    fn fit_grnboost2_early_stopping() {
        // Pure noise target: all zero. Early stopping should kick in fast.
        let n_cells = 100;
        let n_features = 2;
        let mut rng = SmallRng::seed_from_u64(99);

        let mut data = vec![0u8; n_features * n_cells];
        for i in 0..data.len() {
            data[i] = rng.random_range(0..=255u8);
        }
        let store = make_store(data, n_cells, n_features);

        // Empty target (all zeros)
        let target = SparseAxis::from_vecs_to_csc(Vec::new(), Vec::new());

        let config = GradientBoostingConfig {
            n_trees_max: 500,
            learning_rate: 0.01,
            max_depth: 3,
            min_samples_leaf: 5,
            early_stop_window: 10,
            subsample_rate: 0.9,
            n_features_split: 0,
        };

        let imp = fit_grnboost2_sparse(&target, &store, n_cells, &config, 0);

        // With zero target, importances should all be zero (no variance to
        // reduce) or the function should terminate quickly.
        let total: f32 = imp.iter().sum();
        assert!(
            total < 1e-6,
            "expected near-zero importance for zero target, got {}",
            total
        );
    }

    #[test]
    fn fit_grnboost2_deterministic() {
        let n_cells = 100;
        let n_features = 2;
        let mut rng = SmallRng::seed_from_u64(7);

        let mut data = vec![0u8; n_features * n_cells];
        for i in 0..data.len() {
            data[i] = rng.random_range(0..=255u8);
        }
        let store = make_store(data, n_cells, n_features);

        let mut t_idx: Vec<u16> = Vec::new();
        let mut t_val: Vec<f32> = Vec::new();
        let col = store.get_col(0);
        for c in 0..n_cells {
            let v = col[c] as f32;
            if v > 0.0 {
                t_idx.push(c as u16);
                t_val.push(v);
            }
        }
        let target = SparseAxis::from_vecs_to_csc(t_idx, t_val);

        let config = GradientBoostingConfig::default();

        let a = fit_grnboost2_sparse(&target, &store, n_cells, &config, 42);
        let b = fit_grnboost2_sparse(&target, &store, n_cells, &config, 42);
        assert_eq!(a, b, "same seed should produce identical results");
    }

    #[test]
    fn pool_acquire_release_cycle() {
        let mut pool = HistogramPool::new(3, 1);
        let a = pool.acquire();
        let b = pool.acquire();
        let c = pool.acquire();
        assert_ne!(a, b);
        assert_ne!(b, c);
        pool.release(b);
        let d = pool.acquire();
        assert_eq!(d, b, "released slot should be reused");
    }

    #[test]
    #[should_panic(expected = "histogram pool exhausted")]
    fn pool_exhaustion_panics() {
        let mut pool = HistogramPool::new(2, 1);
        let _ = pool.acquire();
        let _ = pool.acquire();
        let _ = pool.acquire(); // should panic
    }
}
