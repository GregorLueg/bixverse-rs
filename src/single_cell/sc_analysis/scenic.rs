//! Contains the SCENIC implementation from Aibar, et al., Nat Methods, 2017.
//! Several modifications were however implemented:
//!
//! a.) Usage of quantisation and histogram-based splitting. This reduces the
//! size of the predictor variables substantially.
//! b.) Multi-output batching. The original version would create one regression
//! learner per given gene with the TF expression as predictors. In this
//! implementation genes are batched together to reduce number of learners
//! to be trained.
//! c.) To ensure that sensible genes are batched together, the module provides
//! two methods to batch genes together. Completely random to avoid biases of
//! the gene index order generally speaking (fast, but potentially not
//! optimal). And SVD on a subset of cells (for very large data sets) with
//! k-means clustering on the gene loadings to put similar genes together.

use ann_search_rs::prelude::*;
use ann_search_rs::utils::k_means_utils::{assign_all_parallel, train_centroids};
use faer::Mat;
use indexmap::IndexSet;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_processing::pca::pca_on_sc_streaming;

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
    counts: [usize; 256],
    /// Per-bin, per-target Y sums; layout
    y_sums: Vec<f64>,
    /// Per-bin, per-target Y sum-of-squares; same layout as `y_sums`.
    y_sum_sqs: Vec<f64>,
    /// Prefix-sum of `counts` over bins.
    cum_counts: [usize; 256],
    /// Prefix-sum of `y_sums` over bins; same layout as `y_sums`.
    cum_y_sums: Vec<f64>,
    /// Prefix-sum of `y_sum_sqs` over bins; same layout as `y_sums`.
    cum_y_sum_sqs: Vec<f64>,
    /// Left-child Y sums captured at the best split.
    best_y_sums_l: Vec<f64>,
    /// Left-child Y sum-of-squares captured at the best split.
    best_y_sum_sqs_l: Vec<f64>,
    /// Per-target parent variance scratch space.
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
    #[allow(dead_code)]
    fn new(n_features: usize, n_samples: usize, n_targets: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            left_buf: vec![0; n_samples],
            right_buf: vec![0; n_samples],
            left_y_buf: vec![0.0; n_samples * n_targets],
            right_y_buf: vec![0.0; n_samples * n_targets],
            counts: [0usize; 256],
            y_sums: vec![0_f64; 256 * n_targets],
            y_sum_sqs: vec![0_f64; 256 * n_targets],
            cum_counts: [0usize; 256],
            cum_y_sums: vec![0_f64; 256 * n_targets],
            cum_y_sum_sqs: vec![0_f64; 256 * n_targets],
            best_y_sums_l: vec![0_f64; n_targets],
            best_y_sum_sqs_l: vec![0_f64; n_targets],
            parent_vars: vec![0_f64; n_targets],
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
                let y = y_slice[y_base + k] as f64;
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
                let y = tgt_values[i] as f64;
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
            for k in 0..n_targets {
                self.cum_y_sums[curr + k] = self.cum_y_sums[prev + k] + self.y_sums[curr + k];
                self.cum_y_sum_sqs[curr + k] =
                    self.cum_y_sum_sqs[prev + k] + self.y_sum_sqs[curr + k];
            }
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
fn node_variance_f64(sum: f64, sum_sq: f64, n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    f64::max(0.0, sum_sq / nf - (sum / nf) * (sum / nf))
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

    let mut score = 0_f64;
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
        best_y_sums_l.copy_from_slice(&cum_y_sums[h_base..h_base + n_targets]);
        best_y_sum_sqs_l.copy_from_slice(&cum_y_sum_sqs[h_base..h_base + n_targets]);
    }
}

///////////////////
// Tree building //
///////////////////

/// Recursively build a single tree node using sparse Y.
///
/// The key difference from the dense version: no y_slice is passed or
/// partitioned. Only sample_slice is partitioned in place. Y lookups go
/// through the static SparseYBatch.
///
/// ### Params
///
/// * `sparse_y` - SparseYBatch structure
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
#[allow(clippy::too_many_arguments)]
fn build_node_multi_sparse(
    sparse_y: &SparseYBatch,
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

    let mut best_score = 0.0f64;
    let mut best_feature = usize::MAX;
    let mut best_threshold_u8 = 0u8;
    let mut best_n_left = 0usize;

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

    // Accumulate importance (unchanged)
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

    // Partition sample_slice only -- no Y copying
    let mut left_y_sums = [0.0f64; MULTI_OUTPUT_BATCH];
    let mut left_y_sum_sqs = [0.0f64; MULTI_OUTPUT_BATCH];

    left_y_sums[..n_targets].copy_from_slice(&bufs.best_y_sums_l[..n_targets]);
    left_y_sum_sqs[..n_targets].copy_from_slice(&bufs.best_y_sum_sqs_l[..n_targets]);

    let mut right_y_sums = [0.0f64; MULTI_OUTPUT_BATCH];
    let mut right_y_sum_sqs = [0.0f64; MULTI_OUTPUT_BATCH];
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

    let mut total_parent_var = 0_f64;
    for k in 0..n_targets {
        let v = node_variance_f64(y_sums[k], y_sum_sqs[k], n);
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

    let mut best_score = 0_f64;
    let mut best_feature = usize::MAX;
    let mut best_threshold_u8 = 0u8;
    let mut best_n_left = 0usize;

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

    // stack-allocated buffers: n_targets <= MULTI_OUTPUT_BATCH always holds.
    // avoids heap allocations
    let mut left_y_sums = [0_f64; MULTI_OUTPUT_BATCH];
    let mut left_y_sum_sqs = [0_f64; MULTI_OUTPUT_BATCH];
    let mut right_y_sums = [0_f64; MULTI_OUTPUT_BATCH];
    let mut right_y_sum_sqs = [0_f64; MULTI_OUTPUT_BATCH];

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
    let mut importances = vec![0_f64; n_features * n_targets];
    let mut y_sums_root = vec![0_f64; n_targets];
    let mut y_sum_sqs_root = vec![0_f64; n_targets];

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
        let mut total = 0_f64;
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

    // build once, shared across all trees
    let sparse_y = SparseYBatch::from_targets(targets, n_samples);

    let mut sample_indices: Vec<u32> = vec![0; n_samples];
    // no more root_y_buf, left_y_buf, right_y_buf
    let mut bufs = TreeBuffers::new_sparse(n_features, n_samples, n_targets);
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

        // compute root sufficient statistics from sparse Y
        y_sums_root.fill(0.0);
        y_sum_sqs_root.fill(0.0);
        for &s in active.iter() {
            let (tgt_idx, tgt_val) = sparse_y.cell_entries(s as usize);
            for i in 0..tgt_idx.len() {
                let k = tgt_idx[i] as usize;
                let y = tgt_val[i] as f64;
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

    // normalise importances
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

/// Run SCENIC GRN inference and return a TF-by-gene importance matrix
///
/// ### Params
///
/// * `f_path` - Path to the sparse gene expression file.
/// * `cell_indices` - Indices of cells to use.
/// * `gene_indices` - Target gene indices.
/// * `tf_indices` - Transcription factor gene indices (predictors).
/// * `scenic_params` - Reference to the SCENIC parameters indicating the gene
///   batch strategy, gene batch size,
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
/// ensembles via `fit_multi_trees_sparse`. Batches within a chunk are
/// parallelised across threads with Rayon.
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
    let n_multi_output = scenic_params
        .gene_batch_size
        .unwrap_or(MULTI_OUTPUT_BATCH)
        .min(MULTI_OUTPUT_BATCH);

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

    if verbose {
        println!(
            "Loaded, filtered and quantised TF data (n: {}) in: {:.2?}",
            tf_data.n_features.separate_with_underscores(),
            start_reading.elapsed()
        );
    }

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

    let n_genes = gene_indices.len();
    let n_tfs = tf_data.n_features;
    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];

    let config: &dyn TreeRegressorConfig = match &scenic_params.regression_learner {
        RegressionLearner::ExtraTrees(cfg) => cfg,
        RegressionLearner::RandomForest(cfg) => cfg,
    };

    if verbose {
        println!(
            "Running SCENIC on {} genes ({} TFs, {} cells, batches of {})",
            n_genes, n_tfs, n_cells, n_multi_output,
        );
    }

    thread::scope(|scope| {
        let (tx, rx) = mpsc::sync_channel::<(Vec<usize>, Vec<SparseAxis<u16, f32>>)>(2);

        let cell_set_ref = &cell_set;
        let reader_ref = &reader;

        // Producer: read and filter gene data in chunks
        scope.spawn(move || {
            for chunk in ordered_genes.chunks(SCENIC_GENE_CHUNK_SIZE) {
                let gene_ids = chunk.to_vec();
                let mut gene_chunks_target: Vec<CscGeneChunk> =
                    reader_ref.read_gene_parallel(chunk);
                gene_chunks_target.iter_mut().for_each(|c| {
                    c.filter_selected_cells(cell_set_ref);
                });
                let sparse_columns: Vec<SparseAxis<u16, f32>> = gene_chunks_target
                    .iter()
                    .map(|c| c.to_sparse_axis(cell_set_ref.len()))
                    .collect();
                if tx.send((gene_ids, sparse_columns)).is_err() {
                    break;
                }
            }
        });

        // Consumer: fit multi-output ensembles in parallel batches
        for (chunk_idx, (gene_ids, sparse_columns)) in rx.iter().enumerate() {
            let start_chunk = Instant::now();

            // Pair gene_ids with their sparse columns, then chunk into
            // batches of n_multi_output
            let id_chunks: Vec<&[usize]> = gene_ids.chunks(n_multi_output).collect();
            let col_chunks: Vec<&[SparseAxis<u16, f32>]> =
                sparse_columns.chunks(n_multi_output).collect();

            let total_batches = col_chunks.len();
            let batches_done = AtomicUsize::new(0);

            let batch_results: Vec<Vec<Vec<f32>>> = col_chunks
                .par_iter()
                .enumerate()
                .map(|(batch_idx, batch)| {
                    let batch_seed = seed.wrapping_add((chunk_idx * 1000 + batch_idx) * 2654435761);
                    let result =
                        fit_multi_trees_sparse(batch, &tf_data, n_cells, config, batch_seed);

                    if verbose && total_batches >= 4 {
                        let done = batches_done.fetch_add(1, Ordering::Relaxed) + 1;
                        let pct = done * 100 / total_batches;
                        let prev_pct = (done - 1) * 100 / total_batches;
                        if [25, 50, 75, 100].iter().any(|&q| prev_pct < q && pct >= q) {
                            println!(
                                "    Chunk {}: ~{}% of batches done ({}/{})",
                                chunk_idx + 1,
                                pct,
                                done,
                                total_batches
                            );
                        }
                    }

                    result
                })
                .collect();

            for (batch_idx, batch_result) in batch_results.into_iter().enumerate() {
                let batch_gene_ids = id_chunks[batch_idx];
                for (local_idx, imp) in batch_result.into_iter().enumerate() {
                    let gene_id = batch_gene_ids[local_idx];
                    let original_pos = gene_id_to_pos[&gene_id];
                    importance_scores[original_pos] = imp;
                }
            }

            if verbose {
                let genes_done = ((chunk_idx + 1) * SCENIC_GENE_CHUNK_SIZE).min(n_genes);
                println!(
                    "  Chunk {}: {}/{} genes done in {:.2?}",
                    chunk_idx + 1,
                    genes_done,
                    n_genes,
                    start_chunk.elapsed()
                );
            }
        }
    });

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
        let mut cum_counts = [0usize; 256];
        let mut cum_y_sums = vec![0.0f64; 256];
        let mut cum_y_sum_sqs = vec![0.0f64; 256];

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

        let parent_vars = vec![node_variance_f64(26.0, 254.0, 4)];
        let y_sums_total = vec![26.0f64];
        let y_sum_sqs_total = vec![254.0f64];

        let mut best_score = 0.0f64;
        let mut best_feature = usize::MAX;
        let mut best_threshold = 0u8;
        let mut best_n_left = 0usize;
        let mut best_ys_l = vec![0.0f64; 1];
        let mut best_yss_l = vec![0.0f64; 1];

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
}
