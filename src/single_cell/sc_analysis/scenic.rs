use faer::Mat;
use indexmap::IndexSet;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::prelude::*;
use crate::utils::simd::sum_simd_f32;

const SCENIC_GENE_CHUNK_SIZE: usize = 1000;

///////////
// Enums //
///////////

/// Enum to define the type of Regression learner to use
#[derive(Clone, Debug)]
pub enum RegressionLearner {
    ExtraTrees(ExtraTreesConfig),
    RandomForest(RandomForestConfig),
}

/// Default implementation for RegressionLearner
impl Default for RegressionLearner {
    fn default() -> Self {
        RegressionLearner::ExtraTrees(ExtraTreesConfig::default())
    }
}

/// Parse the regression learner
///
/// ### Params
///
/// * `s` - String to parse
///
/// ### Returns
///
/// The Option of the `RegressionLearner` Enum with the chosen learner
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

/// Trait to access different tree regression-based hyper parameters
trait TreeRegressorConfig: Sync {
    /// Number of trees
    ///
    /// ### Returns
    ///
    /// The number of trees to use
    fn n_trees(&self) -> usize;

    /// Minimum number of samples
    ///
    /// ### Returns
    ///
    /// The minimum number of samples per given leaf
    fn min_samples_leaf(&self) -> usize;

    /// How many features to split
    ///
    /// ### Returns
    ///
    /// The number of features to split
    fn n_features_split(&self) -> usize;

    /// Shall a random threshold be used (ExtraTrees) or an optimal threshold
    /// (RF)
    ///
    /// ### Returns
    ///
    /// Boolean
    fn random_threshold(&self) -> bool;

    /// Number of samples to subsample
    ///
    /// ### Returns
    ///
    /// How many of the samples to subsample per given tree
    fn subsample_rate(&self) -> f32 {
        1.0
    }

    /// Shall the subsampling be done with bootstrapping
    ///
    /// ### Returns
    ///
    /// Boolean indicating if bootstrapping should be used
    fn bootstrap(&self) -> bool {
        false
    }

    /// Optional max_depth parameter
    ///
    /// ### Returns
    ///
    /// Shall the tree-depth be limited
    fn max_depth(&self) -> Option<usize> {
        None
    }

    /// Min variance in a given leaf
    ///
    /// ### Returns
    ///
    /// Returns the min variance threshold
    fn min_variance(&self) -> f32 {
        1e-10
    }

    /// Number of random thresholds to use
    ///
    /// Used for ManyTrees
    ///
    /// ### Returns
    ///
    /// Number of thresholds to use
    fn n_thresholds(&self) -> usize {
        1
    }
}

////////////
// Params //
////////////

////////////////
// ExtraTrees //
////////////////

/// Parameters for the extra tree regression
///
/// ### Fields
///
/// * `n_trees` - Number of trees to fit
/// * `min_samples_leaf` - Minimum number of samples per leaf. Will control
///   the depth.
/// * `n_features_split` - Number of features to per split
///   (if zero = sqrt(features))
/// * `n_thresholds` - Number of random thresholds to test
#[derive(Clone, Debug)]
pub struct ExtraTreesConfig {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub n_features_split: usize,
    pub n_thresholds: usize,
    pub max_depth: Option<usize>,
}

/// Default implementation for the ExtraTreesConfig
impl Default for ExtraTreesConfig {
    fn default() -> Self {
        Self {
            n_trees: 500,
            min_samples_leaf: 50,
            n_features_split: 0,
            n_thresholds: 1,
            max_depth: Some(15),
        }
    }
}

/// Trait implementation for ExtraTreesConfig
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
}

//////////////////
// RandomForest //
//////////////////

/// Parameters for the Random Forest regression
///
/// ### Fields
///
/// * `n_trees` - Number of trees to fit
/// * `min_samples_leaf` - Minimum number of samples per leaf. Will control
///   the depth.
/// * `n_features_split` - Number of features to per split
///   (if zero = sqrt(features))
/// * `subsample_rate` - Number of samples to use per tree
/// * `bootstrap` - Shall the subsampling be bootstrapped
/// * `max_depth` - Shall the the tree depth be limited
#[derive(Clone, Debug)]
pub struct RandomForestConfig {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub n_features_split: usize,
    pub subsample_rate: f32,
    pub bootstrap: bool,
    pub max_depth: Option<usize>,
}

/// Default implementation for the RandomForestConfig
impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 200,
            min_samples_leaf: 50,
            n_features_split: 0,
            subsample_rate: 0.632,
            bootstrap: false,
            max_depth: None,
        }
    }
}

/// Trait implementation for RandomForestConfig
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
}

/////////////////////
// Storage helpers //
/////////////////////

///////////////
// Quantiser //
///////////////

/// Structure to store quantised (dense) values
///
/// ### Fields
///
/// * `data` - Flat store of the quantised expression values
/// * `n_cells` - Number of cells
/// * `n_features` - Number of features
/// * `feature_mins` - Minimum values for reconstruction
/// * `feature_range` - Range values for reconstruction
pub struct DenseQuantisedStore {
    /// Flattened column-major data: [TF0_cell0, TF0_cell1, ..., TF1_cell0, ...]
    data: Vec<u8>,
    n_cells: usize,
    n_features: usize,
    feature_min: Vec<f32>,
    feature_range: Vec<f32>,
}

impl DenseQuantisedStore {
    /// Generate a new instance from the CompressedSparseData
    ///
    /// ### Params
    ///
    /// * `mat` - The initial matrix
    /// * `n_cells` - Number of cells being tested
    ///
    /// ### Returns
    ///
    /// Initialised self
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

            // find min/max
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

            // quantise
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

    /// Get the values of a given column
    ///
    /// ### Returns
    ///
    /// Quantised self
    #[inline(always)]
    pub fn get_col(&self, tf_idx: usize) -> &[u8] {
        let start = tf_idx * self.n_cells;
        &self.data[start..start + self.n_cells]
    }
}

////////////////
// Histograms //
////////////////

/// Structure for the histograms
///
/// ### Params
///
/// * `count` - Number of samples
/// * `y_sum` - Sum of the prediction variable
/// * `y_sum_sq` - Squared sum of the prediction variable
#[derive(Clone, Copy, Default)]
struct HistogramBin {
    count: usize,
    y_sum: f32,
    y_sum_sq: f32,
}

//////////////////
// Tree helpers //
//////////////////

/// Represents a node in a regression decision tree.
#[allow(dead_code)]
enum Node {
    /// A leaf (terminal) node containing a constant prediction.
    ///
    /// ### Fields
    ///
    /// * `mean` - The predicted regression value
    Leaf { mean: f32 },

    /// An internal split node that partitions samples by a feature threshold.
    ///
    /// ### Fields
    ///
    /// * `feature_idx` - Index of the feature used for the split
    /// * `threshold` - Threshold value; samples where feature < threshold go
    ///   left
    /// * `left` - Index of the left child node
    /// * `right` - Index of the right child node
    /// * `weighted_impurity_decrease` - Importance metric:
    ///   `variance_reduction * (n_node / n_total)`  following sklearn's
    ///   convention
    Split {
        feature_idx: usize,
        threshold: f32,
        left: usize,
        right: usize,
        weighted_impurity_decrease: f32,
    },
}

/// Variance of the node
///
/// Computes sample variance using the computational formula:
/// `Var = E[X²] - E[X]²`
/// (Does clamping to avoid issues)
///
/// ### Params
///
/// * `sum` - Sum of all values in the node
/// * `sum_sq` - Sum of squared values
/// * `n` - Number of samples in the node
///
/// ### Returns
///
/// Sample variance.
#[inline]
fn node_variance(sum: f32, sum_sq: f32, n: usize) -> f32 {
    if n < 2 {
        return 0_f32;
    }
    let nf = n as f32;
    f32::max(0_f32, sum_sq / nf - (sum / nf) * (sum / nf))
}

///////////////////
// Tree building //
///////////////////

/// Per-tree scratch buffers, allocated once and reused across all nodes.
struct TreeBuffers {
    feat_buf: Vec<usize>,
    left_buf: Vec<u32>,
    right_buf: Vec<u32>,
    hist: [HistogramBin; 256],
    cum_hist: [HistogramBin; 256], // Prefix sum for O(1) split evaluation
}

impl TreeBuffers {
    fn new(n_features: usize, n_samples: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            left_buf: vec![0; n_samples],
            right_buf: vec![0; n_samples],
            hist: [HistogramBin::default(); 256],
            cum_hist: [HistogramBin::default(); 256],
        }
    }

    /// Builds the histogram and the cumulative prefix-sum histogram
    #[inline]
    fn build_histograms(&mut self, tf_col: &[u8], sample_slice: &[u32], y_dense: &[f32]) {
        self.hist.fill(HistogramBin::default());

        // O(N) branchless construction
        for &s in sample_slice {
            let bin_idx = tf_col[s as usize] as usize;
            let y = y_dense[s as usize];
            self.hist[bin_idx].count += 1;
            self.hist[bin_idx].y_sum += y;
            self.hist[bin_idx].y_sum_sq += y * y;
        }

        // O(256) cumulative prefix sum
        let mut acc = HistogramBin::default();
        for i in 0..256 {
            acc.count += self.hist[i].count;
            acc.y_sum += self.hist[i].y_sum;
            acc.y_sum_sq += self.hist[i].y_sum_sq;
            self.cum_hist[i] = acc;
        }
    }
}

/// Helper function to evaluate variance reduction in O(1) using the cumulative histogram
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split(
    threshold: usize,
    feat: usize,
    parent_var: f32,
    n: usize,
    y_sum: f32,
    y_sum_sq: f32,
    bufs: &TreeBuffers,
    config: &dyn TreeRegressorConfig,
    best_score: &mut f32,
    best_feature: &mut usize,
    best_threshold_u8: &mut u8,
    best_y_sum_l: &mut f32,
    best_y_sum_sq_l: &mut f32,
) {
    let left_stats = &bufs.cum_hist[threshold];
    let n_left = left_stats.count;
    let n_right = n - n_left;

    if n_left < config.min_samples_leaf() || n_right < config.min_samples_leaf() {
        return;
    }

    let y_sum_l = left_stats.y_sum;
    let y_sum_sq_l = left_stats.y_sum_sq;
    let y_sum_r = y_sum - y_sum_l;
    let y_sum_sq_r = y_sum_sq - y_sum_sq_l;

    let score = parent_var
        - (n_left as f32 / n as f32) * node_variance(y_sum_l, y_sum_sq_l, n_left)
        - (n_right as f32 / n as f32) * node_variance(y_sum_r, y_sum_sq_r, n_right);

    if score > *best_score {
        *best_score = score;
        *best_feature = feat;
        *best_threshold_u8 = threshold as u8;
        *best_y_sum_l = y_sum_l;
        *best_y_sum_sq_l = y_sum_sq_l;
    }
}

/// Recursively builds a regression tree node using variance reduction.
///
/// Key differences from original:
/// - `y_dense` is a pre-computed dense array, eliminating binary searches
/// - Adaptive intersection (galloping vs merge) for sparse column lookups
/// - Single scratch `partition_buf` reused across all recursive calls
/// - Single-pass split evaluation for ExtraTrees
#[allow(clippy::too_many_arguments)]
fn build_node(
    y_dense: &[f32],
    x: &DenseQuantisedStore,
    sample_slice: &mut [u32],
    y_sum: f32,
    y_sum_sq: f32,
    n_total: usize,
    n_features_split: usize,
    config: &dyn TreeRegressorConfig,
    depth: usize,
    nodes: &mut Vec<Node>,
    bufs: &mut TreeBuffers,
    rng: &mut SmallRng,
) -> usize {
    let n = sample_slice.len();
    let mean = y_sum / n as f32;
    let parent_var = node_variance(y_sum, y_sum_sq, n);

    let max_depth_reached = config.max_depth().map_or(false, |d| depth >= d);

    if n < 2 * config.min_samples_leaf() || parent_var < config.min_variance() || max_depth_reached
    {
        let idx = nodes.len();
        nodes.push(Node::Leaf { mean });
        return idx;
    }

    let n_features = x.n_features;
    let k = n_features_split.min(n_features);

    // Partial Fisher-Yates: feat_buf is initialised once in TreeBuffers::new
    // and always contains each index exactly once. Just shuffle the first k.
    for i in 0..k {
        let j = rng.random_range(i..n_features);
        bufs.feat_buf.swap(i, j);
    }

    let mut best_score = 0.0f32;
    let mut best_feature = usize::MAX;
    let mut best_threshold_u8 = 0u8;
    let mut best_y_sum_l = 0.0f32;
    let mut best_y_sum_sq_l = 0.0f32;

    for fi_idx in 0..k {
        let feat = bufs.feat_buf[fi_idx];
        let tf_col = x.get_col(feat);

        bufs.build_histograms(tf_col, sample_slice, y_dense);

        let min_bin = bufs.hist.iter().position(|b| b.count > 0).unwrap_or(0);
        let max_bin = bufs.hist.iter().rposition(|b| b.count > 0).unwrap_or(255);

        if min_bin == max_bin {
            continue;
        }

        if config.random_threshold() {
            for _ in 0..config.n_thresholds() {
                let threshold = rng.random_range(min_bin..max_bin);
                evaluate_split(
                    threshold,
                    feat,
                    parent_var,
                    n,
                    y_sum,
                    y_sum_sq,
                    bufs,
                    config,
                    &mut best_score,
                    &mut best_feature,
                    &mut best_threshold_u8,
                    &mut best_y_sum_l,
                    &mut best_y_sum_sq_l,
                );
            }
        } else {
            for threshold in min_bin..max_bin {
                evaluate_split(
                    threshold,
                    feat,
                    parent_var,
                    n,
                    y_sum,
                    y_sum_sq,
                    bufs,
                    config,
                    &mut best_score,
                    &mut best_feature,
                    &mut best_threshold_u8,
                    &mut best_y_sum_l,
                    &mut best_y_sum_sq_l,
                );
            }
        }
    }

    if best_feature == usize::MAX {
        let idx = nodes.len();
        nodes.push(Node::Leaf { mean });
        return idx;
    }

    let tf_col = x.get_col(best_feature);
    let mut l_idx = 0;
    let mut r_idx = 0;

    for i in 0..n {
        let s = sample_slice[i];
        let val = tf_col[s as usize];
        let is_right = (val > best_threshold_u8) as usize;
        let is_left = 1 - is_right;

        bufs.left_buf[l_idx] = s;
        bufs.right_buf[r_idx] = s;

        l_idx += is_left;
        r_idx += is_right;
    }

    sample_slice[..l_idx].copy_from_slice(&bufs.left_buf[..l_idx]);
    sample_slice[l_idx..].copy_from_slice(&bufs.right_buf[..r_idx]);

    let y_sum_r = y_sum - best_y_sum_l;
    let y_sum_sq_r = y_sum_sq - best_y_sum_sq_l;

    let node_idx = nodes.len();
    nodes.push(Node::Split {
        feature_idx: best_feature,
        threshold: x.feature_min[best_feature]
            + (best_threshold_u8 as f32 / 255.0) * x.feature_range[best_feature],
        left: usize::MAX,
        right: usize::MAX,
        weighted_impurity_decrease: (n as f32 / n_total as f32) * best_score,
    });

    let (left_sl, right_sl) = sample_slice.split_at_mut(l_idx);

    let left_idx = build_node(
        y_dense,
        x,
        left_sl,
        best_y_sum_l,
        best_y_sum_sq_l,
        n_total,
        n_features_split,
        config,
        depth + 1,
        nodes,
        bufs,
        rng,
    );
    let right_idx = build_node(
        y_dense,
        x,
        right_sl,
        y_sum_r,
        y_sum_sq_r,
        n_total,
        n_features_split,
        config,
        depth + 1,
        nodes,
        bufs,
        rng,
    );

    if let Node::Split { left, right, .. } = &mut nodes[node_idx] {
        *left = left_idx;
        *right = right_idx;
    }

    node_idx
}

/// Calculate the importance values
///
/// ### Params
///
/// * `nodes` - The slice of nodes
/// * `importances` - Mutable references to a slice of importances
fn accumulate_importances(nodes: &[Node], importances: &mut [f32]) {
    for node in nodes {
        if let Node::Split {
            feature_idx,
            weighted_impurity_decrease,
            ..
        } = node
        {
            importances[*feature_idx] += weighted_impurity_decrease;
        }
    }
}

/// Build the dense y-lookup from sparse target data.
///
/// Returns (y_dense, y_sum, y_sum_sq) where y_dense[cell_id] = expression value.
fn build_y_dense(target_variable: &SparseAxis<u16, f32>, n_samples: usize) -> (Vec<f32>, f32, f32) {
    let (y_indices, y_data) = target_variable.get_indices_data_2();
    let mut y_dense = vec![0.0f32; n_samples];
    let mut y_sum = 0.0f32;
    let mut y_sum_sq = 0.0f32;
    for (i, &idx) in y_indices.iter().enumerate() {
        let v = y_data[i];
        y_dense[idx] = v;
        y_sum += v;
        y_sum_sq += v * v;
    }
    (y_dense, y_sum, y_sum_sq)
}

/// Compute y_sum and y_sum_sq for a subsample from the dense lookup.
#[inline]
fn y_stats_from_dense(y_dense: &[f32], samples: &[u32]) -> (f32, f32) {
    let mut sum = 0_f32;
    let mut sum_sq = 0_f32;
    for &s in samples {
        let v = y_dense[s as usize];
        sum += v;
        sum_sq += v * v;
    }
    (sum, sum_sq)
}

/// Fit trees with nested parallelism.
///
/// Trees are built in parallel via rayon. Each thread gets its own RNG,
/// node vec, and scratch buffers. Importances are merged after all trees
/// complete.
fn fit_trees(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    seed: usize,
) -> Vec<f32> {
    let n_features = feature_matrix.n_features;
    let n_features_split = if config.n_features_split() == 0 {
        ((n_features as f64).sqrt() as usize).max(1)
    } else {
        config.n_features_split()
    };

    let n_sub = if config.subsample_rate() >= 1.0 {
        n_samples
    } else {
        ((n_samples as f32 * config.subsample_rate()).round() as usize)
            .max(2 * config.min_samples_leaf())
    };

    let (y_dense, _y_sum_full, _y_sum_sq_full) = build_y_dense(target_variable, n_samples);

    // Sequential tree loop: reuse all buffers across trees
    let mut sample_indices: Vec<u32> = (0..n_samples).map(|x| x as u32).collect();
    let mut bufs = TreeBuffers::new(n_features, n_samples);
    let mut nodes: Vec<Node> = Vec::new();
    let mut importances = vec![0.0f32; n_features];

    for tree_idx in 0..config.n_trees() {
        nodes.clear();

        let mut rng =
            SmallRng::seed_from_u64(seed.wrapping_add(tree_idx * 6364136223846793005) as u64);

        let active_len = if n_sub < n_samples {
            if config.bootstrap() {
                for i in 0..n_sub {
                    sample_indices[i] = rng.random_range(0..n_samples) as u32;
                }
                let mut w = 1usize;
                for r in 1..n_sub {
                    if sample_indices[r] != sample_indices[r - 1] {
                        sample_indices[w] = sample_indices[r];
                        w += 1;
                    }
                }
                w
            } else {
                sample_indices
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v = i as u32);
                for i in 0..n_sub {
                    let j = rng.random_range(i..n_samples);
                    sample_indices.swap(i, j);
                }
                sample_indices[..n_sub].sort_unstable();
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
        let (y_sum, y_sum_sq) = y_stats_from_dense(&y_dense, active);

        build_node(
            &y_dense,
            feature_matrix,
            active,
            y_sum,
            y_sum_sq,
            active_len,
            n_features_split,
            config,
            0,
            &mut nodes,
            &mut bufs,
            &mut rng,
        );

        accumulate_importances(&nodes, &mut importances);
    }

    let total: f32 = sum_simd_f32(&importances);
    if total > 0.0 {
        importances.iter_mut().for_each(|v| *v /= total);
    }
    importances
}

////////////////
// ExtraTrees //
////////////////

/// Fit an Extra Trees regressor and return normalised feature importances.
///
/// Thin wrapper around `fit_trees` using `ExtraTreesConfig`. Splits are
/// chosen at a random threshold between the min and max of each feature,
/// with no subsampling.
///
/// ### Params
///
/// * `target_variable` - Sparse target gene expression to predict
/// * `feature_matrix` - CSC sparse matrix of TF expression levels
/// * `n_samples` - Number of cells
/// * `config` - ExtraTrees hyperparameters
/// * `seed` - Base random seed; each tree derives its own seed from this
///
/// ### Returns
///
/// A `Vec<f32>` of length `n_tfs` with importances normalised to sum to 1.
fn fit_extra_trees(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &ExtraTreesConfig,
    seed: usize,
) -> Vec<f32> {
    fit_trees(target_variable, feature_matrix, n_samples, config, seed)
}

//////////////////
// RandomForest //
//////////////////

/// Fit a Random Forest regressor and return normalised feature importances.
///
/// Thin wrapper around `fit_trees` using `RandomForestConfig`. Splits are
/// chosen by sweeping all candidate thresholds in sorted order. Supports
/// subsampling with or without bootstrap.
///
/// ### Params
///
/// * `target_variable` - Sparse target gene expression to predict
/// * `feature_matrix` - CSC sparse matrix of TF expression levels
/// * `n_samples` - Number of cells
/// * `config` - RandomForest hyperparameters
/// * `seed` - Base random seed; each tree derives its own seed from this
///
/// ### Returns
///
/// A `Vec<f32>` of length `n_tfs` with importances normalised to sum to 1.
fn fit_random_forest(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &RandomForestConfig,
    seed: usize,
) -> Vec<f32> {
    fit_trees(target_variable, feature_matrix, n_samples, config, seed)
}

//////////
// Main //
//////////

/// Returns the genes to include in a SCENIC analysis
///
/// Returns genes that make sense to include in the scenic analysis.
///
/// ### Params
///
/// * `min_total_counts` - Minimum number of total counts across all cells that
///   the gene has to be expressed in.
/// * `min_cells` - Proportion of cells that the gene has to be expressed in.
///
/// ### Returns
///
/// Vec of usizes with gene indices to include
pub fn scenic_gene_filter(
    f_path: &str,
    cell_indices: &[usize],
    min_counts: usize,
    min_cells: f32,
) -> Vec<usize> {
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let total_genes = reader.get_header().total_genes;
    let all_gene_indices: Vec<usize> = (0..total_genes).collect();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_indices.len();

    let mut passing = Vec::new();

    for chunk in all_gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE) {
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
    }

    passing
}

/// Generate the gene-regulatory network using the SCENIC pipeline.
///
/// Predicts the expression of each gene of interest from TF expression levels
/// using an ensemble tree regressor (ExtraTrees or RandomForest). Feature
/// importances are accumulated per gene and stored in `importance_scores`.
/// Genes are processed in chunks of 100 to bound peak memory.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file.
/// * `cell_indices` - Indices of cells to include in the analysis.
/// * `gene_indices` - Indices of target genes to regress.
/// * `tf_indices` - Indices of transcription factor features.
/// * `learner` - Regression algorithm and its configuration.
/// * `seed` - Random seed. Each tree gets its own seed.
/// * `verbose` - Controls verbosity.
pub fn run_scenic_grn(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    learner: &RegressionLearner,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_reading = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();
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
            "Loaded in and filtered TF data (n: {}) to cells of interest: {:.2?}",
            tf_data.n_features, end_reading
        );
    }

    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); gene_indices.len()];

    for (chunk_idx, chunk) in gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        if verbose {
            println!(
                "Processing gene chunk {}/{} ({} genes)",
                chunk_idx + 1,
                gene_indices.len().div_ceil(SCENIC_GENE_CHUNK_SIZE),
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

        // outer parallelism over genes; inner parallelism over trees
        // i will trust rayon to handle this
        let chunk_importances: Vec<Vec<f32>> = sparse_columns
            .par_iter()
            .map(|gene| match learner {
                RegressionLearner::ExtraTrees(cfg) => {
                    fit_trees(gene, &tf_data, cell_set.len(), cfg, seed)
                }
                RegressionLearner::RandomForest(cfg) => {
                    fit_trees(gene, &tf_data, cell_set.len(), cfg, seed)
                }
            })
            .collect();

        let base = chunk_idx * SCENIC_GENE_CHUNK_SIZE;
        for (i, importances) in chunk_importances.into_iter().enumerate() {
            importance_scores[base + i] = importances;
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

    let n_genes = importance_scores.len();
    let n_tfs = if n_genes > 0 {
        importance_scores[0].len()
    } else {
        0
    };

    Mat::from_fn(n_genes, n_tfs, |i, j| importance_scores[i][j])
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_variance_basic() {
        let (sum, sum_sq) = (6.0f32, 14.0f32);
        let v = node_variance(sum, sum_sq, 3);
        assert!((v - 2.0 / 3.0).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn node_variance_uniform() {
        let (sum, sum_sq) = (9.0f32, 27.0f32);
        let v = node_variance(sum, sum_sq, 3);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn branchless_partitioning_logic() {
        let tf_col: Vec<u8> = vec![10, 50, 200, 30, 250, 100];
        // sample slice only contains subset of indices
        let sample_slice: Vec<usize> = vec![0, 1, 2, 4]; // values: 10, 50, 200, 250

        let mut left_buf = vec![0; 4];
        let mut right_buf = vec![0; 4];
        let threshold = 100u8;

        let mut l_idx = 0;
        let mut r_idx = 0;

        for &s in sample_slice.iter() {
            let val = tf_col[s];
            let is_right = (val > threshold) as usize;
            let is_left = 1 - is_right;

            left_buf[l_idx] = s;
            right_buf[r_idx] = s;

            l_idx += is_left;
            r_idx += is_right;
        }

        assert_eq!(l_idx, 2); // 10 and 50 are <= 100
        assert_eq!(r_idx, 2); // 200 and 250 are > 100

        assert_eq!(&left_buf[..l_idx], &[0, 1]);
        assert_eq!(&right_buf[..r_idx], &[2, 4]);
    }
}
