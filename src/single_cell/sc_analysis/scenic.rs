use faer::Mat;
use indexmap::IndexSet;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::time::Instant;

use crate::prelude::*;
use crate::utils::simd::sum_simd_f32;

const SCENIC_GENE_CHUNK_SIZE: usize = 100;

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
}

/// Default implementation for the ExtraTreesConfig
impl Default for ExtraTreesConfig {
    fn default() -> Self {
        Self {
            n_trees: 500,
            min_samples_leaf: 50,
            n_features_split: 0,
            n_thresholds: 1,
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

//////////////////
// Tree helpers //
//////////////////

/// Represents a node in a regression decision tree.
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

/// Extracts a column from a CompressedSparseData
///
/// This will specifically extract the second data layer!
///
/// ### Params
///
/// * `mat` - The CompressedSparseData. Needs to be in CSC format.
/// * `j` - Column index to extract
///
/// ### Returns
///
/// Tuple of `(indices, data)`
#[inline]
fn csc_column<'a>(mat: &'a CompressedSparseData<u16, f32>, j: usize) -> (&'a [usize], &'a [f32]) {
    assert!(mat.cs_type.is_csc(), "Needs to be CSC matrix");

    let s = mat.indptr[j];
    let e = mat.indptr[j + 1];
    let vals = mat.data_2.as_ref().expect("TF matrix requires data_2");

    (&mat.indices[s..e], &vals[s..e])
}

/// Galloping (exponential) search for `target` in a sorted `slice`,
/// starting from `hint`.
///
/// ### Params
///
/// TODO
///
/// ### Returns
///
/// TODO
#[inline]
fn gallop_lb(slice: &[usize], target: usize, hint: usize) -> usize {
    let len = slice.len();
    if hint >= len {
        return len;
    }
    // exponential expansion
    let mut lo = hint;
    let mut step = 1usize;
    while lo + step < len && slice[lo + step] < target {
        lo += step;
        step <<= 1;
    }
    let hi = (lo + step).min(len);
    // binary search in [lo, hi)
    let sub = &slice[lo..hi];
    lo + sub.partition_point(|&v| v < target)
}

/// Adaptive intersection of a sorted sparse column (col_indices, col_vals)
/// with a sorted sample_slice.  Writes matching (position_in_sample_slice,
/// feature_val) pairs into `out`.
///
/// Uses galloping search when sample_slice is much smaller than the column,
/// merge scan otherwise.
#[inline]
fn intersect_sparse_samples(
    col_indices: &[usize],
    col_vals: &[f32],
    sample_slice: &[usize],
    out: &mut Vec<(usize, f32)>,
) {
    out.clear();
    let cn = col_indices.len();
    let sn = sample_slice.len();

    if sn == 0 || cn == 0 {
        return;
    }

    // Heuristic: if scanning from sample_slice with galloping is cheaper
    // than a full merge, use galloping.
    // Cost of merge ~ cn + sn;  cost of galloping ~ sn * log2(cn)
    let gallop_cost = sn as f64 * (cn as f64).log2();
    let merge_cost = (cn + sn) as f64;

    if gallop_cost < merge_cost {
        // Galloping: for each sample, find it in the column
        let mut col_hint = 0usize;
        for (si, &sample) in sample_slice.iter().enumerate() {
            col_hint = gallop_lb(col_indices, sample, col_hint);
            if col_hint < cn && col_indices[col_hint] == sample {
                out.push((si, col_vals[col_hint]));
                col_hint += 1;
            }
        }
    } else {
        // Merge scan
        let (mut a, mut b) = (0, 0);
        while a < cn && b < sn {
            match col_indices[a].cmp(&sample_slice[b]) {
                Equal => {
                    out.push((b, col_vals[a]));
                    a += 1;
                    b += 1;
                }
                Less => a += 1,
                Greater => b += 1,
            }
        }
    }
}

///////////////////
// Tree building //
///////////////////

/// Per-tree scratch buffers, allocated once and reused across all nodes.
struct TreeBuffers {
    feat_buf: Vec<usize>,
    nz_buf: Vec<(usize, f32)>,
    rf_buf: Vec<(f32, f32)>,
    partition_buf: Vec<usize>,
}

impl TreeBuffers {
    fn new(n_features: usize) -> Self {
        Self {
            feat_buf: Vec::with_capacity(n_features),
            nz_buf: Vec::new(),
            rf_buf: Vec::new(),
            partition_buf: Vec::new(),
        }
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
    x: &CompressedSparseData<u16, f32>,
    sample_slice: &mut [usize],
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

    let n_features = x.indptr.len() - 1;
    let k = n_features_split.min(n_features);

    bufs.feat_buf.clear();
    bufs.feat_buf.extend(0..n_features);
    for i in 0..k {
        let j = rng.random_range(i..n_features);
        bufs.feat_buf.swap(i, j);
    }

    let mut best_score = 0.0f32;
    let mut best_feature = usize::MAX;
    let mut best_threshold = 0.0f32;
    let mut best_y_sum_r = 0.0f32;
    let mut best_y_sum_sq_r = 0.0f32;

    for fi_idx in 0..k {
        let feat = bufs.feat_buf[fi_idx];
        let (col_indices, col_vals) = csc_column(x, feat);

        intersect_sparse_samples(col_indices, col_vals, sample_slice, &mut bufs.nz_buf);

        if bufs.nz_buf.is_empty() {
            continue;
        }

        let (min_v, max_v) = bufs
            .nz_buf
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &(_, v)| {
                (mn.min(v), mx.max(v))
            });

        let lo = if bufs.nz_buf.len() < n { 0.0f32 } else { min_v };
        if max_v - lo < 1e-10 {
            continue;
        }

        if config.random_threshold() {
            // ExtraTrees: try n_thresholds random thresholds, keep best.
            // Single pass: accumulate n_right, y_sum_r, y_sum_sq_r together.
            for _ in 0..config.n_thresholds() {
                let threshold = rng.random_range(lo..max_v);

                let mut n_right = 0usize;
                let mut y_sum_r = 0.0f32;
                let mut y_sum_sq_r = 0.0f32;

                for &(si, fval) in bufs.nz_buf.iter() {
                    if fval > threshold {
                        n_right += 1;
                        let v = y_dense[sample_slice[si]];
                        y_sum_r += v;
                        y_sum_sq_r += v * v;
                    }
                }

                let n_left = n - n_right;
                if n_left < config.min_samples_leaf() || n_right < config.min_samples_leaf() {
                    continue;
                }

                let y_sum_l = y_sum - y_sum_r;
                let y_sum_sq_l = y_sum_sq - y_sum_sq_r;
                let score = parent_var
                    - (n_left as f32 / n as f32) * node_variance(y_sum_l, y_sum_sq_l, n_left)
                    - (n_right as f32 / n as f32) * node_variance(y_sum_r, y_sum_sq_r, n_right);

                if score > best_score {
                    best_score = score;
                    best_feature = feat;
                    best_threshold = threshold;
                    best_y_sum_r = y_sum_r;
                    best_y_sum_sq_r = y_sum_sq_r;
                }
            }
        } else {
            // RF: sorted sweep over all candidate thresholds
            bufs.rf_buf.clear();
            bufs.rf_buf.extend(bufs.nz_buf.iter().map(|&(si, fval)| {
                let y = y_dense[sample_slice[si]];
                (fval, y)
            }));
            bufs.rf_buf
                .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Equal));

            let (mut y_sum_r, mut y_sum_sq_r) = (0.0f32, 0.0f32);
            let mut n_right = 0usize;

            for i in (0..bufs.rf_buf.len()).rev() {
                let (fval, yval) = bufs.rf_buf[i];
                y_sum_r += yval;
                y_sum_sq_r += yval * yval;
                n_right += 1;

                if i > 0 && (bufs.rf_buf[i - 1].0 - fval).abs() < 1e-10 {
                    continue;
                }

                let n_left = n - n_right;
                if n_left < config.min_samples_leaf() || n_right < config.min_samples_leaf() {
                    continue;
                }

                let y_sum_l = y_sum - y_sum_r;
                let y_sum_sq_l = y_sum_sq - y_sum_sq_r;
                let score = parent_var
                    - (n_left as f32 / n as f32) * node_variance(y_sum_l, y_sum_sq_l, n_left)
                    - (n_right as f32 / n as f32) * node_variance(y_sum_r, y_sum_sq_r, n_right);

                if score > best_score {
                    best_score = score;
                    best_feature = feat;
                    best_threshold = if i > 0 {
                        (bufs.rf_buf[i - 1].0 + fval) / 2.0
                    } else {
                        fval / 2.0
                    };
                    best_y_sum_r = y_sum_r;
                    best_y_sum_sq_r = y_sum_sq_r;
                }
            }
        }
    }

    if best_feature == usize::MAX {
        let idx = nodes.len();
        nodes.push(Node::Leaf { mean });
        return idx;
    }

    // Partition sample_slice into left/right using a single reusable buffer
    let (col_indices, col_vals) = csc_column(x, best_feature);

    bufs.partition_buf.clear();
    bufs.partition_buf.reserve(n);

    // We'll write "right" samples into partition_buf, then rearrange
    // sample_slice in-place as [left..., right...]
    let mut n_left = 0usize;
    {
        let mut a = 0;
        for i in 0..sample_slice.len() {
            let s = sample_slice[i];
            while a < col_indices.len() && col_indices[a] < s {
                a += 1;
            }
            if a < col_indices.len() && col_indices[a] == s && col_vals[a] > best_threshold {
                bufs.partition_buf.push(s);
            } else {
                // Write left samples in-place from the front
                sample_slice[n_left] = s;
                n_left += 1;
            }
        }
    }
    // Copy right samples after left
    sample_slice[n_left..].copy_from_slice(&bufs.partition_buf);

    let y_sum_l = y_sum - best_y_sum_r;
    let y_sum_sq_l = y_sum_sq - best_y_sum_sq_r;

    let node_idx = nodes.len();
    nodes.push(Node::Split {
        feature_idx: best_feature,
        threshold: best_threshold,
        left: usize::MAX,
        right: usize::MAX,
        weighted_impurity_decrease: (n as f32 / n_total as f32) * best_score,
    });

    let (left_sl, right_sl) = sample_slice.split_at_mut(n_left);

    let left_idx = build_node(
        y_dense,
        x,
        left_sl,
        y_sum_l,
        y_sum_sq_l,
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
        best_y_sum_r,
        best_y_sum_sq_r,
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
fn y_stats_from_dense(y_dense: &[f32], samples: &[usize]) -> (f32, f32) {
    let mut sum = 0_f32;
    let mut sum_sq = 0_f32;
    for &s in samples {
        let v = y_dense[s];
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
    feature_matrix: &CompressedSparseData<u16, f32>,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    seed: usize,
) -> Vec<f32> {
    let n_features = feature_matrix.indptr.len() - 1;
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

    // Dense y-lookup: O(n_samples) memory, eliminates all binary searches
    let (y_dense, _y_sum_full, _y_sum_sq_full) = build_y_dense(target_variable, n_samples);

    // Parallel tree building
    let tree_importances: Vec<Vec<f32>> = (0..config.n_trees())
        .into_par_iter()
        .map(|tree_idx| {
            let mut rng =
                SmallRng::seed_from_u64(seed.wrapping_add(tree_idx * 6364136223846793005) as u64);

            let mut sample_indices: Vec<usize> = (0..n_samples).collect();
            let mut bufs = TreeBuffers::new(n_features);
            let mut nodes: Vec<Node> = Vec::new();

            let active_len = if n_sub < n_samples {
                if config.bootstrap() {
                    for i in 0..n_sub {
                        sample_indices[i] = rng.random_range(0..n_samples);
                    }
                    sample_indices[..n_sub].sort_unstable();
                    // Deduplicate
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
                        .for_each(|(i, v)| *v = i);
                    for i in 0..n_sub {
                        let j = rng.random_range(i..n_samples);
                        sample_indices.swap(i, j);
                    }
                    sample_indices[..n_sub].sort_unstable();
                    n_sub
                }
            } else {
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

            let mut importances = vec![0.0f32; n_features];
            accumulate_importances(&nodes, &mut importances);
            importances
        })
        .collect();

    // Merge importances across trees
    let mut importances = vec![0.0f32; n_features];
    for tree_imp in &tree_importances {
        for (i, &v) in tree_imp.iter().enumerate() {
            importances[i] += v;
        }
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
pub fn fit_extra_trees(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &CompressedSparseData<u16, f32>,
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
pub fn fit_random_forest(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &CompressedSparseData<u16, f32>,
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

    if verbose {
        println!(
            "Loaded in and filtered TF data to cells of interest: {:.2?}",
            end_reading
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

        // Outer parallelism over genes; inner parallelism over trees
        // happens inside fit_trees. Rayon's work-stealing handles the nesting.
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
    fn node_variance_single_sample() {
        assert_eq!(node_variance(5.0, 25.0, 1), 0.0);
    }

    #[test]
    fn gallop_lb_basic() {
        let data = [0, 2, 5, 7, 10, 15, 20];
        assert_eq!(gallop_lb(&data, 5, 0), 2);
        assert_eq!(gallop_lb(&data, 6, 0), 3);
        assert_eq!(gallop_lb(&data, 0, 0), 0);
        assert_eq!(gallop_lb(&data, 25, 0), 7);
    }

    #[test]
    fn gallop_lb_with_hint() {
        let data = [0, 2, 5, 7, 10, 15, 20];
        assert_eq!(gallop_lb(&data, 10, 3), 4);
        assert_eq!(gallop_lb(&data, 15, 4), 5);
    }

    #[test]
    fn y_stats_from_dense_basic() {
        let y = [1.0f32, 2.0, 3.0, 0.0, 5.0];
        let samples = [0, 1, 2];
        let (sum, sum_sq) = y_stats_from_dense(&y, &samples);
        assert!((sum - 6.0).abs() < 1e-6);
        assert!((sum_sq - 14.0).abs() < 1e-6);
    }
}
