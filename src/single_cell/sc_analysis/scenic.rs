use indexmap::IndexSet;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::time::Instant;

use crate::prelude::*;
use crate::utils::simd::sum_simd_f32;

///////////
// Enums //
///////////

/// Enum to define the type of Regression learner to use
#[derive(Clone, Debug, Default)]
pub enum RegressionLearner {
    /// ExtraTree regression. Simple and fast; might not yield the best
    /// regression results.
    #[default]
    ExtraTrees,
    /// RandomForest learner. To be implemented
    RandomForest,
    /// GradientBoosted learner. To be implemented
    GradientBoosted,
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
        "extratrees" => Some(RegressionLearner::ExtraTrees),
        "rf" | "randomforest" => Some(RegressionLearner::RandomForest),
        "boost" | "gradient_boost" => Some(RegressionLearner::GradientBoosted),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Parameters for the extra tree regression
///
/// ### Fields
///
/// * `n_trees` - Number of trees to fit
/// * `min_samples_leaf` - Minimum number of samples per leaf. Will control
///   the depth.
/// * `n_features_split` - Number of features to per split
///   (if zero = sqrt(features))
#[derive(Clone, Debug)]
pub struct ExtraTreesConfig {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub n_features_split: usize,
}

/// Default implementation for the Config
impl Default for ExtraTreesConfig {
    fn default() -> Self {
        Self {
            n_trees: 500,
            min_samples_leaf: 50,
            n_features_split: 0,
        }
    }
}

///////////////////////
// ExtraTree helpers //
///////////////////////

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

/// Sorted merge intersection
///
/// ### Params
///
/// * `y_indices` - Sorted array of indices into `y_data`
/// * `y_data` - Target variable values; indexed by `y_indices`
/// * `samples` - Sorted array of sample indices to intersect with `y_indices`
///
/// ### Returns
///
/// A tuple `(sum, sum_sq)`
#[inline]
fn y_stats_merge(y_indices: &[usize], y_data: &[f32], samples: &[usize]) -> (f32, f32) {
    let (mut sum, mut sum_sq) = (0_f32, 0_f32);
    let (mut a, mut b) = (0, 0);
    while a < y_indices.len() && b < samples.len() {
        match y_indices[a].cmp(&samples[b]) {
            Equal => {
                let v = y_data[a];
                sum += v;
                sum_sq += v * v;
                a += 1;
                b += 1;
            }
            Less => a += 1,
            Greater => b += 1,
        }
    }
    (sum, sum_sq)
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

/// Recursively builds a regression tree node and its subtree.
///
/// Greedily selects the best split by randomly sampling feature thresholds
/// (ExtraTrees style). Partitions samples in-place and recursively builds
/// left and right children. Allocates the node before recursing to enable
/// backfilling child indices.
///
/// ### Params
///
/// * `y_indices` - Sorted array of sample indices into `y_data`
/// * `y_data` - Target variable values
/// * `x` - Sparse feature matrix in CSC format
/// * `sample_slice` - Current sample indices (sorted); partitioned in-place
/// * `y_sum` - Sum of target values for current samples
/// * `y_sum_sq` - Sum of squared target values for current samples
/// * `n_total` - Total number of samples in full dataset (for importance
///   weighting)
/// * `n_features_split` - Number of features to split.
/// * `min_samples_leaf` - Minimum number of samples per leaf.
/// * `nodes` - Vector accumulating all tree nodes; this function appends to it
/// * `feat_buf` - Scratch buffer for candidate feature indices
/// * `nz_buf` - Scratch buffer for non-zero feature values in current sample set
/// * `rng` - Random number generator for feature and threshold selection
///
/// ### Returns
///
/// Index of the node created in the `nodes` vector.
///
/// ### Algorithm
///
/// 1. If `n < 2 * min_samples_leaf` or variance negligible, create leaf
/// 2. Randomly select `k = min(n_features_split, n_features)` candidate features
/// 3. For each candidate feature:
///    - Find non-zero values in current samples
///    - Randomly sample a threshold in `[lo, max_v)`
///    - Evaluate variance reduction after split
/// 4. Apply best split (if any), partition samples in-place
/// 5. Recursively build left and right children
/// 6. Return index of split node
#[allow(clippy::too_many_arguments)]
fn build_node(
    y_indices: &[usize],
    y_data: &[f32],
    x: &CompressedSparseData<u16, f32>,
    sample_slice: &mut [usize], // sorted; partitioned in-place
    y_sum: f32,
    y_sum_sq: f32,
    n_total: usize,
    n_features_split: usize,
    min_samples_leaf: usize,
    nodes: &mut Vec<Node>,
    feat_buf: &mut Vec<usize>,
    nz_buf: &mut Vec<(usize, f32)>,
    rng: &mut SmallRng,
) -> usize {
    let n = sample_slice.len();
    let mean = y_sum / n as f32;
    let parent_var = node_variance(y_sum, y_sum_sq, n);

    if n < 2 * min_samples_leaf || parent_var < 1e-10 {
        let idx = nodes.len();
        nodes.push(Node::Leaf { mean });
        return idx;
    }

    let n_features = x.indptr.len() - 1;
    let k = n_features_split.min(n_features);

    // partial Fisher-Yates for k candidate features
    feat_buf.clear();
    feat_buf.extend(0..n_features);
    for i in 0..k {
        let j = rng.gen_range(i..n_features);
        feat_buf.swap(i, j);
    }

    let mut best_score = 0.0f32;
    let mut best_feature = usize::MAX;
    let mut best_threshold = 0.0f32;
    let mut best_y_sum_r = 0.0f32;
    let mut best_y_sum_sq_r = 0.0f32;

    for &feat in &feat_buf[..k] {
        let (fi, fv) = csc_column(x, feat);

        // collect nonzeros for this feature within current sample set
        nz_buf.clear();
        let (mut a, mut b) = (0, 0);
        while a < fi.len() && b < sample_slice.len() {
            match fi[a].cmp(&sample_slice[b]) {
                std::cmp::Ordering::Equal => {
                    nz_buf.push((b, fv[a]));
                    a += 1;
                    b += 1;
                }
                std::cmp::Ordering::Less => a += 1,
                std::cmp::Ordering::Greater => b += 1,
            }
        }

        if nz_buf.is_empty() {
            continue;
        }

        let (min_v, max_v) = nz_buf
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &(_, v)| {
                (mn.min(v), mx.max(v))
            });

        // tf there are zeros in the sample set, effective lower bound is 0.
        // this ensures zeros (which go left) can be split from nonzeros.
        let lo = if nz_buf.len() < n { 0.0f32 } else { min_v };
        if max_v - lo < 1e-10 {
            continue;
        }

        let threshold = rng.random_range(lo..max_v);

        let n_right = nz_buf.iter().filter(|&&(_, v)| v > threshold).count();
        let n_left = n - n_right;
        if n_left < min_samples_leaf || n_right < min_samples_leaf {
            continue;
        }

        // y stats for right partition: binary search into sorted y_indices
        // for each right sample. vvoids materialising a sorted right-sample
        // vec.
        let (mut y_sum_r, mut y_sum_sq_r) = (0.0f32, 0.0f32);
        for &(si, fval) in nz_buf.iter() {
            if fval > threshold {
                let cell = sample_slice[si];
                if let Ok(pos) = y_indices.binary_search(&cell) {
                    let v = y_data[pos];
                    y_sum_r += v;
                    y_sum_sq_r += v * v;
                }
            }
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

    if best_feature == usize::MAX {
        let idx = nodes.len();
        nodes.push(Node::Leaf { mean });
        return idx;
    }

    // partition sample_slice in place:
    // left (feature <= threshold) | right (feature > threshold).
    // zeros go left since threshold >= 0 for expression data - it's for single
    // cell
    let (fi, fv) = csc_column(x, best_feature);
    let mut left_buf: Vec<usize> = Vec::with_capacity(n);
    let mut right_buf: Vec<usize> = Vec::with_capacity(n);

    let mut a = 0;
    for &s in sample_slice.iter() {
        while a < fi.len() && fi[a] < s {
            a += 1;
        }
        if a < fi.len() && fi[a] == s && fv[a] > best_threshold {
            right_buf.push(s);
        } else {
            left_buf.push(s);
        }
    }

    let n_left = left_buf.len();
    sample_slice[..n_left].copy_from_slice(&left_buf);
    sample_slice[n_left..].copy_from_slice(&right_buf);
    drop(left_buf);
    drop(right_buf);

    let y_sum_l = y_sum - best_y_sum_r;
    let y_sum_sq_l = y_sum_sq - best_y_sum_sq_r;

    // allocate node slot before recursing so child indices can be filled after
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
        y_indices,
        y_data,
        x,
        left_sl,
        y_sum_l,
        y_sum_sq_l,
        n_total,
        n_features_split,
        min_samples_leaf,
        nodes,
        feat_buf,
        nz_buf,
        rng,
    );
    let right_idx = build_node(
        y_indices,
        y_data,
        x,
        right_sl,
        best_y_sum_r,
        best_y_sum_sq_r,
        n_total,
        n_features_split,
        min_samples_leaf,
        nodes,
        feat_buf,
        nz_buf,
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

/// Fit the extra tree regression
///
/// This is done sequential to have the outer loop be run in parallel
///
/// ### Params
///
/// * `target_variable` - The target variable, for SCENIC the gene expression
///   to predict from the TF levels.
/// * `feature_matrix` - The feature variables, for SCENIC the TF expression
///   levels.
/// * `n_samples` - Number of samples, cells.
/// * `config` - The ExtraTreesConfig
/// * `seed` -
pub fn fit_extra_trees(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &CompressedSparseData<u16, f32>,
    n_samples: usize,
    config: &ExtraTreesConfig,
    seed: usize,
) -> Vec<f32> {
    let n_features = feature_matrix.indptr.len() - 1;
    let n_features_split = if config.n_features_split == 0 {
        ((n_features as f64).sqrt() as usize).max(1)
    } else {
        config.n_features_split
    };

    let (y_indices, y_data) = target_variable.get_indices_data_2();

    // allocate scratch once per gene, reused across all trees
    let mut sample_indices: Vec<usize> = (0..n_samples).collect();
    let mut feat_buf: Vec<usize> = Vec::with_capacity(n_features);
    let mut nz_buf: Vec<(usize, f32)> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut importances = vec![0.0f32; n_features];

    for tree_idx in 0..config.n_trees {
        // reset sample indices without reallocating
        sample_indices
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = i);
        nodes.clear();

        let (y_sum, y_sum_sq) = y_stats_merge(y_indices, y_data, &sample_indices);
        let mut rng =
            SmallRng::seed_from_u64(seed.wrapping_add(tree_idx * 6364136223846793005) as u64);

        build_node(
            y_indices,
            y_data,
            feature_matrix,
            &mut sample_indices,
            y_sum,
            y_sum_sq,
            n_samples,
            n_features_split,
            config.min_samples_leaf,
            &mut nodes,
            &mut feat_buf,
            &mut nz_buf,
            &mut rng,
        );

        accumulate_importances(&nodes, &mut importances);
    }

    // average and normalise across trees
    let total: f32 = sum_simd_f32(&importances);
    if total > 0.0 {
        importances.iter_mut().for_each(|v| *v /= total);
    }
    importances
}

//////////
// Main //
//////////

pub fn run_scenic(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    seed: usize,
    verbose: bool,
) {
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
}
