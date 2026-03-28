//! Binary classification via histogram-based gradient-boosted trees
//! with logistic (log) loss.
//!
//! Designed for the scDblFinder doublet detection pipeline where
//! the classifier must:
//!
//! - Return predicted probabilities for **all** samples (including
//!   those excluded from training).
//! - Store tree structure so that excluded samples can be scored
//!   after each round.
//! - Support early stopping on OOB log-loss.
//!
//! The split criterion is XGBoost-style second-order gain with L2
//! regularisation and minimum hessian weight constraints.

use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::single_cell::sc_utils::utils_tree::{QuantisedStore, train_oob_split, tree_seed};

////////////
// Params //
////////////

/// Parameters for the logistic GBM classifier.
///
/// Defaults are aligned with R's scDblFinder XGBoost call:
/// `max_depth = 4`, `learning_rate = 0.3`, `subsample = 0.75`,
/// `nrounds` up to 200.
#[derive(Clone, Debug)]
pub struct LogisticGbmConfig {
    /// Maximum number of boosting rounds.
    pub max_rounds: usize,
    /// Shrinkage per tree.
    pub learning_rate: f32,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum training samples in a leaf.
    pub min_samples_leaf: usize,
    /// L2 regularisation on leaf weights (XGBoost's `lambda`).
    pub lambda: f32,
    /// Minimum sum-of-hessians in a child (XGBoost's
    /// `min_child_weight`).
    pub min_child_weight: f32,
    /// Fraction of eligible samples used per tree.
    pub subsample_rate: f32,
    /// Stop if OOB log-loss hasn't improved for this many rounds.
    pub early_stop_rounds: usize,
}

impl Default for LogisticGbmConfig {
    fn default() -> Self {
        Self {
            max_rounds: 200,
            learning_rate: 0.3,
            max_depth: 4,
            min_samples_leaf: 20,
            lambda: 1.0,
            min_child_weight: 1.0,
            subsample_rate: 0.75,
            early_stop_rounds: 3,
        }
    }
}

//////////
// Loss //
//////////

/// Compute the sigmoid (logistic) function.
///
/// ### Params
///
/// * `x` - Raw logit value.
///
/// ### Returns
///
/// `1 / (1 + exp(-x))`, in `(0, 1)`.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the binary log-loss for a single sample.
///
/// ### Params
///
/// * `label` - Ground truth; `true` for positive class.
/// * `raw` - Raw logit score (pre-sigmoid).
///
/// ### Returns
///
/// `-ln(p)` if `label` is true, `-ln(1-p)` otherwise, where
/// `p = sigmoid(raw)` clamped to `[1e-15, 1 - 1e-15]`.
fn logloss(label: bool, raw: f32) -> f32 {
    let p = sigmoid(raw).clamp(1e-15, 1.0 - 1e-15);
    if label { -p.ln() } else { -(1.0 - p).ln() }
}

/////////////////////////
// Classification hist //
/////////////////////////

/// Per-feature histogram over 256 quantisation bins for
/// classification.
///
/// Tracks count, gradient sum, and hessian sum per bin. These are
/// the sufficient statistics for the XGBoost-style second-order
/// split gain criterion.
struct FeatureHistogram {
    /// Number of samples in each bin.
    count: [u32; 256],
    /// Sum of gradients per bin.
    grad_sum: [f32; 256],
    /// Sum of hessians per bin.
    hess_sum: [f32; 256],
}

impl FeatureHistogram {
    /// Create a zeroed histogram.
    ///
    /// ### Returns
    ///
    /// A `FeatureHistogram` with all bins at zero.
    fn new() -> Self {
        Self {
            count: [0; 256],
            grad_sum: [0.0; 256],
            hess_sum: [0.0; 256],
        }
    }

    /// Reset all bins to zero.
    fn reset(&mut self) {
        self.count = [0; 256];
        self.grad_sum = [0.0; 256];
        self.hess_sum = [0.0; 256];
    }
}

/// Histograms for all features at a single tree node.
///
/// Wraps one `FeatureHistogram` per feature. Reused across nodes
/// within a single tree since parent histograms are not needed
/// after finding the split (no subtraction trick -- the
/// classification trees are shallow enough that rebuilding from
/// scratch at each node is cheap).
struct NodeHistogram {
    /// One histogram per feature.
    features: Vec<FeatureHistogram>,
}

impl NodeHistogram {
    /// Allocate histograms for the given number of features.
    ///
    /// ### Params
    ///
    /// * `n_features` - Number of features.
    ///
    /// ### Returns
    ///
    /// A `NodeHistogram` with all bins zeroed.
    fn new(n_features: usize) -> Self {
        Self {
            features: (0..n_features).map(|_| FeatureHistogram::new()).collect(),
        }
    }

    /// Populate histograms from the given training samples.
    ///
    /// Iterates sample-major: for each sample, its gradient and
    /// hessian are accumulated into all feature histograms at once.
    /// This trades cache locality on the histogram side for a single
    /// pass over the sample set.
    ///
    /// ### Params
    ///
    /// * `store` - Quantised feature store.
    /// * `samples` - Indices of active training samples.
    /// * `grads` - Dense gradient array indexed by sample id.
    /// * `hess` - Dense hessian array indexed by sample id.
    fn build(&mut self, store: &QuantisedStore, samples: &[u32], grads: &[f32], hess: &[f32]) {
        for fh in self.features.iter_mut() {
            fh.reset();
        }
        for &s in samples {
            let si = s as usize;
            let g = grads[si];
            let h = hess[si];
            for (f, fh) in self.features.iter_mut().enumerate() {
                let bin = store.get_col(f)[si] as usize;
                fh.count[bin] += 1;
                fh.grad_sum[bin] += g;
                fh.hess_sum[bin] += h;
            }
        }
    }

    /// Scan all features for the split with highest gain.
    ///
    /// Uses the XGBoost second-order gain formula:
    ///
    /// ```text
    /// gain = 0.5 * (G_L^2/(H_L+lam) + G_R^2/(H_R+lam)
    ///              - G^2/(H+lam))
    /// ```
    ///
    /// Splits are rejected if either child has fewer than
    /// `min_samples_leaf` samples or less than `min_child_weight`
    /// hessian mass.
    ///
    /// ### Params
    ///
    /// * `g_total` - Sum of gradients in this node.
    /// * `h_total` - Sum of hessians in this node.
    /// * `n_total` - Number of training samples in this node.
    /// * `config` - Classifier configuration.
    ///
    /// ### Returns
    ///
    /// `Some(SplitCandidate)` if an improving split was found,
    /// `None` otherwise.
    fn find_best_split(
        &self,
        g_total: f32,
        h_total: f32,
        n_total: u32,
        config: &LogisticGbmConfig,
    ) -> Option<SplitCandidate> {
        let base = g_total * g_total / (h_total + config.lambda);
        let min_leaf = config.min_samples_leaf as u32;
        let min_hw = config.min_child_weight;
        let lam = config.lambda;
        let mut best: Option<SplitCandidate> = None;

        for (f, fh) in self.features.iter().enumerate() {
            let mut gl = 0.0f32;
            let mut hl = 0.0f32;
            let mut nl = 0u32;

            for b in 0..255usize {
                gl += fh.grad_sum[b];
                hl += fh.hess_sum[b];
                nl += fh.count[b];

                let nr = n_total - nl;
                if nl < min_leaf || nr < min_leaf {
                    continue;
                }
                if hl < min_hw || (h_total - hl) < min_hw {
                    continue;
                }

                let gr = g_total - gl;
                let hr = h_total - hl;
                let gain = 0.5 * (gl * gl / (hl + lam) + gr * gr / (hr + lam) - base);

                if gain > best.as_ref().map_or(1e-7, |s| s.gain) {
                    best = Some(SplitCandidate {
                        feature: f,
                        threshold: b as u8,
                        gain,
                        grad_left: gl,
                        hess_left: hl,
                    });
                }
            }
        }

        best
    }
}

/////////////////////
// Split candidate //
/////////////////////

/// Information about a candidate split in a classification tree.
struct SplitCandidate {
    /// Feature index.
    feature: usize,
    /// Bin threshold; samples with `bin <= threshold` go left.
    threshold: u8,
    /// XGBoost-style second-order gain.
    gain: f32,
    /// Sum of gradients in the left child.
    grad_left: f32,
    /// Sum of hessians in the left child.
    hess_left: f32,
}

//////////////////
// Tree storage //
//////////////////

/// A single node in a stored classification tree.
///
/// Unlike the SCENIC regression trees (which discard structure and
/// only accumulate importance), classification trees must be stored
/// so that excluded / OOB samples can be scored after each round.
enum TreeNode {
    /// Internal split node.
    Internal {
        /// Feature index used for the split.
        feature: usize,
        /// Bin threshold; `<= threshold` routes left.
        threshold: u8,
        /// Index of the left child in the `Tree.nodes` array.
        left: usize,
        /// Index of the right child in the `Tree.nodes` array.
        right: usize,
    },
    /// Leaf node containing a raw prediction value (Newton step).
    Leaf(f32),
}

/// A complete boosted classification tree with its learning rate
/// baked in.
///
/// Trees are stored as flat `Vec<TreeNode>` arrays. The root is
/// always at index 0.
struct Tree {
    /// Flat array of tree nodes.
    nodes: Vec<TreeNode>,
    /// Learning rate applied when accumulating predictions.
    lr: f32,
}

impl Tree {
    /// Route samples through the tree, accumulating `lr * leaf_value`
    /// onto `raw_scores`.
    ///
    /// ### Params
    ///
    /// * `store` - Quantised feature store.
    /// * `raw_scores` - Dense raw logit array; updated in place.
    /// * `samples` - Sample indices to route through the tree.
    fn predict_update(&self, store: &QuantisedStore, raw_scores: &mut [f32], samples: &[u32]) {
        for &s in samples {
            let si = s as usize;
            let mut idx = 0;
            loop {
                match &self.nodes[idx] {
                    TreeNode::Leaf(v) => {
                        raw_scores[si] += self.lr * v;
                        break;
                    }
                    TreeNode::Internal {
                        feature,
                        threshold,
                        left,
                        right,
                    } => {
                        idx = if store.get_col(*feature)[si] <= *threshold {
                            *left
                        } else {
                            *right
                        };
                    }
                }
            }
        }
    }
}

///////////////////
// Tree building //
///////////////////

/// Recursively build a classification tree node.
///
/// Histograms are rebuilt from scratch at each node (no subtraction
/// trick -- simpler, and the shallow depth makes the asymptotic
/// cost identical in practice). A single `NodeHistogram` buffer is
/// reused across all nodes since parent histograms are not needed
/// after finding the split.
///
/// Samples are partitioned in place via swap-based partitioning
/// (no external buffer needed).
///
/// ### Params
///
/// * `nodes` - Flat node array being built; new nodes are pushed
///   onto the end.
/// * `store` - Quantised feature store.
/// * `grads` - Dense gradient array indexed by sample id.
/// * `hess` - Dense hessian array indexed by sample id.
/// * `samples` - Active sample indices; partitioned in place.
/// * `g_sum` - Sum of gradients in this node.
/// * `h_sum` - Sum of hessians in this node.
/// * `config` - Classifier configuration.
/// * `depth` - Current tree depth.
/// * `hist` - Reusable histogram buffer.
///
/// ### Returns
///
/// Index of the newly created node in `nodes`.
#[allow(clippy::too_many_arguments)]
fn build_node(
    nodes: &mut Vec<TreeNode>,
    store: &QuantisedStore,
    grads: &[f32],
    hess: &[f32],
    samples: &mut [u32],
    g_sum: f32,
    h_sum: f32,
    config: &LogisticGbmConfig,
    depth: usize,
    hist: &mut NodeHistogram,
) -> usize {
    let my_idx = nodes.len();
    let n = samples.len() as u32;

    // stopping: depth, sample count, or hessian mass
    if depth >= config.max_depth
        || (n as usize) < 2 * config.min_samples_leaf
        || h_sum < config.min_child_weight
    {
        nodes.push(TreeNode::Leaf(-g_sum / (h_sum + config.lambda)));
        return my_idx;
    }

    // build histogram, find split
    hist.build(store, samples, grads, hess);
    let split = match hist.find_best_split(g_sum, h_sum, n, config) {
        Some(s) => s,
        None => {
            nodes.push(TreeNode::Leaf(-g_sum / (h_sum + config.lambda)));
            return my_idx;
        }
    };

    // reserve slot (filled after recursion)
    nodes.push(TreeNode::Leaf(0.0));

    // partition samples in place via swaps
    let col = store.get_col(split.feature);
    let mut left_end = 0usize;
    for i in 0..samples.len() {
        if col[samples[i] as usize] <= split.threshold {
            samples.swap(i, left_end);
            left_end += 1;
        }
    }
    let (left, right) = samples.split_at_mut(left_end);
    let g_right = g_sum - split.grad_left;
    let h_right = h_sum - split.hess_left;

    let left_idx = build_node(
        nodes,
        store,
        grads,
        hess,
        left,
        split.grad_left,
        split.hess_left,
        config,
        depth + 1,
        hist,
    );
    let right_idx = build_node(
        nodes,
        store,
        grads,
        hess,
        right,
        g_right,
        h_right,
        config,
        depth + 1,
        hist,
    );

    nodes[my_idx] = TreeNode::Internal {
        feature: split.feature,
        threshold: split.threshold,
        left: left_idx,
        right: right_idx,
    };
    my_idx
}

//////////
// Main //
//////////

/// Train a logistic GBM and return predicted probabilities for
/// **all** samples.
///
/// Excluded samples are omitted from training but still receive
/// predictions (matching R's XGBoost behaviour in scDblFinder).
/// After each tree is built, it is applied to all samples so that
/// excluded cells accumulate predictions across rounds.
///
/// Early stopping monitors OOB log-loss: if no improvement is seen
/// for `config.early_stop_rounds` consecutive rounds, boosting
/// terminates.
///
/// ### Params
///
/// * `store` - Quantised feature store (all samples, all features).
/// * `labels` - Ground truth labels; `true` = positive (doublet),
///   `false` = negative (singlet).
/// * `exclude` - Per-sample exclusion mask; `true` means the sample
///   is excluded from training and OOB evaluation but still
///   receives predictions.
/// * `config` - Classifier configuration.
/// * `seed` - Base seed for reproducibility.
///
/// ### Returns
///
/// Vector of length `n_samples` with predicted probabilities in
/// `[0, 1]`.
pub fn fit_logistic_gbm(
    store: &QuantisedStore,
    labels: &[bool],
    exclude: &[bool],
    config: &LogisticGbmConfig,
    seed: u64,
) -> Vec<f32> {
    let n = store.n_samples;
    assert_eq!(labels.len(), n);
    assert_eq!(exclude.len(), n);

    let eligible: Vec<u32> = (0..n as u32).filter(|&i| !exclude[i as usize]).collect();
    let n_eligible = eligible.len();

    // degenerate case: too few samples to split
    if n_eligible < 2 * config.min_samples_leaf {
        let n_pos = eligible.iter().filter(|&&i| labels[i as usize]).count();
        let p = (n_pos as f32 / n_eligible.max(1) as f32).clamp(0.01, 0.99);
        return vec![p; n];
    }

    // initialise raw scores from base rate
    let n_pos = eligible.iter().filter(|&&i| labels[i as usize]).count();
    let base_rate = (n_pos as f32 / n_eligible as f32).clamp(0.01, 0.99);
    let init_logit = (base_rate / (1.0 - base_rate)).ln();
    let mut raw_scores = vec![init_logit; n];

    let mut grads = vec![0.0f32; n];
    let mut hess = vec![0.0f32; n];
    let mut hist = NodeHistogram::new(store.n_features);

    let n_train = ((n_eligible as f32 * config.subsample_rate).round() as usize)
        .max(2 * config.min_samples_leaf)
        .min(n_eligible);

    let mut elig_buf = eligible.clone();
    let all_samples: Vec<u32> = (0..n as u32).collect();

    let mut best_oob_loss = f32::INFINITY;
    let mut rounds_no_improve = 0usize;

    for round in 0..config.max_rounds {
        // compute gradients and hessians for eligible samples
        for &s in &eligible {
            let si = s as usize;
            let p = sigmoid(raw_scores[si]);
            let y = if labels[si] { 1.0f32 } else { 0.0 };
            grads[si] = p - y;
            hess[si] = (p * (1.0 - p)).max(1e-8);
        }

        // subsample eligible into train / OOB
        let mut rng = SmallRng::seed_from_u64(tree_seed(seed as usize, round));
        elig_buf.copy_from_slice(&eligible);
        let actual_n_train = train_oob_split(&mut elig_buf, n_train, &mut rng);

        let (train_slice, oob_slice) = elig_buf.split_at(actual_n_train);
        let mut train = train_slice.to_vec();

        let (g_sum, h_sum) = train.iter().fold((0.0f32, 0.0f32), |(gs, hs), &s| {
            let si = s as usize;
            (gs + grads[si], hs + hess[si])
        });

        // build tree
        let mut tree_nodes = Vec::with_capacity(2usize.pow(config.max_depth as u32 + 1));
        build_node(
            &mut tree_nodes,
            store,
            &grads,
            &hess,
            &mut train,
            g_sum,
            h_sum,
            config,
            0,
            &mut hist,
        );

        let tree = Tree {
            nodes: tree_nodes,
            lr: config.learning_rate,
        };

        // update raw scores for ALL samples (including excluded)
        tree.predict_update(store, &mut raw_scores, &all_samples);

        // OOB early stopping on log-loss
        if !oob_slice.is_empty() {
            let oob_loss: f32 = oob_slice
                .iter()
                .map(|&s| logloss(labels[s as usize], raw_scores[s as usize]))
                .sum::<f32>()
                / oob_slice.len() as f32;

            if oob_loss < best_oob_loss - 1e-6 {
                best_oob_loss = oob_loss;
                rounds_no_improve = 0;
            } else {
                rounds_no_improve += 1;
            }
            if rounds_no_improve >= config.early_stop_rounds {
                break;
            }
        }
    }

    // Return probabilities
    raw_scores.iter().map(|&s| sigmoid(s)).collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute AUC via sorting (trapezoidal rule).
    ///
    /// ### Params
    ///
    /// * `scores` - Predicted probabilities or scores.
    /// * `labels` - Ground truth labels.
    ///
    /// ### Returns
    ///
    /// Area under the ROC curve.
    fn auc(scores: &[f32], labels: &[bool]) -> f32 {
        let mut pairs: Vec<(f32, bool)> =
            scores.iter().copied().zip(labels.iter().copied()).collect();
        pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let n_pos = labels.iter().filter(|&&l| l).count() as f32;
        let n_neg = labels.iter().filter(|&&l| !l).count() as f32;
        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5;
        }

        let mut tp = 0.0f32;
        let mut fp = 0.0f32;
        let mut auc_val = 0.0f32;
        let mut prev_fp = 0.0f32;
        let mut prev_tp = 0.0f32;

        for (_, label) in &pairs {
            if *label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            if fp != prev_fp {
                auc_val += (fp - prev_fp) * (tp + prev_tp) / 2.0;
                prev_fp = fp;
                prev_tp = tp;
            }
        }
        auc_val / (n_pos * n_neg)
    }

    /// Generate two Gaussian blobs in `n_features` dimensions.
    ///
    /// Class 0 is centred at `-separation/2`, class 1 at
    /// `+separation/2` on the first `n_informative` features;
    /// remaining features are pure noise.
    ///
    /// ### Params
    ///
    /// * `n_per_class` - Samples per class.
    /// * `n_informative` - Number of informative features.
    /// * `n_noise` - Number of noise features.
    /// * `separation` - Distance between class centres.
    /// * `seed` - Random seed.
    ///
    /// ### Returns
    ///
    /// `(columns, labels)` where `columns` is one `Vec<f32>` per
    /// feature and `labels` is a `Vec<bool>`.
    fn make_blobs(
        n_per_class: usize,
        n_informative: usize,
        n_noise: usize,
        separation: f32,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<bool>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let n = 2 * n_per_class;
        let n_feat = n_informative + n_noise;
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n); n_feat];
        let mut labels = Vec::with_capacity(n);

        for i in 0..n {
            let is_pos = i >= n_per_class;
            labels.push(is_pos);
            let offset = if is_pos {
                separation / 2.0
            } else {
                -separation / 2.0
            };
            for j in 0..n_feat {
                let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
                let val = if j < n_informative {
                    offset + noise
                } else {
                    noise
                };
                columns[j].push(val);
            }
        }

        (columns, labels)
    }

    #[test]
    fn test_sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
        assert!(sigmoid(1.0) > sigmoid(0.0));
        assert!(sigmoid(0.0) > sigmoid(-1.0));
    }

    #[test]
    fn test_separable_blobs() {
        let (cols, labels) = make_blobs(500, 3, 2, 6.0, 42);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 99);

        assert_eq!(probs.len(), store.n_samples);
        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.95,
            "expected AUC > 0.95 for well-separated data, got {:.4}",
            auc_val
        );
    }

    #[test]
    fn test_overlapping_blobs() {
        let (cols, labels) = make_blobs(500, 3, 5, 2.0, 123);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 7);

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.70,
            "expected AUC > 0.70 for overlapping data, got {:.4}",
            auc_val
        );
    }

    #[test]
    fn test_imbalanced() {
        let mut rng = SmallRng::seed_from_u64(55);
        let n_pos = 50;
        let n = 500;
        let n_feat = 4;
        let mut columns: Vec<Vec<f32>> = (0..n_feat)
            .map(|_| Vec::with_capacity(n))
            .collect::<Vec<_>>();
        let mut labels = Vec::with_capacity(n);

        for i in 0..n {
            let is_pos = i < n_pos;
            labels.push(is_pos);
            for j in 0..n_feat {
                let base: f32 = if is_pos && j < 2 { 3.0 } else { 0.0 };
                let val = base + rng.random::<f32>() * 2.0 - 1.0;
                columns[j].push(val);
            }
        }

        let store = QuantisedStore::from_columns(&columns);
        let exclude = vec![false; n];

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 42);

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.90,
            "expected AUC > 0.90 for imbalanced data, got {:.4}",
            auc_val
        );

        let mean_pos: f32 = probs
            .iter()
            .zip(&labels)
            .filter(|&(_, &l)| l)
            .map(|(&p, _)| p)
            .sum::<f32>()
            / n_pos as f32;
        let mean_neg: f32 = probs
            .iter()
            .zip(&labels)
            .filter(|&(_, &l)| !l)
            .map(|(&p, _)| p)
            .sum::<f32>()
            / (n - n_pos) as f32;
        assert!(
            mean_pos > mean_neg * 2.0,
            "positive mean ({:.4}) should be much higher than negative mean ({:.4})",
            mean_pos,
            mean_neg
        );
    }

    #[test]
    fn test_exclusion_still_predicts() {
        let (cols, labels) = make_blobs(300, 3, 0, 5.0, 88);
        let store = QuantisedStore::from_columns(&cols);
        let n = store.n_samples;

        let mut exclude = vec![false; n];
        for i in (0..n).step_by(5) {
            exclude[i] = true;
        }

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 11);

        let excluded_probs: Vec<f32> = probs
            .iter()
            .zip(&exclude)
            .filter(|&(_, &e)| e)
            .map(|(&p, _)| p)
            .collect();
        let min_ex = excluded_probs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_ex = excluded_probs
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_ex - min_ex > 0.1,
            "excluded predictions should vary; range was {:.4}",
            max_ex - min_ex
        );

        let excluded_labels: Vec<bool> = labels
            .iter()
            .zip(&exclude)
            .filter(|&(_, &e)| e)
            .map(|(&l, _)| l)
            .collect();
        let excl_auc = auc(&excluded_probs, &excluded_labels);
        assert!(
            excl_auc > 0.80,
            "excluded sample AUC should be decent, got {:.4}",
            excl_auc
        );
    }

    #[test]
    fn test_early_stopping_pure_noise() {
        let mut rng = SmallRng::seed_from_u64(999);
        let n_train = 500;
        let n_test = 500;
        let n = n_train + n_test;
        let n_feat = 3;
        let mut columns: Vec<Vec<f32>> = (0..n_feat)
            .map(|_| Vec::with_capacity(n))
            .collect::<Vec<_>>();
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            labels.push(rng.random_bool(0.5));
            for j in 0..n_feat {
                columns[j].push(rng.random::<f32>());
            }
        }

        let store = QuantisedStore::from_columns(&columns);

        let mut exclude = vec![false; n];
        for i in n_train..n {
            exclude[i] = true;
        }

        let config = LogisticGbmConfig {
            max_rounds: 500,
            early_stop_rounds: 5,
            ..Default::default()
        };

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &config, 0);

        let test_probs: Vec<f32> = probs[n_train..].to_vec();
        let test_labels: Vec<bool> = labels[n_train..].to_vec();
        let auc_val = auc(&test_probs, &test_labels);
        assert!(
            auc_val < 0.65,
            "with random labels, held-out AUC should be near chance, got {:.4}",
            auc_val
        );
    }

    #[test]
    fn test_deterministic() {
        let (cols, labels) = make_blobs(200, 2, 1, 4.0, 33);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let a = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 42);
        let b = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 42);

        assert_eq!(a, b, "same seed should produce identical results");
    }

    #[test]
    fn test_xor_nonlinear() {
        let mut rng = SmallRng::seed_from_u64(777);
        let n = 800;
        let mut columns: Vec<Vec<f32>> = (0..2).map(|_| Vec::with_capacity(n)).collect::<Vec<_>>();
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            let x: f32 = rng.random_range(-2.0..2.0);
            let y: f32 = rng.random_range(-2.0..2.0);
            let is_pos = (x > 0.0) ^ (y > 0.0);
            columns[0].push(x);
            columns[1].push(y);
            labels.push(is_pos);
        }

        let store = QuantisedStore::from_columns(&columns);
        let exclude = vec![false; n];

        let config = LogisticGbmConfig {
            max_depth: 4,
            max_rounds: 100,
            ..Default::default()
        };

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &config, 13);

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.85,
            "tree ensemble should handle XOR, got AUC {:.4}",
            auc_val
        );
    }

    #[test]
    fn test_doublet_like_scenario() {
        let mut rng = SmallRng::seed_from_u64(2024);
        let n_singlets = 900;
        let n_doublets = 100;
        let n = n_singlets + n_doublets;

        let n_feat = 9;
        let mut columns: Vec<Vec<f32>> = (0..n_feat)
            .map(|_| Vec::with_capacity(n))
            .collect::<Vec<_>>();
        let mut labels = Vec::with_capacity(n);

        for i in 0..n {
            let is_dbl = i >= n_singlets;
            labels.push(is_dbl);

            if is_dbl {
                columns[0].push(0.4 + rng.random::<f32>() * 0.4);
                columns[1].push(0.35 + rng.random::<f32>() * 0.3);
                columns[2].push(0.3 + rng.random::<f32>() * 0.3);
                columns[3].push(0.4 + rng.random::<f32>() * 0.3);
                columns[4].push(1.5 + rng.random::<f32>() * 1.0);
                columns[5].push(800.0 + rng.random::<f32>() * 400.0);
                columns[6].push(0.5 + rng.random::<f32>() * 0.4);
                columns[7].push(rng.random::<f32>() * 4.0 - 2.0);
                columns[8].push(rng.random::<f32>() * 4.0 - 2.0);
            } else {
                columns[0].push(rng.random::<f32>() * 0.3);
                columns[1].push(rng.random::<f32>() * 0.25);
                columns[2].push(rng.random::<f32>() * 0.2);
                columns[3].push(rng.random::<f32>() * 0.3);
                columns[4].push(0.8 + rng.random::<f32>() * 0.4);
                columns[5].push(400.0 + rng.random::<f32>() * 400.0);
                columns[6].push(rng.random::<f32>() * 0.4);
                columns[7].push(rng.random::<f32>() * 4.0 - 2.0);
                columns[8].push(rng.random::<f32>() * 4.0 - 2.0);
            }
        }

        let store = QuantisedStore::from_columns(&columns);
        let exclude = vec![false; n];

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 42);

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.92,
            "doublet-like scenario should achieve high AUC, got {:.4}",
            auc_val
        );

        let mean_dbl: f32 = probs[n_singlets..].iter().sum::<f32>() / n_doublets as f32;
        let mean_sng: f32 = probs[..n_singlets].iter().sum::<f32>() / n_singlets as f32;
        assert!(
            mean_dbl > mean_sng * 3.0,
            "doublet mean ({:.4}) should be much higher than singlet mean ({:.4})",
            mean_dbl,
            mean_sng
        );
    }
}
