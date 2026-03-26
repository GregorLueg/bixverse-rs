//! Binary classification via histogram-based gradient-boosted trees with
//! logistic (log) loss.

use rand::prelude::*;
use rand::rngs::SmallRng;

////////////
// Params //
////////////

/// Parameters for the logistic GBM classifier.
///
/// Defaults are aligned with R's scDblFinder XGBoost call: max_depth = 4,
/// learning_rate = 0.3, subsample = 0.75, nrounds up to 200.
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
    /// L2 regularisation on leaf weights (XGBoost's lambda).
    pub lambda: f32,
    /// Minimum sum-of-hessians in a child (XGBoost's min_child_weight).
    pub min_child_weight: f32,
    /// Fraction of eligible samples used per tree.
    pub subsample_rate: f32,
    /// Stop if OOB logloss hasn't improved for this many rounds.
    pub early_stop_rounds: usize,
}

/// Default implementation LogisticGbmConfig
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

/////////////
// Helpers //
/////////////

/// Sigmoid
///
/// ### Param
///
/// * `x` - Value for which to calculate the sigmoid
///
/// ### Returns
///
/// The sigmoid
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Logarithmic loss
///
/// ### Params
///
/// * `label` - Boolean indicating if
fn logloss(label: bool, raw: f32) -> f32 {
    let p = sigmoid(raw).clamp(1e-15, 1.0 - 1e-15);
    if label { -p.ln() } else { -(1.0 - p).ln() }
}

/////////////////////////////
// Quantised feature store //
/////////////////////////////

/// Column-major store of u8-quantised feature values.
///
/// Each feature column is independently scaled to [0, 255].
/// Layout: data[feature_idx * n_samples + sample_idx].
pub struct QuantisedStore {
    /// Quantised data
    data: Vec<u8>,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
}

impl QuantisedStore {
    /// Build from per-feature column vectors.
    ///
    /// Each inner Vec has length n_samples. Columns are independently
    /// scaled to [0, 255].
    ///
    /// ### Params
    ///
    /// * `columns` - !TODO!
    ///
    /// ### Returns
    ///
    /// Self
    pub fn from_columns(columns: &[Vec<f32>]) -> Self {
        let n_features = columns.len();
        assert!(!columns.is_empty(), "need at least one feature");
        let n_samples = columns[0].len();
        let mut data = vec![0u8; n_features * n_samples];

        for (j, col) in columns.iter().enumerate() {
            assert_eq!(col.len(), n_samples);
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &v in col {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            let range = max_v - min_v;
            let offset = j * n_samples;
            if range > 1e-10 {
                let scale = 255.0 / range;
                for (i, &v) in col.iter().enumerate() {
                    data[offset + i] = ((v - min_v) * scale).round().min(255.0) as u8;
                }
            }
        }

        Self {
            data,
            n_samples,
            n_features,
        }
    }

    /// Build from a flat row-major slice (n_samples rows, n_features cols).
    ///
    /// ### Params
    ///
    /// * `flat` - Flat representation of the data in row major
    /// * `n_samples` - Number of samples
    /// * `n_features` - Number of features
    ///
    /// ### Returns
    ///
    /// Self
    pub fn from_row_major(flat: &[f32], n_samples: usize, n_features: usize) -> Self {
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n_samples); n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                columns[j].push(flat[i * n_features + j]);
            }
        }
        Self::from_columns(&columns)
    }

    /// Return a given (quantised) feature column
    ///
    /// ### Params
    ///
    /// * `feature` - Index of the feature/column
    ///
    /// ### Returns
    ///
    ///
    #[inline(always)]
    pub fn get_col(&self, feature: usize) -> &[u8] {
        let start = feature * self.n_samples;
        &self.data[start..start + self.n_samples]
    }
}

///////////////
// Histogram //
///////////////

/// Per-feature histogram over 256 quantisation bins.
///
/// Tracks count, gradient sum, and hessian sum per bin.
struct FeatureHistogram {
    /// Count per bin
    count: [u32; 256],
    /// Gradients per bin
    grad_sum: [f32; 256],
    /// Hessian sum per bin
    hess_sum: [f32; 256],
}

impl FeatureHistogram {
    /// Initialise a new instance
    ///
    /// ### Returns
    ///
    /// Self
    fn new() -> Self {
        Self {
            count: [0; 256],
            grad_sum: [0.0; 256],
            hess_sum: [0.0; 256],
        }
    }

    fn reset(&mut self) {
        self.count = [0; 256];
        self.grad_sum = [0.0; 256];
        self.hess_sum = [0.0; 256];
    }
}

/// Histograms for all features at a single tree node.
struct NodeHistogram {
    features: Vec<FeatureHistogram>,
}

impl NodeHistogram {
    fn new(n_features: usize) -> Self {
        Self {
            features: (0..n_features).map(|_| FeatureHistogram::new()).collect(),
        }
    }

    /// Populate histograms from the given training samples.
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
    /// Gain = 0.5 * (G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam))
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

            // Scan bins 0..254; threshold at bin b means <= b goes left
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
                        n_left: nl,
                    });
                }
            }
        }

        best
    }
}

// ================================================================
// Split candidate
// ================================================================

struct SplitCandidate {
    feature: usize,
    threshold: u8,
    gain: f32,
    grad_left: f32,
    hess_left: f32,
    n_left: u32,
}

// ================================================================
// Tree storage
// ================================================================

/// A single node in a stored tree.
enum TreeNode {
    Internal {
        feature: usize,
        threshold: u8,
        left: usize,
        right: usize,
    },
    Leaf(f32),
}

/// A complete boosted tree with its learning rate baked in.
struct Tree {
    nodes: Vec<TreeNode>,
    lr: f32,
}

impl Tree {
    /// Route samples through the tree, accumulating lr * leaf_value
    /// onto raw_scores.
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

// ================================================================
// Tree building (recursive)
// ================================================================

/// Recursively build a classification tree node.
///
/// Histograms are built from scratch at each node (no subtraction
/// trick -- simpler, same asymptotic cost). A single `NodeHistogram`
/// buffer is reused across all nodes since parent histograms are not
/// needed after finding the split.
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

    // partition samples in place
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

// ================================================================
// Main entry point
// ================================================================

/// Train a logistic GBM and return predicted probabilities for ALL samples.
///
/// Excluded samples are omitted from training but still receive predictions
/// (matching R's XGBoost behaviour in scDblFinder).
///
/// Labels: true = positive (doublet), false = negative (singlet).
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

    // Degenerate case
    if n_eligible < 2 * config.min_samples_leaf {
        let n_pos = eligible.iter().filter(|&&i| labels[i as usize]).count();
        let p = (n_pos as f32 / n_eligible.max(1) as f32).clamp(0.01, 0.99);
        return vec![p; n];
    }

    // Initialise raw scores from base rate
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
        // Compute gradients and hessians for eligible samples
        for &s in &eligible {
            let si = s as usize;
            let p = sigmoid(raw_scores[si]);
            let y = if labels[si] { 1.0f32 } else { 0.0 };
            grads[si] = p - y;
            hess[si] = (p * (1.0 - p)).max(1e-8); // floor to avoid zero hessian
        }

        // Subsample eligible into train / oob
        let mut rng = SmallRng::seed_from_u64(
            seed.wrapping_add((round as u64).wrapping_mul(6_364_136_223_846_793_005)),
        );
        elig_buf.copy_from_slice(&eligible);
        for i in 0..n_train {
            let j = rng.gen_range(i..n_eligible);
            elig_buf.swap(i, j);
        }
        let (train_slice, oob_slice) = elig_buf.split_at(n_train);
        let mut train = train_slice.to_vec();

        let (g_sum, h_sum) = train.iter().fold((0.0f32, 0.0f32), |(gs, hs), &s| {
            let si = s as usize;
            (gs + grads[si], hs + hess[si])
        });

        // Build tree
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

        // Update raw scores for ALL samples
        tree.predict_update(store, &mut raw_scores, &all_samples);

        // OOB early stopping
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

    /// Simple AUC via sorting. Good enough for testing.
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
            // Trapezoidal rule when fp changes
            if fp != prev_fp {
                auc_val += (fp - prev_fp) * (tp + prev_tp) / 2.0;
                prev_fp = fp;
                prev_tp = tp;
            }
        }
        auc_val / (n_pos * n_neg)
    }

    /// Generate two Gaussian blobs in n_features dimensions.
    ///
    /// Class 0 centred at -separation/2, class 1 at +separation/2 (on
    /// the first `n_informative` features; remaining features are noise).
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
                let noise: f32 = rng.random::<f32>() * 2.0 - 1.0; // uniform [-1, 1]
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
        // Monotonicity
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

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 77);

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.70,
            "expected AUC > 0.70 for overlapping data, got {:.4}",
            auc_val
        );
    }

    #[test]
    fn test_probabilities_bounded() {
        let (cols, labels) = make_blobs(200, 2, 0, 4.0, 7);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 0);

        for &p in &probs {
            assert!(p >= 0.0 && p <= 1.0, "probability out of range: {}", p);
        }
    }

    #[test]
    fn test_imbalanced_classes() {
        let mut rng = SmallRng::seed_from_u64(555);
        let n = 2000;
        let n_pos = 100; // 5%
        let n_feat = 4;
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n); n_feat];
        let mut labels = vec![false; n];

        for i in 0..n {
            let is_pos = i < n_pos;
            labels[i] = is_pos;
            // Positives have higher values on features 0 and 1
            let offset = if is_pos { 3.0 } else { 0.0 };
            for j in 0..n_feat {
                let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
                let val = if j < 2 { offset + noise } else { noise * 3.0 };
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

        // Check calibration: mean predicted prob for positives should be
        // significantly higher than for negatives
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

    // ---- excluded samples still get predictions ----

    #[test]
    fn test_exclusion_still_predicts() {
        let (cols, labels) = make_blobs(300, 3, 0, 5.0, 88);
        let store = QuantisedStore::from_columns(&cols);
        let n = store.n_samples;

        // Exclude 20% of samples
        let mut exclude = vec![false; n];
        for i in (0..n).step_by(5) {
            exclude[i] = true;
        }

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 11);

        // Excluded samples should have non-trivial predictions (not all identical)
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

        // Excluded predictions should still be reasonable
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

    // ---- early stopping with noise ----

    #[test]
    fn test_early_stopping_pure_noise() {
        let mut rng = SmallRng::seed_from_u64(999);
        let n_train = 500;
        let n_test = 500;
        let n = n_train + n_test;
        let n_feat = 3;
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n); n_feat];
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            labels.push(rng.random_bool(0.5));
            for j in 0..n_feat {
                columns[j].push(rng.random::<f32>());
            }
        }

        let store = QuantisedStore::from_columns(&columns);

        // Exclude the test portion from training
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

        // Evaluate on the held-out test set only
        let test_probs: Vec<f32> = probs[n_train..].to_vec();
        let test_labels: Vec<bool> = labels[n_train..].to_vec();
        let auc_val = auc(&test_probs, &test_labels);
        assert!(
            auc_val < 0.65,
            "with random labels, held-out AUC should be near chance, got {:.4}",
            auc_val
        );
    }

    // ---- deterministic with same seed ----

    #[test]
    fn test_deterministic() {
        let (cols, labels) = make_blobs(200, 2, 1, 4.0, 33);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let a = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 42);
        let b = fit_logistic_gbm(&store, &labels, &exclude, &LogisticGbmConfig::default(), 42);

        assert_eq!(a, b, "same seed should produce identical results");
    }

    // ---- XOR-like pattern (tests nonlinearity) ----

    #[test]
    fn test_xor_nonlinear() {
        let mut rng = SmallRng::seed_from_u64(777);
        let n = 800;
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n); 2];
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            let x: f32 = rng.random_range(-2.0..2.0);
            let y: f32 = rng.random_range(-2.0..2.0);
            let is_pos = (x > 0.0) ^ (y > 0.0); // XOR quadrants
            columns[0].push(x);
            columns[1].push(y);
            labels.push(is_pos);
        }

        let store = QuantisedStore::from_columns(&columns);
        let exclude = vec![false; n];

        let config = LogisticGbmConfig {
            max_depth: 4, // needs depth >= 2 for XOR
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

    // ---- semi-realistic doublet scenario ----

    #[test]
    fn test_doublet_like_scenario() {
        // Mimics the scDblFinder feature space:
        // - 3 clusters of "singlets"
        // - "doublets" are mixtures with higher library size ratios,
        //   higher knn_ratios, and higher cxds-like scores
        let mut rng = SmallRng::seed_from_u64(2024);
        let n_singlets = 900;
        let n_doublets = 100; // ~10% doublet rate
        let n = n_singlets + n_doublets;

        // Features: [knn_ratio_k3, knn_ratio_k10, knn_ratio_k25,
        //            weighted, lib_ratio, nfeatures, cxds_score,
        //            pc1, pc2]
        let n_feat = 9;
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n); n_feat];
        let mut labels = Vec::with_capacity(n);

        for i in 0..n {
            let is_dbl = i >= n_singlets;
            labels.push(is_dbl);

            if is_dbl {
                // Doublets: higher knn ratios, library size, cxds
                columns[0].push(0.4 + rng.random::<f32>() * 0.4); // knn_ratio_k3
                columns[1].push(0.35 + rng.random::<f32>() * 0.3); // knn_ratio_k10
                columns[2].push(0.3 + rng.random::<f32>() * 0.3); // knn_ratio_k25
                columns[3].push(0.4 + rng.random::<f32>() * 0.3); // weighted
                columns[4].push(1.5 + rng.random::<f32>() * 1.0); // lib_ratio
                columns[5].push(800.0 + rng.random::<f32>() * 400.0); // nfeatures
                columns[6].push(0.5 + rng.random::<f32>() * 0.4); // cxds_score
                columns[7].push(rng.random::<f32>() * 4.0 - 2.0); // pc1
                columns[8].push(rng.random::<f32>() * 4.0 - 2.0); // pc2
            } else {
                // Singlets: lower knn ratios
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

        // Check that doublets get higher scores on average
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
