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

use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

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
    /// Minimum sum-of-hessians in a child (XGBoost's `min_child_weight`).
    pub min_child_weight: f32,
    /// Fraction of eligible samples used per tree.
    pub subsample_rate: f32,
    /// Number of CV folds for round selection.
    pub n_folds: usize,
    /// Early stopping patience per CV fold.
    pub cv_early_stop: usize,
    /// Multiplier on std for the SE rule (0.25 matches R's
    /// `nrounds=0.25`).
    pub se_fraction: f32,
}

impl Default for LogisticGbmConfig {
    fn default() -> Self {
        Self {
            max_rounds: 200,
            learning_rate: 0.3,
            max_depth: 4,
            min_samples_leaf: 10,
            lambda: 1.0,
            min_child_weight: 1.0,
            subsample_rate: 0.75,
            n_folds: 5,
            cv_early_stop: 2,
            se_fraction: 1.0,
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

/////////////////////////
// Single boosting run //
/////////////////////////

/// Run boosting on a fixed train set, recording per-round validation
/// loss on a fixed val set. Returns the per-round val losses.
///
/// `raw_scores` is modified in place. Only `train_samples` are used
/// for gradient computation; `score_samples` are scored after each
/// tree (for the final run this is all samples; for CV folds this
/// is train+val only).
///
/// ### Params
///
/// * `store` - Quantised feature store.
/// * `labels` - Ground truth labels; `true` = positive (doublet).
/// * `train_pool` - Sample indices eligible for training; subsampled
///   each round.
/// * `val_set` - Sample indices used to compute validation loss.
///   Pass an empty slice to skip validation.
/// * `score_samples` - Sample indices that receive score updates
///   after each tree.
/// * `raw_scores` - Dense raw logit array; updated in place.
/// * `config` - Classifier configuration.
/// * `max_rounds` - Maximum number of boosting rounds to run.
/// * `early_stop_patience` - Stop after this many rounds without
///   improvement. Pass `0` to disable early stopping.
/// * `seed` - Base seed for per-round RNG.
///
/// ### Returns
///
/// Per-round validation losses; length equals the number of rounds
/// actually run (may be shorter than `max_rounds` if early stopping
/// triggers).
#[allow(clippy::too_many_arguments)]
fn boost_run(
    store: &QuantisedStore,
    labels: &[bool],
    train_pool: &[u32],
    val_set: &[u32],
    score_samples: &[u32],
    raw_scores: &mut [f32],
    config: &LogisticGbmConfig,
    max_rounds: usize,
    early_stop_patience: usize,
    seed: u64,
) -> Vec<f32> {
    let n = raw_scores.len();
    let mut grads = vec![0.0f32; n];
    let mut hess = vec![0.0f32; n];
    let mut hist = NodeHistogram::new(store.n_features);

    let n_subsample = ((train_pool.len() as f32 * config.subsample_rate).round() as usize)
        .max(2 * config.min_samples_leaf)
        .min(train_pool.len());

    let mut val_losses: Vec<f32> = Vec::with_capacity(max_rounds);
    let mut best_val_loss = f32::INFINITY;
    let mut rounds_no_improve = 0usize;

    for round in 0..max_rounds {
        for &s in train_pool {
            let si = s as usize;
            let p = sigmoid(raw_scores[si]);
            let y = if labels[si] { 1.0f32 } else { 0.0 };
            grads[si] = p - y;
            hess[si] = (p * (1.0 - p)).max(1e-8);
        }

        let mut rng = SmallRng::seed_from_u64(tree_seed(seed as usize, round));
        let mut pool_buf = train_pool.to_vec();
        let actual_n = train_oob_split(&mut pool_buf, n_subsample, &mut rng);
        let mut train_slice = pool_buf[..actual_n].to_vec();

        let (g_sum, h_sum) = train_slice.iter().fold((0.0f32, 0.0f32), |(gs, hs), &s| {
            let si = s as usize;
            (gs + grads[si], hs + hess[si])
        });

        let mut tree_nodes = Vec::with_capacity(2usize.pow(config.max_depth as u32 + 1));
        build_node(
            &mut tree_nodes,
            store,
            &grads,
            &hess,
            &mut train_slice,
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

        tree.predict_update(store, raw_scores, score_samples);

        let val_loss: f32 = if val_set.is_empty() {
            0.0
        } else {
            val_set
                .iter()
                .map(|&s| logloss(labels[s as usize], raw_scores[s as usize]))
                .sum::<f32>()
                / val_set.len() as f32
        };
        val_losses.push(val_loss);

        if val_loss < best_val_loss - 1e-6 {
            best_val_loss = val_loss;
            rounds_no_improve = 0;
        } else {
            rounds_no_improve += 1;
        }

        if early_stop_patience > 0 && rounds_no_improve >= early_stop_patience {
            break;
        }
    }

    val_losses
}

//////////////////////////////////
// Stratified k-fold generation //
//////////////////////////////////

/// Create stratified k-fold indices from positive and negative
/// sample vectors.
///
/// ### Params
///
/// * `pos` - Sample indices belonging to the positive class.
/// * `neg` - Sample indices belonging to the negative class.
/// * `k` - Number of folds.
/// * `rng` - Random number generator used to shuffle before
///   assignment.
///
/// ### Returns
///
/// `(train_sets, val_sets)` each of length `k`.
fn stratified_kfold(
    pos: &[u32],
    neg: &[u32],
    k: usize,
    rng: &mut SmallRng,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut pos_shuffled = pos.to_vec();
    let mut neg_shuffled = neg.to_vec();
    pos_shuffled.shuffle(rng);
    neg_shuffled.shuffle(rng);

    let mut fold_assignments_pos = vec![0usize; pos_shuffled.len()];
    let mut fold_assignments_neg = vec![0usize; neg_shuffled.len()];
    for (i, fa) in fold_assignments_pos.iter_mut().enumerate() {
        *fa = i % k;
    }
    for (i, fa) in fold_assignments_neg.iter_mut().enumerate() {
        *fa = i % k;
    }

    let mut train_sets = Vec::with_capacity(k);
    let mut val_sets = Vec::with_capacity(k);

    for fold in 0..k {
        let mut train = Vec::new();
        let mut val = Vec::new();

        for (i, &s) in pos_shuffled.iter().enumerate() {
            if fold_assignments_pos[i] == fold {
                val.push(s);
            } else {
                train.push(s);
            }
        }
        for (i, &s) in neg_shuffled.iter().enumerate() {
            if fold_assignments_neg[i] == fold {
                val.push(s);
            } else {
                train.push(s);
            }
        }

        train_sets.push(train);
        val_sets.push(val);
    }

    (train_sets, val_sets)
}

////////////////////////
// CV round selection //
////////////////////////

/// Run k-fold CV and return the selected number of rounds using
/// the SE rule: pick the earliest round whose mean loss is within
/// `se_fraction * std` of the best round's mean loss.
///
/// ### Params
///
/// * `store` - Quantised feature store.
/// * `labels` - Ground truth labels; `true` = positive (doublet).
/// * `eligible` - Sample indices available for training (i.e.
///   non-excluded).
/// * `config` - Classifier configuration.
/// * `init_logit` - Initial raw score for all samples; typically
///   the log-odds of the base rate.
/// * `seed` - Base seed; fold seeds are derived from this.
/// * `verbose` - Additional verbosity for this function. For debugging
///   purposes.
///
/// ### Returns
///
/// Number of boosting rounds to use for the final model (at least `1`).
fn cv_select_rounds(
    store: &QuantisedStore,
    labels: &[bool],
    eligible: &[u32],
    config: &LogisticGbmConfig,
    init_logit: f32,
    seed: u64,
    verbose: bool,
) -> usize {
    let pos: Vec<u32> = eligible
        .iter()
        .copied()
        .filter(|&i| labels[i as usize])
        .collect();
    let neg: Vec<u32> = eligible
        .iter()
        .copied()
        .filter(|&i| !labels[i as usize])
        .collect();

    let k = config.n_folds.min(pos.len()).min(neg.len()).max(2);
    let mut rng = SmallRng::seed_from_u64(seed);
    let (train_sets, val_sets) = stratified_kfold(&pos, &neg, k, &mut rng);

    let n = store.n_samples;

    // per-fold state
    let mut fold_raw_scores: Vec<Vec<f32>> = (0..k).map(|_| vec![init_logit; n]).collect();
    let mut fold_grads: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; n]).collect();
    let mut fold_hess: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; n]).collect();
    let mut fold_hists: Vec<NodeHistogram> = (0..k)
        .map(|_| NodeHistogram::new(store.n_features))
        .collect();

    // aggregated loss tracking
    let mut mean_losses: Vec<f32> = Vec::with_capacity(config.max_rounds);
    let mut std_losses: Vec<f32> = Vec::with_capacity(config.max_rounds);
    let mut best_mean_loss = f32::INFINITY;
    let mut rounds_no_improve = 0usize;

    let n_subsample_per_fold: Vec<usize> = train_sets
        .iter()
        .map(|t| {
            ((t.len() as f32 * config.subsample_rate).round() as usize)
                .max(2 * config.min_samples_leaf)
                .min(t.len())
        })
        .collect();

    for round in 0..config.max_rounds {
        let fold_losses: Vec<f32> = fold_raw_scores
            .par_iter_mut()
            .zip(fold_grads.par_iter_mut())
            .zip(fold_hess.par_iter_mut())
            .zip(fold_hists.par_iter_mut())
            .zip(train_sets.par_iter())
            .zip(val_sets.par_iter())
            .zip(n_subsample_per_fold.par_iter())
            .enumerate()
            .map(
                |(fold, ((((((raw, grads), hess), hist), train), val), &n_sub))| {
                    for &s in train.iter() {
                        let si = s as usize;
                        let p = sigmoid(raw[si]);
                        let y = if labels[si] { 1.0f32 } else { 0.0 };
                        grads[si] = p - y;
                        hess[si] = (p * (1.0 - p)).max(1e-8);
                    }

                    let fold_seed = seed.wrapping_add(fold as u64 * 7_919);
                    let mut sub_rng = SmallRng::seed_from_u64(tree_seed(fold_seed as usize, round));
                    let mut pool_buf = train.to_vec();
                    let actual_n = train_oob_split(&mut pool_buf, n_sub, &mut sub_rng);
                    let mut train_slice = pool_buf[..actual_n].to_vec();

                    let (g_sum, h_sum) =
                        train_slice.iter().fold((0.0f32, 0.0f32), |(gs, hs), &s| {
                            let si = s as usize;
                            (gs + grads[si], hs + hess[si])
                        });

                    let mut tree_nodes =
                        Vec::with_capacity(2usize.pow(config.max_depth as u32 + 1));
                    build_node(
                        &mut tree_nodes,
                        store,
                        grads,
                        hess,
                        &mut train_slice,
                        g_sum,
                        h_sum,
                        config,
                        0,
                        hist,
                    );

                    let tree = Tree {
                        nodes: tree_nodes,
                        lr: config.learning_rate,
                    };

                    let mut fold_samples: Vec<u32> = Vec::with_capacity(train.len() + val.len());
                    fold_samples.extend_from_slice(train);
                    fold_samples.extend_from_slice(val);
                    tree.predict_update(store, raw, &fold_samples);

                    val.iter()
                        .map(|&s| logloss(labels[s as usize], raw[s as usize]))
                        .sum::<f32>()
                        / val.len().max(1) as f32
                },
            )
            .collect();

        // aggregate across folds
        let mean: f32 = fold_losses.iter().sum::<f32>() / k as f32;
        let var: f32 = fold_losses.iter().map(|&l| (l - mean).powi(2)).sum::<f32>() / k as f32;
        let std = var.sqrt();

        mean_losses.push(mean);
        std_losses.push(std);

        if verbose && (round < 10 || round % 10 == 0) {
            println!(
                "  CV round {}: mean_loss={:.5}, best={:.5}, no_improve={}",
                round, mean, best_mean_loss, rounds_no_improve
            );
        }

        // early stopping on the cross-fold mean
        if mean < best_mean_loss - 5e-3 {
            best_mean_loss = mean;
            rounds_no_improve = 0;
        } else {
            rounds_no_improve += 1;
        }

        if rounds_no_improve >= config.cv_early_stop {
            break;
        }
    }

    // SE rule: best round, then earliest round within ceiling
    let best_round = mean_losses
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let ceiling = mean_losses[best_round] + config.se_fraction * std_losses[best_round];

    let selected = mean_losses
        .iter()
        .position(|&l| l <= ceiling)
        .unwrap_or(best_round);

    if verbose {
        println!(
            "CV: ran {} rounds, best_round={}, best_mean_loss={:.4}, \
                 std={:.4}, ceiling={:.4}, selected={}",
            mean_losses.len(),
            best_round,
            mean_losses[best_round],
            std_losses[best_round],
            ceiling,
            selected + 1,
        );
    }

    (selected + 1).max(1)
}

//////////
// Main //
//////////

/// Train a logistic GBM and return predicted probabilities for
/// **all** samples.
///
/// Uses k-fold cross-validation to select the number of boosting
/// rounds, applying the SE rule (matching XGBoost's `nrounds=0.25`
/// behaviour in R's scDblFinder). The final model trains on all
/// eligible non-excluded samples for exactly the selected number
/// of rounds.
///
/// Excluded samples are omitted from training but still receive
/// predictions (matching R's XGBoost behaviour in scDblFinder).
///
/// ### Params
///
/// * `store` - Quantised feature store (all samples, all features).
/// * `labels` - Ground truth labels; `true` = positive (doublet),
///   `false` = negative (singlet).
/// * `exclude` - Per-sample exclusion mask; `true` means the sample
///   is excluded from training but still receives predictions.
/// * `config` - Classifier configuration.
/// * `seed` - Base seed for reproducibility.
/// * `verbose` - Controls verbosity of the function.
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
    verbose: bool,
) -> Vec<f32> {
    let n = store.n_samples;
    assert_eq!(labels.len(), n);
    assert_eq!(exclude.len(), n);

    let eligible: Vec<u32> = (0..n as u32).filter(|&i| !exclude[i as usize]).collect();
    let n_eligible = eligible.len();

    if n_eligible < 2 * config.min_samples_leaf {
        let n_pos = eligible.iter().filter(|&&i| labels[i as usize]).count();
        let p = (n_pos as f32 / n_eligible.max(1) as f32).clamp(0.01, 0.99);
        return vec![p; n];
    }

    let n_pos = eligible.iter().filter(|&&i| labels[i as usize]).count();
    let base_rate = (n_pos as f32 / n_eligible as f32).clamp(0.01, 0.99);
    let init_logit = (base_rate / (1.0 - base_rate)).ln();

    // phase 1: lockstep CV to select n_rounds
    let n_rounds = cv_select_rounds(store, labels, &eligible, config, init_logit, seed, verbose);

    // phase 2: final training on all eligible for exactly n_rounds
    let mut raw_scores = vec![init_logit; n];
    let all_samples: Vec<u32> = (0..n as u32).collect();

    boost_run(
        store,
        labels,
        &eligible,
        &[],
        &all_samples,
        &mut raw_scores,
        config,
        n_rounds,
        0,
        seed,
    );

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

    /// Check that predicted probabilities are not pushed to
    /// extremes on genuinely ambiguous data.
    fn assert_not_bimodal(probs: &[f32], labels: &[bool]) {
        let n = probs.len() as f32;
        let n_extreme = probs
            .iter()
            .filter(|&&p| !(0.02..0.98).contains(&p))
            .count();
        let extreme_frac = n_extreme as f32 / n;
        assert!(
            extreme_frac < 0.4,
            "too many extreme predictions ({:.1}%): model is likely overfitting",
            100.0 * extreme_frac
        );
        let mean_pos: f32 = probs
            .iter()
            .zip(labels)
            .filter(|&(_, &l)| l)
            .map(|(&p, _)| p)
            .sum::<f32>()
            / labels.iter().filter(|&&l| l).count() as f32;
        let mean_neg: f32 = probs
            .iter()
            .zip(labels)
            .filter(|&(_, &l)| !l)
            .map(|(&p, _)| p)
            .sum::<f32>()
            / labels.iter().filter(|&&l| !l).count() as f32;
        assert!(
            (0.0..0.98).contains(&mean_pos),
            "positive class mean {:.4} is saturated",
            mean_pos
        );
        assert!(
            (0.02..1.0).contains(&mean_neg),
            "negative class mean {:.4} is saturated",
            mean_neg
        );
        assert!(
            mean_pos > mean_neg,
            "positive mean ({:.4}) should exceed negative mean ({:.4})",
            mean_pos,
            mean_neg,
        );
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

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            99,
            false,
        );

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
        // weak signal: separation=1.0 with lots of noise features
        // means classes heavily overlap
        let (cols, labels) = make_blobs(500, 2, 8, 1.0, 123);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            7,
            false,
        );

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.55,
            "expected AUC > 0.55 for weakly overlapping data, got {:.4}",
            auc_val
        );

        // with heavy overlap, predictions should stay moderate
        assert_not_bimodal(&probs, &labels);
    }

    #[test]
    fn test_calibration_moderate_separation() {
        // genuinely ambiguous: 1 informative feature, 6 noise, weak signal
        let (cols, labels) = make_blobs(500, 1, 9, 0.8, 77);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            55,
            false,
        );

        let auc_val = auc(&probs, &labels);
        assert!(auc_val > 0.60, "expected AUC > 0.60, got {:.4}", auc_val);

        assert_not_bimodal(&probs, &labels);
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

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            42,
            false,
        );

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

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            11,
            false,
        );

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

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            0,
            false,
        );

        let test_probs: Vec<f32> = probs[n_train..].to_vec();
        let test_labels: Vec<bool> = labels[n_train..].to_vec();
        let auc_val = auc(&test_probs, &test_labels);
        assert!(
            auc_val < 0.65,
            "with random labels, held-out AUC should be near chance, got {:.4}",
            auc_val
        );

        // predictions on pure noise should cluster around the base
        // rate, not be extreme
        let mean_pred: f32 = test_probs.iter().sum::<f32>() / test_probs.len() as f32;
        assert!(
            mean_pred > 0.3 && mean_pred < 0.7,
            "pure noise predictions should be near 0.5, got mean {:.4}",
            mean_pred
        );
    }

    #[test]
    fn test_deterministic() {
        let (cols, labels) = make_blobs(200, 2, 1, 4.0, 33);
        let store = QuantisedStore::from_columns(&cols);
        let exclude = vec![false; store.n_samples];

        let a = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            42,
            false,
        );
        let b = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            42,
            false,
        );

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

        let probs = fit_logistic_gbm(&store, &labels, &exclude, &config, 13, false);

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.85,
            "tree ensemble should handle XOR, got AUC {:.4}",
            auc_val
        );
    }

    #[test]
    fn test_cv_selects_few_rounds_on_noise() {
        // on pure noise, CV should select very few rounds
        let mut rng = SmallRng::seed_from_u64(12345);
        let n = 600;
        let n_feat = 5;
        let mut columns: Vec<Vec<f32>> = (0..n_feat).map(|_| Vec::with_capacity(n)).collect();
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            labels.push(rng.random_bool(0.5));
            for j in 0..n_feat {
                columns[j].push(rng.random::<f32>());
            }
        }

        let store = QuantisedStore::from_columns(&columns);
        let eligible: Vec<u32> = (0..n as u32).collect();
        let config = LogisticGbmConfig::default();
        let init_logit = 0.0f32; // balanced

        let n_rounds = cv_select_rounds(&store, &labels, &eligible, &config, init_logit, 42, false);

        assert!(
            n_rounds <= 10,
            "CV should select very few rounds on noise, got {}",
            n_rounds
        );
    }

    #[test]
    fn test_cv_selects_more_rounds_on_signal() {
        // on separable data, CV should allow more rounds
        let (cols, labels) = make_blobs(400, 3, 2, 4.0, 999);
        let store = QuantisedStore::from_columns(&cols);
        let eligible: Vec<u32> = (0..store.n_samples as u32).collect();
        let n_pos = labels.iter().filter(|&&l| l).count();
        let base_rate = n_pos as f32 / labels.len() as f32;
        let init_logit = (base_rate / (1.0 - base_rate)).ln();
        let config = LogisticGbmConfig::default();

        let n_rounds = cv_select_rounds(&store, &labels, &eligible, &config, init_logit, 42, false);

        assert!(
            n_rounds >= 3,
            "CV should select at least a few rounds on separable data, got {}",
            n_rounds
        );
        assert!(
            n_rounds <= 80,
            "CV should not select too many rounds even on separable data, got {}",
            n_rounds
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

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            42,
            false,
        );

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.92,
            "doublet-like scenario should achieve high AUC, got {:.4}",
            auc_val
        );

        let mean_dbl: f32 = probs[n_singlets..].iter().sum::<f32>() / n_doublets as f32;
        let mean_sng: f32 = probs[..n_singlets].iter().sum::<f32>() / n_singlets as f32;
        assert!(
            mean_dbl > mean_sng * 2.0,
            "doublet mean ({:.4}) should be much higher than singlet mean ({:.4})",
            mean_dbl,
            mean_sng
        );
    }

    #[test]
    fn test_doublet_like_not_overfit() {
        // doublet-like but with significant overlap between classes
        // (noise features dominate, informative features have
        // overlapping ranges)
        let mut rng = SmallRng::seed_from_u64(2024);
        let n_singlets = 900;
        let n_doublets = 100;
        let n = n_singlets + n_doublets;

        let n_feat = 9;
        let mut columns: Vec<Vec<f32>> = (0..n_feat).map(|_| Vec::with_capacity(n)).collect();
        let mut labels = Vec::with_capacity(n);

        for i in 0..n {
            let is_dbl = i >= n_singlets;
            labels.push(is_dbl);

            if is_dbl {
                // weak signal: overlapping ranges on informative features
                columns[0].push(0.15 + rng.random::<f32>() * 0.5);
                columns[1].push(0.10 + rng.random::<f32>() * 0.4);
                columns[2].push(0.10 + rng.random::<f32>() * 0.4);
                // pure noise
                columns[3].push(rng.random::<f32>());
                columns[4].push(rng.random::<f32>() * 2.0);
                columns[5].push(rng.random::<f32>() * 1000.0);
                columns[6].push(rng.random::<f32>());
                columns[7].push(rng.random::<f32>() * 4.0 - 2.0);
                columns[8].push(rng.random::<f32>() * 4.0 - 2.0);
            } else {
                columns[0].push(rng.random::<f32>() * 0.4);
                columns[1].push(rng.random::<f32>() * 0.35);
                columns[2].push(rng.random::<f32>() * 0.3);
                columns[3].push(rng.random::<f32>());
                columns[4].push(rng.random::<f32>() * 2.0);
                columns[5].push(rng.random::<f32>() * 1000.0);
                columns[6].push(rng.random::<f32>());
                columns[7].push(rng.random::<f32>() * 4.0 - 2.0);
                columns[8].push(rng.random::<f32>() * 4.0 - 2.0);
            }
        }

        let store = QuantisedStore::from_columns(&columns);
        let exclude = vec![false; n];

        let probs = fit_logistic_gbm(
            &store,
            &labels,
            &exclude,
            &LogisticGbmConfig::default(),
            42,
            false,
        );

        let auc_val = auc(&probs, &labels);
        assert!(
            auc_val > 0.60,
            "weakly separable doublet-like data should still discriminate somewhat, got {:.4}",
            auc_val
        );

        // singlet predictions should NOT all be pinned near zero
        let singlet_probs = &probs[..n_singlets];
        let n_near_zero = singlet_probs.iter().filter(|&&p| p < 0.01).count();
        let frac_near_zero = n_near_zero as f32 / n_singlets as f32;
        assert!(
            frac_near_zero < 0.8,
            "singlet predictions are too concentrated near 0 ({:.1}%): \
                 model is overfitting",
            100.0 * frac_near_zero
        );

        // doublet predictions should NOT all be pinned near one
        let doublet_probs = &probs[n_singlets..];
        let n_near_one = doublet_probs.iter().filter(|&&p| p > 0.99).count();
        let frac_near_one = n_near_one as f32 / n_doublets as f32;
        assert!(
            frac_near_one < 0.8,
            "doublet predictions are too concentrated near 1 ({:.1}%): \
                 model is overfitting",
            100.0 * frac_near_one
        );
    }
}
