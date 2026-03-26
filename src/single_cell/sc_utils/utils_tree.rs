//! Shared infrastructure for histogram-based tree ensembles.
//!
//! Provides quantised feature storage, histogram accumulation, split
//! evaluation, sample partitioning, and early stopping -- the building
//! blocks used by both the SCENIC regression GBM and the scDblFinder
//! logistic GBM.

use faer::MatRef;
use rand::Rng;
use rand::rngs::SmallRng;

use crate::prelude::*;

/////////////////////
// Quantised store //
/////////////////////

/// Dense column-major store of quantised (u8) feature values.
///
/// Each feature column is independently scaled to `[0, 255]`. The
/// layout is `data[feature_idx * n_samples + sample_idx]`, giving
/// cache-friendly access when iterating samples within a single
/// feature column -- the hot path for histogram construction.
///
/// Per-feature `feature_min` and `feature_range` are stored for
/// optional reconstruction of original values but are not required
/// by the tree-building machinery.
pub struct QuantisedStore {
    /// Flat quantised data in column-major order.
    pub data: Vec<u8>,
    /// Number of samples (rows).
    pub n_samples: usize,
    /// Number of features (columns).
    pub n_features: usize,
    /// Per-feature minimum value prior to quantisation.
    pub feature_min: Vec<f32>,
    /// Per-feature range (`max - min`) prior to quantisation.
    pub feature_range: Vec<f32>,
}

impl QuantisedStore {
    /// Build from a CSC sparse matrix.
    ///
    /// Each feature column is independently scaled to `[0, 255]` using
    /// its observed min and max. Implicit zeros in the sparse structure
    /// are handled correctly provided all values are non-negative
    /// (the minimum is initialised to `0.0`). Features with range
    /// <= 1e-10 (effectively constant) are left at zero.
    ///
    /// ### Params
    ///
    /// * `mat` - Sparse CSC feature matrix (features as columns).
    /// * `n_samples` - Total number of samples (rows) in the matrix.
    ///
    /// ### Returns
    ///
    /// A fully populated `QuantisedStore`.
    pub fn from_csc(mat: &CompressedSparseData2<u16, f32>, n_samples: usize) -> Self {
        let n_features = mat.indptr.len() - 1;
        let mut data = vec![0u8; n_features * n_samples];
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

            let offset = j * n_samples;
            if range > 1e-10 {
                let scale = 255.0 / range;
                for i in 0..col_indices.len() {
                    let cell_idx = col_indices[i];
                    let val = col_vals[i];
                    data[offset + cell_idx] = ((val - min_v) * scale).round() as u8;
                }
            }
        }

        Self {
            data,
            n_samples,
            n_features,
            feature_min: mins,
            feature_range: ranges,
        }
    }

    /// Build from per-feature column vectors.
    ///
    /// Each inner `Vec<f32>` must have length `n_samples`. Columns are
    /// independently scaled to `[0, 255]`.
    ///
    /// ### Params
    ///
    /// * `columns` - One `Vec<f32>` per feature, each of length `n_samples`.
    ///
    /// ### Returns
    ///
    /// A fully populated `QuantisedStore`.
    ///
    /// ### Panics
    ///
    /// Panics if `columns` is empty or if any column has a different
    /// length to the first.
    pub fn from_columns(columns: &[Vec<f32>]) -> Self {
        let n_features = columns.len();
        assert!(!columns.is_empty(), "need at least one feature");
        let n_samples = columns[0].len();
        let mut data = vec![0u8; n_features * n_samples];
        let mut mins = Vec::with_capacity(n_features);
        let mut ranges = Vec::with_capacity(n_features);

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
            mins.push(min_v);
            ranges.push(range);

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
            feature_min: mins,
            feature_range: ranges,
        }
    }

    /// Build from a flat row-major `f32` slice.
    ///
    /// Transposes internally to column-major layout, then quantises
    /// each column independently.
    ///
    /// ### Params
    ///
    /// * `flat` - Row-major data of length `n_samples * n_features`.
    /// * `n_samples` - Number of rows (samples).
    /// * `n_features` - Number of columns (features).
    ///
    /// ### Returns
    ///
    /// A fully populated `QuantisedStore`.
    pub fn from_row_major(flat: &[f32], n_samples: usize, n_features: usize) -> Self {
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n_samples); n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                columns[j].push(flat[i * n_features + j]);
            }
        }
        Self::from_columns(&columns)
    }

    /// Build from a dense `faer::MatRef<f32>`.
    ///
    /// Each matrix column is independently scaled to `[0, 255]`.
    ///
    /// ### Params
    ///
    /// * `features` - Dense feature matrix, shape `(n_samples, n_features)`.
    ///
    /// ### Returns
    ///
    /// A fully populated `QuantisedStore`.
    pub fn from_mat(features: MatRef<f32>) -> Self {
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let mut data = vec![0u8; n_features * n_samples];
        let mut mins = Vec::with_capacity(n_features);
        let mut ranges = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for i in 0..n_samples {
                let v = *features.get(i, j);
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

            let offset = j * n_samples;
            if range > 1e-10 {
                let scale = 255.0 / range;
                for i in 0..n_samples {
                    let v = *features.get(i, j);
                    data[offset + i] = ((v - min_v) * scale).round() as u8;
                }
            }
        }

        Self {
            data,
            n_samples,
            n_features,
            feature_min: mins,
            feature_range: ranges,
        }
    }

    /// Build directly from pre-quantised data.
    ///
    /// Used by tests and any caller that already holds u8-quantised
    /// data in column-major layout.
    ///
    /// ### Params
    ///
    /// * `data` - Pre-quantised column-major data.
    /// * `n_samples` - Number of samples.
    /// * `n_features` - Number of features.
    ///
    /// ### Returns
    ///
    /// A `QuantisedStore` with zeroed `feature_min` and `feature_range`
    /// (reconstruction metadata is unavailable for pre-quantised data).
    pub fn from_raw(data: Vec<u8>, n_samples: usize, n_features: usize) -> Self {
        assert_eq!(data.len(), n_samples * n_features);
        Self {
            data,
            n_samples,
            n_features,
            feature_min: vec![0.0; n_features],
            feature_range: vec![0.0; n_features],
        }
    }

    /// Return the quantised values for a single feature column.
    ///
    /// ### Params
    ///
    /// * `feature` - Feature (column) index.
    ///
    /// ### Returns
    ///
    /// Slice of length `n_samples` containing u8-quantised values.
    #[inline(always)]
    pub fn get_col(&self, feature: usize) -> &[u8] {
        let start = feature * self.n_samples;
        &self.data[start..start + self.n_samples]
    }
}

////////////
// Helper //
////////////

/// Compute the variance of a node from sufficient statistics.
///
/// Uses the identity `var = E[X^2] - E[X]^2`, clamped to zero to
/// guard against floating-point underflow.
///
/// ### Params
///
/// * `sum` - Sum of values in the node.
/// * `sum_sq` - Sum of squared values in the node.
/// * `n` - Number of samples in the node.
///
/// ### Returns
///
/// The variance, clamped to `0.0`. Returns `0.0` for nodes with
/// fewer than 2 samples.
#[inline]
pub fn node_variance(sum: f32, sum_sq: f32, n: usize) -> f32 {
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    f32::max(0.0, sum_sq / nf - (sum / nf) * (sum / nf))
}

/// Split eligible sample indices into a training set and an
/// out-of-bag (OOB) set via partial Fisher-Yates shuffle.
///
/// After the call, `buf[..n_train]` contains the training indices
/// and `buf[n_train..]` contains the OOB indices. Both slices are
/// unordered.
///
/// ### Params
///
/// * `buf` - Mutable buffer of eligible sample indices. Modified
///   in place; on return it is partitioned into train and OOB.
/// * `n_train` - Number of training samples to select. Clamped to
///   `buf.len()` if larger.
/// * `rng` - Random number generator.
///
/// ### Returns
///
/// The actual number of training samples (may be less than
/// `n_train` if `buf` is shorter).
pub fn train_oob_split(buf: &mut [u32], n_train: usize, rng: &mut SmallRng) -> usize {
    let n = buf.len();
    let n_train = n_train.min(n);
    for i in 0..n_train {
        let j = rng.random_range(i..n);
        buf.swap(i, j);
    }
    n_train
}

/// Reset a buffer to the identity permutation `[0, 1, ..., n-1]`
/// then split into train/OOB.
///
/// Convenience wrapper that combines index initialisation with
/// `train_oob_split`.
///
/// ### Params
///
/// * `buf` - Buffer of length >= `n`. Will be overwritten with
///   `0..n` then partitioned.
/// * `n` - Total number of samples.
/// * `n_train` - Number of training samples.
/// * `rng` - Random number generator.
///
/// ### Returns
///
/// The actual number of training samples.
pub fn init_and_split(buf: &mut [u32], n: usize, n_train: usize, rng: &mut SmallRng) -> usize {
    for i in 0..n {
        buf[i] = i as u32;
    }
    train_oob_split(&mut buf[..n], n_train, rng)
}

/// Reusable scratch buffers for partitioning samples around a split.
///
/// Holds temporary left/right arrays for both training and OOB
/// sample sets. Allocated once and reused across all nodes within
/// a tree to avoid repeated heap allocation.
pub struct PartitionBuffers {
    /// Scratch for left training samples.
    pub train_left: Vec<u32>,
    /// Scratch for right training samples.
    pub train_right: Vec<u32>,
    /// Scratch for left OOB samples.
    pub oob_left: Vec<u32>,
    /// Scratch for right OOB samples.
    pub oob_right: Vec<u32>,
}

impl PartitionBuffers {
    /// Allocate partition buffers for the given maximum sample count.
    ///
    /// ### Params
    ///
    /// * `n_samples` - Upper bound on the number of samples that
    ///   will be partitioned (typically the total cell count).
    ///
    /// ### Returns
    ///
    /// Zeroed `PartitionBuffers`.
    pub fn new(n_samples: usize) -> Self {
        Self {
            train_left: vec![0u32; n_samples],
            train_right: vec![0u32; n_samples],
            oob_left: vec![0u32; n_samples],
            oob_right: vec![0u32; n_samples],
        }
    }

    /// Partition training samples by a threshold on a quantised
    /// feature column.
    ///
    /// Samples with `tf_col[s] <= threshold` go left; the rest go right.
    /// `train_samples` is rewritten in place so that
    /// `train_samples[..n_left]` holds the left partition and
    /// `train_samples[n_left..]` holds the right.
    ///
    /// ### Params
    ///
    /// * `train_samples` - Training sample indices; modified in place.
    /// * `tf_col` - Quantised feature column for all cells.
    /// * `threshold` - Split threshold (inclusive on the left side).
    ///
    /// ### Returns
    ///
    /// Number of samples in the left partition.
    pub fn partition_train(
        &mut self,
        train_samples: &mut [u32],
        tf_col: &[u8],
        threshold: u8,
    ) -> usize {
        let mut tl = 0usize;
        let mut tr = 0usize;
        for &s in train_samples.iter() {
            if tf_col[s as usize] <= threshold {
                self.train_left[tl] = s;
                tl += 1;
            } else {
                self.train_right[tr] = s;
                tr += 1;
            }
        }
        train_samples[..tl].copy_from_slice(&self.train_left[..tl]);
        train_samples[tl..tl + tr].copy_from_slice(&self.train_right[..tr]);
        tl
    }

    /// Partition OOB samples by a threshold on a quantised feature column.
    ///
    /// Same semantics as `partition_train` but operates on the OOB
    /// set.
    ///
    /// ### Params
    ///
    /// * `oob_samples` - OOB sample indices; modified in place.
    /// * `tf_col` - Quantised feature column for all cells.
    /// * `threshold` - Split threshold (inclusive on the left side).
    ///
    /// ### Returns
    ///
    /// Number of samples in the left partition.
    pub fn partition_oob(
        &mut self,
        oob_samples: &mut [u32],
        tf_col: &[u8],
        threshold: u8,
    ) -> usize {
        let mut ol = 0usize;
        let mut or_ = 0usize;
        for &s in oob_samples.iter() {
            if tf_col[s as usize] <= threshold {
                self.oob_left[ol] = s;
                ol += 1;
            } else {
                self.oob_right[or_] = s;
                or_ += 1;
            }
        }
        oob_samples[..ol].copy_from_slice(&self.oob_left[..ol]);
        oob_samples[ol..ol + or_].copy_from_slice(&self.oob_right[..or_]);
        ol
    }
}

/// Rolling-window early stopping monitor.
///
/// Tracks a fixed-size ring of per-round improvement values and
/// triggers a stop when the rolling average drops to zero or below.
pub struct EarlyStopMonitor {
    /// Ring buffer of recent improvement values.
    ring: Vec<f32>,
    /// Maximum number of entries before evaluation begins.
    window: usize,
}

impl EarlyStopMonitor {
    /// Create a new monitor with the given window size.
    ///
    /// No stopping decision is made until the ring contains at
    /// least `window` entries.
    ///
    /// ### Params
    ///
    /// * `window` - Number of recent rounds to average.
    ///
    /// ### Returns
    ///
    /// An empty `EarlyStopMonitor`.
    pub fn new(window: usize) -> Self {
        Self {
            ring: Vec::with_capacity(window),
            window,
        }
    }

    /// Record an improvement value and return whether to stop.
    ///
    /// ### Params
    ///
    /// * `improvement` - The improvement metric for the current
    ///   round (e.g. per-sample OOB MSE reduction).
    ///
    /// ### Returns
    ///
    /// `true` if the rolling average over the last `window` rounds
    /// is <= 0, indicating that boosting should stop.
    pub fn push(&mut self, improvement: f32) -> bool {
        if self.ring.len() >= self.window {
            self.ring.remove(0);
        }
        self.ring.push(improvement);
        if self.ring.len() >= self.window {
            let avg: f32 = self.ring.iter().sum::<f32>() / self.ring.len() as f32;
            avg <= 0.0
        } else {
            false
        }
    }

    /// Reset the monitor, clearing all recorded values.
    pub fn reset(&mut self) {
        self.ring.clear();
    }
}

/// Normalise an importance vector to sum to 1.0.
///
/// If the total is zero or negative (no variance was reduced), the
/// vector is left unchanged (all zeros).
///
/// ### Params
///
/// * `importances` - Mutable slice of per-feature importance values.
pub fn normalise_importances(importances: &mut [f32]) {
    let total: f32 = importances.iter().sum();
    if total > 0.0 {
        let inv = 1.0 / total;
        importances.iter_mut().for_each(|v| *v *= inv);
    }
}

/// Partial Fisher-Yates shuffle to select `k` features from the
/// permutation buffer.
///
/// After the call, `feat_buf[..k]` contains the selected feature
/// indices (in random order). The rest of `feat_buf` is in an
/// unspecified but valid permutation state.
///
/// ### Params
///
/// * `feat_buf` - Feature index permutation buffer; length must be
///   >= `n_features`.
/// * `n_features` - Total number of features.
/// * `k` - Number of features to select. Clamped to `n_features`.
/// * `rng` - Random number generator.
///
/// ### Returns
///
/// The effective number of features selected (min of `k` and
/// `n_features`).
pub fn sample_features(
    feat_buf: &mut [usize],
    n_features: usize,
    k: usize,
    rng: &mut SmallRng,
) -> usize {
    let k = k.min(n_features);
    for i in 0..k {
        let j = rng.random_range(i..n_features);
        feat_buf.swap(i, j);
    }
    k
}

/// Compute the effective number of features to evaluate per split.
///
/// If `configured` is 0, returns `sqrt(n_features)` (clamped to
/// at least 1). Otherwise returns `configured` as-is.
///
/// ### Params
///
/// * `configured` - The user-configured value; `0` means "auto".
/// * `n_features` - Total number of available features.
///
/// ### Returns
///
/// The number of features to sample per split.
pub fn resolve_n_features_split(configured: usize, n_features: usize) -> usize {
    if configured == 0 {
        ((n_features as f32).sqrt() as usize).max(1)
    } else {
        configured
    }
}

////////////////////
// GBM split info //
////////////////////

/// Information about a split found in a node's histograms.
///
/// Captures everything needed to partition samples and propagate
/// sufficient statistics to child nodes.
#[derive(Clone, Debug)]
pub struct GbmSplitInfo {
    /// Feature index of the best split.
    pub feature: usize,
    /// Bin threshold; samples with `bin <= threshold` go left.
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

/////////////////////////////////////
// Node histograms (single target) //
/////////////////////////////////////

/// Full-feature bin histograms for a single GBM tree node.
///
/// Stores per-bin sample counts, residual sums, and residual
/// sum-of-squares for every feature simultaneously. This enables
/// the histogram subtraction trick: build the smaller child from
/// scratch, derive the larger as `parent - smaller` in
/// O(n_features * 256).
///
/// All arrays use layout `[feature * 256 + bin]`.
pub struct NodeHistograms {
    /// Per-bin sample counts.
    pub counts: Vec<u32>,
    /// Per-bin residual sums.
    pub y_sums: Vec<f32>,
    /// Per-bin residual sum-of-squares.
    pub y_sum_sqs: Vec<f32>,
    /// Number of features.
    pub n_features: usize,
}

impl NodeHistograms {
    /// Allocate zeroed histograms for the given number of features.
    ///
    /// ### Params
    ///
    /// * `n_features` - Number of features.
    ///
    /// ### Returns
    ///
    /// A `NodeHistograms` with all bins set to zero.
    pub fn new(n_features: usize) -> Self {
        let n = n_features * 256;
        Self {
            counts: vec![0u32; n],
            y_sums: vec![0.0f32; n],
            y_sum_sqs: vec![0.0f32; n],
            n_features,
        }
    }

    /// Zero all bins across all features.
    pub fn reset(&mut self) {
        let n = self.n_features * 256;
        self.counts[..n].fill(0);
        self.y_sums[..n].fill(0.0);
        self.y_sum_sqs[..n].fill(0.0);
    }

    /// Build histograms from a set of sample indices.
    ///
    /// Iterates feature-major for cache-friendly access to the
    /// quantised feature columns. The per-feature histogram (256
    /// bins * 12 bytes) fits comfortably in L1.
    ///
    /// ### Params
    ///
    /// * `x` - Quantised feature store.
    /// * `samples` - Indices of the samples to include.
    /// * `residuals` - Dense residual (or target) array indexed by
    ///   cell id.
    pub fn build_from_samples(&mut self, x: &QuantisedStore, samples: &[u32], residuals: &[f32]) {
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

    /// Find the best split across all (or a random subset of)
    /// features.
    ///
    /// For each candidate feature, computes prefix sums over the 256 bins and
    /// evaluates every valid threshold. Returns `None` if no improving split
    /// exists.
    ///
    /// ### Params
    ///
    /// * `total_sum` - Sum of residuals in this node.
    /// * `total_sum_sq` - Sum of squared residuals in this node.
    /// * `n_samples` - Number of training samples in this node.
    /// * `min_samples_leaf` - Minimum samples per child.
    /// * `n_features_split` - Features to evaluate; `0` means all.
    /// * `feat_buf` - Scratch buffer for feature permutation
    ///   (length >= `n_features`).
    /// * `rng` - RNG for feature subsampling.
    ///
    /// ### Returns
    ///
    /// `Some(GbmSplitInfo)` if an improving split was found, `None`
    /// otherwise.
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

        sample_features(feat_buf, n_features, k_feats, rng);

        let parent_var = node_variance(total_sum, total_sum_sq, n_samples as usize);
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

////////////////////
// Histogram pool //
////////////////////

/// Pre-allocated pool of `NodeHistograms` with acquire/release
/// semantics.
///
/// Avoids repeated heap allocation during tree building. The pool
/// should be sized for the maximum recursion depth at construction
/// time (typically `2 * max_depth + 3`).
pub struct HistogramPool {
    /// The histogram storage. Indexed by pool slot.
    pub histograms: Vec<NodeHistograms>,
    /// Stack of free slot indices.
    free: Vec<usize>,
}

impl HistogramPool {
    /// Create a pool with `capacity` histograms, each sized for
    /// `n_features` features.
    ///
    /// ### Params
    ///
    /// * `capacity` - Number of histogram slots to pre-allocate.
    /// * `n_features` - Number of features per histogram.
    ///
    /// ### Returns
    ///
    /// A fully initialised `HistogramPool`.
    pub fn new(capacity: usize, n_features: usize) -> Self {
        let histograms = (0..capacity)
            .map(|_| NodeHistograms::new(n_features))
            .collect();
        let free = (0..capacity).rev().collect();
        Self { histograms, free }
    }

    /// Acquire a histogram slot from the pool.
    ///
    /// ### Returns
    ///
    /// Index into `self.histograms` for the acquired slot.
    ///
    /// ### Panics
    ///
    /// Panics if the pool is exhausted.
    pub fn acquire(&mut self) -> usize {
        self.free.pop().expect("histogram pool exhausted")
    }

    /// Return a histogram slot to the pool.
    ///
    /// ### Params
    ///
    /// * `idx` - Slot index to release.
    pub fn release(&mut self, idx: usize) {
        self.free.push(idx);
    }

    /// Compute `out = parent - child` element-wise for all bins.
    ///
    /// This is the histogram subtraction trick: after building the
    /// smaller child's histogram from scratch, the larger child's
    /// histogram is derived in O(n_features * 256) without scanning
    /// the larger child's samples.
    ///
    /// ### Params
    ///
    /// * `parent` - Slot index of the parent node histogram.
    /// * `child` - Slot index of the smaller child histogram.
    /// * `out` - Slot index to write the result into.
    ///
    /// ### Panics
    ///
    /// Debug-asserts that all three indices are distinct.
    pub fn subtract(&mut self, parent: usize, child: usize, out: usize) {
        debug_assert_ne!(parent, out);
        debug_assert_ne!(child, out);
        let n = self.histograms[0].n_features * 256;
        for i in 0..n {
            self.histograms[out].counts[i] =
                self.histograms[parent].counts[i] - self.histograms[child].counts[i];
            self.histograms[out].y_sums[i] =
                self.histograms[parent].y_sums[i] - self.histograms[child].y_sums[i];
            self.histograms[out].y_sum_sqs[i] =
                self.histograms[parent].y_sum_sqs[i] - self.histograms[child].y_sum_sqs[i];
        }
    }
}

/////////////////
// GBM scratch //
/////////////////

/// Reusable scratch buffers for GBM tree building.
///
/// Combines the feature permutation buffer with partition scratch
/// space. Allocated once per tree and reused across all nodes.
pub struct GbmScratch {
    /// Feature permutation buffer for partial Fisher-Yates.
    pub feat_buf: Vec<usize>,
    /// Partition buffers for train and OOB sample sets.
    pub partitions: PartitionBuffers,
}

impl GbmScratch {
    /// Allocate scratch buffers.
    ///
    /// ### Params
    ///
    /// * `n_features` - Number of features (determines `feat_buf`
    ///   length).
    /// * `n_samples` - Maximum number of samples (determines
    ///   partition buffer lengths).
    ///
    /// ### Returns
    ///
    /// An initialised `GbmScratch`.
    pub fn new(n_features: usize, n_samples: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            partitions: PartitionBuffers::new(n_samples),
        }
    }
}

//////////////////////
// Leaf application //
// (regression GBM) //
//////////////////////

/// Apply a leaf prediction in a regression GBM: accumulate OOB
/// improvement then update residuals.
///
/// OOB improvement is computed **before** residual updates so it
/// reflects the pre-update state. The per-sample improvement is
/// `2 * lr * pred * r[s] - (lr * pred)^2`, which equals
/// `r[s]^2 - (r[s] - lr * pred)^2`.
///
/// Both training and OOB residuals are updated by subtracting the
/// shrunk prediction.
///
/// ### Params
///
/// * `residuals` - Dense residual array indexed by cell id;
///   modified in place.
/// * `train_samples` - Training sample indices in this leaf.
/// * `oob_samples` - OOB sample indices in this leaf.
/// * `y_sum_train` - Sum of training residuals in this leaf.
/// * `n_train` - Number of training samples in this leaf.
/// * `learning_rate` - Shrinkage factor applied to the leaf
///   prediction.
/// * `oob_improvement` - Accumulated OOB squared-error improvement;
///   updated in place.
pub fn apply_regression_leaf(
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

    for &s in oob_samples {
        let r = residuals[s as usize];
        *oob_improvement += 2.0 * lr_pred * r - lr_pred_sq;
    }

    for &s in train_samples {
        residuals[s as usize] -= lr_pred;
    }
    for &s in oob_samples {
        residuals[s as usize] -= lr_pred;
    }
}

/// Derive a per-tree seed from a base seed and tree index.
///
/// Uses the same LCG constant as the existing code to maintain
/// determinism.
///
/// ### Params
///
/// * `base_seed` - The base seed for the ensemble.
/// * `tree_idx` - Index of the current tree.
///
/// ### Returns
///
/// A `u64` seed suitable for `SmallRng::seed_from_u64`.
#[inline]
pub fn tree_seed(base_seed: usize, tree_idx: usize) -> u64 {
    base_seed.wrapping_add(tree_idx.wrapping_mul(6_364_136_223_846_793_005)) as u64
}
