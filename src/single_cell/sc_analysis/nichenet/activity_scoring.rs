//! Per-ligand activity scoring against a binary gene set.
//!
//! Given a ligand-target matrix sliced to (rows = ligands of interest,
//! cols = background gene set) and a binary response over the background,
//! compute AUROC, AUPR, AUPR-corrected, Pearson, and Spearman per ligand.
//!
//! All metrics use the same prediction/response pair — they are different
//! summaries of the same alignment between a ligand's target potential
//! vector and the gene set membership vector.

use faer::Mat;
use rayon::prelude::*;

use crate::core::math::vector_helpers::rank_vector;
use crate::prelude::*;

/////////////
// Results //
/////////////

/// Output of `ligand_activity_scores`. Each Vec has length = number of ligands.
#[derive(Clone, Debug)]
pub struct LigandActivityScores<T> {
    /// Area Under the Receiver Operating Characteristic
    pub auroc: Vec<T>,
    /// Area Under the Precision-Recall Curve
    pub aupr: Vec<T>,
    /// Area Under the Precision-Recall Curve (corrected)
    pub aupr_corrected: Vec<T>,
    /// Pearson correlations
    pub pearson: Vec<T>,
    /// Spearman correlations
    pub spearman: Vec<T>,
}

/////////////
// Helpers //
/////////////

/// AUROC via Mann-Whitney rank-sum identity.
///
/// Tie-correct ranks from `rank_vector`.
///
/// ### Params
///
/// * `ranks` - Slice of ranks
/// * `pos_indices` - Indices of the in-geneset
/// * `n_pos` - Number of positive hits
/// * `n_neg` - Number of negatives
///
/// ### Returns
///
/// The AUROC
fn auroc_from_ranks<T: BixverseFloat + std::iter::Sum>(
    ranks: &[T],
    pos_indices: &[usize],
    n_pos: usize,
    n_neg: usize,
) -> T {
    if n_pos == 0 || n_neg == 0 {
        return T::nan();
    }
    let sum_pos_ranks: T = pos_indices.iter().map(|&i| ranks[i]).sum();
    let n_pos_t = T::from_usize(n_pos).unwrap();
    let n_neg_t = T::from_usize(n_neg).unwrap();
    let half = T::from_f64(0.5).unwrap();
    let u = sum_pos_ranks - n_pos_t * (n_pos_t + T::one()) * half;
    u / (n_pos_t * n_neg_t)
}

/// AUPR via trapezoidal integration over (recall, precision) points.
///
/// Ties at the same prediction value are accumulated together before the
/// point is emitted — standard convention. Starts from (recall=0, precision=1).
///
/// ### Params
///
/// * `pred` - The predictions
/// * `response` - The response variables
/// * `n_pos` - Number of positive hits
///
/// ### Returns
///
/// The AUPR
fn aupr_value<T: BixverseFloat>(pred: &[T], response: &[bool], n_pos: usize) -> T {
    let n = pred.len();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return T::nan();
    }

    let mut idx: Vec<usize> = (0..n).collect();
    // descending by prediction
    idx.sort_unstable_by(|&a, &b| pred[b].total_cmp(&pred[a]));

    let n_pos_t = T::from_usize(n_pos).unwrap();
    let half = T::from_f64(0.5).unwrap();

    let mut tp: usize = 0;
    let mut fp: usize = 0;
    let mut prev_recall = T::zero();
    let mut prev_precision = T::one();
    let mut aupr = T::zero();

    let mut i = 0;
    while i < n {
        let current = pred[idx[i]];
        let mut j = i;
        while j < n && pred[idx[j]] == current {
            if response[idx[j]] {
                tp += 1;
            } else {
                fp += 1;
            }
            j += 1;
        }
        let tp_t = T::from_usize(tp).unwrap();
        let recall = tp_t / n_pos_t;
        let precision = tp_t / T::from_usize(tp + fp).unwrap();
        aupr += (recall - prev_recall) * (precision + prev_precision) * half;
        prev_recall = recall;
        prev_precision = precision;
        i = j;
    }
    aupr
}

/// Pearson correlation on two equal-length vectors.
///
/// Biased variance is used (1/n); the factor cancels in the ratio so this is
/// identical to the sample-variance version.
///
/// ### Params
///
/// * `x` - Slice of x
/// * `y` - Slice of y
///
/// ### Returns
///
/// The Pearson correlation (or `NaN` for zero variance vectors)
fn pearson_generic<T: BixverseFloat + std::iter::Sum>(x: &[T], y: &[T]) -> T {
    let n = x.len();
    if n < 2 {
        return T::nan();
    }
    let n_t = T::from_usize(n).unwrap();
    let sum_x: T = x.iter().copied().sum();
    let sum_y: T = y.iter().copied().sum();
    let sum_xy: T = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let sum_x2: T = x.iter().map(|&a| a * a).sum();
    let sum_y2: T = y.iter().map(|&a| a * a).sum();
    let mean_x = sum_x / n_t;
    let mean_y = sum_y / n_t;
    let var_x = sum_x2 / n_t - mean_x * mean_x;
    let var_y = sum_y2 / n_t - mean_y * mean_y;
    if var_x <= T::zero() || var_y <= T::zero() {
        return T::nan();
    }
    let cov = sum_xy / n_t - mean_x * mean_y;
    cov / (var_x.sqrt() * var_y.sqrt())
}

//////////
// Main //
//////////

/// Compute per-ligand activity scores.
///
/// ### Params
///
/// * `predictions` - Dense `(n_ligands, n_background)` matrix.
/// * `response` - Binary membership vector of length `n_background`.
///
/// ### Returns
///
/// Degenerate cases (no positives, no negatives, zero-variance prediction)
/// return `T::nan()` for the affected metric. Other returns the
/// [LigandActivityScores].
pub fn ligand_activity_scores<T>(predictions: &Mat<T>, response: &[bool]) -> LigandActivityScores<T>
where
    T: BixverseFloat + std::iter::Sum + Send + Sync,
{
    let n_ligands = predictions.nrows();
    let n = response.len();
    assert_eq!(
        predictions.ncols(),
        n,
        "prediction columns must match response length"
    );

    let n_pos: usize = response.iter().filter(|&&b| b).count();
    let n_neg: usize = n - n_pos;

    // constants derived from the (shared) response vector.
    let response_t: Vec<T> = response
        .iter()
        .map(|&b| if b { T::one() } else { T::zero() })
        .collect();
    let half = T::from_f64(0.5).unwrap();
    let rank_neg = (T::from_usize(n_neg).unwrap() + T::one()) * half;
    let rank_pos =
        T::from_usize(n_neg).unwrap() + (T::from_usize(n_pos).unwrap() + T::one()) * half;
    let response_ranks: Vec<T> = response
        .iter()
        .map(|&b| if b { rank_pos } else { rank_neg })
        .collect();
    let pos_indices: Vec<usize> = response
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();

    let aupr_random = if n > 0 {
        T::from_usize(n_pos).unwrap() / T::from_usize(n).unwrap()
    } else {
        T::nan()
    };

    let results: Vec<(T, T, T, T)> = (0..n_ligands)
        .into_par_iter()
        .map(|i| {
            let pred_row: Vec<T> = (0..n).map(|j| predictions[(i, j)]).collect();
            let pred_ranks = rank_vector(&pred_row);
            let auroc = auroc_from_ranks(&pred_ranks, &pos_indices, n_pos, n_neg);
            let aupr = aupr_value(&pred_row, response, n_pos);
            let pearson = pearson_generic(&pred_row, &response_t);
            let spearman = pearson_generic(&pred_ranks, &response_ranks);

            (auroc, aupr, pearson, spearman)
        })
        .collect();

    let mut auroc = Vec::with_capacity(n_ligands);
    let mut aupr = Vec::with_capacity(n_ligands);
    let mut aupr_corrected = Vec::with_capacity(n_ligands);
    let mut pearson = Vec::with_capacity(n_ligands);
    let mut spearman = Vec::with_capacity(n_ligands);
    for (a, ap, p, s) in results {
        auroc.push(a);
        aupr.push(ap);
        aupr_corrected.push(ap - aupr_random);
        pearson.push(p);
        spearman.push(s);
    }

    LigandActivityScores {
        auroc,
        aupr,
        aupr_corrected,
        pearson,
        spearman,
    }
}
