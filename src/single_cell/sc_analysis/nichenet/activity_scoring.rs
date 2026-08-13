//! Per-ligand activity scoring against a binary gene set.
//!
//! Given a ligand-target matrix sliced to (rows = ligands of interest,
//! cols = background gene set) and a binary response over the background,
//! compute AUROC, AUPR, AUPR-corrected, Pearson, and Spearman per ligand.
//!
//! All metrics use the same prediction/response pair — they are different
//! summaries of the same alignment between a ligand's target potential
//! vector and the gene set membership vector.

use faer::MatRef;
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
pub fn ligand_activity_scores<T>(
    predictions: &MatRef<T>,
    response: &[bool],
) -> LigandActivityScores<T>
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// AUROC saturates at 1.0 when every positive outranks every negative.
    #[test]
    fn auroc_perfect_ranking() {
        // positives hold the top ranks
        let ranks = vec![1.0_f64, 2.0, 3.0, 4.0];
        let pos = vec![2usize, 3];
        assert!((auroc_from_ranks(&ranks, &pos, 2, 2) - 1.0).abs() < 1e-12);
    }

    /// The mirror case gives 0.0, which pins the direction of the rank convention.
    #[test]
    fn auroc_inverse_ranking() {
        let ranks = vec![1.0_f64, 2.0, 3.0, 4.0];
        let pos = vec![0usize, 1];
        assert!(auroc_from_ranks(&ranks, &pos, 2, 2).abs() < 1e-12);
    }

    /// A half-right ranking lands on the hand-computed value, not something near chance.
    #[test]
    fn auroc_partial() {
        // positives at ranks 1, 3 -> u = 4 - 3 = 1 -> auroc = 0.25
        let ranks = vec![1.0_f64, 2.0, 3.0, 4.0];
        let pos = vec![0usize, 2];
        assert!((auroc_from_ranks(&ranks, &pos, 2, 2) - 0.25).abs() < 1e-12);
    }

    /// With no positives or no negatives the AUROC is undefined, so it must be NaN rather than 0.
    #[test]
    fn auroc_degenerate_returns_nan() {
        let ranks = vec![1.0_f64, 2.0];
        assert!(auroc_from_ranks(&ranks, &[], 0, 2).is_nan());
        assert!(auroc_from_ranks(&ranks, &[0, 1], 2, 0).is_nan());
    }

    /// AUPR reaches 1.0 when the positives take the top predictions.
    #[test]
    fn aupr_perfect_ranking() {
        let pred = vec![4.0_f64, 3.0, 2.0, 1.0];
        let response = vec![true, true, false, false];
        assert!((aupr_value(&pred, &response, 2) - 1.0).abs() < 1e-12);
    }

    /// An all-positive or zero-positive response leaves the AUPR undefined.
    #[test]
    fn aupr_degenerate_returns_nan() {
        let pred = vec![1.0_f64, 2.0];
        assert!(aupr_value(&pred, &[true, true], 2).is_nan());
        assert!(aupr_value(&pred, &[false, false], 0).is_nan());
    }

    /// A perfectly linear pair correlates at exactly 1.0.
    #[test]
    fn pearson_perfect_positive() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y = vec![2.0_f64, 4.0, 6.0, 8.0];
        assert!((pearson_generic(&x, &y) - 1.0).abs() < 1e-12);
    }

    /// A reversed pair correlates at exactly -1.0, so the sign is not lost.
    #[test]
    fn pearson_perfect_negative() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y = vec![4.0_f64, 3.0, 2.0, 1.0];
        assert!((pearson_generic(&x, &y) + 1.0).abs() < 1e-12);
    }

    /// A constant vector has no variance, so the result is NaN instead of a division by zero.
    #[test]
    fn pearson_zero_variance_returns_nan() {
        let x = vec![1.0_f64, 1.0, 1.0];
        let y = vec![1.0_f64, 2.0, 3.0];
        assert!(pearson_generic(&x, &y).is_nan());
    }

    /// All four activity metrics agree on a cleanly separated single-ligand case.
    #[test]
    fn ligand_activity_perfect_separation() {
        // single ligand, positives ranked above negatives
        let mut pred = Mat::<f64>::zeros(1, 4);
        pred[(0, 0)] = 4.0;
        pred[(0, 1)] = 3.0;
        pred[(0, 2)] = 2.0;
        pred[(0, 3)] = 1.0;
        let response = vec![true, true, false, false];
        let s = ligand_activity_scores(&pred.as_ref(), &response);
        assert!((s.auroc[0] - 1.0).abs() < 1e-12);
        assert!((s.aupr[0] - 1.0).abs() < 1e-12);
        // aupr_random = 2/4 = 0.5
        assert!((s.aupr_corrected[0] - 0.5).abs() < 1e-12);
        // Pearson/Spearman are not 1.0 here because the response has ties,
        // but they should be strongly positive.
        assert!(s.pearson[0] > 0.0);
        assert!(s.spearman[0] > 0.0);
    }
}
