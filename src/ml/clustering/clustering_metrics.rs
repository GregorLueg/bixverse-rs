//! Metrics for clustering consistency. For now, it contains the adjusted Rand
//! index.

use rustc_hash::FxHashMap;

/// Adjusted Rand Index between two clusterings
///
/// Computes ARI from the contingency table using the combinatorial
/// formulation. Both inputs must have the same length.
///
/// ### Params
///
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
///
/// ### Returns
///
/// ARI score in [-1, 1]. 1 = perfect agreement, 0 = chance.
pub fn adjusted_rand_index(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    assert_eq!(n, labels_pred.len());

    let mut contingency: FxHashMap<(usize, usize), u64> = FxHashMap::default();
    for i in 0..n {
        *contingency
            .entry((labels_true[i], labels_pred[i]))
            .or_insert(0) += 1;
    }

    // Row and column sums
    let mut row_sums: FxHashMap<usize, u64> = FxHashMap::default();
    let mut col_sums: FxHashMap<usize, u64> = FxHashMap::default();
    for (&(r, c), &count) in &contingency {
        *row_sums.entry(r).or_insert(0) += count;
        *col_sums.entry(c).or_insert(0) += count;
    }

    let comb2 = |x: u64| -> i64 { (x as i64) * (x as i64 - 1) / 2 };

    let sum_comb_nij: i64 = contingency.values().map(|&v| comb2(v)).sum();
    let sum_comb_a: i64 = row_sums.values().map(|&v| comb2(v)).sum();
    let sum_comb_b: i64 = col_sums.values().map(|&v| comb2(v)).sum();
    let comb_n = comb2(n as u64);

    let expected = (sum_comb_a as f64 * sum_comb_b as f64) / comb_n as f64;
    let max_index = 0.5 * (sum_comb_a as f64 + sum_comb_b as f64);
    let denom = max_index - expected;

    if denom == 0.0 {
        if sum_comb_nij as f64 == expected {
            1.0
        } else {
            0.0
        }
    } else {
        (sum_comb_nij as f64 - expected) / denom
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_agreement() {
        let labels = vec![0, 0, 1, 1, 2, 2];
        let ari = adjusted_rand_index(&labels, &labels);
        assert!((ari - 1.0).abs() < 1e-10);
    }

    #[test]
    fn complete_disagreement() {
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 1, 0, 1];
        let ari = adjusted_rand_index(&labels_true, &labels_pred);
        assert!(ari < 0.5);
    }

    #[test]
    fn all_same_cluster() {
        let labels_true = vec![0, 1, 2, 3];
        let labels_pred = vec![0, 0, 0, 0];
        let ari = adjusted_rand_index(&labels_true, &labels_pred);
        assert!((ari - 0.0).abs() < 1e-10);
    }

    #[test]
    fn single_element() {
        let ari = adjusted_rand_index(&[0], &[0]);
        assert!((ari - 1.0).abs() < 1e-10);
    }

    #[test]
    fn known_value() {
        // sklearn: adjusted_rand_score([0,0,1,1], [0,1,1,0]) == 0.0
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 1, 1, 0];
        let ari = adjusted_rand_index(&labels_true, &labels_pred);
        assert!((ari - 0.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn mismatched_lengths() {
        adjusted_rand_index(&[0, 1], &[0]);
    }
}
