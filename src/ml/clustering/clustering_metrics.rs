//! Metrics for clustering consistency and cluster quality. Contains the
//! adjusted Rand index and a silhouette score specialised to unit-norm rows,
//! where the closed form avoids the usual all-pairs distance matrix.

use faer::{Mat, MatRef};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::prelude::*;

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
    if n <= 1 {
        return 1.0;
    }

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

////////////////
// Silhouette //
////////////////

/// Silhouette score for unit-L2 rows under cosine distance.
///
/// Cosine distance between unit-norm rows is `1 - <a, b>`, so the mean distance
/// from row `i` to every member of cluster `c` collapses to
/// `(|c| - <x_i, S_c>) / |c|`, where `S_c` is the plain sum of the rows in `c`.
/// Scoring row `i` therefore needs only its `n_clusters` dot products against
/// those sums, giving `O(n * n_clusters * dim)` without ever materialising the
/// `n x n` distance matrix the textbook formulation needs. The own-cluster mean
/// subtracts the self term out of `<x_i, S_own>` explicitly, so it does not lean
/// on `<x_i, x_i>` being exactly one.
///
/// The sums and the dot products are accumulated in `f64` regardless of `F`.
/// This is not optional: `<x_i, S_c>` is approximately `|c|`, so
/// `|c| - <x_i, S_c>` subtracts two nearly equal numbers to recover a value near
/// zero for a tight cluster. Accumulating in `f32` understates a clean pair of
/// clusters by around 0.2. It is the same reason
/// [`crate::methods::nmf_hals::refit`] evaluates its objective in `f64`. The cost
/// is a `n_clusters x dim` `f64` buffer, and no copy of `data`, which matters
/// because `dim` can be the cell count on the single-cell path.
///
/// Rows are assumed to have unit L2 norm. The caller owns that invariant; a
/// non-normalised input silently gives a different (and meaningless) metric, and
/// a score marginally outside `[-1, 1]` is the symptom. Singleton clusters score
/// `0` by the usual convention, and clusters that are declared but empty are
/// skipped rather than counted.
///
/// ### Params
///
/// * `data` - Row-major points, shape `n x dim`. Rows must have unit L2 norm.
/// * `labels` - Cluster index per row, each in `0..n_clusters`. Length `n`.
/// * `n_clusters` - Number of clusters. Must be at least 2.
///
/// ### Returns
///
/// Tuple of `(per-row silhouette, mean silhouette)`. Both are `0` when `n` is
/// zero or fewer than two clusters are actually populated.
///
/// ### References
///
/// Rousseeuw, Journal of Computational and Applied Mathematics, 1987
pub fn silhouette_cosine_unit<F>(
    data: MatRef<F>,
    labels: &[usize],
    n_clusters: usize,
) -> (Vec<F>, F)
where
    F: BixverseFloat + Send + Sync,
{
    let n = data.nrows();
    let dim = data.ncols();
    assert_eq!(
        labels.len(),
        n,
        "labels length must match the number of rows"
    );

    if n == 0 || n_clusters < 2 {
        return (vec![F::zero(); n], F::zero());
    }

    // Cluster sums and sizes. S_c is the plain sum, not the mean: the closed
    // form below wants the sum so that the size cancels explicitly.
    let mut sums = Mat::<f64>::zeros(n_clusters, dim);
    let mut sizes = vec![0usize; n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        assert!(label < n_clusters, "label {label} is out of range");
        sizes[label] += 1;
        for j in 0..dim {
            sums[(label, j)] += data[(i, j)].to_f64().unwrap();
        }
    }

    let n_populated = sizes.iter().filter(|&&s| s > 0).count();
    if n_populated < 2 {
        return (vec![F::zero(); n], F::zero());
    }

    let per_row: Vec<F> = (0..n)
        .into_par_iter()
        // One scratch buffer per thread: row i needs its dot product against
        // every cluster sum, and walking `dim` once while accumulating all of
        // them reads the data row a single time.
        .map_init(
            || vec![0f64; n_clusters],
            |dots, i| {
                dots.iter_mut().for_each(|d| *d = 0.0);
                let mut self_dot = 0f64;
                for j in 0..dim {
                    let x = data[(i, j)].to_f64().unwrap();
                    self_dot += x * x;
                    for (c, dot) in dots.iter_mut().enumerate() {
                        *dot += x * sums[(c, j)];
                    }
                }

                let own = labels[i];
                let own_size = sizes[own];
                if own_size <= 1 {
                    return F::zero();
                }

                // Own-cluster mean over the OTHER members. `dots[own]` includes
                // the self term, so it is subtracted explicitly rather than
                // folded into the count: that would assume `<x_i, x_i> == 1`,
                // which f32 rows only satisfy to about 1e-7. The resulting bias
                // is systematic across all members and is enough to push the
                // score above 1 on tight clusters.
                let others = (own_size - 1) as f64;
                let a = (others - (dots[own] - self_dot)) / others;

                let mut b = f64::INFINITY;
                for c in 0..n_clusters {
                    if c == own || sizes[c] == 0 {
                        continue;
                    }
                    let count = sizes[c] as f64;
                    let mean = (count - dots[c]) / count;
                    if mean < b {
                        b = mean;
                    }
                }

                if !b.is_finite() {
                    return F::zero();
                }

                let max_ab = a.max(b);
                let score = if max_ab > 0.0 { (b - a) / max_ab } else { 0.0 };

                // Deliberately not clamped to [-1, 1]. With the algebra above and
                // genuinely unit-norm rows the score is in range; a value outside
                // it means the caller's normalisation is off, and clamping would
                // hide that as readily as it would hide a bug here.
                F::from_f64(score).unwrap()
            },
        )
        .collect();

    let total: f64 = per_row.iter().map(|x| x.to_f64().unwrap()).sum();
    let mean = F::from_f64(total / n as f64).unwrap();

    (per_row, mean)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Identical labellings sit at the top of the scale, exactly 1.
    #[test]
    fn perfect_agreement() {
        let labels = vec![0, 0, 1, 1, 2, 2];
        let ari = adjusted_rand_index(&labels, &labels);
        assert!((ari - 1.0).abs() < 1e-10);
    }

    /// A single-cluster prediction carries no information, so it scores at chance.
    #[test]
    fn all_same_cluster() {
        let labels_true = vec![0, 1, 2, 3];
        let labels_pred = vec![0, 0, 0, 0];
        let ari = adjusted_rand_index(&labels_true, &labels_pred);
        assert!((ari - 0.0).abs() < 1e-10);
    }

    /// One element forms no pairs, so the combinatorial form must not divide by zero.
    #[test]
    fn single_element() {
        let ari = adjusted_rand_index(&[0], &[0]);
        assert!((ari - 1.0).abs() < 1e-10);
    }

    /// Pins a worse-than-chance, negative score against an external oracle.
    #[test]
    fn known_value() {
        // sklearn: adjusted_rand_score([0,0,1,1], [0,1,1,0]) == -0.5. ARI is
        // invariant to relabelling, so the other cross pairing matches.
        let labels_true = vec![0, 0, 1, 1];
        let ari = adjusted_rand_index(&labels_true, &[0, 1, 1, 0]);
        assert!((ari - (-0.5)).abs() < 1e-10);
        let ari = adjusted_rand_index(&labels_true, &[0, 1, 0, 1]);
        assert!((ari - (-0.5)).abs() < 1e-10);
    }

    /// Unequal label lengths are a caller error and must panic, not truncate.
    #[test]
    #[should_panic]
    fn mismatched_lengths() {
        adjusted_rand_index(&[0, 1], &[0]);
    }

    /// Textbook silhouette over an explicit cosine distance matrix. The oracle
    /// the closed form is checked against.
    fn silhouette_naive(data: MatRef<f64>, labels: &[usize], n_clusters: usize) -> Vec<f64> {
        let n = data.nrows();
        let dim = data.ncols();
        let dist = |i: usize, j: usize| {
            let mut dot = 0.0;
            for c in 0..dim {
                dot += data[(i, c)] * data[(j, c)];
            }
            1.0 - dot
        };

        (0..n)
            .map(|i| {
                let mut sums = vec![0.0; n_clusters];
                let mut counts = vec![0usize; n_clusters];
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    sums[labels[j]] += dist(i, j);
                    counts[labels[j]] += 1;
                }

                let own = labels[i];
                if counts[own] == 0 {
                    return 0.0;
                }
                let a = sums[own] / counts[own] as f64;

                let mut b = f64::INFINITY;
                for c in 0..n_clusters {
                    if c == own || counts[c] == 0 {
                        continue;
                    }
                    let mean = sums[c] / counts[c] as f64;
                    if mean < b {
                        b = mean;
                    }
                }
                if !b.is_finite() {
                    return 0.0;
                }

                let max_ab = a.max(b);
                if max_ab > 0.0 { (b - a) / max_ab } else { 0.0 }
            })
            .collect()
    }

    /// Builds unit-L2 rows from a deterministic pseudo-random generator.
    fn unit_rows(n: usize, dim: usize, seed: u64) -> Mat<f64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };

        let mut mat = Mat::<f64>::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                mat[(i, j)] = next();
            }
            let mut norm = 0.0;
            for j in 0..dim {
                norm += mat[(i, j)] * mat[(i, j)];
            }
            let norm = norm.sqrt();
            for j in 0..dim {
                mat[(i, j)] /= norm;
            }
        }
        mat
    }

    /// The GEMM closed form must agree with the all-pairs oracle to machine
    /// precision. This is the load-bearing test for the whole shortcut.
    #[test]
    fn silhouette_matches_naive_reference() {
        let data = unit_rows(40, 7, 11);
        let labels: Vec<usize> = (0..40).map(|i| i % 4).collect();

        let (per_row, mean) = silhouette_cosine_unit(data.as_ref(), &labels, 4);
        let expected = silhouette_naive(data.as_ref(), &labels, 4);

        for (got, want) in per_row.iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-12);
        }
        let expected_mean = expected.iter().sum::<f64>() / expected.len() as f64;
        assert_relative_eq!(mean, expected_mean, epsilon = 1e-12);
    }

    /// Two tight, orthogonal groups are as separable as it gets, so the score
    /// sits near the top of the scale.
    #[test]
    fn silhouette_separated_clusters_score_high() {
        // Two directions 90 degrees apart, each with a small jitter.
        let data: Mat<f64> = Mat::from_fn(6, 2, |i, j| {
            let angle = if i < 3 {
                0.02 * i as f64
            } else {
                1.5 + 0.02 * i as f64
            };
            if j == 0 { angle.cos() } else { angle.sin() }
        });
        let labels = vec![0, 0, 0, 1, 1, 1];

        let (_, mean) = silhouette_cosine_unit(data.as_ref(), &labels, 2);
        assert!(mean > 0.9, "mean silhouette {mean}");
    }

    /// A singleton has no own-cluster distance to average, so convention puts it
    /// at exactly 0 rather than dividing by zero.
    #[test]
    fn silhouette_singleton_scores_zero() {
        let data = unit_rows(5, 4, 3);
        let labels = vec![0, 0, 0, 0, 1];

        let (per_row, _) = silhouette_cosine_unit(data.as_ref(), &labels, 2);
        assert_relative_eq!(per_row[4], 0.0, epsilon = 1e-12);
    }

    /// Fewer than two clusters carries no separation information, so the metric
    /// degenerates to zero instead of returning an infinity from the `b` term.
    #[test]
    fn silhouette_needs_two_clusters() {
        let data = unit_rows(6, 3, 5);

        let (per_row, mean) = silhouette_cosine_unit(data.as_ref(), &[0; 6], 1);
        assert!(per_row.iter().all(|&x| x == 0.0));
        assert_relative_eq!(mean, 0.0, epsilon = 1e-12);

        // Two declared clusters but only one populated behaves the same way.
        let (_, mean) = silhouette_cosine_unit(data.as_ref(), &[0; 6], 2);
        assert_relative_eq!(mean, 0.0, epsilon = 1e-12);
    }

    /// Labels outside the declared cluster count are a caller error.
    #[test]
    #[should_panic]
    fn silhouette_label_out_of_range() {
        let data = unit_rows(4, 3, 7);
        silhouette_cosine_unit(data.as_ref(), &[0, 1, 2, 0], 2);
    }

    /// Clusters that are declared but never populated must be skipped by the `b`
    /// minimum rather than contributing a spurious zero-size mean. This is the
    /// case consensus NMF hits whenever k-means leaves a cluster empty.
    #[test]
    fn silhouette_skips_declared_but_empty_clusters() {
        let data = unit_rows(12, 5, 23);
        // Clusters 2 and 4 declared, never used.
        let labels: Vec<usize> = (0..12).map(|i| if i < 6 { 0 } else { 3 }).collect();

        let (per_row, mean) = silhouette_cosine_unit(data.as_ref(), &labels, 5);
        let expected = silhouette_naive(data.as_ref(), &labels, 5);

        for (got, want) in per_row.iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-12);
        }
        assert!(mean.is_finite());
    }

    /// Two tight clusters a small angle apart, the regime the closed form is
    /// worst in: both `a` and `b` are tiny, so both suffer the cancellation and
    /// the ratio between them is what breaks. Near-orthogonal clusters do not
    /// exercise this, because there `b` is order 1 and swamps any error in `a`.
    fn two_close_tight_clusters(n: usize, dim: usize, jitter: f64, separation: f64) -> Mat<f64> {
        let u = unit_rows(1, dim, 91);
        let w = unit_rows(1, dim, 92);

        // Orthogonalise w against u, so `separation` is a genuine angle.
        let dot: f64 = (0..dim).map(|j| w[(0, j)] * u[(0, j)]).sum();
        let mut v: Vec<f64> = (0..dim).map(|j| w[(0, j)] - dot * u[(0, j)]).collect();
        let vnorm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= vnorm);

        let bases: Vec<Vec<f64>> = vec![
            (0..dim).map(|j| u[(0, j)]).collect(),
            (0..dim).map(|j| u[(0, j)] + separation * v[j]).collect(),
        ];

        let noise = unit_rows(n, dim, 93);
        let mut mat = Mat::<f64>::zeros(n, dim);
        for i in 0..n {
            let base = &bases[i / (n / 2)];
            for j in 0..dim {
                mat[(i, j)] = base[j] + jitter * noise[(i, j)];
            }
            let norm = (0..dim).map(|j| mat[(i, j)].powi(2)).sum::<f64>().sqrt();
            for j in 0..dim {
                mat[(i, j)] /= norm;
            }
        }
        mat
    }

    /// The closed form subtracts two numbers of size `|c|` to recover a value near
    /// zero, so it only survives in `f32` because the accumulation is forced to
    /// `f64`. Accumulating in `F` instead returns a mean around 0.44 here where
    /// the truth for the very same numbers is 0.99999.
    ///
    /// The reference is the f64 oracle run on the f32 values upcast losslessly, so
    /// the comparison isolates the arithmetic. Comparing against the original f64
    /// data instead would be unfair: at this jitter the true intra-cluster distance
    /// is 5e-11, far below the ~1e-7 noise floor of an f32 unit vector, so the two
    /// inputs genuinely describe different clusterings.
    #[test]
    fn silhouette_f32_matches_f64_on_identical_values() {
        let (n, dim) = (40, 8000);
        let f32_data: Mat<f32> = {
            let d = two_close_tight_clusters(n, dim, 1e-5, 3e-3);
            Mat::from_fn(n, dim, |i, j| d[(i, j)] as f32)
        };
        // f32 -> f64 is exact, so both paths see bit-identical values.
        let upcast: Mat<f64> = Mat::from_fn(n, dim, |i, j| f32_data[(i, j)] as f64);
        let labels: Vec<usize> = (0..n).map(|i| i / (n / 2)).collect();

        let (per_row_ref, mean_ref) = silhouette_cosine_unit(upcast.as_ref(), &labels, 2);
        let (per_row_f32, mean_f32) = silhouette_cosine_unit(f32_data.as_ref(), &labels, 2);

        // The oracle agrees with the closed form on these values too.
        // Loose by the standards of the other tests: with `a` at 1e-6 recovered
        // from terms of size 20, the two summation orders legitimately differ in
        // the last few bits.
        let oracle = silhouette_naive(upcast.as_ref(), &labels, 2);
        for (got, want) in per_row_ref.iter().zip(oracle.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-7);
        }

        for (got, want) in per_row_f32.iter().zip(per_row_ref.iter()) {
            assert_relative_eq!(*got as f64, want, epsilon = 1e-6);
        }
        assert_relative_eq!(mean_f32 as f64, mean_ref, epsilon = 1e-6);
    }

    /// The own-cluster mean must subtract the self term explicitly rather than
    /// folding it into the count, which would assume `<x_i, x_i> == 1`. Rows that
    /// are only approximately unit-norm break that assumption, and the resulting
    /// bias is systematic across every member of the cluster.
    #[test]
    fn silhouette_excludes_self_without_assuming_unit_norm() {
        let base = unit_rows(20, 6, 41);
        // Scale each row a little off unit, as f32 normalisation would leave it.
        let data: Mat<f64> = Mat::from_fn(20, 6, |i, j| {
            base[(i, j)] * (1.0 + 0.02 * ((i % 5) as f64 - 2.0))
        });
        let labels: Vec<usize> = (0..20).map(|i| i % 3).collect();

        let (per_row, _) = silhouette_cosine_unit(data.as_ref(), &labels, 3);
        // The oracle skips i == j outright, so it never relies on the self norm.
        let oracle = silhouette_naive(data.as_ref(), &labels, 3);

        for (got, want) in per_row.iter().zip(oracle.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-12);
        }
    }
}
