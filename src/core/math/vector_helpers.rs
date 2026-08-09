//! Various helper functions that work on vectors in Rust

use num_traits::Float;
use rayon::prelude::*;

use crate::prelude::BixverseFloat;

////////////
// Consts //
////////////

/// MAD scaling constant. R's `mad()` applies this factor by default.
pub const MAD_SCALE: f64 = 1.482_602_218_505_602;

///////////////
// Functions //
///////////////

/// Generate the rank of a vector with tie correction.
///
/// ### Params
///
/// * `vec` - The slice of numericals to rank.
///
/// ### Returns
///
/// The ranked vector (also f64)
pub fn rank_vector<T>(vec: &[T]) -> Vec<T>
where
    T: Float,
{
    let n = vec.len();
    if n == 0 {
        return Vec::new();
    }
    let mut indexed_values: Vec<(T, usize)> = vec
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed_values
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![T::zero(); n];
    let mut i = 0;
    while i < n {
        let current_value = indexed_values[i].0;
        let start = i;
        while i < n && indexed_values[i].0 == current_value {
            i += 1;
        }
        let avg_rank = (start + i + 1) as f64 / 2.0;
        let rank_value = T::from(avg_rank).unwrap();
        for j in start..i {
            ranks[indexed_values[j].1] = rank_value;
        }
    }
    ranks
}

/// Get the median
///
/// ### Params
///
/// * `x` - The slice for which to calculate the median for.
///
/// ### Results
///
/// The median (if the vector is not empty)
pub fn median<T>(x: &[T]) -> Option<T>
where
    T: BixverseFloat,
{
    if x.is_empty() {
        return None;
    }
    let mut data = x.to_vec();
    let len = data.len();
    if len.is_multiple_of(2) {
        let (_, median1, right) =
            data.select_nth_unstable_by(len / 2 - 1, |a, b| a.partial_cmp(b).unwrap());
        let median2 = right
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        Some((*median1 + *median2) / T::from_f64(2.0).unwrap())
    } else {
        let (_, median, _) = data.select_nth_unstable_by(len / 2, |a, b| a.partial_cmp(b).unwrap());
        Some(*median)
    }
}

/// Calculate the MAD
///
/// ### Params
///
/// * `x` - Slice for which to calculate the MAD for
/// * `scale` - Optional scaling factor. Pass `Some(1.4826)` for consistency with
///   the standard deviation under normality (R's default). `None` returns the raw MAD.
///
/// ### Results
///
/// The (optionally scaled) MAD of the slice.
pub fn mad<T>(x: &[T], scale: Option<T>) -> Option<T>
where
    T: BixverseFloat,
{
    if x.is_empty() {
        return None;
    }
    let median_val = median(x)?;
    let deviations: Vec<T> = x.iter().map(|&val| (val - median_val).abs()).collect();
    let raw = median(&deviations)?;
    Some(match scale {
        Some(k) => raw * k,
        None => raw,
    })
}

/// Standard deviation
///
/// ### Params
///
/// * `x` Slice of `f64`
///
/// ### Returns
///
/// The standard deviation
pub fn standard_deviation<T>(x: &[T]) -> T
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = T::from_usize(x.len()).unwrap();
    let mean: T = x.iter().copied().sum::<T>() / n;
    let variance = x.iter().map(|&val| (val - mean).powi(2)).sum::<T>() / (n - T::one());
    variance.sqrt()
}

/// Calculate the mean while removing NaNs
///
/// ### Params
///
/// * `x` - The slice of floats for which to calculate the mean (while
///   ignoring `NaN`'s)
///
/// ### Returns
///
/// The mean of the slice without `NaN`s
pub fn mean_nan<T>(x: &[T]) -> T
where
    T: BixverseFloat + std::iter::Sum,
{
    let finite: Vec<T> = x.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        T::nan()
    } else {
        finite.iter().copied().sum::<T>() / T::from_usize(finite.len()).unwrap()
    }
}

/// Linearly interpolated quantile over a pre-sorted slice.
///
/// ### Params
///
/// * `sorted` - Sorted slice of values.
/// * `q` - Quantile in `[0.0, 1.0]`.
///
/// ### Returns
///
/// The interpolated quantile value, or `0.0` if `sorted` is empty.
pub fn quantile_sorted<T>(sorted: &[T], q: T) -> T
where
    T: BixverseFloat,
{
    if sorted.is_empty() {
        return T::zero();
    }
    let pos = q.clamp(T::zero(), T::one()) * T::from_usize(sorted.len() - 1).unwrap();
    let lo = pos.floor().to_usize().unwrap();
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - T::from_usize(lo).unwrap();
    sorted[lo] * (T::one() - frac) + sorted[hi] * frac
}

/// Linearly interpolated quantile over an unsorted slice, matching numpy's
/// default.
///
/// ### Params
///
/// * `values` - Unsorted slice of values.
/// * `q` - Quantile in `[0.0, 1.0]`.
///
/// ### Returns
///
/// The interpolated quantile value, or `0.0` if `values` is empty.
pub fn quantile<T>(values: &[T], q: T) -> T
where
    T: BixverseFloat,
{
    let mut sorted: Vec<T> = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    quantile_sorted(&sorted, q)
}

/// Pearson correlation between two equal-length vectors.
///
/// Two passes, both rayon fold-reduce, so it stays cheap on the very long
/// vectors it is typically pointed at. Accumulation is in `f64` regardless of
/// `T`.
///
/// The means are subtracted before the moments are taken rather than
/// reconstructing the central moments from raw ones. The raw-moment form is one
/// pass cheaper but catastrophically cancels on offset data: `x[i] = 1e8 +
/// i * 1e-4` loses `var_x` entirely and the function then reports two identical
/// vectors as uncorrelated.
///
/// ### Params
///
/// * `x` - First vector.
/// * `y` - Second vector, same length as `x`.
///
/// ### Returns
///
/// The correlation, or `None` when the lengths differ, fewer than two elements
/// are supplied, or either vector is constant.
pub fn pearson_correlation<T>(x: &[T], y: &[T]) -> Option<f64>
where
    T: BixverseFloat + Sync,
{
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;

    let (sum_x, sum_y) = x
        .par_iter()
        .zip(y.par_iter())
        .fold(
            || (0.0f64, 0.0f64),
            |acc, (&a, &b)| {
                (
                    acc.0 + a.to_f64().unwrap_or(0.0),
                    acc.1 + b.to_f64().unwrap_or(0.0),
                )
            },
        )
        .reduce(|| (0.0f64, 0.0f64), |a, b| (a.0 + b.0, a.1 + b.1));

    let (mean_x, mean_y) = (sum_x / n, sum_y / n);

    let (cov, var_x, var_y) = x
        .par_iter()
        .zip(y.par_iter())
        .fold(
            || (0.0f64, 0.0f64, 0.0f64),
            |acc, (&a, &b)| {
                let da = a.to_f64().unwrap_or(0.0) - mean_x;
                let db = b.to_f64().unwrap_or(0.0) - mean_y;
                (acc.0 + da * db, acc.1 + da * da, acc.2 + db * db)
            },
        )
        .reduce(
            || (0.0f64, 0.0f64, 0.0f64),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    let denom = (var_x * var_y).sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        return None;
    }

    Some(cov / denom)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let perfect: Vec<f64> = x.iter().map(|v| 3.0 * v + 1.0).collect();
        let inverse: Vec<f64> = x.iter().map(|v| -2.0 * v).collect();

        assert!((pearson_correlation(&x, &perfect).unwrap() - 1.0).abs() < 1e-12);
        assert!((pearson_correlation(&x, &inverse).unwrap() + 1.0).abs() < 1e-12);

        // Constant vectors have no correlation to report.
        assert_eq!(pearson_correlation(&x, &[2.0; 5]), None);
        // Mismatched lengths and degenerate sizes are rejected.
        assert_eq!(pearson_correlation(&x, &[1.0, 2.0]), None);
        assert_eq!(pearson_correlation(&[1.0], &[1.0]), None);
    }

    #[test]
    fn test_pearson_correlation_survives_a_large_offset() {
        // Values sit at 1e8 with 1e-4 of spread, so the raw-moment formula
        // loses var_x completely and reports None for a vector against itself.
        let x: Vec<f64> = (0..1000).map(|i| 1e8 + i as f64 * 1e-4).collect();
        let y: Vec<f64> = x.iter().map(|v| -2.0 * v + 5.0).collect();

        let self_corr = pearson_correlation(&x, &x).expect("identical vectors correlate at one");
        assert!(
            (self_corr - 1.0).abs() < 1e-9,
            "self correlation is {self_corr}"
        );

        let inverse = pearson_correlation(&x, &y).expect("an affine map correlates at minus one");
        assert!(
            (inverse + 1.0).abs() < 1e-9,
            "inverse correlation is {inverse}"
        );
    }

    #[test]
    fn test_rank_vector() {
        let vec = vec![3.0, 1.0, 2.0, 3.0];
        // Ranks should account for ties:
        // sorted: 1.0 (idx 1), 2.0 (idx 2), 3.0 (idx 0), 3.0 (idx 3)
        // ranks: 1.0 -> 1.0, 2.0 -> 2.0, 3.0 -> 3.5, 3.0 -> 3.5
        let ranks = rank_vector(&vec);
        assert_eq!(ranks, vec![3.5, 1.0, 2.0, 3.5]);

        let empty: Vec<f64> = vec![];
        assert_eq!(rank_vector(&empty), Vec::<f64>::new());
    }

    #[test]
    fn test_median() {
        let vec_odd = vec![1.0, 3.0, 2.0];
        assert_eq!(median(&vec_odd), Some(2.0));

        let vec_even = vec![1.0, 4.0, 2.0, 3.0];
        assert_eq!(median(&vec_even), Some(2.5));

        let empty: Vec<f64> = vec![];
        assert_eq!(median(&empty), None);
    }

    #[test]
    fn test_mad() {
        let vec = vec![1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0];
        // Median is 2.0
        // Absolute deviations: [1.0, 1.0, 0.0, 0.0, 2.0, 4.0, 7.0]
        // Sorted deviations: [0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 7.0]
        // MAD (median of deviations) is 1.0
        assert_eq!(mad(&vec, None), Some(1.0));
    }

    #[test]
    fn test_standard_deviation() {
        let vec = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = standard_deviation(&vec);
        assert!((std - 2.1380899352).abs() < 1e-6);
    }
}
