//! Various helper functions that work on vectors in Rust

use num_traits::{Float, ToPrimitive};
use rayon::prelude::*;

use crate::prelude::{BixverseFloat, BixverseNumeric};

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

/// Sum of squares of a slice, accumulated in `f64`.
///
/// Accumulating in the storage type is the wrong default for anything this is
/// used for. A naive `f32` sum over `n` positive terms drifts once the running
/// total dwarfs the increment, and squaring first makes that happen sooner.
/// Measured on real NMF inputs: **2.5% low** over 1.5e7 dense entries and
/// **4.4% low** over 3e7 sparse non-zeros. That is not a rounding artefact, it is
/// percentage-level error in a quantity users read.
///
/// Rayon's reduction is a tree across chunks, which helps, but each chunk still
/// sums serially, so the accumulator type is what actually decides the result.
///
/// ### Params
///
/// * `values` - The values to square and sum
///
/// ### Returns
///
/// `sum_i values[i]^2` in `f64`.
pub fn sum_sq_f64<T>(values: &[T]) -> f64
where
    T: BixverseNumeric + ToPrimitive,
{
    values
        .par_iter()
        .with_min_len(10_000)
        .map(|&v| {
            // Infallible for every primitive numeric type, which is all this is
            // ever instantiated with.
            let x = v.to_f64().expect("numeric type does not convert to f64");
            x * x
        })
        .sum()
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

///////////////////
// Interpolation //
///////////////////

/// Linear interpolation at a single point, extrapolating past both ends.
///
/// Outside the knot range the nearest end segment is extended rather than
/// clamped, which is `scipy.interpolate.interp1d(..., fill_value="extrapolate")`
/// and not R's `approx(..., rule = 2)`. A single knot gives a constant.
///
/// ### Params
///
/// * `x` - Knot positions, strictly ascending and non-empty
/// * `y` - Knot values, the same length as `x`
/// * `at` - Where to evaluate
///
/// ### Returns
///
/// The interpolated (or extrapolated) value. An empty `x`, or a `y` shorter
/// than `x`, returns `at` unchanged rather than indexing out of bounds; both
/// are caller errors that [`crate::enrichment::blitzgsea::BlitzGseaNull`]
/// rejects up front, and neither is worth a `Result` on a path this hot.
pub fn interp_linear_at<T>(x: &[T], y: &[T], at: T) -> T
where
    T: BixverseFloat,
{
    let n = x.len();
    if n == 0 || y.len() < n {
        return at;
    }
    if n == 1 {
        return y[0];
    }

    // Index of the segment to use: the interior segment containing `at`, or the
    // nearest end segment when `at` falls outside the knots
    let upper = x.partition_point(|&k| k <= at);
    let j = upper.clamp(1, n - 1) - 1;

    let dx = x[j + 1] - x[j];
    if dx <= T::zero() {
        return y[j];
    }

    y[j] + (y[j + 1] - y[j]) * ((at - x[j]) / dx)
}

/// Linear interpolation on a log-spaced knot grid.
///
/// Brackets on `x` directly, which is valid because the logarithm is monotone,
/// then interpolates the position within that bracket in log space. On a grid
/// whose knots grow geometrically, straight linear interpolation gives the
/// widest segments the coarsest resolution; this spaces the resolution evenly
/// instead.
///
/// ### Params
///
/// * `x` - Knot positions, strictly ascending and strictly positive
/// * `y` - Knot values, the same length as `x`
/// * `at` - Where to evaluate, strictly positive
///
/// ### Returns
///
/// The interpolated (or extrapolated) value. Degenerate inputs are handled as
/// in [`interp_linear_at`]; a non-positive `at` or knot falls back to linear
/// interpolation, since the logarithm is undefined there.
pub fn interp_log_linear_at<T>(x: &[T], y: &[T], at: T) -> T
where
    T: BixverseFloat,
{
    let n = x.len();
    if n == 0 || y.len() < n {
        return at;
    }
    if n == 1 {
        return y[0];
    }

    let upper = x.partition_point(|&k| k <= at);
    let j = upper.clamp(1, n - 1) - 1;

    if at <= T::zero() || x[j] <= T::zero() || x[j + 1] <= T::zero() {
        return interp_linear_at(x, y, at);
    }

    let dx = x[j + 1].ln() - x[j].ln();
    if dx <= T::zero() {
        return y[j];
    }

    y[j] + (y[j + 1] - y[j]) * ((at.ln() - x[j].ln()) / dx)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// The sum of squares has to survive a large running total swamping small
    /// increments, which is the failure mode that put a 2.5% error into every
    /// relative NMF loss.
    ///
    /// Built so the gate can actually fail: the leading entry of `1e4` squares to
    /// `1e8`, where the f32 spacing is 8, so every subsequent `1.0` is lost
    /// outright and naive f32 returns `1e8`, missing the entire tail.
    ///
    /// The large value has to come *first*. Put it last and the naive sum
    /// accumulates the small terms perfectly well before the big one arrives, and
    /// the test proves nothing. That order dependence is the point: a naive sum's
    /// answer is a property of the traversal, not of the data.
    #[test]
    fn test_sum_sq_f64_survives_a_swamped_accumulator() {
        let n_ones = 1_000_000usize;
        let mut values = vec![1e4f32];
        values.extend(std::iter::repeat_n(1.0f32, n_ones));

        let exact = 1e8 + n_ones as f64;
        let got = sum_sq_f64(&values);
        assert!(
            (got - exact).abs() <= 1e-6 * exact,
            "f64 accumulation is off: got {got}, exact {exact}"
        );

        // The same reduction in f32, serially, as the NMF backends used to do.
        let naive = values
            .iter()
            .fold(0f32, |acc, &v| acc + v * v)
            .to_f64()
            .unwrap();
        assert!(
            (naive - exact).abs() > 0.005 * exact,
            "the f32 reference was supposed to be visibly wrong, so this test proves nothing"
        );
    }

    /// Integer storage types go through the same path, since the sparse layers
    /// can hold raw counts.
    #[test]
    fn test_sum_sq_f64_on_integer_storage() {
        let values: Vec<u32> = (1..=1000).collect();
        let exact: f64 = (1..=1000u64).map(|v| (v * v) as f64).sum();
        assert!((sum_sq_f64(&values) - exact).abs() < 1e-6);
    }

    /// Correlation hits plus or minus one on affine inputs and returns `None` on degenerate ones.
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

    /// Regression: a large common offset cancelled the variance and returned `None`.
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

    /// Tied values share the average of the ranks they span, and an empty input ranks to nothing.
    #[test]
    fn test_rank_vector() {
        let vec = vec![3.0, 1.0, 2.0, 3.0];
        let ranks = rank_vector(&vec);
        assert_eq!(ranks, vec![3.5, 1.0, 2.0, 3.5]);

        let empty: Vec<f64> = vec![];
        assert_eq!(rank_vector(&empty), Vec::<f64>::new());
    }

    /// Odd lengths take the middle value, even ones the midpoint, and an empty input gives `None`.
    #[test]
    fn test_median() {
        let vec_odd = vec![1.0, 3.0, 2.0];
        assert_eq!(median(&vec_odd), Some(2.0));

        let vec_even = vec![1.0, 4.0, 2.0, 3.0];
        assert_eq!(median(&vec_even), Some(2.5));

        let empty: Vec<f64> = vec![];
        assert_eq!(median(&empty), None);
    }

    /// MAD is the median of the absolute deviations from the median, unscaled without a constant.
    #[test]
    fn test_mad() {
        let vec = vec![1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0];
        assert_eq!(mad(&vec, None), Some(1.0));
    }

    /// The `n - 1` denominator: the population standard deviation here would be exactly 2.0.
    #[test]
    fn test_standard_deviation() {
        let vec = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = standard_deviation(&vec);
        assert!((std - 2.1380899352).abs() < 1e-6);
    }

    ///////////////////
    // Interpolation //
    ///////////////////

    #[test]
    fn test_interp_linear_hits_the_knots() {
        let x = [0.0, 1.0, 3.0, 6.0];
        let y = [10.0, 20.0, 0.0, 30.0];

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!((interp_linear_at(&x, &y, xi) - yi).abs() < 1e-12);
        }
    }

    #[test]
    fn test_interp_linear_interior() {
        let x = [0.0, 1.0, 3.0];
        let y = [0.0, 10.0, 30.0];

        assert!((interp_linear_at(&x, &y, 0.5) - 5.0).abs() < 1e-12);
        assert!((interp_linear_at(&x, &y, 2.0) - 20.0).abs() < 1e-12);
    }

    /// Past either end the nearest segment is extended, not clamped.
    #[test]
    fn test_interp_linear_extrapolates() {
        let x = [1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0];

        assert!((interp_linear_at(&x, &y, 0.0) - 0.0).abs() < 1e-12);
        assert!((interp_linear_at(&x, &y, 5.0) - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_interp_linear_single_knot_is_constant() {
        assert!((interp_linear_at(&[2.0], &[7.0], -100.0) - 7.0).abs() < 1e-12);
        assert!((interp_linear_at(&[2.0], &[7.0], 100.0) - 7.0).abs() < 1e-12);
    }

    /// Degenerate knots must not index out of bounds. `BlitzGseaNull` rejects
    /// these up front, but the functions are public and must not panic.
    #[test]
    fn test_interp_degenerate_knots() {
        assert!((interp_linear_at::<f64>(&[], &[], 3.0) - 3.0).abs() < 1e-12);
        assert!((interp_linear_at(&[1.0, 2.0, 3.0], &[1.0], 2.5) - 2.5).abs() < 1e-12);
        assert!((interp_log_linear_at::<f64>(&[], &[], 3.0) - 3.0).abs() < 1e-12);
        assert!((interp_log_linear_at(&[1.0, 2.0, 3.0], &[1.0], 2.5) - 2.5).abs() < 1e-12);
    }

    /// Log-space interpolation puts the midpoint of a geometric segment at the
    /// geometric mean, not the arithmetic one.
    #[test]
    fn test_interp_log_linear_geometric_midpoint() {
        let x = [1.0, 100.0];
        let y = [0.0, 1.0];

        assert!((interp_log_linear_at(&x, &y, 10.0) - 0.5).abs() < 1e-12);
        assert!((interp_log_linear_at(&x, &y, 1.0) - 0.0).abs() < 1e-12);
        assert!((interp_log_linear_at(&x, &y, 100.0) - 1.0).abs() < 1e-12);
        assert!((interp_log_linear_at(&x, &y, 10_000.0) - 2.0).abs() < 1e-12);
    }

    /// A non-positive query has no logarithm, so it falls back to linear.
    #[test]
    fn test_interp_log_linear_falls_back_on_non_positive() {
        let x = [1.0, 3.0];
        let y = [10.0, 30.0];

        let linear = interp_linear_at(&x, &y, -1.0);
        assert!((interp_log_linear_at(&x, &y, -1.0) - linear).abs() < 1e-12);
    }
}
