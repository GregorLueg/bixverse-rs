//! Various helper functions that work on vectors in Rust

use num_traits::Float;

use crate::prelude::BixverseFloat;

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
///
/// ### Results
///
/// The MAD of the slice.
pub fn mad<T>(x: &[T]) -> Option<T>
where
    T: BixverseFloat,
{
    if x.is_empty() {
        return None;
    }
    let median_val = median(x)?;
    let deviations: Vec<T> = x.iter().map(|&val| (val - median_val).abs()).collect();
    median(&deviations)
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(mad(&vec), Some(1.0));
    }

    #[test]
    fn test_standard_deviation() {
        let vec = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = standard_deviation(&vec);
        assert!((std - 2.1380899352).abs() < 1e-6);
    }
}
