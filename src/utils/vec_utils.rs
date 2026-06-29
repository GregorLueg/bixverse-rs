//! Various utility functions for vectors: flattening them, min-max values,
//! cumulative sums and scaling them.

use num_traits::Float;
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rustc_hash::FxHashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::AddAssign;

use crate::core::math::vector_helpers::*;
use crate::prelude::*;

////////////////////
// Vector helpers //
////////////////////

/// Flatten a nested vector
///
/// ### Params
///
/// * `vec` - The vector to flatten
///
/// ### Returns
///
/// The flattened vector
pub fn flatten_vector<I, T>(vec: I) -> Vec<T>
where
    I: IntoIterator,
    I::Item: IntoIterator<Item = T>,
{
    vec.into_iter().flatten().collect()
}

/// Get the maximum value of an array
///
/// ### Params
///
/// * `arr` - The array of values
///
/// ### Returns
///
/// The maximum value found in the array
pub fn array_max<T>(arr: &[T]) -> T
where
    T: PartialOrd + Copy,
{
    let mut max_val = arr[0];
    for number in arr {
        if *number > max_val {
            max_val = *number
        }
    }
    max_val
}

/// Get the minimum value of an array
///
/// ### Params
///
/// * `arr` - The array of values
///
/// ### Returns
///
/// The minimum value found in the array
pub fn array_min<T>(arr: &[T]) -> T
where
    T: PartialOrd + Copy,
{
    let mut min_val = arr[0];
    for number in arr {
        if *number < min_val {
            min_val = *number
        }
    }
    min_val
}

/// Get the maximum and minimum value of an array
///
/// ### Params
///
/// * `arr` - The array of values
///
/// ### Returns
///
/// Tuple of `(max, min)` in the array
pub fn array_max_min<T>(arr: &[T]) -> (T, T)
where
    T: PartialOrd + Copy,
{
    let mut min_val = arr[0];
    let mut max_val = arr[0];
    for number in arr {
        if *number < min_val {
            min_val = *number
        }
        if *number > max_val {
            max_val = *number
        }
    }

    (max_val, min_val)
}

/// Get unique elements from a slice of any hashable, equatable numeric type.
///
/// ### Params
///
/// * `vec` - The slice of numerical values.
///
/// ### Returns
///
/// The unique elements of `vec` as a Vec.
pub fn unique<T>(vec: &[T]) -> Vec<T>
where
    T: Copy + Eq + Hash + Debug,
{
    let mut set = FxHashSet::default();
    vec.iter()
        .filter(|&&item| set.insert(item))
        .cloned()
        .collect()
}

/// Calculate the cumulative sum over a vector
///
/// ### Params
///
/// * `x` - The slice of numerical values
///
/// ### Returns
///
/// The cumulative sum over the vector.
pub fn cumsum<T>(x: &[T]) -> Vec<T>
where
    T: Float + AddAssign<T>,
{
    let mut sum = T::zero();
    x.iter()
        .map(|&val| {
            sum += val;
            sum
        })
        .collect()
}

/// Split a vector randomly into two chunks
///
/// Splits a vector randomly into two of [..x] and the other [x..]
///
/// ### Params
///
/// * `vec` - Slice of the vector you want to split
/// * `x` - Length of the first vector; the rest will be put into the second vector
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// A tuple of the pieces of the vector
pub fn split_vector_randomly<T>(vec: &[T], x: usize, seed: u64) -> (Vec<T>, Vec<T>)
where
    T: Clone,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shuffled = vec.to_vec();
    shuffled.shuffle(&mut rng);

    let (first_set, second_set) = shuffled.split_at(x);

    (first_set.to_vec(), second_set.to_vec())
}

/////////////
// Scalers //
/////////////

/// Min-max scale a vector to the range [0, 1].
///
/// ### Params
///
/// * `vec` - The slice of numerical values.
///
/// ### Returns
///
/// The scaled vector. Returns a zero vector if all values are identical.
pub fn min_max_scale<T>(vec: &[T]) -> Vec<T>
where
    T: Float,
{
    if vec.is_empty() {
        return Vec::new();
    }
    let (max_val, min_val) = array_max_min(vec);
    let range = max_val - min_val;
    if range == T::zero() {
        return vec![T::zero(); vec.len()];
    }
    vec.iter().map(|&x| (x - min_val) / range).collect()
}

/// Rank-normalise a vector by ranking values and dividing by n.
///
/// Produces values in (0, 1] with tie correction via averaged ranks.
///
/// ### Params
///
/// * `vec` - The slice of numerical values.
///
/// ### Returns
///
/// The rank-normalised vector.
pub fn rank_normalise<T>(vec: &[T]) -> Vec<T>
where
    T: Float,
{
    if vec.is_empty() {
        return Vec::new();
    }
    let n = T::from(vec.len()).unwrap();
    rank_vector(vec).into_iter().map(|r| r / n).collect()
}

/// Z-score normalise a vector (zero mean, unit variance).
///
/// ### Params
///
/// * `vec` - The slice of numerical values.
///
/// ### Returns
///
/// The normalised vector. Returns a zero vector if standard deviation is zero.
pub fn zscore_normalise<T>(vec: &[T]) -> Vec<T>
where
    T: Float,
{
    if vec.is_empty() {
        return Vec::new();
    }
    let n = T::from(vec.len()).unwrap();
    let mean = vec.iter().copied().fold(T::zero(), |acc, x| acc + x) / n;
    let variance = vec.iter().copied().fold(T::zero(), |acc, x| {
        let d = x - mean;
        acc + d * d
    }) / n;
    let std_dev = variance.sqrt();
    if std_dev == T::zero() {
        return vec![T::zero(); vec.len()];
    }
    vec.iter().map(|&x| (x - mean) / std_dev).collect()
}

/// Robust-scale a vector using median and MAD.
///
/// Computes `(x - median) / MAD` for each element. More resistant to outliers
/// than z-score normalisation.
///
/// ### Params
///
/// * `vec` - The slice of numerical values.
///
/// ### Returns
///
/// The scaled vector, or `None` if the input is empty. Returns a zero vector
/// if MAD is zero.
pub fn robust_scale<T>(vec: &[T]) -> Option<Vec<T>>
where
    T: BixverseFloat,
{
    if vec.is_empty() {
        return None;
    }
    let median_val = median(vec)?;
    let mad_val = mad(vec, Some(T::from_f64(MAD_SCALE).unwrap()))?;
    if mad_val == T::zero() {
        return Some(vec![T::zero(); vec.len()]);
    }
    Some(vec.iter().map(|&x| (x - median_val) / mad_val).collect())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec_approx_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| approx_eq(x, y))
    }

    #[test]
    fn test_flatten_vector_basic() {
        let nested = vec![vec![1, 2], vec![3], vec![4, 5]];
        assert_eq!(flatten_vector(nested), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_flatten_vector_empty_inner() {
        let nested: Vec<Vec<i32>> = vec![vec![], vec![1], vec![]];
        assert_eq!(flatten_vector(nested), vec![1]);
    }

    #[test]
    fn test_array_max() {
        assert_eq!(array_max(&[3, 1, 4, 1, 5, 9, 2, 6]), 9);
    }

    #[test]
    fn test_array_max_single() {
        assert_eq!(array_max(&[42]), 42);
    }

    #[test]
    fn test_array_min() {
        assert_eq!(array_min(&[3, 1, 4, 1, 5, 9, 2, 6]), 1);
    }

    #[test]
    fn test_array_min_single() {
        assert_eq!(array_min(&[42]), 42);
    }

    #[test]
    fn test_array_max_min() {
        let (max, min) = array_max_min(&[3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0]);
        assert!(approx_eq(max, 9.0));
        assert!(approx_eq(min, 1.0));
    }

    #[test]
    fn test_array_max_min_single() {
        let (max, min) = array_max_min(&[7]);
        assert_eq!(max, 7);
        assert_eq!(min, 7);
    }

    #[test]
    fn test_unique_removes_duplicates() {
        let mut result = unique(&[1, 2, 2, 3, 3, 3]);
        result.sort();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_unique_no_duplicates() {
        let mut result = unique(&[4, 2, 1]);
        result.sort();
        assert_eq!(result, vec![1, 2, 4]);
    }

    #[test]
    fn test_cumsum_basic() {
        let result = cumsum(&[1.0_f64, 2.0, 3.0, 4.0]);
        assert!(vec_approx_eq(&result, &[1.0, 3.0, 6.0, 10.0]));
    }

    #[test]
    fn test_cumsum_single() {
        let result = cumsum(&[5.0_f64]);
        assert!(vec_approx_eq(&result, &[5.0]));
    }

    #[test]
    fn test_split_vector_randomly_lengths() {
        let vec: Vec<i32> = (0..10).collect();
        let (a, b) = split_vector_randomly(&vec, 4, 42);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn test_split_vector_randomly_contains_all() {
        let vec: Vec<i32> = (0..10).collect();
        let (mut a, mut b) = split_vector_randomly(&vec, 4, 42);
        a.append(&mut b);
        a.sort();
        assert_eq!(a, vec);
    }

    #[test]
    fn test_split_vector_randomly_reproducible() {
        let vec: Vec<i32> = (0..10).collect();
        let (a1, _) = split_vector_randomly(&vec, 4, 99);
        let (a2, _) = split_vector_randomly(&vec, 4, 99);
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_min_max_scale_basic() {
        let result = min_max_scale(&[0.0_f64, 1.0, 2.0, 3.0, 4.0]);
        assert!(vec_approx_eq(&result, &[0.0, 0.25, 0.5, 0.75, 1.0]));
    }

    #[test]
    fn test_min_max_scale_uniform() {
        let result = min_max_scale(&[3.0_f64, 3.0, 3.0]);
        assert!(vec_approx_eq(&result, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_min_max_scale_empty() {
        let result = min_max_scale::<f64>(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rank_normalise_basic() {
        let result = rank_normalise(&[10.0_f64, 30.0, 20.0]);
        // ranks: 1, 3, 2 -> divided by 3
        assert!(vec_approx_eq(&result, &[1.0 / 3.0, 1.0, 2.0 / 3.0]));
    }

    #[test]
    fn test_rank_normalise_ties() {
        let result = rank_normalise(&[1.0_f64, 1.0, 3.0]);
        // ranks: avg(1,2)=1.5, 1.5, 3 -> divided by 3
        assert!(vec_approx_eq(&result, &[0.5, 0.5, 1.0]));
    }

    #[test]
    fn test_rank_normalise_empty() {
        let result = rank_normalise::<f64>(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_zscore_normalise_mean_zero() {
        let result = zscore_normalise(&[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!(approx_eq(mean, 0.0));
    }

    #[test]
    fn test_zscore_normalise_unit_variance() {
        let result = zscore_normalise(&[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        let variance: f64 =
            result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64;
        assert!(approx_eq(variance, 1.0));
    }

    #[test]
    fn test_zscore_normalise_uniform() {
        let result = zscore_normalise(&[5.0_f64, 5.0, 5.0]);
        assert!(vec_approx_eq(&result, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_zscore_normalise_empty() {
        let result = zscore_normalise::<f64>(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_robust_scale_basic() {
        // median = 3, MAD = median(|1,0,2,1,2|) = median of deviations from 3
        // values: [1, 3, 2, 5, 4] -> sorted [1,2,3,4,5], median = 3
        // deviations: [2, 0, 1, 2, 1] -> sorted [0,1,1,2,2], MAD = 1
        let result = robust_scale(&[1.0_f64, 3.0, 2.0, 5.0, 4.0]).unwrap();
        assert!(vec_approx_eq(&result, &[-2.0, 0.0, -1.0, 2.0, 1.0]));
    }

    #[test]
    fn test_robust_scale_uniform_mad() {
        let result = robust_scale(&[4.0_f64, 4.0, 4.0]).unwrap();
        assert!(vec_approx_eq(&result, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_robust_scale_empty() {
        let result = robust_scale::<f64>(&[]);
        assert!(result.is_none());
    }
}
