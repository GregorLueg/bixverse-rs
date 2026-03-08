//! Various utility functions fro vectors

use num_traits::Float;
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rustc_hash::FxHashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::AddAssign;

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
