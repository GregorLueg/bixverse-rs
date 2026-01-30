use crate::prelude::*;

/// Simple linear regression
///
/// Fits y = b0 + b1 * x using ordinary least squares.
///
/// ### Params
///
/// * `x` - Independent variable
/// * `y` - Dependent variable
///
/// ### Returns
///
/// Tuple of (intercept, slope)
pub fn linear_regression<T>(x: &[T], y: &[T]) -> (T, T)
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = T::from(x.len()).unwrap();
    let sum_x: T = x.iter().cloned().sum();
    let sum_y: T = y.iter().cloned().sum();
    let sum_xy: T = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();
    let sum_xx: T = x.iter().map(|&xi| xi * xi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (intercept, slope)
}
