use rayon::prelude::*;

use crate::prelude::*;

/// Structure to store the Loess results
///
/// ### Params
///
/// * `fitted_vals` - The values fitted by the function.
/// * `residuals` - The residuals.
/// * `valid_indices` - Which index positions were valid.
#[derive(Debug, Clone)]
pub struct LoessRes<T> {
    pub fitted_vals: Vec<T>,
    pub residuals: Vec<T>,
    pub valid_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum LoessFunc {
    /// Linear version of the Loess function
    Linear,
    /// Quadratic version of the Loess function
    Quadratic,
}

/// Parse the type of Loess function
///
/// ### Params
///
/// * `option` - Usize defining the degrees of freedom
///
/// ### Return
///
/// The option of the `LoessFunc`
pub fn parse_loess_fun(option: &usize) -> Option<LoessFunc> {
    match option {
        1 => Some(LoessFunc::Linear),
        2 => Some(LoessFunc::Quadratic),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Solve a 3 x 3 system of linear equations
///
/// ### Params
///
/// * `a` - The coefficient matrix
/// * `b` - The right-hand side vector
///
/// ### Return
///
/// The solution vector if the system is solvable, otherwise None
fn solve_3x3_system<T>(a: &[[T; 3]; 3], b: &[T; 3]) -> Option<[T; 3]>
where
    T: BixverseFloat,
{
    let mut matrix = *a;
    let mut rhs = *b;

    for i in 0..3 {
        let mut pivot_row = i;
        for j in (i + 1)..3 {
            if matrix[j][i].abs() > matrix[pivot_row][i].abs() {
                pivot_row = j;
            }
        }

        if pivot_row != i {
            matrix.swap(i, pivot_row);
            rhs.swap(i, pivot_row);
        }

        if matrix[i][i].abs() < T::from_f64(1e-12).unwrap() {
            return None;
        }

        for j in (i + 1)..3 {
            let factor = matrix[j][i] / matrix[i][i];
            for k in i..3 {
                matrix[j][k] -= factor * matrix[i][k];
            }
            rhs[j] -= factor * rhs[i];
        }
    }

    let mut solution = [T::zero(); 3];
    for i in (0..3).rev() {
        solution[i] = rhs[i];
        for j in (i + 1)..3 {
            solution[i] -= matrix[i][j] * solution[j];
        }
        solution[i] /= matrix[i][i];
    }

    Some(solution)
}

///////////
// Loess //
///////////

/// Struct for performing a loess regression
///
/// ### Fields
///
/// * `span` - The span of the loess regression
/// * `loess_type` - The type of loess regression to perform
pub struct LoessRegression<T> {
    span: T,
    loess_type: LoessFunc,
}

impl<T> LoessRegression<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Generate a new instance of the Loess regression
    ///
    /// ### Params
    ///
    /// * `span` -
    /// * `degree` -
    ///
    /// ### Return
    ///
    /// Initialised class
    pub fn new(span: T, degree: usize) -> Self {
        assert!(
            span > T::zero() && span <= T::one(),
            "Span must be between 0 and 1"
        );
        assert!(
            degree == 1 || degree == 2,
            "Only linear (1) and quadratic (2) supported"
        );

        let loess_type: LoessFunc = parse_loess_fun(&degree).unwrap();

        Self { span, loess_type }
    }

    /// Fit the loess function (for a two variable system)
    ///
    /// ### Params
    ///
    /// * `x` - The response variable
    /// * `y` - The predictor variable
    ///
    /// ### Returns
    ///
    /// The fit results in form of a `LoessRes`
    pub fn fit(&self, x: &[T], y: &[T]) -> LoessRes<T> {
        let n = x.len();
        let valid: Vec<(usize, T, T)> = x
            .iter()
            .zip(y.iter())
            .enumerate()
            .filter_map(|(i, (&x, &y))| {
                if x.is_finite() && y.is_finite() {
                    Some((i, x, y))
                } else {
                    None
                }
            })
            .collect();

        if valid.is_empty() {
            return LoessRes {
                fitted_vals: vec![T::zero(); n],
                residuals: vec![T::zero(); n],
                valid_indices: Vec::new(),
            };
        }

        let mut sorted_points = valid;
        sorted_points.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n_valid = sorted_points.len();
        let no_neighbours = (T::from_usize(n_valid).unwrap() * self.span)
            .max(T::one())
            .to_usize()
            .unwrap();

        let mut fitted_values = vec![T::zero(); n];
        let mut residuals = vec![T::zero(); n];

        let results: Vec<_> = sorted_points
            .par_iter()
            .map(|(orig_idx, x_val, y_val)| {
                let fitted_val = self.fit_point(&sorted_points, *x_val, no_neighbours);
                (*orig_idx, fitted_val, *y_val - fitted_val)
            })
            .collect();

        for (orig_idx, fitted_val, residual) in results {
            fitted_values[orig_idx] = fitted_val;
            residuals[orig_idx] = residual;
        }

        LoessRes {
            fitted_vals: fitted_values,
            residuals,
            valid_indices: sorted_points.iter().map(|(idx, _, _)| *idx).collect(),
        }
    }

    /// Fits a given point
    ///
    /// ### Params
    ///
    /// * `sorted_points` - A slice of tuples of the position, x and y value
    /// * `target_x` - The target value
    /// * `k` - Number of neighbours
    ///
    /// ### Returns
    ///
    /// The fitted value
    fn fit_point(&self, sorted_points: &[(usize, T, T)], target_x: T, k: usize) -> T {
        let neighbors = self.find_neighbors_binary(sorted_points, target_x, k);

        if neighbors.is_empty() {
            return T::zero();
        }

        let max_dist = neighbors
            .iter()
            .map(|&i| (sorted_points[i].1 - target_x).abs())
            .fold(T::zero(), T::max);

        if max_dist == T::zero() {
            let sum: T = neighbors.iter().map(|&i| sorted_points[i].2).sum();
            return sum / T::from_usize(neighbors.len()).unwrap();
        }

        let mut x_vals = [T::zero(); 64];
        let mut y_vals = [T::zero(); 64];
        let mut weights = [T::zero(); 64];

        let n_neighbors = neighbors.len().min(64);
        let inv_max_dist = T::one() / max_dist;

        for (i, &idx) in neighbors.iter().take(n_neighbors).enumerate() {
            let (_, nx, ny) = sorted_points[idx];
            x_vals[i] = nx;
            y_vals[i] = ny;
            weights[i] = self.tricube_weight((nx - target_x).abs() * inv_max_dist);
        }

        self.weighted_polynomial_fit(
            target_x,
            &x_vals[..n_neighbors],
            &y_vals[..n_neighbors],
            &weights[..n_neighbors],
        )
    }

    /// Find neighbours via binary search
    ///
    /// Uses binary search under the hood for speed.
    ///
    /// ### Params
    ///
    /// `sorted_points` - A slice of tuples of the position, x and y value
    /// `target_x` - The target value
    /// `k` - Number of neighbour
    fn find_neighbors_binary(
        &self,
        sorted_points: &[(usize, T, T)],
        target_x: T,
        k: usize,
    ) -> Vec<usize> {
        let n = sorted_points.len();
        if k >= n {
            return (0..n).collect();
        }

        let insert_pos = sorted_points
            .binary_search_by(|probe| probe.1.partial_cmp(&target_x).unwrap())
            .unwrap_or_else(|pos| pos);

        let mut l = insert_pos;
        let mut r = insert_pos;
        let mut neighbors = Vec::with_capacity(k);

        for _ in 0..k {
            let left_dist = if l > 0 {
                (sorted_points[l - 1].1 - target_x).abs()
            } else {
                T::infinity()
            };
            let right_dist = if r < n {
                (sorted_points[r].1 - target_x).abs()
            } else {
                T::infinity()
            };

            if left_dist <= right_dist && l > 0 {
                l -= 1;
                neighbors.push(l);
            } else if r < n {
                neighbors.push(r);
                r += 1;
            } else {
                break;
            }
        }

        neighbors
    }

    /// Tricube weight function: (1 - |u|³)³ for |u| < 1, 0 otherwise
    ///
    /// ### Params
    ///
    /// * `u` - The value
    ///
    /// ### Returns
    ///
    /// The tricube weight
    #[inline]
    fn tricube_weight(&self, u: T) -> T {
        if u >= T::one() {
            T::zero()
        } else {
            let temp = T::one() - u * u * u;
            temp * temp * temp
        }
    }

    /// Weighted fit helper
    ///
    /// ### Params
    ///
    /// * `target_x` - The target variable at this point
    /// * `x` - Slice of the predictor variable
    /// * `y` - Slice of the response variable
    /// * `w` - Slice of the weights
    ///
    /// ### Returns
    ///
    /// Value at this position
    fn weighted_polynomial_fit(&self, target_x: T, x: &[T], y: &[T], w: &[T]) -> T {
        match self.loess_type {
            LoessFunc::Linear => self.weighted_linear_fit(target_x, x, y, w),
            LoessFunc::Quadratic => self.weighted_quadratic_fit(target_x, x, y, w),
        }
    }

    /// Helper function for linear fits
    ///
    /// ### Params
    ///
    /// * `target_x` - The target variable at this point
    /// * `x` - Slice of the predictor variable
    /// * `y` - Slice of the response variable
    /// * `w` - Slice of the weights
    ///
    /// ### Return
    ///
    /// Value at this position with a linear regression
    fn weighted_linear_fit(&self, target_x: T, x: &[T], y: &[T], w: &[T]) -> T {
        let mut w_sum = T::zero();
        let mut wx_sum = T::zero();
        let mut wy_sum = T::zero();
        let mut wxx_sum = T::zero();
        let mut wxy_sum = T::zero();

        for i in 0..x.len() {
            let wi = w[i];
            let xi = x[i];
            let yi = y[i];

            w_sum += wi;
            wx_sum += wi * xi;
            wy_sum += wi * yi;
            wxx_sum += wi * xi * xi;
            wxy_sum += wi * xi * yi;
        }

        if w_sum == T::zero() {
            let sum: T = y.iter().copied().sum();
            return sum / T::from_usize(y.len()).unwrap();
        }

        let inv_w_sum = T::one() / w_sum;
        let x_mean = wx_sum * inv_w_sum;
        let y_mean = wy_sum * inv_w_sum;

        let numerator = wxy_sum - w_sum * x_mean * y_mean;
        let denominator = wxx_sum - w_sum * x_mean * x_mean;

        if denominator.abs() < T::from_f64(1e-12).unwrap() {
            return y_mean;
        }

        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;

        intercept + slope * target_x
    }

    /// Helper function for quadratic fits
    ///
    /// ### Params
    ///
    /// * `target_x` - The target variable at this point
    /// * `x` - Slice of the predictor variable
    /// * `y` - Slice of the response variable
    /// * `w` - Slice of the weights
    ///
    /// ### Return
    ///
    /// Value at this position with a quadratic regression
    fn weighted_quadratic_fit(&self, target_x: T, x: &[T], y: &[T], w: &[T]) -> T {
        if x.len() < 3 {
            return self.weighted_linear_fit(target_x, x, y, w);
        }

        let mut a = [[T::zero(); 3]; 3];
        let mut b = [T::zero(); 3];

        for i in 0..x.len() {
            let xi = x[i];
            let yi = y[i];
            let wi = w[i];
            let xi2 = xi * xi;

            a[0][0] += wi;
            a[0][1] += wi * xi;
            a[0][2] += wi * xi2;
            a[1][1] += wi * xi2;
            a[1][2] += wi * xi * xi2;
            a[2][2] += wi * xi2 * xi2;

            b[0] += wi * yi;
            b[1] += wi * xi * yi;
            b[2] += wi * xi2 * yi;
        }

        a[1][0] = a[0][1];
        a[2][0] = a[0][2];
        a[2][1] = a[1][2];

        match solve_3x3_system(&a, &b) {
            Some(coeffs) => coeffs[0] + coeffs[1] * target_x + coeffs[2] * target_x * target_x,
            None => self.weighted_linear_fit(target_x, x, y, w),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-8, "{} != {}", a, b);
    }

    #[test]
    fn test_loess_construction() {
        let loess: LoessRegression<f64> = LoessRegression::new(0.5, 1);
        // Should not panic
        let _res = loess.fit(&[1.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "Span must be between 0 and 1")]
    fn test_loess_invalid_span_high() {
        let _loess: LoessRegression<f64> = LoessRegression::new(1.5, 1);
    }

    #[test]
    #[should_panic(expected = "Span must be between 0 and 1")]
    fn test_loess_invalid_span_zero() {
        let _loess: LoessRegression<f64> = LoessRegression::new(0.0, 1);
    }

    #[test]
    #[should_panic(expected = "Only linear (1) and quadratic (2) supported")]
    fn test_loess_invalid_degree() {
        let _loess: LoessRegression<f64> = LoessRegression::new(0.5, 3);
    }

    #[test]
    fn test_perfect_linear_fit() {
        // Data: y = 2x + 1
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();

        // Span 0.5, Linear (degree 1)
        let loess = LoessRegression::new(0.5, 1);
        let res = loess.fit(&x, &y);

        assert_eq!(res.fitted_vals.len(), 10);

        for (i, &val) in res.fitted_vals.iter().enumerate() {
            assert_approx_eq(val, y[i]);
            assert_approx_eq(res.residuals[i], 0.0);
        }
    }

    #[test]
    fn test_perfect_quadratic_fit() {
        // Data: y = x^2
        let x: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();

        // Span 1.0 (use all points), Quadratic (degree 2)
        let loess = LoessRegression::new(1.0, 2);
        let res = loess.fit(&x, &y);

        for (i, &val) in res.fitted_vals.iter().enumerate() {
            // Should be very close to exact parabola
            assert_approx_eq(val, y[i]);
        }
    }

    #[test]
    fn test_quadratic_fallback_on_small_sample() {
        // Only 2 points provided. Quadratic fit needs at least 3.
        // Should fallback to linear gracefully.
        let x = vec![1.0, 2.0];
        let y = vec![2.0, 4.0]; // y = 2x

        let loess = LoessRegression::new(1.0, 2); // Request quadratic
        let res = loess.fit(&x, &y);

        assert_approx_eq(res.fitted_vals[0], 2.0);
        assert_approx_eq(res.fitted_vals[1], 4.0);
    }

    #[test]
    fn test_handling_nans_and_order() {
        // Input contains NaNs.
        // Indices: 0(valid), 1(NaN), 2(valid)
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, f64::NAN, 6.0];

        let loess = LoessRegression::new(1.0, 1);
        let res = loess.fit(&x, &y);

        // Check vectors length
        assert_eq!(res.fitted_vals.len(), 3);

        // Index 1 should be 0.0 (or whatever default, logic sets it to 0.0 for invalid)
        assert_eq!(res.fitted_vals[1], 0.0);

        // Indices 0 and 2 should be fitted.
        // With only points (1,2) and (3,6), line is y = 2x.
        assert_approx_eq(res.fitted_vals[0], 2.0);
        assert_approx_eq(res.fitted_vals[2], 6.0);

        // Valid indices check
        assert_eq!(res.valid_indices, vec![0, 2]);
    }

    #[test]
    fn test_empty_input() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        let loess = LoessRegression::new(0.5, 1);
        let res = loess.fit(&x, &y);

        assert!(res.fitted_vals.is_empty());
        assert!(res.valid_indices.is_empty());
    }
}
