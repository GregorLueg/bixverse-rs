//! Locally Estimated Scatterplot Smoothing (LOESS) implementation.
//!
//! Two evaluation surfaces, matching R's `loess()`. [`LoessSurface::Direct`]
//! runs a local regression at every point. [`LoessSurface::Interpolate`], the
//! default in R and here, runs it at a small set of anchor vertices and joins
//! them with a cubic Hermite spline. The anchor count is a function of the span
//! rather than of the data size, so the interpolating surface is effectively
//! linear in `n` where the direct one is quadratic.
//!
//! Local fits are centred on the target and scaled to the neighbourhood width,
//! which keeps the normal equations well conditioned and hands back the
//! derivative the spline needs for free. Moments accumulate in `f64` whatever
//! `T` is, since the fourth moment of a weighted sum loses `f32` fast.

use rayon::prelude::*;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Fraction of the neighbourhood size used as the anchor cell budget.
///
/// R's `loess.control(cell = 0.2)`. The kd-tree splits until a cell holds no
/// more than `span * n * cell` points, so the anchor count lands near
/// `1 / (span * cell)` and is independent of `n`. At the HVG default of
/// `span = 0.3` that is 17 cells whether there are two thousand genes or two
/// hundred thousand.
const LOESS_CELL: f64 = 0.2;

/// Point count below which [`LoessSurface::Interpolate`] defers to
/// [`LoessSurface::Direct`].
///
/// Interpolation only pays when the anchors are far fewer than the points. On
/// a small input the anchors approach the data and the spline buys nothing but
/// approximation error, so fit everything exactly instead.
const LOESS_INTERPOLATE_MIN_POINTS: usize = 256;

/// Relative pivot tolerance for the local normal equations.
///
/// The system is normalised by its zeroth moment before solving, so every
/// entry is `O(1)` and this is a meaningful absolute threshold.
const LOESS_PIVOT_TOLERANCE: f64 = 1e-12;

/////////////
// Results //
/////////////

/// Structure to store the Loess results
#[derive(Debug, Clone)]
pub struct LoessRes<T> {
    /// The values fitted by the function. Non-finite input points keep `0.0`.
    pub fitted_vals: Vec<T>,
    /// The residuals. Non-finite input points keep `0.0`.
    pub residuals: Vec<T>,
    /// Which index positions were valid, in ascending order of `x`.
    pub valid_indices: Vec<usize>,
}

///////////
// Enums //
///////////

/// Which LoessFunction to use
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LoessFunc {
    /// Linear version of the Loess function (default)
    #[default]
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

/// How the fitted surface is evaluated away from the anchor points
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LoessSurface {
    /// Fit at anchor vertices and join them with a cubic Hermite spline.
    ///
    /// R's default, and what Seurat therefore gets. Cost is
    /// `O(n log n + n / cell)` rather than `O(span * n^2)`.
    #[default]
    Interpolate,
    /// Run the local regression at every point.
    ///
    /// Exact, and the reference the interpolating surface is tested against.
    /// Quadratic in `n`, so it gets painful above a few thousand points.
    Direct,
}

/// Parse the loess surface
///
/// ### Params
///
/// * `s` - The surface as a string
///
/// ### Returns
///
/// Option of the `LoessSurface`
pub fn parse_loess_surface(s: &str) -> Option<LoessSurface> {
    match s.to_lowercase().as_str() {
        "interpolate" => Some(LoessSurface::Interpolate),
        "direct" => Some(LoessSurface::Direct),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Solve a symmetric 3 x 3 system by Gaussian elimination with partial pivoting
///
/// The caller normalises the system so that every entry is `O(1)`, which makes
/// [`LOESS_PIVOT_TOLERANCE`] a meaningful singularity guard.
///
/// ### Params
///
/// * `a` - The coefficient matrix
/// * `b` - The right-hand side vector
///
/// ### Return
///
/// The solution vector if the system is solvable, otherwise `None`
fn solve_3x3_system(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
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

        if matrix[i][i].abs() < LOESS_PIVOT_TOLERANCE {
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

    let mut solution = [0f64; 3];
    for i in (0..3).rev() {
        solution[i] = rhs[i];
        for j in (i + 1)..3 {
            solution[i] -= matrix[i][j] * solution[j];
        }
        solution[i] /= matrix[i][i];
    }

    Some(solution)
}

/// Start index of the `q` nearest neighbours of `target` in a sorted slice.
///
/// The `q` nearest of a sorted sequence are contiguous, so the neighbourhood is
/// the window `[start, start + q)`. Shifting the window right swaps `xs[start]`
/// out for `xs[start + q]`, and that trade stops being worthwhile exactly once
/// the entering point is no closer than the leaving one. The predicate is
/// monotone in `start`, so binary search finds the boundary and no state
/// carries between targets, which keeps the sweep parallel.
///
/// ### Params
///
/// * `xs` - Predictor values, ascending
/// * `q` - Neighbourhood size, clamped to `xs.len()` by the caller
/// * `target` - Point the neighbourhood is centred on
///
/// ### Returns
///
/// The window start, in `0..=xs.len() - q`.
fn window_start<T: BixverseFloat>(xs: &[T], q: usize, target: T) -> usize {
    let n = xs.len();
    if q >= n {
        return 0;
    }

    let (mut lo, mut hi) = (0usize, n - q);
    while lo < hi {
        // `mid < hi <= n - q`, so `mid + q` stays in bounds
        let mid = lo + (hi - lo) / 2;
        let leaving = (target - xs[mid]).abs();
        let entering = (xs[mid + q] - target).abs();
        if entering < leaving {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    lo
}

/// Weighted local polynomial fit, centred and scaled on the target.
///
/// Regressing on `u = (x - target) / d_max` rather than on `x` does three
/// things: the fitted value at the target is the intercept, the first
/// coefficient is the derivative in `u` (so the spline gets its slope without a
/// second fit), and every moment lands in `[-1, 1]` so the normal equations
/// stay conditioned even at degree 2.
///
/// ### Params
///
/// * `xs` - Predictor values, ascending
/// * `ys` - Response values in the same order
/// * `start` - First index of the neighbourhood
/// * `q` - Neighbourhood size
/// * `target` - Point to fit at
/// * `degree` - Local polynomial degree
///
/// ### Returns
///
/// A tuple of `(fitted value, derivative with respect to x)`.
fn local_fit<T: BixverseFloat>(
    xs: &[T],
    ys: &[T],
    start: usize,
    q: usize,
    target: T,
    degree: LoessFunc,
) -> (T, T) {
    let end = (start + q).min(xs.len());
    if start >= end {
        return (T::zero(), T::zero());
    }

    let to_f64 = |v: T| v.to_f64().unwrap_or(0.0);
    let from_f64 = |v: f64| T::from_f64(v).unwrap_or_else(T::zero);

    let target_f = to_f64(target);
    let mean_of_window = || {
        let count = (end - start) as f64;
        ys[start..end].iter().map(|&v| to_f64(v)).sum::<f64>() / count
    };

    // The window is contiguous and sorted, so the farthest neighbour is one of
    // the two ends.
    let d_max = (target_f - to_f64(xs[start])).max(to_f64(xs[end - 1]) - target_f);
    // Non-finite inputs are filtered before this point, so d_max cannot be NaN
    // and the plain comparison is enough.
    if d_max <= 0.0 {
        return (from_f64(mean_of_window()), T::zero());
    }

    let inv_d_max = 1.0 / d_max;
    let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut t0, mut t1, mut t2) = (0f64, 0f64, 0f64);

    for i in start..end {
        let u = (to_f64(xs[i]) - target_f) * inv_d_max;
        let distance = u.abs();
        if distance >= 1.0 {
            continue;
        }

        let tricube = 1.0 - distance * distance * distance;
        let w = tricube * tricube * tricube;
        let y = to_f64(ys[i]);

        let wu = w * u;
        let wuu = wu * u;

        s0 += w;
        s1 += wu;
        s2 += wuu;
        s3 += wuu * u;
        s4 += wuu * u * u;

        t0 += w * y;
        t1 += wu * y;
        t2 += wuu * y;
    }

    if s0 <= 0.0 {
        return (from_f64(mean_of_window()), T::zero());
    }

    // Normalising by the zeroth moment leaves every entry O(1), which is what
    // makes the absolute pivot tolerance in `solve_3x3_system` meaningful.
    let inv_s0 = 1.0 / s0;
    let (n1, n2, n3, n4) = (s1 * inv_s0, s2 * inv_s0, s3 * inv_s0, s4 * inv_s0);
    let (r0, r1, r2) = (t0 * inv_s0, t1 * inv_s0, t2 * inv_s0);

    let quadratic = degree == LoessFunc::Quadratic && (end - start) >= 3;
    if quadratic {
        let a = [[1.0, n1, n2], [n1, n2, n3], [n2, n3, n4]];
        if let Some(coefficients) = solve_3x3_system(&a, &[r0, r1, r2]) {
            return (
                from_f64(coefficients[0]),
                from_f64(coefficients[1] * inv_d_max),
            );
        }
    }

    // Degree 1, and the fallback whenever the quadratic system is singular
    let determinant = n2 - n1 * n1;
    if determinant.abs() < LOESS_PIVOT_TOLERANCE {
        return (from_f64(r0), T::zero());
    }

    let slope = (r1 - n1 * r0) / determinant;
    let intercept = r0 - slope * n1;

    (from_f64(intercept), from_f64(slope * inv_d_max))
}

/// Anchor vertex indices for the interpolating surface.
///
/// Equal-count cells, which is what R's median-splitting kd-tree converges to
/// in one dimension. The first and last vertices are the data extremes, so
/// every point falls inside the spline and nothing is extrapolated.
///
/// ### Params
///
/// * `xs` - Predictor values, ascending
/// * `q` - Neighbourhood size
///
/// ### Returns
///
/// Ascending indices into `xs`, with duplicate `x` values collapsed.
fn vertex_indices<T: BixverseFloat>(xs: &[T], q: usize) -> Vec<usize> {
    let n = xs.len();
    let bucket = ((q as f64) * LOESS_CELL).floor().max(1.0) as usize;
    let n_cells = n.div_ceil(bucket).max(1);

    let mut vertices: Vec<usize> = Vec::with_capacity(n_cells + 1);
    for j in 0..=n_cells {
        let idx = (j * (n - 1)) / n_cells;
        match vertices.last() {
            // A tie in x would give the spline a zero-width interval
            Some(&previous) if xs[previous] == xs[idx] => continue,
            _ => vertices.push(idx),
        }
    }

    vertices
}

/// Evaluate a cubic Hermite spline at `x`.
///
/// ### Params
///
/// * `knots` - Ascending knot positions, at least one
/// * `values` - Fitted value at each knot
/// * `slopes` - Derivative at each knot
/// * `x` - Where to evaluate
///
/// ### Returns
///
/// The interpolated value. Outside the knot range the nearest end segment is
/// extended linearly, which only matters for numerically equal endpoints.
fn hermite_eval<T: BixverseFloat>(knots: &[T], values: &[T], slopes: &[T], x: T) -> T {
    let m = knots.len();
    if m == 1 {
        return values[0];
    }

    if x <= knots[0] {
        return values[0] + slopes[0] * (x - knots[0]);
    }
    if x >= knots[m - 1] {
        return values[m - 1] + slopes[m - 1] * (x - knots[m - 1]);
    }

    // Index of the segment containing x, i.e. the last knot at or below it
    let upper = knots.partition_point(|&k| k <= x);
    let j = upper.saturating_sub(1).min(m - 2);

    let h = knots[j + 1] - knots[j];
    if h <= T::zero() {
        return values[j];
    }

    let t = (x - knots[j]) / h;
    let t2 = t * t;
    let t3 = t2 * t;

    let two = T::one() + T::one();
    let three = two + T::one();

    let h00 = two * t3 - three * t2 + T::one();
    let h10 = t3 - two * t2 + t;
    let h01 = three * t2 - two * t3;
    let h11 = t3 - t2;

    h00 * values[j] + h10 * h * slopes[j] + h01 * values[j + 1] + h11 * h * slopes[j + 1]
}

///////////
// Loess //
///////////

/// Struct for performing a loess regression
pub struct LoessRegression<T> {
    /// The span of the loess regression
    span: T,
    /// The type of loess regression to perform
    loess_type: LoessFunc,
    /// How the surface is evaluated away from the anchors
    surface: LoessSurface,
}

impl<T> LoessRegression<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Generate a new instance of the Loess regression
    ///
    /// Uses [`LoessSurface::Interpolate`], matching R's default.
    ///
    /// ### Params
    ///
    /// * `span` - Fraction of the data in each local neighbourhood, in `(0, 1]`
    /// * `degree` - Local polynomial degree, `1` or `2`
    ///
    /// ### Return
    ///
    /// Initialised class
    pub fn new(span: T, degree: usize) -> Self {
        Self::with_surface(span, degree, LoessSurface::default())
    }

    /// Generate a new instance with an explicit evaluation surface
    ///
    /// ### Params
    ///
    /// * `span` - Fraction of the data in each local neighbourhood, in `(0, 1]`
    /// * `degree` - Local polynomial degree, `1` or `2`
    /// * `surface` - How to evaluate away from the anchor points
    ///
    /// ### Return
    ///
    /// Initialised class
    pub fn with_surface(span: T, degree: usize, surface: LoessSurface) -> Self {
        assert!(
            span > T::zero() && span <= T::one(),
            "Span must be between 0 and 1"
        );
        assert!(
            degree == 1 || degree == 2,
            "Only linear (1) and quadratic (2) supported"
        );

        let loess_type: LoessFunc = parse_loess_fun(&degree).unwrap_or_default();

        Self {
            span,
            loess_type,
            surface,
        }
    }

    /// Fit the loess function (for a two variable system)
    ///
    /// Points where either coordinate is non-finite are dropped, and their
    /// fitted value and residual stay at `0.0`. Callers rely on that: an
    /// all-zero gene reaches this with `x = -inf` and must come back finite.
    ///
    /// ### Params
    ///
    /// * `x` - The predictor variable
    /// * `y` - The response variable
    ///
    /// ### Returns
    ///
    /// The fit results in form of a `LoessRes`
    pub fn fit(&self, x: &[T], y: &[T]) -> LoessRes<T> {
        let n = x.len();

        let mut valid: Vec<(usize, T, T)> = x
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

        valid.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // Split into parallel arrays: the inner loops stream xs and ys, and a
        // struct of arrays keeps them contiguous.
        let n_valid = valid.len();
        let mut indices = Vec::with_capacity(n_valid);
        let mut xs = Vec::with_capacity(n_valid);
        let mut ys = Vec::with_capacity(n_valid);
        for (idx, xv, yv) in valid {
            indices.push(idx);
            xs.push(xv);
            ys.push(yv);
        }

        let q = (T::from_usize(n_valid).unwrap_or_else(T::one) * self.span)
            .to_usize()
            .unwrap_or(n_valid)
            .clamp(1, n_valid);

        let interpolate =
            self.surface == LoessSurface::Interpolate && n_valid >= LOESS_INTERPOLATE_MIN_POINTS;

        let fitted_sorted = if interpolate {
            self.fit_interpolate(&xs, &ys, q)
        } else {
            self.fit_direct(&xs, &ys, q)
        };

        let mut fitted_values = vec![T::zero(); n];
        let mut residuals = vec![T::zero(); n];
        for (position, &original) in indices.iter().enumerate() {
            fitted_values[original] = fitted_sorted[position];
            residuals[original] = ys[position] - fitted_sorted[position];
        }

        LoessRes {
            fitted_vals: fitted_values,
            residuals,
            valid_indices: indices,
        }
    }

    /// Run the local regression at every point.
    ///
    /// ### Params
    ///
    /// * `xs` - Predictor values, ascending
    /// * `ys` - Response values in the same order
    /// * `q` - Neighbourhood size
    ///
    /// ### Returns
    ///
    /// Fitted values in the order of `xs`.
    fn fit_direct(&self, xs: &[T], ys: &[T], q: usize) -> Vec<T> {
        xs.par_iter()
            .map(|&target| {
                let start = window_start(xs, q, target);
                local_fit(xs, ys, start, q, target, self.loess_type).0
            })
            .collect()
    }

    /// Fit at the anchor vertices and interpolate everything else.
    ///
    /// ### Params
    ///
    /// * `xs` - Predictor values, ascending
    /// * `ys` - Response values in the same order
    /// * `q` - Neighbourhood size
    ///
    /// ### Returns
    ///
    /// Fitted values in the order of `xs`.
    fn fit_interpolate(&self, xs: &[T], ys: &[T], q: usize) -> Vec<T> {
        let vertices = vertex_indices(xs, q);

        let anchors: Vec<(T, T, T)> = vertices
            .par_iter()
            .map(|&idx| {
                let target = xs[idx];
                let start = window_start(xs, q, target);
                let (value, slope) = local_fit(xs, ys, start, q, target, self.loess_type);
                (target, value, slope)
            })
            .collect();

        let mut knots = Vec::with_capacity(anchors.len());
        let mut values = Vec::with_capacity(anchors.len());
        let mut slopes = Vec::with_capacity(anchors.len());
        for (knot, value, slope) in anchors {
            knots.push(knot);
            values.push(value);
            slopes.push(slope);
        }

        xs.par_iter()
            .map(|&target| hermite_eval(&knots, &values, &slopes, target))
            .collect()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-8, "{} != {}", a, b);
    }

    /// Smooth curve with real curvature, so an undersmoothed fit and a properly
    /// spanned one cannot agree by accident.
    fn curved_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 6.0 - 3.0).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 1.7).sin() * 3.0).collect();
        (x, y)
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

    #[test]
    fn test_infinite_x_is_dropped_not_propagated() {
        // hvg.rs sends log10(0) = -inf for all-zero genes and needs a finite
        // fitted value back
        let x = vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0];
        let y = vec![5.0, 2.0, 4.0, 6.0];

        let res = LoessRegression::new(1.0, 1).fit(&x, &y);

        assert_eq!(res.fitted_vals[0], 0.0);
        assert_eq!(res.residuals[0], 0.0);
        assert!(res.fitted_vals.iter().all(|v| v.is_finite()));
        assert_eq!(res.valid_indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_span_changes_the_fit() {
        // The regression that started this: a fixed 64-neighbour cap made the
        // span inert for any input above ~213 points
        let (x, y) = curved_data(2_000);

        let tight = LoessRegression::with_surface(0.05, 2, LoessSurface::Direct).fit(&x, &y);
        let loose = LoessRegression::with_surface(0.90, 2, LoessSurface::Direct).fit(&x, &y);

        let max_delta = tight
            .fitted_vals
            .iter()
            .zip(loose.fitted_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);

        assert!(
            max_delta > 0.5,
            "span must move the fit, saw max delta {max_delta}"
        );
    }

    #[test]
    fn test_wide_span_tracks_the_signal() {
        // A correctly spanned quadratic loess should sit close to a smooth
        // generating curve
        let (x, y) = curved_data(1_000);
        let res = LoessRegression::with_surface(0.2, 2, LoessSurface::Direct).fit(&x, &y);

        let worst = res
            .fitted_vals
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);

        assert!(
            worst < 0.05,
            "quadratic loess should track sin, saw {worst}"
        );
    }

    /// Largest absolute gap between the two surfaces on the same input.
    ///
    /// ### Params
    ///
    /// * `x` - Predictor values
    /// * `y` - Response values
    /// * `span` - Span to fit both surfaces at
    ///
    /// ### Returns
    ///
    /// The maximum absolute difference in fitted values.
    fn surface_gap(x: &[f64], y: &[f64], span: f64) -> f64 {
        let direct = LoessRegression::with_surface(span, 2, LoessSurface::Direct).fit(x, y);
        let interpolated =
            LoessRegression::with_surface(span, 2, LoessSurface::Interpolate).fit(x, y);

        direct
            .fitted_vals
            .iter()
            .zip(interpolated.fitted_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max)
    }

    #[test]
    fn test_interpolate_matches_direct_on_oscillating_signal() {
        // Adversarial for the spline: 1.6 periods of a sine against roughly 11
        // anchors per period. R quotes about 1% of the range for its own
        // interpolating surface, and this sits inside that.
        let (x, y) = curved_data(4_000);
        let range = y.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v))
            - y.iter().fold(f64::INFINITY, |a, &v| a.min(v));

        let worst = surface_gap(&x, &y, 0.3);
        assert!(
            worst < 0.01 * range,
            "interpolating surface drifted from direct by {worst} over a range of {range}"
        );
    }

    #[test]
    fn test_interpolate_matches_direct_on_a_monotone_trend() {
        // The shape HVG actually feeds it: log10 variance against log10 mean is
        // smooth and monotone, where the spline should be near exact.
        let n = 4_000;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 4.0 - 3.0).collect();
        let y: Vec<f64> = x.iter().map(|&v| 1.8 * v + 0.35 * v * v - 0.4).collect();
        let range = y.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v))
            - y.iter().fold(f64::INFINITY, |a, &v| a.min(v));

        let worst = surface_gap(&x, &y, 0.3);
        assert!(
            worst < 1e-4 * range,
            "spline should be near exact on a smooth trend, saw {worst} over a range of {range}"
        );
    }

    #[test]
    fn test_small_input_ignores_interpolation() {
        // Below LOESS_INTERPOLATE_MIN_POINTS the two surfaces must agree
        // exactly, since Interpolate defers to Direct
        let (x, y) = curved_data(100);

        let direct = LoessRegression::with_surface(0.4, 2, LoessSurface::Direct).fit(&x, &y);
        let interpolated =
            LoessRegression::with_surface(0.4, 2, LoessSurface::Interpolate).fit(&x, &y);

        assert_eq!(direct.fitted_vals, interpolated.fitted_vals);
    }

    #[test]
    fn test_window_start_picks_nearest_neighbours() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();

        // target at the left edge cannot slide
        assert_eq!(window_start(&xs, 4, 0.0), 0);
        // target at the right edge is pinned to the last window
        assert_eq!(window_start(&xs, 4, 9.0), 6);
        // centred target keeps the window centred
        let start = window_start(&xs, 4, 4.5);
        assert!((2..=3).contains(&start), "unexpected start {start}");
        // q covering everything always starts at zero
        assert_eq!(window_start(&xs, 10, 7.0), 0);
    }

    #[test]
    fn test_constant_x_does_not_divide_by_zero() {
        let x = vec![2.0; 50];
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();

        let res = LoessRegression::new(0.5, 2).fit(&x, &y);

        assert!(res.fitted_vals.iter().all(|v| v.is_finite()));
        // every neighbourhood is the same degenerate point cloud, so every fit
        // collapses to the local mean
        let first = res.fitted_vals[0];
        assert!(res.fitted_vals.iter().all(|&v| (v - first).abs() < 1e-9));
    }

    #[test]
    fn test_reproducible() {
        let (x, y) = curved_data(1_000);
        let a = LoessRegression::new(0.3, 2).fit(&x, &y);
        let b = LoessRegression::new(0.3, 2).fit(&x, &y);

        assert_eq!(a.fitted_vals, b.fitted_vals);
    }

    #[test]
    fn test_f32_and_f64_agree() {
        let (x64, y64) = curved_data(1_000);
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = y64.iter().map(|&v| v as f32).collect();

        let res64 = LoessRegression::new(0.3, 2).fit(&x64, &y64);
        let res32 = LoessRegression::new(0.3_f32, 2).fit(&x32, &y32);

        let worst = res64
            .fitted_vals
            .iter()
            .zip(res32.fitted_vals.iter())
            .map(|(&a, &b)| (a - b as f64).abs())
            .fold(0f64, f64::max);

        assert!(worst < 1e-4, "f32 and f64 paths diverged by {worst}");
    }
}
