use faer::{
    Mat, MatRef, Scale,
    linalg::solvers::{PartialPivLu, Solve},
    traits::AddByRef,
};

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

////////////////////
// Matrix solvers //
////////////////////

/// Sylvester solver for three matrix systems
///
/// Solves a system of `AX + XB = C`. Pending on the size of the underlying
/// matrices, the algorithm will solve this directly or iteratively.
///
/// ### Params
///
/// * `mat_a` - Matrix A of the system
/// * `mat_b` - Matrix B of the system
/// * `mat_c` - Matrix C of the system
///
/// ### Returns
///
/// The matrix X
pub fn sylvester_solver<T: BixverseFloat>(
    mat_a: &MatRef<T>,
    mat_b: &MatRef<T>,
    mat_c: &MatRef<T>,
) -> Mat<T> {
    let m = mat_a.nrows();
    let n = mat_b.ncols();

    if m * n < 1000 {
        sylvester_solver_direct(mat_a, mat_b, mat_c)
    } else {
        sylvester_solver_iterative(mat_a, mat_b, mat_c, 50, T::from_f64(1e-6).unwrap())
    }
}

/// Iterative Sylvester solver using fixed-point iteration
///
/// Solves a system of `AX + XB = C`. Uses an iterative approach more
/// appropriate for large matrix systems.
///
/// ### Params
///
/// * `mat_a` - Matrix A of the system
/// * `mat_b` - Matrix B of the system
/// * `mat_c` - Matrix C of the system
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Tolerance parameter
///
/// Returns
///
/// The matrix X
fn sylvester_solver_iterative<T: BixverseFloat>(
    mat_a: &MatRef<T>,
    mat_b: &MatRef<T>,
    mat_c: &MatRef<T>,
    max_iter: usize,
    tolerance: T,
) -> Mat<T> {
    let m = mat_a.nrows();
    let n = mat_b.ncols();

    let mut x = mat_c.to_owned();
    let mut x_new = Mat::zeros(m, n);
    let mut residual = Mat::zeros(m, n);

    let mut alpha = T::from_f64(0.5).unwrap();
    let alpha_min = T::from_f64(0.01).unwrap();
    let alpha_max = T::one();
    let mut prev_residual_norm = T::infinity();

    let c_norm = mat_c.norm_l2();
    let rel_tolerance = tolerance * c_norm.max(T::one());

    for iter in 0..max_iter {
        let ax = mat_a * &x;
        let xb = &x * mat_b;

        residual.copy_from(&mat_c);
        residual -= &ax;
        residual -= &xb;

        let residual_norm = residual.norm_l2();

        if residual_norm < rel_tolerance {
            break;
        }

        if iter > 0 {
            if residual_norm < prev_residual_norm {
                alpha = (alpha * T::from_f64(1.1).unwrap()).min(alpha_max);
            } else {
                alpha = (alpha * T::from_f64(0.5).unwrap()).max(alpha_min);
            }
        }

        x_new.copy_from(&x);
        x_new.add_by_ref(&(residual.as_ref() * Scale(alpha)));

        std::mem::swap(&mut x, &mut x_new);
        prev_residual_norm = residual_norm;

        if iter > 10 && residual_norm > T::from_f64(0.99).unwrap() * prev_residual_norm {
            break;
        }
    }

    x
}

/// Direct version for small matrices
///
/// Uses partial LU decomposition to solve: `AX + XB = C`. Slow for large
/// matrix systems.
///
/// ### Params
///
/// * `mat_a` - Matrix A of the system
/// * `mat_b` - Matrix B of the system
/// * `mat_c` - Matrix C of the system
///
/// ### Returns
///
/// The matrix X
fn sylvester_solver_direct<T: BixverseFloat>(
    mat_a: &MatRef<T>,
    mat_b: &MatRef<T>,
    mat_c: &MatRef<T>,
) -> Mat<T> {
    let m = mat_a.nrows();
    let n = mat_b.ncols();
    let mn = m * n;

    let mut coeff_matrix: Mat<T> = Mat::zeros(mn, mn);

    for i in 0..m {
        for j in 0..n {
            let row_idx = i * n + j;

            for k in 0..m {
                let col_idx = k * n + j;
                coeff_matrix[(row_idx, col_idx)] = mat_a[(i, k)];
            }

            for l in 0..n {
                let col_idx = i * n + l;
                coeff_matrix[(row_idx, col_idx)] += mat_b[(l, j)];
            }
        }
    }

    let mut c_vec: Mat<T> = Mat::zeros(mn, 1);
    for i in 0..m {
        for j in 0..n {
            c_vec[(i * n + j, 0)] = mat_c[(i, j)];
        }
    }

    let lu = PartialPivLu::new(coeff_matrix.as_ref());
    let solved = lu.solve(&c_vec);

    let mut res = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            res[(i, j)] = solved[(i * n + j, 0)];
        }
    }

    res
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![3.0, 5.0, 7.0]; // Formula: y = 2x + 1
        let (intercept, slope): (f64, f64) = linear_regression(&x, &y);

        assert!((intercept - 1.0).abs() < 1e-6);
        assert!((slope - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sylvester_solver() {
        // Solving AX + XB = C
        // Let A = 2*I, B = 3*I. Then 2IX + 3XI = 5X = C
        // If C = 10*I, then X = 2*I
        let mat_a: Mat<f64> = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.0 });
        let mat_b: Mat<f64> = Mat::from_fn(2, 2, |i, j| if i == j { 3.0 } else { 0.0 });
        let mat_c: Mat<f64> = Mat::from_fn(2, 2, |i, j| if i == j { 10.0 } else { 0.0 });

        let x = sylvester_solver(&mat_a.as_ref(), &mat_b.as_ref(), &mat_c.as_ref());

        assert!((x[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((x[(1, 1)] - 2.0).abs() < 1e-6);
        assert!((x[(0, 1)] - 0.0).abs() < 1e-6);
        assert!((x[(1, 0)] - 0.0).abs() < 1e-6);
    }
}
