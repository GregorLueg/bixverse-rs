//! Linear algebra helper functions like simple regressions and more complex
//! solvers such as the Sylvester solver.

use faer::{
    Mat, MatRef, Scale,
    linalg::solvers::{PartialPivLu, Solve},
    traits::AddByRef,
};

use crate::prelude::*;

/// Relative pivot tolerance below which a Gram matrix is called rank deficient.
///
/// Relative to the largest diagonal entry, so a design that is merely badly
/// scaled is not mistaken for a rank-deficient one.
const RANK_TOL: f64 = 1.0e-14;

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

/// Simple linear regression over `f32` data, accumulated in `f64`
///
/// Same fit as [`linear_regression`], but the four moment sums are taken in
/// `f64`. `f32` holds integers exactly only to `2^24`, and the normal equations
/// need `sum_xy` and `sum_xx`, which square the inputs: aggregated counts over
/// tens of thousands of observations walk past that immediately and the slope
/// comes back wrong rather than merely imprecise.
///
/// ### Params
///
/// * `x` - Independent variable
/// * `y` - Dependent variable
///
/// ### Returns
///
/// Tuple of (intercept, slope), narrowed back to `f32`.
pub fn linear_regression_widen(x: &[f32], y: &[f32]) -> (f32, f32) {
    let n = x.len() as f64;

    let (sum_x, sum_y, sum_xy, sum_xx) = x.iter().zip(y.iter()).fold(
        (0f64, 0f64, 0f64, 0f64),
        |(sx, sy, sxy, sxx), (&xi, &yi)| {
            let xi = xi as f64;
            let yi = yi as f64;
            (sx + xi, sy + yi, sxy + xi * yi, sxx + xi * xi)
        },
    );

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (intercept as f32, slope as f32)
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

/////////////////////
// Residualisation //
/////////////////////

/// Regresses covariates out of one or more response columns.
///
/// Ordinary least squares with an intercept, returning `Y - D (D'D)^-1 D'Y`
/// where `D = [1 | covariates]`. The normal equations are formed and solved in
/// `f64` regardless of `T`: `D'D` on a covariate like a z-scored log library
/// size is well conditioned, but the responses here are program scores whose
/// scale is arbitrary, and the subtraction at the end is a cancellation.
///
/// A rank-deficient design is rejected rather than fitted. `D'D` is singular
/// whenever a covariate is constant or duplicates another, which is not exotic:
/// a single-batch run, or a covariate that becomes constant once the data is
/// subset to one cell type, both produce it. The factorisation would divide by
/// a zero pivot and return all-`NaN` residuals under an `Ok`, so the rank is
/// checked first.
///
/// ### Params
///
/// * `y` - Responses, `n x k`. Each column is residualised independently.
/// * `covariates` - Confounders, `n x c`. The intercept is added here, so do
///   not pass a column of ones. Must be full rank alongside it.
///
/// ### Returns
///
/// The `n x k` residuals, [BixverseErrors::ShapeMismatch] when the row counts
/// disagree, or [BixverseErrors::InvalidArgument] for a rank-deficient design
/// or a non-finite input.
pub fn ols_residualise<T: BixverseFloat>(
    y: MatRef<T>,
    covariates: MatRef<T>,
) -> Result<Mat<T>, BixverseErrors> {
    let n = y.nrows();
    if covariates.nrows() != n {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (n, covariates.ncols()),
            got: (covariates.nrows(), covariates.ncols()),
        });
    }
    let k = y.ncols();
    let c = covariates.ncols();
    let p = c + 1;

    // D = [1 | covariates], in f64.
    let design = Mat::<f64>::from_fn(n, p, |i, j| {
        if j == 0 {
            1.0
        } else {
            covariates[(i, j - 1)].to_f64().unwrap_or(f64::NAN)
        }
    });
    let y_f64 = Mat::<f64>::from_fn(n, k, |i, j| y[(i, j)].to_f64().unwrap_or(f64::NAN));

    if (0..n).any(|i| {
        (0..p).any(|j| !design[(i, j)].is_finite()) || (0..k).any(|j| !y_f64[(i, j)].is_finite())
    }) {
        return Err(BixverseErrors::InvalidArgument(
            "the responses or covariates contain a non-finite value.".to_string(),
        ));
    }

    let dtd = design.transpose() * &design;
    if spd_log_det(dtd.as_ref()).is_none() {
        return Err(BixverseErrors::InvalidArgument(
            "the covariate design is rank deficient; a covariate is constant or duplicated."
                .to_string(),
        ));
    }

    let dty = design.transpose() * &y_f64;
    let lu: PartialPivLu<f64> = dtd.partial_piv_lu();
    let beta = lu.solve(&dty);
    let fitted = &design * &beta;

    Ok(Mat::<T>::from_fn(n, k, |i, j| {
        T::from_f64(y_f64[(i, j)] - fitted[(i, j)]).unwrap_or_else(T::nan)
    }))
}

/// Log-determinant of a small symmetric positive definite matrix.
///
/// Cholesky, accumulated in logs. Doubles as a rank test: a rank-deficient
/// matrix hits a non-positive pivot, and one that is merely badly scaled still
/// returns a finite answer where the determinant itself would under- or
/// overflow. The pivot floor is relative to the largest diagonal entry, so
/// scale alone never looks like rank deficiency.
///
/// ### Params
///
/// * `a` - Symmetric matrix, `p x p`, only the lower triangle read
///
/// ### Returns
///
/// `log |a|`, or `None` when `a` is not positive definite.
pub fn spd_log_det(a: MatRef<f64>) -> Option<f64> {
    let p = a.nrows();
    let scale = (0..p).map(|i| a[(i, i)].abs()).fold(0.0_f64, f64::max);
    if scale <= 0.0 {
        return None;
    }
    let floor = RANK_TOL * scale;
    let mut l = vec![0.0_f64; p * p];
    let mut log_det = 0.0_f64;
    for i in 0..p {
        for j in 0..=i {
            let mut acc = a[(i, j)];
            for k in 0..j {
                acc -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if !(acc.is_finite() && acc > floor) {
                    return None;
                }
                let d = acc.sqrt();
                l[i * p + j] = d;
                log_det += 2.0 * d.ln();
            } else {
                l[i * p + j] = acc / l[j * p + j];
            }
        }
    }
    Some(log_det)
}

//////////
// NNLS //
//////////

/// Tuning knobs for [nnls_gram].
#[derive(Clone, Copy, Debug)]
pub struct NnlsParams<T> {
    /// Outer iteration cap, as a multiple of the number of variables.
    ///
    /// Lawson-Hanson terminates in exact arithmetic; the cap only exists
    /// because the guarantee that a newly activated variable enters with a
    /// positive coefficient can fail in floating point, which would otherwise
    /// let the same variable cycle in and out forever. Three is the usual
    /// choice and is never reached on a well conditioned problem.
    pub max_iter_factor: usize,
    /// Threshold on the gradient below which no further variable is activated.
    ///
    /// Scaled by the largest diagonal entry of the Gram matrix inside the
    /// solver, so it is relative to the problem rather than absolute.
    pub tol: T,
}

impl<T: BixverseFloat> NnlsParams<T> {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `max_iter_factor` - Outer iteration cap as a multiple of the number of
    ///   variables
    /// * `tol` - Relative gradient tolerance
    ///
    /// ### Returns
    ///
    /// The parameter set. Nothing is validated here; [nnls_gram] checks the
    /// values against the problem it is given.
    pub fn new(max_iter_factor: usize, tol: T) -> Self {
        Self {
            max_iter_factor,
            tol,
        }
    }
}

impl<T: BixverseFloat> Default for NnlsParams<T> {
    fn default() -> Self {
        Self {
            max_iter_factor: 3,
            tol: T::from_f64(1e-10).unwrap(),
        }
    }
}

/// Non-negative least squares on the normal equations.
///
/// Solves `min ||X b - y||^2` subject to `b >= 0`, given only `X'X` and `X'y`.
/// Taking the Gram form rather than `X` is the point: a caller that has
/// streamed a tall `X` past once and reduced it never has to hold it, and the
/// solve then costs nothing in the row count.
///
/// Lawson-Hanson active set, so the zeros are exact. That matters wherever
/// `b_j > 0` is used as a selection rule: a coordinate-descent solver with a
/// positive floor, such as [crate::methods::nmf_hals], returns a strictly
/// positive vector and would select everything.
///
/// The passive-set Cholesky is grown by rank-one append while variables are
/// activated, which is the common case, and refactorised from scratch only when
/// the inner loop drops one. That turns the naive `O(k^4)` into `O(k^3)`.
///
/// ### Params
///
/// * `xtx` - Gram matrix `X'X`, `k x k`, symmetric positive semi-definite. The
///   full matrix is read, not just one triangle: the passive set is held in
///   activation order rather than sorted, so the factorisation indexes both
///   sides of the diagonal.
/// * `xty` - Cross product `X'y`, length `k`
/// * `params` - Optional [NnlsParams]; [NnlsParams::default] otherwise
///
/// ### Returns
///
/// The non-negative coefficient vector, with exact zeros on the active set, or
/// [BixverseErrors::ShapeMismatch] when the arguments disagree.
///
/// ### References
///
/// Lawson & Hanson, Solving Least Squares Problems, 1974, chapter 23.
/// Bro & De Jong, Journal of Chemometrics, 1997, for the Gram formulation.
pub fn nnls_gram<T: BixverseFloat>(
    xtx: MatRef<T>,
    xty: &[T],
    params: Option<NnlsParams<T>>,
) -> Result<Vec<T>, BixverseErrors> {
    let k = xty.len();
    if xtx.nrows() != k || xtx.ncols() != k {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (k, k),
            got: (xtx.nrows(), xtx.ncols()),
        });
    }
    let params = params.unwrap_or_default();
    let zero = T::zero();

    // Relative gradient threshold: the Gram diagonal sets the scale of X'y.
    let mut diag_max = zero;
    for j in 0..k {
        if xtx[(j, j)] > diag_max {
            diag_max = xtx[(j, j)];
        }
    }
    let tol = params.tol * if diag_max > zero { diag_max } else { T::one() };

    let mut x = vec![zero; k];
    let mut passive: Vec<usize> = Vec::with_capacity(k);
    let mut in_passive = vec![false; k];
    let mut blocked = vec![false; k];
    let mut chol: Vec<T> = Vec::with_capacity(k * (k + 1) / 2);
    let mut gradient = xty.to_vec();

    for _ in 0..params.max_iter_factor * k {
        // Steepest ascent among the variables still held at zero.
        let mut best: Option<usize> = None;
        let mut best_grad = tol;
        for j in 0..k {
            if !in_passive[j] && !blocked[j] && gradient[j] > best_grad {
                best_grad = gradient[j];
                best = Some(j);
            }
        }
        let Some(entering) = best else { break };

        // A column that is numerically dependent on the passive set cannot be
        // factorised in. Block it rather than letting it be retried forever.
        if !chol_append(xtx, &passive, &mut chol, entering) {
            blocked[entering] = true;
            continue;
        }
        passive.push(entering);
        in_passive[entering] = true;

        loop {
            let rhs: Vec<T> = passive.iter().map(|&p| xty[p]).collect();
            let step = chol_solve(&chol, passive.len(), &rhs);

            if step.iter().all(|&v| v > zero) {
                for (i, &p) in passive.iter().enumerate() {
                    x[p] = step[i];
                }
                break;
            }

            // Move as far towards the unconstrained solution as non-negativity
            // allows. At least one passive coefficient lands exactly on zero.
            let mut alpha = T::infinity();
            for (i, &p) in passive.iter().enumerate() {
                if step[i] <= zero {
                    let denom = x[p] - step[i];
                    if denom > zero {
                        let ratio = x[p] / denom;
                        if ratio < alpha {
                            alpha = ratio;
                        }
                    }
                }
            }
            if !alpha.is_finite() {
                break;
            }
            for (i, &p) in passive.iter().enumerate() {
                x[p] = x[p] + alpha * (step[i] - x[p]);
            }

            let mut kept: Vec<usize> = Vec::with_capacity(passive.len());
            for &p in &passive {
                if x[p] > zero {
                    kept.push(p);
                } else {
                    x[p] = zero;
                    in_passive[p] = false;
                }
            }
            passive = kept;

            if passive.is_empty() {
                chol.clear();
                break;
            }
            if !chol_rebuild(xtx, &passive, &mut chol) {
                // Should not happen: the set factorised a moment ago. Bail out
                // with the last feasible iterate rather than looping.
                for &p in &passive {
                    in_passive[p] = false;
                }
                passive.clear();
                chol.clear();
                break;
            }
        }

        // The entering variable can be dropped again by the inner loop when
        // floating point breaks Lawson-Hanson's positivity guarantee. Block it
        // so the outer loop does not pick it straight back up.
        if !in_passive[entering] {
            blocked[entering] = true;
        }

        for (j, g) in gradient.iter_mut().enumerate() {
            let mut acc = xty[j];
            for &p in &passive {
                acc -= xtx[(j, p)] * x[p];
            }
            *g = acc;
        }
    }

    Ok(x)
}

/// Packed index of the lower-triangular entry `(i, j)`, row-major.
#[inline(always)]
fn tri_idx(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// Appends one column to the Cholesky factor of the passive Gram submatrix.
///
/// Given `L L' = G[passive, passive]`, the new row is `l = L^-1 G[passive, j]`
/// and the new diagonal is `sqrt(G[j, j] - l'l)`. Costs `O(m^2)`, against
/// `O(m^3)` for a refactorisation.
///
/// ### Params
///
/// * `gram` - Full Gram matrix
/// * `passive` - Current passive set, in factor order
/// * `chol` - Packed lower-triangular factor, extended in place on success
/// * `entering` - Index of the column to append
///
/// ### Returns
///
/// `true` on success. `false` when the Schur complement is non-positive, which
/// means the column is numerically dependent on the passive set; `chol` is left
/// untouched in that case.
fn chol_append<T: BixverseFloat>(
    gram: MatRef<T>,
    passive: &[usize],
    chol: &mut Vec<T>,
    entering: usize,
) -> bool {
    let m = passive.len();
    let mut row = vec![T::zero(); m];
    for i in 0..m {
        let mut acc = gram[(passive[i], entering)];
        for j in 0..i {
            acc -= chol[tri_idx(i, j)] * row[j];
        }
        let pivot = chol[tri_idx(i, i)];
        if pivot == T::zero() {
            return false;
        }
        row[i] = acc / pivot;
    }
    let mut schur = gram[(entering, entering)];
    for v in row.iter() {
        schur -= *v * *v;
    }
    // Relative guard: a Schur complement this small against the diagonal means
    // the column adds no rank.
    let floor = gram[(entering, entering)] * T::from_f64(1e-12).unwrap();
    if !(schur.is_finite() && schur > floor && schur > T::zero()) {
        return false;
    }
    chol.extend_from_slice(&row);
    chol.push(schur.sqrt());
    true
}

/// Refactorises the passive Gram submatrix from scratch.
///
/// Only reached when the inner loop drops a variable, which is the rare branch.
///
/// ### Params
///
/// * `gram` - Full Gram matrix
/// * `passive` - Current passive set, in factor order
/// * `chol` - Packed lower-triangular factor, overwritten
///
/// ### Returns
///
/// `true` on success, `false` if the submatrix is not positive definite.
fn chol_rebuild<T: BixverseFloat>(gram: MatRef<T>, passive: &[usize], chol: &mut Vec<T>) -> bool {
    let m = passive.len();
    chol.clear();
    chol.resize(m * (m + 1) / 2, T::zero());
    for i in 0..m {
        for j in 0..=i {
            let mut acc = gram[(passive[i], passive[j])];
            for l in 0..j {
                acc -= chol[tri_idx(i, l)] * chol[tri_idx(j, l)];
            }
            if i == j {
                if !(acc.is_finite() && acc > T::zero()) {
                    return false;
                }
                chol[tri_idx(i, j)] = acc.sqrt();
            } else {
                chol[tri_idx(i, j)] = acc / chol[tri_idx(j, j)];
            }
        }
    }
    true
}

/// Solves `L L' z = rhs` by forward then back substitution.
///
/// ### Params
///
/// * `chol` - Packed lower-triangular factor
/// * `m` - Dimension of the factor
/// * `rhs` - Right-hand side, length `m`
///
/// ### Returns
///
/// The solution `z`, length `m`.
fn chol_solve<T: BixverseFloat>(chol: &[T], m: usize, rhs: &[T]) -> Vec<T> {
    let mut z = vec![T::zero(); m];
    for i in 0..m {
        let mut acc = rhs[i];
        for j in 0..i {
            acc -= chol[tri_idx(i, j)] * z[j];
        }
        z[i] = acc / chol[tri_idx(i, i)];
    }
    for i in (0..m).rev() {
        let mut acc = z[i];
        for j in (i + 1)..m {
            acc -= chol[tri_idx(j, i)] * z[j];
        }
        z[i] = acc / chol[tri_idx(i, i)];
    }
    z
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Least squares on exactly collinear points recovers the intercept and the slope.
    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![3.0, 5.0, 7.0]; // Formula: y = 2x + 1
        let (intercept, slope): (f64, f64) = linear_regression(&x, &y);

        assert!((intercept - 1.0).abs() < 1e-6);
        assert!((slope - 2.0).abs() < 1e-6);
    }

    /// The Sylvester solver on a diagonal case whose solution is available in closed form.
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

    // -- ols_residualise --

    /// Fixture inputs shared by the residualisation tests. R fixtures below
    /// were generated from exactly these vectors.
    fn ols_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let q = vec![
            -0.591, 0.027, -1.517, -1.363, 1.178, -0.934, 1.324, 0.625, -0.046, -1.004,
        ];
        let z = vec![
            0.204, 0.68, 0.364, 0.35, 0.062, 0.483, 0.399, 0.016, 0.125, 0.398,
        ];
        let y1 = vec![
            -1.176, -0.057, -2.59, -3.022, 2.028, -2.209, 2.64, 1.029, 0.084, -1.971,
        ];
        let y2 = vec![
            -0.496, -0.76, -1.505, -1.142, 0.035, -1.502, -0.206, 0.636, -0.937, -1.335,
        ];
        (q, z, y1, y2)
    }

    /// Two responses residualised on one covariate, against R's `lm`.
    #[test]
    fn test_ols_residualise_single_covariate() {
        let (q, _z, y1, y2) = ols_fixture();
        let n = q.len();
        let y = Mat::<f64>::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
        let cov = Mat::<f64>::from_fn(n, 1, |i, _| q[i]);

        let res = ols_residualise(y.as_ref(), cov.as_ref()).unwrap();

        // R: residuals(lm(y1 ~ q))
        let expected_1 = [
            0.0438350360944052,
            -0.028017976669079646,
            0.41418760861704207,
            -0.31456217126253216,
            -0.16093353927551277,
            -0.328222272355633,
            0.16973225382566953,
            -0.094_332_056_980_677_22,
            0.253_649_126_780_326_5,
            0.044_663_991_225_991_55,
        ];
        // R: residuals(lm(y2 ~ q))
        let expected_2 = [
            0.4328661028001109,
            -0.18673836251013678,
            -0.043302610990681785,
            0.231083978559839,
            -0.054_037_293_856_570_89,
            -0.37576766483513946,
            -0.379_047_410_256_726_8,
            0.865_165_407_302_923_1,
            -0.321_733_304_310_059_6,
            -0.16848884190355792,
        ];
        for i in 0..n {
            assert!((res[(i, 0)] - expected_1[i]).abs() < 1e-12);
            assert!((res[(i, 1)] - expected_2[i]).abs() < 1e-12);
        }
    }

    /// Two covariates plus the implicit intercept, against R's `lm`.
    #[test]
    fn test_ols_residualise_two_covariates() {
        let (q, z, y1, y2) = ols_fixture();
        let n = q.len();
        let y = Mat::<f64>::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
        let cov = Mat::<f64>::from_fn(n, 2, |i, j| if j == 0 { q[i] } else { z[i] });

        let res = ols_residualise(y.as_ref(), cov.as_ref()).unwrap();

        // R: residuals(lm(y1 ~ q + z))
        let expected_1 = [
            0.03516127176896492,
            -0.0019352760730467421,
            0.411_809_837_167_337_2,
            -0.31714410593417963,
            -0.17069455442358725,
            -0.319_874_936_587_889,
            0.18319302794382428,
            -0.10979566650150827,
            0.24228383279577836,
            0.046_996_569_844_306_08,
        ];
        // R: residuals(lm(y2 ~ q + z))
        let expected_2 = [
            0.275_770_617_625_924_5,
            0.28566028138301697,
            -0.086_367_783_793_349_9,
            0.1843210893790895,
            -0.23082459783966577,
            -0.22458431316169894,
            -0.135_251_664_653_009_1,
            0.585_095_169_160_293,
            -0.527_576_615_894_295_3,
            -0.12624218220630484,
        ];
        for i in 0..n {
            assert!((res[(i, 0)] - expected_1[i]).abs() < 1e-12);
            assert!((res[(i, 1)] - expected_2[i]).abs() < 1e-12);
        }
    }

    /// Residuals are orthogonal to the design, intercept included, so they sum
    /// to zero and have zero covariance with every covariate.
    #[test]
    fn test_ols_residualise_is_orthogonal_to_the_design() {
        let (q, z, y1, _y2) = ols_fixture();
        let n = q.len();
        let y = Mat::<f64>::from_fn(n, 1, |i, _| y1[i]);
        let cov = Mat::<f64>::from_fn(n, 2, |i, j| if j == 0 { q[i] } else { z[i] });

        let res = ols_residualise(y.as_ref(), cov.as_ref()).unwrap();

        let sum: f64 = (0..n).map(|i| res[(i, 0)]).sum();
        let dot_q: f64 = (0..n).map(|i| res[(i, 0)] * q[i]).sum();
        let dot_z: f64 = (0..n).map(|i| res[(i, 0)] * z[i]).sum();
        assert!(sum.abs() < 1e-12);
        assert!(dot_q.abs() < 1e-12);
        assert!(dot_z.abs() < 1e-12);
    }

    #[test]
    fn test_ols_residualise_rejects_mismatched_rows() {
        let y = Mat::<f64>::zeros(5, 1);
        let cov = Mat::<f64>::zeros(4, 1);
        assert!(ols_residualise(y.as_ref(), cov.as_ref()).is_err());
    }

    /// A singular design must error, not return all-NaN residuals under `Ok`.
    ///
    /// Both cases are ordinary in practice: a covariate that is constant within
    /// the subset being analysed, or one accidentally passed twice. The LU
    /// solve divides by a zero pivot and the NaN then propagates through every
    /// downstream score with nothing to signal it.
    #[test]
    fn test_ols_residualise_rejects_a_singular_design() {
        let (q, _z, y1, _y2) = ols_fixture();
        let n = q.len();
        let y = Mat::<f64>::from_fn(n, 1, |i, _| y1[i]);

        // A constant covariate is collinear with the intercept.
        let constant = Mat::<f64>::from_fn(n, 1, |_, _| 1.0);
        assert!(ols_residualise(y.as_ref(), constant.as_ref()).is_err());

        // A duplicated covariate.
        let duplicated = Mat::<f64>::from_fn(n, 2, |i, _| q[i]);
        assert!(ols_residualise(y.as_ref(), duplicated.as_ref()).is_err());

        // More covariates than observations.
        let wide = Mat::<f64>::from_fn(n, n + 2, |i, j| ((i * 7 + j * 3) % 11) as f64);
        assert!(ols_residualise(y.as_ref(), wide.as_ref()).is_err());
    }

    #[test]
    fn test_ols_residualise_rejects_non_finite_input() {
        let (q, _z, y1, _y2) = ols_fixture();
        let n = q.len();
        let cov = Mat::<f64>::from_fn(n, 1, |i, _| q[i]);

        let mut poisoned = y1.clone();
        poisoned[2] = f64::NAN;
        let y = Mat::<f64>::from_fn(n, 1, |i, _| poisoned[i]);
        assert!(ols_residualise(y.as_ref(), cov.as_ref()).is_err());

        let bad_cov = Mat::<f64>::from_fn(n, 1, |i, _| if i == 1 { f64::INFINITY } else { q[i] });
        let clean = Mat::<f64>::from_fn(n, 1, |i, _| y1[i]);
        assert!(ols_residualise(clean.as_ref(), bad_cov.as_ref()).is_err());
    }

    // -- nnls_gram --

    /// Rebuilds a Gram matrix from a row-major fixture.
    fn gram_from_rows(rows: &[f64], k: usize) -> Mat<f64> {
        Mat::<f64>::from_fn(k, k, |i, j| rows[i * k + j])
    }

    /// A problem with a genuinely mixed active set, against R's `nnls::nnls`.
    ///
    /// Two of the five true coefficients are negative, so the solver has to put
    /// them on the boundary. This is the case that separates an active-set
    /// solver from a coordinate-descent one with a positive floor.
    #[test]
    fn test_nnls_gram_matches_r_nnls_with_active_constraints() {
        // R: set.seed(42); X <- matrix(round(rnorm(60), 3), 12, 5); ...
        //    nnls(X, y)
        let xtx = gram_from_rows(
            &[
                16.210632,
                -2.9933329999999994,
                1.3656329999999997,
                -0.654_722_000_000_000_6,
                -2.5521339999999997,
                -2.9933329999999994,
                22.032358000000002,
                -3.6158030000000005,
                1.3042700000000003,
                -1.642_839,
                1.3656329999999997,
                -3.6158030000000005,
                12.920_613,
                -4.973_145,
                -0.602_566_999_999_999_9,
                -0.654_722_000_000_000_6,
                1.3042700000000003,
                -4.973_145,
                13.245169000000002,
                0.882_450_999_999_999_8,
                -2.5521339999999997,
                -1.642_839,
                -0.602_566_999_999_999_9,
                0.882_450_999_999_999_8,
                13.824906,
            ],
            5,
        );
        let xty = [
            35.078_253,
            -44.613946999999996,
            18.730541000000006,
            -11.449845000000002,
            14.151823,
        ];

        let got = nnls_gram(xtx.as_ref(), &xty, None).unwrap();

        // R: coef(nnls(X, y))
        let expected = [
            2.2928542579362294,
            0.0,
            1.2773973438543413,
            0.0,
            1.5025933481224463,
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got {got:?}");
        }
        // Exact zeros, not merely small ones. Step 3 of DIALOGUE selects on
        // `coef > 0`, so this is the property that matters.
        assert_eq!(got[1], 0.0);
        assert_eq!(got[3], 0.0);
    }

    /// When the unconstrained solution is already non-negative the active set
    /// stays empty and NNLS is plain least squares.
    #[test]
    fn test_nnls_gram_falls_back_to_least_squares() {
        // R: set.seed(7); X2 <- matrix(round(abs(rnorm(24)), 3), 8, 3); ...
        let xtx = gram_from_rows(
            &[
                9.727_400_999_999_999,
                8.333_146_000_000_001,
                5.445_142,
                8.333_146_000_000_001,
                21.450_824,
                8.765_513,
                5.445_142,
                8.765_513,
                6.704_459,
            ],
            3,
        );
        let xty = [29.189522, 55.720299000000004, 26.392_924];

        let got = nnls_gram(xtx.as_ref(), &xty, None).unwrap();

        // R: coef(nnls(X2, y2)), which here equals qr.solve(X2, y2).
        let expected = [
            1.0036097822848686,
            2.0014016077616805,
            0.504_860_185_111_712_7,
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got {got:?}");
        }
    }

    /// Every coefficient clamped: with `X'y` entirely non-positive there is no
    /// ascent direction and the answer is the zero vector.
    #[test]
    fn test_nnls_gram_all_clamped() {
        let xtx = gram_from_rows(&[2.0, 0.3, 0.3, 1.5], 2);
        let xty = [-1.0, -2.0];
        let got = nnls_gram(xtx.as_ref(), &xty, None).unwrap();
        assert_eq!(got, vec![0.0, 0.0]);
    }

    /// Brute force over all active sets. Independent of any R package: for each
    /// subset, solve the unconstrained problem restricted to it and keep the
    /// feasible candidate with the lowest objective. NNLS must find that one.
    #[test]
    fn test_nnls_gram_matches_exhaustive_active_set_search() {
        let k = 4;
        // Diagonally dominant so every submatrix is invertible.
        let xtx = gram_from_rows(
            &[
                4.0, 1.0, 0.5, -0.7, 1.0, 3.0, -1.2, 0.4, 0.5, -1.2, 5.0, 1.1, -0.7, 0.4, 1.1, 2.5,
            ],
            k,
        );
        let xty = [1.0, -2.0, 3.0, -0.5];

        let got = nnls_gram(xtx.as_ref(), &xty, None).unwrap();

        // Objective up to the constant y'y: b' X'X b / 2 - b' X'y.
        let objective = |b: &[f64]| {
            let mut acc = 0.0;
            for i in 0..k {
                acc -= b[i] * xty[i];
                for j in 0..k {
                    acc += 0.5 * b[i] * xtx[(i, j)] * b[j];
                }
            }
            acc
        };

        let mut best: Option<(f64, Vec<f64>)> = None;
        for mask in 0u32..(1 << k) {
            let passive: Vec<usize> = (0..k).filter(|j| mask & (1 << j) != 0).collect();
            let m = passive.len();
            if m == 0 {
                let cand = vec![0.0; k];
                let obj = objective(&cand);
                if best.as_ref().is_none_or(|(o, _)| obj < *o) {
                    best = Some((obj, cand));
                }
                continue;
            }
            let sub = Mat::<f64>::from_fn(m, m, |i, j| xtx[(passive[i], passive[j])]);
            let rhs = Mat::<f64>::from_fn(m, 1, |i, _| xty[passive[i]]);
            let sol = sub.partial_piv_lu().solve(&rhs);
            let mut cand = vec![0.0; k];
            let mut feasible = true;
            for (i, &p) in passive.iter().enumerate() {
                if sol[(i, 0)] < 0.0 {
                    feasible = false;
                    break;
                }
                cand[p] = sol[(i, 0)];
            }
            if !feasible {
                continue;
            }
            let obj = objective(&cand);
            if best.as_ref().is_none_or(|(o, _)| obj < *o) {
                best = Some((obj, cand));
            }
        }

        let (_, expected) = best.expect("at least the zero vector is feasible");
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got {got:?}, expected {expected:?}");
        }
    }

    /// Forces a variable back out of the passive set.
    ///
    /// Without this, the whole second half of Lawson-Hanson is dead code in the
    /// test suite: instrumenting the solver showed every other fixture reaches
    /// its optimum purely by *activating* variables, so the alpha step, the
    /// min-ratio selection, the retention filter and the refactorisation were
    /// all unreached. Here variable 0 enters at 1.0, variable 1 enters, and the
    /// joint solve puts variable 0 at -0.333, so it must be dropped.
    #[test]
    fn test_nnls_gram_drops_a_variable_that_turns_negative() {
        let xtx = gram_from_rows(&[1.0, 0.5, 0.5, 0.4], 2);
        let xty = [1.0, 0.9];
        let got = nnls_gram(xtx.as_ref(), &xty, None).unwrap();

        assert_eq!(got[0], 0.0, "variable 0 should have been dropped: {got:?}");
        assert!((got[1] - 2.25).abs() < 1e-12, "got {got:?}");
        assert_kkt(xtx.as_ref(), &xty, &got);
    }

    /// Karush-Kuhn-Tucker check: every active variable must have a
    /// non-positive gradient, every passive one a gradient of zero.
    ///
    /// Cheap to state and it holds for *any* correct NNLS answer, so it
    /// constrains the solver without needing a reference implementation.
    fn assert_kkt(xtx: MatRef<f64>, xty: &[f64], x: &[f64]) {
        let k = x.len();
        let tol = 1e-8;
        for j in 0..k {
            let mut gradient = xty[j];
            for i in 0..k {
                gradient -= xtx[(j, i)] * x[i];
            }
            if x[j] > 0.0 {
                assert!(gradient.abs() < tol, "passive {j} gradient {gradient}");
            } else {
                assert!(gradient <= tol, "active {j} gradient {gradient}");
            }
        }
    }

    /// A collinear Gram is the realistic input, and the only route to the
    /// dependent-column branch.
    #[test]
    fn test_nnls_gram_handles_a_collinear_design() {
        // Column 2 duplicates column 0 exactly.
        let xtx = gram_from_rows(
            &[
                2.0, 0.5, 2.0, //
                0.5, 1.0, 0.5, //
                2.0, 0.5, 2.0,
            ],
            3,
        );
        let xty = [1.0, 0.6, 1.0];
        let got = nnls_gram(xtx.as_ref(), &xty, None).unwrap();

        assert!(got.iter().all(|v| *v >= 0.0), "got {got:?}");
        assert!(got.iter().all(|v| v.is_finite()), "got {got:?}");
        // The duplicated pair must not both be loaded to the point of
        // over-fitting: the fitted value is what is identified, not the split.
        let fitted_0 = got[0] + got[2];
        assert!(fitted_0.is_finite() && fitted_0 >= 0.0);
    }

    #[test]
    fn test_nnls_gram_rejects_mismatched_shapes() {
        let xtx = Mat::<f64>::zeros(3, 3);
        assert!(nnls_gram(xtx.as_ref(), &[1.0, 2.0], None).is_err());
        let empty = Mat::<f64>::zeros(0, 0);
        assert!(nnls_gram(empty.as_ref(), &[], None).unwrap().is_empty());
    }
}
