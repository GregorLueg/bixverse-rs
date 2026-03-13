//! Helper functions for Principal Component type analyses with implementations
//! for randomised SVD on dense or sparse matrices.

use faer::{Mat, MatMut, MatRef};
use num_traits::Float;
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::prelude::*;

////////////////
// Structures //
////////////////

/// Structure for random SVD results
#[derive(Clone, Debug)]
pub struct RandomSvdResults<T> {
    /// Matrix u of the SVD decomposition
    pub u: Mat<T>,
    /// Matrix v of the SVD decomposition
    pub v: Mat<T>,
    /// Eigen vectors of the SVD decomposition
    pub s: Vec<T>,
}

/// Structure for SVD results
///
/// ### Fields
///
/// * `u` - Matrix u of the SVD decomposition
/// * `v` - Matrix v of the SVD decomposition
/// * `s` - Eigen vectors of the SVD decomposition
#[derive(Clone, Debug)]
pub struct SvdResults<T> {
    /// Matrix u of the SVD decomposition
    pub u: Mat<T>,
    /// Matrix v of the SVD decomposition
    pub v: Mat<T>,
    /// Eigen vectors of the SVD decomposition
    pub s: Vec<T>,
}

/// Trait to return the different matrices from the Svd Resuls
pub trait SvdResult<T> {
    /// Returns the matrix u of the SVD decomposition
    fn u(&self) -> &faer::Mat<T>;
    /// Returns the matrix v of the SVD decomposition
    fn v(&self) -> &faer::Mat<T>;
    /// Returns the eigen vectors of the SVD decomposition
    fn s(&self) -> &[T];
}

/// Implementations of the SvdResult trait for randomised SvdResults
impl<T> SvdResult<T> for RandomSvdResults<T> {
    fn u(&self) -> &faer::Mat<T> {
        &self.u
    }
    fn v(&self) -> &faer::Mat<T> {
        &self.v
    }
    fn s(&self) -> &[T] {
        &self.s
    }
}

/// Implementations of the SvdResult trait for SvdResults
impl<T> SvdResult<T> for SvdResults<T> {
    fn u(&self) -> &faer::Mat<T> {
        &self.u
    }
    fn v(&self) -> &faer::Mat<T> {
        &self.v
    }
    fn s(&self) -> &[T] {
        &self.s
    }
}

/////////////
// Helpers //
/////////////

/// Calculate the principal component scores from the SVD results
///
/// ### Params
///
/// * `svd_results` - The (randomised) SVD results
///
/// ### Returns
///
/// The principal component scores
pub fn compute_pc_scores<T, S>(svd_results: &S) -> Mat<T>
where
    T: Float,
    S: SvdResult<T>,
{
    let n_cells = svd_results.u().nrows();
    let n_pcs = svd_results.s().len();

    Mat::from_fn(n_cells, n_pcs, |i, j| {
        svd_results.u()[(i, j)] * svd_results.s()[j]
    })
}

///////////////
// Functions //
///////////////

/// Get the eigenvalues and vectors from a covar or cor matrix
///
/// Function will panic if the matrix is not symmetric
///
/// ### Params
///
/// * `matrix` - The correlation or co-variance matrix
/// * `top_n` - How many of the top eigen vectors and values to return.
///
/// ### Returns
///
/// A vector of tuples corresponding to the top eigen pairs.
pub fn get_top_eigenvalues<T>(matrix: &Mat<T>, top_n: usize) -> Vec<(T, Vec<T>)>
where
    T: BixverseFloat,
{
    // Ensure the matrix is square
    assert_symmetric_mat!(matrix);

    let eigendecomp = matrix.eigen().unwrap();

    let s = eigendecomp.S();
    let u = eigendecomp.U();

    // Extract the real part of the eigenvalues and vectors
    let mut eigenpairs = s
        .column_vector()
        .iter()
        .zip(u.col_iter())
        .map(|(l, v)| {
            let l_real = l.re;
            let v_real = v.iter().map(|v_i| v_i.re).collect::<Vec<T>>();
            (l_real, v_real)
        })
        .collect::<Vec<(T, Vec<T>)>>();

    // Sort and return Top N
    eigenpairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let res: Vec<(T, Vec<T>)> = eigenpairs.into_iter().take(top_n).collect();

    res
}

/// Randomised SVD
///
/// ### Params
///
/// * `x` - The matrix on which to apply the randomised SVD.
/// * `rank` - The target rank of the approximation (number of singular values,
///   vectors to compute).
/// * `seed` - Random seed for reproducible results.
/// * `oversampling` - Additional samples beyond the target rank to improve
///   accuracy. Defaults to 10 if not specified.
/// * `n_power_iter` - Number of power iterations to perform for better
///   approximation quality. More iterations generally improve accuracy but
///   increase computation time. Defaults to 2 if not specified.
///
/// ### Returns
///
/// The randomised SVD results in form of `RandomSvdResults`.
///
/// ### Algorithm Details
///
/// 1. Generate a random Gaussian matrix Ω of size n × (rank + oversampling)
/// 2. Compute Y = X * Ω to capture the range of X
/// 3. Orthogonalize Y using QR decomposition to get Q
/// 4. Apply power iterations: for each iteration, compute Z = X^T * Q, then Q = QR(X * Z)
/// 5. Form B = Q^T * X and compute its SVD
/// 6. Reconstruct the final SVD: U = Q * U_B, V = V_B, S = S_B
pub fn randomised_svd<T>(
    x: MatRef<T>,
    rank: usize,
    seed: usize,
    oversampling: Option<usize>,
    n_power_iter: Option<usize>,
) -> RandomSvdResults<T>
where
    T: BixverseFloat,
{
    let ncol = x.ncols();
    let nrow = x.nrows();
    let os = oversampling.unwrap_or(10);
    let sample_size = (rank + os).min(ncol.min(nrow));
    let n_iter = n_power_iter.unwrap_or(2);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Mat::from_fn(ncol, sample_size, |_, _| {
        T::from_f64(normal.sample(&mut rng)).unwrap()
    });

    let y = x * omega;
    let mut q = y.qr().compute_thin_Q();
    for _ in 0..n_iter {
        let z = x.transpose() * q;
        q = (x * z).qr().compute_thin_Q();
    }

    let b = q.transpose() * x;
    let svd = b.thin_svd().unwrap();
    RandomSvdResults {
        u: q * svd.U(),
        v: svd.V().cloned(),
        s: svd.S().column_vector().iter().copied().collect(),
    }
}

///////////////////////////
// Sparse randomised SVD //
///////////////////////////

/// Randomised sparse SVD - never forms dense intermediate matrices
///
/// ### Params
///
/// * `matrix` - Sparse matrix (CSR or CSC)
/// * `rank` - Target rank
/// * `seed` - For reproducibility
/// * `use_second_layer` - Whether to use the second layer of the sparse matrix
///   for SVD calculation.
/// * `oversampling` - Additional samples (default 10)
/// * `n_power_iter` - Power iterations for accuracy (default 2)
///
/// ### Returns
///
/// `RandomSvdResults` containing U (n×k), S (length k), and V (m×k)
#[allow(clippy::too_many_arguments)]
pub fn randomised_sparse_svd<T, F>(
    matrix: &CompressedSparseData2<T>,
    rank: usize,
    seed: u64,
    use_second_layer: bool,
    oversampling: Option<usize>,
    n_power_iter: Option<usize>,
    col_means: Option<&[F]>,
    col_stds: Option<&[F]>,
) -> RandomSvdResults<F>
where
    T: BixverseNumeric + Into<F>,
    F: BixverseFloat,
{
    let (n, m) = matrix.shape;
    let os = oversampling.unwrap_or(10);
    let sample_size = (rank + os).min(m).min(n);
    let n_iter = n_power_iter.unwrap_or(2);

    let csr_owned;
    let csr: &CompressedSparseData2<T> = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix,
        CompressedSparseFormat::Csc => {
            csr_owned = matrix.transform();
            &csr_owned
        }
    };

    let data_float: Vec<F> = if use_second_layer {
        csr.data_2
            .as_ref()
            .expect("data_2 is None but use_second_layer is true")
            .iter()
            .map(|&v| v.into())
            .collect()
    } else {
        csr.data.iter().map(|&v| v.into()).collect()
    };

    // pre-divide input (m × ncols) by col_stds once, avoiding per-nonzero division.
    let prescale = |x: MatRef<F>| -> Option<Mat<F>> {
        col_stds.map(|sd| Mat::from_fn(x.nrows(), x.ncols(), |i, col| x[(i, col)] / sd[i]))
    };

    // expects x_scaled to already be divided by σ if applicable.
    // y = A * x_scaled - 1 * (μᵀ * x_scaled)
    let sparse_matvec_a = |x_scaled: MatRef<F>, y: MatMut<F>| {
        let ncols = x_scaled.ncols();

        // μᵀ * x_scaled — a (1×m)*(m×ncols) product; faer handles SIMD internally.
        let mean_dots: Vec<F> = if let Some(mu) = col_means {
            let mu_row = MatRef::from_row_major_slice(mu, 1, m);
            let result = mu_row * x_scaled;
            (0..ncols).map(|col| result[(0, col)]).collect()
        } else {
            vec![]
        };

        let y_ptr = y.as_ptr_mut() as usize;
        let y_row_stride = y.row_stride();
        let y_col_stride = y.col_stride();

        (0..n).into_par_iter().for_each(|i| {
            let base = y_ptr as *mut F;

            for col in 0..ncols {
                unsafe {
                    *base.offset(i as isize * y_row_stride + col as isize * y_col_stride) =
                        F::zero();
                }
            }

            for idx in csr.indptr[i]..csr.indptr[i + 1] {
                let j = csr.indices[idx];
                let a_val = data_float[idx];
                for col in 0..ncols {
                    unsafe {
                        let ptr =
                            base.offset(i as isize * y_row_stride + col as isize * y_col_stride);
                        *ptr += a_val * x_scaled[(j, col)];
                    }
                }
            }

            if col_means.is_some() {
                for col in 0..ncols {
                    unsafe {
                        let ptr =
                            base.offset(i as isize * y_row_stride + col as isize * y_col_stride);
                        *ptr -= mean_dots[col];
                    }
                }
            }
        });
    };

    // y = (Aᵀx - μ * col_sumsᵀ) / σ
    // col_sums accumulated inside the fold — no separate O(n * ncols) pass.
    let sparse_matvec_at = |x: MatRef<F>, mut y: MatMut<F>| {
        let ncols = x.ncols();

        let (result, col_sums) = (0..n)
            .into_par_iter()
            .fold(
                || (vec![F::zero(); m * ncols], vec![F::zero(); ncols]),
                |(mut acc, mut cs), i| {
                    for col in 0..ncols {
                        cs[col] += x[(i, col)];
                    }
                    for idx in csr.indptr[i]..csr.indptr[i + 1] {
                        let j = csr.indices[idx];
                        let a_val = data_float[idx];
                        for col in 0..ncols {
                            acc[j * ncols + col] += a_val * x[(i, col)];
                        }
                    }
                    (acc, cs)
                },
            )
            .reduce(
                || (vec![F::zero(); m * ncols], vec![F::zero(); ncols]),
                |(mut a, mut cs_a), (b, cs_b)| {
                    for i in 0..a.len() {
                        a[i] += b[i];
                    }
                    for i in 0..ncols {
                        cs_a[i] += cs_b[i];
                    }
                    (a, cs_a)
                },
            );

        for j in 0..m {
            for col in 0..ncols {
                let mut val = result[j * ncols + col];
                if let Some(mu) = col_means {
                    val -= mu[j] * col_sums[col];
                }
                if let Some(sd) = col_stds {
                    val /= sd[j];
                }
                y[(j, col)] = val;
            }
        }
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Mat::from_fn(m, sample_size, |_, _| {
        F::from_f64(normal.sample(&mut rng)).unwrap()
    });

    let omega_ref = prescale(omega.as_ref());
    let mut y = Mat::<F>::zeros(n, sample_size);
    sparse_matvec_a(
        omega_ref
            .as_ref()
            .map(|m| m.as_ref())
            .unwrap_or(omega.as_ref()),
        y.as_mut(),
    );

    let mut q = y.qr().compute_thin_Q();

    // reuse buffers across power iterations.
    let mut z = Mat::<F>::zeros(m, sample_size);
    let mut y_new = Mat::<F>::zeros(n, sample_size);

    for _ in 0..n_iter {
        sparse_matvec_at(q.as_ref(), z.as_mut());
        let z_scaled = prescale(z.as_ref());
        sparse_matvec_a(
            z_scaled.as_ref().map(|m| m.as_ref()).unwrap_or(z.as_ref()),
            y_new.as_mut(),
        );
        q = y_new.qr().compute_thin_Q();
    }

    // B = Qᵀ * (A - 1μᵀ) / σ = sparse_matvec_at(Q)ᵀ — no duplicate fold needed.
    let mut b_t = Mat::<F>::zeros(m, sample_size);
    sparse_matvec_at(q.as_ref(), b_t.as_mut());
    let b = b_t.transpose().to_owned();

    let svd = b.thin_svd().unwrap();

    let u = &q * svd.U();
    let s: Vec<F> = svd.S().column_vector().iter().copied().collect();
    let v = svd.V().to_owned();

    RandomSvdResults { u, s, v }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_compute_pc_scores() {
        let u: Mat<f64> = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let v: Mat<f64> = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let s = vec![3.0, 1.5];

        let svd_res = SvdResults { u, v, s };
        let scores = compute_pc_scores(&svd_res);

        // PC scores = U * S
        assert!((scores[(0, 0)] - 3.0).abs() < 1e-6);
        assert!((scores[(0, 1)] - 0.0).abs() < 1e-6);
        assert!((scores[(1, 0)] - 0.0).abs() < 1e-6);
        assert!((scores[(1, 1)] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_get_top_eigenvalues() {
        let mat: Mat<f64> = Mat::from_fn(
            2,
            2,
            |i, j| if i == j { 3.0 - (i as f64 * 2.0) } else { 0.0 },
        );
        let top_eigen = get_top_eigenvalues(&mat, 2);

        assert_eq!(top_eigen.len(), 2);
        assert!((top_eigen[0].0 - 3.0).abs() < 1e-6);
        assert!((top_eigen[1].0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_randomised_svd_logic() {
        // Create a dense rank-1 matrix A = x * y^T
        // x = [1.0, 2.0, 3.0, 4.0]^T
        // y = [1.0, 0.5, 0.25]^T
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 0.5, 0.25];
        let mut mat: Mat<f64> = Mat::zeros(4, 3);
        for i in 0..4 {
            for j in 0..3 {
                mat[(i, j)] = x[i] * y[j];
            }
        }

        // We only need the top 1 PC
        let svd = randomised_svd(mat.as_ref(), 1, 42, Some(5), Some(4));

        // Test U (Left Singular Vector) correlation with x
        let u_col = svd.u.col(0);
        let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut dot_u = 0.0;
        for i in 0..4 {
            dot_u += u_col[i] * (x[i] / x_norm);
        }

        // Test V (Right Singular Vector) correlation with y
        let v_col = svd.v.col(0);
        let y_norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut dot_v = 0.0;
        for j in 0..3 {
            dot_v += v_col[j] * (y[j] / y_norm);
        }

        // The absolute correlation should be > 0.999 (allowing for sign flips)
        assert!(dot_u.abs() > 0.999);
        assert!(dot_v.abs() > 0.999);
    }

    #[test]
    fn test_randomised_sparse_svd_logic() {
        // Create a sparse rank-1 matrix A = x * y^T
        // x = [0.0, 2.0, 0.0, 4.0]^T
        // y = [1.0, 0.0, 0.5]^T
        // Non-zeros will only exist where x_i != 0 AND y_j != 0
        let data = vec![2.0, 1.0, 4.0, 2.0];
        let indices = vec![0, 2, 0, 2];
        let indptr = vec![0, 0, 2, 2, 4];
        let shape = (4, 3);

        let csr = CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, shape);

        let no_params: Option<&[f64]> = None;
        let svd = randomised_sparse_svd(&csr, 1, 42, false, Some(5), Some(4), no_params, no_params);

        let u_col = svd.u.col(0);
        let x_norm = (2.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt(); // sqrt(20)
        let dot_u = (u_col[1] * 2.0 + u_col[3] * 4.0) / x_norm;
        assert!(dot_u.abs() > 0.999);

        let v_col = svd.v.col(0);
        let y_norm = (1.0_f64.powi(2) + 0.5_f64.powi(2)).sqrt(); // sqrt(1.25)
        let dot_v = (v_col[0] * 1.0 + v_col[2] * 0.5) / y_norm;
        assert!(dot_v.abs() > 0.999);
    }
}
