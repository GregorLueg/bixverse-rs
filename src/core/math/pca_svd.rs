//! Helper functions for Principal Component type analyses with implementations
//! for randomised SVD on dense or sparse matrices.

use faer::{Mat, MatRef};
use num_traits::Float;
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

use super::*;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Column standard deviation below which [`sparse_covariance_svd`] treats a
/// feature as constant. Its centred cross-products are pure rounding residue
/// (about `n x^2 eps` for a constant `x`), and dividing that by `sd^2` would
/// hand the eigenproblem a huge spurious variance. Matches the dense path's
/// zero-variance cut-off.
const COV_ZERO_SD: f64 = 1e-8;

/// Rows per block when transposing between row-major scratch and `Mat`
const TRANSPOSE_ROW_BLOCK: usize = 64;

/// Columns written per task when filling a `Mat` from a row-major source
const TRANSPOSE_COL_BLOCK: usize = 8;

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
pub fn get_top_eigenvalues<T>(
    matrix: &Mat<T>,
    top_n: usize,
) -> Result<Vec<(T, Vec<T>)>, BixverseErrors>
where
    T: BixverseFloat,
{
    // Ensure the matrix is square
    assert_symmetric_mat!(matrix);

    let eigendecomp = matrix.eigen().map_err(|_| BixverseErrors::FaerEigenError)?;

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

    Ok(res)
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
) -> Result<RandomSvdResults<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let ncol = x.ncols();
    let nrow = x.nrows();
    let os = oversampling.unwrap_or(DEFAULT_OVERSAMPLING_RAND_SVD);
    let sample_size = (rank + os).min(ncol.min(nrow));
    let n_iter = n_power_iter.unwrap_or(DEFAULT_N_POWER_ITERS_RAND_SVD);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Mat::from_fn(ncol, sample_size, |_, _| {
        T::from_f64(normal.sample(&mut rng)).unwrap()
    });

    let y = x * omega;
    let mut q = orthonormalise(y);
    for _ in 0..n_iter {
        let z = x.transpose() * q;
        q = orthonormalise(x * z);
    }

    let b = q.transpose() * x;
    let svd = b
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
    let u = q * svd.U();

    Ok(RandomSvdResults {
        u,
        v: svd.V().cloned(),
        s: svd.S().column_vector().iter().copied().collect(),
    })
}

////////////////////////////////
// Matrix-free randomised SVD //
////////////////////////////////

/// Randomised SVD for a linear operator `M` defined only through its
/// matrix-vector product closures. Used by anchor-based batch correction
/// (Seurat-CCA) where `M = X1^T @ X2` would materialise an N1 x N2 dense
/// intermediate; factoring through `M @ V = X1^T @ (X2 @ V)` keeps memory at
/// `O((N1 + N2) * (rank + oversampling))`.
///
/// ### Params
///
/// * `n_rows` - Rows of the implicit operator `M`
/// * `n_cols` - Columns of the implicit operator `M`
/// * `rank` - Target number of singular triples
/// * `seed` - Random seed for the Gaussian sketch
/// * `oversampling` - Extra samples on top of `rank`. Defaults to
///   [DEFAULT_OVERSAMPLING_RAND_SVD]
/// * `n_power_iter` - Power iterations. Defaults to
///   [DEFAULT_N_POWER_ITERS_RAND_SVD]
/// * `apply` - Closure computing `M @ V` for a block `V` of shape
///   `(n_cols, k)`, returning `(n_rows, k)`
/// * `apply_t` - Closure computing `M^T @ U` for a block `U` of shape
///   `(n_rows, k)`, returning `(n_cols, k)`
///
/// ### Returns
///
/// [RandomSvdResults] with `u: (n_rows, rank+os)`, `v: (n_cols, rank+os)`
/// and `s` singular values. Consumers should truncate to `rank` themselves.
#[allow(clippy::too_many_arguments)]
pub fn randomised_svd_matfree<T, FA, FT>(
    n_rows: usize,
    n_cols: usize,
    rank: usize,
    seed: usize,
    oversampling: Option<usize>,
    n_power_iter: Option<usize>,
    apply: FA,
    apply_t: FT,
) -> Result<RandomSvdResults<T>, BixverseErrors>
where
    T: BixverseFloat,
    FA: Fn(MatRef<T>) -> Mat<T>,
    FT: Fn(MatRef<T>) -> Mat<T>,
{
    let os = oversampling.unwrap_or(DEFAULT_OVERSAMPLING_RAND_SVD);
    let sample_size = (rank + os).min(n_cols.min(n_rows));
    let n_iter = n_power_iter.unwrap_or(DEFAULT_N_POWER_ITERS_RAND_SVD);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Mat::from_fn(n_cols, sample_size, |_, _| {
        T::from_f64(normal.sample(&mut rng)).unwrap()
    });

    let y = apply(omega.as_ref());
    let mut q = y.qr().compute_thin_Q();

    for _ in 0..n_iter {
        let z = apply_t(q.as_ref());
        q = apply(z.as_ref()).qr().compute_thin_Q();
    }

    // B = Q^T @ M has shape (sample_size, n_cols). We only have M^T @ Q,
    // so form B via its transpose: (M^T @ Q).T = Q^T @ M.
    let bt = apply_t(q.as_ref());
    let b = bt.transpose().to_owned();

    let svd = b
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;

    Ok(RandomSvdResults {
        u: q * svd.U(),
        v: svd.V().cloned(),
        s: svd.S().column_vector().iter().copied().collect(),
    })
}

///////////////////////////
// Sparse randomised SVD //
///////////////////////////

/// Randomised sparse SVD - never forms dense intermediate matrices
///
/// ### Params
///
/// * `matrix` - Sparse matrix (CSR or CSC), consumed
/// * `rank` - Target rank
/// * `seed` - For reproducibility
/// * `use_second_layer` - Whether to use the second layer of the sparse matrix
///   for SVD calculation.
/// * `oversampling` - Additional samples (default 10)
/// * `n_power_iter` - Power iterations for accuracy (default 2)
/// * `col_means` - Optional column means for implicit mean centering
/// * `col_stds` - Optional column sds for implicit variance normalising
/// * `row_offsets` - Additional offsets (for example for CLR-type PCA in single
///   cell).
///
/// ### Returns
///
/// `RandomSvdResults` containing U (n x k), S (length k), and V (m x k)
#[allow(clippy::too_many_arguments)]
pub fn randomised_sparse_svd<T, F>(
    matrix: CompressedSparseData2<T>,
    rank: usize,
    seed: u64,
    use_second_layer: bool,
    oversampling: Option<usize>,
    n_power_iter: Option<usize>,
    col_means: Option<&[F]>,
    col_stds: Option<&[F]>,
    row_offsets: Option<&[F]>,
) -> Result<RandomSvdResults<F>, BixverseErrors>
where
    T: BixverseNumeric + Into<F>,
    F: BixverseFloat,
{
    let (n, m) = matrix.shape;
    let os = oversampling.unwrap_or(DEFAULT_OVERSAMPLING_RAND_SVD);
    let sample_size = (rank + os).min(m).min(n);
    let n_iter = n_power_iter.unwrap_or(DEFAULT_N_POWER_ITERS_RAND_SVD);

    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix,
        CompressedSparseFormat::Csc => matrix.transform_single_layer(use_second_layer)?,
    };
    let csr_data: &[T] = if use_second_layer {
        csr.data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?
            .as_slice()
    } else {
        csr.data.as_slice()
    };
    let val = |idx: usize| -> f64 { Into::<F>::into(csr_data[idx]).to_f64().unwrap() };

    let mu: Option<Vec<f64>> = col_means.map(|v| v.iter().map(|x| x.to_f64().unwrap()).collect());
    let sd: Option<Vec<f64>> = col_stds.map(|v| v.iter().map(|x| x.to_f64().unwrap()).collect());
    let off: Option<Vec<f64>> =
        row_offsets.map(|v| v.iter().map(|x| x.to_f64().unwrap()).collect());

    // row-major f64 copy of a faer matrix, optionally dividing row j by sd[j]
    let to_row_major = |x: MatRef<F>, scale: Option<&[f64]>| -> Vec<f64> {
        let k = x.ncols();
        let mut out = vec![0f64; x.nrows() * k];
        // Row blocks, feature outer: the reads walk contiguous column slices and
        // the strided writes stay inside one cache-resident block.
        out.par_chunks_mut(TRANSPOSE_ROW_BLOCK * k)
            .enumerate()
            .for_each(|(b, block)| {
                let j0 = b * TRANSPOSE_ROW_BLOCK;
                for c in 0..k {
                    for r in 0..block.len() / k {
                        let d = scale.map(|s| s[j0 + r]).unwrap_or(1.0);
                        block[r * k + c] = x[(j0 + r, c)].to_f64().unwrap() / d;
                    }
                }
            });
        out
    };

    // y = (A - o 1^T - 1 mu^T) D^-1 x, row by row over CSR
    let sparse_matvec_a = |x: MatRef<F>| -> Mat<F> {
        let k = x.ncols();
        let x_rm = to_row_major(x, sd.as_deref());
        let mean_dots: Vec<f64> = match &mu {
            Some(mu) => (0..k)
                .map(|c| (0..m).map(|j| mu[j] * x_rm[j * k + c]).sum())
                .collect(),
            None => vec![0.0; k],
        };
        let x_sums: Vec<f64> = match &off {
            Some(_) => (0..k)
                .map(|c| (0..m).map(|j| x_rm[j * k + c]).sum())
                .collect(),
            None => vec![0.0; k],
        };

        let mut y_rm = vec![0f64; n * k];
        y_rm.par_chunks_mut(k).enumerate().for_each(|(i, acc)| {
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                let a = val(idx);
                let xr = &x_rm[csr.indices[idx] as usize * k..][..k];
                for (acc_c, &x_c) in acc.iter_mut().zip(xr) {
                    *acc_c += a * x_c;
                }
            }
            let o_i = off.as_ref().map(|o| o[i]).unwrap_or(0.0);
            for c in 0..k {
                acc[c] -= mean_dots[c] + o_i * x_sums[c];
            }
        });
        mat_from_row_major(n, k, |i, c| F::from_f64(y_rm[i * k + c]).unwrap())
    };

    // y = D^-1 (A - o 1^T - 1 mu^T)^T x. One contiguous block of rows per
    // thread, each with its own m x k accumulator (small enough for L2), so x
    // is streamed once in order rather than gathered at random.
    let n_threads = rayon::current_num_threads().max(1);
    let sparse_matvec_at = |x: MatRef<F>| -> Mat<F> {
        let k = x.ncols();
        let x_rm = to_row_major(x, None);
        let col_sums: Vec<f64> = (0..k)
            .map(|c| (0..n).map(|i| x_rm[i * k + c]).sum())
            .collect();
        let o_dot_x: Vec<f64> = match &off {
            Some(o) => (0..k)
                .map(|c| (0..n).map(|i| o[i] * x_rm[i * k + c]).sum())
                .collect(),
            None => vec![0.0; k],
        };

        let chunk = n.div_ceil(n_threads).max(1);
        let y_rm = (0..n)
            .step_by(chunk)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|start| {
                let mut acc = vec![0f64; m * k];
                for i in start..(start + chunk).min(n) {
                    let xr = &x_rm[i * k..(i + 1) * k];
                    for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                        let a = val(idx);
                        let j = csr.indices[idx] as usize;
                        for (acc_c, &x_c) in acc[j * k..(j + 1) * k].iter_mut().zip(xr) {
                            *acc_c += a * x_c;
                        }
                    }
                }
                acc
            })
            .reduce(
                || vec![0f64; m * k],
                |mut a, b| {
                    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y);
                    a
                },
            );

        mat_from_row_major(m, k, |j, c| {
            let mu_j = mu.as_ref().map(|v| v[j]).unwrap_or(0.0);
            let sd_j = sd.as_ref().map(|v| v[j]).unwrap_or(1.0);
            F::from_f64((y_rm[j * k + c] - mu_j * col_sums[c] - o_dot_x[c]) / sd_j).unwrap()
        })
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Mat::from_fn(m, sample_size, |_, _| {
        F::from_f64(normal.sample(&mut rng)).unwrap()
    });

    let y = sparse_matvec_a(omega.as_ref());
    drop(omega);

    let mut q = orthonormalise(y);
    for _ in 0..n_iter {
        let z = sparse_matvec_at(q.as_ref());
        q = orthonormalise(sparse_matvec_a(z.as_ref()));
    }

    let b = sparse_matvec_at(q.as_ref()).transpose().to_owned();
    let svd = b
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;

    let u = &q * svd.U();
    let s: Vec<F> = svd.S().column_vector().iter().copied().collect();
    let v = svd.V().to_owned();

    Ok(RandomSvdResults { u, s, v })
}

/// Build a column-major matrix from a function over `(row, column)`
///
/// Parallel over small column blocks with rows outer, so a row-major source is
/// read one cache line at a time rather than once per column.
///
/// ### Params
///
/// * `n` - Number of rows
/// * `k` - Number of columns
/// * `f` - Entry at `(row, column)`
///
/// ### Returns
///
/// The `n x k` matrix.
fn mat_from_row_major<F, G>(n: usize, k: usize, f: G) -> Mat<F>
where
    F: BixverseFloat,
    G: Fn(usize, usize) -> F + Sync,
{
    let mut out = Mat::<F>::zeros(n, k);
    out.par_col_chunks_mut(TRANSPOSE_COL_BLOCK)
        .enumerate()
        .for_each(|(b, mut block)| {
            let c0 = b * TRANSPOSE_COL_BLOCK;
            for i in 0..n {
                for c in 0..block.ncols() {
                    block[(i, c)] = f(i, c0 + c);
                }
            }
        });
    out
}

/// Orthonormalise the columns of a tall matrix
///
/// CholeskyQR2: two passes of `Q = Y L^-T` with `L L^T = Y^T Y`. Each pass is
/// two tall GEMMs plus a `k x k` Cholesky, which parallelises far better than
/// a Householder QR of a tall-skinny matrix. The second pass repairs the
/// orthogonality the first loses when `Y` is ill-conditioned. Falls back to
/// Householder QR if the Cholesky factorisation fails.
///
/// ### Params
///
/// * `y` - Tall matrix (n x k, n >= k) to orthonormalise, consumed
///
/// ### Returns
///
/// `Q` (n x k) with orthonormal columns spanning the range of `y`.
fn orthonormalise<F: BixverseFloat>(y: Mat<F>) -> Mat<F> {
    fn cholesky_qr<F: BixverseFloat>(y: &Mat<F>) -> Option<Mat<F>> {
        let k = y.ncols();
        let gram = y.transpose() * y;
        let llt = gram.llt(faer::Side::Lower).ok()?;
        // Q = Y L^-T, with the small k x k inverse formed explicitly so the
        // tall product is a single GEMM
        let mut l_inv = Mat::<F>::identity(k, k);
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            llt.L(),
            l_inv.as_mut(),
            faer::Par::Seq,
        );
        Some(y * l_inv.transpose())
    }

    match cholesky_qr(&y).and_then(|q1| cholesky_qr(&q1)) {
        Some(q) => q,
        None => y.qr().compute_thin_Q(),
    }
}

///////////////////////////
// Dense covariance PCA //
///////////////////////////

/// Exact PCA of an already centred and scaled dense matrix via `X^T X`
///
/// One GEMM for the `m x m` cross-product, a symmetric eigendecomposition,
/// and `U = X V / s`. Costs `2 n m^2`, so it only beats a randomised SVD when
/// `m` is small relative to the sketch width times the number of passes.
/// Stays in f64: the eigenvalues of the cross-product are the squared
/// singular values, which halves the usable precision.
///
/// ### Params
///
/// * `x` - Centred (and optionally scaled) matrix, samples x features
/// * `rank` - Number of components to return
///
/// ### Returns
///
/// `RandomSvdResults` containing U (n x rank), S (length rank), and V
/// (m x rank).
pub fn dense_covariance_svd(
    x: MatRef<f64>,
    rank: usize,
) -> Result<RandomSvdResults<f64>, BixverseErrors> {
    let (n, m) = (x.nrows(), x.ncols());
    let rank = rank.min(m).min(n);

    let gram = x.transpose() * x;
    let (eig_vals, eig_vecs) =
        crate::core::math::sparse::symmetric_eigen_descending(gram.as_ref(), m)?;

    let s: Vec<f64> = eig_vals[..rank]
        .iter()
        .map(|&l| l.max(0.0).sqrt())
        .collect();
    let v = eig_vecs.subcols(0, rank).to_owned();
    let mut u = x * &v;
    for (c, &s_c) in s.iter().enumerate() {
        let inv = if s_c > 0.0 { 1.0 / s_c } else { 0.0 };
        u.col_mut(c).iter_mut().for_each(|x| *x *= inv);
    }

    Ok(RandomSvdResults { u, v, s })
}

///////////////////////////
// Sparse covariance PCA //
///////////////////////////

/// Exact sparse PCA via the gene-gene cross-product
///
/// Forms `Z^T Z` for the implicitly shifted and scaled matrix
/// `Z = (A - o 1^T - 1 c^T) D^-1` without densifying `A`, then
/// eigendecomposes it. `A^T A` comes from one pass over the rows: each row
/// with `r` non-zeros adds its `r^2 / 2` outer products to the upper triangle,
/// and all updates for one entry land in one row of the accumulator, which
/// stays in L1. Centring, the CLR offsets and scaling are rank-one
/// corrections applied afterwards in f64. Cost is `sum(r_i^2) / 2` plus an
/// `m x m` eigendecomposition, so this pays off when `m` is a few thousand
/// features and rows are sparse, as with HVG-restricted single cell data.
///
/// ### Params
///
/// * `matrix` - Sparse matrix (CSR or CSC), cells x features, consumed
/// * `rank` - Number of components to return
/// * `use_second_layer` - Whether to use the second layer of the sparse matrix
/// * `col_means` - Optional column means for implicit mean centring. With
///   `row_offsets`, these are the means of the offset-corrected matrix, as
///   returned by `sparse_csc_column_means`.
/// * `col_stds` - Optional column sds for implicit variance normalising
/// * `row_offsets` - Additional offsets (for example for CLR-type PCA in single
///   cell).
///
/// ### Returns
///
/// `RandomSvdResults` containing U (n x rank), S (length rank), and V
/// (m x rank). Memory is one `m x m` f64 accumulator per Rayon thread.
#[allow(clippy::too_many_arguments)]
pub fn sparse_covariance_svd<T, F>(
    matrix: CompressedSparseData2<T>,
    rank: usize,
    use_second_layer: bool,
    col_means: Option<&[F]>,
    col_stds: Option<&[F]>,
    row_offsets: Option<&[F]>,
) -> Result<RandomSvdResults<F>, BixverseErrors>
where
    T: BixverseNumeric + Into<F>,
    F: BixverseFloat,
{
    let (n, m) = matrix.shape;
    let rank = rank.min(m).min(n);
    let n_f = n as f64;

    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix,
        CompressedSparseFormat::Csc => matrix.transform_single_layer(use_second_layer)?,
    };
    let values: &[T] = if use_second_layer {
        csr.data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?
            .as_slice()
    } else {
        csr.data.as_slice()
    };
    let val = |idx: usize| -> f64 { Into::<F>::into(values[idx]).to_f64().unwrap() };

    let off: Option<Vec<f64>> =
        row_offsets.map(|o| o.iter().map(|x| x.to_f64().unwrap()).collect());
    // column shift c: the means of A - o 1^T when centring, else nothing
    let shift: Vec<f64> = match col_means {
        Some(mu) => mu.iter().map(|x| x.to_f64().unwrap()).collect(),
        None => vec![0.0; m],
    };
    let sd: Vec<f64> = match col_stds {
        Some(s) => s.iter().map(|x| x.to_f64().unwrap()).collect(),
        None => vec![1.0; m],
    };

    // relabel features by descending frequency: most pair updates then hit the
    // small, cache-resident top-left block of the accumulator
    let mut freq = vec![0usize; m];
    for &j in csr.indices.iter() {
        freq[j as usize] += 1;
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by(|&a, &b| freq[b].cmp(&freq[a]));
    let mut relabel = vec![0usize; m];
    for (new_j, &old_j) in order.iter().enumerate() {
        relabel[old_j] = new_j;
    }

    // pass 1: one triangle of A^T A (in relabelled space, rows are no longer
    // sorted, so an entry can land in either triangle), the column sums
    // A^T 1 and w = A^T o
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(n_threads).max(1);
    let (gram, col_sums, w) = (0..n)
        .step_by(chunk)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let mut g = vec![0f64; m * m];
            let mut cs = vec![0f64; m];
            let mut w = vec![0f64; m];
            let mut row_vals: Vec<f64> = Vec::new();
            let mut row_idx: Vec<usize> = Vec::new();
            for i in start..(start + chunk).min(n) {
                let (lo, hi) = (csr.indptr[i] as usize, csr.indptr[i + 1] as usize);
                row_vals.clear();
                row_vals.extend((lo..hi).map(val));
                row_idx.clear();
                row_idx.extend(csr.indices[lo..hi].iter().map(|&j| relabel[j as usize]));
                for (p, (&a, &va)) in row_idx.iter().zip(row_vals.iter()).enumerate() {
                    cs[a] += va;
                    if let Some(o) = &off {
                        w[a] += va * o[i];
                    }
                    let row = &mut g[a * m..(a + 1) * m];
                    for (&b, &vb) in row_idx[p..].iter().zip(row_vals[p..].iter()) {
                        row[b] += va * vb;
                    }
                }
            }
            (g, cs, w)
        })
        .reduce(
            || (vec![0f64; m * m], vec![0f64; m], vec![0f64; m]),
            |(mut g1, mut cs1, mut w1), (g2, cs2, w2)| {
                g1.par_iter_mut()
                    .zip(g2.par_iter())
                    .for_each(|(a, b)| *a += b);
                cs1.iter_mut().zip(cs2.iter()).for_each(|(a, b)| *a += b);
                w1.iter_mut().zip(w2.iter()).for_each(|(a, b)| *a += b);
                (g1, cs1, w1)
            },
        );

    let (o_sum, o_sq): (f64, f64) = off
        .as_ref()
        .map(|o| (o.iter().sum(), o.iter().map(|x| x * x).sum()))
        .unwrap_or((0.0, 0.0));

    // Z^T Z = A^T A - w 1^T - 1 w^T - s c^T - c s^T + (o . o) 1 1^T
    //         + (1^T o)(1 c^T + c 1^T) + n c c^T, then scaled,
    // with s = A^T 1 and w = A^T o
    let gram_ref = &gram;
    let constant: Vec<bool> = sd.iter().map(|&s| s < COV_ZERO_SD).collect();
    let cov_entry = |a: usize, b: usize| -> f64 {
        if constant[a] || constant[b] {
            return 0.0;
        }
        let (ra, rb) = (relabel[a], relabel[b]);
        let ata = if ra == rb {
            gram_ref[ra * m + ra]
        } else {
            gram_ref[ra * m + rb] + gram_ref[rb * m + ra]
        };
        let (sa, sb) = (col_sums[ra], col_sums[rb]);
        let (ca, cb) = (shift[a], shift[b]);
        let raw =
            ata - w[ra] - w[rb] - sa * cb - ca * sb + o_sq + o_sum * (ca + cb) + n_f * ca * cb;
        raw / (sd[a] * sd[b])
    };
    let mut cov = Mat::<f64>::zeros(m, m);
    cov.par_col_iter_mut().enumerate().for_each(|(b, mut col)| {
        for a in 0..m {
            col[a] = cov_entry(a, b);
        }
    });
    drop(gram);

    let (eig_vals, eig_vecs) =
        crate::core::math::sparse::symmetric_eigen_descending(cov.as_ref(), m)?;

    let s: Vec<f64> = eig_vals[..rank]
        .iter()
        .map(|&l| l.max(0.0).sqrt())
        .collect();
    // V scaled by 1 / sd, row-major (m x rank) for the score pass
    let v_scaled: Vec<f64> = (0..m)
        .flat_map(|j| (0..rank).map(move |c| (j, c)))
        .map(|(j, c)| {
            if constant[j] {
                0.0
            } else {
                eig_vecs[(j, c)] / sd[j]
            }
        })
        .collect();
    let shift_v: Vec<f64> = (0..rank)
        .map(|c| (0..m).map(|j| shift[j] * v_scaled[j * rank + c]).sum())
        .collect();
    let one_v: Vec<f64> = (0..rank)
        .map(|c| (0..m).map(|j| v_scaled[j * rank + c]).sum())
        .collect();

    // pass 2: U = Z V / s, row by row
    let u_rm: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut acc = vec![0f64; rank];
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                let j = csr.indices[idx] as usize;
                let a = val(idx);
                let vr = &v_scaled[j * rank..(j + 1) * rank];
                for c in 0..rank {
                    acc[c] += a * vr[c];
                }
            }
            let oi = off.as_ref().map(|o| o[i]).unwrap_or(0.0);
            for c in 0..rank {
                let z = acc[c] - shift_v[c] - oi * one_v[c];
                acc[c] = if s[c] > 0.0 { z / s[c] } else { 0.0 };
            }
            acc
        })
        .collect();
    Ok(RandomSvdResults {
        u: mat_from_row_major(n, rank, |i, c| F::from_f64(u_rm[i * rank + c]).unwrap()),
        v: Mat::from_fn(m, rank, |j, c| F::from_f64(eig_vecs[(j, c)]).unwrap()),
        s: s.iter().map(|&x| F::from_f64(x).unwrap()).collect(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// PC scores are `U * S`, so an identity `U` hands the singular values straight back.
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

    /// Eigenvalues come back in descending order.
    #[test]
    fn test_get_top_eigenvalues() {
        let mat: Mat<f64> = Mat::from_fn(
            2,
            2,
            |i, j| if i == j { 3.0 - (i as f64 * 2.0) } else { 0.0 },
        );
        let top_eigen = get_top_eigenvalues(&mat, 2).unwrap();

        assert_eq!(top_eigen.len(), 2);
        assert!((top_eigen[0].0 - 3.0).abs() < 1e-6);
        assert!((top_eigen[1].0 - 1.0).abs() < 1e-6);
    }

    /// Randomised SVD recovers the singular vectors of a dense rank-one matrix, up to sign.
    #[test]
    fn test_randomised_svd_logic() {
        // Create a dense rank-1 matrix A = x * y^T
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 0.5, 0.25];
        let mut mat: Mat<f64> = Mat::zeros(4, 3);
        for i in 0..4 {
            for j in 0..3 {
                mat[(i, j)] = x[i] * y[j];
            }
        }

        // We only need the top 1 PC
        let svd = randomised_svd(mat.as_ref(), 1, 42, Some(5), Some(4)).unwrap();

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

    /// The matrix-free path must agree with the dense one on the same operator.
    #[test]
    fn test_randomised_svd_matfree_matches_dense() {
        // Build A = X1^T @ X2 explicitly, then compare matfree(X1, X2) SVD
        // to dense SVD of A.
        let n_hvg = 5;
        let n1 = 6;
        let n2 = 4;
        let rank = 2;

        let x1: Mat<f64> = Mat::from_fn(n_hvg, n1, |i, j| ((i + 1) as f64 * 0.3 + j as f64).sin());
        let x2: Mat<f64> = Mat::from_fn(n_hvg, n2, |i, j| ((j + 2) as f64 * 0.7 + i as f64).cos());

        let a = x1.transpose() * &x2;
        let dense = randomised_svd(a.as_ref(), rank, 42, Some(5), Some(4)).unwrap();

        let apply = |v: MatRef<f64>| -> Mat<f64> { x1.transpose() * (&x2 * v) };
        let apply_t = |u: MatRef<f64>| -> Mat<f64> { x2.transpose() * (&x1 * u) };
        let matfree =
            randomised_svd_matfree(n1, n2, rank, 42, Some(5), Some(4), apply, apply_t).unwrap();

        // Compare singular values (matfree may return more than rank; truncate).
        for k in 0..rank {
            assert!(
                (dense.s[k] - matfree.s[k]).abs() < 1e-6,
                "singular value {k}: dense {} vs matfree {}",
                dense.s[k],
                matfree.s[k]
            );
        }

        // Absolute correlation of top singular vectors (up to sign).
        for k in 0..rank {
            let mut dot_u = 0.0_f64;
            for i in 0..n1 {
                dot_u += dense.u[(i, k)] * matfree.u[(i, k)];
            }
            let mut dot_v = 0.0_f64;
            for j in 0..n2 {
                dot_v += dense.v[(j, k)] * matfree.v[(j, k)];
            }
            assert!(dot_u.abs() > 0.999, "u[{k}] |dot| = {}", dot_u.abs());
            assert!(dot_v.abs() > 0.999, "v[{k}] |dot| = {}", dot_v.abs());
        }
    }

    /// The sparse path recovers a rank-one factorisation despite two structurally empty rows.
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
        let svd = randomised_sparse_svd(
            csr,
            1,
            42,
            false,
            Some(5),
            Some(4),
            no_params,
            no_params,
            None,
        )
        .unwrap();

        let u_col = svd.u.col(0);
        let x_norm = (2.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt(); // sqrt(20)
        let dot_u = (u_col[1] * 2.0 + u_col[3] * 4.0) / x_norm;
        assert!(dot_u.abs() > 0.999);

        let v_col = svd.v.col(0);
        let y_norm = (1.0_f64.powi(2) + 0.5_f64.powi(2)).sqrt(); // sqrt(1.25)
        let dot_v = (v_col[0] * 1.0 + v_col[2] * 0.5) / y_norm;
        assert!(dot_v.abs() > 0.999);
    }

    /// A constant non-zero feature contributes nothing, rather than its
    /// centring residue blown up by a tiny standard deviation.
    #[test]
    fn test_sparse_covariance_svd_ignores_constant_feature() {
        let (n, m) = (40, 6);
        let constant = m - 1;
        let dense = Mat::<f64>::from_fn(n, m, |i, j| {
            if j == constant {
                2.0
            } else if (i * 7 + j * 3) % 4 == 0 {
                ((i * 13 + j * 5) % 11) as f64 * 0.25 + 0.5
            } else {
                0.0
            }
        });

        let (mut data, mut indices, mut indptr) = (Vec::new(), Vec::new(), vec![0u32]);
        for j in 0..m {
            for i in 0..n {
                if dense[(i, j)] != 0.0 {
                    data.push(dense[(i, j)]);
                    indices.push(i as u32);
                }
            }
            indptr.push(data.len() as u32);
        }

        let means: Vec<f64> = (0..m)
            .map(|j| dense.col(j).iter().sum::<f64>() / n as f64)
            .collect();
        let sds: Vec<f64> = (0..m)
            .map(|j| {
                let ss: f64 = dense.col(j).iter().map(|&x| (x - means[j]).powi(2)).sum();
                (ss / (n - 1) as f64).sqrt().max(f64::EPSILON)
            })
            .collect();

        let scaled = Mat::<f64>::from_fn(n, m, |i, j| {
            if j == constant {
                0.0
            } else {
                (dense[(i, j)] - means[j]) / sds[j]
            }
        });
        let reference = scaled.thin_svd().unwrap();

        let csc =
            CompressedSparseData2::<f64, f64>::new_csc(&data, &indices, &indptr, None, (n, m));
        let rank = 3;
        let res =
            sparse_covariance_svd::<f64, f64>(csc, rank, false, Some(&means), Some(&sds), None)
                .unwrap();

        for c in 0..rank {
            let expected = reference.S()[c];
            assert!(res.s[c].is_finite());
            assert!(
                (res.s[c] - expected).abs() < 1e-8 * expected.max(1.0),
                "s[{c}] = {} against {expected}",
                res.s[c]
            );
            assert_eq!(res.v[(constant, c)], 0.0);
        }
    }
}
