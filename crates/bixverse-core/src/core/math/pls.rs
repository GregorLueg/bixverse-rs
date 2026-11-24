//! Partial least squares regression with the kernel algorithm of Dayal and
//! MacGregor, as `pls::kernelpls.fit` implements it.
//!
//! The predictors come in as a CSR matrix and are centred implicitly, so a
//! sparse matrix never densifies. Each component costs one pass for `X r` and
//! one for `X' t`; the response only enters through `X' Y`, computed once.
//! Signs of the components are arbitrary, as for any eigenvector.

use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Rows per rayon task in the `X' v` reductions. Each task owns a
/// `n_cols x k` accumulator, so fewer, larger tasks keep the merge cheap. Not
/// tuned.
const PLS_ROWS_PER_TASK: usize = 2048;

////////////////
// Structures //
////////////////

/// A fitted PLS model.
///
/// `n` observations, `p` predictors, `q` responses and `a` components. Scores
/// of new data are `(x - x_means) projection`.
#[derive(Clone, Debug)]
pub struct PlsFit<T> {
    /// Loading weights `W`, `p x a`
    pub weights: Mat<T>,
    /// Projection `R = W (P' W)^-1`, `p x a`. Maps centred predictors to scores
    pub projection: Mat<T>,
    /// Predictor loadings `P`, `p x a`
    pub x_loadings: Mat<T>,
    /// Response loadings `Q`, `q x a`
    pub y_loadings: Mat<T>,
    /// Predictor scores `T`, `n x a`
    pub x_scores: Mat<T>,
    /// Column means of the predictors, length `p`
    pub x_means: Vec<T>,
    /// Column means of the responses, length `q`
    pub y_means: Vec<T>,
}

/////////////
// Helpers //
/////////////

/// `X' V` for a CSR `X` (`n x p`) and a dense `V` (`n x k`).
///
/// Fold-reduce over blocks of rows, each with its own `p x k` accumulator.
///
/// ### Params
///
/// * `x` - CSR matrix, `n x p`
/// * `v` - Dense matrix, `n x k`
///
/// ### Returns
///
/// `X' V` as a `p x k` matrix.
fn csr_t_dense<T>(x: &CompressedSparseData2<T>, v: MatRef<T>) -> Mat<T>
where
    T: BixverseFloat + BixverseNumeric,
{
    let (n, p) = x.shape;
    let k = v.ncols();
    let acc = (0..n)
        .into_par_iter()
        .with_min_len(PLS_ROWS_PER_TASK)
        .fold(
            || vec![T::zero(); p * k],
            |mut acc, i| {
                let (s, e) = (x.indptr[i] as usize, x.indptr[i + 1] as usize);
                for c in 0..k {
                    let vi = v[(i, c)];
                    if vi == T::zero() {
                        continue;
                    }
                    let col = &mut acc[c * p..(c + 1) * p];
                    for q in s..e {
                        col[x.indices[q] as usize] += vi * x.data[q];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![T::zero(); p * k],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            },
        );
    Mat::from_fn(p, k, |j, c| acc[c * p + j])
}

/// `X r - (mu' r) 1`: the centred predictors times a vector.
///
/// ### Params
///
/// * `x` - CSR matrix, `n x p`
/// * `mu` - Column means of `x`, length `p`
/// * `r` - Vector of length `p`
///
/// ### Returns
///
/// Vector of length `n`.
fn centred_matvec<T>(x: &CompressedSparseData2<T>, mu: &[T], r: &[T]) -> Vec<T>
where
    T: BixverseFloat + BixverseNumeric,
{
    let shift = mu
        .iter()
        .zip(r)
        .fold(T::zero(), |acc, (&m, &v)| acc + m * v);
    (0..x.shape.0)
        .into_par_iter()
        .map(|i| {
            let (s, e) = (x.indptr[i] as usize, x.indptr[i + 1] as usize);
            let mut sum = T::zero();
            for q in s..e {
                sum += x.data[q] * r[x.indices[q] as usize];
            }
            sum - shift
        })
        .collect()
}

/// Leading eigenvector of a symmetric matrix.
///
/// ### Params
///
/// * `m` - Symmetric matrix
///
/// ### Returns
///
/// The eigenvector of the largest eigenvalue.
fn leading_eigenvector<T>(m: MatRef<T>) -> Result<Vec<T>, CoreErrors>
where
    T: BixverseFloat,
{
    let eig = m
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| CoreErrors::FaerEigenError)?;
    let (vals, vecs) = (eig.S(), eig.U());
    let top = (0..m.nrows())
        .max_by(|&a, &b| vals[a].total_cmp(&vals[b]))
        .expect("a non-empty matrix has an eigenvalue");
    Ok((0..m.nrows()).map(|j| vecs[(j, top)]).collect())
}

/// Scale a vector to unit Euclidean norm.
///
/// ### Params
///
/// * `v` - The vector, normalised in place
fn normalise_l2<T: BixverseFloat>(v: &mut [T]) {
    let norm = v.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt();
    if norm > T::zero() {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

//////////
// Main //
//////////

/// Kernel PLS regression of `Y` on a sparse `X`.
///
/// Both blocks are centred, `X` implicitly. Per component the weight is the
/// dominant left singular direction of the deflated `X' Y`; with one response
/// that is `X' y` itself, with `q < p` it comes from the eigenproblem of the
/// `q x q` matrix `(X' Y)' X' Y` and otherwise from the `p x p` one, the same
/// three branches `pls::kernelpls.fit` takes. Only `X' Y` is deflated, so `X`
/// is read but never modified.
///
/// ### Params
///
/// * `x` - Predictors, CSR, `n x p`
/// * `y` - Responses, dense, `n x q`. A factor goes in one-hot encoded
/// * `n_comp` - Number of components, at most `min(n - 1, p)`
///
/// ### Returns
///
/// The [PlsFit]. Errors on a CSC `x`, mismatched rows, an out-of-range
/// `n_comp`, or a component with zero variance (rank exhausted).
///
/// ### References
///
/// Dayal & MacGregor, J Chemometrics, 1997; Mevik & Wehrens, J Stat Softw,
/// 2007
pub fn kernel_pls<T>(
    x: &CompressedSparseData2<T>,
    y: MatRef<T>,
    n_comp: usize,
) -> Result<PlsFit<T>, CoreErrors>
where
    T: BixverseFloat + BixverseNumeric,
{
    if !x.cs_type.is_csr() {
        return Err(CoreErrors::InvalidArgument(
            "kernel_pls needs X in CSR format.".into(),
        ));
    }
    let (n, p) = x.shape;
    let q = y.ncols();
    if y.nrows() != n {
        return Err(CoreErrors::InvalidArgument(format!(
            "X has {n} rows but Y has {}.",
            y.nrows()
        )));
    }
    if q == 0 {
        return Err(CoreErrors::InvalidArgument("Y has no columns.".into()));
    }
    if n_comp == 0 || n_comp > p.min(n.saturating_sub(1)) {
        return Err(CoreErrors::InvalidArgument(format!(
            "n_comp must lie in 1..={} for {n} observations and {p} predictors, got {n_comp}.",
            p.min(n.saturating_sub(1))
        )));
    }

    let n_t = T::from_usize(n).expect("n fits the float type");
    let ones = Mat::<T>::from_fn(n, 1, |_, _| T::one());
    let x_means: Vec<T> = csr_t_dense(x, ones.as_ref())
        .col(0)
        .iter()
        .map(|&s| s / n_t)
        .collect();
    let y_means: Vec<T> = (0..q)
        .map(|c| y.col(c).iter().fold(T::zero(), |a, &v| a + v) / n_t)
        .collect();
    let y_c = Mat::<T>::from_fn(n, q, |i, c| y[(i, c)] - y_means[c]);

    // X_c' Y_c = X' Y_c - mu (1' Y_c); the second term is zero up to rounding
    let mut xty = csr_t_dense(x, y_c.as_ref());
    let y_col_sums: Vec<T> = (0..q)
        .map(|c| y_c.col(c).iter().fold(T::zero(), |a, &v| a + v))
        .collect();
    for c in 0..q {
        for j in 0..p {
            xty[(j, c)] -= x_means[j] * y_col_sums[c];
        }
    }

    let mut weights = Mat::<T>::zeros(p, n_comp);
    let mut projection = Mat::<T>::zeros(p, n_comp);
    let mut x_loadings = Mat::<T>::zeros(p, n_comp);
    let mut y_loadings = Mat::<T>::zeros(q, n_comp);
    let mut x_scores = Mat::<T>::zeros(n, n_comp);

    for a in 0..n_comp {
        let mut w: Vec<T> = if q == 1 {
            xty.col(0).iter().copied().collect()
        } else if q < p {
            let gram = xty.transpose() * &xty;
            let v = leading_eigenvector(gram.as_ref())?;
            (0..p)
                .map(|j| (0..q).fold(T::zero(), |acc, c| acc + xty[(j, c)] * v[c]))
                .collect()
        } else {
            let gram = &xty * xty.transpose();
            leading_eigenvector(gram.as_ref())?
        };
        normalise_l2(&mut w);

        // r = w - R (P' w), over the components fitted so far
        let mut r = w.clone();
        for b in 0..a {
            let pw = (0..p).fold(T::zero(), |acc, j| acc + x_loadings[(j, b)] * w[j]);
            for (j, rj) in r.iter_mut().enumerate() {
                *rj -= pw * projection[(j, b)];
            }
        }

        let t = centred_matvec(x, &x_means, &r);
        let tsq = t.iter().fold(T::zero(), |acc, &v| acc + v * v);
        if !tsq.is_finite() || tsq <= T::zero() {
            return Err(CoreErrors::InvalidArgument(format!(
                "PLS component {} has zero variance; X supports at most {a} components.",
                a + 1
            )));
        }

        // p = X_c' t / tsq, with X_c' t = X' t - mu (1' t)
        let t_sum = t.iter().fold(T::zero(), |acc, &v| acc + v);
        let t_mat = MatRef::from_column_major_slice(&t, n, 1);
        let xt = csr_t_dense(x, t_mat);
        let p_a: Vec<T> = (0..p)
            .map(|j| (xt[(j, 0)] - x_means[j] * t_sum) / tsq)
            .collect();
        let q_a: Vec<T> = (0..q)
            .map(|c| (0..p).fold(T::zero(), |acc, j| acc + xty[(j, c)] * r[j]) / tsq)
            .collect();

        for j in 0..p {
            for c in 0..q {
                xty[(j, c)] -= tsq * p_a[j] * q_a[c];
            }
            weights[(j, a)] = w[j];
            projection[(j, a)] = r[j];
            x_loadings[(j, a)] = p_a[j];
        }
        for c in 0..q {
            y_loadings[(c, a)] = q_a[c];
        }
        for (i, &v) in t.iter().enumerate() {
            x_scores[(i, a)] = v;
        }
    }

    Ok(PlsFit {
        weights,
        projection,
        x_loadings,
        y_loadings,
        x_scores,
        x_means,
        y_means,
    })
}
