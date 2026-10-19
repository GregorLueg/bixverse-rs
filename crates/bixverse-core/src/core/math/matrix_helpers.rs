//! Various helper functions that act on faer matrices

use faer::{Mat, MatMut, MatRef, Scale};
use rayon::prelude::*;

use crate::core::math::vector_helpers::*;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Matrix size (elements) from which column-wise loops fan out over rayon
const PAR_COL_MIN_ELEMS: usize = 1 << 17;

///////////////
// Functions //
///////////////

/// Scale a matrix
///
/// ### Params
///
/// * `mat` - The matrix on which to apply column-wise scaling
/// * `scale_sd` - Shall the standard deviation be equalised across columns
///
/// ### Returns
///
/// The scaled matrix.
pub fn scale_matrix_col<T>(mat: &MatRef<T>, scale_sd: bool) -> Mat<T>
where
    T: BixverseFloat,
{
    let n_rows = mat.nrows();
    let n_rows_t = T::from_usize(n_rows).unwrap();
    let sd_floor = T::from_f64(1e-10).unwrap();

    let scale_col = |col: &mut [T]| {
        let mut mean = T::zero();
        for &v in col.iter() {
            mean += v;
        }
        mean /= n_rows_t;
        for v in col.iter_mut() {
            *v -= mean;
        }
        if !scale_sd {
            return;
        }
        let mut ss = T::zero();
        for &v in col.iter() {
            ss += v * v;
        }
        let mut sd = (ss / (n_rows_t - T::one())).sqrt();
        if sd < sd_floor {
            sd = T::one();
        }
        for v in col.iter_mut() {
            *v /= sd;
        }
    };

    let mut result = mat.to_owned();
    if n_rows * mat.ncols() >= PAR_COL_MIN_ELEMS {
        result.par_col_iter_mut().for_each(|mut col| {
            scale_col(col.as_mut().try_as_col_major_mut().unwrap().as_slice_mut());
        });
    } else {
        for j in 0..result.ncols() {
            scale_col(
                result
                    .col_mut(j)
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut(),
            );
        }
    }

    result
}

/// Column wise L2 normalisation
///
/// ### Params
///
/// * `mat` - The matrix on which to apply column-wise L2 normalisation
///
/// ### Returns
///
/// The matrix with the columns being L2 normalised.
pub fn normalise_matrix_col_l2<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let mut normalised = mat.to_owned();

    for j in 0..mat.ncols() {
        let col = mat.col(j);
        let norm = col.norm_l2();

        if norm > T::from_f64(1e-10).unwrap() {
            for i in 0..mat.nrows() {
                normalised[(i, j)] = mat[(i, j)] / norm;
            }
        }
    }

    normalised
}

/// Normalise the data across rows
///
/// ### Params
///
/// * `mat` - The matrix to normalise.
///
/// ### Returns
///
/// The (row-)normalised matrix.
pub fn normalise_rows_l1<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let (nrows, ncols) = mat.shape();
    let mut row_sums = vec![T::zero(); nrows];
    for j in 0..ncols {
        for (s, &v) in row_sums.iter_mut().zip(mat.col(j).iter()) {
            *s += v;
        }
    }
    Mat::from_fn(nrows, ncols, |i, j| {
        let row_sum = row_sums[i];
        if row_sum > T::zero() {
            *mat.get(i, j) / row_sum
        } else {
            T::zero()
        }
    })
}

/// Column wise rank normalisation
///
/// ### Params
///
/// * `mat` - The matrix on which to apply column-wise rank normalisation
///
/// ### Returns
///
/// The matrix with the columns being rank normalised.
pub fn rank_matrix_col<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let mut ranked_mat: Mat<T> = Mat::zeros(mat.nrows(), mat.ncols());

    // Parallel ranking directly into the matrix
    ranked_mat
        .par_col_iter_mut()
        .enumerate()
        .for_each(|(col_idx, mut col)| {
            let original_col: Vec<T> = mat.col(col_idx).iter().copied().collect();
            let ranks = rank_vector(&original_col);

            // Write ranks directly to the matrix column
            for (row_idx, &rank) in ranks.iter().enumerate() {
                col[row_idx] = rank;
            }
        });

    ranked_mat
}

/// Calculates the column sums of a matrix
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the column-wise sums
///
/// ### Returns
///
/// Vector of the column sums.
pub fn col_sums<T>(mat: MatRef<T>) -> Vec<T>
where
    T: BixverseFloat,
{
    let n_rows = mat.nrows();
    let ones = Mat::from_fn(n_rows, 1, |_, _| T::one());
    let col_sums = ones.transpose() * mat;

    col_sums.row(0).iter().cloned().collect()
}

/// Calculates the columns means of a matrix
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the column-wise means
///
/// ### Returns
///
/// Vector of the column means.
pub fn col_means<T>(mat: MatRef<T>) -> Vec<T>
where
    T: BixverseFloat,
{
    let n_rows = mat.nrows();
    let ones = Mat::from_fn(n_rows, 1, |_, _| T::one());
    let means = ones.transpose() * mat / Scale(T::from_usize(n_rows).unwrap());

    means.row(0).iter().cloned().collect()
}

/// Calculate the column standard deviations
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the column-wise standard
///   deviations
///
/// ### Returns
///
/// Vector of the column standard deviations.
pub fn col_sds<T>(mat: MatRef<T>) -> Vec<T>
where
    T: BixverseFloat,
{
    let n = T::from_usize(mat.nrows()).unwrap();
    let n_cols = mat.ncols();

    let col_sd = |j: usize| {
        let mut mean = T::zero();
        let mut m2 = T::zero();
        let mut count = T::zero();
        for &x in mat.col(j).iter() {
            count += T::one();
            let delta = x - mean;
            mean += delta / count;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
        (m2 / (n - T::one())).sqrt()
    };

    if mat.nrows() * n_cols >= PAR_COL_MIN_ELEMS {
        (0..n_cols).into_par_iter().map(col_sd).collect()
    } else {
        (0..n_cols).map(col_sd).collect()
    }
}

/////////////////////////
// Matrix manipulation //
/////////////////////////

/// Copy the lower triangle of a square matrix into its upper triangle
///
/// Blocked so that both the strided reads and the contiguous writes stay in
/// cache, instead of one cache line per element on the transposed side.
///
/// ### Params
///
/// * `mat` - Square matrix whose lower triangle is filled
pub fn mirror_lower_to_upper<T>(mut mat: MatMut<T>)
where
    T: Copy,
{
    const BLOCK: usize = 64;
    let n = mat.nrows();
    debug_assert_eq!(n, mat.ncols());

    for bj in (0..n).step_by(BLOCK) {
        let j_end = (bj + BLOCK).min(n);
        for bi in (0..=bj).step_by(BLOCK) {
            let i_end = (bi + BLOCK).min(n);
            for j in bj..j_end {
                for i in bi..i_end.min(j) {
                    let v = mat[(j, i)];
                    mat[(i, j)] = v;
                }
            }
        }
    }
}

/// Stack two matrices
///
/// ### Params
///
/// * `x` - First matrix
/// * `y` - Second matrix
///
/// ### Returns
///
/// Stacked matrix
pub fn stack_rows<T: BixverseFloat>(x: MatRef<T>, y: MatRef<T>) -> Mat<T> {
    faer::concat![[x], [y]]
}

/// Subset rows
///
/// ### Params
///
/// * `x` - The matrix to subset
/// * `idx` - The row indices
///
/// ### Returns
///
/// The subsetted matrix
pub fn subset_rows<T: BixverseFloat>(x: MatRef<T>, idx: &[usize]) -> Mat<T> {
    let d = x.ncols();
    Mat::from_fn(idx.len(), d, |i, j| *x.get(idx[i], j))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Column sums and means run down the rows, not across them.
    #[test]
    fn test_col_sums_and_means() {
        let mat: Mat<f64> = Mat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);

        let sums = col_sums(mat.as_ref());
        assert_eq!(sums, vec![5.0, 7.0, 9.0]);

        let means = col_means(mat.as_ref());
        assert_eq!(means, vec![2.5, 3.5, 4.5]);
    }

    /// Each row is divided by its own sum, so rows with different sums scale differently.
    #[test]
    fn test_normalise_rows_l1() {
        let mat: Mat<f64> = Mat::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
        let norm = normalise_rows_l1(&mat.as_ref());

        assert!((norm[(0, 0)] - 1.0 / 3.0).abs() < 1e-6);
        assert!((norm[(0, 1)] - 2.0 / 3.0).abs() < 1e-6);
        assert!((norm[(1, 0)] - 3.0 / 7.0).abs() < 1e-6);
        assert!((norm[(1, 1)] - 4.0 / 7.0).abs() < 1e-6);
    }

    /// Without the standard deviation the column is only centred, never rescaled.
    #[test]
    fn test_scale_matrix_col() {
        let mat: Mat<f64> = Mat::from_fn(3, 1, |i, _| (i + 1) as f64);
        let scaled_no_sd = scale_matrix_col(&mat.as_ref(), false);

        assert!((scaled_no_sd[(0, 0)] - (-1.0)).abs() < 1e-6);
        assert!((scaled_no_sd[(1, 0)] - 0.0).abs() < 1e-6);
        assert!((scaled_no_sd[(2, 0)] - 1.0).abs() < 1e-6);
    }
}
