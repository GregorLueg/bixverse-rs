use faer::{Mat, MatRef, Scale};
use rayon::prelude::*;

use crate::core::math::vector_helpers::*;
use crate::prelude::*;

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
    let n_cols = mat.ncols();

    let mut means = vec![T::zero(); n_cols];
    for j in 0..n_cols {
        for i in 0..n_rows {
            means[j] += mat[(i, j)];
        }
        means[j] /= T::from_usize(n_rows).unwrap();
    }

    let mut result = mat.to_owned();
    for j in 0..n_cols {
        let mean = means[j];
        for i in 0..n_rows {
            result[(i, j)] -= mean;
        }
    }

    if !scale_sd {
        return result;
    }

    let mut std_devs = vec![T::zero(); n_cols];
    for j in 0..n_cols {
        for i in 0..n_rows {
            let val = result[(i, j)];
            std_devs[j] += val * val;
        }
        std_devs[j] = (std_devs[j] / (T::from_usize(n_rows).unwrap() - T::one())).sqrt();
        if std_devs[j] < T::from_f64(1e-10).unwrap() {
            std_devs[j] = T::one();
        }
    }

    for j in 0..n_cols {
        let std_dev = std_devs[j];
        for i in 0..n_rows {
            result[(i, j)] /= std_dev;
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
    Mat::from_fn(nrows, ncols, |i, j| {
        let row_sum: T = (0..ncols).map(|k| mat[(i, k)]).sum();
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

    let (_, m2): (Vec<T>, Vec<T>) = (0..n_cols)
        .map(|j| {
            let mut mean = T::zero();
            let mut m2 = T::zero();
            let mut count = T::zero();
            for i in 0..mat.nrows() {
                count += T::one();
                let delta = mat[(i, j)] - mean;
                mean += delta / count;
                let delta2 = mat[(i, j)] - mean;
                m2 += delta * delta2;
            }
            (mean, (m2 / (n - T::one())).sqrt())
        })
        .unzip();
    m2
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_col_sums_and_means() {
        // Matrix:
        // [[1.0, 2.0, 3.0],
        //  [4.0, 5.0, 6.0]]
        let mat: Mat<f64> = Mat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);

        let sums = col_sums(mat.as_ref());
        assert_eq!(sums, vec![5.0, 7.0, 9.0]);

        let means = col_means(mat.as_ref());
        assert_eq!(means, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_normalise_rows_l1() {
        // Matrix:
        // [[1.0, 2.0],
        //  [3.0, 4.0]]
        let mat: Mat<f64> = Mat::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
        let norm = normalise_rows_l1(&mat.as_ref());

        // Row 1 sum = 3.0 -> [1/3, 2/3]
        // Row 2 sum = 7.0 -> [3/7, 4/7]
        assert!((norm[(0, 0)] - 1.0 / 3.0).abs() < 1e-6);
        assert!((norm[(0, 1)] - 2.0 / 3.0).abs() < 1e-6);
        assert!((norm[(1, 0)] - 3.0 / 7.0).abs() < 1e-6);
        assert!((norm[(1, 1)] - 4.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_matrix_col() {
        let mat: Mat<f64> = Mat::from_fn(3, 1, |i, _| (i + 1) as f64); // [1.0, 2.0, 3.0]^T
        let scaled_no_sd = scale_matrix_col(&mat.as_ref(), false);

        // Mean is 2.0, so centering should yield [-1.0, 0.0, 1.0]^T
        assert!((scaled_no_sd[(0, 0)] - (-1.0)).abs() < 1e-6);
        assert!((scaled_no_sd[(1, 0)] - 0.0).abs() < 1e-6);
        assert!((scaled_no_sd[(2, 0)] - 1.0).abs() < 1e-6);
    }
}
