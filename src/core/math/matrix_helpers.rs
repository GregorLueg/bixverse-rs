use faer::{Mat, MatRef};
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
