use faer::{Mat, MatRef, concat};

use crate::prelude::*;
use crate::utils::vec_utils::flatten_vector;

/// Transform a nested vector into a faer matrix
///
/// ### Params
///
/// * `nested_vec` - The nested vector
/// * `col_wise` - If set to `True` it will column bind (outer vector represents)
///   the columns. If set to `False` it will row bind (outer vector represents
///   the rows).
///
/// ### Returns
///
/// The row or column bound matrix.
pub fn nested_vector_to_faer_mat<T>(nested_vec: Vec<Vec<T>>, col_wise: bool) -> Mat<T>
where
    T: BixverseFloat,
{
    let (nrow, ncol) = if col_wise {
        (nested_vec[0].len(), nested_vec.len())
    } else {
        (nested_vec.len(), nested_vec[0].len())
    };

    let data = flatten_vector(nested_vec);

    if col_wise {
        Mat::from_fn(nrow, ncol, |i, j| data[i + j * nrow])
    } else {
        Mat::from_fn(nrow, ncol, |i, j| data[j + i * ncol])
    }
}

/// Create a diagonal matrix from vector values
///
/// ### Params
///
/// * `vec` - The vector of values to put in the diagonal of the matrix
///
/// ### Returns
///
/// The diagonal matrix as a faer Matrix.
pub fn faer_diagonal_from_vec<T>(vec: Vec<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let len = vec.len();
    Mat::from_fn(
        len,
        len,
        |row, col| if row == col { vec[row] } else { T::zero() },
    )
}

/// Get the index positions of the upper triangle of a symmetric matrix
///
/// Function will panic if offset > 1.
///
/// ### Params
///
/// * `n_dim` - The dimensions of the symmetric matrix
/// * `offset` - Do you want to include the diagonal values (offset = 0) or exclude
///   them (offset = 1).
///
/// ### Returns
///
/// A tuple of the row and column index positions of the upper triangle of the
/// matirx
pub fn upper_triangle_indices(n_dim: usize, offset: usize) -> (Vec<usize>, Vec<usize>) {
    if offset >= n_dim {
        return (Vec::new(), Vec::new());
    }
    assert!(offset <= 1, "The offset should be 0 or 1");

    // Precise calculation of total elements
    let total_elements: usize = (0..n_dim)
        .map(|row| n_dim.saturating_sub(row + offset))
        .sum();

    let mut row_indices = Vec::with_capacity(total_elements);
    let mut col_indices = Vec::with_capacity(total_elements);

    for row in 0..n_dim {
        let start_col = row + offset;
        if start_col < n_dim {
            let end_col = n_dim;
            let elements_in_row = end_col - start_col;
            // Use repeat_n for better performance and clarity
            row_indices.extend(std::iter::repeat_n(row, elements_in_row));
            col_indices.extend(start_col..end_col);
        }
    }

    (row_indices, col_indices)
}

/// Create from the upper triangle values a symmetric matrix
///
/// Generates the full dense matrix of values representing the upper triangle
/// of a symmetric matrix.
///
/// ### Params
///
/// * `data` - Slice of the values
/// * `shift` - Was the diagonal included (= 0) or not (= 1). If not included,
///   the diagonal is set to 1.
/// * `n` - Original dimension of the symmetric matrix.
///
/// ### Return
///
/// The symmetric, dense matrix.
pub fn upper_triangle_to_sym_faer<T>(data: &[T], shift: usize, n: usize) -> faer::Mat<T>
where
    T: BixverseFloat,
{
    let mut mat = Mat::<T>::zeros(n, n);
    let mut idx = 0;
    for i in 0..n {
        for j in i..n {
            if shift == 1 && i == j {
                mat[(i, j)] = T::one();
            } else {
                mat[(i, j)] = data[idx];
                mat[(j, i)] = data[idx];
                idx += 1;
            }
        }
    }

    mat
}

/// Store the upper triangle values as a flat vector from a faer matrix
///
/// ### Params
///
/// * `x` The faer matrix
/// * `shift` Shall the diagonal be included (shift = 0) or not (shift = 1).
///
/// ### Returns
///
/// A vector representing the upper triangle values (row major ordered)
pub fn faer_mat_to_upper_triangle<T>(x: MatRef<T>, shift: usize) -> Vec<T>
where
    T: BixverseFloat,
{
    assert!(shift <= 1, "The shift should be 0 or 1");

    let n = x.ncols();
    let total_elements = if shift == 0 {
        n * (n + 1) / 2
    } else {
        n * (n - 1) / 2
    };
    let mut vals = Vec::with_capacity(total_elements);
    for i in 0..n {
        let start_j = i + shift;
        for j in start_j..n {
            vals.push(*x.get(i, j));
        }
    }

    vals
}

/// Slice out a single row and return the remaining matrix
///
/// ### Params
///
/// * `x` - The matrix from which to remove a single row
/// * `idx_to_remove` - The index of the row to remove.
///
/// ### Returns
///
/// The matrix minus the specified row.
pub fn mat_rm_row<T>(x: MatRef<T>, idx_to_remove: usize) -> Mat<T>
where
    T: BixverseFloat,
{
    assert!(
        idx_to_remove <= x.nrows(),
        "The specified index is larger than the matrix"
    );

    let total_rows = x.nrows();

    if idx_to_remove == 0 {
        x.subrows(1, total_rows - 1).to_owned()
    } else if idx_to_remove == total_rows - 1 {
        x.subrows(0, total_rows - 1).to_owned()
    } else {
        let upper = x.subrows(0, idx_to_remove);
        let lower = x.subrows(idx_to_remove + 1, total_rows - idx_to_remove - 1);
        concat![[upper], [lower]]
    }
}

/// Rowbind a vector of faer Matrices
///
/// The function will panic if the number of columns of the matrices differ in
/// the vector
///
/// ### Params
///
/// * `matrices` - Vector of faer matrix to row bind
///
/// ### Returns
///
/// One row bound matrix from the initial matrices
#[allow(dead_code)]
pub fn rowbind_matrices<T>(matrices: Vec<Mat<T>>) -> Mat<T>
where
    T: BixverseFloat,
{
    let ncols = matrices[0].ncols();
    let total_row = matrices.iter().map(|m| m.nrows()).sum();
    let mut result: Mat<T> = Mat::zeros(total_row, ncols);
    let mut row_offset = 0;
    for matrix in matrices {
        assert_eq!(
            matrix.ncols(),
            ncols,
            "All matrices must have the same number of columns"
        );
        let nrows = matrix.nrows();
        for i in 0..nrows {
            for j in 0..ncols {
                result[(row_offset + i, j)] = matrix[(i, j)]
            }
        }
        row_offset += nrows;
    }

    result
}

/// Colbind a vector of faer Matrices
///
/// The function will panic if the number of rows of the matrices differ in
/// the vector
///
/// ### Params
///
/// * `matrices` - Vector of faer matrix to column bind
///
/// ### Returns
///
/// One column bound matrix from the initial matrices
pub fn colbind_matrices<T>(matrices: &[Mat<T>]) -> Mat<T>
where
    T: BixverseFloat,
{
    let nrows = matrices[0].nrows();
    let total_col = matrices.iter().map(|m| m.ncols()).sum();
    let mut result: Mat<T> = Mat::zeros(nrows, total_col);
    let mut col_offset = 0;
    for matrix in matrices {
        assert_eq!(
            matrix.nrows(),
            nrows,
            "All matrices must have the same number of columns"
        );
        let ncols = matrix.ncols();
        for i in 0..nrows {
            for j in 0..ncols {
                result[(i, col_offset + j)] = matrix[(i, j)]
            }
        }
        col_offset += ncols;
    }

    result
}
