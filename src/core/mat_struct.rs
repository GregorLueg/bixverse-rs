use faer::{Mat, MatRef};
use std::collections::BTreeMap;

use crate::prelude::*;

//////////////
// Matrices //
//////////////

/// Structure to store named matrices
///
/// ### Fields
///
/// * `col_names` - A BTreeMap representing the column names and the col indices
/// * `row_names` - A BTreeMap representing the row names and the row indices
/// * `values` - A faer matrix reference representing the matrix values.
#[derive(Clone, Debug)]
pub struct NamedMatrix<'a, T> {
    pub col_names: BTreeMap<String, usize>,
    pub row_names: BTreeMap<String, usize>,
    pub values: faer::MatRef<'a, T>,
}

impl<'a, T> NamedMatrix<'a, T>
where
    T: BixverseFloat,
{
    /// Return a submatrix based on the row names and columns to select.
    ///
    /// If no rows or columns are specified, returns the full matrix. If empty slices are
    /// provided, returns None.
    ///
    /// ### Params
    ///
    /// * `rows_to_select` - Names of the row features
    /// * `cols_to_select` - Names of the column features
    ///
    /// ### Returns
    ///
    /// The faer Matrix with the values from the rows and columns.
    pub fn get_sub_mat(
        &self,
        rows_to_select: Option<&[&str]>,
        cols_to_select: Option<&[&str]>,
    ) -> Option<Mat<T>> {
        // Determine which rows to select
        let row_indices: Vec<usize> = match rows_to_select {
            None => (0..self.values.nrows()).collect(), // All rows
            Some(rows) => {
                if rows.is_empty() {
                    return None;
                }
                let mut indices = Vec::new();
                for &row_name in rows {
                    if let Some(&index) = self.row_names.get(row_name) {
                        indices.push(index);
                    }
                }
                if indices.is_empty() {
                    return None;
                }
                indices
            }
        };

        // Determine which columns to select
        let col_indices: Vec<usize> = match cols_to_select {
            None => (0..self.values.ncols()).collect(), // All columns
            Some(cols) => {
                if cols.is_empty() {
                    return None;
                }
                let mut indices = Vec::new();
                for &col_name in cols {
                    if let Some(&index) = self.col_names.get(col_name) {
                        indices.push(index);
                    }
                }
                if indices.is_empty() {
                    return None;
                }
                indices
            }
        };

        // Create new matrix by copying values
        let mut result = Mat::<T>::zeros(row_indices.len(), col_indices.len());
        for (new_row, &old_row) in row_indices.iter().enumerate() {
            for (new_col, &old_col) in col_indices.iter().enumerate() {
                result[(new_row, new_col)] = self.values[(old_row, old_col)];
            }
        }

        Some(result)
    }

    /// Convenience method to get the full matrix
    ///
    /// ### Returns
    ///
    /// Faer matrix representing the full data.
    pub fn get_full_mat(&self) -> Mat<T> {
        self.get_sub_mat(None, None).unwrap()
    }

    /// Convenience method to get submatrix with only row selection
    ///
    /// ### Params
    ///
    /// * `rows_to_select` - Names of the row features
    ///
    /// ### Returns
    ///
    /// Faer matrix with the selected rows.
    pub fn get_rows(&self, rows_to_select: &[&str]) -> Option<Mat<T>> {
        self.get_sub_mat(Some(rows_to_select), None)
    }

    /// Convenience method to get submatrix with only column selection
    ///
    /// ### Params
    ///
    /// * `cols_to_select` - Names of the column features
    ///
    /// ### Returns
    ///
    /// Faer matrix with the selected columns.
    pub fn get_cols(&self, cols_to_select: &[&str]) -> Option<Mat<T>> {
        self.get_sub_mat(None, Some(cols_to_select))
    }

    /// Get column names as references (for temporary use within same scope)
    ///
    /// ### Returns
    ///
    /// The column names as a reference
    pub fn get_col_names_refs(&self) -> Vec<&String> {
        self.col_names.keys().collect()
    }

    /// Get row names as references (for temporary use within same scope)
    ///
    /// ### Returns
    ///
    /// The row names as a reference
    pub fn get_row_names_refs(&self) -> Vec<&String> {
        self.row_names.keys().collect()
    }

    /// Get the row indices based on names
    ///
    /// ### Params
    ///
    /// * `rows_to_select` - Names of the row features
    pub fn get_row_indices(&self, rows_to_select: Option<&[&str]>) -> Vec<usize> {
        let row_indices: Vec<usize> = match rows_to_select {
            None => (0..self.values.nrows()).collect(), // All rows
            Some(rows) => {
                if rows.is_empty() {
                    return vec![];
                }
                let mut indices = Vec::new();
                for &row_name in rows {
                    if let Some(&index) = self.row_names.get(row_name) {
                        indices.push(index);
                    }
                }
                if indices.is_empty() {
                    return vec![];
                }
                indices
            }
        };

        row_indices
    }

    /// Get the col indices based on names
    ///
    /// ### Params
    ///
    /// * `rows_to_select` - Names of the row features
    pub fn get_col_indices(&self, cols_to_select: Option<&[&str]>) -> Vec<usize> {
        let col_indices: Vec<usize> = match cols_to_select {
            None => (0..self.values.ncols()).collect(), // All columns
            Some(cols) => {
                if cols.is_empty() {
                    return vec![];
                }
                let mut indices = Vec::new();
                for &col_name in cols {
                    if let Some(&index) = self.col_names.get(col_name) {
                        indices.push(index);
                    }
                }
                if indices.is_empty() {
                    return vec![];
                }
                indices
            }
        };

        col_indices
    }
}

/// Matrix slice view
///
/// Structure to help creating sub slices of a given matrix when needed.
/// Due to memory structure, slicing creates deep copies, but this function
/// avoids generating them until the last possible moment.
///
/// ### Fields
///
/// * `data` - The faer MatRef (original matrix)
/// * `row_indices` - The row indices you want to slice out.
/// * `col_indices` - The col indices you want to slice out.
#[derive(Clone, Debug)]
pub struct MatSliceView<'a, 'r, 'c, T> {
    data: MatRef<'a, T>,
    row_indices: &'r [usize],
    col_indices: &'c [usize],
}

impl<'a, 'r, 'c, T> MatSliceView<'a, 'r, 'c, T>
where
    T: BixverseFloat,
{
    /// Generate a new MatSliceView
    ///
    /// This function will panic if you try to select indices larger than the
    /// underlying matrix.
    ///
    /// ### Params
    ///
    /// * `data` - The original MatRef from which you want to slice out data
    /// * `row_indices` - The row indices you want to slice out.
    /// * `col_indices` - The col indices you want to slice out.
    pub fn new(
        data: MatRef<'a, T>,
        row_indices: &'r [usize],
        col_indices: &'c [usize],
    ) -> MatSliceView<'a, 'r, 'c, T> {
        let max_col_index = col_indices.iter().copied().fold(0, usize::max);
        let max_row_index = row_indices.iter().copied().fold(0, usize::max);

        assert!(
            max_col_index < data.ncols(),
            "You selected indices larger than ncol."
        );
        assert!(
            max_row_index < data.nrows(),
            "You selected indices larger than nrow."
        );

        Self {
            data,
            row_indices,
            col_indices,
        }
    }

    /// Return the number of rows
    ///
    /// ### Returns
    ///
    /// Number of rows
    pub fn nrows(&self) -> usize {
        self.row_indices.len()
    }

    /// Return the number of columns
    ///
    /// ### Returns
    ///
    /// Number of columns
    pub fn ncols(&self) -> usize {
        self.col_indices.len()
    }

    /// Return an owned matrix.
    ///
    /// Deep copying cannot be circumvented due to memory accessing at this point.
    /// Subsequent matrix algebra needs a continouos view into memory.
    ///
    /// ### Returns
    ///
    /// Owned sliced matrix for subsequent usage.
    pub fn to_owned(&self) -> Mat<T> {
        Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            *self.data.get(self.row_indices[i], self.col_indices[j])
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use std::collections::BTreeMap;

    fn setup_named_matrix() -> (Mat<f64>, BTreeMap<String, usize>, BTreeMap<String, usize>) {
        // 3x3 Matrix:
        // [[0.0, 1.0, 2.0],
        //  [3.0, 4.0, 5.0],
        //  [6.0, 7.0, 8.0]]
        let mat: Mat<f64> = Mat::from_fn(3, 3, |i, j| (i * 3 + j) as f64);

        let mut row_names = BTreeMap::new();
        row_names.insert("r1".to_string(), 0);
        row_names.insert("r2".to_string(), 1);
        row_names.insert("r3".to_string(), 2);

        let mut col_names = BTreeMap::new();
        col_names.insert("c1".to_string(), 0);
        col_names.insert("c2".to_string(), 1);
        col_names.insert("c3".to_string(), 2);

        (mat, row_names, col_names)
    }

    #[test]
    fn test_named_matrix_sub_mat() {
        let (mat, row_names, col_names) = setup_named_matrix();
        let named_mat = NamedMatrix {
            col_names,
            row_names,
            values: mat.as_ref(),
        };

        // Select row 3 and row 1 (in that order), and column 2
        let sub = named_mat
            .get_sub_mat(Some(&["r3", "r1"]), Some(&["c2"]))
            .unwrap();

        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 1);

        // "r3", "c2" -> original (2, 1) -> value 7.0
        assert!((sub[(0, 0)] - 7.0).abs() < 1e-6);
        // "r1", "c2" -> original (0, 1) -> value 1.0
        assert!((sub[(1, 0)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_named_matrix_empty_selection() {
        let (mat, row_names, col_names) = setup_named_matrix();
        let named_mat = NamedMatrix {
            col_names,
            row_names,
            values: mat.as_ref(),
        };

        let empty_rows = named_mat.get_sub_mat(Some(&[]), None);
        assert!(empty_rows.is_none());

        let invalid_rows = named_mat.get_sub_mat(Some(&["r_invalid"]), None);
        assert!(invalid_rows.is_none());
    }

    #[test]
    fn test_mat_slice_view() {
        let (mat, _, _) = setup_named_matrix();
        let r_idx = vec![0, 2]; // row 0, row 2
        let c_idx = vec![1, 2]; // col 1, col 2

        let view = MatSliceView::new(mat.as_ref(), &r_idx, &c_idx);
        assert_eq!(view.nrows(), 2);
        assert_eq!(view.ncols(), 2);

        let owned = view.to_owned();
        // original (0, 1) -> 1.0
        assert!((owned[(0, 0)] - 1.0).abs() < 1e-6);
        // original (2, 2) -> 8.0
        assert!((owned[(1, 1)] - 8.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "You selected indices larger than ncol.")]
    fn test_mat_slice_view_out_of_bounds_col() {
        let (mat, _, _) = setup_named_matrix();
        let r_idx = vec![0];
        let c_idx = vec![5]; // Out of bounds
        let _view = MatSliceView::new(mat.as_ref(), &r_idx, &c_idx);
    }
}
