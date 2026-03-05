//! Helper structure to store named matrices

use faer::{Mat, MatRef};
use std::collections::BTreeMap;

use crate::prelude::*;

//////////////
// Matrices //
//////////////

/// Structure to store named matrices
#[derive(Clone, Debug)]
pub struct NamedMatrix<'a, T> {
    /// A BTreeMap representing the column names and the col indices
    pub col_names: BTreeMap<String, usize>,
    /// A BTreeMap representing the row names and the row indices
    pub row_names: BTreeMap<String, usize>,
    /// A faer matrix reference representing the matrix values.
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
