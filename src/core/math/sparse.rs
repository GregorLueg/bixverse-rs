use ann_search_rs::utils::dist::SimdDistance;
use faer::{Mat, MatRef};
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul};

use crate::core::math::pca_svd::SvdResults;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Counts the zeroes in a given faer matrix
///
/// ### Params
///
/// * `mat` - The respective faer matrix
///
/// ### Returns
///
/// A tuple with the first being the total zeroes, the second the zeroes per
/// row and the last element being the column zeroes.
pub fn count_zeroes<T>(mat: &MatRef<T>) -> (usize, Vec<usize>, Vec<usize>)
where
    T: BixverseFloat,
{
    let (nrow, ncol) = mat.shape();
    let mut total_zeroes = 0_usize;
    let mut row_zeroes = vec![0_usize; nrow];
    let mut col_zeroes = vec![0_usize; ncol];

    let zero = T::zero();

    for j in 0..ncol {
        for i in 0..nrow {
            let val = unsafe { mat.get_unchecked(i, j) };
            if *val == zero {
                total_zeroes += 1;
                row_zeroes[i] += 1;
                col_zeroes[j] += 1;
            }
        }
    }

    (total_zeroes, row_zeroes, col_zeroes)
}

//////////////////////////////
// Sparse format conversion //
//////////////////////////////

/// Type to describe the CompressedSparseFormat
#[derive(Debug, Clone)]
pub enum CompressedSparseFormat {
    /// CSC-formatted data
    Csc,
    /// CSR-formatted data
    Csr,
}

/// Helper function to parse compressed sparse format
///
/// ### Params
///
/// * `s` - String specifying the type
///
/// ### Return
///
/// Returns an `Option<CompressedSparseFormat>`
pub fn parse_compressed_sparse_format(s: &str) -> Option<CompressedSparseFormat> {
    match s.to_lowercase().as_str() {
        "csr" => Some(CompressedSparseFormat::Csr),
        "csc" => Some(CompressedSparseFormat::Csc),
        _ => None,
    }
}

#[allow(dead_code)]
impl CompressedSparseFormat {
    /// Returns boolean if it's CSC
    pub fn is_csc(&self) -> bool {
        matches!(self, CompressedSparseFormat::Csc)
    }
    /// Returns boolean if it's CSR
    pub fn is_csr(&self) -> bool {
        matches!(self, CompressedSparseFormat::Csr)
    }
}

/// Structure to store compressed sparse data of either type
///
/// ### Fields
///
/// * `data` - The values
/// * `indices` - The indices of the values
/// * `indptr` - The index pointers
/// * `cs_type` - Is the data stored in `Csr` or `Csc`.
/// * `data_2` - An optional second data layer
/// * `shape` - The shape of the underlying matrix
#[derive(Debug, Clone)]
pub struct CompressedSparseData<T, U = T>
where
    T: Clone,
    U: Clone,
{
    pub data: Vec<T>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    pub cs_type: CompressedSparseFormat,
    pub data_2: Option<Vec<U>>,
    pub shape: (usize, usize),
}

impl<T, U> CompressedSparseData<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    /// Generate a nes CSC version of the matrix
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data
    /// * `indices` - The index positions (in this case row indices)
    /// * `indptr` - The index pointer (in this case the column index pointers)
    /// * `data2` - An optional second layer
    #[allow(dead_code)]
    pub fn new_csc(
        data: &[T],
        indices: &[usize],
        indptr: &[usize],
        data2: Option<&[U]>,
        shape: (usize, usize),
    ) -> Self {
        Self {
            data: data.to_vec(),
            indices: indices.to_vec(),
            indptr: indptr.to_vec(), // Fixed: was using indices instead of indptr
            cs_type: CompressedSparseFormat::Csc,
            data_2: data2.map(|d| d.to_vec()),
            shape,
        }
    }

    /// Generate a nes CSR version of the matrix
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data
    /// * `indices` - The index positions (in this case row indices)
    /// * `indptr` - The index pointer (in this case the column index pointers)
    /// * `data2` - An optional second layer
    pub fn new_csr(
        data: &[T],
        indices: &[usize],
        indptr: &[usize],
        data2: Option<&[U]>,
        shape: (usize, usize),
    ) -> Self {
        Self {
            data: data.to_vec(),
            indices: indices.to_vec(),
            indptr: indptr.to_vec(), // Fixed: was using indices instead of indptr
            cs_type: CompressedSparseFormat::Csr,
            data_2: data2.map(|d| d.to_vec()),
            shape,
        }
    }

    /// Transform from CSC to CSR or vice versa
    ///
    /// ### Returns
    ///
    /// The transformed/transposed version
    pub fn transform(&self) -> Self {
        match self.cs_type {
            CompressedSparseFormat::Csc => csc_to_csr(self),
            CompressedSparseFormat::Csr => csr_to_csc(self),
        }
    }

    /// Transpose and convert
    ///
    /// This is a helper to deal with the h5ad madness. Takes in for example
    /// a genes x cell CSR matrix from h5ad and transforms it into a cell x
    /// genes CSR matrix which bixverse expects. Same for CSC.
    ///
    /// ### Returns
    ///
    /// The transformed/transposed version
    pub fn transpose_and_convert(&self) -> Self {
        match self.cs_type {
            CompressedSparseFormat::Csr => {
                // convert first and then switch around
                let csc_version = csr_to_csc(self);
                CompressedSparseData {
                    data: csc_version.data,
                    indices: csc_version.indices,
                    indptr: csc_version.indptr,
                    cs_type: CompressedSparseFormat::Csr, // relabel as CSR
                    data_2: csc_version.data_2,
                    shape: (self.shape.1, self.shape.0), // swap dimensions
                }
            }
            CompressedSparseFormat::Csc => {
                // no conversion needed here! simple transpose is enough...
                CompressedSparseData {
                    data: self.data.clone(),
                    indices: self.indices.clone(),
                    indptr: self.indptr.clone(),
                    cs_type: CompressedSparseFormat::Csr,
                    data_2: self.data_2.clone(),
                    shape: (self.shape.1, self.shape.0),
                }
            }
        }
    }

    /// Transpose the matrix
    #[allow(dead_code)]
    pub fn transpose_from_h5ad(&self) -> Self {
        CompressedSparseData {
            data: self.data.clone(),
            indices: self.indices.clone(),
            indptr: self.indptr.clone(),
            cs_type: self.cs_type.clone(),
            data_2: self.data_2.clone(),
            shape: (self.shape.1, self.shape.0),
        }
    }

    /// Generates a sparse matrix from a dense matrix
    ///
    /// ### Params
    ///
    /// * `mat`: The dense matrix to convert to a sparse matrix
    /// * `format`: The format of the sparse matrix to generate
    ///
    /// ### Returns
    ///
    /// * `Self`: The sparse matrix generated from the dense matrix
    pub fn from_dense_matrix(mat: faer::MatRef<T>, format: CompressedSparseFormat) -> Self
    where
        T: BixverseFloat,
    {
        let (nrows, ncols) = (mat.nrows(), mat.ncols());
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = Vec::new();

        match format {
            CompressedSparseFormat::Csr => {
                indptr.push(0);
                for i in 0..nrows {
                    for j in 0..ncols {
                        let val = mat[(i, j)];
                        if val != T::zero() {
                            data.push(val);
                            indices.push(j);
                        }
                    }
                    indptr.push(data.len());
                }
            }
            CompressedSparseFormat::Csc => {
                indptr.push(0);
                for j in 0..ncols {
                    for i in 0..nrows {
                        let val = mat[(i, j)];
                        if val != T::zero() {
                            data.push(val);
                            indices.push(i);
                        }
                    }
                    indptr.push(data.len());
                }
            }
        }

        Self {
            data,
            indices,
            indptr,
            cs_type: format,
            data_2: None,
            shape: (nrows, ncols),
        }
    }

    /// Create a sparse matrix from an upper triangular matrix.
    ///
    /// ### Params
    ///
    /// * `upper_triangle` - The upper triangular matrix.
    /// * `n` - The number of rows and columns in the matrix.
    /// * `include_diagonal` - Whether to include the diagonal elements.
    /// * `format` - The format of the sparse matrix.
    ///
    /// ### Returns
    ///
    /// A sparse matrix.
    pub fn from_upper_triangle_sym(
        upper_triangle: &[T],
        n: usize,
        include_diagonal: bool,
        format: CompressedSparseFormat,
    ) -> Self
    where
        T: BixverseFloat,
    {
        // lambda function in Rust style...
        let get_value = |row: usize, col: usize| -> T {
            if row == col {
                if include_diagonal {
                    let offset = row * n - row * (row + 1) / 2 + col;
                    upper_triangle[offset]
                } else {
                    T::one()
                }
            } else if row < col {
                let offset = if include_diagonal {
                    row * n - row * (row + 1) / 2 + col
                } else {
                    row * (n - 1) - row * (row + 1) / 2 + col - 1
                };
                upper_triangle[offset]
            } else {
                let offset = if include_diagonal {
                    col * n - col * (col + 1) / 2 + row
                } else {
                    col * (n - 1) - col * (col + 1) / 2 + row - 1
                };
                upper_triangle[offset]
            }
        };

        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = Vec::new();

        match format {
            CompressedSparseFormat::Csr => {
                indptr.push(0);
                for row in 0..n {
                    for col in 0..n {
                        let value = get_value(row, col);
                        if value != T::zero() {
                            data.push(value);
                            indices.push(col);
                        }
                    }
                    indptr.push(data.len());
                }
            }
            CompressedSparseFormat::Csc => {
                indptr.push(0);
                for col in 0..n {
                    for row in 0..n {
                        let value = get_value(row, col);
                        if value != T::zero() {
                            data.push(value);
                            indices.push(row);
                        }
                    }
                    indptr.push(data.len());
                }
            }
        }

        Self {
            data,
            indices,
            indptr,
            cs_type: format,
            data_2: None,
            shape: (n, n),
        }
    }

    /// Returns the shape of the matrix
    ///
    /// ### Returns
    ///
    /// A tuple of `(nrow, ncol)`
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns the NNZ
    ///
    /// ### Returns
    ///
    /// The number of NNZ
    pub fn get_nnz(&self) -> usize {
        self.data.len()
    }

    /// Return the second layer
    ///
    /// If this does not exist, the function will panic
    ///
    /// ### Returns
    ///
    /// Vector of the second layer
    pub fn get_data2_unsafe(&self) -> Vec<U> {
        self.data_2.clone().unwrap()
    }
}

/// Transforms a CompressedSparseData that is CSC to CSR
///
/// ### Params
///
/// * `sparse_data` - The CompressedSparseData you want to transform
pub fn csc_to_csr<T, U>(sparse_data: &CompressedSparseData<T, U>) -> CompressedSparseData<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let (nrow, _) = sparse_data.shape();
    let nnz = sparse_data.get_nnz();
    let mut row_ptr = vec![0; nrow + 1];

    for &r in &sparse_data.indices {
        row_ptr[r + 1] += 1;
    }

    for i in 0..nrow {
        row_ptr[i + 1] += row_ptr[i];
    }

    let mut csr_data = vec![T::default(); nnz];
    let mut csr_data2 = sparse_data.data_2.as_ref().map(|_| vec![U::default(); nnz]);
    let mut csr_col_ind = vec![0; nnz];
    let mut next = row_ptr[..nrow].to_vec();

    for col in 0..(sparse_data.indptr.len() - 1) {
        for idx in sparse_data.indptr[col]..sparse_data.indptr[col + 1] {
            let row = sparse_data.indices[idx];
            let pos = next[row];

            csr_data[pos] = sparse_data.data[idx];
            csr_col_ind[pos] = col;

            // Handle the second layer data
            if let (Some(source_data2), Some(csr_d2)) = (&sparse_data.data_2, &mut csr_data2) {
                csr_d2[pos] = source_data2[idx];
            }

            next[row] += 1;
        }
    }

    CompressedSparseData {
        data: csr_data,
        indices: csr_col_ind,
        indptr: row_ptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: csr_data2,
        shape: sparse_data.shape(),
    }
}

/// Transform CSR stored data into CSC stored data
///
/// This version does a full memory copy of the data.
///
/// ### Params
///
/// * `sparse_data` - The CompressedSparseData you want to transform
///
/// ### Returns
///
/// The data in CSC format, i.e., `CscData`
pub fn csr_to_csc<T, U>(sparse_data: &CompressedSparseData<T, U>) -> CompressedSparseData<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let nnz = sparse_data.get_nnz();
    let (_, ncol) = sparse_data.shape();
    let mut col_ptr = vec![0; ncol + 1];

    // count occurrences per column
    for &c in &sparse_data.indices {
        col_ptr[c + 1] += 1;
    }

    // cumulative sum to get column pointers
    for i in 0..ncol {
        col_ptr[i + 1] += col_ptr[i];
    }

    let mut csc_data = vec![T::default(); nnz];
    let mut csc_data2 = sparse_data.data_2.as_ref().map(|_| vec![U::default(); nnz]);
    let mut csc_row_ind = vec![0; nnz];
    let mut next = col_ptr[..ncol].to_vec();

    // iterate through rows and place data in CSC format
    for row in 0..(sparse_data.indptr.len() - 1) {
        for idx in sparse_data.indptr[row]..sparse_data.indptr[row + 1] {
            let col = sparse_data.indices[idx];
            let pos = next[col];

            csc_data[pos] = sparse_data.data[idx];
            csc_row_ind[pos] = row;

            // handle the second layer data
            if let (Some(source_data2), Some(csc_d2)) = (&sparse_data.data_2, &mut csc_data2) {
                csc_d2[pos] = source_data2[idx];
            }

            next[col] += 1;
        }
    }

    CompressedSparseData {
        data: csc_data,
        indices: csc_row_ind,
        indptr: col_ptr,
        cs_type: CompressedSparseFormat::Csc,
        data_2: csc_data2,
        shape: sparse_data.shape(),
    }
}

/// Transform COO stored data into CSR
///
/// ### Params
///
/// * `rows` - Row indices
/// * `cols` - Col indices
/// * `vals` - The values to store in the matrix
///
/// ### Returns
///
/// `CompressedSparseData` in CSR format
pub fn coo_to_csr<T>(
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
    shape: (usize, usize),
) -> CompressedSparseData<T>
where
    T: BixverseNumeric,
{
    let n_rows = shape.0;

    // sort by (row, col) and merge duplicates
    let mut entries: Vec<(usize, usize, T)> = rows
        .iter()
        .zip(cols.iter())
        .zip(vals.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();

    entries.sort_unstable_by_key(|&(r, c, _)| (r, c));

    // merge duplicates; can happen during additions
    let mut merged_entries = Vec::new();
    if !entries.is_empty() {
        let mut current = entries[0];

        for &(r, c, v) in &entries[1..] {
            if r == current.0 && c == current.1 {
                current.2 += v;
            } else {
                if current.2 != T::default() {
                    merged_entries.push(current);
                }
                current = (r, c, v);
            }
        }
        if current.2 != T::default() {
            merged_entries.push(current);
        }
    }

    // build CSR from merged entries
    let final_nnz = merged_entries.len();
    let mut data = Vec::with_capacity(final_nnz);
    let mut indices = Vec::with_capacity(final_nnz);
    let mut indptr = vec![0usize; n_rows + 1];

    for &(row, col, val) in &merged_entries {
        data.push(val);
        indices.push(col);
        indptr[row + 1] += 1;
    }

    // Convert counts to cumulative offsets
    for i in 0..n_rows {
        indptr[i + 1] += indptr[i];
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, None, shape)
}

/// Optimised COO to CSR - assumes input is already sorted by (row, col)
///
/// ### Params
///
/// * `rows` - Row indices (must be sorted by row first, then col)
/// * `cols` - Col indices
/// * `vals` - Values
/// * `shape` - Matrix dimensions
/// * `is_sorted` - If true, skips sorting step
///
/// ### Returns
///
/// CSR matrix
pub fn coo_to_csr_presorted<T>(
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
    shape: (usize, usize),
) -> CompressedSparseData<T>
where
    T: BixverseNumeric,
{
    let n_rows = shape.0;
    let nnz = rows.len();

    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    let mut indptr = vec![0usize; n_rows + 1];

    // unsafe to squeeze out performance...
    unsafe {
        data.set_len(nnz);
        indices.set_len(nnz);

        let data_ptr: *mut T = data.as_mut_ptr();
        let indices_ptr: *mut usize = indices.as_mut_ptr();
        let indptr_ptr: *mut usize = indptr.as_mut_ptr();

        for i in 0..nnz {
            *data_ptr.add(i) = *vals.get_unchecked(i);
            *indices_ptr.add(i) = *cols.get_unchecked(i);
            let row = *rows.get_unchecked(i);
            *indptr_ptr.add(row + 1) += 1;
        }

        for i in 0..n_rows {
            *indptr_ptr.add(i + 1) += *indptr_ptr.add(i);
        }
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, None, shape)
}

/// Add two CSR matrices together
///
/// ### Params
///
/// * `a` - Reference to the first CompressedSparseData (in CSR format!)
/// * `b` - Reference to the second CompressedSparseData (in CSR format!)
///
/// ### Returns
///
/// `CompressedSparseData` with added values between the two.
pub fn sparse_add_csr<T>(
    a: &CompressedSparseData<T>,
    b: &CompressedSparseData<T>,
) -> CompressedSparseData<T>
where
    T: BixverseNumeric + Into<f64> + Add<Output = T>,
{
    assert_eq!(a.shape, b.shape);
    assert!(a.cs_type.is_csr() && b.cs_type.is_csr());

    const EPSILON: f32 = 1e-9;
    let n_rows = a.shape.0;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n_rows {
        let a_start = a.indptr[i];
        let a_end = a.indptr[i + 1];
        let b_start = b.indptr[i];
        let b_end = b.indptr[i + 1];

        let mut a_idx = a_start;
        let mut b_idx = b_start;

        while a_idx < a_end || b_idx < b_end {
            if a_idx < a_end && (b_idx >= b_end || a.indices[a_idx] < b.indices[b_idx]) {
                rows.push(i);
                cols.push(a.indices[a_idx]);
                vals.push(a.data[a_idx]);
                a_idx += 1;
            } else if b_idx < b_end && (a_idx >= a_end || b.indices[b_idx] < a.indices[a_idx]) {
                rows.push(i);
                cols.push(b.indices[b_idx]);
                vals.push(b.data[b_idx]);
                b_idx += 1;
            } else {
                let val = a.data[a_idx] + b.data[b_idx];
                if val.into().abs() > EPSILON as f64 {
                    rows.push(i);
                    cols.push(a.indices[a_idx]);
                    vals.push(val);
                }
                a_idx += 1;
                b_idx += 1;
            }
        }
    }

    // output is already sorted by (row, col), build CSR directly
    coo_to_csr_presorted(&rows, &cols, &vals, a.shape)
}

/// Scalar multiplication of CSR matrix
///
/// ### Params
///
/// * `a` - Reference to the first CompressedSparseData (in CSR format!)
/// * `scalar` - The scalar value to multiply with
///
/// ### Returns
///
/// `CompressedSparseData` with the data multiplied by the scalar.
pub fn sparse_scalar_multiply_csr<T>(
    a: &CompressedSparseData<T>,
    scalar: T,
) -> CompressedSparseData<T>
where
    T: BixverseNumeric,
    <T as Mul>::Output: Send,
    Vec<T>: FromParallelIterator<<T as Mul>::Output>,
{
    let data: Vec<T> = a.data.par_iter().map(|&v| v * scalar).collect();
    CompressedSparseData::new_csr(&data, &a.indices, &a.indptr, None, a.shape)
}

/// Sparse matrix subtraction
///
/// ### Params
///
/// * `a` - Reference to the first CompressedSparseData (in CSR format!)
/// * `b` - Reference to the second CompressedSparseData (in CSR format!)
///
/// ### Returns
///
/// The subtracted new matrix
pub fn sparse_subtract_csr<T>(
    a: &CompressedSparseData<T>,
    b: &CompressedSparseData<T>,
) -> CompressedSparseData<T>
where
    T: BixverseNumeric + Into<f64>,
{
    assert_eq!(a.shape, b.shape);
    assert!(a.cs_type.is_csr() && b.cs_type.is_csr());

    const EPSILON: f32 = 1e-9;
    let n_rows = a.shape.0;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n_rows {
        let a_start = a.indptr[i];
        let a_end = a.indptr[i + 1];
        let b_start = b.indptr[i];
        let b_end = b.indptr[i + 1];

        let mut a_idx = a_start;
        let mut b_idx = b_start;

        while a_idx < a_end || b_idx < b_end {
            if a_idx < a_end && (b_idx >= b_end || a.indices[a_idx] < b.indices[b_idx]) {
                rows.push(i);
                cols.push(a.indices[a_idx]);
                vals.push(a.data[a_idx]);
                a_idx += 1;
            } else if b_idx < b_end && (a_idx >= a_end || b.indices[b_idx] < a.indices[a_idx]) {
                rows.push(i);
                cols.push(b.indices[b_idx]);
                vals.push(T::default() - b.data[b_idx]);
                b_idx += 1;
            } else {
                let val = a.data[a_idx] - b.data[b_idx];
                if val.into().abs() > EPSILON as f64 {
                    rows.push(i);
                    cols.push(a.indices[a_idx]);
                    vals.push(val);
                }
                a_idx += 1;
                b_idx += 1;
            }
        }
    }

    // already sorted
    coo_to_csr_presorted(&rows, &cols, &vals, a.shape)
}

/// Element-wise sparse multiplication
///
/// ### Params
///
/// * `a` - Reference to the first CompressedSparseData (in CSR format!)
/// * `b` - Reference to the second CompressedSparseData (in CSR format!)
///
/// ### Returns
///
/// The multiplied matrix.
pub fn sparse_multiply_elementwise_csr<T>(
    a: &CompressedSparseData<T>,
    b: &CompressedSparseData<T>,
) -> CompressedSparseData<T>
where
    T: BixverseNumeric,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    assert_eq!(a.shape, b.shape);
    assert!(a.cs_type.is_csr() && b.cs_type.is_csr());
    let n_rows = a.shape.0;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n_rows {
        let a_start = a.indptr[i];
        let a_end = a.indptr[i + 1];
        let b_start = b.indptr[i];
        let b_end = b.indptr[i + 1];
        let mut a_idx = a_start;
        let mut b_idx = b_start;
        while a_idx < a_end && b_idx < b_end {
            match a.indices[a_idx].cmp(&b.indices[b_idx]) {
                std::cmp::Ordering::Less => {
                    a_idx += 1;
                }
                std::cmp::Ordering::Greater => {
                    b_idx += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Same column - multiply
                    let val = a.data[a_idx] * b.data[b_idx];
                    if val != T::default() {
                        rows.push(i);
                        cols.push(a.indices[a_idx]);
                        vals.push(val);
                    }
                    a_idx += 1;
                    b_idx += 1;
                }
            }
        }
    }
    coo_to_csr(&rows, &cols, &vals, a.shape)
}

/// Normalises the columns of a CSR matrix to a sum of 1 (L1 norm)
///
/// ### Params
///
/// * `csr` - Mutable reference to the CSR matrix (modified in-place)
pub fn normalise_csr_columns_l1<T>(csr: &mut CompressedSparseData<T>)
where
    T: BixverseNumeric + Into<f64>,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    assert!(csr.cs_type.is_csr(), "Matrix must be in CSR format");

    let ncols = csr.shape.1;

    let mut col_sums = vec![T::default(); ncols];

    for (idx, &col) in csr.indices.iter().enumerate() {
        col_sums[col] += csr.data[idx]
    }

    for (idx, &col) in csr.indices.iter().enumerate() {
        let sum = col_sums[col];
        if sum.into() > 1e-15 {
            csr.data[idx] /= sum;
        }
    }
}

/// Normalises the rows of a CSR matrix to a sum of 1 (L1 norm)
///
/// ### Params
///
/// * `csr` - Mutable reference to the CSR matrix (modified in-place)
#[allow(dead_code)]
pub fn normalise_csr_rows_l1<T>(csr: &mut CompressedSparseData<T>)
where
    T: BixverseNumeric + Into<f64>,
    // We also need the `Sum` trait for `iter().sum()`
    T: std::iter::Sum<T>,
{
    assert!(csr.cs_type.is_csr(), "Matrix must be in CSR format");

    let nrows = csr.shape.0;

    for i in 0..nrows {
        let start = csr.indptr[i];
        let end = csr.indptr[i + 1];
        let row_data_slice = &mut csr.data[start..end];

        let row_sum: T = row_data_slice.iter().copied().sum();

        if row_sum.into() > 1e-15 {
            for val in row_data_slice.iter_mut() {
                *val /= row_sum;
            }
        } else {
            panic!(
                "Row {} has sum {}, indicating isolated node",
                i,
                row_sum.into()
            );
        }
    }
}

/// Compute Frobenius norm of sparse matrix
///
/// ### Params
///
/// * `mat` - Sparse matrix in CSR or CSC format
///
/// ### Returns
///
/// Frobenius norm ||A||_F = sqrt(sum(A_ij^2))
pub fn frobenius_norm<T>(mat: &CompressedSparseData<T>) -> f32
where
    T: BixverseNumeric + Into<f32>,
{
    mat.data
        .par_iter()
        .with_min_len(10000)
        .map(|&v| {
            let val: f32 = v.into();
            val * val
        })
        .sum::<f32>()
        .sqrt()
}

/// Remove zeros from sparse matrix
///
/// ### Params
///
/// * `mat` - Matrix from which to remove the zeroes
///
/// ### Returns
///
/// The Matrix with 0's removed.
pub fn eliminate_zeros_csr<T>(mat: CompressedSparseData<T>) -> CompressedSparseData<T>
where
    T: BixverseNumeric,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    let n_rows = mat.shape.0;
    for i in 0..n_rows {
        let start = mat.indptr[i];
        let end = mat.indptr[i + 1];

        for j in start..end {
            if mat.data[j] != T::default() {
                rows.push(i);
                cols.push(mat.indices[j]);
                vals.push(mat.data[j]);
            }
        }
    }

    coo_to_csr(&rows, &cols, &vals, mat.shape)
}

/// Sparse matrix-vector multiplication
///
/// Multiply a sparse CSR matrix with a vector
///
/// ### Params
///
/// * `mat` - The Compressed Sparse matrix in CSR format
/// * `vec` - The Vector to multiply with
///
/// ### Params
///
/// The resulting vector
pub fn csr_matvec<T>(mat: &CompressedSparseData<T>, vec: &[T]) -> Vec<T>
where
    T: BixverseNumeric,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    let mut result = vec![T::default(); mat.shape.0];
    for i in 0..mat.shape.0 {
        let row_start = mat.indptr[i];
        let row_end = mat.indptr[i + 1];
        let mut sum = T::default();
        for idx in row_start..row_end {
            sum += mat.data[idx] * vec[mat.indices[idx]];
        }
        result[i] = sum;
    }
    result
}

/// Sparse accumulator for efficient sparse matrix multiplication
///
/// ### Fields
///
/// * `values` - Vector storing accumulated values for each index
/// * `indices` - Vector of active (non-zero) indices
/// * `flags` - Boolean flags indicating which indices are active
struct SparseAccumulator<T>
where
    T: Copy + Default + AddAssign,
{
    values: Vec<T>,
    indices: Vec<usize>,
    flags: Vec<bool>,
}

impl<T> SparseAccumulator<T>
where
    T: Copy + Default + AddAssign,
{
    /// Create a new sparse accumulator
    ///
    /// ### Params
    ///
    /// * `size` - Maximum number of indices to accumulate
    fn new(size: usize) -> Self {
        Self {
            values: vec![T::default(); size],
            indices: Vec::with_capacity(size / 10),
            flags: vec![false; size],
        }
    }

    /// Add a value to the accumulator at the given index
    ///
    /// ### Params
    ///
    /// * `idx` - Index to accumulate at
    /// * `val` - Value to add
    ///
    /// ### Safety
    ///
    /// `idx` must be less than the size specified during construction
    #[inline]
    unsafe fn add(&mut self, idx: usize, val: T) {
        unsafe {
            if !*self.flags.get_unchecked(idx) {
                *self.flags.get_unchecked_mut(idx) = true;
                self.indices.push(idx);
                *self.values.get_unchecked_mut(idx) = val;
            } else {
                *self.values.get_unchecked_mut(idx) += val;
            }
        }
    }

    /// Extract accumulated values as sorted index-value pairs and reset the accumulator
    ///
    /// ### Returns
    ///
    /// Vector of (index, value) pairs sorted by index
    #[inline]
    fn extract_sorted(&mut self) -> Vec<(usize, T)> {
        self.indices.sort_unstable();
        let result: Vec<(usize, T)> = unsafe {
            self.indices
                .iter()
                .map(|&i| (i, *self.values.get_unchecked(i)))
                .collect()
        };
        // Reset for next use
        unsafe {
            for &idx in &self.indices {
                *self.flags.get_unchecked_mut(idx) = false;
                *self.values.get_unchecked_mut(idx) = T::default();
            }
        }
        self.indices.clear();
        result
    }
}

/// Multiply two CSR matrices using sparse accumulators and parallel processing
///
/// ### Params
///
/// * `a` - Left CSR matrix
/// * `b` - Right CSR matrix
///
/// ### Returns
///
/// Product matrix in CSR format
pub fn csr_matmul_csr<T>(
    a: &CompressedSparseData<T>,
    b: &CompressedSparseData<T>,
) -> CompressedSparseData<T>
where
    T: BixverseNumeric,
{
    assert!(a.cs_type.is_csr() && b.cs_type.is_csr());
    assert_eq!(a.shape.1, b.shape.0, "Dimension mismatch");

    let nrows = a.shape.0;
    let ncols = b.shape.1;

    let row_results: Vec<Vec<(usize, T)>> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let mut acc = SparseAccumulator::new(ncols);

            unsafe {
                let a_indptr = a.indptr.as_ptr();
                let a_indices = a.indices.as_ptr();
                let a_data = a.data.as_ptr();
                let b_indptr = b.indptr.as_ptr();
                let b_indices = b.indices.as_ptr();
                let b_data = b.data.as_ptr();

                let a_start = *a_indptr.add(i);
                let a_end = *a_indptr.add(i + 1);

                for a_idx in a_start..a_end {
                    let k = *a_indices.add(a_idx);
                    let a_val = *a_data.add(a_idx);

                    let b_start = *b_indptr.add(k);
                    let b_end = *b_indptr.add(k + 1);

                    for b_idx in b_start..b_end {
                        let j = *b_indices.add(b_idx);
                        let b_val = *b_data.add(b_idx);
                        acc.add(j, a_val * b_val);
                    }
                }
            }

            acc.extract_sorted()
        })
        .collect();

    // direct CSR construction
    let total_nnz: usize = row_results.iter().map(|r| r.len()).sum();
    let mut data = Vec::with_capacity(total_nnz);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut indptr = Vec::with_capacity(nrows + 1);
    indptr.push(0);

    for row in row_results {
        for (col, val) in row {
            data.push(val);
            indices.push(col);
        }
        indptr.push(data.len());
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, None, (nrows, ncols))
}

/////////////////////////////////////
// Lanczos Eigenvalue calculations //
/////////////////////////////////////

/// Helper function for dot product of two vectors
///
/// ### Params
///
/// * `a` - Vector a
/// * `b` - Vector b
///
/// ### Returns
///
/// Dot product of the two vectors
fn dot<T>(a: &[T], b: &[T]) -> T
where
    T: SimdDistance,
{
    assert_same_len!(a, b);
    T::dot_simd(a, b)
}

/// Helper function to normalise a vector
///
/// ### Params
///
/// * `v` - Initial vector
///
/// ### Returns
///
/// Normalised dot product of the vector `v`
fn norm<T>(v: &[T]) -> T
where
    T: SimdDistance + BixverseFloat,
{
    let dot = dot(v, v);
    dot.sqrt()
}

/// Helper function to normalise a vector
///
/// ### Params
///
/// * `v` - Mutable reference of the vector to normalise
fn normalise<T>(v: &mut [T])
where
    T: SimdDistance + BixverseFloat,
{
    let n = norm(v);
    v.par_iter_mut().for_each(|x| *x /= n);
}

/// Helper function to calculate eigenvalues
///
/// ### Params
///
/// * `alpha` - alpha vector
/// * `beta` - beta vector
///
/// ### Returns
///
/// Tuple of `(eigenvectors, eigenvalues)`
fn tridiag_eig<T>(alpha: &[T], beta: &[T]) -> (Vec<T>, Mat<T>)
where
    T: BixverseFloat,
{
    let n = alpha.len();
    let mut t = Mat::<T>::zeros(n, n);

    for i in 0..n {
        t[(i, i)] = alpha[i];
        if i < n - 1 {
            t[(i, i + 1)] = beta[i];
            t[(i + 1, i)] = beta[i];
        }
    }

    let eig = t.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let evals = eig.S().column_vector().iter().copied().collect();
    let evecs = eig.U().to_owned();

    (evals, evecs)
}

/// Compute largest eigenvalues and eigenvectors using Lanczos
///
/// ### Params
///
/// * `matrix` - Sparse matrix in CSR format
/// * `n_components` - Number of eigenpairs to compute
/// * `seed` - For reproducibility
///
/// ### Returns
///
/// (eigenvalues, eigenvectors) where eigenvectors[i][j] is element j of
/// eigenvector i
pub fn compute_largest_eigenpairs_lanczos<T>(
    matrix: &CompressedSparseData<T>,
    n_components: usize,
    seed: u64,
) -> (Vec<f32>, Vec<Vec<f32>>)
where
    T: BixverseNumeric + SimdDistance + Into<f64>,
{
    let n = matrix.shape.0;
    let n_iter = (n_components * 2 + 10).max(n_components).min(n);

    // Convert to CSR for efficient row access
    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix.clone(),
        CompressedSparseFormat::Csc => matrix.transform(),
    };

    let data_f64: Vec<f64> = csr.data.iter().map(|&v| v.into()).collect();

    // Parallelised matvec: y = A * x
    let matvec = |x: &[f64], y: &mut [f64]| {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = 0.0;
            for idx in csr.indptr[i]..csr.indptr[i + 1] {
                let j = csr.indices[idx];
                sum += data_f64[idx] * x[j];
            }
            *yi = sum;
        });
    };

    // Lanczos iteration
    let mut v = vec![0.0; n];
    let mut v_old = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut v_matrix = vec![vec![0.0; n]; n_iter];

    let mut rng = StdRng::seed_from_u64(seed);

    for i in 0..n {
        v[i] = rng.random::<f64>() - 0.5;
    }
    normalise(&mut v);

    let mut alpha = vec![0.0; n_iter];
    let mut beta = vec![0.0; n_iter];

    for j in 0..n_iter {
        v_matrix[j].copy_from_slice(&v);

        matvec(&v, &mut w);
        alpha[j] = dot(&w, &v);

        // w = w - alpha[j]*v - beta[j-1]*v_old
        for i in 0..n {
            w[i] -= alpha[j] * v[i];
            if j > 0 {
                w[i] -= beta[j - 1] * v_old[i];
            }
        }

        beta[j] = norm(&w);
        if beta[j] < 1e-12 {
            break;
        }

        v_old.copy_from_slice(&v);
        v.copy_from_slice(&w);
        normalise(&mut v);
    }

    let (evals, evecs) = tridiag_eig(&alpha[..n_iter], &beta[..n_iter - 1]);

    let mut indices: Vec<usize> = (0..evals.len()).collect();
    indices.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap());

    let mut largest_evals: Vec<f32> = Vec::with_capacity(n_components);
    let mut largest_evecs: Vec<Vec<f32>> = Vec::with_capacity(n_components);

    for &idx in indices.iter().take(n_components) {
        // Transform eigenvector back to original space: v_original = V * v_tridiag
        let mut evec = vec![0.0; n];
        for i in 0..n {
            for j in 0..n_iter {
                evec[i] += v_matrix[j][i] * evecs[(j, idx)].to_f64().unwrap();
            }
        }

        // Normalise the transformed eigenvector
        let norm: f64 = evec.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut evec {
            *x /= norm;
        }

        largest_evals.push(evals[idx].to_f64().unwrap() as f32);
        largest_evecs.push(evec.iter().map(|&x| x as f32).collect());
    }

    let mut transposed = vec![vec![0.0f32; n_components]; n];
    for comp_idx in 0..n_components {
        for point_idx in 0..n {
            transposed[point_idx][comp_idx] = largest_evecs[comp_idx][point_idx];
        }
    }

    (largest_evals, transposed)
}

/////////////////
// Lanczos SVD //
/////////////////

/// Compute sparse SVD using Lanczos on A^T A or AA^T
///
/// ### Params
///
/// * `matrix` - Sparse matrix (CSR or CSC)
/// * `n_components` - Number of singular values/vectors to compute
/// * `seed` - For reproducibility
/// * `use_second_layer` - If true, use data_2 instead of data
///
/// ### Returns
///
/// (U, S, V^T) where U is n×k, S is length k, V^T is k×m
/// Compute sparse SVD using Lanczos on A^T A or AA^T
///
/// ### Params
///
/// * `matrix` - Sparse matrix (CSR or CSC)
/// * `n_components` - Number of singular values/vectors to compute
/// * `seed` - For reproducibility
/// * `use_second_layer` - If true, use data_2 instead of data
///
/// ### Returns
///
/// `RandomSvdResults` containing U (n×k), S (length k), and V (m×k)
pub fn sparse_svd_lanczos<T, U, F>(
    matrix: &CompressedSparseData<T, U>,
    n_components: usize,
    seed: u64,
    use_second_layer: bool,
) -> SvdResults<F>
where
    T: BixverseNumeric + SimdDistance + Into<F>,
    U: BixverseNumeric + Into<F> + Clone,
    F: BixverseFloat + SimdDistance + std::iter::Sum,
{
    let (n, m) = matrix.shape;

    let use_ata = n > m;
    let krylov_dim = if use_ata { m } else { n };

    let n_iter = (n_components * 2 + 10).max(n_components).min(krylov_dim);

    // ensure we have CSR for efficient operations
    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix.clone(),
        CompressedSparseFormat::Csc => matrix.transform(),
    };

    let data_f: Vec<F> = if use_second_layer {
        csr.data_2
            .as_ref()
            .expect("data_2 is None but use_second_layer is true")
            .iter()
            .map(|&v| v.into())
            .collect()
    } else {
        csr.data.iter().map(|&v| v.into()).collect()
    };

    // Parallelized A*x: y = A * x
    let matvec_a = |x: &[F], y: &mut [F]| {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = F::zero();
            for idx in csr.indptr[i]..csr.indptr[i + 1] {
                let j = csr.indices[idx];
                sum += data_f[idx] * x[j];
            }
            *yi = sum;
        });
    };

    // Parallelized A^T*x: y = A^T * x
    let matvec_at = |x: &[F], y: &mut [F]| {
        let partial_sums: Vec<F> = (0..n)
            .into_par_iter()
            .fold(
                || vec![F::zero(); m],
                |mut acc, i| {
                    for idx in csr.indptr[i]..csr.indptr[i + 1] {
                        let j = csr.indices[idx];
                        acc[j] += data_f[idx] * x[i];
                    }
                    acc
                },
            )
            .reduce(
                || vec![F::zero(); m],
                |mut a, b| {
                    for i in 0..m {
                        a[i] += b[i];
                    }
                    a
                },
            );
        y.copy_from_slice(&partial_sums);
    };

    // define (A^T A)*x or (AA^T)*x without forming the product
    let matvec_gram: Box<dyn Fn(&[F], &mut [F]) + Sync> = if use_ata {
        Box::new(|x: &[F], y: &mut [F]| {
            let mut temp = vec![F::zero(); n];
            matvec_a(x, &mut temp);
            matvec_at(&temp, y);
        })
    } else {
        Box::new(|x: &[F], y: &mut [F]| {
            let mut temp = vec![F::zero(); m];
            matvec_at(x, &mut temp);
            matvec_a(&temp, y);
        })
    };

    // lanczos iteration
    let mut v = vec![F::zero(); krylov_dim];
    let mut v_old = vec![F::zero(); krylov_dim];
    let mut w = vec![F::zero(); krylov_dim];
    let mut v_matrix = vec![vec![F::zero(); krylov_dim]; n_iter];

    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..krylov_dim {
        v[i] = F::from(rng.random::<f64>() - 0.5).unwrap();
    }
    normalise(&mut v);

    let mut alpha = vec![F::zero(); n_iter];
    let mut beta = vec![F::zero(); n_iter];

    for j in 0..n_iter {
        v_matrix[j].copy_from_slice(&v);

        matvec_gram(&v, &mut w);
        alpha[j] = dot(&w, &v);

        for i in 0..krylov_dim {
            w[i] -= alpha[j] * v[i];
            if j > 0 {
                w[i] -= beta[j - 1] * v_old[i];
            }
        }

        // Full reorthogonalisation against all previous Lanczos vectors
        for k in 0..=j {
            let coeff = dot(&w, &v_matrix[k]);
            for i in 0..krylov_dim {
                w[i] -= coeff * v_matrix[k][i];
            }
        }

        beta[j] = norm(&w);
        if beta[j] < F::from(1e-12).unwrap() {
            break;
        }

        v_old.copy_from_slice(&v);
        v.copy_from_slice(&w);
        normalise(&mut v);
    }

    let (evals, evecs) = tridiag_eig(&alpha[..n_iter], &beta[..n_iter - 1]);

    let mut indices: Vec<usize> = (0..evals.len()).collect();
    indices.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap());

    let mut singular_values: Vec<F> = Vec::with_capacity(n_components);
    let mut u_vecs: Vec<Vec<F>> = Vec::with_capacity(n_components);
    let mut v_vecs: Vec<Vec<F>> = Vec::with_capacity(n_components);

    for &idx in indices.iter().take(n_components) {
        let eval = evals[idx];
        if eval <= F::zero() {
            continue;
        }

        let sigma = eval.sqrt();
        singular_values.push(sigma);

        // transform eigenvector back to original space
        let mut gram_evec = vec![F::zero(); krylov_dim];
        for i in 0..krylov_dim {
            for j in 0..n_iter {
                gram_evec[i] += v_matrix[j][i] * evecs[(j, idx)];
            }
        }

        // normalise
        let norm_val: F = gram_evec.iter().map(|x| *x * *x).sum::<F>().sqrt();
        for x in &mut gram_evec {
            *x /= norm_val;
        }

        if use_ata {
            // gram_evec is eigenvector of A^T A, so it's V
            // U = (1/σ) * A * V
            let mut u_vec = vec![F::zero(); n];
            matvec_a(&gram_evec, &mut u_vec);
            for x in &mut u_vec {
                *x /= sigma;
            }

            u_vecs.push(u_vec);
            v_vecs.push(gram_evec);
        } else {
            // gram_evec is eigenvector of AA^T, so it's U
            // V = (1/σ) * A^T * U
            let mut v_vec = vec![F::zero(); m];
            matvec_at(&gram_evec, &mut v_vec);
            for x in &mut v_vec {
                *x /= sigma;
            }

            u_vecs.push(gram_evec);
            v_vecs.push(v_vec);
        }
    }

    // Convert to faer matrices
    let u = Mat::from_fn(n, singular_values.len(), |i, j| u_vecs[j][i]);
    let v = Mat::from_fn(m, singular_values.len(), |i, j| v_vecs[j][i]);

    SvdResults {
        u,
        s: singular_values,
        v,
    }
}
