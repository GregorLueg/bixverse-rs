//! Sparse matrix formats, sparse operations and helpers to transform different
//! formats into each other.

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

///////////////////////
// Sparse structures //
///////////////////////

/// Type to describe the CompressedSparseFormat
#[derive(Debug, Clone)]
pub enum CompressedSparseFormat {
    /// CSC-formatted data
    Csc,
    /// CSR-formatted data
    Csr,
}

impl CompressedSparseFormat {
    /// Returns boolean if it's CSC
    ///
    /// ### Returns
    ///
    /// Boolean indicating if CSC
    #[inline(always)]
    pub fn is_csc(&self) -> bool {
        matches!(self, CompressedSparseFormat::Csc)
    }
    /// Returns boolean if it's CSR
    ///
    /// ### Returns
    ///
    /// Boolean indicating if CSR
    #[inline(always)]
    pub fn is_csr(&self) -> bool {
        matches!(self, CompressedSparseFormat::Csr)
    }
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

/// Generate structure to store sparse rows or columns
pub struct SparseAxis<T, U = T> {
    /// The indices of the values/non-zero positions
    pub indices: Vec<usize>,
    /// The values in the row/columns
    pub data: Vec<T>,
    /// An optional second data layer
    pub data_2: Option<Vec<U>>,
    /// Is the data stored in `Csr` or `Csc`. `Csr` -> sparse row; `Csc` ->
    /// sparse column
    pub cs_type: CompressedSparseFormat,
    /// Total values in that dimension
    pub len: usize,
}

impl<T, U> SparseAxis<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    /// Generate a new `SparseAxis` in CSC format
    ///
    /// ### Params
    ///
    /// * `indices` - The indices of the values/non-zero positions
    /// * `data` - The values in the row/columns
    /// * `data_2` - An optional second data layer
    /// * `len` - Number of rows in this sparse column
    pub fn new_csc(indices: Vec<usize>, data: Vec<T>, data_2: Option<Vec<U>>, len: usize) -> Self {
        SparseAxis {
            indices,
            data,
            data_2,
            cs_type: CompressedSparseFormat::Csc,
            len,
        }
    }

    /// Get references to the indices and second layer
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, data_2)`
    pub fn get_indices_data_2(&self) -> (&[usize], &[U]) {
        let indices = &self.indices;
        let data_2 = self.data_2.as_ref().expect("target gene requires data_2");

        (indices, data_2)
    }
}

/// Structure to store compressed sparse data of either type
#[derive(Debug, Clone)]
pub struct CompressedSparseData2<T, U = T>
where
    T: Clone,
    U: Clone,
{
    /// The first data slot for this compressed sparse data format
    pub data: Vec<T>,
    /// The indices of the data points
    pub indices: Vec<usize>,
    /// The indptr of the data points
    pub indptr: Vec<usize>,
    /// Enum defining if the data is stored in CSC or CSR
    pub cs_type: CompressedSparseFormat,
    /// Optional second data slot for a different layer of the data (for
    /// example raw and normalised counts)
    pub data_2: Option<Vec<U>>,
    /// Shape of the data (rows, cols)
    pub shape: (usize, usize),
}

impl<T, U> CompressedSparseData2<T, U>
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
        transpose_sparse(self)
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
                let csc_version = transpose_sparse(self);
                CompressedSparseData2 {
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
                CompressedSparseData2 {
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
        CompressedSparseData2 {
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

////////////////////////
// Format conversions //
////////////////////////

/// Transpose a compressed sparse matrix (CSC→CSR or CSR→CSC).
///
/// This is the standard two-pass sparse transpose in O(nnz) time.
///
/// ### Params
///
/// * `sparse_data`: The input compressed sparse matrix to be transformed.
///
/// ### Returns
///
/// The transposed compressed sparse matrix.
pub fn transpose_sparse<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
) -> CompressedSparseData2<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let nnz = sparse_data.get_nnz();
    let (nrow, ncol) = sparse_data.shape();

    // the "minor" dimension is what becomes the new indptr axis.
    let (new_major, new_type) = match sparse_data.cs_type {
        CompressedSparseFormat::Csc => (nrow, CompressedSparseFormat::Csr),
        CompressedSparseFormat::Csr => (ncol, CompressedSparseFormat::Csc),
    };

    // first pass: count entries per new-major index
    let mut new_indptr = vec![0usize; new_major + 1];
    for &idx in &sparse_data.indices {
        new_indptr[idx + 1] += 1;
    }
    for i in 0..new_major {
        new_indptr[i + 1] += new_indptr[i];
    }

    // second pass: scatter data
    let mut new_data: Vec<T> = vec![T::default(); nnz];
    let mut new_indices: Vec<usize> = vec![0usize; nnz];
    let mut new_data2: Option<Vec<U>> =
        sparse_data.data_2.as_ref().map(|_| vec![U::default(); nnz]);
    unsafe {
        new_data.set_len(nnz);
        new_indices.set_len(nnz);
    }

    // Reuse new_indptr as the write cursor — we'll restore it afterwards.
    // Work on a mutable window so we don't need a separate `next` vec.
    // We iterate old major indices and scatter into new positions.
    let old_major_len = sparse_data.indptr.len() - 1;
    for major in 0..old_major_len {
        for idx in sparse_data.indptr[major]..sparse_data.indptr[major + 1] {
            let minor = sparse_data.indices[idx];
            let pos = new_indptr[minor];

            // SAFETY: pos < nnz guaranteed by the counting pass
            unsafe {
                *new_data.get_unchecked_mut(pos) = sparse_data.data[idx];
                *new_indices.get_unchecked_mut(pos) = major;
            }

            if let (Some(src), Some(dst)) = (&sparse_data.data_2, &mut new_data2) {
                unsafe {
                    *dst.get_unchecked_mut(pos) = src[idx];
                }
            }

            new_indptr[minor] += 1;
        }
    }

    // restore new_indptr: the scatter pass shifted every entry forward by its
    // count, so we shift the whole array right by one position.
    for i in (1..=new_major).rev() {
        new_indptr[i] = new_indptr[i - 1];
    }
    new_indptr[0] = 0;

    CompressedSparseData2 {
        data: new_data,
        indices: new_indices,
        indptr: new_indptr,
        cs_type: new_type,
        data_2: new_data2,
        shape: (nrow, ncol),
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
/// `CompressedSparseData2` in CSR format
pub fn coo_to_csr<T>(
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
    shape: (usize, usize),
) -> CompressedSparseData2<T>
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

    CompressedSparseData2::new_csr(&data, &indices, &indptr, None, shape)
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
) -> CompressedSparseData2<T>
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

    CompressedSparseData2::new_csr(&data, &indices, &indptr, None, shape)
}

///////////////////////
// Sparse operations //
///////////////////////

/// Add two CSR matrices together
///
/// ### Params
///
/// * `a` - Reference to the first CompressedSparseData2 (in CSR format!)
/// * `b` - Reference to the second CompressedSparseData2 (in CSR format!)
///
/// ### Returns
///
/// `CompressedSparseData2` with added values between the two.
pub fn sparse_add_csr<T>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> CompressedSparseData2<T>
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
/// * `a` - Reference to the first CompressedSparseData2 (in CSR format!)
/// * `scalar` - The scalar value to multiply with
///
/// ### Returns
///
/// `CompressedSparseData2` with the data multiplied by the scalar.
pub fn sparse_scalar_multiply_csr<T>(
    a: &CompressedSparseData2<T>,
    scalar: T,
) -> CompressedSparseData2<T>
where
    T: BixverseNumeric,
    <T as Mul>::Output: Send,
    Vec<T>: FromParallelIterator<<T as Mul>::Output>,
{
    let data: Vec<T> = a.data.par_iter().map(|&v| v * scalar).collect();
    CompressedSparseData2::new_csr(&data, &a.indices, &a.indptr, None, a.shape)
}

/// Sparse matrix subtraction
///
/// ### Params
///
/// * `a` - Reference to the first CompressedSparseData2 (in CSR format!)
/// * `b` - Reference to the second CompressedSparseData2 (in CSR format!)
///
/// ### Returns
///
/// The subtracted new matrix
pub fn sparse_subtract_csr<T>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> CompressedSparseData2<T>
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
/// * `a` - Reference to the first CompressedSparseData2 (in CSR format!)
/// * `b` - Reference to the second CompressedSparseData2 (in CSR format!)
///
/// ### Returns
///
/// The multiplied matrix.
pub fn sparse_multiply_elementwise_csr<T>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> CompressedSparseData2<T>
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
pub fn normalise_csr_columns_l1<T>(csr: &mut CompressedSparseData2<T>)
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
pub fn normalise_csr_rows_l1<T>(csr: &mut CompressedSparseData2<T>)
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
pub fn frobenius_norm<T>(mat: &CompressedSparseData2<T>) -> f32
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
pub fn eliminate_zeros_csr<T>(mat: CompressedSparseData2<T>) -> CompressedSparseData2<T>
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
pub fn csr_matvec<T>(mat: &CompressedSparseData2<T>, vec: &[T]) -> Vec<T>
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
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> CompressedSparseData2<T>
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

    CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (nrows, ncols))
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
    matrix: &CompressedSparseData2<T>,
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

        // full reorthogonalisation
        for k in 0..=j {
            let coeff = dot(&w, &v_matrix[k]);
            for i in 0..n {
                w[i] -= coeff * v_matrix[k][i];
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
/// `SvdResults` containing U (n×k), S (length k), and V (m×k)
pub fn sparse_svd_lanczos<T, U, F>(
    matrix: &CompressedSparseData2<T, U>,
    n_components: usize,
    seed: u64,
    use_second_layer: bool,
    col_means: Option<&[F]>,
    col_stds: Option<&[F]>,
) -> SvdResults<F>
where
    T: BixverseNumeric + SimdDistance + Into<F> + Clone,
    U: BixverseNumeric + Into<F> + Clone,
    F: BixverseFloat + SimdDistance + std::iter::Sum,
{
    let (n, m) = matrix.shape;
    let use_ata = n > m;
    let krylov_dim = if use_ata { m } else { n };
    let n_iter = (n_components * 2 + 10).max(n_components).min(krylov_dim);

    // keep both representations to make matvec operations fast
    let (csr, csc);
    let csr_owned;
    let csc_owned;

    match matrix.cs_type {
        CompressedSparseFormat::Csr => {
            csr = matrix;
            csc_owned = matrix.transform();
            csc = &csc_owned;
        }
        CompressedSparseFormat::Csc => {
            csc = matrix;
            csr_owned = matrix.transform();
            csr = &csr_owned;
        }
    };

    // helper to extract the right data layer and cast to F
    let extract_data = |mat: &CompressedSparseData2<T, U>| -> Vec<F> {
        if use_second_layer {
            mat.data_2
                .as_ref()
                .expect("data_2 is None but use_second_layer is true")
                .iter()
                .copied()
                .map(|v| v.into())
                .collect()
        } else {
            mat.data.iter().copied().map(|v| v.into()).collect()
        }
    };

    let data_csr_f = extract_data(csr);
    let data_csc_f = extract_data(csc);

    // matrix-vector product for A (using CSR)
    let matvec_a = |x: &[F], y: &mut [F]| {
        let x_scaled: Vec<F> = if let Some(sd) = col_stds {
            x.iter().enumerate().map(|(j, &v)| v / sd[j]).collect()
        } else {
            x.to_vec()
        };
        let mean_dot: F = if let Some(mu) = col_means {
            x_scaled.iter().enumerate().map(|(j, &v)| mu[j] * v).sum()
        } else {
            F::zero()
        };

        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = F::zero();
            for idx in csr.indptr[i]..csr.indptr[i + 1] {
                let j = csr.indices[idx];
                sum += data_csr_f[idx] * x_scaled[j];
            }
            if col_means.is_some() {
                sum -= mean_dot;
            }
            *yi = sum;
        });
    };

    // matrix-vector product for A^T (using CSC for memory contiguity)
    let matvec_at = |x: &[F], y: &mut [F]| {
        let x_sum: F = x.iter().copied().sum();

        y.par_iter_mut().enumerate().for_each(|(j, yj)| {
            let mut sum = F::zero();
            for idx in csc.indptr[j]..csc.indptr[j + 1] {
                let i = csc.indices[idx];
                sum += data_csc_f[idx] * x[i];
            }

            if let Some(mu) = col_means {
                sum -= mu[j] * x_sum;
            }
            if let Some(sd) = col_stds {
                sum /= sd[j];
            }
            *yj = sum;
        });
    };

    // select Gram matrix operator
    #[allow(clippy::type_complexity)]
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
    let mut v_matrix = Mat::<F>::zeros(krylov_dim, n_iter);
    let mut v = vec![F::zero(); krylov_dim];
    let mut v_old = vec![F::zero(); krylov_dim];
    let mut w = vec![F::zero(); krylov_dim];
    let mut w_faer = faer::Col::<F>::zeros(krylov_dim);

    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..krylov_dim {
        v[i] = F::from(rng.random::<f64>() - 0.5).unwrap();
    }
    normalise(&mut v);

    let mut alpha = vec![F::zero(); n_iter];
    let mut beta = vec![F::zero(); n_iter];

    for j in 0..n_iter {
        for i in 0..krylov_dim {
            v_matrix[(i, j)] = v[i];
        }

        matvec_gram(&v, &mut w);
        alpha[j] = dot(&w, &v);

        for i in 0..krylov_dim {
            w[i] -= alpha[j] * v[i];
            if j > 0 {
                w[i] -= beta[j - 1] * v_old[i];
            }
        }

        // Gram-Schmidt / Orthogonalisation
        // w -= Vj * (Vj^T * w)
        for i in 0..krylov_dim {
            w_faer[i] = w[i];
        }
        let vj = v_matrix.as_ref().subcols(0, j + 1);
        let coeffs = vj.transpose() * w_faer.as_ref();
        let proj = vj * coeffs.as_ref();
        for i in 0..krylov_dim {
            w[i] -= proj[i];
        }

        beta[j] = norm(&w);
        if beta[j] < F::from(1e-12).unwrap() {
            break;
        }

        v_old.copy_from_slice(&v);
        v.copy_from_slice(&w);
        normalise(&mut v);
    }

    // eigendecomposition and reconstruction
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

        // gram_evec = v_matrix * evecs.col(idx) via faer operator overloading
        let gram_col = &v_matrix * evecs.col(idx);
        let norm_val: F = gram_col.iter().map(|x| *x * *x).sum::<F>().sqrt();
        let gram_evec: Vec<F> = gram_col.iter().map(|x| *x / norm_val).collect();

        if use_ata {
            let mut u_vec = vec![F::zero(); n];
            matvec_a(&gram_evec, &mut u_vec);
            for x in &mut u_vec {
                *x /= sigma;
            }
            u_vecs.push(u_vec);
            v_vecs.push(gram_evec);
        } else {
            let mut v_vec = vec![F::zero(); m];
            matvec_at(&gram_evec, &mut v_vec);
            for x in &mut v_vec {
                *x /= sigma;
            }
            u_vecs.push(gram_evec);
            v_vecs.push(v_vec);
        }
    }

    let u = Mat::from_fn(n, singular_values.len(), |i, j| u_vecs[j][i]);
    let v = Mat::from_fn(m, singular_values.len(), |i, j| v_vecs[j][i]);

    SvdResults {
        u,
        s: singular_values,
        v,
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_parse_sparse_format() {
        assert!(parse_compressed_sparse_format("csr").unwrap().is_csr());
        assert!(parse_compressed_sparse_format("CSC").unwrap().is_csc());
        assert!(parse_compressed_sparse_format("dense").is_none());
    }

    #[test]
    fn test_from_dense_and_count_zeroes() {
        let mat: Mat<f64> = Mat::from_fn(3, 2, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        let (total_zeroes, row_zeroes, col_zeroes) = count_zeroes(&mat.as_ref());

        assert_eq!(total_zeroes, 4);
        assert_eq!(row_zeroes, vec![1, 1, 2]);
        assert_eq!(col_zeroes, vec![2, 2]);

        let csr = CompressedSparseData2::<f64, f64>::from_dense_matrix(
            mat.as_ref(),
            CompressedSparseFormat::Csr,
        );
        assert_eq!(csr.shape, (3, 2));
        assert_eq!(csr.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sparse_add_csr() {
        let shape = (2, 2);
        let a = CompressedSparseData2::<f64, f64>::new_csr(
            &[1.0, 2.0],
            &[0, 1],
            &[0, 1, 2],
            None,
            shape,
        );
        let b = CompressedSparseData2::<f64, f64>::new_csr(
            &[3.0, 4.0],
            &[1, 1],
            &[0, 1, 2],
            None,
            shape,
        );
        let c = sparse_add_csr(&a, &b);

        assert_eq!(c.data, vec![1.0, 3.0, 6.0]);
        assert_eq!(c.indices, vec![0, 1, 1]);
        assert_eq!(c.indptr, vec![0, 2, 3]);
    }

    #[test]
    fn test_csr_matvec() {
        let a = CompressedSparseData2::<f64, f64>::new_csr(
            &[1.0, 2.0, 3.0],
            &[0, 1, 1],
            &[0, 2, 3],
            None,
            (2, 2),
        );
        let vec = vec![2.0, 1.0];
        let result = csr_matvec(&a, &vec);
        assert_eq!(result, vec![4.0, 3.0]);
    }

    #[test]
    fn test_lanczos_eigenpairs_logic() {
        // Symmetric rank-1 matrix M = x * x^T
        // Let x = [1.0, 0.0, 2.0, 0.0]^T
        let data = vec![1.0, 2.0, 2.0, 4.0];
        let indices = vec![0, 2, 0, 2];
        let indptr = vec![0, 2, 2, 4, 4]; // Rows 1 and 3 are empty
        let shape = (4, 4);

        let csr = CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, shape);

        // Lanczos expects symmetric matrix, this one is symmetric
        let (evals, evecs) = compute_largest_eigenpairs_lanczos(&csr, 1, 42);

        // True top eigenvalue should be exactly sum(x_i^2) = 1.0 + 4.0 = 5.0
        assert!((evals[0] - 5.0).abs() < 1e-3);

        // Eigenvectors are returned transposed: evecs[point_idx][comp_idx]
        // So the first principal component is [evecs[0][0], evecs[1][0], evecs[2][0], evecs[3][0]]
        let x_norm = 5.0_f32.sqrt();
        let dot_x = (evecs[0][0] * 1.0 + evecs[2][0] * 2.0) / x_norm;

        assert!(dot_x.abs() > 0.999);
    }

    #[test]
    fn test_sparse_svd_lanczos_logic() {
        // Sparse rank-1 matrix A = x * y^T
        // x = [0.0, 2.0, 0.0, 4.0]^T
        // y = [1.0, 0.0, 0.5]^T
        let data = vec![2.0, 1.0, 4.0, 2.0];
        let indices = vec![0, 2, 0, 2];
        let indptr = vec![0, 0, 2, 2, 4];
        let shape = (4, 3);

        let csr = CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, shape);
        let no_params: Option<&[f64]> = None;

        let svd = sparse_svd_lanczos(&csr, 1, 42, false, no_params, no_params);

        // Test correlation with theoretical U
        let u_col = svd.u.col(0);
        let x_norm = 20.0_f64.sqrt();
        let dot_u = (u_col[1] * 2.0 + u_col[3] * 4.0) / x_norm;
        assert!(dot_u.abs() > 0.999);

        // Test correlation with theoretical V
        let v_col = svd.v.col(0);
        let y_norm = 1.25_f64.sqrt();
        let dot_v = (v_col[0] * 1.0 + v_col[2] * 0.5) / y_norm;
        assert!(dot_v.abs() > 0.999);
    }
}
