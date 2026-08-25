//! Sparse matrix formats, sparse operations and helpers to transform different
//! formats into each other.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef};
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::ops::{Add, AddAssign, Mul};

use crate::core::math::pca_svd::SvdResults;
use crate::core::math::vector_helpers::sum_sq_f64;
use crate::prelude::*;
use crate::utils::faer_parallelism;
use crate::utils::simd::{sum_squared_dev_widen_simd_f32, sum_widen_simd_f32};

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

///////////
// Enums //
///////////

/// Type to describe the CompressedSparseFormat
#[derive(Debug, Clone, Copy)]
pub enum CompressedSparseFormat {
    /// CSC-formatted data
    Csc,
    /// CSR-formatted data
    Csr,
}

/// Display implementation for [CompressedSparseFormat]
impl std::fmt::Display for CompressedSparseFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressedSparseFormat::Csc => write!(f, "CSC"),
            CompressedSparseFormat::Csr => write!(f, "CSR"),
        }
    }
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

////////////////
// SparseAxis //
////////////////

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
    pub fn get_indices_data_2(&self) -> Result<(&[usize], &[U]), BixverseErrors> {
        let indices = &self.indices;
        let data_2 = self
            .data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?;

        Ok((indices, data_2))
    }

    /// Construct a `SparseAxis` in CSC format from index and value vectors.
    ///
    /// Stores `indices` (converted to `usize`) in `self.indices`, leaves
    /// `data` empty, and places `values` into `data_2`. Useful for building
    /// single-target test fixtures and lightweight sparse columns where only
    /// the float layer is needed.
    ///
    /// ### Params
    ///
    /// * `indices` - Non-zero positions (must be convertible to `usize`).
    /// * `values` - Corresponding values, stored in `data_2`.
    ///
    /// ### Panics
    ///
    /// Panics if `indices` and `values` differ in length.
    pub fn from_vecs_to_csc(indices: Vec<T>, values: Vec<U>) -> Result<Self, BixverseErrors>
    where
        T: Into<usize>,
    {
        if indices.len() != values.len() {
            return Err(BixverseErrors::DimensionMisMatchSparse {
                indices_len: indices.len(),
                data_len: values.len(),
            });
        }

        let usize_indices: Vec<usize> = indices.into_iter().map(Into::into).collect();
        let len = usize_indices.last().map_or(0, |&i| i + 1);

        Ok(Self {
            indices: usize_indices,
            data: Vec::new(),
            data_2: Some(values),
            cs_type: CompressedSparseFormat::Csc,
            len,
        })
    }
}

///////////////////////////
// CompressedSparseData2 //
///////////////////////////

/// Structure to store compressed sparse data with two potential data layers.
/// Think for example in single cell raw counts and normalised counts.
#[derive(Debug, Clone)]
pub struct CompressedSparseData2<T, U = T>
where
    T: Clone,
    U: Clone,
{
    /// The first data slot for this compressed sparse data format
    pub data: Vec<T>,
    /// The indices of the data points
    pub indices: Vec<u32>,
    /// The indptr of the data points
    pub indptr: Vec<u32>,
    /// Enum defining if the data is stored in CSC or CSR, see
    /// [CompressedSparseFormat]
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
        indices: &[u32],
        indptr: &[u32],
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
        indices: &[u32],
        indptr: &[u32],
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

    /// Take ownership of buffers that were built for this matrix
    ///
    /// [CompressedSparseData2::new_csr] and its CSC sibling copy all three
    /// buffers, which doubles peak memory for any caller that assembled them
    /// itself and then throws the originals away. At a million cells by thirty
    /// neighbours that is 150 MB built and immediately cloned, so the `u8` edge
    /// layer sold as a byte per edge is two bytes per edge in transit.
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data, moved.
    /// * `indices` - The minor-axis indices, moved.
    /// * `indptr` - The major-axis pointers, moved.
    /// * `data2` - An optional second layer, moved.
    /// * `cs_type` - Which orientation the buffers describe.
    /// * `shape` - `(nrow, ncol)`.
    ///
    /// ### Returns
    ///
    /// The matrix, with no buffer copied.
    pub fn from_parts(
        data: Vec<T>,
        indices: Vec<u32>,
        indptr: Vec<u32>,
        data2: Option<Vec<U>>,
        cs_type: CompressedSparseFormat,
        shape: (usize, usize),
    ) -> Self {
        Self {
            data,
            indices,
            indptr,
            cs_type,
            data_2: data2,
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

    /// Transform from CSC to CSR or vice versa, keeping only one data layer.
    ///
    /// ### Params
    ///
    /// * `use_second_layer` - If `true`, the output keeps `data_2` and `data`
    ///   is empty. If `false`, the output keeps `data` and `data_2` is `None`.
    ///
    /// ### Returns
    ///
    /// The transformed/transposed version with only the requested layer.
    pub fn transform_single_layer(&self, use_second_layer: bool) -> Result<Self, BixverseErrors> {
        transpose_sparse_single_layer(self, use_second_layer)
    }

    /// Transpose and convert
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
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = Vec::new();

        match format {
            CompressedSparseFormat::Csr => {
                indptr.push(0);
                for i in 0..nrows {
                    for j in 0..ncols {
                        let val = mat[(i, j)];
                        if val != T::zero() {
                            data.push(val);
                            indices.push(j as u32);
                        }
                    }
                    indptr.push(data.len() as u32);
                }
            }
            CompressedSparseFormat::Csc => {
                indptr.push(0);
                for j in 0..ncols {
                    for i in 0..nrows {
                        let val = mat[(i, j)];
                        if val != T::zero() {
                            data.push(val);
                            indices.push(i as u32);
                        }
                    }
                    indptr.push(data.len() as u32);
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
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = Vec::new();

        match format {
            CompressedSparseFormat::Csr => {
                indptr.push(0);
                for row in 0..n {
                    for col in 0..n {
                        let value = get_value(row, col);
                        if value != T::zero() {
                            data.push(value);
                            indices.push(col as u32);
                        }
                    }
                    indptr.push(data.len() as u32);
                }
            }
            CompressedSparseFormat::Csc => {
                indptr.push(0);
                for col in 0..n {
                    for row in 0..n {
                        let value = get_value(row, col);
                        if value != T::zero() {
                            data.push(value);
                            indices.push(row as u32);
                        }
                    }
                    indptr.push(data.len() as u32);
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

    /// Return the number of rows
    ///
    /// ### Returns
    ///
    /// Number of rows
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Return the number of columns
    ///
    /// ### Returns
    ///
    /// Number of columns
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Returns the NNZ
    ///
    /// ### Returns
    ///
    /// The number of NNZ
    pub fn get_nnz(&self) -> usize {
        self.indices.len()
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

    /////////////
    // Slicing //
    /////////////

    /// Gather along the major axis (rows for CSR, cols for CSC).
    ///
    /// Arbitrary order allowed; duplicates and out-of-range indices rejected.
    ///
    /// ### Params
    ///
    /// * `keep` - The indices on the major dimensions to keep
    ///
    /// ### Returns
    ///
    /// Self sliced along the major axis
    fn slice_major(&self, keep: &[usize]) -> Result<Self, BixverseErrors> {
        let major = self.major_dim();
        let mut seen = vec![false; major];

        let mut indptr = Vec::with_capacity(keep.len() + 1);
        indptr.push(0u32);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut data_2 = self.data_2.as_ref().map(|_| Vec::new());

        for &m in keep {
            if m >= major {
                return Err(BixverseErrors::SliceIndexOutOfBounds {
                    index: m,
                    len: major,
                });
            }
            if seen[m] {
                return Err(BixverseErrors::DuplicateSliceIndex(m));
            }
            seen[m] = true;

            let s = self.indptr[m] as usize;
            let e = self.indptr[m + 1] as usize;
            indices.extend_from_slice(&self.indices[s..e]);
            data.extend_from_slice(&self.data[s..e]);
            if let (Some(dst), Some(src)) = (data_2.as_mut(), self.data_2.as_ref()) {
                dst.extend_from_slice(&src[s..e]);
            }
            indptr.push(indices.len() as u32);
        }

        let shape = match self.cs_type {
            CompressedSparseFormat::Csr => (keep.len(), self.shape.1),
            CompressedSparseFormat::Csc => (self.shape.0, keep.len()),
        };

        let out = Self {
            data,
            indices,
            indptr,
            cs_type: self.cs_type,
            data_2,
            shape,
        };
        out.assert_invariants();
        Ok(out)
    }

    /// Filter + remap the minor axis (cols for CSR, rows for CSC).
    ///
    /// Arbitrary order allowed; duplicates and out-of-range indices rejected.
    /// Minor indices within each major run are kept ascending in the new space.
    ///
    /// ### Params
    ///
    /// * `keep` - The indices on the major dimensions to keep
    ///
    /// ### Returns
    ///
    /// Self sliced along the minor axis (with remapping)
    fn slice_minor(&self, keep: &[usize]) -> Result<Self, BixverseErrors> {
        let minor = self.minor_dim();
        let mut map: Vec<Option<usize>> = vec![None; minor];
        for (new_idx, &old) in keep.iter().enumerate() {
            if old >= minor {
                return Err(BixverseErrors::SliceIndexOutOfBounds {
                    index: old,
                    len: minor,
                });
            }
            if map[old].is_some() {
                return Err(BixverseErrors::DuplicateSliceIndex(old));
            }
            map[old] = Some(new_idx);
        }

        let major = self.major_dim();
        let mut indptr = Vec::with_capacity(major + 1);
        indptr.push(0u32);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut data_2 = self.data_2.as_ref().map(|_| Vec::new());

        let mut kept: Vec<(usize, usize)> = Vec::new(); // (new_minor, src_pos)
        for m in 0..major {
            let s = self.indptr[m];
            let e = self.indptr[m + 1];
            kept.clear();
            for p in s..e {
                if let Some(nm) = map[self.indices[p as usize] as usize] {
                    kept.push((nm, p as usize));
                }
            }
            kept.sort_unstable_by_key(|&(nm, _)| nm);
            for &(nm, p) in &kept {
                indices.push(nm as u32);
                data.push(self.data[p]);
                if let (Some(dst), Some(src)) = (data_2.as_mut(), self.data_2.as_ref()) {
                    dst.push(src[p]);
                }
            }
            indptr.push(indices.len() as u32);
        }

        let shape = match self.cs_type {
            CompressedSparseFormat::Csr => (self.shape.0, keep.len()),
            CompressedSparseFormat::Csc => (keep.len(), self.shape.1),
        };

        let out = Self {
            data,
            indices,
            indptr,
            cs_type: self.cs_type,
            data_2,
            shape,
        };
        out.assert_invariants();
        Ok(out)
    }

    /// Slice the rows of the matrix
    ///
    /// ### Params
    ///
    /// * `rows` - The row indices to keep
    ///
    /// ### Returns
    ///
    /// Self with the rows to keep
    pub fn slice_rows(&self, rows: &[usize]) -> Result<Self, BixverseErrors> {
        match self.cs_type {
            CompressedSparseFormat::Csr => self.slice_major(rows),
            CompressedSparseFormat::Csc => self.slice_minor(rows),
        }
    }

    /// Slice the columns of the matrix
    ///
    /// ### Params
    ///
    /// * `cols` - The column indices to keep
    ///
    /// ### Returns
    ///
    /// Self with the cols to keep
    pub fn slice_cols(&self, cols: &[usize]) -> Result<Self, BixverseErrors> {
        match self.cs_type {
            CompressedSparseFormat::Csr => self.slice_minor(cols),
            CompressedSparseFormat::Csc => self.slice_major(cols),
        }
    }

    /// Slice across rows and columns
    ///
    /// ### Params
    ///
    /// * `rows` - The row indices to keep
    /// * `cols` - The column indices to keep
    ///
    /// ### Returns
    ///
    /// Self with the rows and columns to keep
    pub fn slice(&self, rows: &[usize], cols: &[usize]) -> Result<Self, BixverseErrors> {
        self.slice_rows(rows)?.slice_cols(cols)
    }

    /////////////
    // Helpers //
    /////////////

    /// Return the major dimensions
    ///
    /// ### Returns
    ///
    /// Major dimension
    fn major_dim(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Returns the minor dimension
    ///
    /// ### Returns
    ///
    /// Minor dimension
    fn minor_dim(&self) -> usize {
        match self.cs_type {
            CompressedSparseFormat::Csr => self.shape.1,
            CompressedSparseFormat::Csc => self.shape.0,
        }
    }

    /// Check every invariant that indexing this matrix relies on.
    ///
    /// [`Self::assert_invariants`] is a `debug_assert` helper and compiles out
    /// in release, and it never checks the indices against the minor axis.
    /// This is the release-mode counterpart, for input that crossed an FFI
    /// boundary: the fields are public, so a caller can hand over an `indptr`
    /// that disagrees with `shape` or an index past the minor axis, and any
    /// consumer that scatters into a dense buffer sized from `shape` would
    /// then panic inside a worker thread or, worse, silently process fewer
    /// major runs than the shape declares.
    ///
    /// A layer that is empty is treated as absent rather than as a length
    /// mismatch, since [`Self::transform_single_layer`] deliberately leaves the
    /// unused layer empty.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the first invariant that failed:
    /// [`BixverseErrors::SparseIndptrInvalid`] for an `indptr` that disagrees
    /// with `shape` or with the number of stored values,
    /// [`BixverseErrors::DimensionMisMatchSparse`] for a populated layer whose
    /// length does not match `indices`, and
    /// [`BixverseErrors::SliceIndexOutOfBounds`] for an index past the minor
    /// axis.
    pub fn validate(&self) -> Result<(), BixverseErrors> {
        let expected_major = match self.cs_type {
            CompressedSparseFormat::Csr => self.shape.0,
            CompressedSparseFormat::Csc => self.shape.1,
        };

        if self.indptr.len() != expected_major + 1 {
            return Err(BixverseErrors::SparseIndptrInvalid {
                detail: "length against the declared major axis",
                expected: expected_major + 1,
                got: self.indptr.len(),
            });
        }

        // `indptr.len()` is now at least 1, so `last` cannot be `None`
        let nnz = *self.indptr.last().expect("indptr length checked above") as usize;
        if nnz != self.indices.len() {
            return Err(BixverseErrors::SparseIndptrInvalid {
                detail: "final offset against the stored indices",
                expected: self.indices.len(),
                got: nnz,
            });
        }

        if !self.indptr.windows(2).all(|w| w[0] <= w[1]) {
            return Err(BixverseErrors::SparseIndptrInvalid {
                detail: "offsets must be non-decreasing",
                expected: nnz,
                got: nnz,
            });
        }

        if !self.data.is_empty() && self.data.len() != self.indices.len() {
            return Err(BixverseErrors::DimensionMisMatchSparse {
                indices_len: self.indices.len(),
                data_len: self.data.len(),
            });
        }

        if let Some(data_2) = &self.data_2
            && !data_2.is_empty()
            && data_2.len() != self.indices.len()
        {
            return Err(BixverseErrors::DimensionMisMatchSparse {
                indices_len: self.indices.len(),
                data_len: data_2.len(),
            });
        }

        let minor = self.minor_dim();
        if let Some(&bad) = self
            .indices
            .par_iter()
            .find_any(|&&idx| idx as usize >= minor)
        {
            return Err(BixverseErrors::SliceIndexOutOfBounds {
                index: bad as usize,
                len: minor,
            });
        }

        Ok(())
    }

    /// Assertion helper for invariants
    pub fn assert_invariants(&self) {
        let expected_major = match self.cs_type {
            CompressedSparseFormat::Csr => self.shape.0,
            CompressedSparseFormat::Csc => self.shape.1,
        };
        debug_assert_eq!(self.indices.len(), self.data.len());
        debug_assert_eq!(self.indptr.len(), expected_major + 1);
        debug_assert_eq!(*self.indptr.last().unwrap() as usize, self.data.len());
        if let Some(d2) = &self.data_2 {
            debug_assert_eq!(d2.len(), self.data.len());
        }
    }
}

////////////////////////
// Format conversions //
////////////////////////

/// Destination pointer handed to every worker of the parallel scatter.
///
/// The transpose partitions the destination by `(chunk, minor)`: the counting
/// pass gives every pair its own half-open run of slots, and a worker only ever
/// advances the cursors of its own chunk. No two workers therefore address the
/// same slot, which is what makes the shared `*mut` sound.
struct ScatterPtr<T>(*mut T);

// SAFETY: see the type doc. Writes go through disjoint per-chunk cursor runs.
unsafe impl<T: Send> Send for ScatterPtr<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for ScatterPtr<T> {}

impl<T> ScatterPtr<T> {
    /// Write one value into the destination.
    ///
    /// ### Params
    ///
    /// * `pos` - Slot to write
    /// * `value` - Value to store
    ///
    /// # Safety
    ///
    /// `pos` must be in bounds of the buffer this was built from, and no other
    /// worker may write the same slot.
    #[inline(always)]
    unsafe fn write(&self, pos: usize, value: T) {
        unsafe { *self.0.add(pos) = value }
    }
}

/// Below this many stored values the transpose stays single-threaded.
///
/// The parallel path pays for `n_threads * new_major` counters plus two sweeps
/// over them, which only earns its keep once the scatter itself is large enough
/// to leave cache. A megabyte of stored values is comfortably past that point
/// on every machine tested and well short of it for the small graphs and toy
/// matrices most callers hand over.
const PARALLEL_TRANSPOSE_MIN_NNZ: usize = 1 << 20;

/// Per-chunk write cursors and the transposed `indptr`.
///
/// Runs the counting phase of the transpose: one histogram per chunk of the old
/// major axis, folded into the new `indptr` and then rewritten in place as the
/// absolute slot each `(chunk, minor)` pair starts at.
///
/// ### Params
///
/// * `sparse_data` - The matrix being transposed
/// * `chunks` - Half-open ranges partitioning the old major axis
/// * `new_major` - Length of the new major axis
///
/// ### Returns
///
/// `(new_indptr, cursors)`, where `cursors` is `chunks.len() * new_major` long
/// and laid out chunk-major.
fn transpose_cursors<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
    chunks: &[(usize, usize)],
    new_major: usize,
) -> (Vec<u32>, Vec<u32>)
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let mut cursors = vec![0u32; chunks.len() * new_major];

    chunks
        .par_iter()
        .zip(cursors.par_chunks_mut(new_major.max(1)))
        .for_each(|(&(start, end), counts)| {
            let lo = sparse_data.indptr[start] as usize;
            let hi = sparse_data.indptr[end] as usize;
            for &minor in &sparse_data.indices[lo..hi] {
                counts[minor as usize] += 1;
            }
        });

    // column sums of the histograms give the entries per new-major index
    let mut new_indptr = vec![0u32; new_major + 1];
    for counts in cursors.chunks(new_major.max(1)) {
        for (total, &count) in new_indptr[1..].iter_mut().zip(counts.iter()) {
            *total += count;
        }
    }
    for i in 0..new_major {
        new_indptr[i + 1] += new_indptr[i];
    }

    // rewrite the counts as absolute starts. Both the cursor row and the
    // running offsets are swept contiguously, so this stays cache-friendly
    // even when the new major axis is hundreds of thousands long.
    let mut running: Vec<u32> = new_indptr[..new_major].to_vec();
    for counts in cursors.chunks_mut(new_major.max(1)) {
        for (count, offset) in counts.iter_mut().zip(running.iter_mut()) {
            let start = *offset;
            *offset += *count;
            *count = start;
        }
    }

    (new_indptr, cursors)
}

/// Scatter the stored values into their transposed positions, in parallel.
///
/// ### Params
///
/// * `sparse_data` - The matrix being transposed
/// * `chunks` - Half-open ranges partitioning the old major axis
/// * `cursors` - Per-chunk write cursors from [`transpose_cursors`], consumed
/// * `new_major` - Length of the new major axis
/// * `out_indices` - Destination for the new minor indices
/// * `out_data` - Optional `(source, destination)` for the counts layer
/// * `out_data2` - Optional `(source, destination)` for the second layer
fn transpose_scatter<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
    chunks: &[(usize, usize)],
    cursors: &mut [u32],
    new_major: usize,
    out_indices: &mut [u32],
    out_data: Option<(&[T], &mut [T])>,
    out_data2: Option<(&[U], &mut [U])>,
) where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let indices_ptr = ScatterPtr(out_indices.as_mut_ptr());
    let (data_src, data_ptr) = match out_data {
        Some((src, dst)) => (Some(src), Some(ScatterPtr(dst.as_mut_ptr()))),
        None => (None, None),
    };
    let (data2_src, data2_ptr) = match out_data2 {
        Some((src, dst)) => (Some(src), Some(ScatterPtr(dst.as_mut_ptr()))),
        None => (None, None),
    };

    chunks
        .par_iter()
        .zip(cursors.par_chunks_mut(new_major.max(1)))
        .for_each(|(&(start, end), cursor)| {
            for major in start..end {
                let lo = sparse_data.indptr[major] as usize;
                let hi = sparse_data.indptr[major + 1] as usize;

                for idx in lo..hi {
                    let minor = sparse_data.indices[idx] as usize;
                    let pos = cursor[minor] as usize;
                    cursor[minor] += 1;

                    // SAFETY: `pos` sits in the run this chunk owns for
                    // `minor`, which no other worker touches, and every run is
                    // inside `0..nnz` by the counting pass.
                    unsafe {
                        indices_ptr.write(pos, major as u32);
                        if let (Some(src), Some(dst)) = (data_src, &data_ptr) {
                            dst.write(pos, src[idx]);
                        }
                        if let (Some(src), Some(dst)) = (data2_src, &data2_ptr) {
                            dst.write(pos, src[idx]);
                        }
                    }
                }
            }
        });
}

/// Re-express a compressed sparse matrix in the other format (CSC→CSR or
/// CSR→CSC).
///
/// Despite the name this does **not** transpose: `shape` is carried over
/// unchanged and only `cs_type` flips, so the result addresses the same
/// elements by the same `(row, col)` pair. Reach for
/// [`CompressedSparseData2::transpose_and_convert`] when the shape is meant to
/// swap.
///
/// A counting sort in O(nnz) time. Above `PARALLEL_TRANSPOSE_MIN_NNZ` the
/// count and the scatter both fan out over rayon: the scatter is a random-write
/// pass over `nnz`-sized destinations, so leaving it single-threaded makes it
/// the slowest step of anything that converts on the way in.
///
/// ### Params
///
/// * `sparse_data`: The input compressed sparse matrix to be converted.
///
/// ### Returns
///
/// The same matrix in the other compressed sparse format, with `shape`
/// unchanged and the indices of each new major run ascending.
pub fn transpose_sparse<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
) -> CompressedSparseData2<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    transpose_sparse_chunked(sparse_data, &major_axis_chunks(sparse_data))
}

/// Half-open ranges the transpose should split the old major axis into.
///
/// One range keeps the whole thing single-threaded, which is what small inputs
/// want. Split out so the tests can force a multi-chunk split without building
/// a matrix past [`PARALLEL_TRANSPOSE_MIN_NNZ`].
///
/// ### Params
///
/// * `sparse_data` - The matrix being transposed
///
/// ### Returns
///
/// The chunks covering the old major axis.
fn major_axis_chunks<T, U>(sparse_data: &CompressedSparseData2<T, U>) -> Vec<(usize, usize)>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let old_major_len = sparse_data.indptr.len() - 1;

    if sparse_data.get_nnz() >= PARALLEL_TRANSPOSE_MIN_NNZ {
        thread_chunks(old_major_len)
    } else {
        vec![(0, old_major_len)]
    }
}

/// Transpose a compressed sparse matrix over a given chunking of the old major
/// axis.
///
/// ### Params
///
/// * `sparse_data` - The input compressed sparse matrix to be transformed
/// * `chunks` - Half-open ranges partitioning the old major axis
///
/// ### Returns
///
/// The transposed compressed sparse matrix.
fn transpose_sparse_chunked<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
    chunks: &[(usize, usize)],
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

    let mut new_data: Vec<T> = vec![T::default(); nnz];
    let mut new_indices: Vec<u32> = vec![0u32; nnz];
    let mut new_data2: Option<Vec<U>> =
        sparse_data.data_2.as_ref().map(|_| vec![U::default(); nnz]);

    let (new_indptr, mut cursors) = transpose_cursors(sparse_data, chunks, new_major);

    let data_pair = Some((sparse_data.data.as_slice(), new_data.as_mut_slice()));
    let data2_pair = match (&sparse_data.data_2, &mut new_data2) {
        (Some(src), Some(dst)) => Some((src.as_slice(), dst.as_mut_slice())),
        _ => None,
    };

    transpose_scatter(
        sparse_data,
        chunks,
        &mut cursors,
        new_major,
        &mut new_indices,
        data_pair,
        data2_pair,
    );

    CompressedSparseData2 {
        data: new_data,
        indices: new_indices,
        indptr: new_indptr,
        cs_type: new_type,
        data_2: new_data2,
        shape: (nrow, ncol),
    }
}

/// Transpose a compressed sparse matrix (CSC→CSR or CSR→CSC), keeping only
/// one of the two data layers.
///
/// Useful when the caller knows the other layer is dead weight (e.g. raw
/// counts when only normalised counts are consumed downstream). Skips the
/// scatter for the unused layer entirely.
///
/// ### Params
///
/// * `sparse_data` - The input compressed sparse matrix to be transposed.
/// * `use_second_layer` - If `true`, only `data_2` is transposed and the
///   output's `data` is empty. If `false`, only `data` is transposed and the
///   output's `data_2` is `None`.
///
/// ### Returns
///
/// The transposed compressed sparse matrix with only the requested layer
/// populated.
pub fn transpose_sparse_single_layer<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
    use_second_layer: bool,
) -> Result<CompressedSparseData2<T, U>, BixverseErrors>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let nnz = sparse_data.get_nnz();
    let (nrow, ncol) = sparse_data.shape();

    let (new_major, new_type) = match sparse_data.cs_type {
        CompressedSparseFormat::Csc => (nrow, CompressedSparseFormat::Csr),
        CompressedSparseFormat::Csr => (ncol, CompressedSparseFormat::Csc),
    };

    let mut new_indices: Vec<u32> = vec![0u32; nnz];

    // allocate only for the kept layer
    let (mut new_data, mut new_data2): (Vec<T>, Option<Vec<U>>) = if use_second_layer {
        (Vec::new(), Some(vec![U::default(); nnz]))
    } else {
        (vec![T::default(); nnz], None)
    };

    let chunks = major_axis_chunks(sparse_data);
    let (new_indptr, mut cursors) = transpose_cursors(sparse_data, &chunks, new_major);

    let (data_pair, data2_pair) = if use_second_layer {
        let src = sparse_data
            .data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?
            .as_slice();
        let dst = new_data2.as_mut().unwrap().as_mut_slice();
        (None, Some((src, dst)))
    } else {
        (
            Some((sparse_data.data.as_slice(), new_data.as_mut_slice())),
            None,
        )
    };

    transpose_scatter(
        sparse_data,
        &chunks,
        &mut cursors,
        new_major,
        &mut new_indices,
        data_pair,
        data2_pair,
    );

    Ok(CompressedSparseData2 {
        data: new_data,
        indices: new_indices,
        indptr: new_indptr,
        cs_type: new_type,
        data_2: new_data2,
        shape: (nrow, ncol),
    })
}

/// Check that a matrix is a structurally sound square CSR and return its order
///
/// Callers that index `partitions[j]` or `parent[j]` from a stored column index
/// rely on every invariant here, and a violated one is otherwise silent: a
/// non-monotonic `indptr` gives an empty Rust range, so the row is skipped and
/// the caller returns a plausible answer computed on a subset of the graph. An
/// out-of-range column index panics through whatever boundary the caller sits
/// behind instead of erroring.
///
/// ### Params
///
/// * `graph` - The adjacency to validate.
///
/// ### Returns
///
/// The node count, or an error naming the violated invariant.
pub fn validate_square_csr<T>(graph: &CompressedSparseData2<T>) -> Result<usize, BixverseErrors>
where
    T: Clone,
{
    if !graph.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    let (rows, cols) = graph.shape;
    if rows != cols {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (rows, rows),
            got: (rows, cols),
        });
    }

    if graph.indptr.len() != rows + 1 {
        return Err(BixverseErrors::MalformedCsr(
            "indptr length must be the row count plus one",
        ));
    }
    if graph.indices.len() != graph.data.len() {
        return Err(BixverseErrors::MalformedCsr(
            "indices and data must have the same length",
        ));
    }
    if graph.indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(BixverseErrors::MalformedCsr(
            "indptr must be non-decreasing",
        ));
    }
    // `indptr` is non-empty by the length check above.
    if graph.indptr[rows] as usize != graph.indices.len() {
        return Err(BixverseErrors::MalformedCsr(
            "the last indptr entry must equal the number of stored values",
        ));
    }
    if graph.indices.iter().any(|&j| j as usize >= rows) {
        return Err(BixverseErrors::MalformedCsr(
            "a column index sits outside the matrix",
        ));
    }

    Ok(rows)
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
    rows: &[u32],
    cols: &[u32],
    vals: &[T],
    shape: (usize, usize),
) -> CompressedSparseData2<T>
where
    T: BixverseNumeric,
{
    let n_rows = shape.0;

    // sort by (row, col) and merge duplicates
    let mut entries: Vec<(u32, u32, T)> = rows
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
    let mut indptr = vec![0u32; n_rows + 1];

    for &(row, col, val) in &merged_entries {
        data.push(val);
        indices.push(col);
        indptr[(row + 1) as usize] += 1;
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
    rows: &[u32],
    cols: &[u32],
    vals: &[T],
    shape: (usize, usize),
) -> CompressedSparseData2<T>
where
    T: BixverseNumeric,
{
    let n_rows = shape.0;
    let nnz = rows.len();

    let mut data = Vec::with_capacity(nnz);
    let mut indices: Vec<u32> = Vec::with_capacity(nnz);
    let mut indptr = vec![0u32; n_rows + 1];

    // unsafe to squeeze out performance...
    unsafe {
        data.set_len(nnz);
        indices.set_len(nnz);

        let data_ptr: *mut T = data.as_mut_ptr();
        let indices_ptr: *mut u32 = indices.as_mut_ptr();
        let indptr_ptr: *mut u32 = indptr.as_mut_ptr();

        for i in 0..nnz {
            *data_ptr.add(i) = *vals.get_unchecked(i);
            *indices_ptr.add(i) = *cols.get_unchecked(i);
            let row = *rows.get_unchecked(i) as usize;
            *indptr_ptr.add(row + 1) += 1;
        }

        for i in 0..n_rows {
            *indptr_ptr.add(i + 1) += *indptr_ptr.add(i);
        }
    }

    CompressedSparseData2::new_csr(&data, &indices, &indptr, None, shape)
}

/// Scatter a sorted undirected edge list into a symmetric CSR adjacency
///
/// Rows come out ascending without a per-row sort or a per-row allocation. The
/// trick is the scatter order: the `hi` endpoint of every edge is written
/// first, filling each row `r` with its `lo < r` partners in ascending order,
/// and the `lo` endpoint second, appending its `hi > r` partners also
/// ascending. Both halves are ascending because the input is sorted by `(lo,
/// hi)`, and the first half sits entirely below the second because `lo < hi`.
///
/// Unlike [coo_to_csr] this neither sorts nor merges duplicates, so the caller
/// owns both invariants. Violating them produces a *corrupt* CSR rather than a
/// panic: unsorted input or `lo > hi` gives descending column indices,
/// `lo == hi` duplicates a diagonal entry, and a repeated pair doubles the nnz.
/// Every other consumer in the crate assumes ascending deduplicated rows, so the
/// invariants are `debug_assert!`ed here and the function stays crate-private.
///
/// ### Params
///
/// * `n` - Node count; the output is `n` square.
/// * `edges` - Deduplicated `(lo, hi, value)` triples with `lo < hi < n`, sorted
///   strictly ascending by `(lo, hi)`.
///
/// ### Returns
///
/// `CompressedSparseData2` in CSR format with both directions of every edge
/// stored at the same value.
pub(crate) fn undirected_edges_to_csr<T>(
    n: usize,
    edges: &[(u32, u32, T)],
) -> CompressedSparseData2<T>
where
    T: BixverseNumeric,
{
    debug_assert!(
        edges
            .iter()
            .all(|&(lo, hi, _)| lo < hi && (hi as usize) < n),
        "undirected_edges_to_csr wants lo < hi < n"
    );
    debug_assert!(
        edges
            .windows(2)
            .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)),
        "undirected_edges_to_csr wants strictly ascending, deduplicated (lo, hi)"
    );

    let mut indptr = vec![0u32; n + 1];
    for &(lo, hi, _) in edges {
        indptr[lo as usize + 1] += 1;
        indptr[hi as usize + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    let nnz = 2 * edges.len();
    let mut indices = vec![0u32; nnz];
    let mut data = vec![T::default(); nnz];
    let mut cursor: Vec<u32> = indptr[..n].to_vec();

    for &(lo, hi, v) in edges {
        let pos = cursor[hi as usize] as usize;
        indices[pos] = lo;
        data[pos] = v;
        cursor[hi as usize] += 1;
    }
    for &(lo, hi, v) in edges {
        let pos = cursor[lo as usize] as usize;
        indices[pos] = hi;
        data[pos] = v;
        cursor[lo as usize] += 1;
    }

    CompressedSparseData2::from_parts(
        data,
        indices,
        indptr,
        None,
        CompressedSparseFormat::Csr,
        (n, n),
    )
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
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseNumeric + Into<f64> + Add<Output = T>,
{
    if !a.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if !b.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if a.shape != b.shape {
        return Err(BixverseErrors::ShapeMismatchSparse);
    }

    const EPSILON: f32 = 1e-9;
    let n_rows = a.shape.0;

    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n_rows {
        let a_start = a.indptr[i] as usize;
        let a_end = a.indptr[i + 1] as usize;
        let b_start = b.indptr[i] as usize;
        let b_end = b.indptr[i + 1] as usize;

        let mut a_idx = a_start;
        let mut b_idx = b_start;

        while a_idx < a_end || b_idx < b_end {
            if a_idx < a_end && (b_idx >= b_end || a.indices[a_idx] < b.indices[b_idx]) {
                rows.push(i as u32);
                cols.push(a.indices[a_idx]);
                vals.push(a.data[a_idx]);
                a_idx += 1;
            } else if b_idx < b_end && (a_idx >= a_end || b.indices[b_idx] < a.indices[a_idx]) {
                rows.push(i as u32);
                cols.push(b.indices[b_idx]);
                vals.push(b.data[b_idx]);
                b_idx += 1;
            } else {
                let val = a.data[a_idx] + b.data[b_idx];
                if val.into().abs() > EPSILON as f64 {
                    rows.push(i as u32);
                    cols.push(a.indices[a_idx]);
                    vals.push(val);
                }
                a_idx += 1;
                b_idx += 1;
            }
        }
    }

    Ok(coo_to_csr_presorted(&rows, &cols, &vals, a.shape))
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
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseNumeric + Into<f64>,
{
    if !a.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if !b.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if a.shape != b.shape {
        return Err(BixverseErrors::ShapeMismatchSparse);
    }

    const EPSILON: f32 = 1e-9;
    let n_rows = a.shape.0;

    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n_rows {
        let a_start = a.indptr[i] as usize;
        let a_end = a.indptr[i + 1] as usize;
        let b_start = b.indptr[i] as usize;
        let b_end = b.indptr[i + 1] as usize;

        let mut a_idx = a_start;
        let mut b_idx = b_start;

        while a_idx < a_end || b_idx < b_end {
            if a_idx < a_end && (b_idx >= b_end || a.indices[a_idx] < b.indices[b_idx]) {
                rows.push(i as u32);
                cols.push(a.indices[a_idx]);
                vals.push(a.data[a_idx]);
                a_idx += 1;
            } else if b_idx < b_end && (a_idx >= a_end || b.indices[b_idx] < a.indices[a_idx]) {
                rows.push(i as u32);
                cols.push(b.indices[b_idx]);
                vals.push(T::default() - b.data[b_idx]);
                b_idx += 1;
            } else {
                let val = a.data[a_idx] - b.data[b_idx];
                if val.into().abs() > EPSILON as f64 {
                    rows.push(i as u32);
                    cols.push(a.indices[a_idx]);
                    vals.push(val);
                }
                a_idx += 1;
                b_idx += 1;
            }
        }
    }

    Ok(coo_to_csr_presorted(&rows, &cols, &vals, a.shape))
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
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseNumeric,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    if !a.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if !b.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if a.shape != b.shape {
        return Err(BixverseErrors::ShapeMismatchSparse);
    }

    let n_rows = a.shape.0;
    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n_rows {
        let a_start = a.indptr[i] as usize;
        let a_end = a.indptr[i + 1] as usize;
        let b_start = b.indptr[i] as usize;
        let b_end = b.indptr[i + 1] as usize;
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
                        rows.push(i as u32);
                        cols.push(a.indices[a_idx]);
                        vals.push(val);
                    }
                    a_idx += 1;
                    b_idx += 1;
                }
            }
        }
    }

    Ok(coo_to_csr(&rows, &cols, &vals, a.shape))
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
        col_sums[col as usize] += csr.data[idx]
    }

    for (idx, &col) in csr.indices.iter().enumerate() {
        let sum = col_sums[col as usize];
        if sum.into() > 1e-15 {
            csr.data[idx] /= sum;
        }
    }
}

/// Normalises the rows of a CSR matrix to a sum of 1 (L1 norm)
///
/// Turns an affinity matrix `K` into the row-stochastic diffusion operator
/// `D^-1 K` in place.
///
/// A row summing to zero is an error rather than a no-op. The reference
/// implementations this backs (Palantir's `diffusion_maps_from_kernel`, MAGIC)
/// guard with `D[D != 0] = 1 / D[D != 0]` and leave the row at zero, but on a
/// kNN-derived kernel every node has `k` out-edges, so a zero row means the
/// weights underflowed and everything downstream of the operator is garbage.
/// Failing loudly beats propagating a silently absorbing state.
///
/// ### Params
///
/// * `csr` - Mutable reference to the CSR matrix (modified in-place)
///
/// ### Returns
///
/// `Ok(())`, or [BixverseErrors::SparseMatrixMustBeCsr] if the matrix is in
/// CSC, or [BixverseErrors::SparseMatrixIsolatedRow] naming the first row that
/// sums to zero.
pub fn normalise_csr_rows_l1<T>(csr: &mut CompressedSparseData2<T>) -> Result<(), BixverseErrors>
where
    T: BixverseNumeric + Into<f64>,
    T: std::iter::Sum<T>,
{
    if !csr.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }

    let nrows = csr.shape.0;

    for i in 0..nrows {
        let start = csr.indptr[i] as usize;
        let end = csr.indptr[i + 1] as usize;
        let row_data_slice = &mut csr.data[start..end];

        let row_sum: T = row_data_slice.iter().copied().sum();

        if row_sum.into() <= 1e-15 {
            return Err(BixverseErrors::SparseMatrixIsolatedRow {
                row: i,
                row_sum: row_sum.into(),
            });
        }

        for val in row_data_slice.iter_mut() {
            *val /= row_sum;
        }
    }

    Ok(())
}

/// Compute Frobenius norm of sparse matrix
///
/// Accumulated in `f64` via [`sum_sq_f64`] and narrowed once at the end. Summing
/// in `f32` drifts by percentage points over tens of millions of non-zeros, and
/// the square root then hides how far off it was.
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
    T: BixverseNumeric + ToPrimitive,
{
    sum_sq_f64(&mat.data).sqrt() as f32
}

/// Squared Frobenius norm of a sparse matrix, in `f64`.
///
/// Prefer this over squaring [`frobenius_norm`] wherever the result is one of
/// several large terms that cancel: narrowing to `f32` and squaring again throws
/// away the precision the cancellation needs.
///
/// ### Params
///
/// * `mat` - Sparse matrix in CSR or CSC format
///
/// ### Returns
///
/// `||A||_F^2`.
pub fn frobenius_norm_sq_f64<T>(mat: &CompressedSparseData2<T>) -> f64
where
    T: BixverseNumeric + ToPrimitive,
{
    sum_sq_f64(&mat.data)
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
    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let mut vals = Vec::new();

    let n_rows = mat.shape.0;
    for i in 0..n_rows {
        let start = mat.indptr[i] as usize;
        let end = mat.indptr[i + 1] as usize;

        for j in start..end {
            if mat.data[j] != T::default() {
                rows.push(i as u32);
                cols.push(mat.indices[j]);
                vals.push(mat.data[j]);
            }
        }
    }

    coo_to_csr(&rows, &cols, &vals, mat.shape)
}

/// Sparse matrix @ dense vector: `M @ v`.
///
/// Computes `result[i] = sum_j M[i, j] * v[j]` by iterating CSR rows.
///
/// ### Params
///
/// * `mat` - CSR matrix of shape `(m, n)`.
/// * `vec` - Dense vector of length `n`.
///
/// ### Returns
///
/// Dense vector of length `m`. Errors if `mat` is not CSR.
pub fn csr_matvec<T>(mat: &CompressedSparseData2<T>, vec: &[T]) -> Result<Vec<T>, BixverseErrors>
where
    T: BixverseNumeric,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    if !mat.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }

    let mut result = vec![T::default(); mat.shape.0];
    for i in 0..mat.shape.0 {
        let row_start = mat.indptr[i] as usize;
        let row_end = mat.indptr[i + 1] as usize;
        let mut sum = T::default();
        for idx in row_start..row_end {
            sum += mat.data[idx] * vec[mat.indices[idx] as usize];
        }
        result[i] = sum;
    }

    Ok(result)
}

/// Dense vector @ sparse matrix: `v @ M`.
///
/// Computes `result[j] = sum_i v[i] * M[i, j]` by iterating CSR rows of `M`
/// and scattering scaled row contributions into the output. Treats `v` as a
/// row vector.
///
/// ### Params
///
/// * `vec` - Dense vector of length `m`.
/// * `mat` - CSR matrix of shape `(m, n)`.
///
/// ### Returns
///
/// Dense vector of length `n`. Errors if `mat` is not CSR.
pub fn csr_vecmat<T>(vec: &[T], mat: &CompressedSparseData2<T>) -> Result<Vec<T>, BixverseErrors>
where
    T: BixverseNumeric,
{
    if !mat.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }

    let n_cols = mat.ncols();
    let mut out = vec![T::default(); n_cols];
    for (j, &vj) in vec.iter().enumerate() {
        if vj == T::default() {
            continue;
        }
        let gs = mat.indptr[j] as usize;
        let ge = mat.indptr[j + 1] as usize;
        for q in gs..ge {
            let t = mat.indices[q] as usize;
            out[t] += vj * mat.data[q];
        }
    }

    Ok(out)
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

    /// Drain accumulated values into caller-provided buffers and reset the
    /// accumulator.
    ///
    /// ### Params
    ///
    /// * `indices_out` - Buffer to append active indices into
    /// * `data_out` - Buffer to append accumulated values into
    ///
    /// ### Returns
    ///
    /// Number of entries written
    #[inline]
    fn extract_into(&mut self, indices_out: &mut Vec<usize>, data_out: &mut Vec<T>) -> usize {
        self.indices.sort_unstable();
        let n = self.indices.len();
        unsafe {
            for &i in &self.indices {
                indices_out.push(i);
                data_out.push(*self.values.get_unchecked(i));
                *self.flags.get_unchecked_mut(i) = false;
                *self.values.get_unchecked_mut(i) = T::default();
            }
        }
        self.indices.clear();
        n
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
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseNumeric,
{
    let ncol_a = a.shape().1;
    let nrow_b = b.shape().0;

    if !a.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if !b.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if ncol_a != nrow_b {
        return Err(BixverseErrors::SparseMatrixMultiplication {
            n_col_a: ncol_a,
            n_row_b: nrow_b,
        });
    }

    let nrows = a.shape.0;
    let ncols = b.shape.1;

    const CHUNK: usize = 256;
    let n_chunks = nrows.div_ceil(CHUNK);

    let chunks: Vec<(Vec<usize>, Vec<T>, Vec<usize>)> = (0..n_chunks)
        .into_par_iter()
        .map_init(
            || SparseAccumulator::new(ncols),
            |acc, chunk_idx| {
                let start = chunk_idx * CHUNK;
                let end = ((chunk_idx + 1) * CHUNK).min(nrows);

                let mut idx_buf = Vec::new();
                let mut data_buf = Vec::new();
                let mut lengths = Vec::with_capacity(end - start);

                for i in start..end {
                    for a_idx in a.indptr[i] as usize..a.indptr[i + 1] as usize {
                        let k = a.indices[a_idx] as usize;
                        let a_val = a.data[a_idx];
                        for b_idx in b.indptr[k] as usize..b.indptr[k + 1] as usize {
                            unsafe {
                                acc.add(b.indices[b_idx] as usize, a_val * b.data[b_idx]);
                            }
                        }
                    }
                    lengths.push(acc.extract_into(&mut idx_buf, &mut data_buf));
                }

                (idx_buf, data_buf, lengths)
            },
        )
        .collect();

    let total_nnz: usize = chunks.iter().map(|(i, _, _)| i.len()).sum();

    let mut indptr: Vec<u32> = Vec::with_capacity(nrows + 1);
    indptr.push(0);
    let mut running = 0usize;
    for (_, _, lengths) in &chunks {
        for &len in lengths {
            running += len;
            indptr.push(running as u32);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(total_nnz);
    let mut data = Vec::with_capacity(total_nnz);
    for (idx_buf, data_buf, _) in chunks {
        indices.extend(idx_buf.into_iter().map(|x| x as u32));
        data.extend(data_buf);
    }

    // build directly rather than via new_csr, which would .to_vec() the lot.
    Ok(CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (nrows, ncols),
    })
}

/////////////////////////////
// Sparse dense operations //
/////////////////////////////

/// Sparse CSR @ sparse CSR -> dense
///
/// Parallel over output rows. For cases where you assume that the resulting
/// matrix will become dense.
///
/// ### Params
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// ### Returns
///
/// Dense matrix product of both CSR matrices
pub fn csr_sparse_matmul_dense<T>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> Result<Mat<T>, BixverseErrors>
where
    T: BixverseFloat + Send + Sync + Default,
{
    let ncol_a = a.shape().1;
    let nrow_b = b.shape().0;

    if !a.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if !b.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    if ncol_a != nrow_b {
        return Err(BixverseErrors::SparseMatrixMultiplication {
            n_col_a: ncol_a,
            n_row_b: nrow_b,
        });
    }

    let n_rows = a.nrows();
    let n_cols = b.ncols();
    let m_indptr = &a.indptr;
    let m_indices = &a.indices;
    let m_data = &a.data;
    let g_indptr = &b.indptr;
    let g_indices = &b.indices;
    let g_data = &b.data;

    let dense_rows: Vec<Vec<T>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![T::zero(); n_cols];
            let rs = m_indptr[i] as usize;
            let re = m_indptr[i + 1] as usize;
            for p in rs..re {
                let j = m_indices[p] as usize;
                let w = m_data[p];
                let gs = g_indptr[j] as usize;
                let ge = g_indptr[j + 1] as usize;
                for q in gs..ge {
                    let t = g_indices[q] as usize;
                    row[t] += w * g_data[q];
                }
            }
            row
        })
        .collect();

    // this part can be sequential...
    let mut out = Mat::zeros(n_rows, n_cols);
    for (i, row) in dense_rows.into_iter().enumerate() {
        for (j, v) in row.into_iter().enumerate() {
            out[(i, j)] = v;
        }
    }

    Ok(out)
}

/// Apply a CSR operator to a dense row-major block: `out = a @ block`
///
/// The kernel behind repeated diffusion, `T @ (T @ (T @ X))`. Applying the
/// operator `t` times is always preferable to forming `T^t`: at `knn = 30` the
/// operator carries ~30 non-zeros per row, `T^2` ~900 and `T^3` ~27,000, so at
/// 100k rows the explicit cube is on the order of 2.7e9 non-zeros while `t`
/// applications stay at ~30 per row for the same answer.
///
/// Both buffers are flat row-major rather than [faer::Mat] on purpose. The
/// operation is `out_row_i = sum_j w_ij * block_row_j`, so a row-major layout
/// makes every inner step a contiguous SIMD axpy of `width` elements, where
/// faer's column-major storage would turn it into a strided scatter.
///
/// Parallel over output rows, which are disjoint, so no synchronisation.
///
/// Note that `single_cell::sc_analysis::meld` has an older private
/// column-at-a-time version of this (`chebyshev_apply_columns`) that restreams
/// the whole operator once per column. It should move over here at some point.
///
/// ### Params
///
/// * `a` - The operator, CSR, `n x k`
/// * `block` - Dense input, row-major, `k * width` elements
/// * `width` - Columns in the block
/// * `out` - Dense output, row-major, `n * width` elements, overwritten
///
/// ### Returns
///
/// `Ok(())`, or [BixverseErrors::SparseMatrixMustBeCsr] if `a` is in CSC, or
/// [BixverseErrors::SparseMatrixMultiplication] if the buffers do not match the
/// operator's shape.
///
/// ### Panics
///
/// If `a` violates its own CSR invariants: a column index at or above
/// `a.shape().1`, or an `indptr` shorter than `a.shape().0 + 1`. Both index
/// safe slices, so this is a panic rather than unsoundness, but it surfaces
/// inside a rayon closure. `CompressedSparseData2` constructors do not validate,
/// so run [assert_invariants](CompressedSparseData2::assert_invariants) on a
/// matrix of uncertain provenance.
pub fn csr_matmul_dense_block<T>(
    a: &CompressedSparseData2<T>,
    block: &[T],
    width: usize,
    out: &mut [T],
) -> Result<(), BixverseErrors>
where
    T: BixverseFloat + BixverseSimd,
{
    if !a.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }

    let (n_rows, n_cols) = a.shape();

    // `width == 0` would make both buffers empty regardless of the operator, so
    // the length checks below could not tell a shape mismatch from a no-op.
    if width == 0 || block.len() != n_cols * width || out.len() != n_rows * width {
        return Err(BixverseErrors::SparseMatrixMultiplication {
            n_col_a: n_cols,
            n_row_b: block.len().checked_div(width).unwrap_or(0),
        });
    }

    let indptr = &a.indptr;
    let indices = &a.indices;
    let data = &a.data;

    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, out_row)| {
            out_row.fill(T::zero());
            for p in indptr[i] as usize..indptr[i + 1] as usize {
                let j = indices[p] as usize;
                T::bxv_axpy_simd(out_row, data[p], &block[j * width..(j + 1) * width]);
            }
        });

    Ok(())
}

///////////////////////
// Sparse statistics //
///////////////////////

/// Calculate the column means for CSC [CompressedSparseData2]
///
/// ### Params
///
/// * `csc` - The [CompressedSparseData2] (needs to have floats)
/// * `use_second_layer` - Use the second data layer
///
/// ### Returns
///
/// The column means
pub fn sparse_col_means_csc<T>(
    csc: &CompressedSparseData2<T>,
    use_second_layer: bool,
) -> Result<Vec<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseSimd,
{
    if csc.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsc);
    }

    let active_data: &[T] = if use_second_layer {
        csc.data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?
            .as_slice()
    } else {
        csc.data.as_slice()
    };

    let (nrows, ncols) = csc.shape();
    let nrows_t = T::from_usize(nrows).unwrap();
    let mut col_means: Vec<T> = Vec::with_capacity(ncols);

    for j in 0..ncols {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        let sum = T::bxv_sum(&active_data[start..end]);
        col_means.push(sum / nrows_t);
    }

    Ok(col_means)
}

/// Calculate the column standard deviations for CSC [CompressedSparseData2]
///
/// ### Params
///
/// * `csc` - The [CompressedSparseData2] (needs to have floats)
/// * `use_second_layer` - Use the second data layer
///
/// ### Returns
///
/// The column standard deviations
pub fn sparse_col_sds_csc<T>(
    csc: &CompressedSparseData2<T>,
    use_second_layer: bool,
) -> Result<Vec<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseSimd,
{
    if csc.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsc);
    }

    let active_data: &[T] = if use_second_layer {
        csc.data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?
            .as_slice()
    } else {
        csc.data.as_slice()
    };

    let (nrows, ncols) = csc.shape();
    let nrows_t = T::from_usize(nrows).unwrap();
    let denom = nrows_t - T::one();
    let mut col_sds: Vec<T> = Vec::with_capacity(ncols);

    for j in 0..ncols {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        let col_slice = &active_data[start..end];
        let implicit_zeros = T::from_usize(nrows - (end - start)).unwrap();

        let mean = T::bxv_sum(col_slice) / nrows_t;
        let ssd_nonzero = T::bxv_sum_squared_deviation(col_slice, mean);
        let ssd_total = ssd_nonzero + implicit_zeros * mean * mean;
        col_sds.push((ssd_total / denom).sqrt());
    }

    Ok(col_sds)
}

/////////////////////////
// Sparse correlations //
/////////////////////////

/// Below this a column counts as constant and its correlations are reported as
/// `0.0` rather than as a division by a vanishing standard deviation.
const SPARSE_COR_SD_EPS: f64 = 1e-8;

/// One sparse column reduced to what [`sparse_pairwise_correlations`] needs.
///
/// The stored entries are kept as they arrive; nothing here densifies, so the
/// memory is `O(nnz)` per column rather than `O(n_rows)`.
#[derive(Clone, Debug)]
pub struct SparseColMoments {
    /// Row indices of the stored entries. Ascending order is not required.
    pub indices: Vec<u32>,
    /// Values at those indices. The raw values for Pearson, the zero-shifted
    /// ranks for Spearman. Every non-stored position is exactly zero in both
    /// cases, which is what makes the closed forms below valid.
    pub values: Vec<f32>,
    /// Mean over all `n_rows` entries, structural zeros included.
    pub mean: f64,
    /// Standard deviation over all `n_rows` entries, sample (n - 1)
    /// denominator.
    pub sd: f64,
}

/// Average ranks of one sparse column, shifted so the zero block sits at zero.
///
/// ### Params
///
/// * `values` - The stored values of the column.
/// * `n_rows` - Full length of the column, structural zeros included.
///
/// ### Returns
///
/// The shifted average ranks, aligned to `values`.
fn shifted_ranks_sparse(values: &[f32], n_rows: usize) -> Vec<f32> {
    let n_stored = values.len();
    let n_implicit = n_rows.saturating_sub(n_stored);

    let n_neg = values.iter().filter(|&&v| v < 0.0).count();
    let n_stored_zero = values.iter().filter(|&&v| v == 0.0).count();
    let zero_block = n_stored_zero + n_implicit;

    // The zero group occupies 1-based ranks (n_neg + 1) ..= (n_neg + zero_block).
    let r_zero = n_neg as f64 + (zero_block as f64 + 1.0) / 2.0;

    let mut order: Vec<u32> = (0..n_stored as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        values[a as usize]
            .partial_cmp(&values[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0_f32; n_stored];
    let mut i = 0;
    while i < n_stored {
        let value = values[order[i] as usize];
        let start = i;
        while i < n_stored && values[order[i] as usize] == value {
            i += 1;
        }

        let avg_rank = if value == 0.0 {
            r_zero
        } else {
            // A stored value at sorted position `p` sits at 1-based full rank
            // `p + 1` when negative, and `p + 1 + n_implicit` when positive:
            // the implicit zeros are spliced in ahead of it.
            let offset = if value < 0.0 { 0 } else { n_implicit };
            (start + i + 1 + 2 * offset) as f64 / 2.0
        };

        let shifted = (avg_rank - r_zero) as f32;
        for &slot in &order[start..i] {
            ranks[slot as usize] = shifted;
        }
    }

    ranks
}

/// Reduce one sparse column to its moments without densifying it.
///
/// ### Params
///
/// * `indices` - Row indices of the stored entries, each below `n_rows` and
///   without duplicates. Order does not matter.
/// * `values` - Values at those indices, aligned to `indices`.
/// * `n_rows` - Full length of the column, structural zeros included. Must be
///   at least `values.len()`.
/// * `spearman` - Rank the column first, for Spearman rather than Pearson.
///
/// ### Returns
///
/// The [`SparseColMoments`] for this column.
pub fn sparse_col_moments(
    indices: &[u32],
    values: &[f32],
    n_rows: usize,
    spearman: bool,
) -> SparseColMoments {
    let stored: Vec<f32> = if spearman {
        shifted_ranks_sparse(values, n_rows)
    } else {
        values.to_vec()
    };

    let n = n_rows as f64;
    let mean = sum_widen_simd_f32(&stored) / n;
    let ss_stored = sum_squared_dev_widen_simd_f32(&stored, mean);
    let ss = ss_stored + n_rows.saturating_sub(stored.len()) as f64 * mean * mean;
    let sd = if n_rows < 2 {
        0.0
    } else {
        (ss / (n - 1.0)).sqrt()
    };

    SparseColMoments {
        indices: indices.to_vec(),
        values: stored,
        mean,
        sd,
    }
}

/// Pearson correlation of specified column pairs, over the sparsity patterns.
///
/// Uses the raw-moment form of the covariance,
///
/// ```text
/// sum (x_a - m_a)(x_b - m_b) = sum x_a x_b - n * m_a * m_b
/// ```
///
/// because `sum x_a x_b` only picks up the intersection of the two sparsity
/// patterns. On single-cell counts that is a hundredth of the column rather
/// than all of it.
///
/// That form is the one [`crate::core::math::vector_helpers::pearson_correlation`]
/// warns against, so the precondition is worth stating: it cancels
/// catastrophically when the data carries a large constant offset relative to
/// its spread. Log-normalised counts have a mean-to-sd ratio around a third, so
/// the two subtracted terms are within about one decimal digit of their
/// difference. Accumulating in `f64` then leaves roughly `1e-13` relative on the
/// numerator. Do not point this at a column with a large offset.
///
/// The intersection is taken by scatter/gather through a scratch buffer rather
/// than by merging two sorted index runs, because the indices are not
/// guaranteed ascending: a filtered read emits them in the order the caller's
/// selection was given. Pairs are grouped by their first column so the scatter
/// is paid once per distinct column rather than once per pair.
///
/// ### Params
///
/// * `moments` - Per-column moments, see [`sparse_col_moments`].
/// * `pairs` - Index pairs into `moments`, one per requested correlation.
/// * `n_rows` - Full column length, structural zeros included.
///
/// ### Returns
///
/// One correlation per entry of `pairs`, in the same order, clamped to
/// `[-1, 1]`. A column whose standard deviation is below
/// [`SPARSE_COR_SD_EPS`] yields `0.0`.
pub fn sparse_pairwise_correlations(
    moments: &[SparseColMoments],
    pairs: &[(usize, usize)],
    n_rows: usize,
) -> Vec<f32> {
    let mut grouped: FxHashMap<usize, Vec<(usize, usize)>> = FxHashMap::default();
    for (slot, &(a, b)) in pairs.iter().enumerate() {
        grouped.entry(a).or_default().push((slot, b));
    }
    let groups: Vec<(usize, Vec<(usize, usize)>)> = grouped.into_iter().collect();

    let n = n_rows as f64;
    let denom = n - 1.0;

    let solved: Vec<Vec<(usize, f32)>> = groups
        .par_iter()
        .map_init(
            || vec![0_f32; n_rows],
            |scratch, (first, partners)| {
                let a = &moments[*first];
                for (&idx, &value) in a.indices.iter().zip(a.values.iter()) {
                    scratch[idx as usize] = value;
                }

                let out = partners
                    .iter()
                    .map(|&(slot, second)| {
                        let b = &moments[second];
                        if a.sd < SPARSE_COR_SD_EPS || b.sd < SPARSE_COR_SD_EPS {
                            return (slot, 0_f32);
                        }
                        // scratch is zero wherever `a` has no entry, so the
                        // product drops out and no branch is needed.
                        let mut cross = 0_f64;
                        for (&idx, &value) in b.indices.iter().zip(b.values.iter()) {
                            cross += scratch[idx as usize] as f64 * value as f64;
                        }
                        let cov = cross - n * a.mean * b.mean;
                        let cor = cov / (denom * a.sd * b.sd);
                        (slot, (cor as f32).clamp(-1_f32, 1_f32))
                    })
                    .collect();

                // Clear only what was written; refilling the whole buffer would
                // cost `n_rows` per group instead of `nnz`.
                for &idx in &a.indices {
                    scratch[idx as usize] = 0_f32;
                }

                out
            },
        )
        .collect();

    let mut res = vec![0_f32; pairs.len()];
    for group in solved {
        for (slot, value) in group {
            res[slot] = value;
        }
    }
    res
}

////////////////////////
// Lanczos Eigenvalue //
////////////////////////

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
    T: BixverseSimd,
{
    assert_same_len!(a, b);
    T::bxv_dot_simd(a, b)
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
    T: BixverseSimd + BixverseFloat,
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
    T: BixverseSimd + BixverseFloat,
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
fn tridiag_eig<T>(alpha: &[T], beta: &[T]) -> Result<(Vec<T>, Mat<T>), BixverseErrors>
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

    let eig = t
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| BixverseErrors::FaerEigenError)?;
    let evals = eig.S().column_vector().iter().copied().collect();
    let evecs = eig.U().to_owned();

    Ok((evals, evecs))
}

/// Ritz vectors retained across a restart, above the requested pair count.
const LANCZOS_RESTART_GUARD: usize = 5;

/// Norm ratio below which a second Gram-Schmidt pass is run.
///
/// The DGKS criterion. A drop past `1 / sqrt(2)` means the projection cancelled
/// at least half the vector, which is where the round-off it leaves behind
/// stops being negligible. Above that the second pass is measurably wasted work.
const LANCZOS_ORTHO_REFINE_RATIO: f64 = 0.707;

/// Lanczos breakdown threshold, relative to the running `||A||` estimate.
///
/// An absolute threshold is meaningless for a matrix that is not `O(1)`. The
/// attainable floor on the residual norm after two Gram-Schmidt passes is
/// `~eps * sqrt(n) * ||A||`, so this sits comfortably above it while still only
/// firing on a genuinely invariant subspace.
const LANCZOS_BREAKDOWN_RATIO: f64 = 1e-12;

/// Parameters for the restarted Lanczos eigensolver.
#[derive(Clone, Copy, Debug)]
pub struct LanczosParams {
    /// Krylov basis vectors held at once. `None` derives it from the requested
    /// component count as `max(2k + 10, k + 20)`, which leaves enough room
    /// above the wanted pairs for the restart to make progress.
    pub basis_size: Option<usize>,
    /// Maximum restart cycles. Each cycle costs one basis worth of
    /// matrix-vector products plus the orthogonalisation against the basis, so
    /// this is the only knob that bounds the run time.
    pub max_restarts: usize,
    /// Relative residual tolerance. A Ritz pair counts as converged once
    /// `||A x - lambda x||` drops below `tol * ||A||`. The scaling matters: an
    /// absolute threshold is unreachable for a matrix whose norm is large and
    /// trivially met for one whose norm is tiny.
    pub tol: f64,
}

impl LanczosParams {
    /// Defaults tuned on a single cell diffusion kernel.
    ///
    /// ### Returns
    ///
    /// Self with a derived basis size, 64 restarts and a relative tolerance of
    /// `1e-8`.
    pub fn new() -> Self {
        Self {
            basis_size: None,
            max_restarts: 64,
            tol: 1e-8,
        }
    }

    /// Resolve the basis size for a given component count.
    ///
    /// ### Params
    ///
    /// * `n_components` - Eigenpairs requested, already capped at `n`.
    /// * `n` - Matrix dimension, which caps the basis.
    ///
    /// ### Returns
    ///
    /// The number of basis vectors to hold, at least one and never above `n`.
    fn resolve_basis(&self, n_components: usize, n: usize) -> usize {
        self.basis_size
            .unwrap_or_else(|| (n_components * 2 + 10).max(n_components + 20))
            .max(n_components + 1)
            .min(n)
            .max(1)
    }
}

impl Default for LanczosParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of a restarted Lanczos solve, with the convergence diagnostics.
///
/// The diagnostics exist because a budget-exhausted solve and a converged one
/// are otherwise indistinguishable from the outside, and the difference is
/// several digits of accuracy in the eigenvalues.
#[derive(Clone, Debug)]
pub struct LanczosResult {
    /// Eigenvalues sorted descending. Kept in `f64` because the callers form
    /// `lambda / (1 - lambda)`, which is catastrophic in `f32` once `lambda`
    /// approaches one, as it does for any diffusion operator.
    pub eigenvalues: Vec<f64>,
    /// `eigenvectors[i][j]` is element `i` of eigenvector `j`. Always exactly
    /// `eigenvalues.len()` columns wide.
    pub eigenvectors: Vec<Vec<f32>>,
    /// Largest residual `||A x - lambda x||` over the returned pairs.
    pub residual: f64,
    /// Scale the residual was tested against, i.e. the `||A||` estimate.
    pub norm_estimate: f64,
    /// Whether `residual` met `tol * norm_estimate`.
    pub converged: bool,
    /// Restart cycles actually run.
    pub restarts: usize,
}

/// Compute the largest eigenpairs of a symmetric sparse matrix.
///
/// Thin wrapper over [compute_largest_eigenpairs_lanczos_diag] for callers that
/// do not inspect the convergence diagnostics.
///
/// ### Params
///
/// * `matrix` - Symmetric sparse matrix. CSC input is converted to CSR.
/// * `n_components` - Number of eigenpairs to compute.
/// * `seed` - Seed for the starting vector.
/// * `params` - Optional [LanczosParams], defaulted when `None`.
///
/// ### Returns
///
/// `(eigenvalues, eigenvectors)` sorted by descending eigenvalue, where
/// `eigenvectors[i][j]` is element `i` of eigenvector `j`. Both outputs carry
/// the same number of components, which is `min(n_components, n)` and can be
/// smaller still if the iteration hit an invariant subspace first.
pub fn compute_largest_eigenpairs_lanczos<T>(
    matrix: &CompressedSparseData2<T>,
    n_components: usize,
    seed: u64,
    params: Option<LanczosParams>,
) -> Result<(Vec<f64>, Vec<Vec<f32>>), BixverseErrors>
where
    T: BixverseNumeric + BixverseSimd + Into<f64>,
{
    let res = compute_largest_eigenpairs_lanczos_diag(matrix, n_components, seed, params)?;
    Ok((res.eigenvalues, res.eigenvectors))
}

/// Compute the largest eigenpairs of a symmetric sparse matrix, with
/// diagnostics.
///
/// Thick-restarted Lanczos. Each cycle builds a Krylov basis of
/// [LanczosParams::basis_size] vectors with full reorthogonalisation, does a
/// Rayleigh-Ritz extraction, and restarts from the best Ritz vectors plus the
/// residual direction. Restarting is the key point: the basis stays bounded
/// while the iteration keeps going, so a clustered spectrum converges without
/// the quadratic orthogonalisation cost a single long run would need.
///
/// The projected matrix is accumulated explicitly as `H[i][j] = <v_i, A v_j>`
/// rather than as a tridiagonal recurrence. After a thick restart the leading
/// basis vectors are Ritz vectors, not Lanczos vectors, so the projection is
/// arrow-shaped rather than tridiagonal; forming it directly handles that
/// without special-casing.
///
/// The basis is held column-major as one flat buffer, so the projection, the
/// reorthogonalisation, the restart and the final Ritz vectors are all faer
/// products rather than scalar loops. Orthogonalisation is classical
/// Gram-Schmidt with a DGKS-conditional second pass, which matches the accuracy
/// of two unconditional modified passes while being expressible as two matrix
/// products instead of 2 * basis_size scalar loops.
///
/// Convergence uses the exact Arnoldi residual |beta_m| * |y[m-1]|, which
/// still holds under thick restart because the relation A V = V H + beta v
/// e^T is preserved across cycles, tested against tol * ||A||.
///
/// Internals are f64 regardless of the input type. Eigenvalues come back in
/// f64, eigenvectors in f32.
///
/// ### Params
///
/// * `matrix` - Symmetric sparse matrix. CSC input is converted to CSR.
/// * `n_components` - Number of eigenpairs to compute. Capped at the matrix
///   dimension rather than erroring, so a request for more pairs than the
///   matrix has simply returns all of them.
/// * `seed` - Seed for the starting vector.
/// * `params` - Optional [LanczosParams], defaulted when None.
///
/// ### Returns
///
/// A [LanczosResult]. The eigenvalue count and the eigenvector column count
/// are always equal.
pub fn compute_largest_eigenpairs_lanczos_diag<T>(
    matrix: &CompressedSparseData2<T>,
    n_components: usize,
    seed: u64,
    params: Option<LanczosParams>,
) -> Result<LanczosResult, BixverseErrors>
where
    T: BixverseNumeric + BixverseSimd + Into<f64>,
{
    let params = params.unwrap_or_default();
    let n = matrix.shape.0;

    if n_components == 0 || n == 0 {
        return Err(BixverseErrors::MustBePositive(
            "n_components and the matrix dimension".to_string(),
        ));
    }

    let n_keep = n_components.min(n);
    let m = params.resolve_basis(n_keep, n);

    // CSR gives contiguous rows for the matrix-vector product.
    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix.clone(),
        CompressedSparseFormat::Csc => matrix.transform(),
    };
    let data_f64: Vec<f64> = csr.data.iter().map(|&v| v.into()).collect();

    let matvec = |x: &[f64], y: &mut [f64]| {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = 0.0;
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                let j = csr.indices[idx] as usize;
                sum += data_f64[idx] * x[j];
            }
            *yi = sum;
        });
    };

    // Column-major basis, one flat buffer, so every pass over it is a `faer`
    // product and column `j` is still a contiguous slice for the matrix product.
    let mut basis = vec![0.0f64; m * n];
    let mut h = Mat::<f64>::zeros(m, m);
    let mut w = vec![0.0f64; n];
    let mut coeffs = vec![0.0f64; m];

    let mut rng = StdRng::seed_from_u64(seed);
    for x in basis[..n].iter_mut() {
        *x = rng.random::<f64>() - 0.5;
    }
    normalise(&mut basis[..n]);

    // Basis vectors carried over from the previous cycle. Zero on the first.
    let mut locked = 0usize;
    let mut ritz_values: Vec<f64> = Vec::new();
    let mut ritz_coeffs = Mat::<f64>::zeros(0, 0);
    let mut m_final = 0usize;

    // Lower bound on ||A||_2, refined as the iteration sees more of the matrix.
    let mut norm_estimate = 0.0f64;
    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut restarts = 0usize;

    let budget = params.max_restarts.max(1);

    for cycle in 0..budget {
        restarts = cycle + 1;
        let mut m_eff = m;
        let mut last_beta = 0.0f64;

        for j in locked..m {
            matvec(&basis[j * n..(j + 1) * n], &mut w);

            let width = j + 1;

            // Rayleigh-Ritz coefficients against the whole basis so far. Taken
            // before any subtraction, so `h` is the exact projection, and they
            // double as the first Gram-Schmidt coefficient set.
            basis_project(&basis, n, width, &w, &mut coeffs[..width]);
            for i in 0..width {
                h[(i, j)] = coeffs[i];
                h[(j, i)] = coeffs[i];
            }

            let before = norm(&w);
            norm_estimate = norm_estimate.max(before);

            basis_subtract(&basis, n, width, &coeffs[..width], &mut w);
            last_beta = norm(&w);

            // DGKS: refine only when the projection cancelled enough of the
            // vector for the round-off it left behind to matter.
            if last_beta < LANCZOS_ORTHO_REFINE_RATIO * before {
                basis_project(&basis, n, width, &w, &mut coeffs[..width]);
                basis_subtract(&basis, n, width, &coeffs[..width], &mut w);
                last_beta = norm(&w);
            }

            // Breakdown: the basis already spans an invariant subspace, so the
            // Ritz pairs from it are exact and there is nothing left to add.
            if last_beta <= LANCZOS_BREAKDOWN_RATIO * norm_estimate {
                m_eff = width;
                last_beta = 0.0;
                break;
            }

            if width < m {
                let inv = 1.0 / last_beta;
                let next = &mut basis[width * n..(width + 1) * n];
                for (dst, &src) in next.iter_mut().zip(w.iter()) {
                    *dst = src * inv;
                }
            }
        }

        let (values, vectors) = symmetric_eigen_descending(h.as_ref(), m_eff)?;

        // Residual of Ritz pair i is |beta| * |y[m_eff - 1, i]|, exact under the
        // Arnoldi relation that thick restart preserves.
        let n_out = n_keep.min(m_eff);
        residual = (0..n_out).fold(0.0f64, |acc, i| {
            acc.max(last_beta * vectors[(m_eff - 1, i)].abs())
        });

        // The extreme Ritz values are the sharpest available lower bound on
        // ||A||, so scale the tolerance by them rather than testing an absolute
        // threshold that a large matrix can never reach.
        norm_estimate = norm_estimate
            .max(values[0].abs())
            .max(values[m_eff - 1].abs());
        converged = residual <= params.tol * norm_estimate;

        ritz_values = values;
        ritz_coeffs = vectors;
        m_final = m_eff;

        if converged || m_eff < m || last_beta == 0.0 || cycle + 1 == budget {
            break;
        }

        // Thick restart: keep the best Ritz vectors, then hand the residual
        // direction over as the next basis vector so the Krylov space continues
        // to grow from where it stopped.
        // Never retain more than half the basis: past that the cycle buys too
        // few new Krylov vectors to pay for the restart, and at `keep = m - 1`
        // it buys exactly one and the iteration effectively stops.
        let keep = (n_keep + LANCZOS_RESTART_GUARD).min(m / 2).max(1);
        let mut restarted = vec![0.0f64; keep * n];
        basis_expand(&basis, n, m_eff, ritz_coeffs.as_ref(), keep, &mut restarted);

        let inv = 1.0 / last_beta;
        for x in w.iter_mut() {
            *x *= inv;
        }

        h.fill(0.0);
        basis[..keep * n].copy_from_slice(&restarted);
        for (i, &value) in ritz_values.iter().take(keep).enumerate() {
            h[(i, i)] = value;
        }
        basis[keep * n..(keep + 1) * n].copy_from_slice(&w);
        locked = keep;
    }

    // Ritz vectors are built once, on the way out. Building them every cycle is
    // pure waste: only the last cycle's are ever returned, and the retained
    // block of the restart is bit-identical to the leading columns anyway.
    let n_out = n_keep.min(m_final);
    let mut ritz = vec![0.0f64; n_out * n];
    basis_expand(&basis, n, m_final, ritz_coeffs.as_ref(), n_out, &mut ritz);

    let eigenvalues: Vec<f64> = ritz_values[..n_out].to_vec();
    let mut eigenvectors = vec![vec![0.0f32; n_out]; n];

    for comp in 0..n_out {
        let column = &ritz[comp * n..(comp + 1) * n];
        let scale: f64 = column.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (point, &v) in column.iter().enumerate() {
            eigenvectors[point][comp] = (v * inv) as f32;
        }
    }

    Ok(LanczosResult {
        eigenvalues,
        eigenvectors,
        residual,
        norm_estimate,
        converged,
        restarts,
    })
}

/// Project a vector onto the leading columns of the basis.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Leading columns to project against.
/// * `v` - Vector of length `n`.
/// * `out` - Receives `V^T v`; must be exactly `width` long.
///
/// ### Returns
///
/// Nothing; `out` is overwritten.
fn basis_project(basis: &[f64], n: usize, width: usize, v: &[f64], out: &mut [f64]) {
    let b = MatRef::from_column_major_slice(&basis[..width * n], n, width);
    let v = MatRef::from_column_major_slice(v, n, 1);
    let mut out = MatMut::from_column_major_slice_mut(out, width, 1);

    matmul(
        out.as_mut(),
        Accum::Replace,
        b.transpose(),
        v,
        1.0,
        faer_parallelism(),
    );
}

/// Subtract the basis expansion of a coefficient vector in place.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Leading columns to expand.
/// * `coeffs` - Coefficients, exactly `width` long.
/// * `v` - Vector of length `n`, updated to `v - V c`.
///
/// ### Returns
///
/// Nothing; `v` is updated in place.
fn basis_subtract(basis: &[f64], n: usize, width: usize, coeffs: &[f64], v: &mut [f64]) {
    let b = MatRef::from_column_major_slice(&basis[..width * n], n, width);
    let c = MatRef::from_column_major_slice(coeffs, width, 1);
    let mut v = MatMut::from_column_major_slice_mut(v, n, 1);

    matmul(v.as_mut(), Accum::Add, b, c, -1.0, faer_parallelism());
}

/// Expand Ritz coefficient columns back into the full space.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Basis columns spanned by the coefficients.
/// * `coeffs` - Rayleigh-Ritz eigenvectors, column-wise.
/// * `n_out` - Leading coefficient columns to expand.
/// * `out` - Receives the `n_out` expanded vectors, column-major.
///
/// ### Returns
///
/// Nothing; `out` is overwritten.
fn basis_expand(
    basis: &[f64],
    n: usize,
    width: usize,
    coeffs: MatRef<f64>,
    n_out: usize,
    out: &mut [f64],
) {
    let b = MatRef::from_column_major_slice(&basis[..width * n], n, width);
    let mut out = MatMut::from_column_major_slice_mut(out, n, n_out);

    matmul(
        out.as_mut(),
        Accum::Replace,
        b,
        coeffs.subcols(0, n_out),
        1.0,
        faer_parallelism(),
    );
}

/// Eigen-decompose the leading `size` block of a symmetric matrix, descending.
///
/// ### Params
///
/// * `h` - Symmetric matrix; only the leading `size` by `size` block is read.
/// * `size` - Size of the block to decompose.
///
/// ### Returns
///
/// `(eigenvalues, eigenvectors)` sorted by descending eigenvalue, the
/// eigenvectors held column-wise.
fn symmetric_eigen_descending(
    h: MatRef<f64>,
    size: usize,
) -> Result<(Vec<f64>, Mat<f64>), BixverseErrors> {
    let block = Mat::<f64>::from_fn(size, size, |i, j| h[(i, j)]);

    let eig = block
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| BixverseErrors::FaerEigenError)?;

    let raw_values = eig.S();
    let raw_vectors = eig.U();

    let mut order: Vec<usize> = (0..size).collect();
    order.sort_by(|&a, &b| raw_values[b].total_cmp(&raw_values[a]));

    let values: Vec<f64> = order.iter().map(|&i| raw_values[i]).collect();
    let vectors = Mat::<f64>::from_fn(size, size, |i, j| raw_vectors[(i, order[j])]);

    Ok((values, vectors))
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
/// * `col_means` - Optional column means for implicit mean centering
/// * `col_stds` - Optional column sds for implicit variance normalising
/// * `row_offsets` - Additional offsets (for example for CLR-type PCA in single
///   cell).
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
    row_offsets: Option<&[F]>,
) -> Result<SvdResults<F>, BixverseErrors>
where
    T: BixverseNumeric + BixverseSimd + Into<F> + Clone,
    U: BixverseNumeric + Into<F> + Clone,
    F: BixverseFloat + BixverseSimd + std::iter::Sum,
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
            csc_owned = matrix.transform_single_layer(use_second_layer)?;
            csc = &csc_owned;
        }
        CompressedSparseFormat::Csc => {
            csc = matrix;
            csr_owned = matrix.transform_single_layer(use_second_layer)?;
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
        let x_sum: F = if row_offsets.is_some() {
            x_scaled.iter().copied().sum()
        } else {
            F::zero()
        };

        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = F::zero();
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                let j = csr.indices[idx] as usize;
                sum += data_csr_f[idx] * x_scaled[j];
            }
            if col_means.is_some() {
                sum -= mean_dot;
            }
            if let Some(off) = row_offsets {
                sum -= off[i] * x_sum;
            }
            *yi = sum;
        });
    };

    let matvec_at = |x: &[F], y: &mut [F]| {
        let x_sum: F = x.iter().copied().sum();
        let m_dot_x: F = if let Some(off) = row_offsets {
            off.iter().zip(x.iter()).map(|(&m_i, &xi)| m_i * xi).sum()
        } else {
            F::zero()
        };

        y.par_iter_mut().enumerate().for_each(|(j, yj)| {
            let mut sum = F::zero();
            for idx in csc.indptr[j] as usize..csc.indptr[j + 1] as usize {
                let i = csc.indices[idx] as usize;
                sum += data_csc_f[idx] * x[i];
            }
            if let Some(mu) = col_means {
                sum -= mu[j] * x_sum;
            }
            if row_offsets.is_some() {
                sum -= m_dot_x;
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
    let (evals, evecs) = tridiag_eig(&alpha[..n_iter], &beta[..n_iter - 1])?;

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

    Ok(SvdResults {
        u,
        s: singular_values,
        v,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::vector_helpers::rank_vector;
    use faer::Mat;

    ////////////////
    // validate() //
    ////////////////

    /// A 3 x 4 CSR with one empty row, both layers populated.
    fn validatable_csr() -> CompressedSparseData2<f32, f32> {
        CompressedSparseData2 {
            data: vec![1.0, 2.0, 3.0, 4.0],
            indices: vec![0, 3, 1, 2],
            indptr: vec![0, 2, 2, 4],
            cs_type: CompressedSparseFormat::Csr,
            data_2: Some(vec![10.0, 20.0, 30.0, 40.0]),
            shape: (3, 4),
        }
    }

    #[test]
    fn test_validate_accepts_a_well_formed_matrix() {
        assert!(validatable_csr().validate().is_ok());
        assert!(validatable_csr().transform().validate().is_ok());
    }

    /// The single-layer transform leaves `data` empty on purpose, so an empty
    /// layer must read as absent rather than as a length mismatch.
    #[test]
    fn test_validate_accepts_an_empty_unused_layer() {
        let single = validatable_csr().transform_single_layer(true).unwrap();

        assert!(single.data.is_empty());
        assert!(single.validate().is_ok());
    }

    /// An indptr that disagrees with the declared major axis would otherwise
    /// silently process fewer rows than the shape claims.
    #[test]
    fn test_validate_rejects_a_short_indptr() {
        let mut m = validatable_csr();
        m.indptr = vec![0, 2, 4];

        assert!(matches!(
            m.validate(),
            Err(BixverseErrors::SparseIndptrInvalid {
                expected: 4,
                got: 3,
                ..
            })
        ));
    }

    #[test]
    fn test_validate_rejects_a_final_offset_past_the_indices() {
        let mut m = validatable_csr();
        m.indptr = vec![0, 2, 2, 9];

        assert!(matches!(
            m.validate(),
            Err(BixverseErrors::SparseIndptrInvalid {
                expected: 4,
                got: 9,
                ..
            })
        ));
    }

    #[test]
    fn test_validate_rejects_decreasing_offsets() {
        let mut m = validatable_csr();
        m.indptr = vec![0, 3, 1, 4];

        assert!(matches!(
            m.validate(),
            Err(BixverseErrors::SparseIndptrInvalid { .. })
        ));
    }

    /// This is the one that matters: an index past the minor axis is what turns
    /// a dense scatter into an out-of-bounds panic in a worker thread.
    #[test]
    fn test_validate_rejects_an_index_past_the_minor_axis() {
        let mut m = validatable_csr();
        m.indices = vec![0, 3, 1, 4];

        assert!(matches!(
            m.validate(),
            Err(BixverseErrors::SliceIndexOutOfBounds { index: 4, len: 4 })
        ));
    }

    /// CSC bounds the indices by the row count, not the column count, so the
    /// same index list is fine in one orientation and not the other.
    #[test]
    fn test_validate_bounds_csc_by_the_row_count() {
        let mut m = validatable_csr();
        m.cs_type = CompressedSparseFormat::Csc;
        m.shape = (4, 3);

        // indices max to 3, rows are 4: fine
        assert!(m.validate().is_ok());

        m.shape = (3, 3);
        assert!(matches!(
            m.validate(),
            Err(BixverseErrors::SliceIndexOutOfBounds { index: 3, len: 3 })
        ));
    }

    #[test]
    fn test_validate_rejects_a_truncated_second_layer() {
        let mut m = validatable_csr();
        m.data_2 = Some(vec![10.0, 20.0]);

        assert!(matches!(
            m.validate(),
            Err(BixverseErrors::DimensionMisMatchSparse {
                indices_len: 4,
                data_len: 2
            })
        ));
    }

    /// The format parser takes either case and rejects anything that is not CSR or CSC.
    #[test]
    fn test_parse_sparse_format() {
        assert!(parse_compressed_sparse_format("csr").unwrap().is_csr());
        assert!(parse_compressed_sparse_format("CSC").unwrap().is_csc());
        assert!(parse_compressed_sparse_format("dense").is_none());
    }

    /// Five rows, both layers populated, and one empty row so the chunk splits
    /// land on an empty `indptr` span at least once.
    fn transpose_fixture() -> CompressedSparseData2<f64, f64> {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data2: Vec<f64> = data.iter().map(|v| v * 10.0).collect();
        let indices: Vec<u32> = vec![0, 2, 1, 0, 1, 3, 2, 3];
        let indptr: Vec<u32> = vec![0, 2, 3, 3, 6, 8];

        CompressedSparseData2::new_csr(&data, &indices, &indptr, Some(&data2), (5, 4))
    }

    /// The parallel path only differs from the serial one in how the old major
    /// axis is split, so forcing an uneven multi-chunk split is what exercises
    /// the per-chunk cursors without building a matrix past
    /// `PARALLEL_TRANSPOSE_MIN_NNZ`.
    #[test]
    fn test_transpose_chunked_matches_single_chunk() {
        let csr = transpose_fixture();

        let serial = transpose_sparse_chunked(&csr, &[(0, 5)]);

        for chunks in [
            vec![(0, 1), (1, 3), (3, 5)],
            vec![(0, 2), (2, 5)],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        ] {
            let parallel = transpose_sparse_chunked(&csr, &chunks);

            assert_eq!(parallel.indptr, serial.indptr);
            assert_eq!(parallel.indices, serial.indices);
            assert_eq!(parallel.data, serial.data);
            assert_eq!(parallel.data_2, serial.data_2);
            assert_eq!(parallel.cs_type.is_csc(), serial.cs_type.is_csc());
            assert_eq!(parallel.shape, serial.shape);
        }
    }

    /// Round-tripping has to give the input back, and the transposed layout has
    /// to be ascending within every new major index.
    #[test]
    fn test_transpose_round_trips_and_stays_sorted() {
        let csr = transpose_fixture();
        let csc = transpose_sparse(&csr);

        assert!(csc.cs_type.is_csc());
        for major in 0..csc.indptr.len() - 1 {
            let lo = csc.indptr[major] as usize;
            let hi = csc.indptr[major + 1] as usize;
            assert!(csc.indices[lo..hi].windows(2).all(|w| w[0] < w[1]));
        }

        let round_trip = transpose_sparse(&csc);
        assert_eq!(round_trip.indptr, csr.indptr);
        assert_eq!(round_trip.indices, csr.indices);
        assert_eq!(round_trip.data, csr.data);
        assert_eq!(round_trip.data_2, csr.data_2);
    }

    /// The single-layer variant drops the layer it was not asked for and keeps
    /// the other one identical to the two-layer transpose.
    #[test]
    fn test_transpose_single_layer_matches_full_transpose() {
        let csr = transpose_fixture();
        let full = transpose_sparse(&csr);

        let counts_only = csr.transform_single_layer(false).unwrap();
        assert_eq!(counts_only.data, full.data);
        assert!(counts_only.data_2.is_none());

        let norm_only = csr.transform_single_layer(true).unwrap();
        assert_eq!(norm_only.data_2, full.data_2);
        assert!(norm_only.data.is_empty());

        assert_eq!(counts_only.indptr, full.indptr);
        assert_eq!(norm_only.indices, full.indices);
    }

    /// Both endpoints of every edge land in the matrix and every row stays ascending.
    #[test]
    fn test_undirected_edges_to_csr_is_symmetric_and_row_sorted() {
        // A path plus one chord, so several rows hold both a `lo < r` partner
        // and a `hi > r` one. That split is the whole trick behind the
        // sort-free scatter, and it is what breaks first if the two passes are
        // ever reordered.
        let edges = vec![(0u32, 1u32, 1.0f64), (0, 3, 2.0), (1, 2, 3.0), (2, 3, 4.0)];

        let csr = undirected_edges_to_csr(4, &edges);

        assert_eq!(csr.shape, (4, 4));
        assert_eq!(csr.indptr, vec![0, 2, 4, 6, 8]);
        assert_eq!(csr.indices, vec![1, 3, 0, 2, 1, 3, 0, 2]);
        assert_eq!(csr.data, vec![1.0, 2.0, 1.0, 3.0, 3.0, 4.0, 2.0, 4.0]);

        // Every row ascending, which every consumer in the crate assumes.
        for i in 0..4 {
            let lo = csr.indptr[i] as usize;
            let hi = csr.indptr[i + 1] as usize;
            assert!(csr.indices[lo..hi].windows(2).all(|w| w[0] < w[1]));
        }
    }

    /// No edges must still produce a well-formed `indptr` of the declared size.
    #[test]
    fn test_undirected_edges_to_csr_handles_empty_input() {
        let csr = undirected_edges_to_csr::<f64>(3, &[]);

        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.indptr, vec![0, 0, 0, 0]);
        assert!(csr.indices.is_empty());
    }

    /// An edgeless node keeps an empty span instead of shifting every later row.
    #[test]
    fn test_undirected_edges_to_csr_keeps_isolated_nodes() {
        // Node 0 has no edges at all, so its span has to be empty rather than
        // shifting every later row.
        let edges = vec![(1u32, 2u32, 7.0f64)];

        let csr = undirected_edges_to_csr(3, &edges);

        assert_eq!(csr.indptr, vec![0, 0, 1, 2]);
        assert_eq!(csr.indices, vec![2, 1]);
    }

    /// A well-formed square CSR passes and reports its dimension.
    #[test]
    fn test_validate_square_csr_accepts_a_sound_matrix() {
        let csr = undirected_edges_to_csr(3, &[(0u32, 1u32, 1.0f64), (1, 2, 2.0)]);

        assert_eq!(validate_square_csr(&csr).unwrap(), 3);
    }

    /// Each structural fault the validator exists for must come back as `MalformedCsr`.
    #[test]
    fn test_validate_square_csr_catches_structural_faults() {
        // Non-monotonic indptr: the Rust range `5..2` is empty, so row 1 is
        // silently skipped rather than read.
        let bad_indptr = CompressedSparseData2::new_csr(
            &[1.0f64; 7],
            &[1u32, 2, 0, 2, 0, 0, 1],
            &[0u32, 5, 2, 7],
            None,
            (3, 3),
        );
        assert!(matches!(
            validate_square_csr(&bad_indptr),
            Err(BixverseErrors::MalformedCsr(_))
        ));

        // A column index past the last row indexes whatever the caller keys off
        // it out of bounds.
        let bad_index = CompressedSparseData2::new_csr(
            &[1.0f64, 1.0],
            &[1u32, 9],
            &[0u32, 1, 2, 2],
            None,
            (3, 3),
        );
        assert!(matches!(
            validate_square_csr(&bad_index),
            Err(BixverseErrors::MalformedCsr(_))
        ));

        // indptr of the wrong length for the declared row count.
        let short_indptr =
            CompressedSparseData2::new_csr(&[1.0f64], &[1u32], &[0u32, 1], None, (3, 3));
        assert!(matches!(
            validate_square_csr(&short_indptr),
            Err(BixverseErrors::MalformedCsr(_))
        ));

        // The last pointer must account for every stored value.
        let short_tail = CompressedSparseData2::new_csr(
            &[1.0f64, 1.0],
            &[1u32, 0],
            &[0u32, 1, 1, 1],
            None,
            (3, 3),
        );
        assert!(matches!(
            validate_square_csr(&short_tail),
            Err(BixverseErrors::MalformedCsr(_))
        ));
    }

    /// The zero counts and the dense-to-CSR conversion agree on what is structurally zero.
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

    /// Adding two CSR matrices merges shared columns rather than storing them twice.
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
        let c = sparse_add_csr(&a, &b).unwrap();

        assert_eq!(c.data, vec![1.0, 3.0, 6.0]);
        assert_eq!(c.indices, vec![0, 1, 1]);
        assert_eq!(c.indptr, vec![0, 2, 3]);
    }

    /// CSR matrix-vector product on a case small enough to check by hand.
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
        let result = csr_matvec(&a, &vec).unwrap();
        assert_eq!(result, vec![4.0, 3.0]);
    }

    /// Lanczos recovers the one nonzero eigenpair of a symmetric rank-one matrix.
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
        let (evals, evecs) = compute_largest_eigenpairs_lanczos(&csr, 1, 42, None).unwrap();

        // True top eigenvalue should be exactly sum(x_i^2) = 1.0 + 4.0 = 5.0
        assert!((evals[0] - 5.0).abs() < 1e-3);

        // Eigenvectors are returned transposed: evecs[point_idx][comp_idx]
        // So the first principal component is [evecs[0][0], evecs[1][0], evecs[2][0], evecs[3][0]]
        let x_norm = 5.0_f32.sqrt();
        let dot_x = (evecs[0][0] * 1.0 + evecs[2][0] * 2.0) / x_norm;

        assert!(dot_x.abs() > 0.999);
    }

    /// Lanczos SVD recovers both singular vectors of a sparse rank-one matrix.
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

        let svd = sparse_svd_lanczos(&csr, 1, 42, false, no_params, no_params, None).unwrap();

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

    use approx::assert_relative_eq;

    /// Adjacency matrix of a path graph on `n` nodes.
    ///
    /// Eigenvalues are `2 cos(k pi / (n + 1))`, separated by `O(1 / n^2)`, which
    /// makes it the cheapest genuinely clustered spectrum to build.
    fn path_graph(n: usize) -> CompressedSparseData2<f64, f64> {
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for i in 0..n - 1 {
            rows[i].push(((i + 1) as u32, 1.0));
            rows[i + 1].push((i as u32, 1.0));
        }
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in rows.iter_mut() {
            row.sort_by_key(|&(j, _)| j);
            for &(j, v) in row.iter() {
                indices.push(j);
                data.push(v);
            }
            indptr.push(indices.len() as u32);
        }

        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
    }

    /// Diagonal matrix in CSR, storing every entry given, zeros included.
    fn diagonal(diag: &[f64]) -> CompressedSparseData2<f64, f64> {
        let indices: Vec<u32> = (0..diag.len() as u32).collect();
        let indptr: Vec<u32> = (0..=diag.len() as u32).collect();

        CompressedSparseData2::new_csr(diag, &indices, &indptr, None, (diag.len(), diag.len()))
    }

    /// `||A x - lambda x||` for the `k`-th returned pair, with `vectors` indexed `[i][k]`.
    fn pair_residual(
        mat: &CompressedSparseData2<f64, f64>,
        values: &[f64],
        vectors: &[Vec<f32>],
        k: usize,
    ) -> f64 {
        let n = mat.shape.0;
        let v: Vec<f64> = (0..n).map(|i| vectors[i][k] as f64).collect();
        let lam = values[k];

        (0..n)
            .map(|i| {
                let mut sum = 0.0;
                for x in mat.indptr[i] as usize..mat.indptr[i + 1] as usize {
                    sum += mat.data[x] * v[mat.indices[x] as usize];
                }
                (sum - lam * v[i]).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Regression: a tightly clustered spectrum used to come back as noise.
    #[test]
    fn test_lanczos_resolves_a_clustered_spectrum_at_defaults() {
        // The regression: this used to come back as noise, with the leading
        // eigenvector uncorrelated with position. The default budget does not
        // drive the residual to zero on a spectrum this tight, so what is
        // asserted here is that the shape is right, not that it is converged.
        let n = 300usize;
        let mat = path_graph(n);

        let (values, vectors) = compute_largest_eigenpairs_lanczos(&mat, 4, 42, None).unwrap();

        for k in 0..4 {
            let expected = 2.0 * ((k + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
            assert_relative_eq!(values[k], expected, epsilon = 1e-3);
        }

        // The leading eigenvector is sin(pi x / (n + 1)), so it has no sign
        // change anywhere. Noise does.
        let sign = vectors[0][0].signum();
        assert!(
            (0..n).all(|i| vectors[i][0].signum() == sign || vectors[i][0] == 0.0),
            "leading eigenvector of a path graph changed sign"
        );
    }

    /// Given a big enough restart budget the same clustered spectrum does converge.
    #[test]
    fn test_lanczos_restarts_converge_a_clustered_spectrum() {
        // Same fixture with a budget large enough to actually converge, which
        // is what the restart machinery exists for. 300 nodes, ~40 cycles.
        let n = 300usize;
        let mat = path_graph(n);
        let params = LanczosParams {
            basis_size: None,
            max_restarts: 64,
            tol: 1e-8,
        };

        let res = compute_largest_eigenpairs_lanczos_diag(&mat, 4, 42, Some(params)).unwrap();

        assert!(res.converged, "residual stalled at {:e}", res.residual);
        for k in 0..4 {
            let expected = 2.0 * ((k + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
            assert_relative_eq!(res.eigenvalues[k], expected, epsilon = 1e-6);

            let residual = pair_residual(&mat, &res.eigenvalues, &res.eigenvectors, k);
            assert!(residual < 1e-5, "pair {k} has residual {residual:e}");
        }
    }

    /// Regression: asking for more components than the matrix has rows used to panic.
    #[test]
    fn test_lanczos_caps_components_at_the_matrix_dimension() {
        // Used to panic inside `clamp` before it ever got to the iteration.
        let mat = diagonal(&[4.0, 3.0, 2.0, 1.0]);

        for requested in [4usize, 5, 40] {
            let (values, vectors) =
                compute_largest_eigenpairs_lanczos(&mat, requested, 11, None).unwrap();

            assert_eq!(values.len(), 4, "requested {requested}");
            assert_eq!(vectors[0].len(), values.len(), "requested {requested}");
            assert_relative_eq!(values[0], 4.0, epsilon = 1e-9);
            assert_relative_eq!(values[3], 1.0, epsilon = 1e-9);
        }
    }

    /// Regression: an early Krylov breakdown used to leave values and vectors different lengths.
    #[test]
    fn test_lanczos_breakdown_keeps_values_and_vectors_the_same_length() {
        // Rank two, so the Krylov space closes after three vectors and there is
        // no way to produce the ten pairs asked for. Both outputs must agree on
        // how many there are; they used to disagree and index out of bounds
        // downstream.
        let mut diag = vec![0.0f64; 12];
        diag[0] = 5.0;
        diag[1] = 3.0;
        let mat = diagonal(&diag);

        let res = compute_largest_eigenpairs_lanczos_diag(&mat, 10, 3, None).unwrap();

        assert!(res.eigenvalues.len() < 10, "expected an early breakdown");
        assert_eq!(res.eigenvalues.len(), res.eigenvectors[0].len());
        assert_relative_eq!(res.eigenvalues[0], 5.0, epsilon = 1e-9);
        assert_relative_eq!(res.eigenvalues[1], 3.0, epsilon = 1e-9);
    }

    /// A solve that ran out of restarts has to report itself as unconverged.
    #[test]
    fn test_lanczos_reports_budget_exhaustion() {
        // One cycle on a path graph cannot converge, and the caller has to be
        // able to tell that apart from a solve that did.
        let mat = path_graph(300);
        let params = LanczosParams {
            basis_size: None,
            max_restarts: 1,
            tol: 1e-8,
        };

        let res = compute_largest_eigenpairs_lanczos_diag(&mat, 4, 42, Some(params)).unwrap();

        assert!(!res.converged);
        assert_eq!(res.restarts, 1);
        assert!(res.residual > params.tol * res.norm_estimate);
    }

    /// Scaling the whole matrix must not change whether the solve converges.
    #[test]
    fn test_lanczos_tolerance_is_relative_to_the_matrix_norm() {
        // Scaling the matrix by 1e8 scales every residual by 1e8 too. An
        // absolute tolerance would make the scaled problem unconvergeable; a
        // relative one converges both in the same number of cycles.
        let small = diagonal(&[4.0, 3.0, 2.0, 1.0]);
        let large = diagonal(&[4e8, 3e8, 2e8, 1e8]);

        let a = compute_largest_eigenpairs_lanczos_diag(&small, 2, 5, None).unwrap();
        let b = compute_largest_eigenpairs_lanczos_diag(&large, 2, 5, None).unwrap();

        assert!(a.converged && b.converged);
        assert_relative_eq!(b.eigenvalues[0] / a.eigenvalues[0], 1e8, epsilon = 1e-1);
    }

    /// A basis smaller than the default still resolves the top eigenvalues via restarts.
    #[test]
    fn test_lanczos_params_basis_size_is_honoured() {
        let n = 40usize;
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            indices.push(i as u32);
            data.push((i + 1) as f64);
            indptr.push(indices.len() as u32);
        }
        let mat = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n));

        // Diagonal matrix: the largest eigenvalue is n regardless of the basis.
        let tight = LanczosParams {
            basis_size: Some(6),
            max_restarts: 200,
            tol: 1e-10,
        };
        let (values, _) = compute_largest_eigenpairs_lanczos(&mat, 3, 7, Some(tight)).unwrap();

        assert_relative_eq!(values[0], 40.0, epsilon = 1e-8);
        assert_relative_eq!(values[1], 39.0, epsilon = 1e-8);
        assert_relative_eq!(values[2], 38.0, epsilon = 1e-8);
    }

    ///////////////////////////
    // Row-stochastic + axpy //
    ///////////////////////////

    /// A row that sums to zero must come back as an error rather than a panic.
    #[test]
    fn test_normalise_csr_rows_l1_isolated_row_errors() {
        // Row 1 is structurally empty, row 2 stores an explicit zero. Both are
        // isolated as far as a row-stochastic normalisation is concerned.
        let data = vec![1.0f64, 3.0, 0.0];
        let indices = vec![0u32, 2, 1];
        let indptr = vec![0u32, 2, 2, 3];
        let mut csr = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (3, 3));

        match normalise_csr_rows_l1(&mut csr) {
            Err(BixverseErrors::SparseMatrixIsolatedRow { row, row_sum }) => {
                assert_eq!(row, 1);
                assert_eq!(row_sum, 0.0);
            }
            other => panic!("expected SparseMatrixIsolatedRow, got {:?}", other),
        }
    }

    /// The happy path has to leave every row summing to exactly one.
    #[test]
    fn test_normalise_csr_rows_l1_rows_sum_to_one() {
        let data = vec![1.0f64, 3.0, 2.0, 2.0, 5.0];
        let indices = vec![0u32, 2, 1, 2, 0];
        let indptr = vec![0u32, 2, 4, 5];
        let mut csr = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (3, 3));

        normalise_csr_rows_l1(&mut csr).unwrap();

        for i in 0..3 {
            let lo = csr.indptr[i] as usize;
            let hi = csr.indptr[i + 1] as usize;
            let sum: f64 = csr.data[lo..hi].iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        }
        assert_relative_eq!(csr.data[0], 0.25, epsilon = 1e-12);
        assert_relative_eq!(csr.data[1], 0.75, epsilon = 1e-12);
    }

    /// Build a small random CSR operator plus a dense block, both deterministic.
    fn dense_block_fixture(
        n: usize,
        k: usize,
        width: usize,
        seed: u64,
    ) -> (CompressedSparseData2<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indptr = vec![0u32];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();

        for _ in 0..n {
            // Ascending, unique column indices; some rows deliberately empty.
            for j in 0..k {
                if rng.random::<f64>() < 0.35 {
                    indices.push(j as u32);
                    data.push(rng.random::<f64>() * 2.0 - 1.0);
                }
            }
            indptr.push(indices.len() as u32);
        }

        let csr = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, k));
        let block: Vec<f64> = (0..k * width)
            .map(|_| rng.random::<f64>() * 2.0 - 1.0)
            .collect();

        (csr, block)
    }

    /// The SIMD dense-block kernel must agree with the obvious scalar triple
    /// loop, including on rows that carry no non-zeros at all.
    #[test]
    fn test_csr_matmul_dense_block_matches_scalar() {
        // Widths straddling the 2-, 4-, 8- and 16-lane boundaries so every
        // axpy tail path is hit.
        for width in [1usize, 3, 4, 7, 16, 17, 33] {
            let (csr, block) = dense_block_fixture(11, 9, width, 4242 + width as u64);

            let mut got = vec![0.0f64; 11 * width];
            csr_matmul_dense_block(&csr, &block, width, &mut got).unwrap();

            let mut want = vec![0.0f64; 11 * width];
            for i in 0..11 {
                for p in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                    let j = csr.indices[p] as usize;
                    for c in 0..width {
                        want[i * width + c] += csr.data[p] * block[j * width + c];
                    }
                }
            }

            for (g, w) in got.iter().zip(want.iter()) {
                assert_relative_eq!(g, w, epsilon = 1e-12, max_relative = 1e-12);
            }
        }
    }

    /// Applying the operator `t` times must equal multiplying by the explicit
    /// `T^t`. This is the whole justification for never materialising the power
    /// in MAGIC, so it is worth pinning at a size where the explicit cube is
    /// still legal to build.
    #[test]
    fn test_csr_matmul_dense_block_repeated_matches_explicit_power() {
        let width = 5usize;
        let n = 8usize;
        let (csr, block) = dense_block_fixture(n, n, width, 77);

        // Three applications, ping-ponging between two buffers.
        let mut buf_a = block.clone();
        let mut buf_b = vec![0.0f64; n * width];
        for _ in 0..3 {
            csr_matmul_dense_block(&csr, &buf_a, width, &mut buf_b).unwrap();
            std::mem::swap(&mut buf_a, &mut buf_b);
        }

        // The explicit cube, applied once.
        let sq = csr_matmul_csr(&csr, &csr).unwrap();
        let cube = csr_matmul_csr(&sq, &csr).unwrap();
        let mut direct = vec![0.0f64; n * width];
        csr_matmul_dense_block(&cube, &block, width, &mut direct).unwrap();

        for (rep, dir) in buf_a.iter().zip(direct.iter()) {
            assert_relative_eq!(rep, dir, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    /// A row-stochastic operator preserves a constant column, which is the
    /// invariant that makes imputed values sit on the input's scale.
    #[test]
    fn test_csr_matmul_dense_block_preserves_constant_column() {
        let n = 6usize;
        let width = 3usize;

        // Dense positive rows, so nothing is isolated, then row-normalise.
        let mut indptr = vec![0u32];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                indices.push(j as u32);
                data.push(1.0 + ((i * n + j) % 3) as f64);
            }
            indptr.push(indices.len() as u32);
        }
        let mut csr = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n));
        normalise_csr_rows_l1(&mut csr).unwrap();

        let block = vec![2.5f64; n * width];
        let mut out = vec![0.0f64; n * width];
        csr_matmul_dense_block(&csr, &block, width, &mut out).unwrap();

        for v in out {
            assert_relative_eq!(v, 2.5, epsilon = 1e-12);
        }
    }

    /// Shape mismatches and a CSC operator must be rejected, not read out of
    /// bounds.
    #[test]
    fn test_csr_matmul_dense_block_rejects_bad_shapes() {
        let (csr, block) = dense_block_fixture(4, 4, 2, 1);
        let mut out = vec![0.0f64; 4 * 2];

        // Wrong width for the block that was built.
        assert!(csr_matmul_dense_block(&csr, &block, 3, &mut out).is_err());
        // Zero width.
        assert!(csr_matmul_dense_block(&csr, &block, 0, &mut out).is_err());
        // Output too short.
        let mut short = vec![0.0f64; 4];
        assert!(csr_matmul_dense_block(&csr, &block, 2, &mut short).is_err());

        let csc = csr.transform();
        assert!(matches!(
            csr_matmul_dense_block(&csc, &block, 2, &mut out),
            Err(BixverseErrors::SparseMatrixMustBeCsr)
        ));
    }

    /// The sparse ranking must reproduce the dense one, up to the shift.
    #[test]
    fn test_shifted_ranks_sparse_matches_dense_rank_vector() {
        let cases: Vec<Vec<f32>> = vec![
            vec![3.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![-2.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            vec![-1.0, -3.0, 0.0, 2.0, 4.0, 0.0, 0.0],
            vec![2.0, 2.0, 0.0, 2.0, 0.0, 5.0],
            vec![-4.0, -4.0, 0.0, 0.0, -4.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, -3.0],
            vec![7.0],
        ];

        for dense in cases {
            let n = dense.len();
            let indices: Vec<u32> = (0..n as u32)
                .filter(|&i| dense[i as usize] != 0.0)
                .collect();
            let values: Vec<f32> = indices.iter().map(|&i| dense[i as usize]).collect();

            let got = shifted_ranks_sparse(&values, n);
            let dense_ranks = rank_vector(&dense);
            assert_eq!(got.len(), indices.len());

            if indices.is_empty() {
                continue;
            }

            // The ranks themselves must match the dense ones. Only up to a
            // constant, because the correlation is shift invariant and a fully
            // dense column has no zero block to pin the shift to.
            for (slot, &i) in indices.iter().enumerate() {
                assert_relative_eq!(
                    got[slot] - got[0],
                    dense_ranks[i as usize] - dense_ranks[indices[0] as usize],
                    epsilon = 1e-6
                );
            }

            // Once a structural zero exists the shift is pinned: it has to put
            // that whole tie group at exactly zero.
            if indices.len() < n {
                let r_zero = dense_ranks[(0..n).find(|&i| dense[i] == 0.0).unwrap()];
                for (slot, &i) in indices.iter().enumerate() {
                    assert_relative_eq!(
                        got[slot],
                        dense_ranks[i as usize] - r_zero,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// An explicitly stored zero must land on exactly the same shifted rank as
    /// a structural one, namely zero.
    ///
    /// This is the invariant the closed-form variance and the scatter/gather
    /// both stand on: every position the caller did not store contributes a
    /// known constant. A stored zero that ranked as its own tie group would
    /// break that silently, and CSC data from R does carry explicit zeros.
    #[test]
    fn test_shifted_ranks_sparse_merges_explicit_zeros() {
        // Six rows, four stored, one of them an explicit zero.
        let values: Vec<f32> = vec![-2.0, 0.0, 3.0, 1.0];
        let got = shifted_ranks_sparse(&values, 6);

        assert_relative_eq!(got[1], 0.0, epsilon = 1e-6);
        assert!(got[0] < 0.0, "the negative must rank below the zero block");
        assert!(got[2] > got[3], "3.0 must rank above 1.0");
        assert!(got[3] > 0.0, "positives must rank above the zero block");
    }
}
