//! Sparse matrix formats, sparse operations and helpers to transform different
//! formats into each other. Free of any dense linear algebra dependency.

use num_traits::Float;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul};

use crate::errors::SparseErrors;
use crate::traits::BixverseNumeric;

/// Split `0..n` into one contiguous range per rayon worker.
///
/// For fan-outs where each worker carries scratch proportional to the whole
/// problem, one chunk per worker is what keeps the number of allocations and
/// merges down to the thread count. Rayon's own splitting is the better choice
/// whenever the per-item work is self-contained.
///
/// ### Params
///
/// * `n` - Length of the range to split
///
/// ### Returns
///
/// Half-open `(start, end)` ranges covering `0..n`. Empty when `n` is zero.
pub fn thread_chunks(n: usize) -> Vec<(usize, usize)> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(n_threads).max(1);

    (0..n)
        .step_by(chunk)
        .map(|start| (start, (start + chunk).min(n)))
        .collect()
}

// Compressed sparse structures //
//////////////////////////////////

/////////////////////////
// Structures: structs //
/////////////////////////

/// Type to describe the CompressedSparseFormat
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    pub fn get_indices_data_2(&self) -> Result<(&[usize], &[U]), SparseErrors> {
        let indices = &self.indices;
        let data_2 = self
            .data_2
            .as_ref()
            .ok_or(SparseErrors::Data2NotAvailable)?;

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
    pub fn from_vecs_to_csc(indices: Vec<T>, values: Vec<U>) -> Result<Self, SparseErrors>
    where
        T: Into<usize>,
    {
        if indices.len() != values.len() {
            return Err(SparseErrors::DimensionMisMatchSparse {
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

/// Structure to store compressed sparse data with two potential data layers.
/// Think for example in single cell raw counts and normalised counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// * `shape` - `(nrow, ncol)`.
    ///
    /// ### Returns
    ///
    /// Self as a CSC-type matrix
    pub fn new_csc(
        data: &[T],
        indices: &[u32],
        indptr: &[u32],
        data2: Option<&[U]>,
        shape: (usize, usize),
    ) -> Self {
        Self::from_parts(
            data.to_vec(),
            indices.to_vec(),
            indptr.to_vec(),
            data2.map(|d| d.to_vec()),
            CompressedSparseFormat::Csc,
            shape,
        )
    }

    /// Generate a nes CSR version of the matrix
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data
    /// * `indices` - The index positions (in this case row indices)
    /// * `indptr` - The index pointer (in this case the column index pointers)
    /// * `data2` - An optional second layer
    /// * `shape` - `(nrow, ncol)`.
    ///
    /// ### Returns
    ///
    /// Self as a CSR-type matrix
    pub fn new_csr(
        data: &[T],
        indices: &[u32],
        indptr: &[u32],
        data2: Option<&[U]>,
        shape: (usize, usize),
    ) -> Self {
        Self::from_parts(
            data.to_vec(),
            indices.to_vec(),
            indptr.to_vec(),
            data2.map(|d| d.to_vec()),
            CompressedSparseFormat::Csr,
            shape,
        )
    }

    /// Take ownership of buffers that were built for this matrix
    ///
    /// [CompressedSparseData2::new_csr] and its CSC sibling copy all three
    /// buffers, which doubles peak memory for any caller that assembled them
    /// itself and then throws the originals away.
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
    pub fn transform_single_layer(&self, use_second_layer: bool) -> Result<Self, SparseErrors> {
        transpose_sparse_single_layer(self, use_second_layer)
    }

    /// Transpose and convert
    ///
    /// ### Returns
    ///
    /// The transformed/transposed version
    pub fn transpose_and_convert(&self) -> Self {
        // A CSC matrix reads as the CSR of its transpose with no data movement.
        // A CSR matrix is converted first, then relabelled the same way.
        let mut out = match self.cs_type {
            CompressedSparseFormat::Csr => transpose_sparse(self),
            CompressedSparseFormat::Csc => self.clone(),
        };
        out.cs_type = CompressedSparseFormat::Csr;
        out.shape = (self.shape.1, self.shape.0);
        out
    }

    /// Create a sparse matrix from an upper triangular matrix.
    ///
    /// ### Params
    ///
    /// * `upper_triangle` - The upper triangular matrix.
    /// * `n` - The number of rows and columns in the matrix.
    /// * `include_diagonal` - Whether to include the diagonal elements.
    /// * `format` - The format of the sparse matrix, see
    ///   [CompressedSparseFormat].
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
        T: BixverseNumeric + Float,
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

        Self::from_entry_fn(n, n, format, get_value)
    }

    /// Returns the shape of the matrix
    ///
    /// ### Returns
    ///
    /// A tuple of `(n_row, n_col)`
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

    /// Split into one owned [`SparseAxis`] per major-axis run.
    ///
    /// Columns of a CSC or rows of a CSR. A layer that is empty (not
    /// populated) stays empty on every axis.
    ///
    /// ### Returns
    ///
    /// The axes in order, each sized to the minor dimension.
    pub fn to_sparse_axes(&self) -> Vec<SparseAxis<T, U>> {
        let len = self.minor_dim();
        let has_data = self.data.len() == self.indices.len();

        self.indptr
            .par_windows(2)
            .map(|w| {
                let (lo, hi) = (w[0] as usize, w[1] as usize);
                SparseAxis {
                    indices: self.indices[lo..hi].iter().map(|&i| i as usize).collect(),
                    data: if has_data {
                        self.data[lo..hi].to_vec()
                    } else {
                        Vec::new()
                    },
                    data_2: self.data_2.as_ref().map(|d| d[lo..hi].to_vec()),
                    cs_type: self.cs_type,
                    len,
                }
            })
            .collect()
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
    fn slice_major(&self, keep: &[usize]) -> Result<Self, SparseErrors> {
        let major = self.major_dim();
        let mut seen = vec![false; major];

        let mut indptr = Vec::with_capacity(keep.len() + 1);
        indptr.push(0u32);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut data_2 = self.data_2.as_ref().map(|_| Vec::new());

        for &m in keep {
            if m >= major {
                return Err(SparseErrors::SliceIndexOutOfBounds {
                    index: m,
                    len: major,
                });
            }
            if seen[m] {
                return Err(SparseErrors::DuplicateSliceIndex(m));
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
    fn slice_minor(&self, keep: &[usize]) -> Result<Self, SparseErrors> {
        let minor = self.minor_dim();
        let mut map: Vec<Option<usize>> = vec![None; minor];
        for (new_idx, &old) in keep.iter().enumerate() {
            if old >= minor {
                return Err(SparseErrors::SliceIndexOutOfBounds {
                    index: old,
                    len: minor,
                });
            }
            if map[old].is_some() {
                return Err(SparseErrors::DuplicateSliceIndex(old));
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
    pub fn slice_rows(&self, rows: &[usize]) -> Result<Self, SparseErrors> {
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
    pub fn slice_cols(&self, cols: &[usize]) -> Result<Self, SparseErrors> {
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
    pub fn slice(&self, rows: &[usize], cols: &[usize]) -> Result<Self, SparseErrors> {
        self.slice_rows(rows)?.slice_cols(cols)
    }

    /////////////
    // Helpers //
    /////////////

    /// Build a matrix by scanning every `(row, col)` entry of a virtual dense
    /// matrix and storing the non-zeros.
    ///
    /// ### Params
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `format` - Format to generate; the scan runs along its major axis
    /// * `get` - Value at `(row, col)`
    ///
    /// ### Returns
    ///
    /// The sparse matrix, with no second layer.
    pub fn from_entry_fn<G>(
        nrows: usize,
        ncols: usize,
        format: CompressedSparseFormat,
        get: G,
    ) -> Self
    where
        T: BixverseNumeric + Float,
        G: Fn(usize, usize) -> T,
    {
        let (major, minor) = match format {
            CompressedSparseFormat::Csr => (nrows, ncols),
            CompressedSparseFormat::Csc => (ncols, nrows),
        };

        let mut data = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = Vec::with_capacity(major + 1);
        indptr.push(0);

        for outer in 0..major {
            for inner in 0..minor {
                let val = match format {
                    CompressedSparseFormat::Csr => get(outer, inner),
                    CompressedSparseFormat::Csc => get(inner, outer),
                };
                if val != T::zero() {
                    data.push(val);
                    indices.push(inner as u32);
                }
            }
            indptr.push(data.len() as u32);
        }

        Self::from_parts(data, indices, indptr, None, format, (nrows, ncols))
    }

    /// Return the major dimensions
    ///
    /// ### Returns
    ///
    /// Major dimension
    fn major_dim(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Major dimension as declared by `shape`, which `indptr` must agree with.
    ///
    /// ### Returns
    ///
    /// Rows for CSR, columns for CSC.
    fn expected_major(&self) -> usize {
        match self.cs_type {
            CompressedSparseFormat::Csr => self.shape.0,
            CompressedSparseFormat::Csc => self.shape.1,
        }
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
    /// [`SparseErrors::SparseIndptrInvalid`] for an `indptr` that disagrees
    /// with `shape` or with the number of stored values,
    /// [`SparseErrors::DimensionMisMatchSparse`] for a populated layer whose
    /// length does not match `indices`, and
    /// [`SparseErrors::SliceIndexOutOfBounds`] for an index past the minor
    /// axis.
    pub fn validate(&self) -> Result<(), SparseErrors> {
        let expected_major = self.expected_major();

        if self.indptr.len() != expected_major + 1 {
            return Err(SparseErrors::SparseIndptrInvalid {
                detail: "length against the declared major axis",
                expected: expected_major + 1,
                got: self.indptr.len(),
            });
        }

        // `indptr.len()` is now at least 1, so `last` cannot be `None`
        let nnz = *self.indptr.last().expect("indptr length checked above") as usize;
        if nnz != self.indices.len() {
            return Err(SparseErrors::SparseIndptrInvalid {
                detail: "final offset against the stored indices",
                expected: self.indices.len(),
                got: nnz,
            });
        }

        if !self.indptr.windows(2).all(|w| w[0] <= w[1]) {
            return Err(SparseErrors::SparseIndptrInvalid {
                detail: "offsets must be non-decreasing",
                expected: nnz,
                got: nnz,
            });
        }

        if !self.data.is_empty() && self.data.len() != self.indices.len() {
            return Err(SparseErrors::DimensionMisMatchSparse {
                indices_len: self.indices.len(),
                data_len: self.data.len(),
            });
        }

        if let Some(data_2) = &self.data_2
            && !data_2.is_empty()
            && data_2.len() != self.indices.len()
        {
            return Err(SparseErrors::DimensionMisMatchSparse {
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
            return Err(SparseErrors::SliceIndexOutOfBounds {
                index: bad as usize,
                len: minor,
            });
        }

        Ok(())
    }

    /// Assertion helper for invariants
    pub fn assert_invariants(&self) {
        debug_assert_eq!(self.indices.len(), self.data.len());
        debug_assert_eq!(self.indptr.len(), self.expected_major() + 1);
        debug_assert_eq!(*self.indptr.last().unwrap() as usize, self.data.len());
        if let Some(d2) = &self.data_2 {
            debug_assert_eq!(d2.len(), self.data.len());
        }
    }
}

/////////////////////////
// Structures: helpers //
/////////////////////////

/// Fail unless the matrix is stored as CSR.
///
/// ### Params
///
/// * `mat` - The matrix to check
///
/// ### Returns
///
/// `Ok(())`, or [SparseErrors::SparseMatrixMustBeCsr].
pub fn require_csr<T, U>(mat: &CompressedSparseData2<T, U>) -> Result<(), SparseErrors>
where
    T: Clone,
    U: Clone,
{
    if mat.cs_type.is_csr() {
        Ok(())
    } else {
        Err(SparseErrors::SparseMatrixMustBeCsr)
    }
}

/// Fail unless the matrix is stored as CSC.
///
/// ### Params
///
/// * `mat` - The matrix to check
///
/// ### Returns
///
/// `Ok(())`, or [SparseErrors::SparseMatrixMustBeCsc].
pub fn require_csc<T, U>(mat: &CompressedSparseData2<T, U>) -> Result<(), SparseErrors>
where
    T: Clone,
    U: Clone,
{
    if mat.cs_type.is_csc() {
        Ok(())
    } else {
        Err(SparseErrors::SparseMatrixMustBeCsc)
    }
}

/// Fail unless both matrices are CSR and share a shape, as the element-wise
/// operations need.
///
/// ### Params
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// ### Returns
///
/// `Ok(())`, [SparseErrors::SparseMatrixMustBeCsr] or
/// [SparseErrors::ShapeMismatchSparse].
fn require_csr_pair_same_shape<T>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> Result<(), SparseErrors>
where
    T: Clone,
{
    require_csr(a)?;
    require_csr(b)?;
    if a.shape != b.shape {
        return Err(SparseErrors::ShapeMismatchSparse);
    }
    Ok(())
}

/// Turn per-run counts into offsets, in place.
///
/// ### Params
///
/// * `indptr` - Counts shifted by one, i.e. `indptr[i + 1]` holds the count of
///   run `i` and `indptr[0]` is zero. Becomes the usual `indptr`.
fn cumulate_indptr(indptr: &mut [u32]) {
    for i in 1..indptr.len() {
        indptr[i] += indptr[i - 1];
    }
}

//////////////////////////////////
// Structures: public functions //
//////////////////////////////////

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
pub fn validate_square_csr<T>(graph: &CompressedSparseData2<T>) -> Result<usize, SparseErrors>
where
    T: Clone,
{
    if !graph.cs_type.is_csr() {
        return Err(SparseErrors::SparseMatrixMustBeCsr);
    }
    let (rows, cols) = graph.shape;
    if rows != cols {
        return Err(SparseErrors::ShapeMismatch {
            expected: (rows, rows),
            got: (rows, cols),
        });
    }

    if graph.indptr.len() != rows + 1 {
        return Err(SparseErrors::MalformedCsr(
            "indptr length must be the row count plus one",
        ));
    }
    if graph.indices.len() != graph.data.len() {
        return Err(SparseErrors::MalformedCsr(
            "indices and data must have the same length",
        ));
    }
    if graph.indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(SparseErrors::MalformedCsr("indptr must be non-decreasing"));
    }
    // `indptr` is non-empty by the length check above.
    if graph.indptr[rows] as usize != graph.indices.len() {
        return Err(SparseErrors::MalformedCsr(
            "the last indptr entry must equal the number of stored values",
        ));
    }
    if graph.indices.iter().any(|&j| j as usize >= rows) {
        return Err(SparseErrors::MalformedCsr(
            "a column index sits outside the matrix",
        ));
    }

    Ok(rows)
}

///////////////////////
// Format conversion //
///////////////////////

////////////////////////
// Conversion: consts //
////////////////////////

/// Below this many stored values the transpose stays single-threaded.
///
/// The parallel path pays for `n_threads * new_major` counters plus two sweeps
/// over them, which only earns its keep once the scatter itself is large enough
/// to leave cache. A megabyte of stored values is comfortably past that point
/// on every machine tested and well short of it for the small graphs and toy
/// matrices most callers hand over.
const PARALLEL_TRANSPOSE_MIN_NNZ: usize = 1 << 20;

/// Width of the band of the new major axis the scatter fills at a time.
///
/// Unbanded, each chunk revisits every new major run once per old major index,
/// so a short, wide input (2000 genes by 1M cells) streams its whole
/// destination through the cache once per gene. Banding keeps each chunk's
/// writes inside `band * nnz / new_major` slots. Measured on a 1M x 2000 CSC
/// at 10% density (M1 Max, 10 threads): 633 ms unbanded, 194-207 ms for bands
/// of 512 to 4096, 280 ms at 16384, 490 ms at 65536.
const TRANSPOSE_MINOR_BAND: usize = 1024;

/////////////////////////
// Conversion: structs //
/////////////////////////

/// Destination pointer handed to every worker of the parallel scatter.
///
/// The transpose partitions the destination by `(chunk, minor)`: the counting
/// pass gives every pair its own half-open run of slots, and a worker only ever
/// advances the cursors of its own chunk. No two workers therefore address the
/// same slot, which is what makes the shared `*mut` sound.
pub struct ScatterPtr<T>(pub *mut T);

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
    pub unsafe fn write(&self, pos: usize, value: T) {
        unsafe { *self.0.add(pos) = value }
    }
}

/////////////////////////
// Conversion: helpers //
/////////////////////////

/// Length of the major axis and the format of the other-format twin.
///
/// The old minor axis becomes the new major axis, so a CSC matrix of `nrow`
/// rows converts into a CSR one with `nrow` runs and vice versa.
///
/// ### Params
///
/// * `sparse_data` - The matrix being converted
///
/// ### Returns
///
/// `(new_major, new_type)`.
fn converted_layout<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
) -> (usize, CompressedSparseFormat)
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let new_type = match sparse_data.cs_type {
        CompressedSparseFormat::Csc => CompressedSparseFormat::Csr,
        CompressedSparseFormat::Csr => CompressedSparseFormat::Csc,
    };
    (sparse_data.minor_dim(), new_type)
}

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
/// `(new_indptr, cursors, sorted)`, where `cursors` is
/// `chunks.len() * new_major` long and laid out chunk-major, and `sorted`
/// says whether every old major run has strictly ascending minor indices.
fn transpose_cursors<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
    chunks: &[(usize, usize)],
    new_major: usize,
) -> (Vec<u32>, Vec<u32>, bool)
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let mut cursors = vec![0u32; chunks.len() * new_major];

    let sorted = chunks
        .par_iter()
        .zip(cursors.par_chunks_mut(new_major.max(1)))
        .map(|(&(start, end), counts)| {
            let mut sorted = true;
            for major in start..end {
                let lo = sparse_data.indptr[major] as usize;
                let hi = sparse_data.indptr[major + 1] as usize;
                let run = &sparse_data.indices[lo..hi];
                for &minor in run {
                    counts[minor as usize] += 1;
                }
                sorted &= run.windows(2).all(|w| w[0] < w[1]);
            }
            sorted
        })
        .reduce(|| true, |a, b| a && b);

    // column sums of the histograms give the entries per new-major index
    let mut new_indptr = vec![0u32; new_major + 1];
    for counts in cursors.chunks(new_major.max(1)) {
        for (total, &count) in new_indptr[1..].iter_mut().zip(counts.iter()) {
            *total += count;
        }
    }
    cumulate_indptr(&mut new_indptr);

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

    (new_indptr, cursors, sorted)
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
/// * `band` - Width of the band of the new major axis each chunk scatters
///   into at a time; `None` walks every run once, front to back. Banding
///   needs strictly ascending minor indices within each old major run.
#[allow(clippy::too_many_arguments)]
fn transpose_scatter<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
    chunks: &[(usize, usize)],
    cursors: &mut [u32],
    new_major: usize,
    band: Option<usize>,
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
            // read position in each old major run, advanced band by band
            let mut read: Vec<usize> = (start..end)
                .map(|major| sparse_data.indptr[major] as usize)
                .collect();
            let band = band.unwrap_or(usize::MAX).max(1);
            let mut band_lo = 0usize;
            while band_lo < new_major.max(1) {
                let band_hi = band_lo.saturating_add(band);
                for (major, rd) in (start..end).zip(read.iter_mut()) {
                    let hi = sparse_data.indptr[major + 1] as usize;
                    while *rd < hi {
                        let idx = *rd;
                        let minor = sparse_data.indices[idx] as usize;
                        if minor >= band_hi {
                            break;
                        }
                        *rd += 1;
                        let pos = cursor[minor] as usize;
                        cursor[minor] += 1;

                        // SAFETY: `pos` sits in the run this chunk owns for
                        // `minor`, which no other worker touches, and every
                        // run is inside `0..nnz` by the counting pass.
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
                band_lo = band_hi;
            }
        });
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
/// * `band` - Scatter band width, see [`TRANSPOSE_MINOR_BAND`]; ignored if a
///   run has unsorted minor indices
///
/// ### Returns
///
/// The transposed compressed sparse matrix.
fn transpose_sparse_chunked<T, U>(
    sparse_data: &CompressedSparseData2<T, U>,
    chunks: &[(usize, usize)],
    band: usize,
) -> CompressedSparseData2<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let nnz = sparse_data.get_nnz();
    let (nrow, ncol) = sparse_data.shape();
    let (new_major, new_type) = converted_layout(sparse_data);

    let mut new_data: Vec<T> = vec![T::default(); nnz];
    let mut new_indices: Vec<u32> = vec![0u32; nnz];
    let mut new_data2: Option<Vec<U>> =
        sparse_data.data_2.as_ref().map(|_| vec![U::default(); nnz]);

    let (new_indptr, mut cursors, sorted) = transpose_cursors(sparse_data, chunks, new_major);

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
        sorted.then_some(band),
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

//////////////////////////////////
// Conversion: public functions //
//////////////////////////////////

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
    transpose_sparse_chunked(
        sparse_data,
        &major_axis_chunks(sparse_data),
        TRANSPOSE_MINOR_BAND,
    )
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
) -> Result<CompressedSparseData2<T, U>, SparseErrors>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    let nnz = sparse_data.get_nnz();
    let (nrow, ncol) = sparse_data.shape();
    let (new_major, new_type) = converted_layout(sparse_data);

    let mut new_indices: Vec<u32> = vec![0u32; nnz];

    // allocate only for the kept layer
    let (mut new_data, mut new_data2): (Vec<T>, Option<Vec<U>>) = if use_second_layer {
        (Vec::new(), Some(vec![U::default(); nnz]))
    } else {
        (vec![T::default(); nnz], None)
    };

    let chunks = major_axis_chunks(sparse_data);
    let (new_indptr, mut cursors, sorted) = transpose_cursors(sparse_data, &chunks, new_major);

    let (data_pair, data2_pair) = if use_second_layer {
        let src = sparse_data
            .data_2
            .as_ref()
            .ok_or(SparseErrors::Data2NotAvailable)?
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
        sorted.then_some(TRANSPOSE_MINOR_BAND),
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

    cumulate_indptr(&mut indptr);

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
    }

    cumulate_indptr(&mut indptr);

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
/// Every other consumer in the crate assumes ascending deduplicated rows, so
/// the invariants are `debug_assert!`ed here and the function stays
/// crate-private.
///
/// ### Params
///
/// * `n` - Node count; the output is `n` square.
/// * `edges` - Deduplicated `(lo, hi, value)` triples with `lo < hi < n`,
///   sorted strictly ascending by `(lo, hi)`.
///
/// ### Returns
///
/// `CompressedSparseData2` in CSR format with both directions of every edge
/// stored at the same value.
pub fn undirected_edges_to_csr<T>(n: usize, edges: &[(u32, u32, T)]) -> CompressedSparseData2<T>
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
    cumulate_indptr(&mut indptr);

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
// Sparse arithmetic //
///////////////////////

////////////////////////
// Arithmetic: consts //
////////////////////////

/// Rows merged per rayon job in [merge_csr_rows]. Not tuned; large enough that
/// a job outweighs the scheduling cost, small enough to balance skewed rows.
const SPARSE_MERGE_ROW_CHUNK: usize = 1024;

/// Magnitude at or below which an overlap of two merged entries counts as
/// cancelled and is dropped. Stored as `f32` to keep the threshold the
/// callers have always seen.
const SPARSE_MERGE_EPSILON: f32 = 1e-9;

/// Output rows per rayon job in [csr_matmul_csr]. Each job reuses one
/// [SparseAccumulator] of the output width, so this also bounds how often the
/// accumulator is re-created. Not tuned.
const SPARSE_MATMUL_ROW_CHUNK: usize = 256;

/// Row or column sum at or below which the L1 normalisation treats the slice as
/// empty, to avoid dividing by something indistinguishable from zero.
const NORMALISE_MIN_SUM: f64 = 1e-15;

/////////////////////////
// Arithmetic: structs //
/////////////////////////

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

/////////////////////////
// Arithmetic: helpers //
/////////////////////////

/// Check that two CSR matrices can be multiplied, `a @ b`.
///
/// ### Params
///
/// * `a` - Left matrix
/// * `b` - Right matrix
///
/// ### Returns
///
/// `Ok(())`, [SparseErrors::SparseMatrixMustBeCsr] or
/// [SparseErrors::SparseMatrixMultiplication] if `a` has a different number of
/// columns than `b` has rows.
pub fn require_matmul_operands<T>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> Result<(), SparseErrors>
where
    T: Clone,
{
    require_csr(a)?;
    require_csr(b)?;

    let (ncol_a, nrow_b) = (a.shape.1, b.shape.0);
    if ncol_a != nrow_b {
        return Err(SparseErrors::SparseMatrixMultiplication {
            n_col_a: ncol_a,
            n_row_b: nrow_b,
        });
    }
    Ok(())
}

/// Merge two CSR matrices row by row over the union of their patterns.
///
/// Rows are merged in parallel chunks of [SPARSE_MERGE_ROW_CHUNK]. An entry
/// present in only one matrix is carried over (through `lone_b` for `b`); an
/// overlap is combined and dropped if the result has magnitude at or below
/// [SPARSE_MERGE_EPSILON]. The per-chunk outputs are stitched into one CSR with
/// ascending columns per row, assuming both inputs have them.
///
/// ### Params
///
/// * `a` - First matrix, CSR, same shape as `b`
/// * `b` - Second matrix, CSR, same shape as `a`
/// * `combine` - Value of an overlap, given the entries of `a` and `b`
/// * `lone_b` - Value of an entry that only `b` stores
///
/// ### Returns
///
/// The merged matrix in CSR format.
fn merge_csr_rows<T, C, N>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
    combine: C,
    lone_b: N,
) -> CompressedSparseData2<T>
where
    T: BixverseNumeric + Into<f64>,
    C: Fn(T, T) -> T + Sync,
    N: Fn(T) -> T + Sync,
{
    let n_rows = a.shape.0;

    let chunks: Vec<(Vec<u32>, Vec<T>, Vec<u32>)> = (0..n_rows.div_ceil(SPARSE_MERGE_ROW_CHUNK))
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * SPARSE_MERGE_ROW_CHUNK;
            let end = (start + SPARSE_MERGE_ROW_CHUNK).min(n_rows);
            let cap = (a.indptr[end] - a.indptr[start] + b.indptr[end] - b.indptr[start]) as usize;
            let mut cols: Vec<u32> = Vec::with_capacity(cap);
            let mut vals: Vec<T> = Vec::with_capacity(cap);
            let mut lengths: Vec<u32> = Vec::with_capacity(end - start);

            for i in start..end {
                let before = cols.len();
                let a_end = a.indptr[i + 1] as usize;
                let b_end = b.indptr[i + 1] as usize;
                let mut a_idx = a.indptr[i] as usize;
                let mut b_idx = b.indptr[i] as usize;

                while a_idx < a_end && b_idx < b_end {
                    let a_col = a.indices[a_idx];
                    let b_col = b.indices[b_idx];
                    if a_col < b_col {
                        cols.push(a_col);
                        vals.push(a.data[a_idx]);
                        a_idx += 1;
                    } else if b_col < a_col {
                        cols.push(b_col);
                        vals.push(lone_b(b.data[b_idx]));
                        b_idx += 1;
                    } else {
                        let val = combine(a.data[a_idx], b.data[b_idx]);
                        if val.into().abs() > SPARSE_MERGE_EPSILON as f64 {
                            cols.push(a_col);
                            vals.push(val);
                        }
                        a_idx += 1;
                        b_idx += 1;
                    }
                }
                cols.extend_from_slice(&a.indices[a_idx..a_end]);
                vals.extend_from_slice(&a.data[a_idx..a_end]);
                cols.extend_from_slice(&b.indices[b_idx..b_end]);
                vals.extend(b.data[b_idx..b_end].iter().map(|&v| lone_b(v)));

                lengths.push((cols.len() - before) as u32);
            }

            (cols, vals, lengths)
        })
        .collect();

    let nnz: usize = chunks.iter().map(|(c, _, _)| c.len()).sum();
    let mut indptr: Vec<u32> = Vec::with_capacity(n_rows + 1);
    let mut indices: Vec<u32> = Vec::with_capacity(nnz);
    let mut data: Vec<T> = Vec::with_capacity(nnz);
    indptr.push(0);
    for (cols, vals, lengths) in chunks {
        let mut running = *indptr.last().unwrap_or(&0);
        for len in lengths {
            running += len;
            indptr.push(running);
        }
        indices.extend_from_slice(&cols);
        data.extend_from_slice(&vals);
    }

    CompressedSparseData2::from_parts(
        data,
        indices,
        indptr,
        None,
        CompressedSparseFormat::Csr,
        a.shape,
    )
}

//////////////////////////////////
// Arithmetic: public functions //
//////////////////////////////////

/// Add two CSR matrices together
///
/// See [merge_csr_rows] for the merge itself; an overlap whose sum has
/// magnitude at or below [SPARSE_MERGE_EPSILON] is dropped.
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
) -> Result<CompressedSparseData2<T>, SparseErrors>
where
    T: BixverseNumeric + Into<f64> + Add<Output = T>,
{
    require_csr_pair_same_shape(a, b)?;
    Ok(merge_csr_rows(a, b, |x, y| x + y, |y| y))
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
/// `a - b`, merged as for [sparse_add_csr]; an overlap whose difference has
/// magnitude at or below [SPARSE_MERGE_EPSILON] is dropped.
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
) -> Result<CompressedSparseData2<T>, SparseErrors>
where
    T: BixverseNumeric + Into<f64>,
{
    require_csr_pair_same_shape(a, b)?;
    Ok(merge_csr_rows(a, b, |x, y| x - y, |y| T::default() - y))
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
) -> Result<CompressedSparseData2<T>, SparseErrors>
where
    T: BixverseNumeric,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    require_csr_pair_same_shape(a, b)?;

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
        if sum.into() > NORMALISE_MIN_SUM {
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
/// `Ok(())`, or [SparseErrors::SparseMatrixMustBeCsr] if the matrix is in
/// CSC, or [SparseErrors::SparseMatrixIsolatedRow] naming the first row that
/// sums to zero.
pub fn normalise_csr_rows_l1<T>(csr: &mut CompressedSparseData2<T>) -> Result<(), SparseErrors>
where
    T: BixverseNumeric + Into<f64>,
    T: std::iter::Sum<T>,
{
    require_csr(csr)?;

    let nrows = csr.shape.0;

    for i in 0..nrows {
        let start = csr.indptr[i] as usize;
        let end = csr.indptr[i + 1] as usize;
        let row_data_slice = &mut csr.data[start..end];

        let row_sum: T = row_data_slice.iter().copied().sum();

        if row_sum.into() <= NORMALISE_MIN_SUM {
            return Err(SparseErrors::SparseMatrixIsolatedRow {
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
pub fn csr_matvec<T>(mat: &CompressedSparseData2<T>, vec: &[T]) -> Result<Vec<T>, SparseErrors>
where
    T: BixverseNumeric,
    <T as std::ops::Add>::Output: std::cmp::PartialEq<T>,
{
    require_csr(mat)?;

    let mut result = vec![T::default(); mat.shape.0];
    for i in 0..mat.shape.0 {
        let row_start = mat.indptr[i] as usize;
        let row_end = mat.indptr[i + 1] as usize;
        let data = &mat.data[row_start..row_end];
        let cols = &mat.indices[row_start..row_end];
        let mut acc = [T::default(); 4];
        let mut data_chunks = data.chunks_exact(4);
        let mut col_chunks = cols.chunks_exact(4);
        for (d, c) in (&mut data_chunks).zip(&mut col_chunks) {
            for l in 0..4 {
                acc[l] += d[l] * vec[c[l] as usize];
            }
        }
        let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for (&d, &c) in data_chunks.remainder().iter().zip(col_chunks.remainder()) {
            sum += d * vec[c as usize];
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
pub fn csr_vecmat<T>(vec: &[T], mat: &CompressedSparseData2<T>) -> Result<Vec<T>, SparseErrors>
where
    T: BixverseNumeric,
{
    require_csr(mat)?;

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
) -> Result<CompressedSparseData2<T>, SparseErrors>
where
    T: BixverseNumeric,
{
    require_matmul_operands(a, b)?;

    let nrows = a.shape.0;
    let ncols = b.shape.1;

    let n_chunks = nrows.div_ceil(SPARSE_MATMUL_ROW_CHUNK);

    let chunks: Vec<(Vec<usize>, Vec<T>, Vec<usize>)> = (0..n_chunks)
        .into_par_iter()
        .map_init(
            || SparseAccumulator::new(ncols),
            |acc, chunk_idx| {
                let start = chunk_idx * SPARSE_MATMUL_ROW_CHUNK;
                let end = ((chunk_idx + 1) * SPARSE_MATMUL_ROW_CHUNK).min(nrows);

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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
            Err(SparseErrors::SparseIndptrInvalid {
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
            Err(SparseErrors::SparseIndptrInvalid {
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
            Err(SparseErrors::SparseIndptrInvalid { .. })
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
            Err(SparseErrors::SliceIndexOutOfBounds { index: 4, len: 4 })
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
            Err(SparseErrors::SliceIndexOutOfBounds { index: 3, len: 3 })
        ));
    }

    #[test]
    fn test_validate_rejects_a_truncated_second_layer() {
        let mut m = validatable_csr();
        m.data_2 = Some(vec![10.0, 20.0]);

        assert!(matches!(
            m.validate(),
            Err(SparseErrors::DimensionMisMatchSparse {
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

        let serial = transpose_sparse_chunked(&csr, &[(0, 5)], usize::MAX);

        for (chunks, band) in [
            (vec![(0, 1), (1, 3), (3, 5)], usize::MAX),
            (vec![(0, 2), (2, 5)], usize::MAX),
            (vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], usize::MAX),
            (vec![(0, 5)], 1),
            (vec![(0, 2), (2, 5)], 1),
            (vec![(0, 1), (1, 3), (3, 5)], 2),
            (vec![(0, 2), (2, 5)], 3),
            (vec![(0, 2), (2, 5)], 4),
        ] {
            let parallel = transpose_sparse_chunked(&csr, &chunks, band);

            assert_eq!(parallel.indptr, serial.indptr);
            assert_eq!(parallel.indices, serial.indices);
            assert_eq!(parallel.data, serial.data);
            assert_eq!(parallel.data_2, serial.data_2);
            assert_eq!(parallel.cs_type.is_csc(), serial.cs_type.is_csc());
            assert_eq!(parallel.shape, serial.shape);
        }
    }

    /// A run with unsorted minor indices cannot be banded; the scatter falls
    /// back to the unbanded walk and still places every entry.
    #[test]
    fn test_transpose_unsorted_runs_ignore_band() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices: Vec<u32> = vec![2, 0, 3, 1, 0];
        let indptr: Vec<u32> = vec![0, 2, 4, 5];
        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, (3, 4));

        let csc = transpose_sparse_chunked(&csr, &[(0, 1), (1, 3)], 1);
        assert_eq!(csc.indptr, vec![0, 2, 3, 4, 5]);
        assert_eq!(csc.indices, vec![0, 2, 1, 0, 1]);
        assert_eq!(csc.data, vec![2.0, 5.0, 4.0, 1.0, 3.0]);
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
            Err(SparseErrors::MalformedCsr(_))
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
            Err(SparseErrors::MalformedCsr(_))
        ));

        // indptr of the wrong length for the declared row count.
        let short_indptr =
            CompressedSparseData2::new_csr(&[1.0f64], &[1u32], &[0u32, 1], None, (3, 3));
        assert!(matches!(
            validate_square_csr(&short_indptr),
            Err(SparseErrors::MalformedCsr(_))
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
            Err(SparseErrors::MalformedCsr(_))
        ));
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
            Err(SparseErrors::SparseMatrixIsolatedRow { row, row_sum }) => {
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
}
