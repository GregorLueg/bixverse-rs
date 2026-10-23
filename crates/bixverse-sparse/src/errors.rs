//! Errors raised by `bixverse-sparse`.

use thiserror::Error;

/// Error variants raised by the sparse structures and operations.
///
/// `bixverse-core` converts these into the identically named `CoreErrors`
/// variants, so callers of the core crate keep matching on `CoreErrors`.
#[derive(Debug, Error)]
pub enum SparseErrors {
    /// Error in situations were data_2 in
    /// [`CompressedSparseData2`](crate::sparse::CompressedSparseData2) is
    /// asked for but not available
    #[error("data_2 slot is None but was requested")]
    Data2NotAvailable,

    /// Error if the slice index is out of bounds
    #[error("The slice index ({index}) is out of bounds (length: {len})")]
    SliceIndexOutOfBounds {
        /// The chosen index
        index: usize,
        /// The actual length
        len: usize,
    },

    /// Error if the slice index is duplicated
    #[error("You provided a duplicated slice index {0}")]
    DuplicateSliceIndex(usize),

    /// Error for sparse multiplication dimension mismatches
    #[error(
        "The dimensions of the matrix do not support sparse multiplication (matrix a n_col: {n_col_a}; matrix b n_row: {n_row_b})"
    )]
    SparseMatrixMultiplication {
        /// Number of columns in matrix a
        n_col_a: usize,
        /// Number of rows in matrix b
        n_row_b: usize,
    },

    /// Dimension mismatch error during construction
    #[error("The indices (len: {indices_len}) and data (len: {data_len}) need to have same length")]
    DimensionMisMatchSparse {
        /// Length indices
        indices_len: usize,
        /// Length data
        data_len: usize,
    },

    /// Error if the `indptr` is inconsistent with the shape it declares or
    /// with the stored values.
    ///
    /// Raised by
    /// [`CompressedSparseData2::validate`](crate::sparse::CompressedSparseData2::validate)
    /// on input that crossed an FFI boundary, where the struct's public fields
    /// could have been populated with anything.
    #[error("The sparse indptr is inconsistent ({detail}): expected {expected}, got {got}")]
    SparseIndptrInvalid {
        /// Which invariant failed
        detail: &'static str,
        /// The value implied by the rest of the structure
        expected: usize,
        /// The value actually found
        got: usize,
    },

    /// Error if there is a dimension mismatch in terms of the two matrices
    #[error("The shape of the two sparse matrices must be the same.")]
    ShapeMismatchSparse,

    /// Error if the sparse matrix is not in CSR.
    #[error("The SparseCompressedData2 must be in CSR format")]
    SparseMatrixMustBeCsr,

    /// A row summed to (effectively) zero where a row-stochastic normalisation
    /// was asked for.
    ///
    /// On a kNN-derived affinity matrix every node has `k` out-edges, so a zero
    /// row means the kernel weights underflowed rather than that the graph is
    /// legitimately disconnected. Anything built on top of the operator would
    /// be meaningless, hence an error rather than leaving the row at zero.
    #[error(
        "Row {row} of the sparse matrix sums to {row_sum}, which is not positive, so it cannot be normalised to a row-stochastic form (isolated node, or the kernel weights underflowed)"
    )]
    SparseMatrixIsolatedRow {
        /// The offending row index
        row: usize,
        /// The row sum that was rejected
        row_sum: f64,
    },

    /// A CSR matrix failed one of its structural invariants.
    ///
    /// Raised by [`validate_square_csr`](crate::sparse::validate_square_csr).
    /// The static string names the invariant, because "the graph is malformed"
    /// alone leaves the caller to bisect their own input.
    #[error("The CSR matrix is malformed: {0}")]
    MalformedCsr(&'static str),

    /// Error if the sparse matrix is not in CSC.
    #[error("The SparseCompressedData2 must be in CSC format")]
    SparseMatrixMustBeCsc,

    /// Shape mismatch problem for matrices
    #[error("Expected this shape {expected:?}; got {got:?}.")]
    ShapeMismatch {
        /// Expected shape
        expected: (usize, usize),
        /// Provided shape
        got: (usize, usize),
    },
}
