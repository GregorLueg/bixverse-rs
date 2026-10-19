//! Errors raised by `bixverse-core`.

use crate::core::math::sparse::CompressedSparseFormat;
use ann_search_rs::utils::dist::Dist;
use thiserror::Error;

/// Error variants raised by `bixverse-core`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum CoreErrors {
    // -- arguments --
    /// General error for invalid arguments to a given function
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Error for wrong integer parameters
    #[error("Parameter {0} must be a positive non-zero integer.")]
    MustBePositive(String),

    // -- Math / Faer --
    /// SVD from faer failed to converge or returned no solution.
    ///
    /// Typically caused by ill-conditioned or degenerate input (e.g. all-zero
    /// rows, NaNs, rank-deficient matrices beyond the requested rank).
    #[error("The faer SVD failed: {0}")]
    FaerSvdError(String),

    /// Eigen decomposition from faer failed.
    ///
    /// Usually indicates a non-symmetric matrix being passed to a symmetric
    /// solver, or numerical breakdown on degenerate input.
    #[error("The faer Eigen decomposition failed - please verify the data")]
    FaerEigenError,

    /// Cholesky decomposition from faer failed.
    #[error("The faer Cholesky failed: {0}")]
    FaerCholeskyError(#[from] faer::linalg::solvers::LltError),

    // -- PCA with missing values --
    /// A row or column of the input to PPCA / BPCA holds no observed value.
    #[error("PCA with missing values: {axis} {index} is entirely missing; remove it first.")]
    PcaAllMissing {
        /// `"row"` or `"column"`
        axis: &'static str,
        /// Zero-based index of the offending row or column
        index: usize,
    },

    /// More components were requested than the data can hold.
    #[error("PCA: requested {n_pcs} components, but at most {max} are possible.")]
    PcaTooManyComponents {
        /// Requested number of components
        n_pcs: usize,
        /// `min(nrow, ncol)`
        max: usize,
    },

    // -- Kernel smoothing --
    /// A kernel bandwidth was requested for fewer than two points.
    #[error("Kernel bandwidth needs at least 2 data points, got {found}.")]
    BandwidthTooFewPoints {
        /// Number of points supplied
        found: usize,
    },

    /// A Sheather-Jones pilot functional came back non-finite or non-positive.
    ///
    /// R raises "sample is too sparse to find TD" / "... alph2" here. It means
    /// the pair-distance counts carry too little signal for the pilot estimate,
    /// which in practice is a degenerate or near-constant sample.
    #[error("Sheather-Jones bandwidth: sample too sparse to find {stage}.")]
    BandwidthTooSparse {
        /// Which pilot functional failed, `TD` or `alph2`
        stage: &'static str,
    },

    /// The Sheather-Jones bracket search never enclosed a root.
    #[error("Sheather-Jones bandwidth: no solution in the search interval.")]
    BandwidthNoBracket,

    /// Two vectors that had to agree in length did not.
    #[error("Length mismatch for '{name}': expected {expected}, got {found}.")]
    LengthMismatch {
        /// Name of the offending input
        name: &'static str,
        /// Length that was required
        expected: usize,
        /// Length that was supplied
        found: usize,
    },

    // -- Gaussian processes --
    /// A Gaussian process was handed empty training data.
    #[error("Gaussian process: {name} is empty")]
    GpEmptyInput {
        /// Which input was empty
        name: &'static str,
    },

    /// The training inputs and responses disagree in length.
    #[error(
        "Gaussian process: {n_x} training inputs against {n_y} response rows; the two must be aligned"
    )]
    GpDimensionMismatch {
        /// Length of the input vector
        n_x: usize,
        /// Rows in the response matrix
        n_y: usize,
    },

    /// A hyperparameter is outside its supported range.
    #[error("Gaussian process: hyperparameter '{name}' is {value}, which is not supported")]
    GpInvalidHyperparameter {
        /// Name of the offending hyperparameter
        name: &'static str,
        /// The value that was rejected
        value: f64,
    },

    /// A non-finite value reached the fit.
    ///
    /// Checked up front because a `NaN` otherwise surfaces as an unexplained
    /// Cholesky failure much later, with nothing pointing back at the input.
    #[error("Gaussian process: {name} contains a non-finite value")]
    GpNonFiniteInput {
        /// Which input carried the non-finite value
        name: &'static str,
    },

    /// The landmark Gram matrix stayed singular through the whole jitter ladder.
    #[error(
        "Gaussian process: the {n_landmarks} x {n_landmarks} landmark Gram is not positive definite even at a jitter of {jitter}; spread the landmarks out or shorten the length scale"
    )]
    GpLandmarkCholeskyFailed {
        /// Largest jitter that was tried
        jitter: f64,
        /// Number of landmarks
        n_landmarks: usize,
    },

    /// The posterior Cholesky failed.
    ///
    /// Distinct from [BixverseErrors::GpLandmarkCholeskyFailed] because this
    /// factorises `A A^T / sigma^2 + I`, whose smallest eigenvalue is at least
    /// one in exact arithmetic. A failure here means a non-finite value got
    /// through the input checks, not that the problem is ill-conditioned.
    #[error(
        "Gaussian process: the posterior Cholesky failed, which should be impossible for a matrix with unit diagonal dominance; the accumulation produced non-finite entries"
    )]
    GpPosteriorCholeskyFailed,

    /// Every landmark sits at the same coordinate.
    #[error(
        "Gaussian process: all landmarks share the same coordinate, so the kernel matrix is degenerate"
    )]
    GpDegenerateLandmarks,

    /// Every sample weight is zero.
    ///
    /// A zero weight drops its observation, so this leaves the fit with no data
    /// and returns the prior everywhere. Indistinguishable from a genuinely
    /// flat response, hence an error rather than a silent zero curve.
    #[error("Gaussian process: every sample weight is zero, so the fit would carry no data")]
    GpAllWeightsZero,

    // -- statrs --
    /// Beta distribution construction failed
    #[error("Beta distribution error: {0}")]
    BetaDistribution(String),

    // -- ann-search-rs --
    /// Propagate errors from the ann-search-rs crate
    #[error("Error from the ann-search-rs crate: {0}")]
    AnnSearchRsError(#[from] ann_search_rs::errors::AnnSearchErrors),

    // -- distances --
    /// Distance type not supported
    #[error("Distance metric '{0}' is not supported for this method.")]
    DistanceNotSupported(Dist),

    // -- matrix algebra errors --
    /// Error if the feature dimensions are not the same
    #[error("The feature dimensions between the two matrices are unequal! x: {dim_x}, y: {dim_y}")]
    NonMatchingFeatureDim {
        /// Feature dimensions matrix x
        dim_x: usize,
        /// Feature dimensions matrix y
        dim_y: usize,
    },

    /// Label length and number of cells do not match for Harmony
    #[error(
        "The labels length ({label_length}) does not match the number of samples ({n_samples})"
    )]
    NumberLabelsNotEqualSampleNumber {
        /// Provided label length
        label_length: usize,
        /// Number of cells
        n_samples: usize,
    },

    /// If the reference is out of range
    #[error("The provided reference is out of range")]
    ReferenceOutOfRange,

    // -- sparse erros --
    /// Error in situations were data_2 in [`crate::prelude::CompressedSparseData2`]
    /// is asked for but not available
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

    /// Error if the `indptr` of a [`crate::prelude::CompressedSparseData2`] is
    /// inconsistent with the shape it declares or with the stored values.
    ///
    /// Raised by [`crate::prelude::CompressedSparseData2::validate`] on input
    /// that crossed an FFI boundary, where the struct's public fields could
    /// have been populated with anything.
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

    /// Error if the [crate::prelude::CompressedSparseData2] is not in CSR.
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
    /// Raised by [crate::core::math::sparse::validate_square_csr]. The static
    /// string names the invariant, because "the graph is malformed" alone leaves
    /// the caller to bisect their own input.
    #[error("The CSR matrix is malformed: {0}")]
    MalformedCsr(&'static str),

    /// Error if the [crate::prelude::CompressedSparseData2] is not in Csc.
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

    /// General error for sparse matrix format mismatches
    #[error("Expected this compressed sparse format {expected}; got {got}.")]
    SparseLayoutMismatch {
        /// The expected [CompressedSparseFormat] enum.
        expected: CompressedSparseFormat,
        /// The provided [CompressedSparseFormat] enum.
        got: CompressedSparseFormat,
    },
}
