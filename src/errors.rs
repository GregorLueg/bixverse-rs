//! Errors in bixverse. Contains all the various errors that can be returned by
//! the crate.

use ann_search_rs::utils::dist::Dist;
#[cfg(feature = "single-cell")]
use std::io;
use thiserror::Error;

use crate::prelude::*;

/// All error variants that can occur across bixverse operations.
///
/// Errors are grouped by subsystem: faer-backed linear algebra, binary file
/// I/O for the single cell store, HDF5/h5ad ingestion, MTX ingestion, shared
/// format parsing and other errors.
#[derive(Debug, Error)]
pub enum BixverseErrors {
    // -- arguments --
    /// General error for invalid arguments to a given function
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Error for wrong integer parameters
    #[error("Parameter {0} must be a positive non-zero integer.")]
    MustBePositive(String),

    // -- extendr --
    /// List to HashMap parsing error occured
    #[error("R list parsing failed: {0}")]
    RListParse(&'static str),

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

    // -- graph based errors ---
    /// Error for algorithms that expect undirected graphs
    ///
    /// For methods that expect an undirected graph, but received a directed
    /// one.
    #[error("The Graph is directed but needs to be undirected for this algorithm.")]
    GraphDirectedError,

    /// Error for community membership/graph node number mismatch
    ///
    /// In cases where the graph and the community membership do not agree.
    #[error("The number of nodes and membership assignments in the communities do not add up.")]
    CommunityAssignmentMismatch,

    /// Error if a partition label sits outside the declared partition count
    #[error("The partition label {label} is out of range for {n_partitions} partitions.")]
    PartitionLabelOutOfRange {
        /// The offending label
        label: usize,
        /// Number of declared partitions
        n_partitions: usize,
    },

    /// Most of the supplied kNN indices point outside the cell set
    ///
    /// The usual cause is a kNN computed against a reference index rather than
    /// the query set, which drops every edge and returns an all-zero
    /// connectivity matrix looking like a genuine result.
    #[cfg(feature = "single-cell")]
    #[error(
        "PAGA: {out_of_range} of {total} kNN entries point outside the {n_cells} cells; the graph was probably built against a different index"
    )]
    PagaKnnIndicesOutOfRange {
        /// Entries pointing past the last cell
        out_of_range: usize,
        /// Entries supplied in total
        total: usize,
        /// Cells in the input
        n_cells: usize,
    },

    /// Error if the partition count exceeds what the dense coarsening accepts
    #[error(
        "The number of partitions ({n_partitions}) exceeds the maximum of {max} for graph abstraction."
    )]
    TooManyPartitions {
        /// Provided number of partitions
        n_partitions: usize,
        /// The supported maximum
        max: usize,
    },

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

    /// General error for sparse matrix format mismatches
    #[error("Expected this compressed sparse format {expected}; got {got}.")]
    SparseLayoutMismatch {
        /// The expected [CompressedSparseFormat] enum.
        expected: CompressedSparseFormat,
        /// The provided [CompressedSparseFormat] enum.
        got: CompressedSparseFormat,
    },

    /// Shape mismatch problem for matrices
    #[error("Expected this shape {expected:?}; got {got:?}.")]
    ShapeMismatch {
        /// Expected shape
        expected: (usize, usize),
        /// Provided shape
        got: (usize, usize),
    },

    // -- Binary file I/O --
    /// Wraps any `std::io::Error` encountered while reading or writing the
    /// bixverse binary sparse format.
    ///
    /// Covers file creation, seeks, buffered writes, and raw reads against
    /// the memory-mapped region. Use `#[from]` conversion so `?` works on
    /// any `io::Result`.
    #[cfg(feature = "single-cell")]
    #[error("I/O error on binary file: {0}")]
    BinaryIo(#[from] io::Error),

    /// Error if counts do not seem to be raw counts during i/o
    #[cfg(feature = "single-cell")]
    #[error("The counts you are trying to load in do not seem to be raw counts")]
    NotRawCounts,

    /// Bincode failed to encode the `FileHeader` or `SparseDataHeader`.
    ///
    /// Practically unreachable for the fixed-layout headers used here, but
    /// kept as a non-panicking path so the writer never aborts a long ingest.
    #[cfg(feature = "single-cell")]
    #[error("Failed to encode file header")]
    HeaderEncodeFailed,

    /// Bincode failed to decode the `FileHeader` or `SparseDataHeader`.
    ///
    /// Indicates the file is truncated, corrupt, or was not produced by
    /// bixverse. The 64-byte fixed file header is the first thing read, so
    /// most "wrong file type" errors surface here.
    #[cfg(feature = "single-cell")]
    #[error("Failed to decode file header - file may be corrupt or truncated")]
    HeaderDecodeFailed,

    /// File version does not match the current `SC_FILE_VERSION`.
    ///
    /// Returned on read when an older or newer binary is opened. The fix is
    /// to regenerate the file with the current bixverse version.
    #[cfg(feature = "single-cell")]
    #[error("File version mismatch: expected {expected}, got {found}")]
    FileVersionMismatch {
        /// Version the current build expects.
        expected: u32,
        /// Version actually read from the file header.
        found: u32,
    },

    /// A chunk buffer was shorter than the minimum header length.
    ///
    /// Raised before any field is parsed from a decompressed chunk, so the
    /// reader can fail early instead of indexing out of bounds.
    #[cfg(feature = "single-cell")]
    #[error("Chunk buffer too small: expected at least {expected} bytes, got {found}")]
    ChunkBufferTooSmall {
        /// Minimum bytes required for the chunk header.
        expected: usize,
        /// Actual bytes available in the buffer.
        found: usize,
    },

    /// LZ4 decompression of a chunk failed.
    ///
    /// The offset is reported to make corrupt regions locatable. Usually
    /// indicates file truncation or a mismatched compressed-size prefix.
    #[cfg(feature = "single-cell")]
    #[error("Failed to decompress chunk at offset {0}")]
    ChunkDecompressionFailed(u64),

    /// A caller-supplied original index was not present in the file's index
    /// map.
    ///
    /// Indicates either a bug in the calling code (asking for an index that
    /// was never written) or a file that was generated with a different set
    /// of cells/genes than the caller assumes.
    #[cfg(feature = "single-cell")]
    #[error("Chunk index {0} not found in file index map")]
    ChunkIndexNotFound(usize),

    /// Two stores that must share a gene axis disagree on how many genes
    /// they hold.
    ///
    /// Raised where a method takes a gene-major and a cell-major reader over
    /// what is meant to be the same experiment. Nothing ties the two files
    /// together, so the check has to be explicit: indices derived from one
    /// store are used against buffers sized from the other.
    #[cfg(feature = "single-cell")]
    #[error(
        "The gene store holds {gene_store} genes but the cell store holds {cell_store}; they must describe the same experiment"
    )]
    GeneAxisMismatch {
        /// Total genes reported by the gene-major store
        gene_store: usize,
        /// Total genes reported by the cell-major store
        cell_store: usize,
    },

    /// A cell-based reader method was called on a gene-based file, or vice
    /// versa.
    ///
    /// Replaces the previous runtime asserts. The two string fields describe
    /// the file's actual layout and the layout the caller requested.
    #[cfg(feature = "single-cell")]
    #[error("Reader mode mismatch: file is {actual}, requested {requested}")]
    ReaderModeMismatch {
        /// What the file is: "cell-based" or "gene-based".
        actual: &'static str,
        /// What the caller requested: "cell-based" or "gene-based".
        requested: &'static str,
    },

    /// A chunk's declared payload lengths exceed the bytes actually present.
    ///
    /// The three length fields in a chunk header are untrusted input. This is
    /// raised after they have been parsed but before any payload slice is
    /// taken, so a corrupt length cannot index out of bounds.
    #[cfg(feature = "single-cell")]
    #[error("Chunk payload truncated: header declares {expected} bytes, buffer holds {found}")]
    ChunkPayloadTruncated {
        /// Total bytes the header's length fields imply.
        expected: usize,
        /// Actual bytes available in the decompressed chunk.
        found: usize,
    },

    /// The raw-count element-size discriminant was not recognised.
    ///
    /// Valid values are `RAW_ELEM_U16` (2), `RAW_ELEM_U32` (4), and 0 for
    /// legacy files where the byte was zero padding. Anything else means the
    /// chunk is corrupt; reinterpreting it would misparse the whole payload.
    #[cfg(feature = "single-cell")]
    #[error("Invalid raw count element size discriminant: {0}")]
    RawElemSizeInvalid(u8),

    /// A raw count read from disk does not fit the requested numeric type.
    ///
    /// Raised by `from_gene_chunks` / `from_cell_chunks` instead of silently
    /// saturating. Pick a wider `T` (`u32`, `f32`, `f64`) for the affected
    /// dataset.
    #[cfg(feature = "single-cell")]
    #[error("Raw count {value} does not fit target type {target_type}")]
    RawCountOverflow {
        /// The count that could not be represented.
        value: u32,
        /// Name of the type it was being converted into.
        target_type: &'static str,
    },

    /// The `target_size` recorded in a file header disagrees with the one the
    /// caller requested.
    ///
    /// Undoing a file's own library-size normalisation requires the exact
    /// factor the writer used, so a mismatch always yields wrong numbers.
    /// Files written before `target_size` entered the header report `0.0` and
    /// are exempt from the check.
    #[cfg(feature = "single-cell")]
    #[error(
        "Target size mismatch: file was normalised against {header}, but {requested} was requested"
    )]
    TargetSizeMismatch {
        /// Value stored in the file header.
        header: f32,
        /// Value the caller supplied.
        requested: f32,
    },

    /// Error for h5 ingestion if feature type is not found
    #[cfg(feature = "single-cell")]
    #[error(
        "The requested feature type ({requested}) was not found. On file found features are {found}"
    )]
    FeatureTypeNotFound {
        /// Requested feature type
        requested: String,
        /// Found feature types
        found: String,
    },

    /// Error if the h5 string type cannot be read
    #[cfg(feature = "single-cell")]
    #[error("h5: Unexpected str type identified: {0}")]
    H5UnexpectedStringType(String),

    // -- HDF5 / h5ad --
    /// Wraps any error from the `hdf5` crate.
    ///
    /// Covers file open failures, missing datasets (`X/data`, `X/indices`,
    /// `X/indptr`, `obs/*`), dtype mismatches, and slice reads. The wrapped
    /// error carries the original HDF5 message.
    #[cfg(feature = "single-cell")]
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    /// Provide custom error message for unsupported file formats.
    #[cfg(feature = "single-cell")]
    #[error("Unknown or unsupported h5ad format: {0}")]
    UnsupportH5ADFormat(String),

    /// A named `obs` column was requested but does not exist in the h5ad.
    ///
    /// Primarily raised by `write_h5_normalised_counts` when the caller
    /// passes an `obs_lib_size_col` that isn't in the file.
    #[cfg(feature = "single-cell")]
    #[error("obs column '{0}' not found in h5ad file")]
    ObsColumnMissing(String),

    /// The library-size vector read from `obs` does not match the number of
    /// cells in `X`.
    ///
    /// Indicates a malformed h5ad — typical cause is a user-edited obs table
    /// whose row count drifted from the matrix.
    #[cfg(feature = "single-cell")]
    #[error("Library size column length ({found}) does not match cell count ({expected})")]
    LibSizeLengthMismatch {
        /// Expected length (number of cells in `X`).
        expected: usize,
        /// Actual length of the obs column.
        found: usize,
    },

    // -- MTX --
    /// The header of an MTX file is malformed.
    ///
    /// The static string describes which part of the header failed (shape
    /// line missing, wrong field count, unparseable counts). Comment lines
    /// starting with `%` are skipped before this fires.
    #[cfg(feature = "single-cell")]
    #[error("Invalid MTX header: {0}")]
    MtxHeaderInvalid(&'static str),

    /// A field of the MTX shape line could not be parsed as a number.
    ///
    /// `field` names the offending column ("cell count", "gene count",
    /// "entry count") so that malformed inputs can be diagnosed without
    /// quoting user data. Which of the first two fields is which depends on
    /// the `cells_as_rows` orientation.
    #[cfg(feature = "single-cell")]
    #[error("Failed to parse MTX value as {field}")]
    MtxParseError {
        /// Which field failed: "cell count", "gene count", or "entry count".
        field: &'static str,
    },
    // -- Batch --
    /// Need at least two batches for this method
    #[cfg(feature = "single-cell")]
    #[error("You need at least two batches. Provided {n_batches} batches.")]
    NeedAtLeastTwoBatches {
        /// Number of provided batches
        n_batches: usize,
    },

    /// A batch has too few cells for the requested anchor search
    #[cfg(feature = "single-cell")]
    #[error("Batch {batch} has {n_cells} cells, needs at least {required} for anchor finding.")]
    TooFewCellsForAnchor {
        /// Batch index
        batch: usize,
        /// Cells in the batch
        n_cells: usize,
        /// Minimum required
        required: usize,
    },
    // -- Harmony --
    /// Sigma length is not equal to the number of clusters
    #[cfg(feature = "single-cell")]
    #[error("Harmony: sigma length must match number of clusters")]
    HarmonySigmaLengthUnequalCluster,

    // -- Format parsing --
    /// The `cs_type` string did not match a known sparse format.
    ///
    /// Accepted values are "csc" and "csr" (case-insensitive at the parser
    /// level). The unrecognised input is echoed back in the message.
    #[cfg(feature = "single-cell")]
    #[error("Unknown compressed sparse format: '{0}' (expected 'csc' or 'csr')")]
    UnknownSparseFormat(String),

    // -- PCA --
    /// Error if user wants CLR-normalised PCA, but did not provide the offsets
    #[cfg(feature = "single-cell")]
    #[error(
        "You did not provide the offsets needed for using the CLR-normalised PCA in single cell."
    )]
    OffsetsNotProvidedForClrPCA,

    /// Error if user wants CLR-normalised PCA, but did not provide the offsets
    #[cfg(feature = "single-cell")]
    #[error(
        "The provided offsets have length {len_offset}; n_cells is {n_cells}. Length mismatch."
    )]
    OffsetsLengthDoesNotMatchNCells {
        /// Length of the offsets
        len_offset: usize,
        /// Number of cells provided
        n_cells: usize,
    },

    // -- HVG --
    /// A requested cell index does not exist in the store.
    #[cfg(feature = "single-cell")]
    #[error("HVG: cell index {index} is outside the store, which holds {total_cells} cells.")]
    HvgCellIndexOutOfRange {
        /// The offending cell index
        index: usize,
        /// Number of cells the store holds
        total_cells: usize,
    },

    /// The same cell was selected twice, which would inflate the denominator.
    #[cfg(feature = "single-cell")]
    #[error("HVG: cell index {index} appears more than once in the selection.")]
    HvgDuplicateCellIndex {
        /// The duplicated cell index
        index: usize,
    },

    /// Batch labels and cell indices disagree in length.
    #[cfg(feature = "single-cell")]
    #[error("HVG: {n_labels} batch labels were given for {n_cells} selected cells.")]
    HvgBatchLabelLengthMismatch {
        /// Number of batch labels provided
        n_labels: usize,
        /// Number of selected cells
        n_cells: usize,
    },

    /// A batch in `0..n_batches` ended up with no cells.
    #[cfg(feature = "single-cell")]
    #[error("HVG: batch {batch} has no cells; labels must densely cover 0..{n_batches}.")]
    HvgEmptyBatch {
        /// The empty batch
        batch: usize,
        /// Number of batches implied by the labels
        n_batches: usize,
    },

    /// No cells were selected at all.
    #[cfg(feature = "single-cell")]
    #[error("HVG: the cell selection is empty.")]
    HvgEmptySelection,

    /// Loess span outside the range [`crate::core::base::loess::LoessRegression`]
    /// accepts.
    #[cfg(feature = "single-cell")]
    #[error("HVG: the loess span is {span}, but it must be within (0, 1].")]
    HvgInvalidLoessSpan {
        /// The provided span
        span: f32,
    },

    /// Bin count of zero, which would divide by zero during binning.
    #[cfg(feature = "single-cell")]
    #[error("HVG: n_bins must be at least 1.")]
    HvgInvalidBinCount,

    // -- NMF --
    /// NMF Rank is too large for the NNDSVD initialisation
    #[error(
        "The requested NMF ({requested}) rank is too large for NNDSVD initialisation (available: {available})."
    )]
    NmfRankTooLarge {
        /// Requested rank
        requested: usize,
        /// Maximum available rank
        available: usize,
    },

    /// Error if NMF values are not finite
    #[error("The NMF values are not finite")]
    NmfNonFinite,

    /// Error if NMF values are negative
    #[error("Negative values were discovered for NMF. Please check the inputs.")]
    NmfNonNegativeViolated,

    /// Consensus NMF received a rank that cannot support a silhouette.
    #[error("Consensus NMF needs k >= 2, but k = {k} was requested.")]
    NmfConsensusInvalidK {
        /// Requested rank
        k: usize,
    },

    /// Consensus NMF received too few restarts to pool over.
    #[error("Consensus NMF needs at least 2 restarts, but n_runs = {n_runs} was requested.")]
    NmfConsensusTooFewRuns {
        /// Requested number of restarts
        n_runs: usize,
    },

    /// The outlier filter removed so many components that k-means cannot run.
    #[error(
        "Consensus NMF: only {n_surviving} components survived the density filter, which is fewer than k = {k}. Raise the density threshold or add restarts."
    )]
    NmfConsensusTooFewComponents {
        /// Requested rank
        k: usize,
        /// Components left after filtering
        n_surviving: usize,
    },

    /// k-means returned an empty cluster, so the consensus factor would carry a
    /// zero row and the refit would be degenerate.
    #[error(
        "Consensus NMF: cluster {cluster} of {k} is empty, so the consensus factor is degenerate."
    )]
    NmfConsensusEmptyCluster {
        /// Index of the empty cluster
        cluster: usize,
        /// Requested rank
        k: usize,
    },

    /// The k sweep was handed nothing to sweep over.
    #[error("The NMF k sweep was given an empty k range.")]
    NmfKSweepEmptyRange,

    /// A consensus step was applied to restarts run at a different rank.
    #[error(
        "Consensus NMF was asked for k = {requested}, but the restarts were run at k = {restarts}."
    )]
    NmfConsensusKMismatch {
        /// Rank requested for the consensus
        requested: usize,
        /// Rank the restarts actually hold
        restarts: usize,
    },

    /// A frozen factor handed to a refit does not line up with V.
    #[error(
        "NMF refit: the frozen factor has a shared dimension of {found}, but V expects {expected}."
    )]
    NmfDimensionMismatch {
        /// Dimension V expects
        expected: usize,
        /// Dimension the frozen factor provides
        found: usize,
    },

    // -- Hotspot --
    /// Invalid model chosen for Hotspot
    #[cfg(feature = "single-cell")]
    #[error("Invalid model type: {0}")]
    HotSpotWrongModel(String),

    // -- Metacells2 --
    /// User needs to have generated select_features prior to the
    /// compute_similarity part.
    #[cfg(feature = "single-cell")]
    #[error("select_features must be called before compute_similarity!")]
    SelectFeaturesBeforeSimilariy,

    // -- Seacells --
    /// The SEACells kernel matrix has not been yet constructed.
    #[cfg(feature = "single-cell")]
    #[error("SEACells: You must construct the kernel matrix first.")]
    SEACellsKernelMatrixMissing,

    /// The SEACells archetypes have not yet been identified.
    #[cfg(feature = "single-cell")]
    #[error("SEACells: You must first initialise the Archetypes.")]
    SEACellsArchetypesMissing,

    /// The SEACells archetypes have not yet been identified.
    #[cfg(feature = "single-cell")]
    #[error("SEACells: The model has not been fitted yet. Please run .fit()")]
    SEACellsModelNotFitted,

    /// The adaptive diffusion kernel came out with non-finite weights.
    ///
    /// Raised by
    /// [crate::single_cell::mc_generation::seacells::compute_diffusion_kernel].
    /// The usual cause is a caller-supplied kNN graph carrying `NaN` distances,
    /// since zero bandwidths are handled by the smallest-positive fallback.
    #[cfg(feature = "single-cell")]
    #[error(
        "The adaptive diffusion kernel over {n_cells} cells holds non-finite weights; check the supplied kNN distances"
    )]
    DiffusionKernelNotFinite {
        /// Cells in the kNN graph
        n_cells: usize,
    },

    // -- FastCluster --
    /// The Fast cluster results do not contain k-means cluster assignments
    #[cfg(feature = "single-cell")]
    #[error("The fast cluster results were generated without any k means cluster assignments")]
    FastClusterNoKmeansAssignments,

    /// The Fast cluster results do not contain k-means centroids
    #[cfg(feature = "single-cell")]
    #[error("The fast cluster results were generated without any centroids")]
    FastClusterNoCentroids,

    // -- MELD --
    /// If the user provides a label which is out-of-range of the expected
    /// number of groups
    #[cfg(feature = "single-cell")]
    #[error("MELD: label {label} out of range for n_groups={n_groups}.")]
    MELDLabelOutOfRange {
        /// The label which is out of range
        label: usize,
        /// The number of expected groups
        n_groups: usize,
    },

    /// If the user asks for less than two Chebyshev coefficients
    #[cfg(feature = "single-cell")]
    #[error("MELD: Need at least 2 Chebyshev coefficients")]
    MELDChebyshevCoefTooLow,

    /// Error if labels and number of cells are not equal
    #[cfg(feature = "single-cell")]
    #[error("MELD: The labels are not equal to the number of cells")]
    MELDLabelUnequalsSamples,

    /// Error of n_groups < 2
    #[cfg(feature = "single-cell")]
    #[error("MELD: labels needs two groups minimum")]
    MELDOnlyOneGroup,

    // -- DGE --
    /// The reference group holds no cells.
    #[cfg(feature = "single-cell")]
    #[error("DGE: the reference group is empty")]
    DgeEmptyReferenceGroup,

    /// No comparison group was supplied at all.
    #[cfg(feature = "single-cell")]
    #[error("DGE: at least one comparison group is required")]
    DgeNoComparisonGroups,

    /// One of the comparison groups holds no cells.
    #[cfg(feature = "single-cell")]
    #[error("DGE: comparison group {group} is empty")]
    DgeEmptyComparisonGroup {
        /// Index of the offending comparison group
        group: usize,
    },

    /// A cell appears both in the reference and in a comparison group.
    ///
    /// Silently biases every AUROC toward 0.5 rather than failing, so it is
    /// worth rejecting outright.
    #[cfg(feature = "single-cell")]
    #[error("DGE: cell {cell} appears in both the reference and comparison group {group}")]
    DgeOverlappingGroups {
        /// Index of the offending comparison group
        group: usize,
        /// The cell shared with the reference group
        cell: usize,
    },

    // -- Palantir --
    /// The user-supplied early cell index sits outside the cell range.
    #[cfg(feature = "single-cell")]
    #[error("Palantir: early cell index {early_cell} is out of range for {n_cells} cells")]
    PalantirEarlyCellOutOfRange {
        /// The offending early cell index
        early_cell: usize,
        /// Number of cells in the input
        n_cells: usize,
    },

    /// A user-supplied terminal state index sits outside the cell range.
    #[cfg(feature = "single-cell")]
    #[error("Palantir: terminal state index {state} is out of range for {n_cells} cells")]
    PalantirTerminalStateOutOfRange {
        /// The offending terminal state index
        state: usize,
        /// Number of cells in the input
        n_cells: usize,
    },

    /// A terminal state cell was never sampled as a waypoint, so it has no
    /// position in the waypoint Markov chain.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: terminal state cell {state} is not among the {n_waypoints} sampled waypoints"
    )]
    PalantirTerminalStateNotAWaypoint {
        /// The offending terminal state cell index
        state: usize,
        /// Number of sampled waypoints
        n_waypoints: usize,
    },

    /// The first waypoint is not the start cell, which every consumer of the
    /// geodesic matrix assumes.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: waypoint 0 is cell {first_waypoint}, but the start cell is {start_cell}; the start cell must be the first waypoint"
    )]
    PalantirStartCellNotFirstWaypoint {
        /// The start cell the caller asked for
        start_cell: usize,
        /// The cell actually sitting in waypoint position zero
        first_waypoint: usize,
    },

    /// The pseudotime refinement cap leaves no room for a single pass.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: max_iterations is {max_iterations}, but at least {minimum} is needed to run a refinement pass"
    )]
    PalantirMaxIterationsTooSmall {
        /// The requested cap
        max_iterations: usize,
        /// Smallest cap that runs at least one pass
        minimum: usize,
    },

    /// The neighbour count is too small to derive the back-edge bandwidth of
    /// the waypoint Markov chain, which indexes neighbour rank
    /// `min(knn / 3 - 1, 30)`.
    ///
    /// Carries the neighbour count the caller asked for, not the one clamped to
    /// the waypoint set. Too few waypoints raises
    /// [BixverseErrors::PalantirTooFewWaypoints] instead, so the message always
    /// names the parameter that can actually be changed.
    #[cfg(feature = "single-cell")]
    #[error("Palantir: knn is {knn}, but at least {minimum} is needed for the adaptive bandwidth")]
    PalantirKnnTooSmall {
        /// The requested neighbour count
        knn: usize,
        /// Smallest neighbour count yielding a valid bandwidth rank
        minimum: usize,
    },

    /// Too few waypoints were sampled to support the back-edge bandwidth rank,
    /// however large the requested `knn` is.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: only {n_waypoints} waypoints were sampled, but at least {minimum} are needed for the adaptive bandwidth; raise num_waypoints or supply more cells"
    )]
    PalantirTooFewWaypoints {
        /// Waypoints actually sampled
        n_waypoints: usize,
        /// Smallest waypoint count yielding a valid bandwidth rank
        minimum: usize,
    },

    /// The waypoint data and the waypoint pseudotime disagree on the waypoint
    /// count.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: {n_waypoints} waypoint rows were supplied against {n_pseudotime} pseudotime values"
    )]
    PalantirWaypointDataMismatch {
        /// Rows of multiscale data restricted to the waypoints
        n_waypoints: usize,
        /// Pseudotime values supplied alongside
        n_pseudotime: usize,
    },

    /// The stationary ranks have no spread, so the outlier cutoff separates
    /// nothing.
    ///
    /// More than half the waypoints share a stationary mass, which puts the
    /// median absolute deviation at exactly zero and the cutoff exactly on the
    /// median. Roughly half of everything then clears it, the components merge,
    /// and the branching structure collapses to a single terminal state.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: the stationary ranks over {n_waypoints} waypoints have zero spread, so terminal states cannot be separated; supply them explicitly or raise knn"
    )]
    PalantirStationaryRanksDegenerate {
        /// Waypoints in the chain
        n_waypoints: usize,
    },

    /// Cells remained unreachable from the start cell after graph repair.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: {n_unreachable} cells remain unreachable from the start cell after {repairs} repair edges; increase knn"
    )]
    PalantirDisconnectedGraph {
        /// Cells still at infinite geodesic distance
        n_unreachable: usize,
        /// Repair edges added before giving up
        repairs: usize,
    },

    /// The pseudotime collapsed to a constant, or the Silverman bandwidth was
    /// zero or non-finite, so the refinement cannot be normalised.
    #[cfg(feature = "single-cell")]
    #[error("Palantir: degenerate pseudotime ({reason}); the manifold is probably collapsed")]
    PalantirDegeneratePseudotime {
        /// Which quantity degenerated
        reason: &'static str,
    },

    /// The stationary power iteration lost all of its mass, which can only
    /// happen when the supplied chain is not row-stochastic or carries
    /// non-finite entries.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: the stationary power iteration lost its mass at iteration {iteration} (total mass {total:e}); the transition matrix is not row-stochastic"
    )]
    PalantirStationaryMassLost {
        /// The pass on which the mass went non-positive or non-finite
        iteration: usize,
        /// The offending total mass
        total: f64,
    },

    /// An eigenvalue of the diffusion operator came back non-finite.
    ///
    /// Clamping it would fabricate a plausible-looking scale factor out of a
    /// `NaN`, since `f64::min` returns the other operand for a `NaN` input.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: the diffusion eigensolve returned a non-finite eigenvalue ({eigenvalue}); the kernel is degenerate"
    )]
    PalantirNonFiniteEigenvalue {
        /// The offending eigenvalue
        eigenvalue: f64,
    },

    /// An explicit `n_eigs` below the smallest usable count was requested.
    ///
    /// The trivial leading eigenvector is always dropped, so fewer than two
    /// eigenvectors leaves no multiscale coordinate at all. The request is
    /// refused rather than raised to the heuristic's floor: silently changing a
    /// pinned count changes every distance downstream.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: n_eigs is {n_eigs}, but at least {minimum} eigenvectors are needed for a multiscale space"
    )]
    PalantirNEigsTooSmall {
        /// The requested eigenvector count
        n_eigs: usize,
        /// Smallest count leaving a usable coordinate
        minimum: usize,
    },

    /// No terminal states survived the rank cutoff and the caller supplied none.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: no terminal states detected; supply them explicitly or increase num_waypoints"
    )]
    PalantirNoTerminalStates,

    /// The multiscale space has no boundary cells, so there is nothing to snap
    /// the start cell or a terminal-state candidate onto.
    ///
    /// Boundaries are the per-component argmin and argmax, so an empty set means
    /// the multiscale space has no columns at all.
    #[cfg(feature = "single-cell")]
    #[error("Palantir: the multiscale space has no boundary cells to snap onto")]
    PalantirNoBoundaryCells,

    /// Cells remained unreachable from the waypoint set when the geodesic
    /// matrix was built.
    ///
    /// Distinct from [BixverseErrors::PalantirDisconnectedGraph], which reports
    /// the repair attempts. This fires after repair, inside the pseudotime
    /// solve, where the repair count is not in scope and claiming zero of them
    /// would be a lie.
    #[cfg(feature = "single-cell")]
    #[error("Palantir: {n_unreachable} cells have no geodesic path to some waypoint; increase knn")]
    PalantirUnreachableFromWaypoints {
        /// Cells at infinite geodesic distance from at least one waypoint
        n_unreachable: usize,
    },

    /// The densified `(I - Q) B = R` solve did not converge.
    ///
    /// Carries the achieved `‖(I - Q) B - R‖_inf`, which is what actually
    /// distinguishes a near-singular system from ordinary rounding.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: the (I - Q) B = R solve over {n_transient} transient waypoints left a residual of {residual:e}, above tolerance"
    )]
    PalantirAbsorbingSolveFailed {
        /// Achieved `‖(I - Q) B - R‖_inf`
        residual: f64,
        /// Total transient waypoints
        n_transient: usize,
    },

    /// The absorption probabilities of one waypoint sum to more than one.
    ///
    /// Split out from [BixverseErrors::PalantirAbsorbingSolveFailed] because the
    /// solve residual says nothing here: the factorisation can be exact to
    /// machine precision and still land on this, which is what makes the
    /// offending sum the only number worth reporting.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: waypoint {waypoint} absorbs with probability {row_sum} across {n_transient} transient waypoints, which is above one"
    )]
    PalantirAbsorbingRowSum {
        /// The offending row sum
        row_sum: f64,
        /// Waypoint index the row belongs to
        waypoint: usize,
        /// Total transient waypoints
        n_transient: usize,
    },

    /// The absorption probabilities of one waypoint are not finite.
    ///
    /// Split from [BixverseErrors::PalantirAbsorbingRowSum] because the two
    /// point at different bugs: a non-finite sum means a `NaN` reached the
    /// solve, while a sum above one means the chain was not sub-stochastic.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: waypoint {waypoint} has a non-finite absorption row sum ({row_sum}) across {n_transient} transient waypoints; the solve produced non-finite entries"
    )]
    PalantirAbsorbingRowNotFinite {
        /// The offending row sum
        row_sum: f64,
        /// Waypoint index the row belongs to
        waypoint: usize,
        /// Total transient waypoints
        n_transient: usize,
    },

    /// The densified `(I - Q)` would exceed the supported size.
    #[cfg(feature = "single-cell")]
    #[error(
        "Palantir: {n_transient} transient waypoints exceed the dense solve limit of {limit}; lower num_waypoints"
    )]
    PalantirTooManyWaypointsForDenseSolve {
        /// Transient waypoint count requested
        n_transient: usize,
        /// The supported ceiling
        limit: usize,
    },

    // -- MAGIC --
    /// The requested imputation would not fit in a sane amount of memory.
    ///
    /// MAGIC output is unavoidably dense, so the only bound available is the
    /// caller's gene selection. This is an error rather than a warning because
    /// the crate is driven from R, where a printed warning is easily missed and
    /// the alternative failure mode is the session being OOM-killed.
    #[cfg(feature = "single-cell")]
    #[error(
        "MAGIC: imputing {n_genes} genes across {n_cells} cells needs {} MB of dense f32, above the {} MB budget; subset the genes or set allow_large",
        (*n_cells as f64 * *n_genes as f64 * 4.0 / 1.048576e6).round(),
        (*max_elements as f64 * 4.0 / 1.048576e6).round()
    )]
    MagicOutputTooLarge {
        /// Cells in the selection
        n_cells: usize,
        /// Genes in the selection
        n_genes: usize,
        /// The element ceiling that was exceeded
        max_elements: usize,
    },

    /// The kNN inputs disagree with the requested cell selection.
    #[cfg(feature = "single-cell")]
    #[error(
        "MAGIC: the kNN graph has {n_knn} rows but {n_selected} cells were selected; the two must be aligned"
    )]
    MagicKnnSelectionMismatch {
        /// Rows in the supplied kNN graph
        n_knn: usize,
        /// Cells in the selection
        n_selected: usize,
    },

    /// A cell index in the selection sits outside the store.
    #[cfg(feature = "single-cell")]
    #[error("MAGIC: cell index {cell} is out of range for a store of {total_cells} cells")]
    MagicCellOutOfRange {
        /// The offending cell index
        cell: usize,
        /// Cells in the store
        total_cells: usize,
    },

    /// A cell index appears twice in the selection.
    #[cfg(feature = "single-cell")]
    #[error("MAGIC: cell index {cell} appears more than once in the selection")]
    MagicDuplicateCell {
        /// The duplicated cell index
        cell: usize,
    },

    /// The operator was built against a different store than the one being read.
    ///
    /// The operator's cell lookup is indexed by global cell id, so reading a
    /// store with more cells than it was sized for would index past its end.
    #[cfg(feature = "single-cell")]
    #[error(
        "MAGIC: the operator was built for a store of {operator_cells} cells but the reader holds {store_cells}; the `total_cells` passed to MagicOperator::from_knn must match the store"
    )]
    MagicStoreSizeMismatch {
        /// Cells the operator was built for
        operator_cells: usize,
        /// Cells the reader actually holds
        store_cells: usize,
    },

    // -- Gene trends --
    /// The expression matrix does not have one row per cell.
    #[cfg(feature = "single-cell")]
    #[error(
        "Gene trends: the expression matrix has {n_cells} rows but {n_pseudotime} pseudotime values were supplied"
    )]
    GeneTrendsShapeMismatch {
        /// Rows in the expression matrix
        n_cells: usize,
        /// Length of the pseudotime vector
        n_pseudotime: usize,
    },

    /// A branch selected no cells at all.
    #[cfg(feature = "single-cell")]
    #[error("Gene trends: branch {branch} selected no cells")]
    GeneTrendsBranchEmpty {
        /// Index of the offending branch
        branch: usize,
    },

    /// A branch has too few cells to fit anything meaningful.
    ///
    /// The reference has no such guard and will happily fit a 500-point grid to
    /// a single cell. The resulting curve is the prior, not the data.
    #[cfg(feature = "single-cell")]
    #[error(
        "Gene trends: branch {branch} has only {n_cells} cells, which is below the minimum of {minimum}"
    )]
    GeneTrendsBranchTooFewCells {
        /// Index of the offending branch
        branch: usize,
        /// Cells the branch selected
        n_cells: usize,
        /// Minimum the fit requires
        minimum: usize,
    },

    /// Every cell in a branch shares the same pseudotime, so there is no grid
    /// to fit over.
    #[cfg(feature = "single-cell")]
    #[error("Gene trends: branch {branch} spans zero pseudotime, so its grid would be degenerate")]
    GeneTrendsDegeneratePseudotime {
        /// Index of the offending branch
        branch: usize,
    },

    /// Fate-probability weighting was asked for without the probabilities.
    #[cfg(feature = "single-cell")]
    #[error(
        "Gene trends: BranchWeighting::FateProbability needs the fate probabilities, but none were supplied"
    )]
    GeneTrendsMissingFateProbabilities,

    /// The fate-probability matrix does not have one column per branch.
    ///
    /// Branch `i` takes column `i` positionally, so a mismatch would silently
    /// weight every cell by the wrong fate rather than failing.
    #[cfg(feature = "single-cell")]
    #[error(
        "Gene trends: the fate probabilities have {n_columns} columns but {n_branches} branches were supplied; branches map to columns positionally, so the two must agree"
    )]
    GeneTrendsFateColumnMismatch {
        /// Columns in the fate probability matrix
        n_columns: usize,
        /// Branches supplied
        n_branches: usize,
    },

    /// The fate-probability matrix does not have one row per cell.
    #[cfg(feature = "single-cell")]
    #[error(
        "Gene trends: the fate probabilities have {n_rows} rows but {n_cells} cells were supplied"
    )]
    GeneTrendsFateShapeMismatch {
        /// Rows in the fate probability matrix
        n_rows: usize,
        /// Cells supplied
        n_cells: usize,
    },

    /// A branch cell index sits outside the cell range.
    #[cfg(feature = "single-cell")]
    #[error(
        "Gene trends: branch {branch} references cell {cell}, which is out of range for {n_cells} cells"
    )]
    GeneTrendsCellOutOfRange {
        /// Index of the offending branch
        branch: usize,
        /// The offending cell index
        cell: usize,
        /// Cells in the input
        n_cells: usize,
    },

    /// Every cell in a branch carries zero fate-probability weight.
    ///
    /// The Gaussian process treats a zero-weight observation as absent, so this
    /// would otherwise return the zero-mean prior and be indistinguishable from
    /// a genuinely flat gene.
    #[cfg(feature = "single-cell")]
    #[error(
        "Gene trends: every cell selected for branch {branch} has zero fate probability, so the fit would carry no data"
    )]
    GeneTrendsZeroWeightBranch {
        /// Index of the offending branch
        branch: usize,
    },

    /// The requested grid is too coarse to fit anything over.
    #[cfg(feature = "single-cell")]
    #[error("Gene trends: a resolution of {resolution} is below the minimum of {minimum}")]
    GeneTrendsResolutionTooLow {
        /// Requested grid points
        resolution: usize,
        /// Minimum the fit requires
        minimum: usize,
    },

    // -- sctype --
    /// Error when number of cluster assignment != the number of cells
    #[cfg(feature = "single-cell")]
    #[error(
        "SCType: The number of cells ({n_cells}) and cluster assignments length ({n_cluster_assignment}) is not the same."
    )]
    ScTypeClusterAssignmentNotEqualNCells {
        /// Number of cells
        n_cells: usize,
        /// Number of cluster assignments
        n_cluster_assignment: usize,
    },

    /// Error when the smoothing graph node count != the number of cells
    #[cfg(feature = "single-cell")]
    #[error("SCType: The graph has {n_nodes} nodes, but there are {n_cells} cells.")]
    ScTypeGraphNodesNotEqualNCells {
        /// Number of cells
        n_cells: usize,
        /// Number of nodes in the smoothing graph
        n_nodes: usize,
    },

    // -- wnn --
    /// Error if the modalities do not have the same number of cells
    #[cfg(feature = "multi-modal")]
    #[error("WNN: Both modalities need to have the same cell numbers.")]
    WNNModalitySampleMismatch,

    /// Error if k_nn > knn_range
    #[cfg(feature = "multi-modal")]
    #[error("WNN: k_nn must be <= knn_range")]
    WNNKnnLargerThanKnnRange,

    /// Error if sigma_idx >= knn_range
    #[cfg(feature = "multi-modal")]
    #[error("WNN: sigma_idx must be < knn_range")]
    WNNSigmaIdxOutOfRange,

    /// Error if s_nn > knn_range
    #[cfg(feature = "multi-modal")]
    #[error("WNN: s_nn must be <= knn_range")]
    WNNSnnLargerThanKnnRange,

    /// Error if a modality's kNN row count does not match n_cells
    #[cfg(feature = "multi-modal")]
    #[error("WNN: modality {modality} kNN row count {found} does not match n_cells {expected}")]
    WNNKnnRowCountMismatch {
        /// 0 or 1
        modality: usize,
        /// Expected count
        expected: usize,
        /// Found count
        found: usize,
    },

    /// Error if a modality's kNN distance row count does not match n_cells
    #[cfg(feature = "multi-modal")]
    #[error(
        "WNN: modality {modality} kNN distance row count {found} does not match n_cells {expected}"
    )]
    WNNKnnDistRowCountMismatch {
        /// 0 or 1
        modality: usize,
        /// Expected count
        expected: usize,
        /// Found count
        found: usize,
    },

    /// Error if a modality has fewer than knn_range neighbours per cell
    #[cfg(feature = "multi-modal")]
    #[error(
        "WNN: modality {modality} needs at least {expected} neighbours per cell, found {found}"
    )]
    WNNInsufficientNeighbours {
        /// 0 or 1
        modality: usize,
        /// knn_range
        expected: usize,
        /// Actual neighbour count in row 0
        found: usize,
    },

    // -- gpu --
    /// A device-limit or runtime error from the `cubecl-utils-rs` primitives.
    ///
    /// Covers the cube-count, grid, binding-size and shared-memory guards, plus
    /// buffer reads that fail on the CubeCL server.
    #[cfg(feature = "gpu")]
    #[error("GPU: {0}")]
    CubeclUtils(#[from] cubecl_utils_rs::CubeclUtilsErrors),
    /// A GPU cubecl matrix multiplication error from the cubek crate
    #[cfg(feature = "gpu")]
    #[error("GPU: A matrix multiplication occurred: {0}")]
    GpuMatmul(String),
    /// A single named buffer exceeds the device's per-binding size limit.
    ///
    /// Over-sized bindings are rejected without an error surfacing: the kernel
    /// does no work and returns zeros, so the condition is caught on the host
    /// before the upload instead.
    ///
    /// `GpuTensor`'s own constructors already guard their allocations and
    /// report `CubeclUtils`. This variant is for the host-side pre-checks that
    /// walk a whole named buffer set before a dispatch, where the buffer name
    /// is worth more than a bare byte count.
    #[cfg(feature = "gpu")]
    #[error(
        "GPU: buffer '{buffer}' needs {bytes} bytes, device per-binding limit is {limit} bytes"
    )]
    GpuBindingTooLarge {
        /// Name of the buffer that does not fit.
        buffer: &'static str,
        /// Bytes the buffer requires.
        bytes: usize,
        /// Per-binding limit reported by the device.
        limit: usize,
    },
    /// The requested NMF rank is above the largest comptime register tier the
    /// GPU HALS sweeps are compiled for.
    ///
    /// The sweeps hold a whole row of `W` or column of `H` in a register array
    /// whose length must be known at compilation, so the dispatch picks the
    /// smallest tier that fits `k`. Past the largest tier the array would spill
    /// to global memory and the kernel would be slower than the CPU path, so
    /// this errors rather than silently degrading.
    #[cfg(feature = "gpu")]
    #[error(
        "NMF GPU: rank k = {k} is above the largest supported rank of {max}; use the CPU entry point for ranks this large"
    )]
    GpuNmfRankTooLarge {
        /// Requested number of components.
        k: usize,
        /// Largest rank the compiled tiers cover.
        max: usize,
    },
    // -- gpu / single cell --
    /// GPU Harmony only supports one co-variate for now (based on Arrowhead)
    #[cfg(feature = "gpu")]
    #[error("Harmony GPU: Only a single co-variate is supported for the GPU path")]
    GpuHarmonySupportsSingleCovariateOnly,
    /// Selected SCENIC regression learner has no GPU implementation. Only the
    /// tree-based learners (ExtraTrees, RandomForest) are ported; GRNBoost2 /
    /// gradient boosting stays on CPU by design.
    #[cfg(feature = "gpu")]
    #[error(
        "SCENIC GPU: the {learner} regression learner has no GPU path; use the CPU entry point (run_scenic_grn / run_scenic_grn_streaming / run_scenic_grn_in_memory)"
    )]
    GpuNotSupportedForLearner {
        /// Name of the requested regression learner.
        learner: &'static str,
    },
    /// The device cannot give `build_score_rf_fused` the threadgroup memory it
    /// asks for, so SCENIC's RandomForest learner has no GPU path on it.
    ///
    /// The kernel's ask is a compile-time constant checked against the tightest
    /// shipping backend floor, so reaching this means a device well outside
    /// what the crate targets rather than an unlucky shape.
    #[cfg(feature = "gpu")]
    #[error(
        "SCENIC GPU: the fused RandomForest kernel needs {bytes} bytes of shared memory but this device offers {limit}; use the CPU entry point (run_scenic_grn / run_scenic_grn_streaming / run_scenic_grn_in_memory) or switch the learner to ExtraTrees"
    )]
    GpuScenicFusedRfUnavailable {
        /// Shared-memory bytes the fused kernel allocates per workgroup.
        bytes: usize,
        /// Shared-memory bytes the device reports.
        limit: usize,
    },
}
