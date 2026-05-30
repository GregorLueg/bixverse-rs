//! Errors in bixverse. Contains all the various errors that can be returned by
//! the crate.

use ann_search_rs::utils::dist::Dist;
#[cfg(feature = "single-cell")]
use std::io;
use thiserror::Error;

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

    // -- extendr --
    /// List to HashMap parsing error occured
    #[error("R list parsing failed: {0}")]
    RListParse(&'static str),

    // -- Math / Faer --
    /// SVD from faer failed to converge or returned no solution.
    ///
    /// Typically caused by ill-conditioned or degenerate input (e.g. all-zero
    /// rows, NaNs, rank-deficient matrices beyond the requested rank).
    #[error("The faer SVD failed - please verify the data")]
    FaerSvdError,

    /// Eigen decomposition from faer failed.
    ///
    /// Usually indicates a non-symmetric matrix being passed to a symmetric
    /// solver, or numerical breakdown on degenerate input.
    #[error("The faer Eigen decomposition failed - please verify the data")]
    FaerEigenError,

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

    /// If serialisation of the [crate::prelude::CompressedSparseData2] to disk failed
    #[cfg(feature = "single-cell")]
    #[error("Serialisation to meta cell CompressedSparseData2 format on disk failed")]
    SerialisationFailed,

    /// If deserialisation of the [crate::prelude::CompressedSparseData2] from disk
    /// failed
    #[cfg(feature = "single-cell")]
    #[error("Serialisation to meta cell CompressedSparseData2 format on disk failed")]
    DeserialisationFailed,

    // -- HDF5 / h5ad --
    /// Wraps any error from the `hdf5` crate.
    ///
    /// Covers file open failures, missing datasets (`X/data`, `X/indices`,
    /// `X/indptr`, `obs/*`), dtype mismatches, and slice reads. The wrapped
    /// error carries the original HDF5 message.
    #[cfg(feature = "single-cell")]
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

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

    /// A field inside an MTX body line could not be parsed.
    ///
    /// `field` names the offending column ("row", "col", "value") so that
    /// malformed inputs can be diagnosed without quoting user data.
    #[cfg(feature = "single-cell")]
    #[error("Failed to parse MTX value as {field}")]
    MtxParseError {
        /// Which field failed: "row", "col", or "value".
        field: &'static str,
    },

    // -- Format parsing --
    /// The `cs_type` string did not match a known sparse format.
    ///
    /// Accepted values are "csc" and "csr" (case-insensitive at the parser
    /// level). The unrecognised input is echoed back in the message.
    #[cfg(feature = "single-cell")]
    #[error("Unknown compressed sparse format: '{0}' (expected 'csc' or 'csr')")]
    UnknownSparseFormat(String),

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

    /// Error if embedding rows do not match the kNN
    #[cfg(feature = "single-cell")]
    #[error("MELD: Embedding rows unequals samples")]
    MELDEmbeddingUnequalsSamples,

    // -- sctype --
    /// Error when number of cluster assignment != the number of cells
    #[error(
        "SCType: The number of cells ({n_cells}) and cluster assignments length ({n_cluster_assignment}) is not the same."
    )]
    ScTypeClusterAssignmentNotEqualNCells {
        /// Number of cells
        n_cells: usize,
        /// Number of cluster assignments
        n_cluster_assignment: usize,
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
}
