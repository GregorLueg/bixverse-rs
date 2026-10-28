//! Errors raised by `bixverse-sc-data`.

use std::io;
use thiserror::Error;

/// Error variants raised by `bixverse-sc-data`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum ScDataErrors {
    // -- lower crates --
    /// Propagates an error from `bixverse-sparse`.
    #[error(transparent)]
    Sparse(#[from] bixverse_sparse::errors::SparseErrors),

    /// Propagates an error from `bixverse-core`.
    #[cfg(feature = "analysis")]
    #[error(transparent)]
    Core(#[from] bixverse_core::errors::CoreErrors),

    // -- arguments --
    /// A caller-supplied argument is invalid.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    // -- Binary file I/O --
    /// Wraps any `std::io::Error` encountered while reading or writing the
    /// bixverse binary sparse format.
    ///
    /// Covers file creation, seeks, buffered writes, and raw reads against
    /// the memory-mapped region. Use `#[from]` conversion so `?` works on
    /// any `io::Result`.
    #[error("I/O error on binary file: {0}")]
    BinaryIo(#[from] io::Error),

    /// Error if counts do not seem to be raw counts during i/o
    #[error("The counts you are trying to load in do not seem to be raw counts")]
    NotRawCounts,

    /// Bincode failed to encode the `FileHeader` or `SparseDataHeader`.
    ///
    /// Practically unreachable for the fixed-layout headers used here, but
    /// kept as a non-panicking path so the writer never aborts a long ingest.
    #[error("Failed to encode file header")]
    HeaderEncodeFailed,

    /// Bincode failed to decode the `FileHeader` or `SparseDataHeader`.
    ///
    /// Indicates the file is truncated, corrupt, or was not produced by
    /// bixverse. The 64-byte fixed file header is the first thing read, so
    /// most "wrong file type" errors surface here.
    #[error("Failed to decode file header - file may be corrupt or truncated")]
    HeaderDecodeFailed,

    /// File version does not match the current `SC_FILE_VERSION`.
    ///
    /// Returned on read when an older or newer binary is opened. The fix is
    /// to regenerate the file with the current bixverse version.
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
    #[error("Failed to decompress chunk at offset {0}")]
    ChunkDecompressionFailed(u64),

    /// A caller-supplied original index was not present in the file's index
    /// map.
    ///
    /// Indicates either a bug in the calling code (asking for an index that
    /// was never written) or a file that was generated with a different set
    /// of cells/genes than the caller assumes.
    #[error("Chunk index {0} not found in file index map")]
    ChunkIndexNotFound(usize),

    /// A cell-based reader method was called on a gene-based file, or vice
    /// versa.
    ///
    /// Replaces the previous runtime asserts. The two string fields describe
    /// the file's actual layout and the layout the caller requested.
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
    #[error("Invalid raw count element size discriminant: {0}")]
    RawElemSizeInvalid(u8),

    /// A v4 chunk is malformed: a bad codec byte, a bit width above 32, a
    /// body that ends before its own header says it should, or layers of
    /// unequal length handed to the encoder.
    #[error("Corrupt v4 chunk: {0}")]
    V4ChunkCorrupt(String),

    /// A v4 chunk drops its norm layer and recomputes it per entry, but the
    /// file carries no per-cell library-size table to recompute it from.
    #[error("v4 gene chunk needs the per-cell library-size table, which the file does not carry")]
    V4MissingCellLibraryTable,

    /// The file is not a bixverse count archive.
    #[error("Not a bixverse count archive (magic string mismatch)")]
    ArchiveMagicMismatch,

    /// The archive was written by an incompatible archive format version.
    #[error("Archive version mismatch: expected {expected}, got {found}")]
    ArchiveVersionMismatch {
        /// Version the current build expects.
        expected: u32,
        /// Version actually read from the archive.
        found: u32,
    },

    /// The archive is truncated or corrupt, or a chunk cannot be archived.
    #[error("Corrupt count archive: {0}")]
    ArchiveCorrupt(String),

    /// A raw count read from disk does not fit the requested numeric type.
    ///
    /// Raised by `from_gene_chunks` / `from_cell_chunks` instead of silently
    /// saturating. Pick a wider `T` (`u32`, `f32`, `f64`) for the affected
    /// dataset.
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
    #[error("h5: Unexpected str type identified: {0}")]
    H5UnexpectedStringType(String),

    // -- HDF5 / h5ad --
    /// Wraps any error from the `hdf5` crate.
    ///
    /// Covers file open failures, missing datasets (`X/data`, `X/indices`,
    /// `X/indptr`, `obs/*`), dtype mismatches, and slice reads. The wrapped
    /// error carries the original HDF5 message.
    #[error("HDF5 error: {0}")]
    #[cfg(feature = "h5")]
    Hdf5(#[from] hdf5::Error),

    /// Provide custom error message for unsupported file formats.
    #[error("Unknown or unsupported h5ad format: {0}")]
    UnsupportH5ADFormat(String),

    /// Two columns of an obs / var table end up with the same name after
    /// snake_case. DuckDB rejects the table, so the export stops before
    /// writing anything.
    #[error("Duplicate column name after snake_case: '{0}'")]
    DuplicateColumnName(String),

    /// Wraps any error from the `parquet` crate while writing an obs / var
    /// table.
    #[cfg(feature = "parquet")]
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// A dataset is compressed with a filter this build cannot decode.
    ///
    /// Raised before the first read, because HDF5 itself reports the missing
    /// filter as a missing plugin directory: the path baked in at build time
    /// is a cargo `OUT_DIR` that no longer exists. Either re-save the file
    /// with `compression="gzip"`, or build with the `hdf5-filters` feature,
    /// which adds LZF and Blosc.
    #[error(
        "HDF5 dataset '{dataset}' uses filter {filter_id} ({filter_name}), which this build cannot decode"
    )]
    UnsupportedH5Filter {
        /// Path of the dataset inside the file.
        dataset: String,
        /// The HDF5 filter identifier.
        filter_id: i32,
        /// Human readable name for that identifier.
        filter_name: String,
    },

    /// A chunk read directly from an HDF5 file did not decode to the size its
    /// dataset says it should, or a read asked for elements past the end.
    #[error("HDF5 chunk decode error: {0}")]
    H5ChunkDecode(String),

    /// A named `obs` column was requested but does not exist in the h5ad.
    ///
    /// Primarily raised by `write_h5_normalised_counts` when the caller
    /// passes an `obs_lib_size_col` that isn't in the file.
    #[error("obs column '{0}' not found in h5ad file")]
    ObsColumnMissing(String),

    /// The library-size vector read from `obs` does not match the number of
    /// cells in `X`.
    ///
    /// Indicates a malformed h5ad — typical cause is a user-edited obs table
    /// whose row count drifted from the matrix.
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
    #[error("Invalid MTX header: {0}")]
    MtxHeaderInvalid(&'static str),

    /// A field of the MTX shape line could not be parsed as a number.
    ///
    /// `field` names the offending column ("cell count", "gene count",
    /// "entry count") so that malformed inputs can be diagnosed without
    /// quoting user data. Which of the first two fields is which depends on
    /// the `cells_as_rows` orientation.
    #[error("Failed to parse MTX value as {field}")]
    MtxParseError {
        /// Which field failed: "cell count", "gene count", or "entry count".
        field: &'static str,
    },

    // -- Format parsing --
    /// The `cs_type` string did not match a known sparse format.
    ///
    /// Accepted values are "csc" and "csr" (case-insensitive at the parser
    /// level). The unrecognised input is echoed back in the message.
    #[error("Unknown compressed sparse format: '{0}' (expected 'csc' or 'csr')")]
    UnknownSparseFormat(String),
}
