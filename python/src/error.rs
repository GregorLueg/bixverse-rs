//! Mapping [`ScDataErrors`] onto Python exceptions.
//!
//! No new error enum: the data layer already has one and these bindings add
//! no failure modes of their own beyond what pyo3 raises directly.
//! `ScDataErrors` and `PyErr` are both foreign here, so the `From` impl
//! needs a local newtype.

use bixverse_sc_data::errors::ScDataErrors;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;

////////////////
// Exceptions //
////////////////

create_exception!(
    _bixverse,
    BixverseError,
    PyException,
    "Base class for every error raised by the bixverse Rust core."
);

create_exception!(
    _bixverse,
    SparseFileError,
    BixverseError,
    "A sparse store is missing, truncated, corrupt, or of the wrong format version or layout."
);

/////////////
// Newtype //
/////////////

/// Carries a [`ScDataErrors`] to the FFI boundary.
///
/// A tuple struct, which means it doubles as the conversion function:
/// `.map_err(BvErr)?` in any method returning [`PyResult`].
pub(crate) struct BvErr(
    /// The error the crate raised.
    pub ScDataErrors,
);

impl From<ScDataErrors> for BvErr {
    fn from(e: ScDataErrors) -> Self {
        Self(e)
    }
}

impl From<BvErr> for PyErr {
    fn from(e: BvErr) -> PyErr {
        let msg = e.0.to_string();
        match e.0 {
            // Filesystem.
            ScDataErrors::BinaryIo(ref io) if io.kind() == std::io::ErrorKind::NotFound => {
                PyFileNotFoundError::new_err(msg)
            }
            ScDataErrors::BinaryIo(_) => PyOSError::new_err(msg),

            // A store that is not what we asked for, or is damaged.
            ScDataErrors::FileVersionMismatch { .. }
            | ScDataErrors::HeaderDecodeFailed
            | ScDataErrors::HeaderEncodeFailed
            | ScDataErrors::ChunkBufferTooSmall { .. }
            | ScDataErrors::ChunkPayloadTruncated { .. }
            | ScDataErrors::ChunkDecompressionFailed(_)
            | ScDataErrors::RawElemSizeInvalid(_)
            | ScDataErrors::ReaderModeMismatch { .. } => SparseFileError::new_err(msg),

            // The caller asked for a cell the store does not hold.
            ScDataErrors::ChunkIndexNotFound(_) => PyValueError::new_err(msg),

            // The h5 and MTX ingestion variants and the wrapped core errors
            // are unreachable from these bindings. The catch-all is required.
            _ => BixverseError::new_err(msg),
        }
    }
}
