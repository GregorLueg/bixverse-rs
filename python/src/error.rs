//! Mapping [`BixverseErrors`] onto Python exceptions.
//!
//! No new error enum: the crate already has one and these bindings add no
//! failure modes of their own beyond what pyo3 raises directly.
//! `BixverseErrors` and `PyErr` are both foreign here, so the `From` impl
//! needs a local newtype.

use bixverse_rs::errors::BixverseErrors;
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

/// Carries a [`BixverseErrors`] to the FFI boundary.
///
/// A tuple struct, which means it doubles as the conversion function:
/// `.map_err(BvErr)?` in any method returning [`PyResult`].
pub(crate) struct BvErr(
    /// The error the crate raised.
    pub BixverseErrors,
);

impl From<BixverseErrors> for BvErr {
    fn from(e: BixverseErrors) -> Self {
        Self(e)
    }
}

impl From<BvErr> for PyErr {
    fn from(e: BvErr) -> PyErr {
        let msg = e.0.to_string();
        match e.0 {
            // Filesystem.
            BixverseErrors::BinaryIo(ref io) if io.kind() == std::io::ErrorKind::NotFound => {
                PyFileNotFoundError::new_err(msg)
            }
            BixverseErrors::BinaryIo(_) => PyOSError::new_err(msg),

            // A store that is not what we asked for, or is damaged.
            BixverseErrors::FileVersionMismatch { .. }
            | BixverseErrors::HeaderDecodeFailed
            | BixverseErrors::HeaderEncodeFailed
            | BixverseErrors::ChunkBufferTooSmall { .. }
            | BixverseErrors::ChunkPayloadTruncated { .. }
            | BixverseErrors::ChunkDecompressionFailed(_)
            | BixverseErrors::RawElemSizeInvalid(_)
            | BixverseErrors::ReaderModeMismatch { .. } => SparseFileError::new_err(msg),

            // The caller asked for a cell the store does not hold.
            BixverseErrors::ChunkIndexNotFound(_) => PyValueError::new_err(msg),

            // `BixverseErrors` covers the whole crate, most of it unreachable
            // from these bindings. The catch-all is required, not laziness.
            _ => BixverseError::new_err(msg),
        }
    }
}
