//! Mapping [`ScErrors`] onto Python exceptions.
//!
//! No new error enum: the crates already have one each and these bindings add
//! no failure modes of their own beyond what pyo3 raises directly. `ScErrors`
//! wraps every lower crate's enum, so one newtype over it covers the lot.
//! `ScErrors` and `PyErr` are both foreign here, so the `From` impl needs a
//! local newtype.

use bixverse_sc::prelude::ScErrors;
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

/// Carries a [`ScErrors`] to the FFI boundary.
///
/// `.map_err(BvErr::from)?` in any function returning [`PyResult`], from
/// either the analysis or the data layer error.
pub(crate) struct BvErr(
    /// The error the crate raised.
    pub ScErrors,
);

impl From<ScErrors> for BvErr {
    fn from(e: ScErrors) -> Self {
        Self(e)
    }
}

impl From<ScDataErrors> for BvErr {
    fn from(e: ScDataErrors) -> Self {
        Self(ScErrors::ScData(e))
    }
}

impl From<BvErr> for PyErr {
    fn from(e: BvErr) -> PyErr {
        match e.0 {
            ScErrors::ScData(data) => data_err(data),
            // Everything else is a precondition the analysis checked: bad
            // shapes, out-of-range indices, degenerate inputs.
            other => BixverseError::new_err(other.to_string()),
        }
    }
}

/// The Python exception for a data layer error.
///
/// ### Params
///
/// * `e` - The data layer error.
///
/// ### Returns
///
/// The matching exception: filesystem errors as `OSError` subclasses, damaged
/// or mismatched stores as `SparseFileError`, the rest as `BixverseError`.
fn data_err(e: ScDataErrors) -> PyErr {
    let msg = e.to_string();
    match e {
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

        _ => BixverseError::new_err(msg),
    }
}
