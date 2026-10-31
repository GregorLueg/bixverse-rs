//! Mapping [`ScErrors`] onto Python exceptions.
//!
//! No new error enum: the crates already have one each and these bindings add
//! no failure modes of their own beyond what pyo3 raises directly. `ScErrors`
//! wraps every lower crate's enum, so one newtype over it covers the lot.
//! `ScErrors` and `PyErr` are both foreign here, so the `From` impl needs a
//! local newtype.
//!
//! The exception classes are created by the `bixverse` extension. The GPU
//! extension raises the same classes, imported at run time, so the mapping is
//! generic over the two class types rather than tied to these.

use bixverse_sc::prelude::ScErrors;
use bixverse_sc_data::errors::ScDataErrors;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;

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

/// Carries a [`ScErrors`] to the FFI boundary of the `bixverse` extension.
///
/// `.map_err(BvErr::from)?` in any function returning [`PyResult`], from
/// either the analysis or the data layer error.
pub struct BvErr(
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
        sc_err::<BixverseError, SparseFileError>(e.0)
    }
}

/////////////
// Mapping //
/////////////

/// The Python exception for an analysis layer error.
///
/// ### Params
///
/// * `e` - The error.
///
/// ### Returns
///
/// The data layer mapping for a wrapped [`ScDataErrors`], else `B`: every
/// other variant is a precondition the analysis checked (bad shapes,
/// out-of-range indices, degenerate inputs).
pub fn sc_err<B: PyTypeInfo, S: PyTypeInfo>(e: ScErrors) -> PyErr {
    match e {
        ScErrors::ScData(data) => data_err::<B, S>(data),
        other => PyErr::new::<B, _>(other.to_string()),
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
/// or mismatched stores as `S`, the rest as `B`.
pub fn data_err<B: PyTypeInfo, S: PyTypeInfo>(e: ScDataErrors) -> PyErr {
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
        | ScDataErrors::ReaderModeMismatch { .. } => PyErr::new::<S, _>(msg),

        // The caller asked for a cell the store does not hold.
        ScDataErrors::ChunkIndexNotFound(_) => PyValueError::new_err(msg),

        _ => PyErr::new::<B, _>(msg),
    }
}
