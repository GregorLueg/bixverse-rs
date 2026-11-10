//! Mapping [`ScGpuErrors`] onto Python exceptions.
//!
//! The classes are the `bixverse` extension's, imported at run time, so a GPU
//! error is caught by the same `except bixverse.BixverseError` as its CPU twin.
//! Everything the CPU layers raise goes through the shared mapping in
//! `py_common::error`.

use bixverse_gpu::errors::GpuErrors;
use bixverse_sc_gpu::errors::ScGpuErrors;
use py_common::error::{data_err, sc_err};
use pyo3::prelude::*;

pyo3::import_exception!(bixverse._bixverse, BixverseError);
pyo3::import_exception!(bixverse._bixverse, SparseFileError);

/// Carries a [`ScGpuErrors`] to the FFI boundary.
///
/// `.map_err(GpuErr::from)?` in any function returning [`PyResult`].
pub(crate) struct GpuErr(
    /// The error the crate raised.
    pub ScGpuErrors,
);

impl From<ScGpuErrors> for GpuErr {
    fn from(e: ScGpuErrors) -> Self {
        Self(e)
    }
}

impl From<GpuErrors> for GpuErr {
    fn from(e: GpuErrors) -> Self {
        Self(ScGpuErrors::Gpu(e))
    }
}

impl From<GpuErr> for PyErr {
    fn from(e: GpuErr) -> PyErr {
        match e.0 {
            ScGpuErrors::Sc(e) => sc_err::<BixverseError, SparseFileError>(e),
            ScGpuErrors::ScData(e) => data_err::<BixverseError, SparseFileError>(e),
            other => BixverseError::new_err(other.to_string()),
        }
    }
}
