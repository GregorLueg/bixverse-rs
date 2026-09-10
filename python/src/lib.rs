//! Python bindings for `bixverse-rs`.
//!
//! The surface is deliberately small: open a cell-major sparse store, and
//! stream minibatches out of it fast enough to keep a GPU training loop fed.
//! Everything numerical happens in the crate; this layer only moves buffers
//! across the FFI boundary and maps errors onto exceptions.

use pyo3::prelude::*;

mod batch;
mod error;
mod loader;
mod reader;
mod testing;

/// Register the extension module.
///
/// ### Params
///
/// * `m` - The module object pyo3 hands us.
///
/// ### Returns
///
/// `Ok(())` once every class, exception and constant is registered.
#[pymodule]
fn _bixverse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__core_version__", bixverse_rs::VERSION)?;

    m.add("BixverseError", m.py().get_type::<error::BixverseError>())?;
    m.add("SparseFileError", m.py().get_type::<error::SparseFileError>())?;

    m.add_class::<reader::PyCellReader>()?;
    m.add_class::<loader::PyCellLoader>()?;
    m.add_class::<loader::PyCellLoaderIter>()?;

    m.add_function(wrap_pyfunction!(testing::write_synthetic_store, m)?)?;

    Ok(())
}
