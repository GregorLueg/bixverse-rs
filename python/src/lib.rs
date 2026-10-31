//! Python bindings for `bixverse-rs`.
//!
//! Open a cell-major sparse store and stream minibatches out of it fast enough
//! to keep a GPU training loop fed, and ingest h5ad, mtx and 10x h5 data into
//! such stores. Everything numerical happens in the crates; this layer only
//! moves buffers across the FFI boundary and maps errors onto exceptions.

use pyo3::prelude::*;

mod batch;
mod error;
mod h5ad_write;
mod io;
mod loader;
mod reader;
mod synthetic;
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
    m.add("__core_version__", bixverse_sc_data::VERSION)?;

    m.add("BixverseError", m.py().get_type::<error::BixverseError>())?;
    m.add(
        "SparseFileError",
        m.py().get_type::<error::SparseFileError>(),
    )?;

    m.add_class::<reader::PyCellReader>()?;
    m.add_class::<loader::PyCellLoader>()?;
    m.add_class::<loader::PyCellLoaderIter>()?;

    m.add_function(wrap_pyfunction!(testing::write_synthetic_store, m)?)?;

    for f in [
        wrap_pyfunction!(io::csr_to_file, m)?,
        wrap_pyfunction!(io::h5ad_to_file, m)?,
        wrap_pyfunction!(io::norm_h5ad_to_file, m)?,
        wrap_pyfunction!(io::mtx_to_file, m)?,
        wrap_pyfunction!(io::tenx_h5_to_file, m)?,
        wrap_pyfunction!(io::multi_h5ad_to_file, m)?,
        wrap_pyfunction!(io::multi_mtx_to_file, m)?,
        wrap_pyfunction!(io::multi_tenx_h5_to_file, m)?,
        wrap_pyfunction!(io::merge_sc_files, m)?,
        wrap_pyfunction!(io::generate_gene_based_data, m)?,
        wrap_pyfunction!(io::gene_nnz, m)?,
        wrap_pyfunction!(io::archive_cells, m)?,
        wrap_pyfunction!(io::restore_cells_archive, m)?,
        wrap_pyfunction!(io::file_version, m)?,
        wrap_pyfunction!(io::upgrade_to_v4, m)?,
        wrap_pyfunction!(io::h5ad_obs_var_to_parquet, m)?,
        wrap_pyfunction!(io::tenx_obs_var_to_parquet, m)?,
        wrap_pyfunction!(io::snake_case, m)?,
        wrap_pyfunction!(io::h5ad_dimensions, m)?,
        wrap_pyfunction!(io::raw_count_slot, m)?,
        wrap_pyfunction!(io::h5_strings, m)?,
        wrap_pyfunction!(io::tenx_h5_metadata, m)?,
        wrap_pyfunction!(io::write_tenx, m)?,
        wrap_pyfunction!(h5ad_write::save_h5ad, m)?,
        wrap_pyfunction!(h5ad_write::write_h5ad, m)?,
        wrap_pyfunction!(synthetic::synthetic_sc_data, m)?,
        wrap_pyfunction!(synthetic::synthetic_adt_data, m)?,
        wrap_pyfunction!(synthetic::synthetic_dialogue_data, m)?,
        wrap_pyfunction!(synthetic::synthetic_cellsweep_data, m)?,
    ] {
        m.add_function(f)?;
    }

    Ok(())
}
