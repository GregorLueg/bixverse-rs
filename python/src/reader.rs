//! The reader handle exposed to Python.
//!
//! A thin wrapper over [`ParallelSparseReader`]. It opens the memory-mapped
//! store, caches the three header scalars a loader needs, and hands the reader
//! out behind an [`Arc`] so the prefetch thread can share it.

use std::sync::Arc;

use bixverse_rs::single_cell::sc_data::data_io::{ParallelSparseReader, SingleCellReading};
use pyo3::prelude::*;

use crate::error::BvErr;

////////////////
// PyCellReader //
////////////////

/// Handle on a bixverse binary sparse store.
#[pyclass(module = "bixverse._bixverse", name = "CellReader", frozen)]
pub struct PyCellReader {
    /// The memory-mapped reader, shared with any prefetch thread.
    pub(crate) inner: Arc<ParallelSparseReader>,
    /// Total cells in the store, cached from the header.
    pub(crate) total_cells: usize,
    /// Total genes in the store, cached from the header.
    pub(crate) total_genes: usize,
    /// Whether the store is cell-major (CSR) rather than gene-major (CSC).
    pub(crate) cell_based: bool,
}

#[pymethods]
impl PyCellReader {
    /// Open a store.
    ///
    /// ### Params
    ///
    /// * `path` - Path to a `.bin` written by the crate's sparse writer.
    ///
    /// ### Returns
    ///
    /// The reader, or an exception if the file is missing, foreign, truncated
    /// or of a different format version.
    #[new]
    fn new(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py.detach(|| ParallelSparseReader::new(path)).map_err(BvErr)?;
        let header = inner.get_header();

        Ok(Self {
            total_cells: header.total_cells,
            total_genes: header.total_genes,
            cell_based: inner.is_cell_based(),
            inner: Arc::new(inner),
        })
    }

    /// Total number of cells in the store.
    #[getter]
    fn total_cells(&self) -> usize {
        self.total_cells
    }

    /// Total number of genes in the store.
    #[getter]
    fn total_genes(&self) -> usize {
        self.total_genes
    }

    /// Whether the store is cell-major. Only cell-major stores can be looped.
    #[getter]
    fn is_cell_based(&self) -> bool {
        self.cell_based
    }

    /// Library size the normalised layer was scaled against, when recorded.
    #[getter]
    fn target_size(&self) -> Option<f32> {
        self.inner.target_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "CellReader(total_cells={}, total_genes={}, cell_based={})",
            self.total_cells, self.total_genes, self.cell_based
        )
    }
}
