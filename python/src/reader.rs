//! The reader handle exposed to Python.
//!
//! A thin wrapper over [`ParallelSparseReader`]. It opens the memory-mapped
//! store, caches the three header scalars a loader needs, and hands the reader
//! out behind an [`Arc`] so the prefetch thread can share it. `read_csr` reads
//! one cell selection into a single CSR matrix, for `get_counts`.

use std::sync::Arc;

use bixverse_sc_data::data_io::{ParallelSparseReader, SingleCellReading};
use bixverse_sc_data::errors::ScDataErrors;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::batch::{Layer, adopt_csr, gene_remap, subset_csr};
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
    /// or of an unsupported format version. Stores of file version 3 and 4
    /// both open.
    #[new]
    fn new(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| ParallelSparseReader::new(path))
            .map_err(BvErr::from)?;
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

    /// Read a set of cells into one CSR matrix.
    ///
    /// The whole selection is decoded in one call to the crate's fused CSR
    /// reader, rayon parallel over cells, with the GIL released throughout.
    /// A gene subset is applied afterwards in one pass over the non-zeros.
    ///
    /// ### Params
    ///
    /// * `cells` - 0-based store cell indices, in the row order wanted.
    /// * `layer` - `"raw"` or `"norm"`.
    /// * `genes` - Optional 0-based store gene indices to keep, in the column
    ///   order wanted. `None` keeps every gene.
    ///
    /// ### Returns
    ///
    /// `(data, indices, indptr, (n_rows, n_cols))`: `f32` values, `uint32`
    /// column indices and `uint32` row pointers. An exception for a gene-major
    /// store, an unknown layer, or an out-of-range or repeated index.
    #[pyo3(signature = (cells, layer = "raw", genes = None))]
    fn read_csr<'py>(
        &self,
        py: Python<'py>,
        cells: PyReadonlyArray1<'py, u32>,
        layer: &str,
        genes: Option<PyReadonlyArray1<'py, u32>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if !self.cell_based {
            return Err(BvErr::from(ScDataErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            })
            .into());
        }
        let layer = match Layer::parse(layer).map_err(PyValueError::new_err)? {
            Layer::Both => {
                return Err(PyValueError::new_err(
                    "read_csr returns one layer: 'raw' or 'norm'",
                ));
            }
            l => l,
        };

        let total_cells = self.total_cells;
        let cells: Vec<usize> = cells.as_slice()?.iter().map(|&i| i as usize).collect();
        if let Some(&bad) = cells.iter().find(|&&i| i >= total_cells) {
            return Err(PyValueError::new_err(format!(
                "cell index {bad} is out of range for a store of {total_cells} cells"
            )));
        }
        let remap = genes
            .map(|g| gene_remap(g.as_slice()?, self.total_genes).map_err(PyValueError::new_err))
            .transpose()?;

        let reader = Arc::clone(&self.inner);
        let batch = py
            .detach(|| {
                reader
                    .read_cells_csr(&cells, &layer.as_data_layer())
                    .map(|b| match remap.as_ref() {
                        Some(r) => subset_csr(adopt_csr(b), r),
                        None => adopt_csr(b),
                    })
            })
            .map_err(BvErr::from)?;

        let shape = (batch.n_rows, batch.n_cols);
        let data = match layer {
            Layer::Raw => batch.raw,
            _ => batch.norm,
        }
        .expect("the requested layer is always materialised");
        let indices = batch.indices.expect("the CSR path always sets indices");
        let indptr = batch.indptr.expect("the CSR path always sets indptr");

        (
            data.into_pyarray(py),
            indices.into_pyarray(py),
            indptr.into_pyarray(py),
            shape,
        )
            .into_pyobject(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CellReader(total_cells={}, total_genes={}, cell_based={})",
            self.total_cells, self.total_genes, self.cell_based
        )
    }
}
