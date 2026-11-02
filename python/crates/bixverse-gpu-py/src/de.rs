//! NEBULA with the per-gene fits on the GPU, the twin of `rs_nebula_sc_gpu`.

use bixverse_sc::sc_analysis::nebula::NebulaScParams;
use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_gpu::nebula_gpu::run_nebula_gpu;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::{from_py, index_vec};
use py_common::error::BvErr;
use py_common::results::nebula_dict;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::device::{ensure_plane_ops, on_device};
use crate::error::GpuErr;

/// NEBULA over a cell selection, stage two on the GPU in `f32`
///
/// Same signature as the CPU `nebula`. `reml` has no GPU path.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the library sizes
/// * `cells_to_keep` - 0-based cells, no duplicates
/// * `gene_indices` - 0-based genes to fit
/// * `subject_ids` - 0-based subject per selected cell
/// * `design` - `(n_cells, n_coef)` `float64` design, C-contiguous
/// * `params` - `to_rust(NebulaGpuParams)` plus `coef` (0-based) or
///   `contrast`
/// * `offset` - Strictly positive scaling per selected cell; `None` uses the
///   library sizes
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`nebula_dict`].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cells_to_keep, gene_indices, subject_ids, design, params,
    offset = None, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nebula<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    subject_ids: PyReadonlyArray1<'py, u32>,
    design: PyReadonlyArray2<'py, f64>,
    params: &Bound<'py, PyAny>,
    offset: Option<PyReadonlyArray1<'py, f64>>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_plane_ops()?;
    let params: NebulaScParams = from_py(params, "NEBULA params")?;
    let cells = index_vec(&cells_to_keep)?;
    let genes = index_vec(&gene_indices)?;
    let subjects = index_vec(&subject_ids)?;
    let n_coef = design.shape()[1];
    let design = design.as_slice()?.to_vec();
    let offset = offset
        .map(|o| o.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let gene_reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let cell_reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                run_nebula_gpu(
                    &gene_reader,
                    &cell_reader,
                    &cells,
                    &genes,
                    &subjects,
                    &design,
                    n_coef,
                    offset.as_deref(),
                    &params,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nebula_dict(py, res)
}
