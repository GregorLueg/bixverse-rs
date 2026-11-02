//! SCENIC GRN inference with the tree ensembles on the GPU, the twins of
//! `rs_scenic_grn_gpu`, `rs_scenic_grn_streaming_gpu` and `rs_mc_scenic_gpu`.
//!
//! Same signatures as the CPU twins plus a trailing `wave_byte_budget`.
//! Gradient boosting has no GPU path and errors.

use bixverse_sc::sc_analysis::scenic::ScenicParams;
use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_gpu::scenic_gpu::{
    ScenicGpuParams, run_scenic_grn_gpu, run_scenic_grn_in_memory_gpu, run_scenic_grn_streaming_gpu,
};
use numpy::PyReadonlyArray1;
use py_common::convert::{csr_from_parts, from_py, index_vec, mat_f64};
use py_common::error::BvErr;
use pyo3::prelude::*;

use crate::device::{ensure_gpu, on_device};
use crate::error::GpuErr;

/// R's default VRAM budget per wave of trees, 4 GiB.
const DEFAULT_WAVE_BYTE_BUDGET: usize = 4 * 1024 * 1024 * 1024;

/// SCENIC GRN inference over a cell-major selection on the GPU
///
/// ### Params
///
/// * `f_path_genes` - The gene-based binary
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based target genes
/// * `tf_indices` - 0-based TF genes, also targets
/// * `params` - `to_rust(ScenicParams)`
/// * `streaming` - Stream the targets
/// * `seed` - Seed for the learners
/// * `verbose` - Controls verbosity
/// * `wave_byte_budget` - VRAM budget per wave of trees, in bytes
///
/// ### Returns
///
/// `(n_targets, n_tfs)` `float64` importances.
#[pyfunction]
#[pyo3(signature = (
    f_path_genes, cell_indices, gene_indices, tf_indices, params, streaming, seed = 42,
    verbose = 1, wave_byte_budget = DEFAULT_WAVE_BYTE_BUDGET
))]
#[allow(clippy::too_many_arguments)]
pub fn scenic_grn<'py>(
    py: Python<'py>,
    f_path_genes: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    tf_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    streaming: bool,
    seed: usize,
    verbose: usize,
    wave_byte_budget: usize,
) -> PyResult<Bound<'py, PyAny>> {
    ensure_gpu()?;
    let params: ScenicParams = from_py(params, "SCENIC params")?;
    let gpu = ScenicGpuParams { wave_byte_budget };
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let tfs = index_vec(&tf_indices)?;
    let reader = ParallelSparseReader::new(f_path_genes).map_err(BvErr::from)?;
    let v = verbose;
    let res = py
        .detach(|| {
            on_device(|device| {
                if streaming {
                    run_scenic_grn_streaming_gpu(
                        &reader, &cells, &genes, &tfs, &params, &gpu, seed, device, v,
                    )
                } else {
                    run_scenic_grn_gpu(
                        &reader, &cells, &genes, &tfs, &params, &gpu, seed, device, v,
                    )
                }
            })
        })
        .map_err(GpuErr::from)?;
    mat_f64(py, &res)
}

/// SCENIC GRN inference over meta cells on the GPU
///
/// ### Params
///
/// * `data` - CSR raw counts, columns the target genes
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_targets)`
/// * `tf_indices` - 0-based TF columns of the matrix
/// * `params` - `to_rust(ScenicParams)`
/// * `seed` - Seed for the learners
/// * `verbose` - Controls verbosity
/// * `wave_byte_budget` - VRAM budget per wave of trees, in bytes
///
/// ### Returns
///
/// `(n_targets, n_tfs)` `float64` importances.
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, tf_indices, params, seed = 42, verbose = 1,
    wave_byte_budget = DEFAULT_WAVE_BYTE_BUDGET
))]
#[allow(clippy::too_many_arguments)]
pub fn mc_scenic_grn<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    tf_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
    wave_byte_budget: usize,
) -> PyResult<Bound<'py, PyAny>> {
    ensure_gpu()?;
    let params: ScenicParams = from_py(params, "SCENIC params")?;
    let gpu = ScenicGpuParams { wave_byte_budget };
    let tfs = index_vec(&tf_indices)?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                run_scenic_grn_in_memory_gpu(&m, &tfs, &params, &gpu, seed, device, verbose)
            })
        })
        .map_err(GpuErr::from)?;
    mat_f64(py, &res)
}
