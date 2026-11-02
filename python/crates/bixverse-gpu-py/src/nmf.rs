//! HALS NMF on the GPU over single cells and meta cells, the twins of
//! `rs_nmf_single_sc_gpu`, `rs_nmf_multi_sc_gpu`, `rs_nmf_consensus_sc_gpu`,
//! `rs_nmf_k_sweep_sc_gpu` and their `_mc` siblings.
//!
//! Same signatures and the same result dicts as the CPU twins. Only the HALS
//! solver runs on the device; the consensus step stays on the host.

use bixverse_gpu::methods_gpu::nmf_consensus_gpu::{
    nmf_consensus_run_sparse_gpu, nmf_k_sweep_run_sparse_gpu, nmf_multiple_run_sparse_gpu,
    nmf_single_run_sparse_gpu,
};
use bixverse_sc::prelude::CompressedSparseData2;
use bixverse_sc_data::data_io::{DataLayerReturn, ParallelSparseReader, SingleCellReading};
use indexmap::IndexSet;
use numpy::PyReadonlyArray1;
use py_common::convert::{csr_from_parts, index_vec};
use py_common::error::BvErr;
use py_common::results::{
    consensus_params, hals_opts, nmf_consensus_dict, nmf_multi_dict, nmf_single_dict,
    nmf_sweep_dict,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::device::{ensure_gpu, on_device};
use crate::error::GpuErr;

/////////////
// Helpers //
/////////////

/// The cells x genes matrix of a selection, read from the gene-based store
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `gene_indices` - 0-based genes
/// * `cell_indices` - 0-based cells
/// * `use_second_layer` - Read the normalised rather than the raw counts
///
/// ### Returns
///
/// The CSC matrix.
fn load_sc(
    f_path_gene: &str,
    gene_indices: &PyReadonlyArray1<'_, u32>,
    cell_indices: &PyReadonlyArray1<'_, u32>,
    use_second_layer: bool,
) -> PyResult<CompressedSparseData2<f32>> {
    let genes = index_vec(gene_indices)?;
    let cells: IndexSet<u32> = cell_indices.as_slice()?.iter().copied().collect();
    let layer = if use_second_layer {
        DataLayerReturn::Norm
    } else {
        DataLayerReturn::Raw
    };
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    Ok(reader
        .read_genes_sparse(&genes, Some(&cells), &layer)
        .map_err(BvErr::from)?)
}

/// A meta cell matrix from its CSR parts
///
/// ### Params
///
/// * `data` - CSR values
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
///
/// ### Returns
///
/// The CSR matrix, the values in both layers.
fn load_mc(
    data: &PyReadonlyArray1<'_, f32>,
    indices: &PyReadonlyArray1<'_, u32>,
    indptr: &PyReadonlyArray1<'_, u32>,
    shape: (usize, usize),
) -> PyResult<CompressedSparseData2<f32>> {
    csr_from_parts((data, indices, indptr, shape), |x| x)
}

//////////////////
// Single cells //
//////////////////

/// One NMF run over single cells on the GPU
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `gene_indices` - 0-based genes
/// * `cell_indices` - 0-based cells
/// * `k` - Number of components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Factorise the normalised counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `seed` - Seed for a random initialisation
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`nmf_single_dict`].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k, preprocessing, use_second_layer, params,
    seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_single_sc<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    gene_indices: PyReadonlyArray1<'py, u32>,
    cell_indices: PyReadonlyArray1<'py, u32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let data = load_sc(f_path_gene, &gene_indices, &cell_indices, use_second_layer)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_single_run_sparse_gpu(
                    data,
                    k,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_single_dict(py, res)
}

/// NMF restarts over single cells on the GPU
///
/// ### Params
///
/// As [`nmf_single_sc`], plus `n_runs`, the number of restarts.
///
/// ### Returns
///
/// See [`nmf_multi_dict`].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k, preprocessing, use_second_layer, params,
    n_runs, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_multi_sc<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    gene_indices: PyReadonlyArray1<'py, u32>,
    cell_indices: PyReadonlyArray1<'py, u32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let data = load_sc(f_path_gene, &gene_indices, &cell_indices, use_second_layer)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_multiple_run_sparse_gpu(
                    data,
                    k,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    n_runs,
                    seed,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_multi_dict(py, res)
}

/// Consensus NMF over single cells, the restarts on the GPU
///
/// ### Params
///
/// As [`nmf_multi_sc`], plus `consensus`, `to_rust(NmfConsensusParams)`.
///
/// ### Returns
///
/// See [`nmf_consensus_dict`].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k, preprocessing, use_second_layer, params,
    consensus, n_runs, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_sc<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    gene_indices: PyReadonlyArray1<'py, u32>,
    cell_indices: PyReadonlyArray1<'py, u32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    consensus: &Bound<'py, PyAny>,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let data = load_sc(f_path_gene, &gene_indices, &cell_indices, use_second_layer)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_consensus_run_sparse_gpu(
                    data,
                    k,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    Some(cons),
                    n_runs,
                    seed,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_consensus_dict(py, res)
}

/// Consensus NMF over a range of k for single cells, the restarts on the GPU
///
/// ### Params
///
/// As [`nmf_consensus_sc`], with `k_range` in place of `k`.
///
/// ### Returns
///
/// See [`nmf_sweep_dict`].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k_range, preprocessing, use_second_layer,
    params, consensus, n_runs, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_k_sweep_sc<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    gene_indices: PyReadonlyArray1<'py, u32>,
    cell_indices: PyReadonlyArray1<'py, u32>,
    k_range: Vec<usize>,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    consensus: &Bound<'py, PyAny>,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let data = load_sc(f_path_gene, &gene_indices, &cell_indices, use_second_layer)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_k_sweep_run_sparse_gpu(
                    data,
                    &k_range,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    Some(cons),
                    n_runs,
                    seed,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_sweep_dict(py, res)
}

////////////////
// Meta cells //
////////////////

/// One NMF run over meta cells on the GPU
///
/// ### Params
///
/// * `data` - CSR values of the selected layer
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `k` - Number of components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Factorise the second layer
/// * `params` - `to_rust(NmfHalsParams)`
/// * `seed` - Seed for a random initialisation
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`nmf_single_dict`].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k, preprocessing, use_second_layer, params, seed = 42,
    verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_single_mc<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let m = load_mc(&data, &indices, &indptr, shape)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_single_run_sparse_gpu(
                    m,
                    k,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_single_dict(py, res)
}

/// NMF restarts over meta cells on the GPU
///
/// ### Params
///
/// As [`nmf_single_mc`], plus `n_runs`, the number of restarts.
///
/// ### Returns
///
/// See [`nmf_multi_dict`].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k, preprocessing, use_second_layer, params, n_runs,
    seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_multi_mc<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let m = load_mc(&data, &indices, &indptr, shape)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_multiple_run_sparse_gpu(
                    m,
                    k,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    n_runs,
                    seed,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_multi_dict(py, res)
}

/// Consensus NMF over meta cells, the restarts on the GPU
///
/// ### Params
///
/// As [`nmf_multi_mc`], plus `consensus`, `to_rust(NmfConsensusParams)`.
///
/// ### Returns
///
/// See [`nmf_consensus_dict`].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k, preprocessing, use_second_layer, params, consensus,
    n_runs, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_mc<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    consensus: &Bound<'py, PyAny>,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let m = load_mc(&data, &indices, &indptr, shape)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_consensus_run_sparse_gpu(
                    m,
                    k,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    Some(cons),
                    n_runs,
                    seed,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_consensus_dict(py, res)
}

/// Consensus NMF over a range of k for meta cells, the restarts on the GPU
///
/// ### Params
///
/// As [`nmf_consensus_mc`], with `k_range` in place of `k`.
///
/// ### Returns
///
/// See [`nmf_sweep_dict`].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k_range, preprocessing, use_second_layer, params,
    consensus, n_runs, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn nmf_k_sweep_mc<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    k_range: Vec<usize>,
    preprocessing: &str,
    use_second_layer: bool,
    params: &Bound<'py, PyAny>,
    consensus: &Bound<'py, PyAny>,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let m = load_mc(&data, &indices, &indptr, shape)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                nmf_k_sweep_run_sparse_gpu(
                    m,
                    &k_range,
                    preprocessing,
                    use_second_layer,
                    Some(opts),
                    Some(cons),
                    n_runs,
                    seed,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    nmf_sweep_dict(py, res)
}
