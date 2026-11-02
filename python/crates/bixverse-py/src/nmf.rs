//! HALS NMF over single cells and meta cells: the Python twins of
//! `rs_nmf_single_sc`, `rs_nmf_multi_sc`, `rs_nmf_consensus_sc`,
//! `rs_nmf_k_sweep_sc` and their `_mc` siblings.
//!
//! The numerics live in `bixverse-sc` and `bixverse-methods`; this only moves
//! buffers. Every index is 0-based. The crates factorise cells x genes, so
//! their `w` is cells x k and their `h` is k x genes; both come back as is and
//! the Python side flips them, as R's constructors do. A meta cell matrix
//! crosses as its CSR parts.

use bixverse_sc::mc_analysis::nmf_mc::{
    nmf_consensus_run_mc, nmf_k_sweep_run_mc, nmf_multiple_run_mc, nmf_single_run_mc,
};
use bixverse_sc::prelude::CompressedSparseData2;
use bixverse_sc::sc_analysis::nmf_sc::{
    nmf_consensus_run_sc, nmf_k_sweep_run_sc, nmf_multiple_run_sc, nmf_single_run_sc,
};
use bixverse_sc_data::data_io::ParallelSparseReader;
use numpy::PyReadonlyArray1;
use py_common::convert::{csr_from_parts, index_vec};
use py_common::error::BvErr;
use py_common::results::{
    consensus_params, hals_opts, nmf_consensus_dict, nmf_multi_dict, nmf_single_dict,
    nmf_sweep_dict,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/////////////
// Helpers //
/////////////

/// The meta cell matrix in `f32`, both layers the values handed over
///
/// ### Params
///
/// * `parts` - `(data, indices, indptr, shape)`
///
/// ### Returns
///
/// The CSR matrix.
fn mc_matrix(
    parts: (
        &PyReadonlyArray1<'_, f32>,
        &PyReadonlyArray1<'_, u32>,
        &PyReadonlyArray1<'_, u32>,
        (usize, usize),
    ),
) -> PyResult<CompressedSparseData2<f32>> {
    csr_from_parts(parts, |x| x)
}

//////////////////
// Single cells //
//////////////////

/// One HALS NMF run over the store
///
/// The twin of `rs_nmf_single_sc`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `gene_indices` - 0-based genes
/// * `cell_indices` - 0-based cells
/// * `k` - Components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Factorise the normalised rather than the raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `seed` - Seed for a random initialisation
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_single_dict].
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
    let opts = hals_opts(params, seed)?;
    let (genes, cells) = (index_vec(&gene_indices)?, index_vec(&cell_indices)?);
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            nmf_single_run_sc(
                &reader,
                &genes,
                &cells,
                k,
                preprocessing,
                use_second_layer,
                Some(opts),
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    nmf_single_dict(py, res)
}

/// Repeated randomly initialised HALS NMF runs over the store
///
/// The twin of `rs_nmf_multi_sc`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `gene_indices` - 0-based genes
/// * `cell_indices` - 0-based cells
/// * `k` - Components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Factorise the normalised rather than the raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `n_runs` - Restarts
/// * `seed` - Base seed of the restarts
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_multi_dict].
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
    let opts = hals_opts(params, seed)?;
    let (genes, cells) = (index_vec(&gene_indices)?, index_vec(&cell_indices)?);
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            nmf_multiple_run_sc(
                &reader,
                &genes,
                &cells,
                k,
                preprocessing,
                use_second_layer,
                Some(opts),
                n_runs,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    nmf_multi_dict(py, res)
}

/// Consensus HALS NMF over the store
///
/// The twin of `rs_nmf_consensus_sc`, the consensus seed injected from
/// `seed` as R does.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `gene_indices` - 0-based genes
/// * `cell_indices` - 0-based cells
/// * `k` - Components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Factorise the normalised rather than the raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `consensus` - `to_rust(NmfConsensusParams)`
/// * `n_runs` - Restarts
/// * `seed` - Base seed of the restarts and the consensus k-means
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_consensus_dict].
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
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let (genes, cells) = (index_vec(&gene_indices)?, index_vec(&cell_indices)?);
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            nmf_consensus_run_sc(
                &reader,
                &genes,
                &cells,
                k,
                preprocessing,
                use_second_layer,
                Some(opts),
                Some(cons),
                n_runs,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    nmf_consensus_dict(py, res)
}

/// Consensus HALS NMF over a range of k on the store
///
/// The twin of `rs_nmf_k_sweep_sc`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `gene_indices` - 0-based genes
/// * `cell_indices` - 0-based cells
/// * `k_range` - The ranks to evaluate
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Factorise the normalised rather than the raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `consensus` - `to_rust(NmfConsensusParams)`
/// * `n_runs` - Restarts per k
/// * `seed` - Base seed of the restarts and the consensus k-means
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_sweep_dict].
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
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let (genes, cells) = (index_vec(&gene_indices)?, index_vec(&cell_indices)?);
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            nmf_k_sweep_run_sc(
                &reader,
                &genes,
                &cells,
                &k_range,
                preprocessing,
                use_second_layer,
                Some(opts),
                Some(cons),
                n_runs,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    nmf_sweep_dict(py, res)
}

////////////////
// Meta cells //
////////////////

/// One HALS NMF run over meta cells
///
/// The twin of `rs_nmf_single_mc`.
///
/// ### Params
///
/// * `data` - CSR counts of the selected meta cells and genes, the layer
///   `use_second_layer` picks
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `k` - Components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Normalised (`True`) or raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `seed` - Seed for a random initialisation
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_single_dict].
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
    let opts = hals_opts(params, seed)?;
    let m = mc_matrix((&data, &indices, &indptr, shape))?;
    let res = py
        .detach(|| nmf_single_run_mc(m, k, preprocessing, use_second_layer, Some(opts), verbose))
        .map_err(BvErr::from)?;
    nmf_single_dict(py, res)
}

/// Repeated randomly initialised HALS NMF runs over meta cells
///
/// The twin of `rs_nmf_multi_mc`.
///
/// ### Params
///
/// * `data` - CSR counts, the layer `use_second_layer` picks
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `k` - Components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Normalised (`True`) or raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `n_runs` - Restarts
/// * `seed` - Base seed of the restarts
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_multi_dict].
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
    let opts = hals_opts(params, seed)?;
    let m = mc_matrix((&data, &indices, &indptr, shape))?;
    let res = py
        .detach(|| {
            nmf_multiple_run_mc(
                m,
                k,
                preprocessing,
                use_second_layer,
                Some(opts),
                n_runs,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    nmf_multi_dict(py, res)
}

/// Consensus HALS NMF over meta cells
///
/// The twin of `rs_nmf_consensus_mc`.
///
/// ### Params
///
/// * `data` - CSR counts, the layer `use_second_layer` picks
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `k` - Components
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Normalised (`True`) or raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `consensus` - `to_rust(NmfConsensusParams)`
/// * `n_runs` - Restarts
/// * `seed` - Base seed of the restarts and the consensus k-means
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_consensus_dict].
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
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let m = mc_matrix((&data, &indices, &indptr, shape))?;
    let res = py
        .detach(|| {
            nmf_consensus_run_mc(
                m,
                k,
                preprocessing,
                use_second_layer,
                Some(opts),
                Some(cons),
                n_runs,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    nmf_consensus_dict(py, res)
}

/// Consensus HALS NMF over a range of k on meta cells
///
/// The twin of `rs_nmf_k_sweep_mc`.
///
/// ### Params
///
/// * `data` - CSR counts, the layer `use_second_layer` picks
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `k_range` - The ranks to evaluate
/// * `preprocessing` - `"none"`, `"sd"` or `"sqrt_sd"`
/// * `use_second_layer` - Normalised (`True`) or raw counts
/// * `params` - `to_rust(NmfHalsParams)`
/// * `consensus` - `to_rust(NmfConsensusParams)`
/// * `n_runs` - Restarts per k
/// * `seed` - Base seed of the restarts and the consensus k-means
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [nmf_sweep_dict].
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
    let opts = hals_opts(params, seed)?;
    let cons = consensus_params(consensus, seed)?;
    let m = mc_matrix((&data, &indices, &indptr, shape))?;
    let res = py
        .detach(|| {
            nmf_k_sweep_run_mc(
                m,
                &k_range,
                preprocessing,
                use_second_layer,
                Some(opts),
                Some(cons),
                n_runs,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    nmf_sweep_dict(py, res)
}
