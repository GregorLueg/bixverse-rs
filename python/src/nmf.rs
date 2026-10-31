//! HALS NMF over single cells and meta cells: the Python twins of
//! `rs_nmf_single_sc`, `rs_nmf_multi_sc`, `rs_nmf_consensus_sc`,
//! `rs_nmf_k_sweep_sc` and their `_mc` siblings.
//!
//! The numerics live in `bixverse-sc` and `bixverse-methods`; this only moves
//! buffers. Every index is 0-based. The crates factorise cells x genes, so
//! their `w` is cells x k and their `h` is k x genes; both come back as is and
//! the Python side flips them, as R's constructors do. A meta cell matrix
//! crosses as its CSR parts.

use bixverse_methods::methods::nmf_hals::consensus::{
    ConsensusNmfResult, ConsensusParams, ConsensusParamsSpec, KSweepEntry,
};
use bixverse_methods::methods::nmf_hals::{HalsOpts, HalsOptsSpec, NmfResult, StabilisedNmfResult};
use bixverse_sc::mc_analysis::nmf_mc::{
    nmf_consensus_run_mc, nmf_k_sweep_run_mc, nmf_multiple_run_mc, nmf_single_run_mc,
};
use bixverse_sc::prelude::{CompressedSparseData2, ScErrors};
use bixverse_sc::sc_analysis::nmf_sc::{
    nmf_consensus_run_sc, nmf_k_sweep_run_sc, nmf_multiple_run_sc, nmf_single_run_sc,
};
use bixverse_sc_data::data_io::ParallelSparseReader;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::de::{mat_f64, u32_array};
use crate::error::BvErr;
use crate::graph::from_py;
use crate::metacells::csr_from_parts;
use crate::pp::index_vec;

/////////////
// Helpers //
/////////////

/// HALS options from the parameter dict
///
/// ### Params
///
/// * `params` - `to_rust(NmfHalsParams)`
/// * `seed` - Seed for the random initialisation
///
/// ### Returns
///
/// The options.
fn hals_opts(params: &Bound<'_, PyAny>, seed: usize) -> PyResult<HalsOpts<f32>> {
    let spec: HalsOptsSpec = from_py(params, "NMF HALS params")?;
    spec.into_opts(seed)
        .map_err(|e| BvErr::from(ScErrors::from(e)).into())
}

/// Consensus options from the parameter dict, seeded from the run seed
///
/// R's `.inject_consensus_seed` does the same: the k-means restarts take the
/// run seed rather than a seed of their own.
///
/// ### Params
///
/// * `params` - `to_rust(NmfConsensusParams)`
/// * `seed` - The run seed
///
/// ### Returns
///
/// The options.
fn consensus_params(params: &Bound<'_, PyAny>, seed: usize) -> PyResult<ConsensusParams<f32>> {
    let mut spec: ConsensusParamsSpec = from_py(params, "NMF consensus params")?;
    spec.consensus_seed = Some(seed as u64);
    spec.into_params()
        .map_err(|e| BvErr::from(ScErrors::from(e)).into())
}

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

/// A single run as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The run
///
/// ### Returns
///
/// Dict with `w` (cells x k), `h` (k x genes), `final_loss`, `n_iter` and
/// `converged`.
fn single_dict<'py>(py: Python<'py>, res: NmfResult<f32>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("w", mat_f64(py, &res.w)?)?;
    out.set_item("h", mat_f64(py, &res.h)?)?;
    out.set_item("final_loss", res.final_loss as f64)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    Ok(out)
}

/// Restarts as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The restarts
///
/// ### Returns
///
/// Dict with `w_all` (cells x `k * n_runs`, run blocks side by side),
/// `h_per_run` (k x genes per run), `losses`, `converged` and the 0-based
/// `best_idx`.
fn multi_dict<'py>(py: Python<'py>, res: StabilisedNmfResult<f32>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let h = PyList::empty(py);
    for m in &res.h_per_run {
        h.append(mat_f64(py, m)?)?;
    }
    out.set_item("w_all", mat_f64(py, &res.w_all)?)?;
    out.set_item("h_per_run", h)?;
    out.set_item(
        "losses",
        res.losses
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    out.set_item("converged", res.converged)?;
    out.set_item("best_idx", res.best_idx)?;
    Ok(out)
}

/// The consensus fit as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The consensus fit
///
/// ### Returns
///
/// Dict with `w` (cells x k), `h` (k x genes), `rel_error`,
/// `rel_run_errors`, `labels` (0-based cluster per pooled component, -1 for
/// a dropped one, `int64`), `local_density`, `kept` (0-based pooled indices),
/// `silhouette` (one per kept component), `stability`, `cluster_sizes`,
/// `n_dropped` and `n_empty_clusters`.
fn consensus_dict<'py>(
    py: Python<'py>,
    res: ConsensusNmfResult<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let f64s = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let c = &res.clusters;
    let labels: Vec<i64> = c
        .labels
        .iter()
        .map(|l| l.map_or(-1, |v| v as i64))
        .collect();
    let out = PyDict::new(py);
    out.set_item("w", mat_f64(py, &res.w)?)?;
    out.set_item("h", mat_f64(py, &res.h)?)?;
    out.set_item("rel_error", res.error as f64)?;
    out.set_item("rel_run_errors", f64s(&res.run_errors).into_pyarray(py))?;
    out.set_item("labels", labels.into_pyarray(py))?;
    out.set_item("local_density", f64s(&c.local_density).into_pyarray(py))?;
    out.set_item("kept", u32_array(py, &c.kept))?;
    out.set_item("silhouette", f64s(&c.silhouette).into_pyarray(py))?;
    out.set_item("stability", c.stability as f64)?;
    out.set_item("cluster_sizes", u32_array(py, &c.sizes))?;
    out.set_item("n_dropped", c.n_dropped)?;
    out.set_item("n_empty_clusters", c.n_empty_clusters)?;
    Ok(out)
}

/// A k sweep as a column dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `entries` - One entry per k
///
/// ### Returns
///
/// Dict with `k`, `stability`, `best_error`, `median_error`,
/// `consensus_failed`, `n_dropped`, `n_empty_clusters` and `n_converged`, one
/// value per k.
fn sweep_dict<'py>(
    py: Python<'py>,
    entries: Vec<KSweepEntry<f32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let ints = |f: fn(&KSweepEntry<f32>) -> usize| entries.iter().map(f).collect::<Vec<_>>();
    let floats =
        |f: fn(&KSweepEntry<f32>) -> f32| entries.iter().map(|e| f(e) as f64).collect::<Vec<f64>>();
    out.set_item("k", u32_array(py, &ints(|e| e.k)))?;
    out.set_item("stability", floats(|e| e.stability).into_pyarray(py))?;
    out.set_item("best_error", floats(|e| e.best_error).into_pyarray(py))?;
    out.set_item("median_error", floats(|e| e.median_error).into_pyarray(py))?;
    out.set_item(
        "consensus_failed",
        entries
            .iter()
            .map(|e| e.consensus_failed)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("n_dropped", u32_array(py, &ints(|e| e.n_dropped)))?;
    out.set_item(
        "n_empty_clusters",
        u32_array(py, &ints(|e| e.n_empty_clusters)),
    )?;
    out.set_item("n_converged", u32_array(py, &ints(|e| e.n_converged)))?;
    Ok(out)
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
/// See [single_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k, preprocessing, use_second_layer, params,
    seed = 42, verbose = false
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
    verbose: bool,
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
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    single_dict(py, res)
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
/// See [multi_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k, preprocessing, use_second_layer, params,
    n_runs, seed = 42, verbose = false
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
    verbose: bool,
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
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    multi_dict(py, res)
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
/// See [consensus_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k, preprocessing, use_second_layer, params,
    consensus, n_runs, seed = 42, verbose = false
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
    verbose: bool,
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
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    consensus_dict(py, res)
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
/// See [sweep_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, gene_indices, cell_indices, k_range, preprocessing, use_second_layer,
    params, consensus, n_runs, seed = 42, verbose = false
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
    verbose: bool,
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
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    sweep_dict(py, res)
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
/// See [single_dict].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k, preprocessing, use_second_layer, params, seed = 42,
    verbose = false
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
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = hals_opts(params, seed)?;
    let m = mc_matrix((&data, &indices, &indptr, shape))?;
    let res = py
        .detach(|| {
            nmf_single_run_mc(
                m,
                k,
                preprocessing,
                use_second_layer,
                Some(opts),
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    single_dict(py, res)
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
/// See [multi_dict].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k, preprocessing, use_second_layer, params, n_runs,
    seed = 42, verbose = false
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
    verbose: bool,
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
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    multi_dict(py, res)
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
/// See [consensus_dict].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k, preprocessing, use_second_layer, params, consensus,
    n_runs, seed = 42, verbose = false
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
    verbose: bool,
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
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    consensus_dict(py, res)
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
/// See [sweep_dict].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, k_range, preprocessing, use_second_layer, params,
    consensus, n_runs, seed = 42, verbose = false
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
    verbose: bool,
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
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    sweep_dict(py, res)
}
