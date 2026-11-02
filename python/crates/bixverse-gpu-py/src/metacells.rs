//! SEACells with both Frank-Wolfe solves on the GPU, the twin of
//! `rs_seacells_gpu`, and Bonsai with Sanity on the GPU, the twins of
//! `rs_sc_bonsai_gpu` and `rs_mc_bonsai_gpu`.
//!
//! Same signatures as the CPU twins. Cell mapping, the kNN, the archetype
//! bookkeeping and the aggregation are the CPU code via
//! `seacells_on_store_with`; only the fit moves.

use bixverse_sc::mc_generation::generate::seacells_on_store_with;
use bixverse_sc::mc_generation::seacells::SEACellsParams;
use bixverse_sc::prelude::parse_verbosity_level;
use bixverse_sc::sc_analysis::bonsai::BonsaiScParams;
use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_gpu::errors::ScGpuErrors;
use bixverse_sc_gpu::sanity_bonsai_gpu::{sanity_bonsai_mc_gpu, sanity_bonsai_sc_gpu};
use bixverse_sc_gpu::seacells_gpu::seacells_fit_gpu;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use py_common::convert::{csr_from_parts, from_py, index_vec, knn_pair, mat_view};
use py_common::error::BvErr;
use py_common::results::{bonsai_dict, seacells_dict};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::device::{ensure_gpu, ensure_plane_ops, on_device};
use crate::error::GpuErr;

/// SEACells over a cell-major store, the fit on the GPU
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `embd` - `(n, d)` embedding over the working rows, C-contiguous
/// * `knn_indices` - Optional `(n, k)` kNN over the working rows
/// * `knn_distances` - Its distances
/// * `cells_to_keep` - Store index of each working row
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - `to_rust(SeacellsParams)`
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`seacells_dict`].
#[pyfunction]
#[pyo3(signature = (
    f_path_cell, embd, knn_indices, knn_distances, cells_to_keep, cells_to_use, params,
    target_size, seed, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn seacells<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_distances: Option<PyReadonlyArray2<'py, f32>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    cells_to_use: Option<PyReadonlyArray1<'py, u32>>,
    params: &Bound<'py, PyAny>,
    target_size: f32,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let params: SEACellsParams = from_py(params, "SEACells params")?;
    let knn = knn_pair(knn_indices.as_ref(), knn_distances.as_ref())?;
    let embd = mat_view(&embd)?;
    let keep = index_vec(&cells_to_keep)?;
    let to_use = cells_to_use.as_ref().map(index_vec).transpose()?;
    let v = verbose;
    let res = py
        .detach(|| {
            on_device(|device| {
                seacells_on_store_with::<ScGpuErrors, _>(
                    f_path_cell,
                    embd,
                    knn.as_ref().map(|(i, d)| (i.as_slice(), d.as_slice())),
                    Some(&keep),
                    to_use.as_deref(),
                    &params,
                    target_size,
                    seed,
                    v,
                    |e, i, d| seacells_fit_gpu(e, i, d, &params, seed, device, v),
                )
            })
        })
        .map_err(GpuErr::from)?;
    seacells_dict(py, res)
}

/// Bonsai over a cell and gene selection, Sanity on the GPU
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based genes
/// * `params` - `to_rust(BonsaiParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`bonsai_dict`].
#[pyfunction]
#[pyo3(signature = (f_path_gene, f_path_cell, cell_indices, gene_indices, params, verbose = 1))]
pub fn sc_bonsai<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_plane_ops()?;
    let params: BonsaiScParams = from_py(params, "Bonsai params")?;
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let gene_reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let cell_reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                sanity_bonsai_sc_gpu(
                    &gene_reader,
                    &cell_reader,
                    &cells,
                    &genes,
                    &params,
                    device,
                    parse_verbosity_level(verbose),
                )
            })
        })
        .map_err(GpuErr::from)?;
    bonsai_dict(py, res)
}

/// Bonsai over meta cell raw counts, Sanity on the GPU
///
/// ### Params
///
/// * `data` - CSR raw counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `gene_indices` - 0-based genes
/// * `params` - `to_rust(BonsaiParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`bonsai_dict`].
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, gene_indices, params, verbose = 1))]
#[allow(clippy::too_many_arguments)]
pub fn mc_bonsai<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    gene_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_plane_ops()?;
    let params: BonsaiScParams = from_py(params, "Bonsai params")?;
    let genes = index_vec(&gene_indices)?;
    let counts = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                sanity_bonsai_mc_gpu(
                    &counts,
                    &genes,
                    &params,
                    device,
                    parse_verbosity_level(verbose),
                )
            })
        })
        .map_err(GpuErr::from)?;
    bonsai_dict(py, res)
}
