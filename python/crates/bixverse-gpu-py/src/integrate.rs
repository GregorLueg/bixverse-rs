//! Batch correction on the GPU, the twins of `rs_harmony_v2_gpu`,
//! `rs_bbknn_gpu` and `rs_fast_mnn_gpu`.
//!
//! Harmony and BBKNN keep the signatures of their CPU twins. GPU fastMNN
//! corrects an embedding the caller hands over rather than computing a PCA
//! from the store, so it has its own.

use bixverse_sc_gpu::bbknn_gpu::{BbknnParamsGpu, bbknn_gpu};
use bixverse_sc_gpu::fast_mnn_gpu::{FastMnnParamsGpu, fast_mnn_gpu};
use bixverse_sc_gpu::harmony_gpu::{HarmonyParamsV2Gpu, harmony_v2_gpu};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use py_common::convert::{from_py, index_vec, mat_array, mat_view};
use py_common::results::{BbknnOut, bbknn_out};
use pyo3::prelude::*;

use crate::device::{ensure_gpu, on_device};
use crate::error::GpuErr;

/// Harmony v2 on the GPU, one batch covariate
///
/// ### Params
///
/// * `pca` - `(n_cells, n_pcs)` `float32` PCA, C-contiguous
/// * `params` - `to_rust(ScHarmonyV2GpuParams)` with `k` resolved
/// * `batch_labels` - One `uint32` code vector per covariate; the GPU path
///   takes exactly one
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `(n_cells, n_pcs)` `float32` corrected embedding.
#[pyfunction]
#[pyo3(signature = (pca, params, batch_labels, seed = 42, verbose = 1))]
pub fn harmony_v2<'py>(
    py: Python<'py>,
    pca: PyReadonlyArray2<'py, f32>,
    params: &Bound<'py, PyAny>,
    batch_labels: Vec<PyReadonlyArray1<'py, u32>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    ensure_gpu()?;
    let params: HarmonyParamsV2Gpu = from_py(params, "GPU harmony v2 params")?;
    let labels = batch_labels
        .iter()
        .map(index_vec)
        .collect::<PyResult<Vec<_>>>()?;
    let view = mat_view(&pca)?;
    let res = py
        .detach(|| {
            on_device(|device| harmony_v2_gpu(view, &labels, &params, seed, device, verbose))
        })
        .map_err(GpuErr::from)?;
    mat_array(py, &res)
}

/// BBKNN with the per-batch searches on the GPU
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `batch_labels` - `uint32` batch code per cell
/// * `params` - `to_rust(ScBbknnGpuParams)`
/// * `k` - Neighbours per cell to keep in the returned kNN
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`bbknn_out`].
#[pyfunction]
#[pyo3(signature = (embd, batch_labels, params, k, seed = 42, verbose = 1))]
pub fn bbknn_graph<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    batch_labels: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    k: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<BbknnOut<'py>> {
    ensure_gpu()?;
    let params: BbknnParamsGpu = from_py(params, "GPU bbknn params")?;
    let labels = index_vec(&batch_labels)?;
    let view = mat_view(&embd)?;
    let (distances, connectivities) = py
        .detach(|| on_device(|device| bbknn_gpu(view, &labels, &params, seed, device, verbose)))
        .map_err(GpuErr::from)?;
    bbknn_out(py, &distances, &connectivities, k)
}

/// fastMNN on an embedding, the neighbour searches on the GPU
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `batch_labels` - `uint32` batch code per cell
/// * `params` - `to_rust(ScFastmnnGpuParams)`
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `(n_cells, n_dims)` `float32` corrected embedding.
#[pyfunction]
#[pyo3(signature = (embd, batch_labels, params, seed = 42, verbose = 1))]
pub fn fast_mnn<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    batch_labels: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    ensure_gpu()?;
    let params: FastMnnParamsGpu = from_py(params, "GPU fastmnn params")?;
    let labels = index_vec(&batch_labels)?;
    let view = mat_view(&embd)?;
    let res = py
        .detach(|| on_device(|device| fast_mnn_gpu(view, &labels, &params, seed, device, verbose)))
        .map_err(GpuErr::from)?;
    mat_array(py, &res)
}
