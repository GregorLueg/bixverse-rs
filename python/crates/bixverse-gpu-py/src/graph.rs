//! kNN, Leiden / Louvain and fast clustering on the GPU, the twins of
//! `rs_gpu_knn`, `rs_leiden_clustering_gpu` and `rs_fast_cluster_gpu`.
//!
//! The kNN and the fast clustering keep the signatures of their CPU twins in
//! `bixverse._bixverse`; the neighbour graph itself is built by the CPU
//! extension from the kNN this hands back.

use bixverse_gpu::graph::leiden_gpu::community_detection_gpu;
use bixverse_methods::graph::graph_structures::SparseGraph;
use bixverse_methods::graph::leiden::LeidenParams;
use bixverse_sc::prelude::CompressedSparseData2;
use bixverse_sc::sc_analysis::fast_clusters::{FastClusterParams, fast_clusters};
use bixverse_sc::sc_processing::knn::pad_knn_rows;
use bixverse_sc_gpu::fast_clusters_gpu::{GpuKMeans, KMeansGpuSpec};
use bixverse_sc_gpu::knn_gpu::{KnnParamsGpu, knn_with_dist_gpu};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use py_common::convert::{flat_array, from_py, mat_view};
use py_common::results::{FastClusterOut, fast_cluster_out};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::device::{ensure_gpu, ensure_plane_ops, on_device};
use crate::error::GpuErr;
use py_common::convert::CSlice;

/// GPU kNN over an embedding, with the distances
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `params` - `to_rust(KnnGpuDefaults)`; `k` sets the neighbours
/// * `seed` - Seed for the index builds
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(indices, distances)` as `(n_cells, k)` `uint32` and `float32`, self
/// excluded, Euclidean as true L2. Rows an approximate index left short are
/// padded with duplicate edges, as the CPU twin does.
#[pyfunction]
#[pyo3(signature = (embd, params, seed = 42, verbose = 1))]
pub fn knn_with_dist<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    ensure_gpu()?;
    let params: KnnParamsGpu = from_py(params, "GPU knn params")?;
    let view = mat_view(&embd)?;
    let (n, k) = (view.nrows(), params.k);
    if k == 0 || k >= n {
        return Err(PyValueError::new_err(format!(
            "k must be in 1..{n} for {n} cells, got {k}"
        )));
    }
    let (mut idx, mut dist) = py
        .detach(|| on_device(|device| knn_with_dist_gpu(view, k, &params, device, seed, verbose)))
        .map_err(GpuErr::from)?;
    let padded = pad_knn_rows(&mut idx, Some(&mut dist), k);
    if padded > 0 && verbose > 0 {
        println!(
            "{padded} of {n} rows came back with fewer than {k} neighbours and were padded \
             with duplicate edges."
        );
    }
    let idx: Vec<u32> = idx.into_iter().flatten().map(|j| j as u32).collect();
    let dist: Vec<f32> = dist.into_iter().flatten().collect();
    Ok((flat_array(py, idx, n, k)?, flat_array(py, dist, n, k)?))
}

/// Leiden or Louvain on a symmetric CSR graph, level-0 local moving on the GPU
///
/// Same signature as the CPU `leiden`. Graphs below
/// `LEIDEN_GPU_MIN_NODES` nodes, or with a node of very high degree, run fully
/// on the CPU with coloured local moving inside the crate.
///
/// ### Params
///
/// * `data` - `float32` edge weights of the symmetric CSR
/// * `indices` - `uint32` column indices
/// * `indptr` - `uint32` row pointers, `n + 1` of them
/// * `params` - Dict with the `LeidenParams` keys
///
/// ### Returns
///
/// `uint32` community per node, 0 the largest.
#[pyfunction]
pub fn leiden<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    ensure_plane_ops()?;
    let params: LeidenParams = from_py(params, "leiden params")?;
    let n = indptr.len()?.saturating_sub(1);
    let adjacency = CompressedSparseData2::new_csr(
        data.c_slice()?,
        indices.c_slice()?,
        indptr.c_slice()?,
        None,
        (n, n),
    );
    let graph = SparseGraph::new(n, adjacency, false);
    let labels = py
        .detach(|| {
            on_device(|device| {
                community_detection_gpu(&graph, &params, usize::MAX, true, &device.client())
            })
        })
        .map_err(GpuErr::from)?;
    let labels: Vec<u32> = labels.into_iter().map(|c| c as u32).collect();
    Ok(labels.into_pyarray(py).into_any())
}

/// k-means coarsening on the GPU, a centroid graph and Louvain / Leiden on it
///
/// Same signature as the CPU `fast_cluster`; the k-means keys are the GPU
/// ones (`k_means_iter`, `k_means_init`, `fixed`, `quantise`).
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `resolutions` - The resolutions, in output order
/// * `params` - `to_rust(ScFastClusterGpuParams)` with `n_centroids` and
///   `n_iterations` resolved
/// * `n_seeds` - Community-detection seeds per resolution
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`fast_cluster_out`].
#[pyfunction]
#[pyo3(signature = (embd, resolutions, params, n_seeds = 1, seed = 42, verbose = 1))]
pub fn fast_cluster<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    resolutions: Vec<f32>,
    params: &Bound<'py, PyAny>,
    n_seeds: usize,
    seed: usize,
    verbose: usize,
) -> PyResult<FastClusterOut<'py>> {
    ensure_gpu()?;
    let fc: FastClusterParams = from_py(params, "fast cluster params")?;
    let km: KMeansGpuSpec = from_py(params, "GPU k-means params")?;
    let km = km.into_params().map_err(PyValueError::new_err)?;
    let view = mat_view(&embd)?;
    let n = view.nrows();
    let res = py
        .detach(|| {
            on_device(|device| {
                fast_clusters(
                    view,
                    &GpuKMeans::new(Some(km), device),
                    &fc,
                    &resolutions,
                    n_seeds,
                    seed,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    fast_cluster_out(py, res, n, n_seeds)
}
