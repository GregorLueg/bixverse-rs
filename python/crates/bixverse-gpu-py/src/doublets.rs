//! Scrublet on the GPU, the twin of `rs_sc_scrublet_gpu`.

use bixverse_sc::sc_processing::knn::KnnParams;
use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_gpu::knn_gpu::KnnParamsGpu;
use bixverse_sc_gpu::scrublet_gpu::{ScrubletGpuSpec, ScrubletKnnBackend, run_scrublet_gpu};
use numpy::PyReadonlyArray1;
use py_common::convert::{from_py, index_vec};
use py_common::error::BvErr;
use py_common::results::scrublet_dict;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::device::{ensure_gpu, on_device};
use crate::error::GpuErr;

/// Scrublet over a cell selection, SVD, projection and (by default) kNN on
/// the GPU
///
/// Same signature as the CPU `scrublet`. The kNN block is read from the same
/// dict as the CPU or the GPU one, as `knn_backend` says.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `cells` - 0-based cells
/// * `params` - `to_rust(ScrubletGpuParams)`
/// * `seed` - Seed
/// * `streaming` - Stream the HVG pass
/// * `return_combined_pca` - Also return the PCA of observed and simulated
///   cells
/// * `return_pairs` - Also return the parents of the simulated doublets
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [`scrublet_dict`].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cells, params, seed = 42, streaming = false,
    return_combined_pca = false, return_pairs = false, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn scrublet<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cells: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    streaming: bool,
    return_combined_pca: bool,
    return_pairs: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_gpu()?;
    let spec: ScrubletGpuSpec = from_py(params, "GPU scrublet params")?;
    let knn = match spec.knn_backend.as_str() {
        "gpu" => ScrubletKnnBackend::Gpu(from_py::<KnnParamsGpu>(params, "GPU knn params")?),
        "cpu" => ScrubletKnnBackend::Cpu(from_py::<KnnParams>(params, "knn params")?),
        other => {
            return Err(PyValueError::new_err(format!(
                "knn_backend must be 'gpu' or 'cpu', got '{other}'"
            )));
        }
    };
    let params = spec.into_params(knn);
    let cells = index_vec(&cells)?;
    let gene_reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let cell_reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            on_device(|device| {
                run_scrublet_gpu(
                    &gene_reader,
                    &cell_reader,
                    &params,
                    &cells,
                    streaming,
                    seed,
                    device,
                    verbose as usize,
                    return_combined_pca,
                    return_pairs,
                )
            })
        })
        .map_err(GpuErr::from)?;
    scrublet_dict(py, res)
}
