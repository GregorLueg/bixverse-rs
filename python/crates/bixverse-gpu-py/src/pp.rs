//! Sparse PCA on the GPU, the twin of `rs_sc_pca_sparse_gpu`.

use bixverse_sc::sc_processing::pca::SingleCellPcaParams;
use bixverse_sc_data::data_io::{ParallelSparseReader, SingleCellReading};
use bixverse_sc_gpu::pca_gpu::pca_on_sc_sparse_gpu;
use numpy::{IntoPyArray, PyReadonlyArray1};
use py_common::convert::{from_py, index_vec, mat_array};
use py_common::error::BvErr;
use pyo3::prelude::*;

use crate::device::{ensure_gpu, on_device};
use crate::error::GpuErr;

/// Sparse PCA over a cell and gene selection on the GPU
///
/// Same signature as the CPU `pca`, so the Python side can swap the two.
/// `svd_solver = "randomised"` runs the randomised SVD, every other solver
/// the exact block Lanczos; there is no dense or covariance GPU path, so
/// `sparse` is ignored.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the PFlogPF offsets
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based genes
/// * `no_pcs` - Number of components
/// * `params` - `to_rust(PcaParams)`
/// * `sparse` - Ignored, always sparse
/// * `seed` - Seed for the sketch or the Lanczos starting block
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(scores, loadings, singular_values)`, `float32`.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cell_indices, gene_indices, no_pcs, params, sparse = true,
    seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn pca<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    no_pcs: usize,
    params: &Bound<'py, PyAny>,
    sparse: bool,
    seed: usize,
    verbose: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let _ = sparse;
    ensure_gpu()?;
    let params: SingleCellPcaParams = from_py(params, "pca params")?;
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let offsets = if params.clr {
        let reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
        let offsets = reader.get_clr_offsets(&cells, None).map_err(BvErr::from)?;
        Some(offsets.iter().map(|&x| x as f32).collect::<Vec<f32>>())
    } else {
        None
    };
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let (scores, loadings, s) = py
        .detach(|| {
            on_device(|device| {
                pca_on_sc_sparse_gpu(
                    &reader,
                    &cells,
                    &genes,
                    no_pcs,
                    offsets.as_deref(),
                    &params,
                    seed,
                    device,
                    verbose,
                )
            })
        })
        .map_err(GpuErr::from)?;
    Ok((
        mat_array(py, &scores)?,
        mat_array(py, &loadings)?,
        s.into_pyarray(py).into_any(),
    ))
}
