//! Batch correction and integration metrics, the Python twins of
//! `rs_harmony`, `rs_harmony_v2`, `rs_bbknn` (with `rs_bbknn_filtering` and
//! the graph symmetrisation), `rs_mnn`, `rs_seurat_cca`, `rs_seurat_rpca`,
//! `rs_kbet`, `rs_lisi`, `rs_batch_silhouette_width`, `rs_pcr`,
//! `rs_cell_type_asw` and `rs_graph_connectivity`.
//!
//! The numerics and the store-level dispatch live in `bixverse-sc`; this only
//! moves buffers. Batch and label codes arrive as `uint32`, kNN indices as
//! `(n, k)` `uint32`, every index 0-based.

use bixverse_sc::sc_batch_correction::bbknn::{BbknnParams, bbknn};
use bixverse_sc::sc_batch_correction::fast_mnn::{FastMnnParams, fast_mnn_on_store};
use bixverse_sc::sc_batch_correction::harmony::{HarmonyParams, harmony as harmony_v1};
use bixverse_sc::sc_batch_correction::harmony_v2::{HarmonyParamsV2, harmony_v2 as harmony2};
use bixverse_sc::sc_batch_correction::seurat_cca::{SeuratCcaParams, seurat_cca_on_store};
use bixverse_sc::sc_batch_correction::seurat_rpca::{SeuratRpcaParams, seurat_rpca_on_store};
use bixverse_sc::sc_processing::metrics::{
    batch_silhouette_width, cell_type_asw as cell_type_asw_rs, graph_connectivity as conn_rs,
    kbet as kbet_rs, lisi as lisi_rs, lisi_weighted, pcr as pcr_rs,
};
use faer::MatRef;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::CSlice;
use py_common::convert::{dist_rows, knn_rows};
use py_common::convert::{from_py, index_vec, mat_array, mat_view};
use py_common::error::BvErr;
use py_common::results::{BbknnOut, bbknn_out};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/////////////
// Helpers //
/////////////

/// One label slice per batch variable
///
/// ### Params
///
/// * `labels` - 0-based codes per variable
///
/// ### Returns
///
/// The codes as `usize`.
fn label_lists(labels: &[PyReadonlyArray1<'_, u32>]) -> PyResult<Vec<Vec<usize>>> {
    labels.iter().map(index_vec).collect()
}

/////////////
// Harmony //
/////////////

/// Harmony on a PCA embedding
///
/// The twin of `rs_harmony`.
///
/// ### Params
///
/// * `pca` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `params` - `to_rust(HarmonyParams)` with `k` resolved
/// * `batch_labels` - One 0-based code vector per batch variable
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `(n_cells, n_dims)` `float32` corrected embedding.
#[pyfunction]
#[pyo3(signature = (pca, params, batch_labels, seed = 42, verbose = 1))]
pub fn harmony<'py>(
    py: Python<'py>,
    pca: PyReadonlyArray2<'py, f32>,
    params: &Bound<'py, PyAny>,
    batch_labels: Vec<PyReadonlyArray1<'py, u32>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let params: HarmonyParams = from_py(params, "harmony params")?;
    let labels = label_lists(&batch_labels)?;
    let view = mat_view(&pca)?;
    let res = py
        .detach(|| harmony_v1(view, &labels, &params, seed, verbose))
        .map_err(BvErr::from)?;
    mat_array(py, &res)
}

/// Harmony v2 on a PCA embedding
///
/// The twin of `rs_harmony_v2`.
///
/// ### Params
///
/// * `pca` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `params` - `to_rust(HarmonyV2Params)` with `k` resolved
/// * `batch_labels` - One 0-based code vector per batch variable
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `(n_cells, n_dims)` `float32` corrected embedding.
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
    let params: HarmonyParamsV2 = from_py(params, "harmony v2 params")?;
    let labels = label_lists(&batch_labels)?;
    let view = mat_view(&pca)?;
    let res = py
        .detach(|| harmony2(view, &labels, &params, seed, verbose))
        .map_err(BvErr::from)?;
    mat_array(py, &res)
}

///////////
// BBKNN //
///////////

/// BBKNN: the batch-balanced kNN and its neighbour graph
///
/// The twin of `rs_bbknn`, then `rs_bbknn_filtering` on the distances and
/// the max symmetrisation of `.bbknn_to_connectivity` on the
/// connectivities. Rows left short of `k` (zero distances are not stored)
/// are padded by repeating their last neighbour, as R does (with a warning).
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `batch_labels` - 0-based batch per cell
/// * `params` - `to_rust(BbknnParams)`
/// * `k` - Neighbours to keep per cell
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(indices, distances, data, cols, indptr)`: the `(n_cells, k)` `uint32`
/// and `float32` kNN, then the symmetric graph CSR, rows sorted, as
/// `float32`, `uint32`, `uint32`.
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
    let params: BbknnParams = from_py(params, "bbknn params")?;
    let labels = index_vec(&batch_labels)?;
    let view = mat_view(&embd)?;
    let (distances, connectivities) = py
        .detach(|| bbknn(view, &labels, &params, seed, verbose))
        .map_err(BvErr::from)?;
    bbknn_out(py, &distances, &connectivities, &labels, k)
}

/////////////////////////
// Store-based anchors //
/////////////////////////

/// The store-level inputs shared by fastMNN, CCA and rPCA.
struct StoreInputs {
    /// 0-based cells
    cells: Vec<usize>,
    /// 0-based genes
    genes: Vec<usize>,
    /// 0-based batch per cell
    batches: Vec<usize>,
    /// Optional pre-computed PCA, owned
    pca: Option<faer::Mat<f32>>,
}

/// Collect the store-level inputs of an anchor-type correction
///
/// ### Params
///
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based genes
/// * `batch_indices` - 0-based batch per cell
/// * `precomputed_pca` - Optional `(n_cells, n_pcs)` `float32` PCA
///
/// ### Returns
///
/// The [StoreInputs].
fn store_inputs(
    cell_indices: &PyReadonlyArray1<'_, u32>,
    gene_indices: &PyReadonlyArray1<'_, u32>,
    batch_indices: &PyReadonlyArray1<'_, u32>,
    precomputed_pca: Option<&PyReadonlyArray2<'_, f32>>,
) -> PyResult<StoreInputs> {
    Ok(StoreInputs {
        cells: index_vec(cell_indices)?,
        genes: index_vec(gene_indices)?,
        batches: index_vec(batch_indices)?,
        pca: precomputed_pca
            .map(|p| mat_view(p).map(|v| v.to_owned()))
            .transpose()?,
    })
}

/// fastMNN over a cell and gene selection
///
/// The twin of `rs_mnn`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the CLR offsets
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based genes, ideally batch-aware HVGs
/// * `batch_indices` - 0-based batch per cell
/// * `precomputed_pca` - Optional `(n_cells, n_pcs)` `float32` PCA to use
///   instead of computing one
/// * `params` - `to_rust(FastmnnParams)`
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `float32` corrected embedding, cells x dimensions.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cell_indices, gene_indices, batch_indices, precomputed_pca,
    params, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn fast_mnn<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    batch_indices: PyReadonlyArray1<'py, u32>,
    precomputed_pca: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let params: FastMnnParams = from_py(params, "fastmnn params")?;
    let s = store_inputs(
        &cell_indices,
        &gene_indices,
        &batch_indices,
        precomputed_pca.as_ref(),
    )?;
    let res = py
        .detach(|| {
            fast_mnn_on_store(
                f_path_gene,
                f_path_cell,
                &s.cells,
                &s.genes,
                &s.batches,
                s.pca,
                &params,
                verbose,
                seed,
            )
        })
        .map_err(BvErr::from)?;
    mat_array(py, &res)
}

/// Seurat CCA integration over a cell and gene selection
///
/// The twin of `rs_seurat_cca`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the CLR offsets
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based genes, ideally batch-aware HVGs
/// * `batch_indices` - 0-based batch per cell
/// * `precomputed_pca` - Optional `(n_cells, n_pcs)` `float32` PCA
/// * `params` - `to_rust(SeuratCcaParams)`
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `float32` corrected embedding, cells x dimensions.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cell_indices, gene_indices, batch_indices, precomputed_pca,
    params, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn seurat_cca<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    batch_indices: PyReadonlyArray1<'py, u32>,
    precomputed_pca: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let params: SeuratCcaParams = from_py(params, "seurat cca params")?;
    let s = store_inputs(
        &cell_indices,
        &gene_indices,
        &batch_indices,
        precomputed_pca.as_ref(),
    )?;
    let res = py
        .detach(|| {
            seurat_cca_on_store(
                f_path_gene,
                f_path_cell,
                &s.cells,
                &s.genes,
                &s.batches,
                s.pca,
                &params,
                verbose,
                seed,
            )
        })
        .map_err(BvErr::from)?;
    mat_array(py, &res)
}

/// Seurat rPCA integration over a cell and gene selection
///
/// The twin of `rs_seurat_rpca`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the CLR offsets
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based genes, ideally batch-aware HVGs
/// * `batch_indices` - 0-based batch per cell
/// * `precomputed_pca` - Optional `(n_cells, n_pcs)` `float32` union PCA
/// * `params` - `to_rust(SeuratRpcaParams)`
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `float32` corrected embedding, cells x dimensions.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cell_indices, gene_indices, batch_indices, precomputed_pca,
    params, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn seurat_rpca<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    batch_indices: PyReadonlyArray1<'py, u32>,
    precomputed_pca: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let params: SeuratRpcaParams = from_py(params, "seurat rpca params")?;
    let s = store_inputs(
        &cell_indices,
        &gene_indices,
        &batch_indices,
        precomputed_pca.as_ref(),
    )?;
    let res = py
        .detach(|| {
            seurat_rpca_on_store(
                f_path_gene,
                f_path_cell,
                &s.cells,
                &s.genes,
                &s.batches,
                s.pca,
                &params,
                verbose,
                seed,
            )
        })
        .map_err(BvErr::from)?;
    mat_array(py, &res)
}

/////////////
// Metrics //
/////////////

/// kBET on a kNN
///
/// The twin of `rs_kbet`.
///
/// ### Params
///
/// * `knn` - `(n, k)` `uint32` neighbour indices
/// * `batches` - Batch code per cell; need not be 0-based
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `p_values`, `chi_square_stats`, `mean_chi_square` and
/// `median_chi_square`.
#[pyfunction]
#[pyo3(signature = (knn, batches, verbose = 1))]
pub fn kbet<'py>(
    py: Python<'py>,
    knn: PyReadonlyArray2<'py, u32>,
    batches: PyReadonlyArray1<'py, u32>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let rows = knn_rows(&knn)?;
    let batches = index_vec(&batches)?;
    let res = py
        .detach(|| kbet_rs(&rows, &batches, verbose > 0))
        .map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("p_values", res.p_values.into_pyarray(py))?;
    out.set_item("chi_square_stats", res.chi_square_stats.into_pyarray(py))?;
    out.set_item("mean_chi_square", res.mean_chi_square)?;
    out.set_item("median_chi_square", res.median_chi_square)?;
    Ok(out)
}

/// LISI on a kNN, uniform or perplexity weighted
///
/// The twin of `rs_lisi`.
///
/// ### Params
///
/// * `knn` - `(n, k)` `uint32` neighbour indices
/// * `knn_dist` - `(n, k)` `float32` distances for the weighted version;
///   `None` weights the neighbours uniformly
/// * `labels` - Label code per cell; need not be 0-based
/// * `perplexity` - Perplexity of the weighted version; `None` is the crate
///   default
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `per_cell`, `mean_lisi`, `median_lisi`, `n_labels`,
/// `ilisi_norm` and `clisi_norm`.
#[pyfunction]
#[pyo3(signature = (knn, knn_dist, labels, perplexity = None, verbose = 1))]
pub fn lisi<'py>(
    py: Python<'py>,
    knn: PyReadonlyArray2<'py, u32>,
    knn_dist: Option<PyReadonlyArray2<'py, f32>>,
    labels: PyReadonlyArray1<'py, u32>,
    perplexity: Option<f64>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let rows = knn_rows(&knn)?;
    let dists = knn_dist
        .as_ref()
        .map(|d| dist_rows(d, knn.shape()))
        .transpose()?;
    let labels = index_vec(&labels)?;
    let res = py
        .detach(|| match &dists {
            Some(d) => lisi_weighted(&rows, d, &labels, perplexity, verbose > 0),
            None => lisi_rs(&rows, &labels, verbose > 0),
        })
        .map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("ilisi_norm", res.ilisi_norm())?;
    out.set_item("clisi_norm", res.clisi_norm())?;
    out.set_item("mean_lisi", res.mean_lisi)?;
    out.set_item("median_lisi", res.median_lisi)?;
    out.set_item("n_labels", res.n_labels)?;
    out.set_item("per_cell", res.per_cell.into_pyarray(py))?;
    Ok(out)
}

/// Batch average silhouette width on an embedding
///
/// The twin of `rs_batch_silhouette_width`.
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `batches` - Batch code per cell; need not be 0-based
/// * `max_cells` - Subsample to this many cells; `None` uses all
/// * `seed` - Seed for the subsampling
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `per_cell`, `mean_asw` and `median_asw`.
#[pyfunction]
#[pyo3(signature = (embd, batches, max_cells, seed = 42, verbose = 1))]
pub fn batch_asw<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    batches: PyReadonlyArray1<'py, u32>,
    max_cells: Option<usize>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let view = mat_view(&embd)?;
    let batches = index_vec(&batches)?;
    let res = py
        .detach(|| batch_silhouette_width(view, &batches, max_cells, seed, verbose > 0))
        .map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("per_cell", res.per_cell.into_pyarray(py))?;
    out.set_item("mean_asw", res.mean_asw)?;
    out.set_item("median_asw", res.median_asw)?;
    Ok(out)
}

/// Cell type average silhouette width on an embedding, rescaled to `[0, 1]`
///
/// The twin of `rs_cell_type_asw`.
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `labels` - Cell type code per cell; need not be 0-based
/// * `max_cells` - Subsample to this many cells; `None` uses all
/// * `seed` - Seed for the subsampling
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `per_cell`, `mean_asw` and `median_asw`.
#[pyfunction]
#[pyo3(signature = (embd, labels, max_cells, seed = 42, verbose = 1))]
pub fn cell_type_asw<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    labels: PyReadonlyArray1<'py, u32>,
    max_cells: Option<usize>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let view = mat_view(&embd)?;
    let labels = index_vec(&labels)?;
    let res = py
        .detach(|| cell_type_asw_rs(view, &labels, max_cells, seed, verbose > 0))
        .map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("per_cell", res.per_cell.into_pyarray(py))?;
    out.set_item("mean_asw", res.mean)?;
    out.set_item("median_asw", res.median)?;
    Ok(out)
}

/// Principal component regression on batch
///
/// The twin of `rs_pcr`. Takes `float64`, as R hands the embedding over.
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float64` embedding, C-contiguous
/// * `batches` - Batch code per cell; need not be 0-based
///
/// ### Returns
///
/// Dict with `var_explained`, `r_squared` and `pcr`.
#[pyfunction]
pub fn pcr<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f64>,
    batches: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let (n, d) = (embd.shape()[0], embd.shape()[1]);
    let view = MatRef::from_row_major_slice(embd.c_slice()?, n, d);
    let batches = index_vec(&batches)?;
    let res = py.detach(|| pcr_rs(view, &batches)).map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("var_explained", res.var_explained.into_pyarray(py))?;
    out.set_item("r_squared", res.r_squared.into_pyarray(py))?;
    out.set_item("pcr", res.pcr)?;
    Ok(out)
}

/// Graph connectivity per cell type on a kNN
///
/// The twin of `rs_graph_connectivity`.
///
/// ### Params
///
/// * `knn` - `(n, k)` `uint32` neighbour indices
/// * `labels` - Cell type code per cell; need not be 0-based
///
/// ### Returns
///
/// Dict with `per_label` (first-appearance order), `mean` and `median`.
#[pyfunction]
pub fn graph_connectivity<'py>(
    py: Python<'py>,
    knn: PyReadonlyArray2<'py, u32>,
    labels: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let rows = knn_rows(&knn)?;
    let labels = index_vec(&labels)?;
    let res = py.detach(|| conn_rs(&rows, &labels)).map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("per_label", res.per_label.into_pyarray(py))?;
    out.set_item("mean", res.mean)?;
    out.set_item("median", res.median)?;
    Ok(out)
}
