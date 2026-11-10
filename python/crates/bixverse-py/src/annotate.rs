//! Cell type annotation and reference mapping, the Python twins of
//! `rs_sc_type`, `rs_sc_type_cluster_assignment`,
//! `rs_sc_type_assign_cells_graph`, `rs_build_symphony_ref`,
//! `rs_symphony_map_query` and `rs_transfer_labels_symphony`.
//!
//! The numerics live in `bixverse-sc`; this only moves buffers. Every index
//! is 0-based, and an Unknown ScType call is `-1` rather than R's `0`. The
//! ScType scores widen to `float64` on the way out, as R's do; the Symphony
//! reference stays `float32`, the precision it was computed in.

use bixverse_core::core::math::sparse::CompressedSparseData2;
use bixverse_methods::graph::graph_label_propagations::KnnLabPropGraph;
use bixverse_methods::graph::graph_structures::SparseGraph;
use bixverse_sc::sc_annotation::sc_type::{
    CellTypeMarkers, ScTypeCellParams, ScTypeHybridRes, SctypeRes, assign_cells, assign_clusters,
    assign_hybrid, run_sctype,
};
use bixverse_sc::sc_annotation::symphony::{
    HarmonyBackend, SymphonyMapParams, build_symphony_reference_on_store,
    generate_knn_cross_with_dist, knn_majority_vote, symphony_map_query_parts,
};
use bixverse_sc::sc_batch_correction::harmony::HarmonyParams;
use bixverse_sc::sc_batch_correction::harmony_v2::HarmonyParamsV2;
use bixverse_sc::sc_processing::knn::KnnParams;
use bixverse_sc::sc_processing::pca::SingleCellPcaParams;
use bixverse_sc_data::data_io::ParallelSparseReader;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::CSlice;
use py_common::convert::f64_array;
use py_common::convert::{flat_array, from_py, index_vec, mat_array, mat_view};
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Three borrowed `(data, indices, indptr)` arrays of a symmetric CSR.
type CsrParts<'py> = (
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
);

/////////////
// Helpers //
/////////////

/// Optional cell type indices as `int64`, `None` becoming `-1`
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The indices
///
/// ### Returns
///
/// The array.
fn opt_index_array<'py>(py: Python<'py>, v: &[Option<usize>]) -> Bound<'py, PyAny> {
    v.iter()
        .map(|x| x.map_or(-1, |i| i as i64))
        .collect::<Vec<i64>>()
        .into_pyarray(py)
        .into_any()
}

/// `usize` values as a numpy `int64` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The values
///
/// ### Returns
///
/// The array.
fn i64_array<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyAny> {
    v.iter()
        .map(|&x| x as i64)
        .collect::<Vec<i64>>()
        .into_pyarray(py)
        .into_any()
}

/// The ScType scores back as the crate structure
///
/// ### Params
///
/// * `cell_types` - The cell types, in column order
/// * `scores` - `(n_cells, n_cell_types)` `float64` scores, C-contiguous
///
/// ### Returns
///
/// The [SctypeRes], or a `ValueError` if the columns and cell types
/// disagree.
fn sctype_res(cell_types: Vec<String>, scores: &PyReadonlyArray2<'_, f64>) -> PyResult<SctypeRes> {
    let (n_cells, n_cell_types) = (scores.shape()[0], scores.shape()[1]);
    if n_cell_types != cell_types.len() {
        return Err(PyValueError::new_err(format!(
            "The scores have {n_cell_types} columns but there are {} cell types.",
            cell_types.len()
        )));
    }
    Ok(SctypeRes {
        cell_types,
        scores: scores.c_slice()?.iter().map(|&x| x as f32).collect(),
        n_cells,
        n_cell_types,
    })
}

/// Per-cluster composition of the hybrid ScType assignment
///
/// ### Params
///
/// * `py` - GIL token
/// * `hybrid` - The hybrid result
/// * `n_ct` - Number of cell types
///
/// ### Returns
///
/// Dict with one array per composition field (`dominant` and `second` with
/// `-1` for none) plus the `(n_clusters, n_ct)` `int64` `counts`.
fn composition_dict<'py>(
    py: Python<'py>,
    hybrid: &ScTypeHybridRes,
    n_ct: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let comp = &hybrid.composition;
    let field = |f: fn(&_) -> f32| comp.iter().map(f).collect::<Vec<f32>>();
    let out = PyDict::new(py);
    out.set_item(
        "cluster_id",
        i64_array(py, &comp.iter().map(|c| c.cluster).collect::<Vec<_>>()),
    )?;
    out.set_item(
        "n_cells",
        i64_array(py, &comp.iter().map(|c| c.n_cells).collect::<Vec<_>>()),
    )?;
    out.set_item(
        "n_unknown",
        i64_array(py, &comp.iter().map(|c| c.n_unknown).collect::<Vec<_>>()),
    )?;
    out.set_item(
        "dominant",
        opt_index_array(py, &comp.iter().map(|c| c.dominant).collect::<Vec<_>>()),
    )?;
    out.set_item(
        "second",
        opt_index_array(py, &comp.iter().map(|c| c.second).collect::<Vec<_>>()),
    )?;
    out.set_item("purity", f64_array(py, &field(|c| c.purity)))?;
    out.set_item(
        "second_fraction",
        f64_array(py, &field(|c| c.second_fraction)),
    )?;
    out.set_item("entropy", f64_array(py, &field(|c| c.entropy)))?;
    // `cluster_mixed` is indexed by cluster id, the composition skips empty
    // clusters
    out.set_item(
        "cluster_mixed",
        comp.iter()
            .map(|c| hybrid.cluster_mixed[c.cluster])
            .collect::<Vec<bool>>()
            .into_pyarray(py),
    )?;
    let counts: Vec<i64> = comp
        .iter()
        .flat_map(|c| c.counts.iter().map(|&x| x as i64))
        .collect();
    out.set_item("counts", flat_array(py, counts, comp.len(), n_ct)?)?;
    Ok(out)
}

////////////
// ScType //
////////////

/// ScType scores per cell
///
/// The twin of `rs_sc_type`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `cell_indices` - 0-based cells, in score row order
/// * `markers` - One dict per cell type: `cell_type`, `positive_indices`,
///   `negative_indices` (0-based genes)
/// * `sensitivity` - Downweight markers shared between cell types
/// * `weight_floor` - Floor of the sensitivity weights; `None` is 0.1
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `cell_types` and the `(n_cells, n_cell_types)` `float64`
/// `scores`.
#[pyfunction]
#[pyo3(signature = (f_path_gene, cell_indices, markers, sensitivity, weight_floor = None, verbose = 1))]
pub fn sc_type<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    markers: &Bound<'py, PyAny>,
    sensitivity: bool,
    weight_floor: Option<f32>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let markers: Vec<CellTypeMarkers> = from_py(markers, "cell markers")?;
    let cells = index_vec(&cell_indices)?;
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            run_sctype(
                &reader,
                &cells,
                &markers,
                sensitivity,
                None,
                weight_floor,
                verbose,
            )
        })
        .map_err(BvErr::from)?;

    let scores: Vec<f64> = res.scores.iter().map(|&x| x as f64).collect();
    let out = PyDict::new(py);
    out.set_item("cell_types", res.cell_types)?;
    out.set_item(
        "scores",
        flat_array(py, scores, res.n_cells, res.n_cell_types)?,
    )?;
    Ok(out)
}

/// One ScType call per cluster
///
/// The twin of `rs_sc_type_cluster_assignment`.
///
/// ### Params
///
/// * `cell_types` - The cell types, in score column order
/// * `scores` - `(n_cells, n_cell_types)` `float64` scores, C-contiguous
/// * `cluster_labels` - 0-based cluster per scored cell
///
/// ### Returns
///
/// Dict with `cluster_id`, `cell_type` (`"Unknown"` below the floor),
/// `scores` and `n_cells`, one entry per non-empty cluster.
#[pyfunction]
pub fn sc_type_cluster_assignment<'py>(
    py: Python<'py>,
    cell_types: Vec<String>,
    scores: PyReadonlyArray2<'py, f64>,
    cluster_labels: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let res = sctype_res(cell_types, &scores)?;
    let labels = index_vec(&cluster_labels)?;
    let calls = assign_clusters(&res, &labels).map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item(
        "cluster_id",
        i64_array(py, &calls.iter().map(|c| c.cluster).collect::<Vec<_>>()),
    )?;
    out.set_item(
        "cell_type",
        calls
            .iter()
            .map(|c| c.cell_type.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "scores",
        f64_array(py, &calls.iter().map(|c| c.score).collect::<Vec<_>>()),
    )?;
    out.set_item(
        "n_cells",
        i64_array(py, &calls.iter().map(|c| c.n_cells).collect::<Vec<_>>()),
    )?;
    Ok(out)
}

/// Per-cell ScType calls, optionally smoothed and with the hybrid
///
/// The twin of `rs_sc_type_assign_cells_graph` (and, without a graph, of
/// `rs_sc_type_assign_cells`).
///
/// ### Params
///
/// * `cell_types` - The cell types, in score column order
/// * `scores` - `(n_cells, n_cell_types)` `float64` scores, C-contiguous
/// * `graph` - Optional `(data, indices, indptr)` of the symmetric neighbour
///   CSR, one node per scored cell
/// * `cluster_labels` - Optional 0-based cluster per scored cell
/// * `params` - `to_rust(SctypeCellsParams)`
///
/// ### Returns
///
/// Dict with `assignments` (`int64`, `-1` Unknown), `scores`, `margins`,
/// `agreement` (`None` without a graph), and with `cluster_labels`
/// `hybrid_assignments` and `composition` (both `None` otherwise).
#[pyfunction]
#[pyo3(signature = (cell_types, scores, params, graph = None, cluster_labels = None))]
pub fn sc_type_assign_cells<'py>(
    py: Python<'py>,
    cell_types: Vec<String>,
    scores: PyReadonlyArray2<'py, f64>,
    params: &Bound<'py, PyAny>,
    graph: Option<CsrParts<'py>>,
    cluster_labels: Option<PyReadonlyArray1<'py, u32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let params: ScTypeCellParams = from_py(params, "sctype cell params")?;
    let res = sctype_res(cell_types, &scores)?;
    let lp_graph = match graph {
        Some((data, indices, indptr)) => {
            let n = indptr.len().saturating_sub(1);
            if n != res.n_cells {
                return Err(PyValueError::new_err(format!(
                    "The neighbour graph has {n} nodes, but the ScType results {} cells.",
                    res.n_cells
                )));
            }
            let adjacency = CompressedSparseData2::new_csr(
                data.c_slice()?,
                indices.c_slice()?,
                indptr.c_slice()?,
                None,
                (n, n),
            );
            let sparse = SparseGraph::new(n, adjacency, false);
            Some(KnnLabPropGraph::from_sparse_graph(&sparse))
        }
        None => None,
    };
    let labels = cluster_labels.as_ref().map(index_vec).transpose()?;

    let (cell_res, hybrid) = py
        .detach(|| {
            let cell_res = assign_cells(&res, lp_graph.as_ref(), Some(params))?;
            let hybrid = labels
                .as_deref()
                .map(|l| assign_hybrid(&res, &cell_res, l, Some(params.purity_threshold)))
                .transpose()?;
            Ok::<_, bixverse_sc::prelude::ScErrors>((cell_res, hybrid))
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("assignments", opt_index_array(py, &cell_res.assignments))?;
    out.set_item("scores", f64_array(py, &cell_res.scores))?;
    out.set_item("margins", f64_array(py, &cell_res.margins))?;
    out.set_item(
        "agreement",
        cell_res.agreement.as_deref().map(|a| f64_array(py, a)),
    )?;
    match hybrid {
        Some(h) => {
            out.set_item("hybrid_assignments", opt_index_array(py, &h.assignments))?;
            out.set_item(
                "composition",
                composition_dict(py, &h, cell_res.cell_types.len())?,
            )?;
        }
        None => {
            out.set_item("hybrid_assignments", py.None())?;
            out.set_item("composition", py.None())?;
        }
    }
    Ok(out)
}

//////////////
// Symphony //
//////////////

/// Build a Symphony reference
///
/// The twin of `rs_build_symphony_ref`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the CLR offsets
/// * `cell_indices` - 0-based cells, in embedding row order
/// * `hvg_indices` - 0-based HVGs, in loading row order
/// * `batch_labels` - One 0-based code vector per batch variable
/// * `pca_params` - `to_rust(PcaParams)`
/// * `no_pcs` - Number of components
/// * `harmony_params` - `to_rust(HarmonyParams)` or `to_rust(HarmonyV2Params)`
///   with `k` resolved
/// * `harmony_version` - `"v1"` or `"v2"`, matching `harmony_params`
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with the `float32` reference terms: `gene_means`, `gene_sds`,
/// `loadings` (n_hvgs x d), `z_orig` and `z_corr` (N x d), `r` (K x N),
/// `centroids` and `c` (K x d), `nr` (K).
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cell_indices, hvg_indices, batch_labels, pca_params, no_pcs,
    harmony_params, harmony_version, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn build_symphony_ref<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    hvg_indices: PyReadonlyArray1<'py, u32>,
    batch_labels: Vec<PyReadonlyArray1<'py, u32>>,
    pca_params: &Bound<'py, PyAny>,
    no_pcs: usize,
    harmony_params: &Bound<'py, PyAny>,
    harmony_version: &str,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let pca_params: SingleCellPcaParams = from_py(pca_params, "pca params")?;
    let backend = match harmony_version {
        "v1" => HarmonyBackend::V1(from_py::<HarmonyParams>(harmony_params, "harmony params")?),
        "v2" => HarmonyBackend::V2(from_py::<HarmonyParamsV2>(
            harmony_params,
            "harmony v2 params",
        )?),
        _ => {
            return Err(PyValueError::new_err(format!(
                "harmony_version must be 'v1' or 'v2', got '{harmony_version}'"
            )));
        }
    };
    let cells = index_vec(&cell_indices)?;
    let hvg = index_vec(&hvg_indices)?;
    let labels: Vec<Vec<usize>> = batch_labels
        .iter()
        .map(index_vec)
        .collect::<PyResult<_>>()?;
    // the crate asserts on these, so they have to fail here
    if labels.is_empty() || labels.iter().any(|l| l.len() != cells.len()) {
        return Err(PyValueError::new_err(
            "Need at least one batch variable, each with one label per cell.",
        ));
    }
    let reference = py
        .detach(|| {
            build_symphony_reference_on_store(
                f_path_gene,
                f_path_cell,
                &cells,
                &hvg,
                &labels,
                &pca_params,
                no_pcs,
                backend,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("gene_means", reference.gene_means.into_pyarray(py))?;
    out.set_item("gene_sds", reference.gene_sds.into_pyarray(py))?;
    out.set_item("loadings", mat_array(py, &reference.loadings)?)?;
    out.set_item("z_orig", mat_array(py, &reference.z_orig)?)?;
    out.set_item("z_corr", mat_array(py, &reference.z_corr)?)?;
    out.set_item("r", mat_array(py, &reference.r)?)?;
    out.set_item("centroids", mat_array(py, &reference.centroids)?)?;
    out.set_item("nr", reference.nr.into_pyarray(py))?;
    out.set_item("c", mat_array(py, &reference.c)?)?;
    Ok(out)
}

/// Map a query onto a Symphony reference
///
/// The twin of `rs_symphony_map_query`.
///
/// ### Params
///
/// * `f_path_query` - The query's gene-based binary
/// * `cell_indices` - 0-based query cells, in output row order
/// * `gene_means` - Reference per-HVG means
/// * `gene_sds` - Reference per-HVG standard deviations
/// * `loadings` - `(n_hvgs, d)` `float32` reference loadings
/// * `centroids` - `(K, d)` `float32` reference centroids
/// * `nr` - Reference cluster sizes
/// * `c_cache` - `(K, d)` `float32` compression term
/// * `gene_map` - Per reference HVG the 0-based query gene, `-1` if absent
/// * `batch_labels` - One 0-based code vector per query batch variable;
///   empty skips the correction
/// * `params` - `to_rust(SymphonyMapParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `z_pca` and `z_corr` (N_q x d) and `r` (K x N_q), `float32`.
#[pyfunction]
#[pyo3(signature = (
    f_path_query, cell_indices, gene_means, gene_sds, loadings, centroids, nr, c_cache,
    gene_map, batch_labels, params, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn symphony_map_query<'py>(
    py: Python<'py>,
    f_path_query: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_means: PyReadonlyArray1<'py, f32>,
    gene_sds: PyReadonlyArray1<'py, f32>,
    loadings: PyReadonlyArray2<'py, f32>,
    centroids: PyReadonlyArray2<'py, f32>,
    nr: PyReadonlyArray1<'py, f32>,
    c_cache: PyReadonlyArray2<'py, f32>,
    gene_map: PyReadonlyArray1<'py, i64>,
    batch_labels: Vec<PyReadonlyArray1<'py, u32>>,
    params: &Bound<'py, PyAny>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: SymphonyMapParams = from_py(params, "symphony map params")?;
    let cells = index_vec(&cell_indices)?;
    let labels: Vec<Vec<usize>> = batch_labels
        .iter()
        .map(index_vec)
        .collect::<PyResult<_>>()?;
    let gene_map: Vec<Option<usize>> = gene_map
        .c_slice()?
        .iter()
        .map(|&x| usize::try_from(x).ok())
        .collect();
    let (means, sds, nr) = (gene_means.c_slice()?, gene_sds.c_slice()?, nr.c_slice()?);
    let (loadings, centroids, c_cache) = (
        mat_view(&loadings)?,
        mat_view(&centroids)?,
        mat_view(&c_cache)?,
    );
    // the crate asserts on these, so they have to fail here
    let n_hvgs = means.len();
    if sds.len() != n_hvgs || loadings.nrows() != n_hvgs || gene_map.len() != n_hvgs {
        return Err(PyValueError::new_err(
            "gene_means, gene_sds, loadings and gene_map must cover the same HVGs.",
        ));
    }
    if labels.iter().any(|l| l.len() != cells.len()) {
        return Err(PyValueError::new_err(
            "Every batch variable needs one label per query cell.",
        ));
    }
    let reader = ParallelSparseReader::new(f_path_query).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            symphony_map_query_parts(
                &reader, &cells, means, sds, loadings, centroids, nr, c_cache, &gene_map, &labels,
                &params, verbose,
            )
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("z_pca", mat_array(py, &res.z_pca)?)?;
    out.set_item("z_corr", mat_array(py, &res.z_corr)?)?;
    out.set_item("r", mat_array(py, &res.r)?)?;
    Ok(out)
}

/// kNN majority vote of reference labels onto query cells
///
/// The twin of `rs_transfer_labels_symphony`.
///
/// ### Params
///
/// * `reference_z_corr` - `(N_ref, d)` `float32` reference embedding
/// * `query_z_corr` - `(N_q, d)` `float32` query embedding
/// * `reference_labels` - 0-based label code per reference cell
/// * `n_labels` - Number of codes
/// * `knn_params` - `to_rust(KnnParams)`
/// * `seed` - Seed for the index build
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(predicted, confidence)`: the `uint32` winning code and its `float64`
/// vote share per query cell.
#[pyfunction]
#[pyo3(signature = (
    reference_z_corr, query_z_corr, reference_labels, n_labels, knn_params, seed = 42,
    verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn transfer_labels_symphony<'py>(
    py: Python<'py>,
    reference_z_corr: PyReadonlyArray2<'py, f32>,
    query_z_corr: PyReadonlyArray2<'py, f32>,
    reference_labels: PyReadonlyArray1<'py, u32>,
    n_labels: usize,
    knn_params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let knn_params: KnnParams = from_py(knn_params, "knn params")?;
    let (reference, query) = (mat_view(&reference_z_corr)?, mat_view(&query_z_corr)?);
    let labels = index_vec(&reference_labels)?;
    if labels.len() != reference.nrows() || labels.iter().any(|&l| l >= n_labels) {
        return Err(PyValueError::new_err(
            "reference_labels needs one code below n_labels per reference cell.",
        ));
    }
    if reference.ncols() != query.ncols() {
        return Err(PyValueError::new_err(format!(
            "The reference embedding has {} dimensions, the query {}.",
            reference.ncols(),
            query.ncols()
        )));
    }
    let (pred, conf) = py
        .detach(|| {
            let (knn, _) =
                generate_knn_cross_with_dist(reference, query, &knn_params, seed, verbose > 0)?;
            Ok::<_, bixverse_sc::prelude::ScErrors>(knn_majority_vote(&knn, &labels, n_labels))
        })
        .map_err(BvErr::from)?;
    let pred: Vec<u32> = pred.into_iter().map(|x| x as u32).collect();
    Ok((pred.into_pyarray(py).into_any(), f64_array(py, &conf)))
}
