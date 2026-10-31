//! Differential expression and abundance, the Python twins of
//! `rs_calculate_dge_mann_whitney`, `rs_calculate_dge_one_vs_rest`,
//! `rs_calculate_dge_one_vs_many`, `rs_pseudobulk_cells_dense` /
//! `_sparse`, `rs_edger_ql`, `rs_nebula_sc`, `rs_make_milor_nhoods`,
//! `rs_spatial_fdr` and `rs_meld_sc`.
//!
//! The numerics live in `bixverse-sc`, `bixverse-methods` and `edge-rs`; this
//! only moves buffers. Every index is 0-based. Per-gene statistics that the
//! crates keep in `f32` widen to `float64` on the way out, as R's do.

use bixverse_methods::methods::dge_bulk::{EdgeRQlParams, run_edger_ql};
use bixverse_sc::mc_generation::cell_aggregation_utils::{
    get_pseudo_bulked_counts_dense, get_pseudo_bulked_counts_sparse, parse_pseudo_bulk,
};
use bixverse_sc::prelude::ScErrors;
use bixverse_sc::sc_analysis::dge_pathway_scores::{
    DgeAurocMultiRes, DgeMannWhitneyRes, calculate_dge_grps_mann_whitney,
    calculate_dge_one_vs_many_auroc, calculate_dge_one_vs_rest_mann_whitney,
};
use bixverse_sc::sc_analysis::meld::{MeldParams, meld as meld_rs};
use bixverse_sc::sc_analysis::milo_r::{MiloRParams, make_milor_nhoods, spatial_fdr as sfdr};
use bixverse_sc::sc_analysis::nebula::{NebulaScParams, run_nebula};
use bixverse_sc_data::data_io::ParallelSparseReader;
use edge_rs::glm::test::Tested;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::{dist_rows, flat_array, from_py, index_vec, knn_rows, mat_view};
use py_common::convert::{f64_array, mat_f64, u32_array};
use py_common::error::BvErr;
use py_common::results::nebula_dict;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/////////////
// Helpers //
/////////////

/// One cell index vector per group
///
/// ### Params
///
/// * `groups` - 0-based cell indices per group
///
/// ### Returns
///
/// The indices as `usize`.
fn group_lists(groups: &[PyReadonlyArray1<'_, u32>]) -> PyResult<Vec<Vec<usize>>> {
    groups.iter().map(index_vec).collect()
}

/// A C-contiguous `float64` matrix as its flat row-major values and shape
///
/// ### Params
///
/// * `m` - The matrix
///
/// ### Returns
///
/// `(values, nrows, ncols)`.
fn rows_f64(m: &PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<f64>, usize, usize)> {
    Ok((m.as_slice()?.to_vec(), m.shape()[0], m.shape()[1]))
}

/// The per-gene fields of a two-group Mann-Whitney result
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The result
///
/// ### Returns
///
/// Dict with `lfc`, `prop1`, `prop2`, `auroc`, `z_scores`, `p_values`, `fdr`
/// over the kept genes and the full-length `genes_to_keep` mask.
fn mann_whitney_dict<'py>(py: Python<'py>, res: DgeMannWhitneyRes) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("lfc", f64_array(py, &res.lfc))?;
    out.set_item("prop1", f64_array(py, &res.prop1))?;
    out.set_item("prop2", f64_array(py, &res.prop2))?;
    out.set_item("auroc", f64_array(py, &res.auroc))?;
    out.set_item("z_scores", res.z_scores.into_pyarray(py))?;
    out.set_item("p_values", res.p_vals.into_pyarray(py))?;
    out.set_item("fdr", res.fdr.into_pyarray(py))?;
    out.set_item("genes_to_keep", res.genes_to_keep.into_pyarray(py))?;
    Ok(out)
}

/// Per-rival vectors as one `(n_rivals, n_kept)` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `rows` - One vector per rival, all of the same length
///
/// ### Returns
///
/// The `float64` array.
fn rival_array<'py, T: Copy + Into<f64>>(
    py: Python<'py>,
    rows: &[Vec<T>],
) -> PyResult<Bound<'py, PyAny>> {
    let k = rows.first().map_or(0, Vec::len);
    let data: Vec<f64> = rows.iter().flatten().map(|&x| x.into()).collect();
    flat_array(py, data, rows.len(), k)
}

/// One reference arm of the one-vs-many test
///
/// ### Params
///
/// * `py` - GIL token
/// * `arm` - The arm
///
/// ### Returns
///
/// Dict with the per-rival fields as `(n_rivals, n_kept)` arrays (`auroc`,
/// `lfc`, `prop_other`, `z_scores`, `p_values`, `fdr`), the summaries over
/// the kept genes and `worst_comparison` as a position among the rivals.
fn one_vs_many_dict<'py>(py: Python<'py>, arm: DgeAurocMultiRes) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("auroc", rival_array(py, &arm.auroc)?)?;
    out.set_item("lfc", rival_array(py, &arm.lfc)?)?;
    out.set_item("prop_other", rival_array(py, &arm.prop_other)?)?;
    out.set_item("z_scores", rival_array(py, &arm.z_scores)?)?;
    out.set_item("p_values", rival_array(py, &arm.p_vals)?)?;
    out.set_item("fdr", rival_array(py, &arm.fdr)?)?;
    out.set_item("prop_ref", f64_array(py, &arm.prop_ref))?;
    out.set_item("median_auroc", f64_array(py, &arm.median_auroc))?;
    out.set_item("min_auroc", f64_array(py, &arm.min_auroc))?;
    out.set_item("mean_auroc", f64_array(py, &arm.mean_auroc))?;
    out.set_item("max_auroc", f64_array(py, &arm.max_auroc))?;
    out.set_item("worst_comparison", u32_array(py, &arm.worst_comparison))?;
    out.set_item("min_rank", u32_array(py, &arm.min_rank))?;
    out.set_item("simes_p", arm.simes_p.into_pyarray(py))?;
    out.set_item("simes_fdr", arm.simes_fdr.into_pyarray(py))?;
    out.set_item("max_p", arm.max_p.into_pyarray(py))?;
    out.set_item("max_p_fdr", arm.max_p_fdr.into_pyarray(py))?;
    out.set_item("genes_to_keep", arm.genes_to_keep.into_pyarray(py))?;
    Ok(out)
}

/////////
// DGE //
/////////

/// Mann-Whitney DGE between two cell groups
///
/// The twin of `rs_calculate_dge_mann_whitney`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `cells_1` - 0-based cells of group 1
/// * `cells_2` - 0-based cells of group 2
/// * `min_prop` - Minimum share of expressing cells in either group
/// * `alternative` - `"twosided"`, `"greater"` or `"less"`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [mann_whitney_dict].
#[pyfunction]
#[pyo3(signature = (f_path_cell, cells_1, cells_2, min_prop, alternative, verbose = false))]
pub fn dge_mann_whitney<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    cells_1: PyReadonlyArray1<'py, u32>,
    cells_2: PyReadonlyArray1<'py, u32>,
    min_prop: f32,
    alternative: &str,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (c1, c2) = (index_vec(&cells_1)?, index_vec(&cells_2)?);
    let reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            calculate_dge_grps_mann_whitney(
                &reader,
                &c1,
                &c2,
                min_prop,
                alternative,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    mann_whitney_dict(py, res)
}

/// One-vs-rest Mann-Whitney DGE for every group
///
/// The twin of `rs_calculate_dge_one_vs_rest`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `cell_groups` - 0-based cells per group
/// * `min_prop` - Minimum share of expressing cells in the group or the rest
/// * `alternative` - `"twosided"`, `"greater"` or `"less"`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// One [mann_whitney_dict] per group, in group order.
#[pyfunction]
#[pyo3(signature = (f_path_gene, cell_groups, min_prop, alternative, verbose = false))]
pub fn dge_one_vs_rest<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    cell_groups: Vec<PyReadonlyArray1<'py, u32>>,
    min_prop: f32,
    alternative: &str,
    verbose: bool,
) -> PyResult<Bound<'py, PyList>> {
    let groups = group_lists(&cell_groups)?;
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let arms = py
        .detach(|| {
            calculate_dge_one_vs_rest_mann_whitney(
                &reader,
                &groups,
                min_prop,
                alternative,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    let out = PyList::empty(py);
    for arm in arms {
        out.append(mann_whitney_dict(py, arm)?)?;
    }
    Ok(out)
}

/// One-vs-many AUROC DGE for specific markers
///
/// The twin of `rs_calculate_dge_one_vs_many`, before R's flattening.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `cell_groups` - 0-based cells per group
/// * `references` - 0-based positions in `cell_groups` of the reference arms
/// * `min_prop` - Minimum share of expressing cells in any group
/// * `alternative` - `"twosided"`, `"greater"` or `"less"`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// One [one_vs_many_dict] per reference, in reference order.
#[pyfunction]
#[pyo3(signature = (f_path_gene, cell_groups, references, min_prop, alternative, verbose = false))]
pub fn dge_one_vs_many<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    cell_groups: Vec<PyReadonlyArray1<'py, u32>>,
    references: PyReadonlyArray1<'py, u32>,
    min_prop: f32,
    alternative: &str,
    verbose: bool,
) -> PyResult<Bound<'py, PyList>> {
    let groups = group_lists(&cell_groups)?;
    let references = index_vec(&references)?;
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let arms = py
        .detach(|| {
            calculate_dge_one_vs_many_auroc(
                &reader,
                &groups,
                &references,
                min_prop,
                alternative,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    let out = PyList::empty(py);
    for arm in arms {
        out.append(one_vs_many_dict(py, arm)?)?;
    }
    Ok(out)
}

////////////////
// Pseudobulk //
////////////////

/// Pseudobulk groups of cells
///
/// The twin of `rs_pseudobulk_cells_dense` and `rs_pseudobulk_cells_sparse`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `cell_groups` - 0-based cells per pseudobulk sample
/// * `assay` - `"raw"` sums the raw counts, `"norm"` averages the normalised
///   ones
/// * `sparse` - Return the CSR parts instead of a dense array
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `(n_groups, n_genes)` `float64` array, or `(data, indices, indptr)`
/// of its CSR.
#[pyfunction]
#[pyo3(signature = (f_path_cell, cell_groups, assay, sparse = false, verbose = false))]
pub fn pseudobulk<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    cell_groups: Vec<PyReadonlyArray1<'py, u32>>,
    assay: &str,
    sparse: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let bulk_type = parse_pseudo_bulk(assay).ok_or_else(|| {
        PyValueError::new_err(format!("assay must be 'raw' or 'norm', got '{assay}'"))
    })?;
    let groups = group_lists(&cell_groups)?;
    let reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
    if sparse {
        let csr = py
            .detach(|| {
                get_pseudo_bulked_counts_sparse(&reader, &groups, bulk_type, verbose as usize)
            })
            .map_err(BvErr::from)?;
        let indptr: Vec<u64> = csr.indptr.iter().map(|&i| i as u64).collect();
        let parts = (
            csr.data.into_pyarray(py),
            csr.indices.into_pyarray(py),
            indptr.into_pyarray(py),
        );
        return Ok(parts.into_pyobject(py)?.into_any());
    }
    let dense = py
        .detach(|| get_pseudo_bulked_counts_dense(&reader, &groups, bulk_type, verbose as usize))
        .map_err(BvErr::from)?;
    mat_f64(py, &dense)
}

/// edgeR's quasi-likelihood chain on a count matrix
///
/// The twin of `rs_edger_ql`.
///
/// ### Params
///
/// * `counts` - `(n_features, n_samples)` `float64` raw counts, C-contiguous
/// * `design` - `(n_samples, n_coef)` `float64` design, C-contiguous
/// * `params` - `to_rust(EdgerQlParams)`
/// * `coef` - 0-based design columns to drop from the null
/// * `contrast` - Column-major `n_coef * n_contrasts` weights; wins over
///   `coef`
/// * `n_contrasts` - Contrast columns
///
/// ### Returns
///
/// Dict with `features_to_keep` over every feature and `log_fc`, `log_cpm`,
/// `f_stat`, `p_values`, `fdr` over the kept ones.
#[pyfunction]
#[pyo3(signature = (counts, design, params, coef = None, contrast = None, n_contrasts = 1))]
pub fn edger_ql<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
    params: &Bound<'py, PyAny>,
    coef: Option<Vec<usize>>,
    contrast: Option<Vec<f64>>,
    n_contrasts: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: EdgeRQlParams = from_py(params, "edgeR params")?;
    let tested = match (contrast, coef) {
        (Some(values), _) => Tested::Contrast {
            values,
            n_contrasts,
        },
        (None, Some(coef)) => Tested::Coef(coef),
        (None, None) => {
            return Err(PyValueError::new_err(
                "the edgeR test needs either `coef` or `contrast`",
            ));
        }
    };
    let (counts, n_features, n_samples) = rows_f64(&counts)?;
    let (design, n_design, n_coef) = rows_f64(&design)?;
    if n_design != n_samples {
        return Err(PyValueError::new_err(format!(
            "The design has {n_design} rows against {n_samples} samples in the counts."
        )));
    }
    let res = py
        .detach(|| {
            run_edger_ql(
                &counts, n_features, n_samples, &design, n_coef, &tested, &params,
            )
        })
        .map_err(|e| BvErr(ScErrors::from(e)))?;

    let out = PyDict::new(py);
    out.set_item("features_to_keep", res.genes_to_keep.into_pyarray(py))?;
    out.set_item("log_fc", res.log_fc.into_pyarray(py))?;
    out.set_item("log_cpm", res.log_cpm.into_pyarray(py))?;
    out.set_item("f_stat", res.f_stat.into_pyarray(py))?;
    out.set_item("p_values", res.p_val.into_pyarray(py))?;
    out.set_item("fdr", res.fdr.into_pyarray(py))?;
    Ok(out)
}

////////////
// NEBULA //
////////////

/// NEBULA over a cell selection
///
/// The twin of `rs_nebula_sc`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the library sizes
/// * `cells_to_keep` - 0-based cells, no duplicates
/// * `gene_indices` - 0-based genes to fit
/// * `subject_ids` - 0-based subject per store cell
/// * `design` - `(n_cells, n_coef)` `float64` design aligned to
///   `cells_to_keep`, C-contiguous
/// * `params` - `to_rust(NebulaParams)` plus `coef` (0-based) or `contrast`
/// * `offset` - Strictly positive scaling per selected cell; `None` uses the
///   library sizes
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with R's keys; `coefficients` and `se` as `(n_kept, n_coef)` arrays,
/// `cell_overdispersion_shrunk` `None` unless shrinkage was on.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cells_to_keep, gene_indices, subject_ids, design, params,
    offset = None, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn nebula<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    subject_ids: PyReadonlyArray1<'py, u32>,
    design: PyReadonlyArray2<'py, f64>,
    params: &Bound<'py, PyAny>,
    offset: Option<PyReadonlyArray1<'py, f64>>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: NebulaScParams = from_py(params, "NEBULA params")?;
    let cells = index_vec(&cells_to_keep)?;
    let genes = index_vec(&gene_indices)?;
    let subjects = index_vec(&subject_ids)?;
    let (design, _, n_coef) = rows_f64(&design)?;
    let offset = offset
        .map(|o| o.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let gene_reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let cell_reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            run_nebula(
                &gene_reader,
                &cell_reader,
                &cells,
                &genes,
                &subjects,
                &design,
                n_coef,
                offset.as_deref(),
                &params,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;

    nebula_dict(py, res)
}

//////////
// Milo //
//////////

/// Sample, refine and count the Milo neighbourhoods
///
/// The twin of `rs_make_milor_nhoods`.
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `knn_indices` - `(n_cells, k)` `uint32` kNN without self
/// * `sample_ids` - 0-based sample per cell
/// * `n_samples` - Number of samples
/// * `params` - The MiloR keys, kNN keys included
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `index_cell`, `nhoods_i`, `nhoods_j` (the membership COO),
/// `n_cells`, `kth_distances`, `sample_counts` as `(n_nhoods, n_samples)`
/// and `nhood_overlap`.
#[pyfunction]
#[pyo3(signature = (embd, knn_indices, sample_ids, n_samples, params, seed = 42, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn milor_nhoods<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: PyReadonlyArray2<'py, u32>,
    sample_ids: PyReadonlyArray1<'py, u32>,
    n_samples: usize,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: MiloRParams = from_py(params, "MiloR params")?;
    let embd = mat_view(&embd)?;
    let knn = knn_rows(&knn_indices)?;
    let samples = index_vec(&sample_ids)?;
    let res = py
        .detach(|| {
            make_milor_nhoods(
                embd,
                &knn,
                &samples,
                n_samples,
                &params,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;

    let n_nhoods = res.index_cells.len();
    let out = PyDict::new(py);
    out.set_item("index_cell", u32_array(py, &res.index_cells))?;
    out.set_item("nhoods_i", u32_array(py, &res.nhoods_i))?;
    out.set_item("nhoods_j", u32_array(py, &res.nhoods_j))?;
    out.set_item("n_cells", res.n_cells)?;
    out.set_item("kth_distances", res.kth_distances.into_pyarray(py))?;
    out.set_item(
        "sample_counts",
        flat_array(py, res.sample_counts, n_nhoods, n_samples)?,
    )?;
    out.set_item("nhood_overlap", res.nhood_overlap.into_pyarray(py))?;
    Ok(out)
}

/// Milo's spatial FDR
///
/// The twin of `rs_spatial_fdr`.
///
/// ### Params
///
/// * `p_values` - Raw p-value per neighbourhood
/// * `connectivity` - Matching k-distance or overlap
///
/// ### Returns
///
/// The adjusted p-values, in input order.
#[pyfunction]
pub fn spatial_fdr<'py>(
    py: Python<'py>,
    p_values: PyReadonlyArray1<'py, f64>,
    connectivity: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let res = sfdr(p_values.as_slice()?, connectivity.as_slice()?).map_err(BvErr::from)?;
    Ok(res.into_pyarray(py).into_any())
}

//////////
// MELD //
//////////

/// MELD over a provided kNN
///
/// The twin of `rs_meld_sc` with its kNN supplied.
///
/// ### Params
///
/// * `knn_indices` - `(n_cells, k)` `uint32` kNN without self
/// * `knn_dist` - The matching `float32` distances
/// * `labels` - 0-based sample per cell
/// * `n_labels` - Number of samples
/// * `params` - `to_rust(MeldParams)`
/// * `seed` - Seed for the largest eigenvalue estimate
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `raw_scores` and `norm_scores`, `(n_cells, n_labels)`
/// `float64`.
#[pyfunction]
#[pyo3(signature = (knn_indices, knn_dist, labels, n_labels, params, seed = 42, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn meld<'py>(
    py: Python<'py>,
    knn_indices: PyReadonlyArray2<'py, u32>,
    knn_dist: PyReadonlyArray2<'py, f32>,
    labels: PyReadonlyArray1<'py, u32>,
    n_labels: usize,
    params: &Bound<'py, PyAny>,
    seed: u64,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: MeldParams = from_py(params, "MELD params")?;
    let knn = knn_rows(&knn_indices)?;
    let dist = dist_rows(&knn_dist, knn_indices.shape())?;
    let labels = index_vec(&labels)?;
    let (raw, norm) = py
        .detach(|| {
            meld_rs(
                &knn,
                &dist,
                &labels,
                n_labels,
                &params,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("raw_scores", mat_f64(py, &raw)?)?;
    out.set_item("norm_scores", mat_f64(py, &norm)?)?;
    Ok(out)
}
