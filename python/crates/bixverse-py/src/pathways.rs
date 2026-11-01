//! Pathway activity, HotSpot and SCENIC: the Python twins of
//! `rs_module_scoring`, `rs_aucell`, `rs_regulon_thresholds`, `rs_vision`,
//! `rs_vision_with_autocorrelation`, `rs_lm_activity_sc`,
//! `rs_hotspot_autocor`, `rs_hotspot_gene_cor`, `rs_hotspot_cluster_genes`,
//! `rs_scenic_gene_filter`, `rs_scenic_grn(_streaming)`, `rs_top_k_targets`,
//! `rs_importance_threshold`, `rs_pairwise_gene_cors`, `rs_cistarget` and
//! their `rs_mc_*` meta cell siblings.
//!
//! The numerics live in `bixverse-sc` and `bixverse-methods`; this only moves
//! buffers. Every index is 0-based. A meta cell matrix crosses as its CSR
//! parts, see [crate::metacells::csr_from_parts]. Scores the crates keep in
//! `f32` widen to `float64` on the way out, as R's do.

use bixverse_methods::enrichment::lm_activity::{
    FeatureSetWeights, LmActivityParams, LmActivityRes,
};
use bixverse_methods::methods::cis_target::{cistarget_gene_sets, parse_rcc_type};
use bixverse_sc::mc_analysis::aucell::calculate_aucell_metacells;
use bixverse_sc::mc_analysis::hotspot_mc::{hotspot_autocor_metacells, hotspot_gene_cor_metacells};
use bixverse_sc::mc_analysis::lm_activity_mc::calculate_lm_activity_metacells;
use bixverse_sc::mc_analysis::metrics::pairwise_gene_correlations_in_memory;
use bixverse_sc::mc_analysis::scenic_metacells::run_scenic_grn_in_memory;
use bixverse_sc::mc_analysis::vision_mc::calculate_vision_metacells;
use bixverse_sc::prelude::{KnnParams, ScErrors};
use bixverse_sc::sc_analysis::dge_pathway_scores::{
    AucellParams, calculate_aucell, calculate_aucell_streaming,
};
use bixverse_sc::sc_analysis::hotspot::{
    HotSpotGeneRes, HotSpotPairRes, HotSpotParams, Hotspot, PAIR_STREAM_BATCH_SIZE,
    hotspot_gene_clusters, panel_size_from_mem,
};
use bixverse_sc::sc_analysis::lm_activity::calculate_lm_activity;
use bixverse_sc::sc_analysis::module_scoring::calculate_module_scores_main;
use bixverse_sc::sc_analysis::regulon_binarise::{BinariseParams, derive_regulon_thresholds};
use bixverse_sc::sc_analysis::scenic::{
    ImportanceLink, ScenicParams, importance_threshold_links, run_scenic_grn,
    run_scenic_grn_streaming, scenic_gene_filter, top_k_links,
};
use bixverse_sc::sc_analysis::vision::{
    SignatureGenes, VisionAutocorRes, calculate_vision, calculate_vision_streaming,
    vision_with_autocorrelation,
};
use bixverse_sc::sc_processing::knn::knn_or_generate;
use bixverse_sc::sc_processing::metrics::pairwise_gene_correlations;
use bixverse_sc_data::data_io::{ParallelSparseReader, SingleCellReading};
use faer::{Mat, MatRef};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::{
    csr_from_parts, f64_array, flat_array, from_py, index_vec, knn_pair, mat_f64, mat_view,
    u32_array,
};
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

///////////
// Types //
///////////

/// A signature as its positive and negative 0-based gene indices.
type PySignature<'py> = (PyReadonlyArray1<'py, u32>, PyReadonlyArray1<'py, u32>);

/// A weighted feature set: 0-based gene indices and their weights.
type PyFeatureSet<'py> = (PyReadonlyArray1<'py, u32>, PyReadonlyArray1<'py, f64>);

/////////////
// Helpers //
/////////////

/// One index vector per gene set
///
/// ### Params
///
/// * `sets` - 0-based gene indices per set
///
/// ### Returns
///
/// The indices as `usize`.
fn index_sets(sets: &[PyReadonlyArray1<'_, u32>]) -> PyResult<Vec<Vec<usize>>> {
    sets.iter().map(index_vec).collect()
}

/// VISION signatures from their index pairs
///
/// ### Params
///
/// * `sigs` - `(positive, negative)` 0-based gene indices per signature
///
/// ### Returns
///
/// The [SignatureGenes].
fn signatures(sigs: &[PySignature<'_>]) -> PyResult<Vec<SignatureGenes>> {
    sigs.iter()
        .map(|(pos, neg)| {
            Ok(SignatureGenes {
                positive: index_vec(pos)?,
                negative: index_vec(neg)?,
            })
        })
        .collect()
}

/// A weighted network as [FeatureSetWeights]
///
/// ### Params
///
/// * `net` - `(indices, weights)` per set
/// * `n_features` - Size of the gene universe
///
/// ### Returns
///
/// The feature sets, or an error if a set is empty, out of range or its
/// weights do not line up.
fn feature_sets(net: &[PyFeatureSet<'_>], n_features: usize) -> PyResult<FeatureSetWeights> {
    let mut sets = Vec::with_capacity(net.len());
    let mut weights = Vec::with_capacity(net.len());
    for (idx, w) in net {
        sets.push(index_vec(idx)?);
        weights.push(w.as_slice()?.to_vec());
    }
    FeatureSetWeights::new(&sets, Some(&weights), n_features)
        .map_err(|e| BvErr::from(ScErrors::from(e)).into())
}

/// Per-set score vectors as a `(n_cells, n_sets)` `float64` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - One vector of cell scores per set
///
/// ### Returns
///
/// The array.
fn by_set_array<'py>(py: Python<'py>, res: &[Vec<f32>]) -> PyResult<Bound<'py, PyAny>> {
    let n_sets = res.len();
    let n_cells = res.first().map_or(0, Vec::len);
    let data: Vec<f64> = (0..n_cells)
        .flat_map(|i| res.iter().map(move |s| s[i] as f64))
        .collect();
    flat_array(py, data, n_cells, n_sets)
}

/// Per-cell score vectors as a `(n_cells, n_sets)` `float64` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - One vector of set scores per cell
/// * `n_sets` - Number of sets, for the empty case
///
/// ### Returns
///
/// The array.
fn by_cell_array<'py>(
    py: Python<'py>,
    res: &[Vec<f32>],
    n_sets: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let data: Vec<f64> = res.iter().flatten().map(|&x| x as f64).collect();
    flat_array(py, data, res.len(), n_sets)
}

/// The kNN to use: the supplied one, or one searched on the embedding
///
/// ### Params
///
/// * `py` - GIL token
/// * `embd` - `(n_cells, n_dims)` `float32` embedding
/// * `knn_indices` - Optional `(n_cells, k)` kNN without self
/// * `knn_dist` - The matching distances
/// * `params` - The kNN keys of the parameter dict
/// * `seed` - Seed for the search
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(indices, distances)`.
#[allow(clippy::type_complexity)]
fn resolve_knn(
    py: Python<'_>,
    embd: &PyReadonlyArray2<'_, f32>,
    knn_indices: Option<&PyReadonlyArray2<'_, u32>>,
    knn_dist: Option<&PyReadonlyArray2<'_, f32>>,
    params: &KnnParams,
    seed: usize,
    verbose: bool,
) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
    let knn = knn_pair(knn_indices, knn_dist)?;
    if let Some((idx, _)) = &knn
        && idx.len() != embd.shape()[0]
    {
        return Err(PyValueError::new_err(format!(
            "the kNN has {} rows, the embedding {}",
            idx.len(),
            embd.shape()[0]
        )));
    }
    let embd = mat_view(embd)?;
    py.detach(|| knn_or_generate(knn, embd, params, seed, verbose))
        .map_err(|e| BvErr::from(e).into())
}

/// The VISION autocorrelation result as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The result
/// * `n_sets` - Number of signatures
///
/// ### Returns
///
/// Dict with `scores` (`(n_cells, n_sets)` `float64`), `auto_cor`, `p_val`
/// and `fdr`.
fn vision_autocor_dict<'py>(
    py: Python<'py>,
    res: VisionAutocorRes,
    n_sets: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("scores", by_cell_array(py, &res.scores, n_sets)?)?;
    out.set_item("auto_cor", res.auto_cor.into_pyarray(py))?;
    out.set_item("p_val", res.p_val.into_pyarray(py))?;
    out.set_item("fdr", res.fdr.into_pyarray(py))?;
    Ok(out)
}

/// Linear model activity scores as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The scores
///
/// ### Returns
///
/// Dict with `es` and `p_values` (`(n_cells, n_sets)` `float64`, the latter
/// `None` unless asked for).
fn lm_dict<'py>(py: Python<'py>, res: LmActivityRes<f32>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("es", mat_f64(py, &res.es)?)?;
    match res.p_values {
        Some(p) => out.set_item("p_values", mat_f64(py, &p)?)?,
        None => out.set_item("p_values", py.None())?,
    }
    Ok(out)
}

/// HotSpot autocorrelations as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - Per gene results
///
/// ### Returns
///
/// Dict with `gene_idx` (`uint32`), `gearys_c`, `z_score`, `pval` and `fdr`.
fn hotspot_gene_dict<'py>(py: Python<'py>, res: HotSpotGeneRes) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("gene_idx", u32_array(py, &res.gene_idx))?;
    out.set_item("gearys_c", res.c.into_pyarray(py))?;
    out.set_item("z_score", res.z.into_pyarray(py))?;
    out.set_item("pval", res.pval.into_pyarray(py))?;
    out.set_item("fdr", res.fdr.into_pyarray(py))?;
    Ok(out)
}

/// HotSpot pairwise local correlations as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The two gene x gene matrices
///
/// ### Returns
///
/// Dict with `cor` and `z`, `(n_genes, n_genes)` `float64`.
fn hotspot_pair_dict<'py>(py: Python<'py>, res: HotSpotPairRes) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("cor", mat_f64(py, &res.cor)?)?;
    out.set_item("z", mat_f64(py, &res.z_scores)?)?;
    Ok(out)
}

/// Importance links as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `links` - `(gene row, TF column, importance)` triplets
///
/// ### Returns
///
/// Dict with `gene` and `tf` (`uint32` positions) and `importance`.
fn links_dict<'py>(py: Python<'py>, links: Vec<ImportanceLink>) -> PyResult<Bound<'py, PyDict>> {
    let genes: Vec<usize> = links.iter().map(|l| l.0).collect();
    let tfs: Vec<usize> = links.iter().map(|l| l.1).collect();
    let imp: Vec<f64> = links.iter().map(|l| l.2).collect();
    let out = PyDict::new(py);
    out.set_item("gene", u32_array(py, &genes))?;
    out.set_item("tf", u32_array(py, &tfs))?;
    out.set_item("importance", imp.into_pyarray(py))?;
    Ok(out)
}

/// A row-major `float64` numpy matrix as a faer view
///
/// ### Params
///
/// * `m` - The C-contiguous matrix
///
/// ### Returns
///
/// The view.
fn mat_view_f64<'a>(m: &'a PyReadonlyArray2<'_, f64>) -> PyResult<MatRef<'a, f64>> {
    let (r, c) = (m.shape()[0], m.shape()[1]);
    Ok(MatRef::from_row_major_slice(m.as_slice()?, r, c))
}

///////////////////
// Module scores //
///////////////////

/// Module scores (Seurat's `AddModuleScore`)
///
/// The twin of `rs_module_scoring`.
///
/// ### Params
///
/// * `f_path_cells` - The cell-based binary
/// * `f_path_genes` - The gene-based binary
/// * `gene_sets` - 0-based gene indices per module
/// * `cells_to_keep` - 0-based cells to score
/// * `nbin` - Expression bins for the control genes
/// * `ctrl` - Control genes per module gene
/// * `seed` - Seed for the control gene draws
/// * `streaming` - Read the cells in chunks
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(n_cells, n_modules)` `float64` scores.
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, f_path_genes, gene_sets, cells_to_keep, nbin, ctrl, seed, streaming,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn module_scores<'py>(
    py: Python<'py>,
    f_path_cells: &str,
    f_path_genes: &str,
    gene_sets: Vec<PyReadonlyArray1<'py, u32>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    nbin: usize,
    ctrl: usize,
    seed: usize,
    streaming: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let sets = index_sets(&gene_sets)?;
    let cells = index_vec(&cells_to_keep)?;
    let gene_reader = ParallelSparseReader::new(f_path_genes).map_err(BvErr::from)?;
    let cell_reader = ParallelSparseReader::new(f_path_cells).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            calculate_module_scores_main(
                &gene_reader,
                &cell_reader,
                &sets,
                &cells,
                nbin,
                ctrl,
                streaming,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    by_set_array(py, &res)
}

////////////
// AUCell //
////////////

/// AUCell scores over the store
///
/// The twin of `rs_aucell`.
///
/// ### Params
///
/// * `f_path` - The cell-based binary
/// * `gene_sets` - 0-based gene indices per set
/// * `cells_to_keep` - 0-based cells to score
/// * `params` - `to_rust(AucellParams)`
/// * `streaming` - Read the cells in chunks
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(n_cells, n_sets)` `float64` scores.
#[pyfunction]
#[pyo3(signature = (f_path, gene_sets, cells_to_keep, params, streaming, verbose = false))]
pub fn aucell<'py>(
    py: Python<'py>,
    f_path: &str,
    gene_sets: Vec<PyReadonlyArray1<'py, u32>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    streaming: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let params: AucellParams = from_py(params, "AUCell params")?;
    let sets = index_sets(&gene_sets)?;
    let cells = index_vec(&cells_to_keep)?;
    let reader = ParallelSparseReader::new(f_path).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            if streaming {
                calculate_aucell_streaming(&reader, &sets, &cells, Some(params), verbose as usize)
            } else {
                calculate_aucell(&reader, &sets, &cells, Some(params), verbose as usize)
            }
        })
        .map_err(BvErr::from)?;
    by_set_array(py, &res)
}

/// AUCell scores over meta cells
///
/// The twin of `rs_mc_aucell`; the raw counts, truncated to integers as R
/// hands them over.
///
/// ### Params
///
/// * `data` - CSR raw counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `gene_sets` - 0-based gene indices per set
/// * `params` - `to_rust(AucellParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(n_metacells, n_sets)` `float64` scores.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, gene_sets, params, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn mc_aucell<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    gene_sets: Vec<PyReadonlyArray1<'py, u32>>,
    params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let params: AucellParams = from_py(params, "AUCell params")?;
    let sets = index_sets(&gene_sets)?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let res = py
        .detach(|| calculate_aucell_metacells(&m, &sets, Some(params), verbose as usize))
        .map_err(BvErr::from)?;
    by_set_array(py, &res)
}

/// On/off thresholds per regulon
///
/// The twin of `rs_regulon_thresholds`.
///
/// ### Params
///
/// * `auc` - `(n_cells, n_regulons)` `float64` scores, C-contiguous
/// * `params` - `to_rust(ScenicBinariseParams)`
///
/// ### Returns
///
/// Dict with `thresholds` (`float64`) and `bimodal` (`bool`), one per
/// regulon.
#[pyfunction]
pub fn regulon_thresholds<'py>(
    py: Python<'py>,
    auc: PyReadonlyArray2<'py, f64>,
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let params: BinariseParams = from_py(params, "binarisation params")?;
    let m = mat_view_f64(&auc)?;
    // the crate wants one row of cell scores per regulon, in f32 as AUCell
    // returns them
    let rows: Vec<Vec<f32>> = (0..m.ncols())
        .map(|j| (0..m.nrows()).map(|i| m[(i, j)] as f32).collect())
        .collect();
    let res = py
        .detach(|| derive_regulon_thresholds(&rows, Some(params)))
        .map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("thresholds", res.thresholds.into_pyarray(py))?;
    out.set_item("bimodal", res.bimodal)?;
    Ok(out)
}

////////////
// VISION //
////////////

/// VISION signature scores over the store
///
/// The twin of `rs_vision`.
///
/// ### Params
///
/// * `f_path` - The cell-based binary
/// * `sigs` - `(positive, negative)` 0-based gene indices per signature
/// * `cells_to_keep` - 0-based cells to score
/// * `streaming` - Read the cells in chunks
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(n_cells, n_signatures)` `float64` scores.
#[pyfunction]
#[pyo3(signature = (f_path, sigs, cells_to_keep, streaming, verbose = false))]
pub fn vision<'py>(
    py: Python<'py>,
    f_path: &str,
    sigs: Vec<PySignature<'py>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    streaming: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let sigs = signatures(&sigs)?;
    let cells = index_vec(&cells_to_keep)?;
    let reader = ParallelSparseReader::new(f_path).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            if streaming {
                calculate_vision_streaming(&reader, &sigs, &cells, verbose as usize)
            } else {
                calculate_vision(&reader, &sigs, &cells, verbose as usize)
            }
        })
        .map_err(BvErr::from)?;
    by_cell_array(py, &res, sigs.len())
}

/// VISION signature scores over meta cells
///
/// The twin of `rs_mc_vision`; the normalised counts.
///
/// ### Params
///
/// * `data` - CSR normalised counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `sigs` - `(positive, negative)` 0-based gene indices per signature
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(n_metacells, n_signatures)` `float64` scores.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, sigs, verbose = false))]
pub fn mc_vision<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    sigs: Vec<PySignature<'py>>,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let sigs = signatures(&sigs)?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x)?;
    let res = py
        .detach(|| calculate_vision_metacells(&m, &sigs, verbose as usize))
        .map_err(BvErr::from)?;
    by_cell_array(py, &res, sigs.len())
}

/// VISION with autocorrelation over the store
///
/// The twin of `rs_vision_with_autocorrelation`.
///
/// ### Params
///
/// * `f_path` - The cell-based binary
/// * `sigs` - `(positive, negative)` 0-based gene indices per signature
/// * `random_sigs` - Permuted signatures, one list per cluster
/// * `cluster_membership` - 0-based cluster per signature
/// * `cells_to_keep` - 0-based cells to score, the rows of `embd`
/// * `embd` - `(n_cells, n_dims)` `float32` embedding for the kNN search
/// * `knn_indices` - Optional `(n_cells, k)` kNN to use instead
/// * `knn_dist` - The matching distances
/// * `params` - `to_rust(VisionParams)`
/// * `streaming` - Read the cells in chunks
/// * `seed` - Seed for the kNN search
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [vision_autocor_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path, sigs, random_sigs, cluster_membership, cells_to_keep, embd, knn_indices,
    knn_dist, params, streaming, seed = 42, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn vision_autocor<'py>(
    py: Python<'py>,
    f_path: &str,
    sigs: Vec<PySignature<'py>>,
    random_sigs: Vec<Vec<PySignature<'py>>>,
    cluster_membership: PyReadonlyArray1<'py, u32>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_dist: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    streaming: bool,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: KnnParams = from_py(params, "VISION params")?;
    let sigs = signatures(&sigs)?;
    let random = random_sigs
        .iter()
        .map(|s| signatures(s))
        .collect::<PyResult<Vec<_>>>()?;
    let membership = index_vec(&cluster_membership)?;
    let cells = index_vec(&cells_to_keep)?;
    if embd.shape()[0] != cells.len() {
        return Err(PyValueError::new_err(
            "the embedding needs one row per cell to keep",
        ));
    }
    let knn = resolve_knn(
        py,
        &embd,
        knn_indices.as_ref(),
        knn_dist.as_ref(),
        &params,
        seed,
        verbose,
    )?;
    let reader = ParallelSparseReader::new(f_path).map_err(BvErr::from)?;
    let v = verbose as usize;
    let res = py
        .detach(|| {
            let score = |s: &[SignatureGenes]| {
                if streaming {
                    calculate_vision_streaming(&reader, s, &cells, v)
                } else {
                    calculate_vision(&reader, s, &cells, v)
                }
            };
            vision_with_autocorrelation(score, &sigs, &random, &membership, knn, v)
        })
        .map_err(BvErr::from)?;
    vision_autocor_dict(py, res, sigs.len())
}

/// VISION with autocorrelation over meta cells
///
/// The twin of `rs_mc_vision_with_autocorrelation`; the normalised counts.
///
/// ### Params
///
/// * `data` - CSR normalised counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `sigs` - `(positive, negative)` 0-based gene indices per signature
/// * `random_sigs` - Permuted signatures, one list per cluster
/// * `cluster_membership` - 0-based cluster per signature
/// * `embd` - `(n_metacells, n_dims)` `float32` embedding for the kNN search
/// * `knn_indices` - Optional `(n_metacells, k)` kNN to use instead
/// * `knn_dist` - The matching distances
/// * `params` - `to_rust(VisionParams)`
/// * `seed` - Seed for the kNN search
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [vision_autocor_dict].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, sigs, random_sigs, cluster_membership, embd, knn_indices,
    knn_dist, params, seed = 42, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn mc_vision_autocor<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    sigs: Vec<PySignature<'py>>,
    random_sigs: Vec<Vec<PySignature<'py>>>,
    cluster_membership: PyReadonlyArray1<'py, u32>,
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_dist: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: KnnParams = from_py(params, "VISION params")?;
    let sigs = signatures(&sigs)?;
    let random = random_sigs
        .iter()
        .map(|s| signatures(s))
        .collect::<PyResult<Vec<_>>>()?;
    let membership = index_vec(&cluster_membership)?;
    if embd.shape()[0] != shape.0 {
        return Err(PyValueError::new_err(
            "the embedding needs one row per meta cell",
        ));
    }
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x)?;
    let knn = resolve_knn(
        py,
        &embd,
        knn_indices.as_ref(),
        knn_dist.as_ref(),
        &params,
        seed,
        verbose,
    )?;
    let v = verbose as usize;
    let res = py
        .detach(|| {
            let score = |s: &[SignatureGenes]| calculate_vision_metacells(&m, s, v);
            vision_with_autocorrelation(score, &sigs, &random, &membership, knn, v)
        })
        .map_err(BvErr::from)?;
    vision_autocor_dict(py, res, sigs.len())
}

/////////////////
// LM activity //
/////////////////

/// ULM / MLM activity scores over the store
///
/// The twin of `rs_lm_activity_sc`; the normalised counts over every gene.
///
/// ### Params
///
/// * `f_path` - The cell-based binary
/// * `net` - `(indices, weights)` per source, 0-based gene indices
/// * `cells_to_keep` - 0-based cells to score
/// * `params` - `to_rust(LmActivityParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [lm_dict].
#[pyfunction]
#[pyo3(signature = (f_path, net, cells_to_keep, params, verbose = false))]
pub fn lm_activity<'py>(
    py: Python<'py>,
    f_path: &str,
    net: Vec<PyFeatureSet<'py>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: LmActivityParams = from_py(params, "lm activity params")?;
    let reader = ParallelSparseReader::new(f_path).map_err(BvErr::from)?;
    let net = feature_sets(&net, reader.get_header().total_genes)?;
    let cells = index_vec(&cells_to_keep)?;
    let res = py
        .detach(|| calculate_lm_activity(&reader, &net, &cells, Some(params), verbose as usize))
        .map_err(BvErr::from)?;
    lm_dict(py, res)
}

/// ULM / MLM activity scores over meta cells
///
/// The twin of `rs_mc_lm_activity`; the normalised counts.
///
/// ### Params
///
/// * `data` - CSR normalised counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `net` - `(indices, weights)` per source, 0-based gene indices
/// * `params` - `to_rust(LmActivityParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [lm_dict].
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, net, params, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn mc_lm_activity<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    net: Vec<PyFeatureSet<'py>>,
    params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: LmActivityParams = from_py(params, "lm activity params")?;
    let net = feature_sets(&net, shape.1)?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x)?;
    let res = py
        .detach(|| calculate_lm_activity_metacells(&m, &net, Some(params), verbose as usize))
        .map_err(BvErr::from)?;
    lm_dict(py, res)
}

/////////////
// HotSpot //
/////////////

/// HotSpot gene autocorrelation over the store
///
/// The twin of `rs_hotspot_autocor`.
///
/// ### Params
///
/// * `f_path_genes` - The gene-based binary
/// * `f_path_cells` - The cell-based binary
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, rows aligned with
///   `cells_to_keep`
/// * `knn_indices` - Optional `(n_cells, k)` kNN to use instead of a search
/// * `knn_dist` - The matching distances
/// * `params` - `to_rust(HotspotParams)`
/// * `cells_to_keep` - 0-based cells
/// * `genes_to_use` - 0-based genes to test
/// * `streaming` - Stream the genes
/// * `seed` - Seed for the kNN search
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [hotspot_gene_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_genes, f_path_cells, embd, knn_indices, knn_dist, params, cells_to_keep,
    genes_to_use, streaming, seed = 42, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn hotspot_autocor<'py>(
    py: Python<'py>,
    f_path_genes: &str,
    f_path_cells: &str,
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_dist: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    genes_to_use: PyReadonlyArray1<'py, u32>,
    streaming: bool,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: HotSpotParams = from_py(params, "HotSpot params")?;
    let cells = index_vec(&cells_to_keep)?;
    let genes = index_vec(&genes_to_use)?;
    if embd.shape()[0] != cells.len() {
        return Err(PyValueError::new_err(
            "the embedding needs one row per cell to keep",
        ));
    }
    let (knn, dist) = resolve_knn(
        py,
        &embd,
        knn_indices.as_ref(),
        knn_dist.as_ref(),
        &params.knn_params,
        seed,
        verbose,
    )?;
    let gene_reader = ParallelSparseReader::new(f_path_genes).map_err(BvErr::from)?;
    let cell_reader = ParallelSparseReader::new(f_path_cells).map_err(BvErr::from)?;
    let v = verbose as usize;
    let res = py
        .detach(|| {
            let mut hs = Hotspot::new(
                &gene_reader,
                &cell_reader,
                &cells,
                &knn,
                &dist,
                Some(params.graph_params),
            )?;
            if streaming {
                hs.compute_all_genes_streaming(&genes, &params.model, params.normalise, v)
            } else {
                hs.compute_all_genes(&genes, &params.model, params.normalise, v)
            }
        })
        .map_err(BvErr::from)?;
    hotspot_gene_dict(py, res)
}

/// HotSpot pairwise local correlations over the store
///
/// The twin of `rs_hotspot_gene_cor`.
///
/// ### Params
///
/// * `f_path_genes` - The gene-based binary
/// * `f_path_cells` - The cell-based binary
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, rows aligned with
///   `cells_to_keep`
/// * `knn_indices` - Optional `(n_cells, k)` kNN to use instead of a search
/// * `knn_dist` - The matching distances
/// * `params` - `to_rust(HotspotParams)`
/// * `cells_to_keep` - 0-based cells
/// * `genes_to_use` - 0-based genes to correlate
/// * `working_mem_gb` - Panel memory budget of the streaming path
/// * `streaming` - Stream the genes
/// * `seed` - Seed for the kNN search
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [hotspot_pair_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_genes, f_path_cells, embd, knn_indices, knn_dist, params, cells_to_keep,
    genes_to_use, working_mem_gb, streaming, seed = 42, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn hotspot_gene_cor<'py>(
    py: Python<'py>,
    f_path_genes: &str,
    f_path_cells: &str,
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_dist: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    genes_to_use: PyReadonlyArray1<'py, u32>,
    working_mem_gb: f64,
    streaming: bool,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: HotSpotParams = from_py(params, "HotSpot params")?;
    let cells = index_vec(&cells_to_keep)?;
    let genes = index_vec(&genes_to_use)?;
    if embd.shape()[0] != cells.len() {
        return Err(PyValueError::new_err(
            "the embedding needs one row per cell to keep",
        ));
    }
    let (knn, dist) = resolve_knn(
        py,
        &embd,
        knn_indices.as_ref(),
        knn_dist.as_ref(),
        &params.knn_params,
        seed,
        verbose,
    )?;
    let gene_reader = ParallelSparseReader::new(f_path_genes).map_err(BvErr::from)?;
    let cell_reader = ParallelSparseReader::new(f_path_cells).map_err(BvErr::from)?;
    let v = verbose as usize;
    let res = py
        .detach(|| {
            let mut hs = Hotspot::new(
                &gene_reader,
                &cell_reader,
                &cells,
                &knn,
                &dist,
                Some(params.graph_params),
            )?;
            if streaming {
                let panel = panel_size_from_mem(working_mem_gb, knn.len(), genes.len());
                hs.compute_gene_cor_streaming(
                    &genes,
                    &params.model,
                    PAIR_STREAM_BATCH_SIZE,
                    panel,
                    v,
                )
            } else {
                hs.compute_gene_cor(&genes, &params.model, v)
            }
        })
        .map_err(BvErr::from)?;
    hotspot_pair_dict(py, res)
}

/// HotSpot gene autocorrelation over meta cells
///
/// The twin of `rs_mc_hotspot_autocor`; the raw counts, truncated to integers
/// as R hands them over.
///
/// ### Params
///
/// * `data` - CSR raw counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `embd` - `(n_kept, n_dims)` `float32` embedding, rows aligned with
///   `cells_to_keep`
/// * `knn_indices` - Optional `(n_kept, k)` kNN to use instead of a search
/// * `knn_dist` - The matching distances
/// * `params` - `to_rust(HotspotParams)`
/// * `cells_to_keep` - 0-based meta cells
/// * `genes_to_use` - 0-based genes to test
/// * `seed` - Seed for the kNN search
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [hotspot_gene_dict].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, embd, knn_indices, knn_dist, params, cells_to_keep,
    genes_to_use, seed = 42, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn mc_hotspot_autocor<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_dist: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    genes_to_use: PyReadonlyArray1<'py, u32>,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: HotSpotParams = from_py(params, "HotSpot params")?;
    let cells = index_vec(&cells_to_keep)?;
    let genes = index_vec(&genes_to_use)?;
    if embd.shape()[0] != cells.len() {
        return Err(PyValueError::new_err(
            "the embedding needs one row per meta cell to keep",
        ));
    }
    let (knn, dist) = resolve_knn(
        py,
        &embd,
        knn_indices.as_ref(),
        knn_dist.as_ref(),
        &params.knn_params,
        seed,
        verbose,
    )?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let res = py
        .detach(|| {
            hotspot_autocor_metacells(&m, &knn, &dist, &cells, &genes, &params, verbose as usize)
        })
        .map_err(BvErr::from)?;
    hotspot_gene_dict(py, res)
}

/// HotSpot pairwise local correlations over meta cells
///
/// The twin of `rs_mc_hotspot_gene_cor`; the raw counts.
///
/// ### Params
///
/// * `data` - CSR raw counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `embd` - `(n_kept, n_dims)` `float32` embedding, rows aligned with
///   `cells_to_keep`
/// * `knn_indices` - Optional `(n_kept, k)` kNN to use instead of a search
/// * `knn_dist` - The matching distances
/// * `params` - `to_rust(HotspotParams)`
/// * `cells_to_keep` - 0-based meta cells
/// * `genes_to_use` - 0-based genes to correlate
/// * `seed` - Seed for the kNN search
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [hotspot_pair_dict].
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, embd, knn_indices, knn_dist, params, cells_to_keep,
    genes_to_use, seed = 42, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn mc_hotspot_gene_cor<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_dist: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    genes_to_use: PyReadonlyArray1<'py, u32>,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: HotSpotParams = from_py(params, "HotSpot params")?;
    let cells = index_vec(&cells_to_keep)?;
    let genes = index_vec(&genes_to_use)?;
    if embd.shape()[0] != cells.len() {
        return Err(PyValueError::new_err(
            "the embedding needs one row per meta cell to keep",
        ));
    }
    let (knn, dist) = resolve_knn(
        py,
        &embd,
        knn_indices.as_ref(),
        knn_dist.as_ref(),
        &params.knn_params,
        seed,
        verbose,
    )?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let res = py
        .detach(|| {
            hotspot_gene_cor_metacells(&m, &knn, &dist, &cells, &genes, &params, verbose as usize)
        })
        .map_err(BvErr::from)?;
    hotspot_pair_dict(py, res)
}

/// HotSpot gene modules from the pairwise Z-scores
///
/// The twin of `rs_hotspot_cluster_genes`.
///
/// ### Params
///
/// * `z` - `(n_genes, n_genes)` `float64` symmetric Z-scores, zero diagonal
/// * `fdr_threshold` - BH level at which a pair counts as significant
/// * `min_size` - Minimum genes per module
///
/// ### Returns
///
/// One `float64` module label per gene, 0-based and dense; `NaN` for an
/// unassigned gene.
#[pyfunction]
pub fn hotspot_cluster_genes<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    fdr_threshold: f64,
    min_size: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if z.shape()[0] != z.shape()[1] {
        return Err(PyValueError::new_err("z must be square"));
    }
    let m = mat_view_f64(&z)?;
    let res = py.detach(|| hotspot_gene_clusters(m, fdr_threshold, min_size));
    Ok(res.into_pyarray(py).into_any())
}

////////////
// SCENIC //
////////////

/// SCENIC gene filter over the store
///
/// The twin of `rs_scenic_gene_filter`.
///
/// ### Params
///
/// * `f_path_genes` - The gene-based binary
/// * `cell_indices` - 0-based cells
/// * `params` - `to_rust(ScenicParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The 0-based genes that pass, `uint32`.
#[pyfunction]
#[pyo3(signature = (f_path_genes, cell_indices, params, verbose = false))]
pub fn scenic_filter<'py>(
    py: Python<'py>,
    f_path_genes: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let params: ScenicParams = from_py(params, "SCENIC params")?;
    let cells = index_vec(&cell_indices)?;
    let reader = ParallelSparseReader::new(f_path_genes).map_err(BvErr::from)?;
    let res = py
        .detach(|| scenic_gene_filter(&reader, &cells, &params, verbose as usize))
        .map_err(BvErr::from)?;
    Ok(u32_array(py, &res))
}

/// SCENIC GRN inference over the store
///
/// The twin of `rs_scenic_grn` and `rs_scenic_grn_streaming`.
///
/// ### Params
///
/// * `f_path_genes` - The gene-based binary
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based target genes
/// * `tf_indices` - 0-based TFs, a subset of the targets
/// * `params` - `to_rust(ScenicParams)`
/// * `streaming` - Stream the targets
/// * `seed` - Seed for the learners
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(n_targets, n_tfs)` `float64` importances.
#[pyfunction]
#[pyo3(signature = (
    f_path_genes, cell_indices, gene_indices, tf_indices, params, streaming, seed = 42,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn scenic_grn<'py>(
    py: Python<'py>,
    f_path_genes: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    tf_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    streaming: bool,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let params: ScenicParams = from_py(params, "SCENIC params")?;
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let tfs = index_vec(&tf_indices)?;
    let reader = ParallelSparseReader::new(f_path_genes).map_err(BvErr::from)?;
    let v = verbose as usize;
    let res: Mat<f32> = py
        .detach(|| {
            if streaming {
                run_scenic_grn_streaming(&reader, &cells, &genes, &tfs, &params, seed, v)
            } else {
                run_scenic_grn(&reader, &cells, &genes, &tfs, &params, seed, v)
            }
        })
        .map_err(BvErr::from)?;
    mat_f64(py, &res)
}

/// SCENIC GRN inference over meta cells
///
/// The twin of `rs_mc_scenic`; the raw counts of the target genes, truncated
/// to integers as R hands them over.
///
/// ### Params
///
/// * `data` - CSR raw counts, columns the target genes
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_targets)`
/// * `tf_indices` - 0-based TF columns of the matrix
/// * `params` - `to_rust(ScenicParams)`
/// * `seed` - Seed for the learners
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(n_targets, n_tfs)` `float64` importances.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, tf_indices, params, seed = 42, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn mc_scenic_grn<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    tf_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let params: ScenicParams = from_py(params, "SCENIC params")?;
    let tfs = index_vec(&tf_indices)?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let res = py
        .detach(|| run_scenic_grn_in_memory(&m, &tfs, &params, seed, verbose as usize))
        .map_err(BvErr::from)?;
    mat_f64(py, &res)
}

/// The strongest TF to gene links per gene or per TF
///
/// The twin of `rs_top_k_targets`.
///
/// ### Params
///
/// * `importance` - `(n_genes, n_tfs)` `float64`, C-contiguous
/// * `k` - Links per gene or TF
/// * `by_gene` - Per gene (`True`) or per TF (`False`)
/// * `min_value` - Optional importance floor
///
/// ### Returns
///
/// See [links_dict].
#[pyfunction]
#[pyo3(signature = (importance, k, by_gene, min_value = None))]
pub fn top_k_targets<'py>(
    py: Python<'py>,
    importance: PyReadonlyArray2<'py, f64>,
    k: usize,
    by_gene: bool,
    min_value: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = mat_view_f64(&importance)?;
    links_dict(py, top_k_links(m, k, by_gene, min_value))
}

/// TF to gene links above a per-gene `mean + n_sd * sd`
///
/// The twin of `rs_importance_threshold`.
///
/// ### Params
///
/// * `importance` - `(n_genes, n_tfs)` `float64`, C-contiguous
/// * `n_sd` - Standard deviations above the mean
/// * `min_value` - Optional importance floor
///
/// ### Returns
///
/// See [links_dict].
#[pyfunction]
#[pyo3(signature = (importance, n_sd, min_value = None))]
pub fn importance_threshold<'py>(
    py: Python<'py>,
    importance: PyReadonlyArray2<'py, f64>,
    n_sd: f64,
    min_value: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = mat_view_f64(&importance)?;
    links_dict(py, importance_threshold_links(m, n_sd, min_value))
}

/// Pairwise gene correlations over the store
///
/// The twin of `rs_pairwise_gene_cors`.
///
/// ### Params
///
/// * `f_path_genes` - The gene-based binary
/// * `genes_1` - 0-based first genes
/// * `genes_2` - 0-based second genes, paired with `genes_1`
/// * `cells_to_keep` - 0-based cells
/// * `spearman` - Spearman rather than Pearson
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// One `float64` correlation per pair.
#[pyfunction]
#[pyo3(signature = (f_path_genes, genes_1, genes_2, cells_to_keep, spearman, verbose = false))]
pub fn pairwise_gene_cors<'py>(
    py: Python<'py>,
    f_path_genes: &str,
    genes_1: PyReadonlyArray1<'py, u32>,
    genes_2: PyReadonlyArray1<'py, u32>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    spearman: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (g1, g2) = (index_vec(&genes_1)?, index_vec(&genes_2)?);
    if g1.len() != g2.len() {
        return Err(PyValueError::new_err(
            "genes_1 and genes_2 need the same length",
        ));
    }
    let cells = index_vec(&cells_to_keep)?;
    let reader = ParallelSparseReader::new(f_path_genes).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            pairwise_gene_correlations(&reader, &g1, &g2, &cells, spearman, verbose as usize)
        })
        .map_err(BvErr::from)?;
    Ok(f64_array(py, &res))
}

/// Pairwise gene correlations over meta cells
///
/// The twin of `rs_pairwise_gene_cors_mc`; the normalised counts.
///
/// ### Params
///
/// * `data` - CSR normalised counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `genes_1` - 0-based first genes
/// * `genes_2` - 0-based second genes, paired with `genes_1`
/// * `spearman` - Spearman rather than Pearson
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// One `float64` correlation per pair.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, genes_1, genes_2, spearman, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn mc_pairwise_gene_cors<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    genes_1: PyReadonlyArray1<'py, u32>,
    genes_2: PyReadonlyArray1<'py, u32>,
    spearman: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (g1, g2) = (index_vec(&genes_1)?, index_vec(&genes_2)?);
    if g1.len() != g2.len() {
        return Err(PyValueError::new_err(
            "genes_1 and genes_2 need the same length",
        ));
    }
    let csc = csr_from_parts((&data, &indices, &indptr, shape), |x| x)?.transform();
    let res = py
        .detach(|| pairwise_gene_correlations_in_memory(&csc, &g1, &g2, spearman, verbose as usize))
        .map_err(BvErr::from)?;
    Ok(f64_array(py, &res))
}

///////////////
// CisTarget //
///////////////

/// Motif enrichment of gene sets over a ranking database
///
/// The twin of `rs_cistarget`.
///
/// ### Params
///
/// * `rankings` - `(n_genes, n_motifs)` `int32` ranks, C-contiguous
/// * `gene_sets` - 0-based gene (row) indices per set
/// * `auc_threshold` - Maximum rank for the AUC
/// * `nes_threshold` - Minimum NES to report a motif
/// * `max_rank` - Depth of the recovery curves
/// * `method` - `"approx"` or `"icistarget"`
/// * `n_mean` - Smoothing window of `"approx"`
/// * `verbose` - Print the progress
///
/// ### Returns
///
/// One dict per set with `motif_idx` (`uint32`), `nes`, `auc`,
/// `rank_at_max` (`uint32`), `n_enriched` (`uint32`) and `leading_edge` (one
/// `uint32` array of 0-based rows per motif).
#[pyfunction]
#[pyo3(signature = (
    rankings, gene_sets, auc_threshold, nes_threshold, max_rank, method, n_mean,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn cistarget<'py>(
    py: Python<'py>,
    rankings: PyReadonlyArray2<'py, i32>,
    gene_sets: Vec<PyReadonlyArray1<'py, u32>>,
    auc_threshold: i32,
    nes_threshold: f64,
    max_rank: i32,
    method: &str,
    n_mean: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyList>> {
    let method = parse_rcc_type(method).ok_or_else(|| {
        PyValueError::new_err(format!(
            "method must be 'approx' or 'icistarget', got '{method}'"
        ))
    })?;
    let sets = index_sets(&gene_sets)?;
    let (r, c) = (rankings.shape()[0], rankings.shape()[1]);
    let m = MatRef::from_row_major_slice(rankings.as_slice()?, r, c);
    let res = py.detach(|| {
        cistarget_gene_sets(
            m,
            &sets,
            auc_threshold,
            nes_threshold,
            max_rank,
            &method,
            n_mean,
            verbose,
        )
    });

    let out = PyList::empty(py);
    for set in res {
        let d = PyDict::new(py);
        let motif: Vec<usize> = set.iter().map(|m| m.motif_idx).collect();
        let rank: Vec<u32> = set.iter().map(|m| m.rank_at_max).collect();
        let n_enriched: Vec<usize> = set.iter().map(|m| m.n_enriched).collect();
        d.set_item("motif_idx", u32_array(py, &motif))?;
        d.set_item(
            "nes",
            set.iter()
                .map(|m| m.nes)
                .collect::<Vec<f64>>()
                .into_pyarray(py),
        )?;
        d.set_item(
            "auc",
            set.iter()
                .map(|m| m.auc)
                .collect::<Vec<f64>>()
                .into_pyarray(py),
        )?;
        d.set_item("rank_at_max", rank.into_pyarray(py))?;
        d.set_item("n_enriched", u32_array(py, &n_enriched))?;
        let le = PyList::empty(py);
        for m in &set {
            le.append(u32_array(py, &m.enriched_gene_indices))?;
        }
        d.set_item("leading_edge", le)?;
        out.append(d)?;
    }
    Ok(out)
}
