//! DIALOGUE: the Python twins of `rs_dialogue_sc` and `rs_mc_dialogue`.
//!
//! The numerics live in `bixverse-sc`; this only moves buffers. Every index is
//! 0-based. The feature matrices cross as row-major `float64`, one per cell
//! type, rows aligned to that cell type's indices.

use bixverse_sc::mc_analysis::dialogue_mc::dialogue_metacells;
use bixverse_sc::sc_analysis::dialogue::{
    DialogueParams, DialogueResult, HlmParams, PmdParams, ProgrammeSignature, RefineParams,
    dialogue_run,
};
use bixverse_sc_data::data_io::ParallelSparseReader;
use faer::MatRef;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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

/// The three stage blocks as one [DialogueParams]
///
/// ### Params
///
/// * `pmd` - `to_rust(DialoguePmdParams)`
/// * `hlm` - `to_rust(DialogueHlmParams)`
/// * `refine` - `to_rust(DialogueRefineParams)`
///
/// ### Returns
///
/// The parameters.
fn dialogue_params(
    pmd: &Bound<'_, PyAny>,
    hlm: &Bound<'_, PyAny>,
    refine: &Bound<'_, PyAny>,
) -> PyResult<DialogueParams> {
    let pmd: PmdParams = from_py(pmd, "DIALOGUE pmd params")?;
    let hlm: HlmParams = from_py(hlm, "DIALOGUE hlm params")?;
    let refine: RefineParams = from_py(refine, "DIALOGUE refine params")?;
    Ok(DialogueParams::new(pmd, hlm, refine))
}

/// Row-major feature matrices as faer views, without copying
///
/// ### Params
///
/// * `features` - C-contiguous `(n_cells_in_type, k_i)` matrices
///
/// ### Returns
///
/// The views.
fn feature_views<'a>(features: &'a [PyReadonlyArray2<'_, f64>]) -> PyResult<Vec<MatRef<'a, f64>>> {
    features
        .iter()
        .map(|m| {
            let (n, d) = (m.shape()[0], m.shape()[1]);
            Ok(MatRef::from_row_major_slice(m.as_slice()?, n, d))
        })
        .collect()
}

/// The `[cell_type][programme]` signatures as nested lists of dicts
///
/// ### Params
///
/// * `py` - GIL token
/// * `sigs` - The signatures
///
/// ### Returns
///
/// A list per cell type of dicts with `up` and `down` (`uint32` genes).
fn signatures<'py>(
    py: Python<'py>,
    sigs: &[Vec<ProgrammeSignature>],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for per_type in sigs {
        let inner = PyList::empty(py);
        for sig in per_type {
            let d = PyDict::new(py);
            d.set_item("up", u32_array(py, &sig.up))?;
            d.set_item("down", u32_array(py, &sig.down))?;
            inner.append(d)?;
        }
        out.append(inner)?;
    }
    Ok(out)
}

/// The [DialogueResult] as a dict, the shape R's list has
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The result
///
/// ### Returns
///
/// Dict with `shared_samples`, `kept_features`, `mcp_cell_types`, `ws`,
/// `scores`, `cca_scores`, `emp_p` (`k x n_pairs`), `pair_cor`,
/// `refit_fidelity` (`n_cell_types x k`), `verdicts` (a dict of columns),
/// `permissive` and `strict`.
fn result_dict<'py>(py: Python<'py>, res: &DialogueResult) -> PyResult<Bound<'py, PyDict>> {
    let mats = |ms: &[faer::Mat<f64>]| -> PyResult<Bound<'py, PyList>> {
        let l = PyList::empty(py);
        for m in ms {
            l.append(mat_f64(py, m)?)?;
        }
        Ok(l)
    };
    let idx_lists = |ls: &[Vec<usize>]| -> PyResult<Bound<'py, PyList>> {
        let l = PyList::empty(py);
        for v in ls {
            l.append(u32_array(py, v))?;
        }
        Ok(l)
    };

    let v = &res.verdicts;
    let col = |f: fn(&bixverse_sc::sc_analysis::dialogue::GeneVerdict) -> usize| {
        u32_array(py, &v.iter().map(f).collect::<Vec<usize>>())
    };
    let verdicts = PyDict::new(py);
    verdicts.set_item("cell_type", col(|g| g.cell_type))?;
    verdicts.set_item("programme", col(|g| g.programme))?;
    verdicts.set_item("gene", col(|g| g.gene))?;
    verdicts.set_item("up", v.iter().map(|g| g.up).collect::<Vec<bool>>())?;
    verdicts.set_item("n_supporting", col(|g| g.n_supporting))?;
    verdicts.set_item(
        "support_fraction",
        v.iter().map(|g| g.support_fraction).collect::<Vec<f64>>(),
    )?;
    verdicts.set_item("p_up", v.iter().map(|g| g.p_up).collect::<Vec<f64>>())?;
    verdicts.set_item("p_down", v.iter().map(|g| g.p_down).collect::<Vec<f64>>())?;
    verdicts.set_item(
        "coefficient",
        v.iter().map(|g| g.coefficient).collect::<Vec<f64>>(),
    )?;

    let out = PyDict::new(py);
    out.set_item("shared_samples", u32_array(py, &res.shared_samples))?;
    out.set_item("kept_features", idx_lists(&res.kept_features)?)?;
    out.set_item("mcp_cell_types", idx_lists(&res.mcp_cell_types)?)?;
    out.set_item("ws", mats(&res.ws)?)?;
    out.set_item("scores", mats(&res.scores)?)?;
    out.set_item("cca_scores", mats(&res.cca_scores)?)?;
    out.set_item("emp_p", mat_f64(py, &res.emp_p)?)?;
    out.set_item("pair_cor", mat_f64(py, &res.pair_cor)?)?;
    out.set_item("refit_fidelity", mat_f64(py, &res.refit_fidelity)?)?;
    out.set_item("verdicts", verdicts)?;
    out.set_item("permissive", signatures(py, &res.permissive)?)?;
    out.set_item("strict", signatures(py, &res.strict)?)?;
    Ok(out)
}

//////////
// Main //
//////////

/// DIALOGUE over a single cell store
///
/// The twin of `rs_dialogue_sc`. Streams the normalised layer off the
/// gene-major file.
///
/// ### Params
///
/// * `f_path_gene` - Path to `counts_genes.bin`
/// * `cell_type_indices` - 0-based store cell indices per cell type
/// * `features` - `(n_cells_in_type, k_i)` `float64` features per cell type,
///   rows aligned to `cell_type_indices`
/// * `sample_ids` - 0-based sample code per store cell, long enough to cover
///   the largest index in use
/// * `cell_quality` - Quality covariate, indexed as `sample_ids`
/// * `gene_indices` - 0-based genes to build signatures from
/// * `pmd` - `to_rust(DialoguePmdParams)`
/// * `hlm` - `to_rust(DialogueHlmParams)`
/// * `refine` - `to_rust(DialogueRefineParams)`
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The result dict, see [result_dict].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn dialogue_sc<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    cell_type_indices: Vec<PyReadonlyArray1<'py, u32>>,
    features: Vec<PyReadonlyArray2<'py, f64>>,
    sample_ids: PyReadonlyArray1<'py, u32>,
    cell_quality: PyReadonlyArray1<'py, f64>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    pmd: &Bound<'py, PyAny>,
    hlm: &Bound<'py, PyAny>,
    refine: &Bound<'py, PyAny>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params = dialogue_params(pmd, hlm, refine)?;
    let cells: Vec<Vec<usize>> = cell_type_indices
        .iter()
        .map(index_vec)
        .collect::<PyResult<_>>()?;
    let views = feature_views(&features)?;
    let samples = index_vec(&sample_ids)?;
    let quality = cell_quality.as_slice()?;
    let genes = index_vec(&gene_indices)?;
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;

    let res = py
        .detach(|| {
            dialogue_run(
                &reader, &cells, &views, &samples, quality, &genes, &params, verbose,
            )
        })
        .map_err(BvErr::from)?;
    result_dict(py, &res)
}

/// DIALOGUE over meta cells in memory
///
/// The twin of `rs_mc_dialogue`. Takes the normalised counts, which become
/// both layers as R hands them over (the raw one truncated to integers).
///
/// ### Params
///
/// * `data` - CSR normalised counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `cell_type_indices` - 0-based meta cell rows per cell type
/// * `features` - `(n_metacells_in_type, k_i)` `float64` features per cell
///   type
/// * `sample_ids` - 0-based sample code per meta cell
/// * `cell_quality` - Quality covariate per meta cell
/// * `gene_indices` - 0-based genes to build signatures from
/// * `pmd` - `to_rust(DialoguePmdParams)`
/// * `hlm` - `to_rust(DialogueHlmParams)`
/// * `refine` - `to_rust(DialogueRefineParams)`
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The result dict, see [result_dict].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mc_dialogue<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    cell_type_indices: Vec<PyReadonlyArray1<'py, u32>>,
    features: Vec<PyReadonlyArray2<'py, f64>>,
    sample_ids: PyReadonlyArray1<'py, u32>,
    cell_quality: PyReadonlyArray1<'py, f64>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    pmd: &Bound<'py, PyAny>,
    hlm: &Bound<'py, PyAny>,
    refine: &Bound<'py, PyAny>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params = dialogue_params(pmd, hlm, refine)?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let cells: Vec<Vec<usize>> = cell_type_indices
        .iter()
        .map(index_vec)
        .collect::<PyResult<_>>()?;
    let views = feature_views(&features)?;
    let samples = index_vec(&sample_ids)?;
    let quality = cell_quality.as_slice()?;
    let genes = index_vec(&gene_indices)?;

    let res = py
        .detach(|| {
            dialogue_metacells(
                &m, &cells, &views, &samples, quality, &genes, &params, verbose,
            )
        })
        .map_err(BvErr::from)?;
    result_dict(py, &res)
}
