//! scSLIDE: the Python twins of `run_scslide`, `landmark_counts` and
//! `sample_diffusion_map` in `bixverse_sc::sc_analysis::scslide`.
//!
//! This only moves buffers. Cell positions are 0-based into `cells_to_keep`,
//! landmarks 0-based into the landmark list.

use bixverse_sc::sc_analysis::scslide::{
    SampleDmParams, ScSlideParams, landmark_counts, run_scslide, sample_diffusion_map,
};
use bixverse_sc_data::store::reader::ParallelSparseReader;
use faer::MatRef;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::{CSlice, flat_array, from_py, index_vec, mat_array, mat_view, u32_array};
use py_common::error::BvErr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/////////////
// Helpers //
/////////////

/// Rows of equal length as a `(n, k)` `uint32` array.
///
/// ### Params
///
/// * `py` - GIL token
/// * `rows` - The rows
/// * `k` - Row length
///
/// ### Returns
///
/// The array.
fn rows_array<'py>(py: Python<'py>, rows: &[Vec<usize>], k: usize) -> PyResult<Bound<'py, PyAny>> {
    let flat: Vec<u32> = rows
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as u32))
        .collect();
    flat_array(py, flat, rows.len(), k)
}

/////////////
// scSLIDE //
/////////////

/// Landmark x donor densities of scSLIDE over a store
///
/// The twin of `run_scslide`.
///
/// ### Params
///
/// * `f_path_cells` - The cell-based binary
/// * `cells_to_keep` - 0-based store cells, in the order of the other inputs
/// * `donors` - 0-based donor per cell
/// * `y_donor` - `(n_donors, q)` `float64` phenotype per donor
/// * `cell_types` - Optional 0-based cell type per cell
/// * `hvg` - 0-based store genes
/// * `pca` - `(n_cells, d)` `float32` embedding
/// * `params` - `to_rust(ScSlideParams)`
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `landmarks`, `counts` `(n_landmarks, n_donors)`,
/// `cell_landmarks` `(n_cells, k_count)`, `pls_embedding`, `donor_folds`,
/// `training_cells` and `pls_genes` (one array per fold).
#[pyfunction]
#[pyo3(signature = (f_path_cells, cells_to_keep, donors, y_donor, cell_types, hvg, pca, params, seed = 42, verbose = true))]
#[allow(clippy::too_many_arguments)]
pub fn scslide<'py>(
    py: Python<'py>,
    f_path_cells: &str,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    donors: PyReadonlyArray1<'py, u32>,
    y_donor: PyReadonlyArray2<'py, f64>,
    cell_types: Option<PyReadonlyArray1<'py, u32>>,
    hvg: PyReadonlyArray1<'py, u32>,
    pca: PyReadonlyArray2<'py, f32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: ScSlideParams = from_py(params, "scSLIDE params")?;
    let cells = index_vec(&cells_to_keep)?;
    let donors = index_vec(&donors)?;
    let types = cell_types.as_ref().map(index_vec).transpose()?;
    let hvg = index_vec(&hvg)?;
    let (n_donors, q) = (y_donor.shape()[0], y_donor.shape()[1]);
    let y = MatRef::from_row_major_slice(y_donor.c_slice()?, n_donors, q);
    let pca = mat_view(&pca)?;
    let reader = ParallelSparseReader::new(f_path_cells).map_err(BvErr::from)?;

    let res = py
        .detach(|| {
            run_scslide(
                &reader,
                &cells,
                &donors,
                y,
                types.as_deref(),
                &hvg,
                pca,
                &params,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;

    let n_landmarks = res.landmarks.len();
    let out = PyDict::new(py);
    out.set_item("landmarks", u32_array(py, &res.landmarks))?;
    out.set_item("counts", flat_array(py, res.counts, n_landmarks, n_donors)?)?;
    out.set_item(
        "cell_landmarks",
        rows_array(py, &res.cell_landmarks, params.k_count)?,
    )?;
    out.set_item("pls_embedding", mat_array(py, &res.pls_embedding)?)?;
    out.set_item("donor_folds", u32_array(py, &res.donor_folds))?;
    out.set_item("training_cells", u32_array(py, &res.training_cells))?;
    let genes = PyList::empty(py);
    for g in &res.pls_genes {
        genes.append(u32_array(py, g))?;
    }
    out.set_item("pls_genes", genes)?;
    Ok(out)
}

/// Landmark x donor counts from two embeddings
///
/// The twin of `landmark_counts`.
///
/// ### Params
///
/// * `emb_one` - `(n_cells, d1)` `float32` first modality
/// * `emb_two` - `(n_cells, d2)` `float32` second modality
/// * `landmarks` - 0-based landmark cells
/// * `donors` - 0-based donor per cell
/// * `n_donors` - Number of donors
/// * `params` - `to_rust(ScSlideParams)`
/// * `seed` - Seed
///
/// ### Returns
///
/// Dict with `counts` `(n_landmarks, n_donors)` and `cell_landmarks`
/// `(n_cells, k_count)`.
#[pyfunction]
#[pyo3(signature = (emb_one, emb_two, landmarks, donors, n_donors, params, seed = 42))]
#[allow(clippy::too_many_arguments)]
pub fn scslide_landmark_counts<'py>(
    py: Python<'py>,
    emb_one: PyReadonlyArray2<'py, f32>,
    emb_two: PyReadonlyArray2<'py, f32>,
    landmarks: PyReadonlyArray1<'py, u32>,
    donors: PyReadonlyArray1<'py, u32>,
    n_donors: usize,
    params: &Bound<'py, PyAny>,
    seed: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: ScSlideParams = from_py(params, "scSLIDE params")?;
    let (one, two) = (mat_view(&emb_one)?, mat_view(&emb_two)?);
    let landmarks = index_vec(&landmarks)?;
    let donors = index_vec(&donors)?;
    let (counts, cell_landmarks) = py
        .detach(|| landmark_counts(one, two, &landmarks, &donors, n_donors, &params, seed))
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("counts", flat_array(py, counts, landmarks.len(), n_donors)?)?;
    out.set_item(
        "cell_landmarks",
        rows_array(py, &cell_landmarks, params.k_count)?,
    )?;
    Ok(out)
}

/// Sample diffusion map from landmark counts
///
/// The twin of `sample_diffusion_map`.
///
/// ### Params
///
/// * `counts` - `(n_landmarks, n_samples)` `float64` counts
/// * `params` - `to_rust(SampleDmParams)`
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `residuals` `(n_landmarks, n_samples)` and `dcs`
/// `(n_samples, n_dcs)`, both `float64`.
#[pyfunction]
#[pyo3(signature = (counts, params, seed = 42, verbose = 0))]
pub fn scslide_diffusion_map<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: SampleDmParams = from_py(params, "sample diffusion map params")?;
    let (n_landmarks, n_samples) = (counts.shape()[0], counts.shape()[1]);
    let counts = counts.c_slice()?;
    let res = py
        .detach(|| sample_diffusion_map(counts, n_landmarks, n_samples, &params, seed, verbose))
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item(
        "residuals",
        flat_array(py, res.residuals, n_landmarks, n_samples)?,
    )?;
    let n_dcs = res.dcs.ncols();
    let dcs: Vec<f64> = (0..n_samples)
        .flat_map(|s| (0..n_dcs).map(move |c| (s, c)))
        .map(|(s, c)| res.dcs[(s, c)])
        .collect();
    out.set_item("dcs", flat_array(py, dcs, n_samples, n_dcs)?)?;
    Ok(out)
}
