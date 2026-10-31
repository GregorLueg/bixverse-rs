//! Multi-modal single cell: the Python twins of `rs_adt_clr`, `rs_dsb`,
//! `rs_prcomp` / `rs_random_svd` as `calculate_pca_adt_sc` calls them, and
//! `rs_wnn`.
//!
//! ADT matrices are small and dense: cells x proteins, row-major `float64` in
//! and out. Every index is 0-based.

use bixverse_core::core::math::matrix_helpers::scale_matrix_col;
use bixverse_core::core::math::pca_svd::randomised_svd;
use bixverse_sc::multi_modal::adt::clr::adt_clr as clr;
use bixverse_sc::multi_modal::adt::dsb::{
    DsbParamsSpec, dsb_normalise, parse_scale_factor, prepare_background_source,
};
use bixverse_sc::multi_modal::wnn::{WnnParams, run_wnn};
use bixverse_sc::prelude::{CoreErrors, ScErrors};
use faer::{Mat, MatRef};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::{f64_array, flat_array, from_py, mat_f64, mat_view};
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/////////////
// Helpers //
/////////////

/// A row-major numpy `float64` matrix as a faer view, without copying
///
/// ### Params
///
/// * `m` - The C-contiguous `(n, d)` matrix
///
/// ### Returns
///
/// The view.
fn mat_view_f64<'a>(m: &'a PyReadonlyArray2<'_, f64>) -> PyResult<MatRef<'a, f64>> {
    let (n, d) = (m.shape()[0], m.shape()[1]);
    Ok(MatRef::from_row_major_slice(m.as_slice()?, n, d))
}

/// A `float64` matrix narrowed to `float32`, as R's `r_matrix_to_faer_fp32`
///
/// ### Params
///
/// * `m` - The matrix
///
/// ### Returns
///
/// The `f32` copy.
fn to_f32(m: MatRef<f64>) -> Mat<f32> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] as f32)
}

/////////
// ADT //
/////////

/// CLR-normalise ADT counts per cell
///
/// The twin of `rs_adt_clr`.
///
/// ### Params
///
/// * `counts` - `(n_cells, n_proteins)` raw counts
/// * `seurat_clr` - Seurat-style `log1p(x / g)` instead of the proper CLR
///
/// ### Returns
///
/// `(n_cells, n_proteins)` `float64` CLR values.
#[pyfunction]
pub fn adt_clr<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    seurat_clr: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let view = mat_view_f64(&counts)?;
    let res = py.detach(|| clr(view, seurat_clr));
    mat_f64(py, &res)
}

/// DSB-normalise ADT counts
///
/// The twin of `rs_dsb`. With `background` the per-protein ambient level comes
/// from the empty droplets, without it from a two-component k-means on the
/// log counts. The counts are narrowed to `float32`, as in R.
///
/// ### Params
///
/// * `raw` - `(n_cells, n_proteins)` raw counts
/// * `background` - Optional `(n_empty, n_proteins)` empty droplet counts
/// * `isotype_indices` - 0-based columns of the isotype controls
/// * `params` - `to_rust(DsbParams)`
/// * `scale_factor` - `"standardise"` or `"mean_subtract"`; only used with
///   `background`
/// * `seed` - Seed for the k-means initialisations
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A dict with `norm_counts` and the `float64` vectors
/// `protein_background_mean`, `protein_background_sd`,
/// `technical_component` and `cellwise_background_mean` (empty when not
/// computed).
#[pyfunction]
#[pyo3(signature = (raw, background, isotype_indices, params, scale_factor, seed, verbose))]
#[allow(clippy::too_many_arguments)]
pub fn dsb<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'py, f64>,
    background: Option<PyReadonlyArray2<'py, f64>>,
    isotype_indices: Vec<usize>,
    params: &Bound<'py, PyAny>,
    scale_factor: &str,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if parse_scale_factor(scale_factor).is_none() {
        return Err(PyValueError::new_err(format!(
            "scale_factor must be 'standardise' or 'mean_subtract', got '{scale_factor}'"
        )));
    }
    let spec: DsbParamsSpec = from_py(params, "DSB params")?;
    let params = spec.into_params(isotype_indices).map_err(BvErr::from)?;
    let raw = to_f32(mat_view_f64(&raw)?);
    let bg = match &background {
        Some(b) => Some(to_f32(mat_view_f64(b)?)),
        None => None,
    };
    let scale_factor = scale_factor.to_string();

    let res = py
        .detach(|| {
            let source = prepare_background_source(bg.as_ref().map(|m| m.as_ref()), scale_factor);
            dsb_normalise(raw.as_ref(), source, &params, verbose, seed)
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("norm_counts", mat_f64(py, &res.normalised)?)?;
    out.set_item("protein_background_mean", res.protein_background_mean)?;
    out.set_item(
        "protein_background_sd",
        res.protein_background_sd.unwrap_or_default(),
    )?;
    out.set_item(
        "technical_component",
        res.technical_component.unwrap_or_default(),
    )?;
    out.set_item(
        "cellwise_background_mean",
        res.cellwise_background_mean.unwrap_or_default(),
    )?;
    Ok(out)
}

/// PCA over the normalised ADT counts, columns centred and scaled
///
/// What `calculate_pca_adt_sc` runs: `rs_prcomp` (thin SVD, singular values
/// divided by `sqrt(n - 1)`) or `rs_random_svd` (raw singular values).
///
/// ### Params
///
/// * `x` - `(n_cells, n_proteins)` normalised counts
/// * `no_pcs` - Number of components, at most `n_proteins`
/// * `randomised` - Randomised SVD instead of the full thin SVD
/// * `seed` - Seed for the randomised SVD
///
/// ### Returns
///
/// `(scores, loadings, singular_values)`: `(n_cells, no_pcs)`,
/// `(n_proteins, no_pcs)` and `no_pcs` values, all `float64`.
#[pyfunction]
pub fn adt_pca<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    no_pcs: usize,
    randomised: bool,
    seed: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Vec<f64>)> {
    let view = mat_view_f64(&x)?;
    if no_pcs == 0 || no_pcs > view.ncols().min(view.nrows()) {
        return Err(PyValueError::new_err(format!(
            "no_pcs must be in [1, {}]",
            view.ncols().min(view.nrows())
        )));
    }
    let (scores, loadings, s) = py
        .detach(|| {
            let scaled = scale_matrix_col(&view, true);
            if randomised {
                let res = randomised_svd(scaled.as_ref(), no_pcs, seed, None, None)?;
                let scores = Mat::from_fn(res.u.nrows(), no_pcs, |i, j| res.u[(i, j)] * res.s[j]);
                let s = res.s.into_iter().take(no_pcs).collect();
                Ok::<_, CoreErrors>((scores, res.v, s))
            } else {
                let svd = scaled
                    .thin_svd()
                    .map_err(|e| CoreErrors::FaerSvdError(format!("{e:?}")))?;
                let scores = &scaled * svd.V();
                let denom = ((view.nrows() as f64) - 1.0).sqrt();
                let s: Vec<f64> = svd
                    .S()
                    .column_vector()
                    .iter()
                    .take(no_pcs)
                    .map(|x| x / denom)
                    .collect();
                Ok((
                    scores.subcols(0, no_pcs).to_owned(),
                    svd.V().subcols(0, no_pcs).to_owned(),
                    s,
                ))
            }
        })
        .map_err(|e| BvErr::from(ScErrors::from(e)))?;
    Ok((mat_f64(py, &scores)?, mat_f64(py, &loadings)?, s))
}

/////////
// WNN //
/////////

/// `(indices, distances, weights_one, weights_two)` of [wnn].
type WnnOut<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

/// Weighted nearest neighbours over two embeddings
///
/// The twin of `rs_wnn`.
///
/// ### Params
///
/// * `emb_one` - `(n_cells, d1)` `float32` embedding of the first modality
/// * `emb_two` - `(n_cells, d2)` `float32` embedding of the second, same
///   cells in the same order
/// * `params` - `to_rust(WnnParams)`
/// * `seed` - Seed for the kNN index builds
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `(indices, distances, weights_one, weights_two)`: `(n_cells, k_nn)`
/// `uint32` and `float32`, then two `float64` vectors.
#[pyfunction]
pub fn wnn<'py>(
    py: Python<'py>,
    emb_one: PyReadonlyArray2<'py, f32>,
    emb_two: PyReadonlyArray2<'py, f32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<WnnOut<'py>> {
    let params: WnnParams = from_py(params, "WNN params")?;
    let (one, two) = (mat_view(&emb_one)?, mat_view(&emb_two)?);
    if one.nrows() != two.nrows() {
        return Err(PyValueError::new_err(
            "both embeddings must have one row per cell",
        ));
    }
    let res = py
        .detach(|| run_wnn(one, two, &params, seed, verbose))
        .map_err(BvErr::from)?;

    let n = res.wnn_indices.len();
    let k = params.k_nn;
    let idx: Vec<u32> = res
        .wnn_indices
        .iter()
        .flat_map(|row| row.iter().map(|&j| j as u32))
        .collect();
    let dist: Vec<f32> = res.wnn_distances.into_iter().flatten().collect();
    if idx.len() != n * k {
        return Err(PyValueError::new_err(
            "the WNN graph came back with rows shorter than k_nn",
        ));
    }
    Ok((
        flat_array(py, idx, n, k)?,
        flat_array(py, dist, n, k)?,
        f64_array(py, &res.modality_weights[0]),
        f64_array(py, &res.modality_weights[1]),
    ))
}
