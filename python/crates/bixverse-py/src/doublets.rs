//! Doublet detection, empty droplets and CellSweep, the Python twins of
//! `rs_sc_scrublet`, `rs_sc_doublet_detection`, `rs_sc_scdblfinder`,
//! `rs_sc_infer_empty_droplets` and `SingleCellCountData$cellsweep`.
//!
//! The numerics live in `bixverse-sc`; this only moves buffers. Every index is
//! 0-based, cell-type codes included. Scores widen to `float64` on the way
//! out, as R's do.

use bixverse_sc::sc_processing::cellsweep::{
    CellSweepFit, CellSweepParams, CellSweepSample, EmptyDropletCall, infer_empty_droplets,
    run_cellsweep,
};
use bixverse_sc::sc_processing::doublet_detection::{BoostClassifier, BoostParams};
use bixverse_sc::sc_processing::scdblfinder::{ScDblFinder, ScDblFinderParams};
use bixverse_sc::sc_processing::scrublet::{Scrublet, ScrubletParams};
use bixverse_sc_data::data_io::ParallelSparseReader;
use numpy::{IntoPyArray, PyReadonlyArray1};
use py_common::convert::CSlice;
use py_common::convert::{f64_array, u32_array};
use py_common::convert::{flat_array, from_py, index_vec, mat_array};
use py_common::error::BvErr;
use py_common::results::scrublet_dict;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/////////////
// Helpers //
/////////////

/// Open the gene- and cell-based stores
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
///
/// ### Returns
///
/// `(gene_reader, cell_reader)`.
fn readers(
    f_path_gene: &str,
    f_path_cell: &str,
) -> PyResult<(ParallelSparseReader, ParallelSparseReader)> {
    Ok((
        ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?,
        ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?,
    ))
}

///////////////////////
// Doublet detection //
///////////////////////

/// Scrublet over a cell selection
///
/// The twin of `rs_sc_scrublet`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `cells` - 0-based cells
/// * `params` - `to_rust(ScrubletParams)`
/// * `seed` - Seed
/// * `streaming` - Stream the HVG pass
/// * `return_combined_pca` - Also return the PCA of observed and simulated
///   cells
/// * `return_pairs` - Also return the parents of the simulated doublets
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with R's keys; `pca`, `pair_1` and `pair_2` are `None` unless asked
/// for.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cells, params, seed = 42, streaming = false,
    return_combined_pca = false, return_pairs = false, verbose = 1
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
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: ScrubletParams = from_py(params, "scrublet params")?;
    let cells = index_vec(&cells)?;
    let (gene_reader, cell_reader) = readers(f_path_gene, f_path_cell)?;
    let res = py
        .detach(|| {
            Scrublet::new(&gene_reader, &cell_reader, params, &cells).run_scrublet(
                streaming,
                seed,
                verbose,
                return_combined_pca,
                return_pairs,
            )
        })
        .map_err(BvErr::from)?;
    scrublet_dict(py, res)
}

/// Boosted doublet classification over a cell selection
///
/// The twin of `rs_sc_doublet_detection`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `cells` - 0-based cells
/// * `params` - `to_rust(BoostParams)`, `n_centroids` resolved
/// * `seed` - Seed
/// * `streaming` - Stream the HVG pass
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `doublet`, `doublet_score` and `voting_avg`.
#[pyfunction]
#[pyo3(signature = (f_path_gene, f_path_cell, cells, params, seed = 42, streaming = false, verbose = 1))]
#[allow(clippy::too_many_arguments)]
pub fn boost<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cells: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    seed: usize,
    streaming: bool,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: BoostParams = from_py(params, "boost params")?;
    let cells = index_vec(&cells)?;
    let (gene_reader, cell_reader) = readers(f_path_gene, f_path_cell)?;
    let res = py
        .detach(|| {
            BoostClassifier::new(&gene_reader, &cell_reader, params, &cells)
                .run_boost(streaming, seed, verbose)
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("doublet", res.predicted_doublets.into_pyarray(py))?;
    out.set_item("doublet_score", f64_array(py, &res.doublet_scores))?;
    out.set_item("voting_avg", f64_array(py, &res.voting_average))?;
    Ok(out)
}

/// scDblFinder over a cell selection
///
/// The twin of `rs_sc_scdblfinder`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `cells` - 0-based cells
/// * `params` - `to_rust(ScdblfinderParams)`, `n_centroids` resolved
/// * `return_features` - Also return the classifier features of the observed
///   cells
/// * `seed` - Seed
/// * `streaming` - Stream the gene selection pass
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with R's keys; `cluster_labels` and `selected_genes` 0-based,
/// `feature_names` and the `(n_cells, n_features)` `float32`
/// `feature_mat` `None` unless asked for.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cells, params, return_features = false, seed = 42,
    streaming = false, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn scdblfinder<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cells: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    return_features: bool,
    seed: usize,
    streaming: bool,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mut params: ScDblFinderParams = from_py(params, "scdblfinder params")?;
    params.return_features = return_features;
    let cells = index_vec(&cells)?;
    let (gene_reader, cell_reader) = readers(f_path_gene, f_path_cell)?;
    let res = py
        .detach(|| {
            ScDblFinder::new(&gene_reader, &cell_reader, params, &cells)
                .run(seed, streaming, verbose)
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item(
        "predicted_doublets",
        res.predicted_doublets.into_pyarray(py),
    )?;
    out.set_item("doublet_scores", f64_array(py, &res.doublet_scores))?;
    out.set_item("cxds_scores", f64_array(py, &res.cxds))?;
    out.set_item("weighted", f64_array(py, &res.weighted))?;
    out.set_item("threshold", res.threshold as f64)?;
    out.set_item("cluster_labels", u32_array(py, &res.cluster_labels))?;
    out.set_item("detected_doublet_rate", res.detected_doublet_rate as f64)?;
    out.set_item("selected_genes", u32_array(py, &res.selected_genes))?;
    match res.features {
        Some(f) => {
            out.set_item("feature_names", f.feature_names)?;
            out.set_item("feature_mat", mat_array(py, &f.values)?)?;
        }
        None => {
            out.set_item("feature_names", py.None())?;
            out.set_item("feature_mat", py.None())?;
        }
    }
    Ok(out)
}

///////////////
// CellSweep //
///////////////

/// Call empty droplets from the library sizes
///
/// The twin of `rs_sc_infer_empty_droplets`.
///
/// ### Params
///
/// * `lib_size` - Library size per barcode, in store order
/// * `params` - `to_rust(EmptyDropletsParams)`; `"supplied"` is rejected,
///   there is nothing to infer
///
/// ### Returns
///
/// A `bool` array, `True` where the barcode is empty.
#[pyfunction]
pub fn empty_droplets<'py>(
    py: Python<'py>,
    lib_size: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let call: EmptyDropletCall = from_py(params, "empty droplet params")?;
    let mask = infer_empty_droplets(lib_size.c_slice()?, call).map_err(BvErr::from)?;
    Ok(mask.into_pyarray(py).into_any())
}

/// One sample's fit as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `fit` - The fit
/// * `sample_id` - The sample
///
/// ### Returns
///
/// Dict with `sample_id`, `alpha`, `z_hat` (0-based), `beta`, `ambient`,
/// `celltype_profiles` as `(n_celltypes, n_genes)` `float64`,
/// `log_likelihood`, `n_iter` and `converged`.
fn fit_dict<'py>(
    py: Python<'py>,
    fit: &CellSweepFit,
    sample_id: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let n_genes = fit.ambient.len();
    let n_celltypes = fit.celltype_profiles.len() / n_genes.max(1);
    let profiles: Vec<f64> = fit.celltype_profiles.iter().map(|&v| v as f64).collect();

    let out = PyDict::new(py);
    out.set_item("sample_id", sample_id)?;
    out.set_item("alpha", fit.alpha.clone().into_pyarray(py))?;
    out.set_item("z_hat", u32_array(py, &fit.z_hat))?;
    out.set_item("beta", fit.beta)?;
    out.set_item("ambient", f64_array(py, &fit.ambient))?;
    out.set_item(
        "celltype_profiles",
        flat_array(py, profiles, n_celltypes, n_genes)?,
    )?;
    out.set_item("log_likelihood", fit.log_likelihood)?;
    out.set_item("n_iter", fit.n_iter)?;
    out.set_item("converged", fit.converged)?;
    Ok(out)
}

/// Fit CellSweep per sample and write the denoised cell-based store
///
/// The twin of `SingleCellCountData$cellsweep`. Writes the cell-based file
/// only; the gene-based companion is built afterwards.
///
/// ### Params
///
/// * `f_path_source` - The raw cell-based binary, empty droplets included
/// * `f_path_target` - The cell-based binary to write
/// * `samples` - Per sample a dict of `sample_id`, `real_cells`,
///   `empty_cells`, `celltype_idx` (0-based lists) and `n_celltypes`
/// * `params` - `to_rust(CellsweepParams)`
/// * `target_size` - Library size the normalised layer is scaled to
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `cell_order` (0-based source indices in output order),
/// `lib_size`, `nnz` and `fits`, one dict per sample.
#[pyfunction]
#[pyo3(signature = (f_path_source, f_path_target, samples, params, target_size, verbose = 1))]
pub fn cellsweep<'py>(
    py: Python<'py>,
    f_path_source: &str,
    f_path_target: &str,
    samples: &Bound<'py, PyAny>,
    params: &Bound<'py, PyAny>,
    target_size: f32,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let samples: Vec<CellSweepSample> = from_py(samples, "cellsweep samples")?;
    let params: CellSweepParams = from_py(params, "cellsweep params")?;
    let reader = ParallelSparseReader::new(f_path_source).map_err(BvErr::from)?;
    let res = py
        .detach(|| {
            run_cellsweep(
                &reader,
                &samples,
                params,
                f_path_target,
                target_size,
                verbose,
            )
        })
        .map_err(BvErr::from)?;

    let fits = PyList::empty(py);
    for (fit, sample) in res.fits.iter().zip(&samples) {
        fits.append(fit_dict(py, fit, &sample.sample_id)?)?;
    }
    let out = PyDict::new(py);
    out.set_item("cell_order", u32_array(py, &res.cell_order))?;
    out.set_item("lib_size", u32_array(py, &res.library_size))?;
    out.set_item("nnz", u32_array(py, &res.nnz))?;
    out.set_item("fits", fits)?;
    Ok(out)
}
