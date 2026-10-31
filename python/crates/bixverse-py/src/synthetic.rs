//! Synthetic single cell data, the Python twin of the extendr generators in
//! `r_synthetic.rs`. The generators live in `bixverse-sc-data`; this only
//! hands their buffers to numpy. Every index comes back 0-based.

use bixverse_sc_data::sc_synthetic_data::{
    CellSweepSyntheticParams, CellTypeConfig, DialogueSyntheticParams, create_adt_synthetic_data,
    create_cellsweep_synthetic_data, create_celltype_sparse_csr_data,
    create_dialogue_synthetic_data, generate_sample_labels, parse_sample_bias,
};
use faer::Mat;
use numpy::{IntoPyArray, PyArrayMethods};
use py_common::convert::u32_array;
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;

/////////////
// Helpers //
/////////////

/// A faer matrix as a row-major numpy array
///
/// ### Params
///
/// * `py` - GIL token
/// * `m` - The matrix
///
/// ### Returns
///
/// The `(nrows, ncols)` `float64` array.
fn mat_array<'py>(py: Python<'py>, m: &Mat<f64>) -> PyResult<Bound<'py, PyAny>> {
    let (r, c) = (m.nrows(), m.ncols());
    let data: Vec<f64> = (0..r)
        .flat_map(|i| (0..c).map(move |j| m[(i, j)]))
        .collect();
    Ok(data.into_pyarray(py).reshape([r, c])?.into_any())
}

/// Deserialise the cell type configs
///
/// ### Params
///
/// * `cell_configs` - A list of `{"marker_genes": [...]}` dicts
///
/// ### Returns
///
/// One [CellTypeConfig] per entry.
fn configs(cell_configs: &Bound<'_, PyAny>) -> PyResult<Vec<CellTypeConfig>> {
    depythonize(cell_configs)
        .map_err(|e| PyValueError::new_err(format!("invalid cell_configs: {e}")))
}

////////////////
// Generators //
////////////////

/// Synthetic RNA counts with planted cell types
///
/// The twin of `rs_synthetic_sc_data_with_cell_types`.
///
/// ### Params
///
/// * `n_cells` - Cells to generate
/// * `n_genes` - Genes to generate
/// * `n_batches` - Batches
/// * `n_samples` - Samples, used together with `sample_bias`
/// * `cell_configs` - One `{"marker_genes": [...]}` per cell type
/// * `batch_effect_strength` - `"strong"`, `"medium"` or `"weak"`
/// * `sample_bias` - `"even"`, `"slightly_uneven"` or `"very_uneven"`
/// * `seed` - Seed
///
/// ### Returns
///
/// `data`, `indices`, `indptr` (CSR over cells, `uint32`), `nrow`, `ncol`,
/// `cell_type_indices`, `batch_indices` and `sample_indices` (`None` unless
/// both sample arguments are given).
#[pyfunction]
#[pyo3(signature = (
    n_cells, n_genes, n_batches, n_samples, cell_configs, batch_effect_strength,
    sample_bias, seed
))]
#[allow(clippy::too_many_arguments)]
pub fn synthetic_sc_data<'py>(
    py: Python<'py>,
    n_cells: usize,
    n_genes: usize,
    n_batches: usize,
    n_samples: Option<usize>,
    cell_configs: &Bound<'py, PyAny>,
    batch_effect_strength: String,
    sample_bias: Option<String>,
    seed: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let configs = configs(cell_configs)?;
    let bias = sample_bias
        .as_deref()
        .map(|b| {
            parse_sample_bias(b)
                .ok_or_else(|| PyValueError::new_err(format!("Invalid sample_bias value '{b}'")))
        })
        .transpose()?;

    let (csr, cell_types, batches, samples) = py.detach(|| {
        let (csr, cell_types, batches) = create_celltype_sparse_csr_data(
            n_cells,
            n_genes,
            configs,
            n_batches,
            &batch_effect_strength,
            seed,
        );
        let samples = match (n_samples, bias) {
            (Some(n), Some(b)) => Some(generate_sample_labels(&cell_types, n, &b, seed)),
            _ => None,
        };
        (csr, cell_types, batches, samples)
    });

    let d = PyDict::new(py);
    d.set_item("data", csr.data.into_pyarray(py))?;
    d.set_item("indices", csr.indices.into_pyarray(py))?;
    d.set_item("indptr", csr.indptr.into_pyarray(py))?;
    d.set_item("nrow", csr.shape.0)?;
    d.set_item("ncol", csr.shape.1)?;
    d.set_item("cell_type_indices", u32_array(py, &cell_types))?;
    d.set_item("batch_indices", u32_array(py, &batches))?;
    d.set_item("sample_indices", samples.map(|s| u32_array(py, &s)))?;
    Ok(d)
}

/// Synthetic ADT counts, cell for cell with [synthetic_sc_data]
///
/// The twin of `rs_synthetic_sc_adt_with_cell_types`.
///
/// ### Params
///
/// * `n_cells` - Cells
/// * `n_proteins` - Proteins
/// * `n_batches` - Batches
/// * `isotype_controls` - 0-based isotype control columns
/// * `cell_configs` - One `{"marker_genes": [...]}` per cell type
/// * `batch_effect_strength` - `"strong"`, `"medium"` or `"weak"`
/// * `seed` - Seed
///
/// ### Returns
///
/// `counts` (`(n_cells, n_proteins)` `uint32`), `cell_type_indices` and
/// `batch_indices`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn synthetic_adt_data<'py>(
    py: Python<'py>,
    n_cells: usize,
    n_proteins: usize,
    n_batches: usize,
    isotype_controls: Vec<usize>,
    cell_configs: &Bound<'py, PyAny>,
    batch_effect_strength: String,
    seed: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let configs = configs(cell_configs)?;
    let (counts, cell_types, batches) = py.detach(|| {
        create_adt_synthetic_data(
            n_cells,
            n_proteins,
            configs,
            isotype_controls,
            n_batches,
            &batch_effect_strength,
            seed,
        )
    });

    let d = PyDict::new(py);
    d.set_item(
        "counts",
        counts.into_pyarray(py).reshape([n_cells, n_proteins])?,
    )?;
    d.set_item("cell_type_indices", u32_array(py, &cell_types))?;
    d.set_item("batch_indices", u32_array(py, &batches))?;
    Ok(d)
}

/// Synthetic data with a planted multicellular programme
///
/// The twin of `rs_synthetic_sc_dialogue_data`.
///
/// ### Params
///
/// * `params` - `to_rust(SyntheticDialogueParams)`
/// * `seed` - Seed
///
/// ### Returns
///
/// `data`, `indices`, `indptr` (CSR over cells), `nrow`, `ncol`,
/// `cell_type_indices` (one array of cells per cell type), `features` (one
/// matrix per cell type), `sample_ids`, `quality`, `latent` and `planted`
/// (one array of genes per cell type).
#[pyfunction]
pub fn synthetic_dialogue_data<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyAny>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let params: DialogueSyntheticParams =
        depythonize(params).map_err(|e| PyValueError::new_err(format!("invalid params: {e}")))?;
    let synthetic = py
        .detach(|| create_dialogue_synthetic_data(&params, seed))
        .map_err(BvErr::from)?;
    // the generator works gene-major; hand back the cell-major view
    let csr = synthetic.matrix.transform();

    let features = synthetic
        .features
        .iter()
        .map(|m| mat_array(py, m))
        .collect::<PyResult<Vec<_>>>()?;
    let cell_types = synthetic
        .cell_type_indices
        .iter()
        .map(|c| u32_array(py, c))
        .collect::<Vec<_>>();
    let planted = synthetic
        .planted
        .iter()
        .map(|g| u32_array(py, g))
        .collect::<Vec<_>>();

    let d = PyDict::new(py);
    d.set_item("data", csr.data.into_pyarray(py))?;
    d.set_item("indices", csr.indices.into_pyarray(py))?;
    d.set_item("indptr", csr.indptr.into_pyarray(py))?;
    d.set_item("nrow", csr.shape.0)?;
    d.set_item("ncol", csr.shape.1)?;
    d.set_item("cell_type_indices", PyList::new(py, cell_types)?)?;
    d.set_item("features", PyList::new(py, features)?)?;
    d.set_item("sample_ids", u32_array(py, &synthetic.sample_ids))?;
    d.set_item("quality", synthetic.quality.into_pyarray(py))?;
    d.set_item("latent", synthetic.latent.into_pyarray(py))?;
    d.set_item("planted", PyList::new(py, planted)?)?;
    Ok(d)
}

/// Synthetic counts with a planted ambient profile
///
/// The twin of `rs_synthetic_sc_cellsweep_data`.
///
/// ### Params
///
/// * `params` - `to_rust(SyntheticCellsweepParams)`
/// * `seed` - Seed
///
/// ### Returns
///
/// `data`, `indices`, `indptr` (CSR over barcodes, real first), `nrow`,
/// `ncol`, `cell_type_indices` (real barcodes only), `is_empty`,
/// `alpha_true`, `ambient_true` and `celltype_profiles_true` (row-major
/// `(n_celltypes, n_genes)`).
#[pyfunction]
pub fn synthetic_cellsweep_data<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyAny>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let params: CellSweepSyntheticParams =
        depythonize(params).map_err(|e| PyValueError::new_err(format!("invalid params: {e}")))?;
    let synthetic = py
        .detach(|| create_cellsweep_synthetic_data(&params, seed))
        .map_err(BvErr::from)?;
    let csr = synthetic.matrix;

    let d = PyDict::new(py);
    d.set_item("data", csr.data.into_pyarray(py))?;
    d.set_item("indices", csr.indices.into_pyarray(py))?;
    d.set_item("indptr", csr.indptr.into_pyarray(py))?;
    d.set_item("nrow", csr.shape.0)?;
    d.set_item("ncol", csr.shape.1)?;
    d.set_item(
        "cell_type_indices",
        u32_array(py, &synthetic.cell_type_indices),
    )?;
    d.set_item("is_empty", synthetic.is_empty.into_pyarray(py))?;
    d.set_item("alpha_true", synthetic.alpha_true.into_pyarray(py))?;
    d.set_item("ambient_true", synthetic.ambient_true.into_pyarray(py))?;
    d.set_item(
        "celltype_profiles_true",
        synthetic
            .celltype_profiles_true
            .into_pyarray(py)
            .reshape([params.n_celltypes, params.n_genes])?,
    )?;
    Ok(d)
}
