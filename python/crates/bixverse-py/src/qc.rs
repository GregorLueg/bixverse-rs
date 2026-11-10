//! Per-cell QC metrics and MAD outlier calls, the Python twins of
//! `rs_mad_outlier`, `rs_sc_get_top_genes_perc` and `rs_sc_get_gene_set_perc`.
//! The numerics live in `bixverse-core` and `bixverse-sc`; this only moves
//! buffers. Every index is 0-based.

use bixverse_core::core::math::stats::{mad_outlier as mad_outlier_core, parse_outlier_type};
use bixverse_sc::sc_processing::qc::{
    get_gene_set_perc, get_gene_set_perc_streaming, get_top_genes_perc,
    get_top_genes_perc_streaming,
};
use bixverse_sc_data::data_io::ParallelSparseReader;
use numpy::{IntoPyArray, PyReadonlyArray1};
use py_common::convert::CSlice;
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/////////////
// Helpers //
/////////////

/// `uint32` cell indices as `usize`
///
/// ### Params
///
/// * `cells` - 0-based cell indices
///
/// ### Returns
///
/// The indices.
fn cell_vec(cells: &PyReadonlyArray1<'_, u32>) -> PyResult<Vec<usize>> {
    Ok(cells.c_slice()?.iter().map(|&x| x as usize).collect())
}

/// Per-cell metric vectors as a list of `float64` arrays
///
/// The crates compute in `f32`; R widens to double on the way out, so the
/// obs columns are DOUBLE on both sides.
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - One vector per metric
///
/// ### Returns
///
/// The list.
fn f64_list<'py>(py: Python<'py>, res: Vec<Vec<f32>>) -> PyResult<Bound<'py, PyList>> {
    let arrays = res.into_iter().map(|v| {
        v.into_iter()
            .map(|x| x as f64)
            .collect::<Vec<f64>>()
            .into_pyarray(py)
    });
    PyList::new(py, arrays)
}

/////////
// MAD //
/////////

/// MAD outlier calls on one metric
///
/// The twin of `rs_mad_outlier`, except an unknown direction raises instead of
/// falling back to two-sided.
///
/// ### Params
///
/// * `x` - The metric
/// * `threshold` - MADs from the median that are acceptable
/// * `direction` - `"twosided"`, `"below"` or `"above"`
/// * `r_standard` - Scale the MAD by 1.4826 like R's `mad()`
///
/// ### Returns
///
/// `(outlier, margin)`: a `bool` array and the applied `threshold * MAD`.
#[pyfunction]
pub fn mad_outlier<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    threshold: f64,
    direction: &str,
    r_standard: bool,
) -> PyResult<(Bound<'py, PyAny>, f64)> {
    let dir = parse_outlier_type(direction)
        .ok_or_else(|| PyValueError::new_err(format!("Invalid direction '{direction}'")))?;
    let (outlier, margin) = mad_outlier_core(x.c_slice()?, threshold, dir, r_standard);
    Ok((outlier.into_pyarray(py).into_any(), margin))
}

/////////////
// Metrics //
/////////////

/// Share of each cell's library taken by its top N genes
///
/// The twin of `rs_sc_get_top_genes_perc`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `top_n_vals` - The N values
/// * `cell_indices` - 0-based cells, in the order the output takes
/// * `streaming` - Read the cells in batches
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// One `float64` array per N value, one entry per cell.
#[pyfunction]
#[pyo3(signature = (f_path_cell, top_n_vals, cell_indices, streaming, verbose = 1))]
pub fn top_genes_perc<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    top_n_vals: Vec<usize>,
    cell_indices: PyReadonlyArray1<'py, u32>,
    streaming: bool,
    verbose: usize,
) -> PyResult<Bound<'py, PyList>> {
    let cells = cell_vec(&cell_indices)?;
    let res = py
        .detach(|| {
            let reader = ParallelSparseReader::new(f_path_cell)?;
            if streaming {
                get_top_genes_perc_streaming(&reader, &top_n_vals, &cells, verbose)
            } else {
                get_top_genes_perc(&reader, &top_n_vals, &cells, verbose)
            }
        })
        .map_err(BvErr::from)?;
    f64_list(py, res)
}

/// Share of each cell's library taken by each of several gene sets
///
/// The twin of `rs_sc_get_gene_set_perc`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `gene_sets` - One array of 0-based gene indices per set
/// * `cell_indices` - 0-based cells, in the order the output takes
/// * `streaming` - Read the cells in batches
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// One `float64` array per gene set, one entry per cell.
#[pyfunction]
#[pyo3(signature = (f_path_cell, gene_sets, cell_indices, streaming, verbose = 1))]
pub fn gene_set_perc<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    gene_sets: Vec<PyReadonlyArray1<'py, u32>>,
    cell_indices: PyReadonlyArray1<'py, u32>,
    streaming: bool,
    verbose: usize,
) -> PyResult<Bound<'py, PyList>> {
    let cells = cell_vec(&cell_indices)?;
    let sets = gene_sets
        .iter()
        .map(|g| Ok(g.c_slice()?.to_vec()))
        .collect::<PyResult<Vec<Vec<u32>>>>()?;
    let res = py
        .detach(|| {
            let reader = ParallelSparseReader::new(f_path_cell)?;
            if streaming {
                get_gene_set_perc_streaming(&reader, sets, &cells, verbose)
            } else {
                get_gene_set_perc(&reader, sets, &cells, verbose)
            }
        })
        .map_err(BvErr::from)?;
    f64_list(py, res)
}
