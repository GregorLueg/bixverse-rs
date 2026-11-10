//! Plot data extraction, the Python twins of `rs_extract_several_genes_plots`
//! and `rs_extract_grouped_gene_stats`.
//!
//! The numerics live in `bixverse-sc-data::plotting`; this only moves
//! buffers. Every index is 0-based.

use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_data::plotting::{extract_grouped_gene_stats, extract_norm_counts_multi};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use py_common::convert::index_vec;
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Dense normalised counts of several genes over a set of cells
///
/// The twin of `rs_extract_several_genes_plots`. The z-score divides by the
/// sample standard deviation (`n - 1`).
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `cell_indices` - 0-based cells, in output order
/// * `gene_indices` - 0-based genes, in output order
/// * `scale` - Z-score each gene across the cells
/// * `clip` - Clip the z-scores to `[-clip, clip]`
///
/// ### Returns
///
/// `(n_genes, n_cells)` `float32` values.
#[pyfunction]
#[pyo3(signature = (f_path_gene, cell_indices, gene_indices, scale = false, clip = None))]
pub fn extract_genes_dense<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    scale: bool,
    clip: Option<f32>,
) -> PyResult<Bound<'py, PyAny>> {
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let res = py
        .detach(|| {
            let reader = ParallelSparseReader::new(f_path_gene)?;
            extract_norm_counts_multi(&reader, &cells, &genes, scale, clip)
        })
        .map_err(BvErr::from)?;
    let flat: Vec<f32> = res.into_iter().flatten().collect();
    Ok(flat
        .into_pyarray(py)
        .reshape([genes.len(), cells.len()])?
        .into_any())
}

/// Per-group mean normalised expression and fraction expressing
///
/// The twin of `rs_extract_grouped_gene_stats`. Means run over every cell of
/// the group, zeros included.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `cell_indices` - 0-based cells
/// * `gene_indices` - 0-based genes, in output order
/// * `group_ids` - 0-based group of each cell, below `n_groups`
/// * `n_groups` - Number of groups
///
/// ### Returns
///
/// `(mean, fraction)`, each `(n_genes, n_groups)` `float32`.
#[pyfunction]
pub fn grouped_gene_stats<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    group_ids: PyReadonlyArray1<'py, u32>,
    n_groups: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let groups = index_vec(&group_ids)?;
    // the crate asserts both, which would surface as a panic
    if groups.len() != cells.len() {
        return Err(PyValueError::new_err(
            "group_ids must hold one entry per cell",
        ));
    }
    if groups.iter().any(|&g| g >= n_groups) {
        return Err(PyValueError::new_err("group_ids must be below n_groups"));
    }
    let levels = vec![String::new(); n_groups];
    let res = py
        .detach(|| {
            let reader = ParallelSparseReader::new(f_path_gene)?;
            extract_grouped_gene_stats(&reader, &cells, &genes, &groups, &levels)
        })
        .map_err(BvErr::from)?;
    let shape = [genes.len(), n_groups];
    Ok((
        res.mean_expression
            .into_pyarray(py)
            .reshape(shape)?
            .into_any(),
        res.pct_expressed
            .into_pyarray(py)
            .reshape(shape)?
            .into_any(),
    ))
}
