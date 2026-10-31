//! Bonsai trees: the Python twins of `rs_sc_bonsai`, `rs_mc_bonsai` and
//! `rs_bonsai_layout`.
//!
//! Sanity, the tree search and the layout live in
//! `bixverse_sc::sc_analysis::bonsai` and `bixverse_sc::mc_analysis::bonsai_mc`;
//! this only moves buffers. Every cell, gene and node index is 0-based, and
//! the root's parent is `-1`.

use bixverse_sc::mc_analysis::bonsai_mc::sanity_bonsai_mc;
use bixverse_sc::prelude::{ParallelSparseReader, verbosity_from_flag};
use bixverse_sc::sc_analysis::bonsai::{
    BonsaiScParams, NO_NODE, bonsai_layout as layout_rs, parse_bonsai_layout, sanity_bonsai_sc,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use py_common::convert::{csr_from_parts, from_py, index_vec};
use py_common::error::BvErr;
use py_common::results::{bonsai_dict, parents_out};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/////////////
// Helpers //
/////////////

////////////
// Bonsai //
////////////

/// Bonsai over the counts on disk
///
/// The twin of `rs_sc_bonsai`: Sanity on the CPU over the given genes and
/// cells, then the tree search and the layout.
///
/// ### Params
///
/// * `f_path_gene` - The gene-major store
/// * `f_path_cell` - The cell-major store, for the library sizes
/// * `cell_indices` - 0-based store cells, the leaf order
/// * `gene_indices` - 0-based candidate genes
/// * `params` - `to_rust(BonsaiParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See `bonsai_dict`.
#[pyfunction]
#[pyo3(signature = (f_path_gene, f_path_cell, cell_indices, gene_indices, params, verbose = false))]
pub fn sc_bonsai<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: BonsaiScParams = from_py(params, "Bonsai params")?;
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let res = py
        .detach(|| {
            let gene_reader = ParallelSparseReader::new(f_path_gene)?;
            let cell_reader = ParallelSparseReader::new(f_path_cell)?;
            sanity_bonsai_sc(
                &gene_reader,
                &cell_reader,
                &cells,
                &genes,
                &params,
                verbosity_from_flag(verbose),
            )
        })
        .map_err(BvErr::from)?;
    bonsai_dict(py, res)
}

/// Bonsai over meta cell counts
///
/// The twin of `rs_mc_bonsai`. Each meta cell is a leaf, its total counts
/// over every gene its library size.
///
/// ### Params
///
/// * `data` - `float32` summed raw counts of the CSR
/// * `indices` - `uint32` gene indices
/// * `indptr` - `uint32` row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `gene_indices` - 0-based candidate genes
/// * `params` - `to_rust(BonsaiParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See `bonsai_dict`, the meta cells as the leaves in their row order.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, gene_indices, params, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn mc_bonsai<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    gene_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: BonsaiScParams = from_py(params, "Bonsai params")?;
    let genes = index_vec(&gene_indices)?;
    let counts = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let res = py
        .detach(|| sanity_bonsai_mc(&counts, &genes, &params, verbosity_from_flag(verbose)))
        .map_err(BvErr::from)?;
    bonsai_dict(py, res)
}

/// Lay out a finished Bonsai tree again
///
/// The twin of `rs_bonsai_layout`. The tree is renumbered, so the ancestors
/// can come back with other indices; the leaves keep theirs.
///
/// ### Params
///
/// * `parent` - `int64` 0-based parents, negative for the root
/// * `branch` - `float64` branch length above each node
/// * `n_leaves` - Leaves, the first `n_leaves` nodes
/// * `layout` - `"equal_angle"`, `"equal_daylight"` or `"dendrogram"`
/// * `hyperbolic` - Project onto the hyperbolic disk
///
/// ### Returns
///
/// Dict with `parent` (`int64`, `-1` for the root), `branch`, `x` and `y`,
/// indexed by the tree's own numbering.
#[pyfunction]
pub fn bonsai_layout<'py>(
    py: Python<'py>,
    parent: PyReadonlyArray1<'py, i64>,
    branch: PyReadonlyArray1<'py, f64>,
    n_leaves: usize,
    layout: &str,
    hyperbolic: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let layout = parse_bonsai_layout(layout)
        .ok_or_else(|| PyValueError::new_err(format!("Invalid Bonsai layout: {layout}")))?;
    let parent: Vec<u32> = parent
        .as_slice()?
        .iter()
        .map(|&p| if p < 0 { NO_NODE } else { p as u32 })
        .collect();
    let (parent, branch, coords) = layout_rs(
        parent,
        branch.as_slice()?.to_vec(),
        n_leaves,
        layout,
        hyperbolic,
    )
    .map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("parent", parents_out(&parent).into_pyarray(py))?;
    out.set_item("branch", branch.into_pyarray(py))?;
    out.set_item("x", coords.x.into_pyarray(py))?;
    out.set_item("y", coords.y.into_pyarray(py))?;
    Ok(out)
}
