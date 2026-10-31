//! Trajectories: the Python twins of `rs_palantir`, `rs_paga` and
//! `rs_gene_trends`.
//!
//! Palantir and PAGA live in `bixverse_sc::sc_trajectory` and take a kNN graph
//! and nothing else; the gene trends take an expression matrix and a finished
//! Palantir run. This only moves buffers. Every cell index is 0-based, into
//! the rows of the kNN graph.

use bixverse_sc::prelude::CompressedSparseData2;
use bixverse_sc::sc_trajectory::gene_trends::{
    BranchSelectionParams, GeneTrendsParams, compute_gene_trends, select_branch_cells,
};
use bixverse_sc::sc_trajectory::paga::{PagaResult, run_paga};
use bixverse_sc::sc_trajectory::palantir::{PalantirParams, run_palantir};
use faer::MatRef;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::u32_array;
use py_common::convert::{dist_rows, flat_array, from_py, index_vec, knn_rows, mat_array};
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/////////////
// Helpers //
/////////////

/// A small symmetric compressed sparse matrix as a dense row-major array
///
/// The PAGA graphs are clusters by clusters, so dense costs nothing and keeps
/// the `float64` values R sees.
///
/// ### Params
///
/// * `py` - GIL token
/// * `m` - The matrix, CSR or CSC; symmetric, so either reads the same
///
/// ### Returns
///
/// The `(n_rows, n_cols)` `float64` array.
fn dense_f64<'py>(py: Python<'py>, m: &CompressedSparseData2<f64>) -> PyResult<Bound<'py, PyAny>> {
    let (r, c) = m.shape;
    let mut out = vec![0.0_f64; r * c];
    for (i, w) in m.indptr.windows(2).enumerate() {
        for p in w[0] as usize..w[1] as usize {
            out[i * c + m.indices[p] as usize] = m.data[p];
        }
    }
    flat_array(py, out, r, c)
}

//////////////
// Palantir //
//////////////

/// Palantir over a kNN graph
///
/// The twin of `rs_palantir`. The graph feeds the diffusion kernel; the
/// geodesics are measured over a second kNN graph Palantir builds on the
/// multiscale space.
///
/// ### Params
///
/// * `knn_indices` - `(n_cells, k)` `uint32` kNN without self
/// * `knn_dist` - The matching `float32` distances
/// * `early_cell` - 0-based row of the early cell
/// * `terminal_states` - Optional 0-based rows of the terminal states; `None`
///   detects them
/// * `params` - `to_rust(PalantirParams)`
/// * `seed` - Seed for the eigensolver, the waypoint sampler and the
///   multiscale kNN
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict with `pseudotime`, `entropy` (`float32` per cell), `branch_probs`
/// and `multiscale` (`float32` cells by states / components),
/// `terminal_states` and `waypoints` (`uint32`), `start_cell`, `iterations`,
/// `converged`, `eigen_converged`, `eigen_residual`, `repair_edges` and
/// `stranded_waypoints`.
#[pyfunction]
#[pyo3(signature = (
    knn_indices, knn_dist, early_cell, terminal_states, params, seed = 42, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn palantir<'py>(
    py: Python<'py>,
    knn_indices: PyReadonlyArray2<'py, u32>,
    knn_dist: PyReadonlyArray2<'py, f32>,
    early_cell: usize,
    terminal_states: Option<PyReadonlyArray1<'py, u32>>,
    params: &Bound<'py, PyAny>,
    seed: u64,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: PalantirParams = from_py(params, "Palantir params")?;
    let knn = knn_rows(&knn_indices)?;
    let dist = dist_rows(&knn_dist, knn_indices.shape())?;
    let terminal = terminal_states.as_ref().map(index_vec).transpose()?;
    let res = py
        .detach(|| {
            run_palantir(
                &knn,
                &dist,
                early_cell,
                terminal.as_deref(),
                params,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("pseudotime", res.pseudotime.into_pyarray(py))?;
    out.set_item("entropy", res.entropy.into_pyarray(py))?;
    out.set_item("branch_probs", mat_array(py, &res.branch_probs)?)?;
    out.set_item("terminal_states", u32_array(py, &res.terminal_states))?;
    out.set_item("waypoints", u32_array(py, &res.waypoints))?;
    out.set_item("start_cell", res.start_cell)?;
    out.set_item("multiscale", mat_array(py, &res.multiscale)?)?;
    out.set_item("iterations", res.iterations)?;
    out.set_item("converged", res.converged)?;
    out.set_item("eigen_converged", res.eigen_converged)?;
    out.set_item("eigen_residual", res.eigen_residual)?;
    out.set_item("repair_edges", res.repair_edges)?;
    out.set_item("stranded_waypoints", res.stranded_waypoints)?;
    Ok(out)
}

//////////
// PAGA //
//////////

/// PAGA over a kNN graph and a clustering
///
/// The twin of `rs_paga`. Only the kNN indices are read.
///
/// ### Params
///
/// * `knn_indices` - `(n_cells, k)` `uint32` kNN without self
/// * `partitions` - 0-based cluster per cell
/// * `n_partitions` - Declared cluster count, so empty levels are kept
///
/// ### Returns
///
/// Dict with `connectivities` and `connectivities_tree` (dense `float64`
/// clusters by clusters) and `sizes` (`uint32` cells per cluster).
#[pyfunction]
pub fn paga<'py>(
    py: Python<'py>,
    knn_indices: PyReadonlyArray2<'py, u32>,
    partitions: PyReadonlyArray1<'py, u32>,
    n_partitions: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let knn = knn_rows(&knn_indices)?;
    let partitions = index_vec(&partitions)?;
    let res: PagaResult<f64> = py
        .detach(|| run_paga(&knn, &partitions, Some(n_partitions)))
        .map_err(BvErr::from)?;

    let out = PyDict::new(py);
    out.set_item("connectivities", dense_f64(py, &res.connectivities)?)?;
    out.set_item(
        "connectivities_tree",
        dense_f64(py, &res.connectivities_tree)?,
    )?;
    out.set_item("sizes", u32_array(py, &res.sizes))?;
    Ok(out)
}

/////////////////
// Gene trends //
/////////////////

/// Gene trends over a Palantir pseudotime
///
/// The twin of `rs_gene_trends`: selects each branch's cells from the fate
/// probabilities, then fits a landmark Matern-5/2 Gaussian process per
/// branch.
///
/// ### Params
///
/// * `expression` - `(n_cells, n_genes)` `float64`, every cell, rows aligned
///   with `pseudotime`
/// * `pseudotime` - `float32` per cell
/// * `branch_probs` - `(n_cells, n_branches)` `float32` fate probabilities
/// * `branch_params` - `to_rust(BranchSelectionParams)`
/// * `gene_trend_params` - `to_rust(GeneTrendsParams)`
///
/// ### Returns
///
/// Dict with `trends` (one `(resolution, n_genes)` `float64` array per
/// branch), `grids` (`float64` per branch), `branch_cells` (`uint32` rows per
/// branch), `n_cells` and `jitter_used`.
#[pyfunction]
pub fn gene_trends<'py>(
    py: Python<'py>,
    expression: PyReadonlyArray2<'py, f64>,
    pseudotime: PyReadonlyArray1<'py, f32>,
    branch_probs: PyReadonlyArray2<'py, f32>,
    branch_params: &Bound<'py, PyAny>,
    gene_trend_params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let selection: BranchSelectionParams = from_py(branch_params, "branch selection params")?;
    let params: GeneTrendsParams = from_py(gene_trend_params, "gene trend params")?;
    let (n, g) = (expression.shape()[0], expression.shape()[1]);
    let (bn, nb) = (branch_probs.shape()[0], branch_probs.shape()[1]);
    if bn != n {
        return Err(PyValueError::new_err(format!(
            "branch_probs has {bn} rows, the expression {n}"
        )));
    }
    let expr = MatRef::from_row_major_slice(expression.as_slice()?, n, g);
    let probs = MatRef::from_row_major_slice(branch_probs.as_slice()?, bn, nb);
    let pt = pseudotime.as_slice()?;

    let (branch_cells, res) = py
        .detach(|| {
            let cells = select_branch_cells(pt, probs, &selection)?;
            let res = compute_gene_trends(expr, pt, &cells, Some(probs), &params)?;
            Ok::<_, bixverse_sc::prelude::ScErrors>((cells, res))
        })
        .map_err(BvErr::from)?;

    let trends = PyList::empty(py);
    for m in &res.trends {
        let (r, c) = (m.nrows(), m.ncols());
        let data: Vec<f64> = (0..r)
            .flat_map(|i| (0..c).map(move |j| m[(i, j)]))
            .collect();
        trends.append(flat_array(py, data, r, c)?)?;
    }
    let grids = PyList::empty(py);
    for grid in res.grids {
        grids.append(grid.into_pyarray(py))?;
    }
    let cells = PyList::empty(py);
    for c in &branch_cells {
        cells.append(u32_array(py, c))?;
    }

    let out = PyDict::new(py);
    out.set_item("trends", trends)?;
    out.set_item("grids", grids)?;
    out.set_item("branch_cells", cells)?;
    out.set_item("n_cells", res.n_cells)?;
    out.set_item("jitter_used", res.jitter_used)?;
    Ok(out)
}
