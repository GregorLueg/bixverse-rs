//! kNN searches, neighbour graphs and clustering, the Python twins of
//! `rs_sc_knn_w_dist`, `rs_sc_knn_graph`, `rs_graph_leiden`,
//! `rs_fast_cluster_sc` and `rs_compare_knn`.
//!
//! R keeps the neighbour graph behind an external pointer; Python keeps its
//! symmetric CSR and hands the three vectors back for clustering. Every index
//! is 0-based.

use bixverse_core::core::math::sparse::CompressedSparseData2;
use bixverse_methods::graph::graph_structures::SparseGraph;
use bixverse_methods::graph::knn_graph::{KnnSymmetrisation, build_knn_graph};
use bixverse_methods::graph::leiden::{LeidenParams, leiden_sparse_graph};
use bixverse_sc::prelude::ScErrors;
use bixverse_sc::sc_analysis::fast_clusters::{
    CpuKMeans, FastClusterParams, GridStats, fast_clusters,
};
use bixverse_sc::sc_processing::knn::{
    KnnParams, compare_knn, generate_knn_with_dist, pad_knn_rows,
};
use faer::MatRef;
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::depythonize;
use serde::de::DeserializeOwned;

use crate::error::BvErr;

/////////////
// Helpers //
/////////////

/// Deserialise a parameter dict into a crate struct
///
/// ### Params
///
/// * `obj` - The dict, as `to_rust(params)` builds it
/// * `what` - What it is, for the error message
///
/// ### Returns
///
/// The struct, or a `ValueError` naming the offending key.
fn from_py<T: DeserializeOwned>(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<T> {
    depythonize(obj).map_err(|e| PyValueError::new_err(format!("invalid {what}: {e}")))
}

/// A row-major numpy matrix as a faer view, without copying
///
/// ### Params
///
/// * `m` - The C-contiguous `(n, d)` matrix
///
/// ### Returns
///
/// The view.
fn mat_view<'a>(m: &'a PyReadonlyArray2<'_, f32>) -> PyResult<MatRef<'a, f32>> {
    let (n, d) = (m.shape()[0], m.shape()[1]);
    Ok(MatRef::from_row_major_slice(m.as_slice()?, n, d))
}

/// Flat row-major values as a numpy `(n, k)` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The values, `n * k` of them
/// * `n` - Rows
/// * `k` - Columns
///
/// ### Returns
///
/// The array.
fn flat_array<'py, T: numpy::Element>(
    py: Python<'py>,
    v: Vec<T>,
    n: usize,
    k: usize,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(v.into_pyarray(py).reshape([n, k])?.into_any())
}

/// A kNN as flat row-major slices: `(indices, distances, n, k)`.
type KnnSlices<'a> = (&'a [u32], Option<&'a [f32]>, usize, usize);

/// A row-major `(n, k)` kNN as flat slices with its shape
///
/// ### Params
///
/// * `indices` - The neighbour indices
/// * `distances` - Optional distances of the same shape
///
/// ### Returns
///
/// `(indices, distances, n, k)`, or a `ValueError` on a shape mismatch.
fn knn_slices<'a>(
    indices: &'a PyReadonlyArray2<'_, u32>,
    distances: Option<&'a PyReadonlyArray2<'_, f32>>,
) -> PyResult<KnnSlices<'a>> {
    let (n, k) = (indices.shape()[0], indices.shape()[1]);
    let dist = match distances {
        Some(d) if d.shape() != indices.shape() => {
            return Err(PyValueError::new_err(
                "indices and distances must have the same shape",
            ));
        }
        Some(d) => Some(d.as_slice()?),
        None => None,
    };
    Ok((indices.as_slice()?, dist, n, k))
}

/////////
// kNN //
/////////

/// kNN indices and distances over an embedding
///
/// The twin of `rs_sc_knn_w_dist`: self excluded, true distances, rows a
/// search left short padded by repeating their last neighbour.
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `params` - `to_rust(NeighboursParams)` or `to_rust(KnnParams)`
/// * `validate_index` - Report the recall of the index on a random subset
/// * `seed` - Seed for the index build
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(indices, distances)`: `(n_cells, k)` `uint32` and `float32`.
#[pyfunction]
#[pyo3(signature = (embd, params, validate_index = false, seed = 42, verbose = false))]
pub fn knn_with_dist<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    params: &Bound<'py, PyAny>,
    validate_index: bool,
    seed: usize,
    verbose: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let params: KnnParams = from_py(params, "knn params")?;
    let view = mat_view(&embd)?;
    let n = view.nrows();
    let k = params.k;
    let (idx, dist) = py
        .detach(|| {
            let (mut idx, dist) =
                generate_knn_with_dist(view, &params, true, validate_index, seed, verbose)?;
            let mut dist = dist.expect("return_dist = true always yields distances");
            let padded = pad_knn_rows(&mut idx, Some(&mut dist), k);
            if padded > 0 && verbose {
                println!(
                    "{padded} of {n} rows came back with fewer than {k} neighbours and were \
                     padded with duplicate edges."
                );
            }
            Ok::<_, ScErrors>((idx, dist))
        })
        .map_err(BvErr::from)?;

    let idx: Vec<u32> = idx
        .into_iter()
        .flat_map(|row| row.into_iter().map(|j| j as u32))
        .collect();
    let dist: Vec<f32> = dist.into_iter().flatten().collect();
    Ok((flat_array(py, idx, n, k)?, flat_array(py, dist, n, k)?))
}

/// Recall and distance ratio of a kNN against a reference
///
/// The twin of `rs_compare_knn`.
///
/// ### Params
///
/// * `idx_ref` - `(n, k)` reference indices
/// * `dist_ref` - `(n, k)` reference distances
/// * `idx_query` - `(n, k)` query indices
/// * `dist_query` - `(n, k)` query distances
///
/// ### Returns
///
/// `(all_matches, all_ratios, final_recall, final_ratio)`: `uint32` and
/// `float64` arrays, then two floats.
#[pyfunction]
pub fn knn_metrics<'py>(
    py: Python<'py>,
    idx_ref: PyReadonlyArray2<'py, u32>,
    dist_ref: PyReadonlyArray2<'py, f32>,
    idx_query: PyReadonlyArray2<'py, u32>,
    dist_query: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, f64, f64)> {
    let (ia, da, _, k) = knn_slices(&idx_ref, Some(&dist_ref))?;
    let (ib, db, _, kb) = knn_slices(&idx_query, Some(&dist_query))?;
    if k != kb {
        return Err(PyValueError::new_err(
            "The two kNN data sets have a different set of neighbours requested",
        ));
    }
    let (da, db) = (da.unwrap_or_default(), db.unwrap_or_default());
    let res = py
        .detach(|| compare_knn(ia, da, ib, db, k))
        .map_err(BvErr::from)?;

    let matches: Vec<u32> = res.all_matches.iter().map(|&m| m as u32).collect();
    Ok((
        matches.into_pyarray(py).into_any(),
        res.all_ratios.into_pyarray(py).into_any(),
        res.final_recall,
        res.final_ratio,
    ))
}

///////////
// Graph //
///////////

/// Symmetrise a kNN into the undirected neighbour graph
///
/// The twin of `rs_sc_knn_graph` followed by `rs_graph_to_csr`.
///
/// ### Params
///
/// * `indices` - `(n, k)` neighbours, self excluded, ascending by distance
/// * `distances` - `(n, k)` distances; needed for `knn_sym = "umap"`
/// * `params` - `to_rust(NeighboursParams)` or `to_rust(KnnGraphDefaults)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(data, indices, indptr)` of the symmetric CSR, rows sorted: `float32`,
/// `uint32`, `uint32`.
#[pyfunction]
#[pyo3(signature = (indices, distances, params, verbose = false))]
pub fn knn_graph<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray2<'py, u32>,
    distances: Option<PyReadonlyArray2<'py, f32>>,
    params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let sym: KnnSymmetrisation = from_py(params, "graph params")?;
    let (idx, dist, n, k) = knn_slices(&indices, distances.as_ref())?;
    // only the UMAP graph reads the distances
    let dist = if matches!(sym, KnnSymmetrisation::Umap { .. }) {
        dist
    } else {
        None
    };
    let graph: SparseGraph<f32> = py
        .detach(|| build_knn_graph(idx, dist, k, n, &sym, verbose as usize))
        .map_err(|e| BvErr::from(ScErrors::from(e)))?;

    let mut indptr: Vec<u32> = Vec::with_capacity(n + 1);
    let mut cols: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    indptr.push(0);
    for i in 0..n {
        let (nbrs, ws) = graph.get_neighbours_raw(i);
        cols.extend_from_slice(nbrs);
        data.extend_from_slice(ws);
        indptr.push(cols.len() as u32);
    }
    Ok((
        data.into_pyarray(py).into_any(),
        cols.into_pyarray(py).into_any(),
        indptr.into_pyarray(py).into_any(),
    ))
}

/// Leiden or Louvain on a neighbour graph
///
/// The twin of `rs_graph_leiden`.
///
/// ### Params
///
/// * `data` - `float32` edge weights of the symmetric CSR
/// * `indices` - `uint32` column indices
/// * `indptr` - `uint32` row pointers, `n + 1` of them
/// * `params` - Dict with the `LeidenParams` keys (`resolution`, `beta`,
///   `n_iterations`, `min_gain`, `refine`, `seed`, `local_moving`)
///
/// ### Returns
///
/// `uint32` community per node, 0 the largest.
#[pyfunction]
pub fn leiden<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let params: LeidenParams = from_py(params, "leiden params")?;
    let n = indptr.len().saturating_sub(1);
    let adjacency = CompressedSparseData2::new_csr(
        data.as_slice()?,
        indices.as_slice()?,
        indptr.as_slice()?,
        None,
        (n, n),
    );
    let graph = SparseGraph::new(n, adjacency, false);
    let labels = py
        .detach(|| leiden_sparse_graph(&graph, Some(params)))
        .map_err(|e| BvErr::from(ScErrors::from(e)))?;
    let labels: Vec<u32> = labels.into_iter().map(|c| c as u32).collect();
    Ok(labels.into_pyarray(py).into_any())
}

//////////////////
// Fast cluster //
//////////////////

/// What [fast_cluster] hands back: memberships, assignments, centroids and
/// the optional stability statistics.
type FastClusterOut<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyDict>>,
);

/// k-means coarsening, a centroid graph and Leiden / Louvain on it
///
/// The twin of `rs_fast_cluster_sc`.
///
/// ### Params
///
/// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
/// * `resolutions` - The resolutions, in output order
/// * `params` - `to_rust(FastClusterParams)` with `n_centroids` and
///   `n_iterations` resolved
/// * `n_seeds` - Community-detection seeds per resolution
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(membership, k_means_cluster, centroids, stats)`: the `(n_cells, n_res)`
/// `uint32` communities, the `uint32` centroid per cell, the `float32`
/// centroids, and the stability statistics as a column dict (`None` for one
/// seed).
#[pyfunction]
#[pyo3(signature = (embd, resolutions, params, n_seeds = 1, seed = 42, verbose = false))]
pub fn fast_cluster<'py>(
    py: Python<'py>,
    embd: PyReadonlyArray2<'py, f32>,
    resolutions: Vec<f32>,
    params: &Bound<'py, PyAny>,
    n_seeds: usize,
    seed: usize,
    verbose: bool,
) -> PyResult<FastClusterOut<'py>> {
    let fc: FastClusterParams = from_py(params, "fast cluster params")?;
    let coarsener: CpuKMeans = from_py(params, "k-means params")?;
    let view = mat_view(&embd)?;
    let n = view.nrows();
    let res = py
        .detach(|| {
            fast_clusters(
                view,
                &coarsener,
                &fc,
                &resolutions,
                n_seeds,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;

    let n_res = res.per_resolution.len();
    let membership: Vec<u32> = (0..n)
        .flat_map(|i| res.per_resolution.iter().map(move |r| r.labels[i] as u32))
        .collect();
    let assignments: Vec<u32> = res.assignments.iter().map(|&c| c as u32).collect();
    let c = &res.centroids;
    let (rows, cols) = (c.nrows(), c.ncols());
    let centroids: Vec<f32> = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| c[(i, j)]))
        .collect();

    let stats = if n_seeds > 1 {
        let stats = PyDict::new(py);
        let get = |f: fn(&GridStats) -> f32| -> Vec<f64> {
            res.per_resolution
                .iter()
                .map(|r| r.stability.as_ref().map_or(f64::NAN, |s| f(s) as f64))
                .collect()
        };
        stats.set_item("mean_ari", get(|s| s.mean_ari).into_pyarray(py))?;
        stats.set_item("median_ari", get(|s| s.median_ari).into_pyarray(py))?;
        stats.set_item(
            "mean_conductance",
            get(|s| s.mean_conductance).into_pyarray(py),
        )?;
        stats.set_item(
            "median_conductance",
            get(|s| s.median_conductance).into_pyarray(py),
        )?;
        stats.set_item(
            "mean_n_comms",
            get(|s| s.mean_n_communities).into_pyarray(py),
        )?;
        Some(stats)
    } else {
        None
    };
    Ok((
        flat_array(py, membership, n, n_res)?,
        assignments.into_pyarray(py).into_any(),
        flat_array(py, centroids, rows, cols)?,
        stats,
    ))
}
