//! Result conversions both extensions hand back the same way.
//!
//! A CPU method and its GPU twin return the same crate result types, so the
//! dicts and arrays built from them live here once. The Python side reads one
//! layout whichever extension produced it.

use bixverse_methods::graph::graph_structures::max_symmetrise_to_sparse_graph;
use bixverse_methods::methods::nmf_hals::consensus::{
    ConsensusNmfResult, ConsensusParams, ConsensusParamsSpec, KSweepEntry,
};
use bixverse_methods::methods::nmf_hals::{HalsOpts, HalsOptsSpec, NmfResult, StabilisedNmfResult};
use bixverse_sc::mc_generation::generate::{MetaCellsRes, SeaCellsRes};
use bixverse_sc::prelude::{CompressedSparseData2, ScErrors};
use bixverse_sc::sc_analysis::bonsai::{BonsaiScResult, NO_NODE};
use bixverse_sc::sc_analysis::fast_clusters::{FastClusterResult, GridStats};
use bixverse_sc::sc_analysis::nebula::NebulaScRes;
use bixverse_sc::sc_batch_correction::bbknn::bbknn_balanced_neighbours;
use bixverse_sc::sc_processing::knn::pad_knn_rows;
use bixverse_sc::sc_processing::scrublet::FinalScrubletRes;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{f64_array, flat_array, from_py, mat_array, mat_f64, u32_array};
use crate::error::BvErr;

///////////
// Types //
///////////

/// What a fast clustering hands back: memberships, assignments, centroids and
/// the optional stability statistics.
pub type FastClusterOut<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyDict>>,
);

/// What BBKNN hands back: the kNN indices and distances, and the symmetric
/// CSR of the connectivities as data, column indices and row pointers.
pub type BbknnOut<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

//////////////////
// Fast cluster //
//////////////////

/// A fast clustering result as NumPy
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The result
/// * `n` - Number of cells
/// * `n_seeds` - Seeds per resolution; the statistics only exist above 1
///
/// ### Returns
///
/// `(membership, k_means_cluster, centroids, stats)`: the `(n_cells, n_res)`
/// `uint32` communities, the `uint32` centroid per cell, the `float32`
/// centroids, and the stability statistics as a column dict (`None` for one
/// seed).
pub fn fast_cluster_out<'py>(
    py: Python<'py>,
    res: FastClusterResult,
    n: usize,
    n_seeds: usize,
) -> PyResult<FastClusterOut<'py>> {
    let n_res = res.per_resolution.len();
    let membership: Vec<u32> = (0..n)
        .flat_map(|i| res.per_resolution.iter().map(move |r| r.labels[i] as u32))
        .collect();
    let assignments: Vec<u32> = res.assignments.iter().map(|&c| c as u32).collect();

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
        mat_array(py, &res.centroids)?,
        stats,
    ))
}

///////////
// BBKNN //
///////////

/// The BBKNN distances and connectivities as a kNN and a symmetric CSR
///
/// The kNN keeps the batch-balanced `k` nearest per row of the distance
/// matrix (`bbknn_balanced_neighbours`), as R's `rs_bbknn_filtering`. Short
/// rows are padded. The connectivities are max-symmetrised.
///
/// ### Params
///
/// * `py` - GIL token
/// * `distances` - The BBKNN distance matrix
/// * `connectivities` - The BBKNN connectivities
/// * `batch_labels` - Batch per cell, as passed to BBKNN
/// * `k` - Neighbours per cell to keep
///
/// ### Returns
///
/// See [`BbknnOut`].
pub fn bbknn_out<'py>(
    py: Python<'py>,
    distances: &CompressedSparseData2<f32>,
    connectivities: &CompressedSparseData2<f32>,
    batch_labels: &[usize],
    k: usize,
) -> PyResult<BbknnOut<'py>> {
    let n = distances.shape.0;
    let (idx, dist, graph) = py.detach(|| {
        let (mut idx, mut dist) = bbknn_balanced_neighbours(distances, batch_labels, k)?;
        pad_knn_rows(&mut idx, Some(&mut dist), k);
        Ok::<_, ScErrors>((idx, dist, max_symmetrise_to_sparse_graph(connectivities)))
    })
    .map_err(BvErr::from)?;

    let idx: Vec<u32> = idx.into_iter().flatten().map(|j| j as u32).collect();
    let dist: Vec<f32> = dist.into_iter().flatten().collect();
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
        flat_array(py, idx, n, k)?,
        flat_array(py, dist, n, k)?,
        data.into_pyarray(py).into_any(),
        cols.into_pyarray(py).into_any(),
        indptr.into_pyarray(py).into_any(),
    ))
}

//////////////
// Scrublet //
//////////////

/// A Scrublet result as a dict with R's keys
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The result, the optional combined PCA and the optional doublet
///   parents
///
/// ### Returns
///
/// The dict; `pca`, `pair_1` and `pair_2` are `None` unless asked for.
pub fn scrublet_dict<'py>(py: Python<'py>, res: FinalScrubletRes) -> PyResult<Bound<'py, PyDict>> {
    let (res, pca, pair_1, pair_2) = res;
    let out = PyDict::new(py);
    out.set_item(
        "predicted_doublets",
        res.predicted_doublets.into_pyarray(py),
    )?;
    out.set_item("doublet_scores_obs", f64_array(py, &res.doublet_scores_obs))?;
    out.set_item("doublet_scores_sim", f64_array(py, &res.doublet_scores_sim))?;
    out.set_item("doublet_errors_obs", f64_array(py, &res.doublet_errors_obs))?;
    out.set_item("z_scores", f64_array(py, &res.z_scores))?;
    out.set_item("threshold", res.threshold as f64)?;
    out.set_item("detected_doublet_rate", res.detected_doublet_rate as f64)?;
    out.set_item(
        "detectable_doublet_fraction",
        res.detectable_doublet_fraction as f64,
    )?;
    out.set_item("overall_doublet_rate", res.overall_doublet_rate as f64)?;
    out.set_item("pca", pca.map(|m| mat_array(py, &m)).transpose()?)?;
    out.set_item("pair_1", pair_1.map(|p| u32_array(py, &p)))?;
    out.set_item("pair_2", pair_2.map(|p| u32_array(py, &p)))?;
    Ok(out)
}

////////////
// NEBULA //
////////////

/// A NEBULA result as a dict with R's keys
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The per-gene fits and the Wald test
///
/// ### Returns
///
/// The dict; `coefficients` and `se` as `(n_kept, n_coef)` arrays,
/// `cell_overdispersion_shrunk` `None` unless shrinkage was on.
pub fn nebula_dict<'py>(py: Python<'py>, res: NebulaScRes) -> PyResult<Bound<'py, PyDict>> {
    let n_kept = res.gene_idx.len();
    let out = PyDict::new(py);
    out.set_item("gene_idx", u32_array(py, &res.gene_idx))?;
    out.set_item(
        "coefficients",
        flat_array(py, res.coefficients, n_kept, res.n_coef)?,
    )?;
    out.set_item("se", flat_array(py, res.se, n_kept, res.n_coef)?)?;
    out.set_item(
        "subject_overdispersion",
        res.subject_overdispersion.into_pyarray(py),
    )?;
    out.set_item(
        "cell_overdispersion",
        res.cell_overdispersion.into_pyarray(py),
    )?;
    out.set_item(
        "cell_overdispersion_shrunk",
        res.cell_overdispersion_shrunk.map(|v| v.into_pyarray(py)),
    )?;
    out.set_item("convergence", res.convergence.into_pyarray(py))?;
    out.set_item("sigma_at_bound", res.sigma_at_bound.into_pyarray(py))?;
    out.set_item("log_fc", res.log_fc.into_pyarray(py))?;
    out.set_item("effect_se", res.effect_se.into_pyarray(py))?;
    out.set_item("z", res.z.into_pyarray(py))?;
    out.set_item("p_values", res.p_val.into_pyarray(py))?;
    out.set_item("fdr", res.fdr.into_pyarray(py))?;
    Ok(out)
}

////////////////
// Meta cells //
////////////////

/// A meta cell generation result as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The memberships and the aggregated counts
///
/// ### Returns
///
/// Dict with `metacells` (store cells per meta cell), the aggregated CSR as
/// `indptr`, `indices`, `raw` and `norm`, its `shape`, and `n_total`.
pub fn metacells_dict<'py>(py: Python<'py>, res: MetaCellsRes) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let metacells = PyList::empty(py);
    for mc in res.metacells {
        let mc: Vec<u32> = mc.into_iter().map(|x| x as u32).collect();
        metacells.append(mc.into_pyarray(py))?;
    }
    let agg = res.aggregated;
    let raw: Vec<f32> = agg.data.iter().map(|&x| x as f32).collect();
    d.set_item("metacells", metacells)?;
    d.set_item("indptr", agg.indptr.into_pyarray(py))?;
    d.set_item("indices", agg.indices.into_pyarray(py))?;
    d.set_item("raw", raw.into_pyarray(py))?;
    d.set_item(
        "norm",
        agg.data_2
            .expect("aggregation writes the normalised layer")
            .into_pyarray(py),
    )?;
    d.set_item("shape", agg.shape)?;
    d.set_item("n_total", res.n_total)?;
    Ok(d)
}

/// A SEACells result as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The meta cells, the RSS history and the archetypes
///
/// ### Returns
///
/// See [`metacells_dict`], plus `rss` and the store index of each `archetype`.
pub fn seacells_dict<'py>(py: Python<'py>, res: SeaCellsRes) -> PyResult<Bound<'py, PyDict>> {
    let archetypes: Vec<u32> = res.archetypes.iter().map(|&x| x as u32).collect();
    let d = metacells_dict(py, res.base)?;
    d.set_item("rss", res.rss.into_pyarray(py))?;
    d.set_item("archetypes", archetypes.into_pyarray(py))?;
    Ok(d)
}

////////////
// Bonsai //
////////////

/// Parent per node, the root's as `-1`
///
/// ### Params
///
/// * `parent` - Parents, [`NO_NODE`] for the root
///
/// ### Returns
///
/// The parents as `int64`.
pub fn parents_out(parent: &[u32]) -> Vec<i64> {
    parent
        .iter()
        .map(|&p| if p == NO_NODE { -1 } else { p as i64 })
        .collect()
}

/// A Bonsai tree as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The tree, its layout and the run log
///
/// ### Returns
///
/// Dict with `parent`, `branch`, `x`, `y`, `n_leaves`, `loglik`, the `steps`
/// and `timings` logs as column dicts, and `genes_used`.
pub fn bonsai_dict<'py>(py: Python<'py>, res: BonsaiScResult) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("parent", parents_out(&res.parent).into_pyarray(py))?;
    out.set_item("branch", res.branch.into_pyarray(py))?;
    out.set_item("x", res.x.into_pyarray(py))?;
    out.set_item("y", res.y.into_pyarray(py))?;
    out.set_item("n_leaves", res.n_leaves)?;
    out.set_item("loglik", res.loglik)?;

    let steps = PyDict::new(py);
    let (mut name, mut loglik, mut secs) = (Vec::new(), Vec::new(), Vec::new());
    for (n, l, s) in res.steps {
        name.push(n);
        loglik.push(l);
        secs.push(s);
    }
    steps.set_item("step", name)?;
    steps.set_item("loglik", loglik)?;
    steps.set_item("seconds", secs)?;
    out.set_item("steps", steps)?;

    let timings = PyDict::new(py);
    let (stage, seconds): (Vec<String>, Vec<f64>) = res.timings.into_iter().unzip();
    timings.set_item("stage", stage)?;
    timings.set_item("seconds", seconds)?;
    out.set_item("timings", timings)?;

    let genes: Vec<u32> = res.genes_used.iter().map(|&g| g as u32).collect();
    out.set_item("genes_used", genes.into_pyarray(py))?;
    Ok(out)
}

/////////
// NMF //
/////////

/// The HALS solver options from a parameter dict
///
/// ### Params
///
/// * `params` - `to_rust(NmfHalsParams)`
/// * `seed` - Seed for a random initialisation
///
/// ### Returns
///
/// The options, or a `ValueError` on an unusable value.
pub fn hals_opts(params: &Bound<'_, PyAny>, seed: usize) -> PyResult<HalsOpts<f32>> {
    let spec: HalsOptsSpec = from_py(params, "NMF HALS params")?;
    spec.into_opts(seed)
        .map_err(|e| BvErr::from(ScErrors::from(e)).into())
}

/// The consensus step options from a parameter dict
///
/// ### Params
///
/// * `params` - `to_rust(NmfConsensusParams)`
/// * `seed` - Seed injected into the consensus k-means, as R does
///
/// ### Returns
///
/// The options, or a `ValueError` on an unusable value.
pub fn consensus_params(params: &Bound<'_, PyAny>, seed: usize) -> PyResult<ConsensusParams<f32>> {
    let mut spec: ConsensusParamsSpec = from_py(params, "NMF consensus params")?;
    spec.consensus_seed = Some(seed as u64);
    spec.into_params()
        .map_err(|e| BvErr::from(ScErrors::from(e)).into())
}

/// One NMF run as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The run; `w` is cells x k, `h` is k x genes
///
/// ### Returns
///
/// Dict with `w`, `h`, `final_loss`, `n_iter` and `converged`.
pub fn nmf_single_dict<'py>(py: Python<'py>, res: NmfResult<f32>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("w", mat_f64(py, &res.w)?)?;
    out.set_item("h", mat_f64(py, &res.h)?)?;
    out.set_item("final_loss", res.final_loss as f64)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;
    Ok(out)
}

/// NMF restarts as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The restarts
///
/// ### Returns
///
/// Dict with `w_all`, `h_per_run`, `losses`, `converged` and the 0-based
/// `best_idx`.
pub fn nmf_multi_dict<'py>(
    py: Python<'py>,
    res: StabilisedNmfResult<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let h = PyList::empty(py);
    for m in &res.h_per_run {
        h.append(mat_f64(py, m)?)?;
    }
    out.set_item("w_all", mat_f64(py, &res.w_all)?)?;
    out.set_item("h_per_run", h)?;
    out.set_item(
        "losses",
        res.losses
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    out.set_item("converged", res.converged)?;
    out.set_item("best_idx", res.best_idx)?;
    Ok(out)
}

/// Consensus NMF as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The consensus factors and the cluster diagnostics
///
/// ### Returns
///
/// Dict with R's keys, every index 0-based and unclustered components `-1`.
pub fn nmf_consensus_dict<'py>(
    py: Python<'py>,
    res: ConsensusNmfResult<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let f64s = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let c = &res.clusters;
    let labels: Vec<i64> = c
        .labels
        .iter()
        .map(|l| l.map_or(-1, |v| v as i64))
        .collect();
    let out = PyDict::new(py);
    out.set_item("w", mat_f64(py, &res.w)?)?;
    out.set_item("h", mat_f64(py, &res.h)?)?;
    out.set_item("rel_error", res.error as f64)?;
    out.set_item("rel_run_errors", f64s(&res.run_errors).into_pyarray(py))?;
    out.set_item("labels", labels.into_pyarray(py))?;
    out.set_item("local_density", f64s(&c.local_density).into_pyarray(py))?;
    out.set_item("kept", u32_array(py, &c.kept))?;
    out.set_item("silhouette", f64s(&c.silhouette).into_pyarray(py))?;
    out.set_item("stability", c.stability as f64)?;
    out.set_item("cluster_sizes", u32_array(py, &c.sizes))?;
    out.set_item("n_dropped", c.n_dropped)?;
    out.set_item("n_empty_clusters", c.n_empty_clusters)?;
    Ok(out)
}

/// A consensus k sweep as a column dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `entries` - One entry per k
///
/// ### Returns
///
/// Dict with one array per sweep statistic.
pub fn nmf_sweep_dict<'py>(
    py: Python<'py>,
    entries: Vec<KSweepEntry<f32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let ints = |f: fn(&KSweepEntry<f32>) -> usize| entries.iter().map(f).collect::<Vec<_>>();
    let floats =
        |f: fn(&KSweepEntry<f32>) -> f32| entries.iter().map(|e| f(e) as f64).collect::<Vec<f64>>();
    out.set_item("k", u32_array(py, &ints(|e| e.k)))?;
    out.set_item("stability", floats(|e| e.stability).into_pyarray(py))?;
    out.set_item("best_error", floats(|e| e.best_error).into_pyarray(py))?;
    out.set_item("median_error", floats(|e| e.median_error).into_pyarray(py))?;
    out.set_item(
        "consensus_failed",
        entries
            .iter()
            .map(|e| e.consensus_failed)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("n_dropped", u32_array(py, &ints(|e| e.n_dropped)))?;
    out.set_item(
        "n_empty_clusters",
        u32_array(py, &ints(|e| e.n_empty_clusters)),
    )?;
    out.set_item("n_converged", u32_array(py, &ints(|e| e.n_converged)))?;
    Ok(out)
}
