//! Ligand-receptor analysis: the Python twins of
//! `rs_generate_ligand_target_influence`, `rs_ligand_activity_scores`,
//! `rs_compute_cluster_expr_stats` and `rs_sc_cellphonedb`.
//!
//! The numerics live in `bixverse-sc`; this only moves buffers. Every index is
//! 0-based. Dense matrices cross as row-major `float64`.

use bixverse_sc::sc_analysis::cellphonedb::{CellPhoneDbParams, LrInteraction, cellphonedb_run};
use bixverse_sc::sc_analysis::nichenet::activity_scoring::{
    LigandActivityScores, ligand_activity_scores as activity_scores,
};
use bixverse_sc::sc_analysis::nichenet::ligand_regulatory_potential::{
    LigandTargetParams, construct_ligand_target_mat,
};
use bixverse_sc::sc_analysis::nichenet::prioritisation::{
    ClusterExpressionStats, compute_cluster_expression_stats,
};
use bixverse_sc_data::data_io::ParallelSparseReader;
use faer::MatRef;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::de::mat_f64;
use crate::error::BvErr;
use crate::graph::{flat_array, from_py};
use crate::pp::index_vec;

/////////////
// Helpers //
/////////////

/// One index vector per group
///
/// ### Params
///
/// * `groups` - 0-based indices per group
///
/// ### Returns
///
/// The indices as `usize`.
fn index_lists(groups: &[PyReadonlyArray1<'_, u32>]) -> PyResult<Vec<Vec<usize>>> {
    groups.iter().map(index_vec).collect()
}

//////////////
// NicheNet //
//////////////

/// The ligand to target influence matrix
///
/// The twin of `rs_generate_ligand_target_influence`.
///
/// ### Params
///
/// * `ligand_seeds` - 0-based seed nodes per ligand or ligand group
/// * `ppi_from` - Signalling layer edge sources
/// * `ppi_to` - Signalling layer edge targets
/// * `ppi_weight` - Signalling layer edge weights
/// * `grn_from` - Gene regulatory layer edge sources
/// * `grn_to` - Gene regulatory layer edge targets
/// * `grn_weight` - Gene regulatory layer edge weights
/// * `n_nodes` - Number of nodes across both layers
/// * `params` - `to_rust(LigandTargetParams)`
///
/// ### Returns
///
/// `(n_seeds, n_nodes)` `float64` influence scores, rows in seed order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ligand_target_influence<'py>(
    py: Python<'py>,
    ligand_seeds: Vec<PyReadonlyArray1<'py, u32>>,
    ppi_from: PyReadonlyArray1<'py, u32>,
    ppi_to: PyReadonlyArray1<'py, u32>,
    ppi_weight: PyReadonlyArray1<'py, f64>,
    grn_from: PyReadonlyArray1<'py, u32>,
    grn_to: PyReadonlyArray1<'py, u32>,
    grn_weight: PyReadonlyArray1<'py, f64>,
    n_nodes: usize,
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let params: LigandTargetParams<f64> = from_py(params, "ligand target params")?;
    let seeds: Vec<Vec<u32>> = ligand_seeds
        .iter()
        .map(|s| Ok(s.as_slice()?.to_vec()))
        .collect::<PyResult<_>>()?;
    let (pf, pt, pw) = (
        ppi_from.as_slice()?,
        ppi_to.as_slice()?,
        ppi_weight.as_slice()?,
    );
    let (gf, gt, gw) = (
        grn_from.as_slice()?,
        grn_to.as_slice()?,
        grn_weight.as_slice()?,
    );
    if pf.len() != pt.len() || pf.len() != pw.len() || gf.len() != gt.len() || gf.len() != gw.len()
    {
        return Err(PyValueError::new_err(
            "from, to and weight of each network need the same length",
        ));
    }
    if let Some(&bad) = pf
        .iter()
        .chain(pt)
        .chain(gf)
        .chain(gt)
        .chain(seeds.iter().flatten())
        .find(|&&v| v as usize >= n_nodes)
    {
        return Err(PyValueError::new_err(format!(
            "node index {bad} is outside the {n_nodes} nodes"
        )));
    }
    let res = py
        .detach(|| construct_ligand_target_mat(n_nodes, pf, pt, pw, gf, gt, gw, &seeds, &params))
        .map_err(BvErr::from)?;
    mat_f64(py, &res)
}

/// Ligand activity scores against gene set memberships
///
/// The twin of `rs_ligand_activity_scores`.
///
/// ### Params
///
/// * `influence` - `(n_ligands, n_background)` `float64` influence, C-contiguous
/// * `in_gene_sets` - One membership vector of length `n_background` per set
///
/// ### Returns
///
/// One dict per gene set with the `float64` per-ligand `auroc`, `aupr`,
/// `aupr_corrected`, `pearson` and `spearman` (`NaN` where undefined).
#[pyfunction]
pub fn ligand_activity_scores<'py>(
    py: Python<'py>,
    influence: PyReadonlyArray2<'py, f64>,
    in_gene_sets: Vec<Vec<bool>>,
) -> PyResult<Bound<'py, PyList>> {
    let (n, d) = (influence.shape()[0], influence.shape()[1]);
    if let Some(bad) = in_gene_sets.iter().find(|s| s.len() != d) {
        return Err(PyValueError::new_err(format!(
            "a gene set membership has length {}, the influence has {d} columns",
            bad.len()
        )));
    }
    let view = MatRef::from_row_major_slice(influence.as_slice()?, n, d);
    let scores: Vec<LigandActivityScores<f64>> = py.detach(|| {
        in_gene_sets
            .iter()
            .map(|s| activity_scores(&view, s))
            .collect()
    });
    let out = PyList::empty(py);
    for s in scores {
        let d = PyDict::new(py);
        d.set_item("auroc", s.auroc)?;
        d.set_item("aupr", s.aupr)?;
        d.set_item("aupr_corrected", s.aupr_corrected)?;
        d.set_item("pearson", s.pearson)?;
        d.set_item("spearman", s.spearman)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Per-cluster mean expression and expressing fraction
///
/// The twin of `rs_compute_cluster_expr_stats`.
///
/// ### Params
///
/// * `f_path_gene` - Path to `counts_genes.bin`
/// * `gene_indices` - 0-based genes
/// * `clusters` - 0-based cell indices per cluster
///
/// ### Returns
///
/// Dict with `mean` and `frac`, `(n_genes, n_clusters)` `float64`.
#[pyfunction]
pub fn cluster_expr_stats<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    gene_indices: PyReadonlyArray1<'py, u32>,
    clusters: Vec<PyReadonlyArray1<'py, u32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let genes = index_vec(&gene_indices)?;
    let clusters = index_lists(&clusters)?;
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
    let res: ClusterExpressionStats<f64> = py
        .detach(|| compute_cluster_expression_stats(&reader, &genes, &clusters))
        .map_err(BvErr::from)?;
    let out = PyDict::new(py);
    out.set_item("mean", mat_f64(py, &res.mean)?)?;
    out.set_item("frac", mat_f64(py, &res.frac)?)?;
    Ok(out)
}

/////////////////
// CellPhoneDB //
/////////////////

/// CellPhoneDB ligand-receptor analysis
///
/// The twin of `rs_sc_cellphonedb`.
///
/// ### Params
///
/// * `f_path_gene` - Path to `counts_genes.bin`
/// * `partner_a` - 0-based subunit genes of partner a per interaction
/// * `partner_b` - The same for partner b
/// * `clusters` - 0-based, disjoint cell indices per cluster
/// * `pairs` - Ordered `(sender, receiver)` cluster pairs; `None` tests all
/// * `deg_genes` - Optional 0-based DEGs per cluster, which replace the
///   expression gate with the DEG gate
/// * `statistical` - Compute the permutation p-values
/// * `params` - `to_rust(CellphonedbParams)`
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// Dict with `means`, `pvals` (`None` unless `statistical`) and `gate`, all
/// `(n_interactions, n_pairs)`, plus the tested `pair_a` / `pair_b`.
#[pyfunction]
#[pyo3(signature = (f_path_gene, partner_a, partner_b, clusters, pairs, deg_genes, statistical, params, verbose))]
#[allow(clippy::too_many_arguments)]
pub fn cellphonedb<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    partner_a: Vec<PyReadonlyArray1<'py, u32>>,
    partner_b: Vec<PyReadonlyArray1<'py, u32>>,
    clusters: Vec<PyReadonlyArray1<'py, u32>>,
    pairs: Option<Vec<(usize, usize)>>,
    deg_genes: Option<Vec<PyReadonlyArray1<'py, u32>>>,
    statistical: bool,
    params: &Bound<'py, PyAny>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if partner_a.len() != partner_b.len() {
        return Err(PyValueError::new_err(
            "partner_a and partner_b need the same length",
        ));
    }
    let params: CellPhoneDbParams<f64> = from_py(params, "CellPhoneDB params")?;
    let interactions: Vec<LrInteraction> = index_lists(&partner_a)?
        .into_iter()
        .zip(index_lists(&partner_b)?)
        .map(|(partner_a, partner_b)| LrInteraction {
            partner_a,
            partner_b,
        })
        .collect();
    let clusters = index_lists(&clusters)?;
    let deg = deg_genes.as_deref().map(index_lists).transpose()?;
    let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;

    let res = py
        .detach(|| {
            cellphonedb_run(
                &reader,
                &interactions,
                &clusters,
                pairs.as_deref(),
                deg.as_deref(),
                statistical,
                params,
                verbose,
            )
        })
        .map_err(BvErr::from)?;

    let n_pairs = res.obs.pairs.len();
    let gate = flat_array(py, res.obs.gate, interactions.len(), n_pairs)?;
    let (pair_a, pair_b): (Vec<usize>, Vec<usize>) = res.obs.pairs.iter().copied().unzip();
    let out = PyDict::new(py);
    out.set_item("means", mat_f64(py, &res.obs.means)?)?;
    match res.pvals {
        Some(p) => out.set_item("pvals", mat_f64(py, &p)?)?,
        None => out.set_item("pvals", py.None())?,
    }
    out.set_item("gate", gate)?;
    out.set_item("pair_a", pair_a)?;
    out.set_item("pair_b", pair_b)?;
    Ok(out)
}
