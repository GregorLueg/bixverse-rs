//! NicheNet ligand-target regulatory potential matrix construction.
//!
//! Pipeline:
//!   1. Optional hub correction on signalling and GRN edge weights.
//!   2. Personalised PageRank from each ligand seed on the signalling graph
//!      (parallel across seeds).
//!   3. Per-seed quantile thresholding of PPR vectors, then per-group mean.
//!      Order matches NicheNet R: threshold first, then average across seeds
//!      within a ligand combination.
//!   4. Sparsify into ligand x node CSR matrix `M`.
//!   5. M @ GRN -> dense ligand x node ligand-target matrix.
//!   6. Optional secondary targets (re-threshold, second matmul, harmonic
//!      combine).
//!   7. Optional topology correction (subtract uniform-PR baseline pushed
//!      through GRN). Per NicheNet docs this is intended to be used with
//!      `ltf_cutoff == 0`.

use faer::{Mat, MatRef};
use petgraph::Graph;
use petgraph::graph::NodeIndex;
use rayon::prelude::*;

use crate::core::math::sparse::{coo_to_csr, csr_sparse_matmul_dense, csr_vecmat};
use crate::core::math::vector_helpers::quantile;
use crate::graph::page_rank::*;
use crate::prelude::*;

////////////
// Params //
////////////

/// Parameters for NicheNet ligand-target matrix construction.
#[derive(Clone, Debug)]
pub struct LigandTargetParams<T: BixverseFloat> {
    /// Hub correction strength for signalling layer (0 disables).
    pub lr_sig_hub: T,
    /// Hub correction strength for GRN layer (0 disables).
    pub gr_hub: T,
    /// Quantile cutoff for per-seed PPR thresholding (0 disables, R default
    /// 0.99).
    pub ltf_cutoff: T,
    /// PPR damping factor (R default 0.5).
    pub damping_factor: T,
    /// Maximum iterations for PPR
    pub max_iter: usize,
    /// Tolerance for PPR
    pub tol: T,
    /// Run secondary-target step (default false in our port).
    pub secondary_targets: bool,
    /// Subtract uniform-PR topology baseline (default false). Intended for use
    /// with `ltf_cutoff == 0`; combining with cutoff > 0 can over-subtract.
    pub topology_correction: bool,
}

/// Defaults based on the R implementation
impl<T: BixverseFloat> Default for LigandTargetParams<T> {
    fn default() -> Self {
        Self {
            lr_sig_hub: T::zero(),
            gr_hub: T::zero(),
            ltf_cutoff: T::from_f64(0.99).unwrap(),
            damping_factor: T::from_f64(0.5).unwrap(),
            max_iter: 1000,
            tol: T::from_f64(1e-7).unwrap(),
            secondary_targets: false,
            topology_correction: false,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Hub correction: weight[i] / indegree(to[i])^h.
///
/// Indegree counts edges, not summed weights -> matches `count(to)` in R.
///
/// ### Params
///
/// * `to` - The `to` nodes.
/// * `weight` - The weight on the node.
/// * `n_nodes` - Number of overall nodes in the graph
/// * `h` - Hub correction factor
///
/// ### Returns
///
/// The hub correction vector
fn hub_correct<T: BixverseFloat>(to: &[u32], weight: &[T], n_nodes: usize, h: T) -> Vec<T> {
    let mut indegree = vec![0u32; n_nodes];
    for &t in to {
        indegree[t as usize] += 1;
    }
    weight
        .par_iter()
        .zip(to.par_iter())
        .map(|(&w, &t)| {
            let d = indegree[t as usize];
            if d == 0 {
                w
            } else {
                w / T::from_u32(d).unwrap().powf(h)
            }
        })
        .collect()
}

/// Build petgraph from indexed edges then convert to PageRankGraph.
///
/// ### Params
///
/// * `n_nodes` - Number of nodes
/// * `from` - Indices of the `from` nodes.
/// * `to` - Indices of the `to` nodes.
/// * `weight` - The edge weight.
///
/// ### Returns
///
/// The [PageRankGraph] for fast personalised page rank calculations.
fn build_ppr_graph<T>(n_nodes: usize, from: &[u32], to: &[u32], weight: &[T]) -> PageRankGraph<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let mut g: Graph<&str, T> = Graph::new();
    for _ in 0..n_nodes {
        g.add_node("");
    }
    for ((&f, &t), &w) in from.iter().zip(to.iter()).zip(weight.iter()) {
        g.add_edge(NodeIndex::new(f as usize), NodeIndex::new(t as usize), w);
    }
    PageRankGraph::from_petgraph(g)
}

/// Threshold each row of a dense matrix at its `q` quantile, return as CSR.
///
/// ### Params
///
/// * `mat` - The dense matrix to sparsify
///
/// ### Returns
///
/// The [CompressedSparseData2] with `.data` populated
fn dense_rows_to_csr_thresholded<T>(mat: &MatRef<T>, q: T) -> CompressedSparseData2<T>
where
    T: BixverseFloat + Send + Sync + Default,
{
    let n_rows = mat.nrows();
    let n_cols = mat.ncols();
    let rows: Vec<Vec<(u32, T)>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row: Vec<T> = (0..n_cols).map(|j| mat[(i, j)]).collect();
            if q > T::zero() {
                let thresh = quantile(&row, q);
                row.iter()
                    .enumerate()
                    .filter_map(|(j, &v)| {
                        if v > thresh {
                            Some((j as u32, v))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                row.iter()
                    .enumerate()
                    .filter_map(|(j, &v)| {
                        if v != T::zero() {
                            Some((j as u32, v))
                        } else {
                            None
                        }
                    })
                    .collect()
            }
        })
        .collect();

    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for sr in &rows {
        for &(j, v) in sr {
            indices.push(j);
            data.push(v);
        }
        indptr.push(indices.len() as u32);
    }
    CompressedSparseData2::new_csr(&data, &indices, &indptr, None::<&[T]>, (n_rows, n_cols))
}

/// Parallel-resistor combination: 1 / (1/primary + 1/secondary), element-wise.
///
/// Zeros are replaced with each matrix's smallest positive value to avoid Inf
/// from 1/0 — matches the R implementation's quirk.
///
/// ### Params
///
/// * `primary` - Primary matrix to update.
/// * `secondary` - Secondary matrix.
fn harmonic_combine_in_place<T: BixverseFloat>(primary: &mut Mat<T>, secondary: &Mat<T>) {
    let n_rows = primary.nrows();
    let n_cols = primary.ncols();

    let min_pos = |m: &Mat<T>| -> T {
        let mut best = T::infinity();
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                let v = m[(i, j)];
                if v > T::zero() && v < best {
                    best = v;
                }
            }
        }
        // Guard against an all-zero matrix (degenerate, but possible).
        if best == T::infinity() {
            T::one()
        } else {
            best
        }
    };

    let mp = min_pos(primary);
    let ms = min_pos(secondary);

    for i in 0..n_rows {
        for j in 0..n_cols {
            let p = if primary[(i, j)] == T::zero() {
                mp
            } else {
                primary[(i, j)]
            };
            let s = if secondary[(i, j)] == T::zero() {
                ms
            } else {
                secondary[(i, j)]
            };
            primary[(i, j)] = T::one() / (T::one() / p + T::one() / s);
        }
    }
}

/// Subtract baseline from each row of `out`, clamp negatives to zero.
///
/// ### Params
///
/// * `out` - Mutable reference to the matrix
/// * `baseline` - The baseline value to subtract
fn subtract_baseline_clamp<T: BixverseFloat>(out: &mut Mat<T>, baseline: &[T]) {
    let n_rows = out.nrows();
    let n_cols = out.ncols();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let v = out[(i, j)] - baseline[j];
            out[(i, j)] = if v < T::zero() { T::zero() } else { v };
        }
    }
}

/// Pack dense rows into CSR, dropping exact zeros.
///
/// ### Params
///
/// * `rows` - Dense row vector
/// * `n_cols` - Number of columns
///
/// ### Returns
///
/// The `CompressedSparseData2` with zeroes removed
fn rows_to_csr<T>(rows: &[Vec<T>], n_cols: usize) -> CompressedSparseData2<T>
where
    T: BixverseFloat + Default,
{
    let n_rows = rows.len();
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in rows {
        for (j, &v) in row.iter().enumerate() {
            if v != T::zero() {
                indices.push(j as u32);
                data.push(v);
            }
        }
        indptr.push(indices.len() as u32);
    }
    CompressedSparseData2::new_csr(&data, &indices, &indptr, None::<&[T]>, (n_rows, n_cols))
}

//////////
// Main //
//////////

/// Construct the ligand-target regulatory potential matrix.
///
/// ### Params
///
/// * `n_nodes` - Number of nodes in the graph
/// * `sig_from` - Signalling network `from` part of the edges
/// * `sig_to` - Signalling network `to` part from the edges
/// * `sig_weight` - Signalling network `weight` of the edges
/// * `grn_from` - Signalling network `from` part of the edges
/// * `grn_to` - Signalling network `to` part from the edges
/// * `grn_weight` - Signalling network `weight` of the edges
/// * `ligand_seeds` - Indices of the ligands to use for the diffusion
/// * `params` - The parameters, see [LigandTargetParams].
///
/// ### Returns
///
/// Returns a dense matrix of shape `(ligand_seeds.len(), n_nodes)` with row
/// order matching `ligand_seeds`. Each entry in `ligand_seeds` is a list of
/// seed node indices for one ligand or ligand combination.
#[allow(clippy::too_many_arguments)]
pub fn construct_ligand_target_mat<T>(
    n_nodes: usize,
    sig_from: &[u32],
    sig_to: &[u32],
    sig_weight: &[T],
    grn_from: &[u32],
    grn_to: &[u32],
    grn_weight: &[T],
    ligand_seeds: &[Vec<u32>],
    params: &LigandTargetParams<T>,
) -> Result<Mat<T>, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum + Send + Sync + Default,
{
    let n_groups = ligand_seeds.len();
    if n_groups == 0 {
        return Ok(Mat::zeros(0, n_nodes));
    }

    // 1. hub correction
    let sig_w = if params.lr_sig_hub > T::zero() {
        hub_correct(sig_to, sig_weight, n_nodes, params.lr_sig_hub)
    } else {
        sig_weight.to_vec()
    };
    let grn_w = if params.gr_hub > T::zero() {
        hub_correct(grn_to, grn_weight, n_nodes, params.gr_hub)
    } else {
        grn_weight.to_vec()
    };

    // 2. build PPR graph and GRN CSR
    let pr_graph = build_ppr_graph(n_nodes, sig_from, sig_to, &sig_w);
    let grn = coo_to_csr(grn_from, grn_to, &grn_w, (n_nodes, n_nodes));

    // 3. flatten seeds across groups
    let mut flat_personalisation: Vec<Vec<T>> = Vec::new();
    let mut seed_to_group: Vec<usize> = Vec::new();
    for (g, group) in ligand_seeds.iter().enumerate() {
        for &seed in group {
            let mut v = vec![T::zero(); n_nodes];
            v[seed as usize] = T::one();
            flat_personalisation.push(v);
            seed_to_group.push(g);
        }
    }

    // 4. run PPR for all seeds in parallel
    let ppr_results: Vec<Vec<T>> = flat_personalisation
        .par_iter()
        .map_init(PageRankWorkingMemory::<T>::new, |wm, p| {
            personalised_page_rank_optimised(
                &pr_graph,
                params.damping_factor,
                p,
                params.max_iter,
                params.tol,
                wm,
            )
        })
        .collect();

    // 5. Per-seed threshold, then per-group accumulate + mean. Matches R
    //    `PPR_wrapper`: threshold each seed independently, then average.
    let mut group_ppr: Vec<Vec<T>> = vec![vec![T::zero(); n_nodes]; n_groups];
    let mut group_counts: Vec<usize> = vec![0; n_groups];
    for (i, mut result) in ppr_results.into_iter().enumerate() {
        if params.ltf_cutoff > T::zero() {
            let thresh = quantile(&result, params.ltf_cutoff);
            for v in result.iter_mut() {
                if *v <= thresh {
                    *v = T::zero();
                }
            }
        }
        let g = seed_to_group[i];
        for (k, v) in result.into_iter().enumerate() {
            group_ppr[g][k] += v;
        }
        group_counts[g] += 1;
    }
    for (g, &count) in group_counts.iter().enumerate() {
        if count > 1 {
            let c = T::from_usize(count).unwrap();
            for v in group_ppr[g].iter_mut() {
                *v /= c;
            }
        }
    }

    // 6. sparsify -> CSR (n_groups x n_nodes)
    let ltf_sparse = rows_to_csr(&group_ppr, n_nodes);

    // 7. M @ GRN
    let mut out = csr_sparse_matmul_dense(&ltf_sparse, &grn)?;

    // 8. Optional secondary targets
    if params.secondary_targets {
        let secondary_sparse = dense_rows_to_csr_thresholded(&out.as_ref(), params.ltf_cutoff);
        let secondary = csr_sparse_matmul_dense(&secondary_sparse, &grn)?;
        harmonic_combine_in_place(&mut out, &secondary);
    }

    // 9. Optional topology correction
    if params.topology_correction {
        let n_t = T::from_usize(n_nodes).unwrap();
        let uniform_p = vec![T::one() / n_t; n_nodes];
        let mut wm = PageRankWorkingMemory::<T>::new();
        let bg_pr = personalised_page_rank_optimised(
            &pr_graph,
            params.damping_factor,
            &uniform_p,
            params.max_iter,
            params.tol,
            &mut wm,
        );
        // No threshold on background: matches `get_pagerank_target` in R.
        let bg_target = csr_vecmat(&bg_pr, &grn)?;
        subtract_baseline_clamp(&mut out, &bg_target);
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A hub exponent of zero leaves every edge weight untouched.
    #[test]
    fn hub_correct_h_zero_is_identity() {
        let to = vec![0u32, 1, 1, 2];
        let w = vec![1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(hub_correct(&to, &w, 3, 0.0), w);
    }

    /// At h = 1 every edge is divided by the in-degree of its target, not its source.
    #[test]
    fn hub_correct_divides_by_indegree() {
        // indegrees: 0 -> 1, 1 -> 2, 2 -> 1
        let to = vec![0u32, 1, 1, 2];
        let w = vec![1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(hub_correct(&to, &w, 3, 1.0), vec![1.0, 1.0, 1.5, 4.0]);
    }

    /// The in-degree is raised to h before dividing, rather than the division being repeated.
    #[test]
    fn hub_correct_power_applied() {
        // node 1 has indegree 2; h=2 -> divide by 4
        let to = vec![1u32, 1];
        let w = vec![8.0_f64, 4.0];
        assert_eq!(hub_correct(&to, &w, 2, 2.0), vec![2.0, 1.0]);
    }

    /// The two matrices combine entrywise as `1 / (1/p + 1/s)`, written back into the primary.
    #[test]
    fn harmonic_combine_basic() {
        let mut p = Mat::<f64>::zeros(1, 2);
        p[(0, 0)] = 2.0;
        p[(0, 1)] = 4.0;
        let mut s = Mat::<f64>::zeros(1, 2);
        s[(0, 0)] = 2.0;
        s[(0, 1)] = 4.0;
        harmonic_combine_in_place(&mut p, &s);
        assert!((p[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((p[(0, 1)] - 2.0).abs() < 1e-12);
    }

    /// Zeros in the primary take its smallest positive entry, keeping the reciprocal finite.
    #[test]
    fn harmonic_combine_replaces_zero_with_min_positive() {
        // primary min positive = 3; (0,0) zero gets replaced by 3
        let mut p = Mat::<f64>::zeros(1, 2);
        p[(0, 0)] = 0.0;
        p[(0, 1)] = 3.0;
        let mut s = Mat::<f64>::zeros(1, 2);
        s[(0, 0)] = 6.0;
        s[(0, 1)] = 6.0;
        harmonic_combine_in_place(&mut p, &s);
        // 1/(1/3 + 1/6) = 2 for both cells
        assert!((p[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((p[(0, 1)] - 2.0).abs() < 1e-12);
    }

    /// The per-column baseline is subtracted and anything below zero is clamped, never left negative.
    #[test]
    fn subtract_baseline_clamps_negatives() {
        let mut m = Mat::<f64>::zeros(2, 3);
        m[(0, 0)] = 0.0;
        m[(0, 1)] = 1.0;
        m[(0, 2)] = 2.0;
        m[(1, 0)] = 1.0;
        m[(1, 1)] = 2.0;
        m[(1, 2)] = 3.0;
        subtract_baseline_clamp(&mut m, &[1.0, 1.0, 1.0]);
        assert_eq!(m[(0, 0)], 0.0); // -1 clamped
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(0, 2)], 1.0);
        assert_eq!(m[(1, 0)], 0.0);
        assert_eq!(m[(1, 1)], 1.0);
        assert_eq!(m[(1, 2)], 2.0);
    }
}
