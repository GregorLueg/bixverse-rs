//! Implementation of the v2 version of Harmony designed for large-scale data
//! sets, see, Patikas, et al., bioRxiv, 2026.
//!
//! Key improvements over v1:
//! - Stabilised diversity penalty (scale-invariant objective)
//! - Batch pruning in ridge regression
//! - Arrowhead matrix inversion for single-covariate case
//! - Dynamic lambda estimation
//! - Theta scaling by batch size

use faer::linalg::solvers::PartialPivLu;
use faer::{Mat, MatRef, linalg::solvers::DenseSolveCore};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::ml::clustering::k_means::KMeansParamsWrappers;
use crate::prelude::parse_verbosity_level;
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::cosine_normalise;

use super::harmony::{
    BatchInfo, HarmonyResult, OEPair, compute_all_diversity_statistics, compute_cosine_distances,
    compute_scaled_distances, create_batch_infos, initialise_r_from_dist, run_kmeans_cosine,
    update_centroids_from_r,
};

////////////
// Params //
////////////

/// Parameters for Harmony v2 batch correction.
pub struct HarmonyParamsV2 {
    /// Number of clusters
    pub k: usize,
    /// Per-cluster diversity weights (length 1 or K)
    pub sigma: Vec<f32>,
    /// Per-variable diversity penalties (length 1 or n_variables)
    pub theta: Vec<f32>,
    /// Ridge penalty (length 1)
    pub lambda: Vec<f32>,
    /// Fraction of cells to update per block (0.0-1.0)
    pub block_size: f32,
    /// Maximum diversity-refinement iterations per Harmony round
    pub max_iter_kmeans: usize,
    /// Maximum Harmony outer iterations
    pub max_iter_harmony: usize,
    /// Clustering convergence threshold
    pub epsilon_kmeans: f32,
    /// Harmony convergence threshold
    pub epsilon_harmony: f32,
    /// Window size for convergence checking
    pub window_size: usize,
    /// Alpha for dynamic lambda estimation (0 < alpha < 1)
    pub alpha: f32,
    /// Tau for theta scaling by batch size (0 = no scaling)
    pub tau: f32,
    /// Batch proportion cutoff for pruning in ridge regression
    pub batch_proportion_cutoff: f32,
    /// Whether to estimate lambda dynamically per cluster
    pub use_dynamic_lambda: bool,
    /// K-mean parameters, see [KMeansParamsWrappers]
    pub kmeans_params: KMeansParamsWrappers,
}

impl Default for HarmonyParamsV2 {
    fn default() -> Self {
        Self {
            k: 100,
            sigma: vec![0.1],
            theta: vec![2.0],
            lambda: vec![1.0],
            block_size: 0.05,
            max_iter_kmeans: 4,
            max_iter_harmony: 10,
            epsilon_kmeans: 1e-3,
            epsilon_harmony: 1e-2,
            window_size: 3,
            alpha: 0.2,
            tau: 0.0,
            batch_proportion_cutoff: 1e-5,
            use_dynamic_lambda: false,
            kmeans_params: KMeansParamsWrappers::default(),
        }
    }
}

/////////////
// Helpers //
/////////////

/// Expand theta per-variable to per-level, with optional scaling by batch size.
///
/// When tau > 0, scales each level's theta by `1 - exp(-(N_b / (K * tau))^2)`,
/// dampening the diversity penalty for small batches.
///
/// ### Params
///
/// * `theta` - Per-variable theta values (length n_variables)
/// * `batch_infos` - Batch information per variable
/// * `k` - Number of clusters
/// * `tau` - Scaling parameter (0 disables scaling)
///
/// ### Returns
///
/// Nested Vec: `[var_idx][level_idx] -> f32` theta value
pub fn expand_theta(theta: &[f32], batch_infos: &[BatchInfo], k: usize, tau: f32) -> Vec<Vec<f32>> {
    batch_infos
        .iter()
        .enumerate()
        .map(|(var_idx, info)| {
            (0..info.n_levels)
                .map(|level_idx| {
                    let base = theta[var_idx];
                    if tau > 0.0 {
                        let n_b = info.batch_indices[level_idx].len() as f32;
                        let ratio = n_b / (k as f32 * tau);
                        base * (1.0 - (-ratio * ratio).exp())
                    } else {
                        base
                    }
                })
                .collect()
        })
        .collect()
}

/// Windowed convergence check.
///
/// Compares the mean objective over the most recent `window_size` values
/// against the preceding window. Returns true when the relative change
/// drops below `epsilon`.
///
/// ### Params
///
/// * `objectives` - Objective values (must have >= 2 * window_size entries)
/// * `window_size` - Number of values per window
/// * `epsilon` - Convergence threshold
///
/// ### Returns
///
/// Whether convergence is reached
pub fn check_convergence(objectives: &[f32], window_size: usize, epsilon: f32) -> bool {
    let n = objectives.len();
    if n < 2 * window_size {
        return false;
    }
    let mut obj_old = 0_f32;
    let mut obj_new = 0_f32;
    for i in 0..window_size {
        obj_old += objectives[n - 1 - window_size - i];
        obj_new += objectives[n - 1 - i];
    }
    let rel_change = (obj_old - obj_new).abs() / obj_old.abs();
    rel_change < epsilon
}

////////////////////
// Objective (v2) //
////////////////////

/// Compute Harmony v2 objective with stabilised diversity penalty.
///
/// Objective = (kmeans_error + entropy + cross_entropy) * 2000/N, where
/// cross_entropy uses `log((O + E + 1) / (2E + 1))` for numerical stability
/// when `O_kb -> 0`. Cell-wise sums are reduced in parallel; results differ
/// from a serial reduction at floating-point rounding level only.
///
/// ### Params
///
/// * `r` - Soft assignments (K x N), columns sum to 1
/// * `dist_mat` - Distance matrix (K x N)
/// * `oe_pairs` - Observed/expected pairs per variable
/// * `sigma` - Per-cluster weights (length K)
/// * `theta_expanded` - Per-level theta values, `[var_idx][level_idx]`
/// * `batch_infos` - Batch information per variable
///
/// ### Returns
///
/// Objective value (lower is better)
pub fn compute_objective_v2(
    r: MatRef<f32>,
    dist_mat: MatRef<f32>,
    oe_pairs: &[OEPair],
    sigma: &[f32],
    theta_expanded: &[Vec<f32>],
    batch_infos: &[BatchInfo],
) -> f32 {
    let k = r.nrows();
    let n = r.ncols();
    let n_vars = batch_infos.len();
    let norm_const = 2000.0 / n as f32;

    // precompute log_ratio per variable (small K x B tables).
    let log_ratios: Vec<Vec<f32>> = batch_infos
        .iter()
        .enumerate()
        .map(|(var_idx, info)| {
            let b = info.n_levels;
            let OEPair { o, e } = &oe_pairs[var_idx];
            let mut lr = vec![0.0f32; k * b];
            for cluster in 0..k {
                for level in 0..b {
                    let o_val = o[(cluster, level)];
                    let e_val = e[(cluster, level)];
                    lr[cluster * b + level] = ((o_val + e_val + 1.0) / (2.0 * e_val + 1.0)).ln();
                }
            }
            lr
        })
        .collect();

    // single fused sweep over cells: kmeans + entropy + all variables'
    // cross-entropy in one go...
    let total: f32 = (0..n)
        .into_par_iter()
        .map(|cell| {
            let mut acc = 0.0f32;
            for cluster in 0..k {
                let r_val = r[(cluster, cell)];
                let s_k = sigma[cluster];
                acc += r_val * dist_mat[(cluster, cell)];
                if r_val > 0.0 {
                    acc += r_val * r_val.ln() * s_k;
                }
                for var_idx in 0..n_vars {
                    let info = &batch_infos[var_idx];
                    let level = info.cell_to_level[cell];
                    let theta_l = theta_expanded[var_idx][level];
                    let lr = log_ratios[var_idx][cluster * info.n_levels + level];
                    acc += r_val * s_k * theta_l * lr;
                }
            }
            acc
        })
        .sum();

    total * norm_const
}

///////////////////
// R update (v2) //
///////////////////

/// Update soft assignments with stabilised diversity penalty.
///
/// For each cell n and cluster k, the assignment is proportional to:
///
/// `scale_dist[k,n] * prod_v ((2*E_v[k,b] + 1) / (O_v[k,b] + E_v[k,b] + 1))^theta_l`
///
/// where `scale_dist` is the column-normalised `exp(-dist/sigma)` base,
/// precomputed once per round by the caller (it is constant across the inner
/// refinement loop). Expected counts are rank-1 (`E[k,b] = r_sum[k]*pr_b[b]`)
/// and are tracked via `r_sum` rather than a full K x B matrix, then rebuilt at
/// the end. The per-block R recompute is parallel over cells.
///
/// ### Params
///
/// * `scale_dist` - Column-normalised base assignments (K x N), see
///   [compute_scaled_distances]
/// * `theta_expanded` - Per-level theta values, `[var_idx][level_idx]`
/// * `batch_infos` - Batch information per variable
/// * `block_size` - Fraction of cells per update block
/// * `seed` - Random seed for shuffling
/// * `r_init` - Initial R matrix (K x N)
/// * `oe_init` - Initial observed/expected pairs per variable
///
/// ### Returns
///
/// Tuple of (R: K x N, Vec of `OEPair` per variable)
#[allow(clippy::too_many_arguments)]
pub fn update_r_with_diversity_v2(
    scale_dist: MatRef<f32>,
    theta_expanded: &[Vec<f32>],
    batch_infos: &[BatchInfo],
    block_size: f32,
    seed: usize,
    r_init: MatRef<f32>,
    oe_init: &[OEPair],
) -> (Mat<f32>, Vec<OEPair>) {
    let k = scale_dist.nrows();
    let n = scale_dist.ncols();
    let n_vars = batch_infos.len();

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut update_order: Vec<usize> = (0..n).collect();
    update_order.shuffle(&mut rng);

    let mut r = r_init.to_owned();
    let mut oe: Vec<OEPair> = oe_init
        .iter()
        .map(|OEPair { o, e }| OEPair {
            o: o.to_owned(),
            e: e.to_owned(),
        })
        .collect();

    let mut r_sum = vec![0.0f32; k];
    for cell_idx in 0..n {
        for cluster_idx in 0..k {
            r_sum[cluster_idx] += r[(cluster_idx, cell_idx)];
        }
    }

    let n_blocks = (1.0 / block_size).ceil() as usize;
    let cells_per_block = (n as f32 * block_size) as usize;

    // re-used across blocks; sized to the largest block.
    let mut new_cols_flat = vec![0.0f32; cells_per_block * k];

    for block_idx in 0..n_blocks {
        let idx_min = block_idx * cells_per_block;
        let idx_max = ((block_idx + 1) * cells_per_block).min(n);
        let block = &update_order[idx_min..idx_max];
        let block_len = block.len();

        // step 1: remove block cells from O and r_sum.
        for &cell_idx in block {
            for cluster_idx in 0..k {
                r_sum[cluster_idx] -= r[(cluster_idx, cell_idx)];
            }
            for var_idx in 0..n_vars {
                let level = batch_infos[var_idx].cell_to_level[cell_idx];
                let o = &mut oe[var_idx].o;
                for cluster_idx in 0..k {
                    o[(cluster_idx, level)] -= r[(cluster_idx, cell_idx)];
                }
            }
        }

        // step 2: recompute R for block cells into a pre-allocated flat buffer.
        {
            let o_refs: Vec<MatRef<f32>> = oe.iter().map(|p| p.o.as_ref()).collect();
            new_cols_flat[..block_len * k]
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(i, col)| {
                    let cell_idx = block[i];
                    let mut col_sum = 0.0f32;
                    for cluster_idx in 0..k {
                        let base = scale_dist[(cluster_idx, cell_idx)];
                        let mut penalty = 1.0f32;
                        for var_idx in 0..n_vars {
                            let level = batch_infos[var_idx].cell_to_level[cell_idx];
                            let o_val = o_refs[var_idx][(cluster_idx, level)];
                            let e_val = r_sum[cluster_idx] * batch_infos[var_idx].pr_b[level];
                            let theta_l = theta_expanded[var_idx][level];
                            penalty *= ((2.0 * e_val + 1.0) / (o_val + e_val + 1.0)).powf(theta_l);
                        }
                        let new_r = base * penalty;
                        col[cluster_idx] = new_r;
                        col_sum += new_r;
                    }
                    if col_sum > 0.0 {
                        for v in col.iter_mut() {
                            *v /= col_sum;
                        }
                    }
                });
        }

        for (i, &cell_idx) in block.iter().enumerate() {
            for cluster_idx in 0..k {
                r[(cluster_idx, cell_idx)] = new_cols_flat[i * k + cluster_idx];
            }
        }

        // step 3: add block cells back to O and r_sum.
        for &cell_idx in block {
            for cluster_idx in 0..k {
                r_sum[cluster_idx] += r[(cluster_idx, cell_idx)];
            }
            for var_idx in 0..n_vars {
                let level = batch_infos[var_idx].cell_to_level[cell_idx];
                let o = &mut oe[var_idx].o;
                for cluster_idx in 0..k {
                    o[(cluster_idx, level)] += r[(cluster_idx, cell_idx)];
                }
            }
        }
    }

    // rebuild E = r_sum (outer) pr_b for each variable.
    for (var_idx, info) in batch_infos.iter().enumerate() {
        let b = info.n_levels;
        let e = &mut oe[var_idx].e;
        for cluster_idx in 0..k {
            let rs = r_sum[cluster_idx];
            for level_idx in 0..b {
                e[(cluster_idx, level_idx)] = rs * info.pr_b[level_idx];
            }
        }
    }

    (r, oe)
}

///////////////////////////
// Ridge regression (v2) //
///////////////////////////

/// Solve `W = inv(design_cov) * phi_z` using arrowhead closed-form inversion.
///
/// When a single covariate is being corrected, the normal-equation matrix
/// from `[intercept | one-hot]` has arrowhead structure, enabling O(p)
/// inversion instead of O(p^3) LU. Uses `f64` to avoid catastrophic
/// cancellation under the hood.
///
/// ### Params
///
/// * `design_cov` - Normal-equation matrix (p x p)
/// * `phi_z` - Right-hand side (p x d)
///
/// ### Returns
///
/// `Some(W)` (p x d) on success, `None` if the Schur complement is
/// degenerate (caller should fall back to LU)
pub fn solve_arrowhead(design_cov: &Mat<f32>, phi_z: &Mat<f32>) -> Option<Mat<f32>> {
    let p = design_cov.nrows();
    let d = phi_z.ncols();

    let mut ac = vec![0.0f64; p];
    for i in 0..p {
        ac[i] = -(design_cov[(0, i)] as f64);
    }
    ac[0] = 1.0;

    let mut b = vec![0.0f64; p];
    for i in 1..p {
        let diag = design_cov[(i, i)] as f64;
        if diag.abs() < 1e-12 {
            return None;
        }
        b[i] = 1.0 / diag;
    }

    let mut u: f64 = design_cov[(0, 0)] as f64;
    for i in 0..p {
        u -= ac[i] * ac[i] * b[i];
    }
    if u.abs() < 1e-10 {
        return None;
    }

    let mut ac_b = vec![0.0f64; p];
    for i in 0..p {
        ac_b[i] = ac[i] * b[i];
    }
    ac_b[0] = 1.0;

    let mut v = vec![0.0f64; d];
    for feat in 0..d {
        for j in 0..p {
            v[feat] += ac_b[j] * phi_z[(j, feat)] as f64;
        }
    }

    let inv_u = 1.0 / u;
    let mut w = Mat::<f32>::zeros(p, d);
    for i in 0..p {
        for feat in 0..d {
            w[(i, feat)] = (inv_u * ac_b[i] * v[feat] + b[i] * phi_z[(i, feat)] as f64) as f32;
        }
    }

    Some(w)
}

/// Fallback LU solve for `W = inv(design_cov) * phi_z`.
///
/// Solve `W = inv(design_cov) * phi_z` via LU decomposition. Uses `f64` to
/// avoid catastrophic cancellation issues.
///
/// ### Params
///
/// * `design_cov` - Normal-equation matrix (p x p)
/// * `phi_z` - Right-hand side (p x d)
///
/// ### Returns
///
/// W (p x d)
pub fn solve_lu(design_cov: &Mat<f32>, phi_z: &Mat<f32>) -> Mat<f32> {
    let p = design_cov.nrows();
    let d = phi_z.ncols();

    let cov_f64 = Mat::<f64>::from_fn(p, p, |i, j| design_cov[(i, j)] as f64);
    let phi_z_f64 = Mat::<f64>::from_fn(p, d, |i, j| phi_z[(i, j)] as f64);

    let lu: PartialPivLu<f64> = cov_f64.partial_piv_lu();
    let inv_cov = lu.inverse();
    let w_f64 = &inv_cov * &phi_z_f64;

    Mat::<f32>::from_fn(p, d, |i, j| w_f64[(i, j)] as f32)
}

/// Single-covariate fast path.
///
/// One pass over cells builds the weighted segmented sum `S[level, cluster, :]`;
/// each cluster's arrowhead system is then assembled from `S`, `O`, and `r_sum`
/// without any further cell scan. Mirrors the GPU formulation in `harmony_gpu`.
///
/// ### Params
///
/// * `z_orig` - Original (uncorrected) data (N x d)
/// * `r` - Soft assignments (K x N)
/// * `info` - Batch information for the single covariate
/// * `oe` - Observed/expected pair (used for pruning and dynamic lambda)
/// * `lambda` - Fixed ridge penalty (used when `use_dynamic_lambda` is false)
/// * `alpha` - Dynamic lambda multiplier: `lambda_kb = alpha * E_kb`
/// * `use_dynamic_lambda` - Whether to use dynamic or fixed lambda
/// * `batch_proportion_cutoff` - Minimum `O[k,b] / N_b` to include a level
///
/// ### Returns
///
/// Corrected data (N x d)
#[allow(clippy::too_many_arguments)]
fn ridge_regression_correction_v2_single(
    z_orig: MatRef<f32>,
    r: MatRef<f32>,
    info: &BatchInfo,
    oe: &OEPair,
    lambda: f32,
    alpha: f32,
    use_dynamic_lambda: bool,
    batch_proportion_cutoff: f32,
) -> Mat<f32> {
    let n = z_orig.nrows();
    let d = z_orig.ncols();
    let k = r.nrows();
    let b = info.n_levels;

    // Step 1: weighted segmented sum S in row-major [b * k * d], single cell sweep.
    let s: Vec<f32> = (0..n)
        .into_par_iter()
        .fold(
            || vec![0.0f32; b * k * d],
            |mut acc, cell| {
                let level = info.cell_to_level[cell];
                for cluster in 0..k {
                    let r_val = r[(cluster, cell)];
                    let base = (level * k + cluster) * d;
                    for feat in 0..d {
                        acc[base + feat] += r_val * z_orig[(cell, feat)];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f32; b * k * d],
            |mut a, b_local| {
                for i in 0..a.len() {
                    a[i] += b_local[i];
                }
                a
            },
        );

    let r_sum: Vec<f32> = (0..k)
        .map(|cluster| (0..b).map(|level| oe.o[(cluster, level)]).sum::<f32>())
        .collect();

    // Step 2: per-cluster arrowhead solve (parallel; tiny systems).
    type Solution = (Vec<usize>, Mat<f32>);
    let solutions: Vec<Option<Solution>> = (0..k)
        .into_par_iter()
        .map(|cluster| {
            let mut passing: Vec<usize> = Vec::new();
            for level in 0..b {
                let n_cells = info.batch_indices[level].len();
                if n_cells == 0 {
                    continue;
                }
                let avg_r = oe.o[(cluster, level)] / n_cells as f32;
                if avg_r > batch_proportion_cutoff {
                    passing.push(level);
                }
            }
            if passing.len() <= 1 {
                return None;
            }

            let p = 1 + passing.len();
            let mut design = Mat::<f32>::zeros(p, p);
            let mut phi_z = Mat::<f32>::zeros(p, d);

            // Intercept: r_sum at (0,0); phi_z[0] = sum_l S[l, cluster, :]
            design[(0, 0)] = r_sum[cluster];
            for level in 0..b {
                let base = (level * k + cluster) * d;
                for feat in 0..d {
                    phi_z[(0, feat)] += s[base + feat];
                }
            }

            // Arrowhead arms for passing levels
            for (col_off, &level) in passing.iter().enumerate() {
                let cc = 1 + col_off;
                let o_val = oe.o[(cluster, level)];
                design[(0, cc)] = o_val;
                design[(cc, 0)] = o_val;
                design[(cc, cc)] = o_val;
                let base = (level * k + cluster) * d;
                for feat in 0..d {
                    phi_z[(cc, feat)] = s[base + feat];
                }
            }

            // Ridge penalty on batch diagonal (intercept unpenalised)
            if use_dynamic_lambda {
                for (col_off, &level) in passing.iter().enumerate() {
                    let e_val = oe.e[(cluster, level)];
                    design[(1 + col_off, 1 + col_off)] += alpha * e_val;
                }
            } else {
                for i in 1..p {
                    design[(i, i)] += lambda;
                }
            }

            let w = solve_arrowhead(&design, &phi_z).unwrap_or_else(|| solve_lu(&design, &phi_z));

            Some((passing, w))
        })
        .collect();

    // Per-cluster level -> column lookup for the apply step.
    let lookups: Vec<Vec<i32>> = solutions
        .iter()
        .map(|sol| {
            let mut lookup = vec![-1i32; b];
            if let Some((passing, _)) = sol {
                for (pos, &level) in passing.iter().enumerate() {
                    lookup[level] = (1 + pos) as i32;
                }
            }
            lookup
        })
        .collect();

    // Step 3: apply correction, parallel over cells.
    let mut out = vec![0.0f32; n * d];
    out.par_chunks_mut(d).enumerate().for_each(|(cell, row)| {
        for feat in 0..d {
            row[feat] = z_orig[(cell, feat)];
        }
        let level = info.cell_to_level[cell];
        for cluster in 0..k {
            let c = lookups[cluster][level];
            if c >= 0
                && let Some((_, w)) = &solutions[cluster]
            {
                let r_val = r[(cluster, cell)];
                let c = c as usize;
                for feat in 0..d {
                    row[feat] -= r_val * w[(c, feat)];
                }
            }
        }
    });

    Mat::from_fn(n, d, |i, j| out[i * d + j])
}

/// Multi-covariate fallback.
///
/// Builds a joint design matrix across all covariates per cluster. Active
/// columns span every qualifying level from every variable; off-diagonal
/// covariance blocks are filled in a single cell sweep. Systems with exactly
/// one active variable are solved via arrowhead inversion; all others use LU.
///
/// ### Params
///
/// * `z_orig` - Original (uncorrected) data (N x d)
/// * `r` - Soft assignments (K x N)
/// * `batch_infos` - Batch information per variable
/// * `oe_pairs` - Observed/expected pairs per variable (used for pruning
///   and dynamic lambda)
/// * `lambda` - Fixed ridge penalty (used when `use_dynamic_lambda` is false)
/// * `alpha` - Dynamic lambda multiplier: `lambda_kb = alpha * E_kb`
/// * `use_dynamic_lambda` - Whether to use dynamic or fixed lambda
/// * `batch_proportion_cutoff` - Minimum `O[k,b] / N_b` to include a level
///
/// ### Returns
///
/// Corrected data (N x d)
#[allow(clippy::too_many_arguments)]
fn ridge_regression_correction_v2_multi(
    z_orig: MatRef<f32>,
    r: MatRef<f32>,
    batch_infos: &[BatchInfo],
    oe_pairs: &[OEPair],
    lambda: f32,
    alpha: f32,
    use_dynamic_lambda: bool,
    batch_proportion_cutoff: f32,
) -> Mat<f32> {
    let n = z_orig.nrows();
    let d = z_orig.ncols();
    let k = r.nrows();
    let n_vars = batch_infos.len();

    let o_refs: Vec<MatRef<f32>> = oe_pairs.iter().map(|p| p.o.as_ref()).collect();
    let e_refs: Vec<MatRef<f32>> = oe_pairs.iter().map(|p| p.e.as_ref()).collect();

    type Solution = (Vec<Vec<i32>>, Mat<f32>);
    let solutions: Vec<Option<Solution>> = (0..k)
        .into_par_iter()
        .map(|cluster_idx| {
            let mut col_map: Vec<(usize, usize)> = Vec::new();
            let mut level_to_col: Vec<Vec<i32>> = batch_infos
                .iter()
                .map(|info| vec![-1i32; info.n_levels])
                .collect();
            let mut n_active_vars = 0usize;

            for (var_idx, info) in batch_infos.iter().enumerate() {
                let mut passing: Vec<usize> = Vec::new();
                for level_idx in 0..info.n_levels {
                    let n_cells_level = info.batch_indices[level_idx].len();
                    if n_cells_level == 0 {
                        continue;
                    }
                    let avg_r = o_refs[var_idx][(cluster_idx, level_idx)] / n_cells_level as f32;
                    if avg_r > batch_proportion_cutoff {
                        passing.push(level_idx);
                    }
                }
                if passing.len() > 1 {
                    n_active_vars += 1;
                    for &level_idx in &passing {
                        level_to_col[var_idx][level_idx] = (1 + col_map.len()) as i32;
                        col_map.push((var_idx, level_idx));
                    }
                }
            }

            if col_map.is_empty() {
                return None;
            }

            let p = 1 + col_map.len();
            let mut design_cov = Mat::<f32>::zeros(p, p);
            let mut phi_z = Mat::<f32>::zeros(p, d);
            let mut active: Vec<usize> = Vec::with_capacity(1 + n_vars);

            for cell_idx in 0..n {
                let r_val = r[(cluster_idx, cell_idx)];

                active.clear();
                active.push(0);
                for var_idx in 0..n_vars {
                    let level = batch_infos[var_idx].cell_to_level[cell_idx];
                    let c = level_to_col[var_idx][level];
                    if c >= 0 {
                        active.push(c as usize);
                    }
                }

                for &c in &active {
                    for feat in 0..d {
                        phi_z[(c, feat)] += r_val * z_orig[(cell_idx, feat)];
                    }
                }

                for (i, &ci) in active.iter().enumerate() {
                    for &cj in &active[i..] {
                        design_cov[(ci, cj)] += r_val;
                        if ci != cj {
                            design_cov[(cj, ci)] += r_val;
                        }
                    }
                }
            }

            if use_dynamic_lambda {
                for (col_offset, &(var_idx, level_idx)) in col_map.iter().enumerate() {
                    let e_val = e_refs[var_idx][(cluster_idx, level_idx)];
                    design_cov[(1 + col_offset, 1 + col_offset)] += alpha * e_val;
                }
            } else {
                for i in 1..p {
                    design_cov[(i, i)] += lambda;
                }
            }

            let w = if n_active_vars == 1 {
                solve_arrowhead(&design_cov, &phi_z)
                    .unwrap_or_else(|| solve_lu(&design_cov, &phi_z))
            } else {
                solve_lu(&design_cov, &phi_z)
            };

            Some((level_to_col, w))
        })
        .collect();

    let mut out = vec![0.0f32; n * d];
    out.par_chunks_mut(d)
        .enumerate()
        .for_each(|(cell_idx, row)| {
            for feat in 0..d {
                row[feat] = z_orig[(cell_idx, feat)];
            }
            for cluster_idx in 0..k {
                if let Some((level_to_col, w)) = &solutions[cluster_idx] {
                    let r_val = r[(cluster_idx, cell_idx)];
                    for var_idx in 0..n_vars {
                        let level = batch_infos[var_idx].cell_to_level[cell_idx];
                        let c = level_to_col[var_idx][level];
                        if c >= 0 {
                            let c = c as usize;
                            for feat in 0..d {
                                row[feat] -= r_val * w[(c, feat)];
                            }
                        }
                    }
                }
            }
        });

    Mat::from_fn(n, d, |i, j| out[i * d + j])
}

/// Ridge regression correction with batch pruning, arrowhead optimisation, and
/// optional dynamic lambda.
///
/// The K per-cluster regressions are independent and are solved in parallel.
/// Each cluster prunes low-occupancy levels, builds a reduced design matrix
/// from the qualifying levels, and solves via arrowhead inversion (single
/// active covariate) or LU. The batch corrections are then subtracted in a
/// single pass that is parallel over cells, summing every cluster's
/// contribution per cell. A per-variable `level -> column` lookup avoids
/// rescanning the column map for every cell.
///
/// ### Params
///
/// * `z_orig` - Original (uncorrected) data (N x d)
/// * `r` - Soft assignments (K x N)
/// * `batch_infos` - Batch information per variable
/// * `oe_pairs` - Observed/expected pairs per variable (used for pruning
///   and dynamic lambda)
/// * `lambda` - Fixed ridge penalty (used when `use_dynamic_lambda` is false)
/// * `alpha` - Dynamic lambda multiplier: `lambda_kb = alpha * E_kb`
/// * `use_dynamic_lambda` - Whether to use dynamic or fixed lambda
/// * `batch_proportion_cutoff` - Minimum `O[k,b] / N_b` to include a level
///
/// ### Returns
///
/// Corrected data (N x d)
#[allow(clippy::too_many_arguments)]
pub fn ridge_regression_correction_v2(
    z_orig: MatRef<f32>,
    r: MatRef<f32>,
    batch_infos: &[BatchInfo],
    oe_pairs: &[OEPair],
    lambda: f32,
    alpha: f32,
    use_dynamic_lambda: bool,
    batch_proportion_cutoff: f32,
) -> Mat<f32> {
    if batch_infos.len() == 1 {
        ridge_regression_correction_v2_single(
            z_orig,
            r,
            &batch_infos[0],
            &oe_pairs[0],
            lambda,
            alpha,
            use_dynamic_lambda,
            batch_proportion_cutoff,
        )
    } else {
        ridge_regression_correction_v2_multi(
            z_orig,
            r,
            batch_infos,
            oe_pairs,
            lambda,
            alpha,
            use_dynamic_lambda,
            batch_proportion_cutoff,
        )
    }
}

//////////////////
// Harmony (v2) //
//////////////////

/// Run Harmony v2 batch correction.
///
/// The outer loop alternates between diversity-penalised soft clustering
/// (with fixed distances per round) and batch-pruned ridge regression.
///
/// ### Params
///
/// * `pca` - PCA embedding (N x d)
/// * `batch_labels` - one label slice per variable, each of length N
/// * `params` - Harmony v2 hyperparameters
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Corrected PCA embedding (N x d)
pub fn harmony_v2_with_state(
    pca: MatRef<f32>,
    batch_labels: &[Vec<usize>],
    params: &HarmonyParamsV2,
    seed: usize,
    verbose: usize,
) -> Result<HarmonyResult, BixverseErrors> {
    let start = Instant::now();

    let verbosity = parse_verbosity_level(verbose);

    let n = pca.nrows();
    let d = pca.ncols();
    let n_vars = batch_labels.len();

    assert!(n_vars >= 1, "At least one batch variable required");

    let batch_infos = create_batch_infos(batch_labels, n)?;

    if verbosity.normal_verbosity() {
        println!(
            "Harmony v2: {} cells, {} dims, {} variable(s), {} clusters",
            n.separate_with_underscores(),
            d,
            n_vars,
            params.k
        );
        for (v, info) in batch_infos.iter().enumerate() {
            println!("  Variable {}: {} levels", v, info.n_levels);
        }
    }

    let sigma = if params.sigma.len() == 1 {
        vec![params.sigma[0]; params.k]
    } else {
        assert_eq!(params.sigma.len(), params.k, "sigma must be length 1 or K");
        params.sigma.clone()
    };

    let theta = if params.theta.len() == 1 {
        vec![params.theta[0]; n_vars]
    } else {
        assert_eq!(
            params.theta.len(),
            n_vars,
            "theta must be length 1 or n_variables"
        );
        params.theta.clone()
    };

    let theta_expanded = expand_theta(&theta, &batch_infos, params.k, params.tau);
    let lambda_scalar = params.lambda[0];

    if verbosity.normal_verbosity() && params.tau > 0.0 {
        for (var_idx, levels) in theta_expanded.iter().enumerate() {
            println!(
                "  Theta (var {}): min={:.4}, max={:.4}",
                var_idx,
                levels.iter().cloned().fold(f32::INFINITY, f32::min),
                levels.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
    }

    let z_orig = pca.to_owned();
    let mut z_cos = cosine_normalise(&z_orig);

    if verbosity.normal_verbosity() {
        println!(" Initial data preparation done in {:.2?}", start.elapsed());
        println!("Running initial k-means...");
    }

    let mut y = run_kmeans_cosine(
        z_cos.as_ref(),
        params.k,
        params.kmeans_params,
        seed,
        verbose,
    )?;

    let mut dist_mat = compute_cosine_distances(y.as_ref(), z_cos.as_ref());
    let mut r = initialise_r_from_dist(dist_mat.as_ref(), &sigma)?;
    let mut oe_pairs = compute_all_diversity_statistics(r.as_ref(), &batch_infos);

    let initial_obj = compute_objective_v2(
        r.as_ref(),
        dist_mat.as_ref(),
        &oe_pairs,
        &sigma,
        &theta_expanded,
        &batch_infos,
    );

    let mut objectives_kmeans: Vec<f32> = vec![initial_obj];
    let mut objectives_harmony: Vec<f32> = vec![initial_obj];
    let mut z_corr = pca.to_owned();

    if verbosity.normal_verbosity() {
        println!("Initial objective: {:.4}", initial_obj);
    }

    for harmony_iter in 0..params.max_iter_harmony {
        if verbosity.normal_verbosity() {
            println!("\n=== Harmony v2 iteration {} ===", harmony_iter + 1);
            println!("  Running k-means clustering...");
        }

        let start_iter = Instant::now();

        // distances are fixed across the inner loop, so the base assignments
        // are computed once per round here rather than inside each update.
        let scale_dist = compute_scaled_distances(dist_mat.as_ref(), &sigma)?;

        // inner loop: refine R with diversity penalty, distances fixed
        for kmeans_iter in 0..params.max_iter_kmeans {
            let (r_new, oe_new) = update_r_with_diversity_v2(
                scale_dist.as_ref(),
                &theta_expanded,
                &batch_infos,
                params.block_size,
                seed + harmony_iter * 1000 + kmeans_iter,
                r.as_ref(),
                &oe_pairs,
            );

            r = r_new;
            oe_pairs = oe_new;

            let obj = compute_objective_v2(
                r.as_ref(),
                dist_mat.as_ref(),
                &oe_pairs,
                &sigma,
                &theta_expanded,
                &batch_infos,
            );
            objectives_kmeans.push(obj);

            if verbosity.detailed_verbosity() {
                println!("  K-means iter {}: obj = {:.4}", kmeans_iter + 1, obj);
            }

            if kmeans_iter >= params.window_size
                && check_convergence(
                    &objectives_kmeans,
                    params.window_size,
                    params.epsilon_kmeans,
                )
            {
                if verbosity.detailed_verbosity() {
                    println!("  K-means converged at iteration {}", kmeans_iter + 1);
                }
                break;
            }
        }

        // ridge regression with batch pruning
        if verbosity.normal_verbosity() {
            println!("  Applying ridge regression correction...");
        }

        z_corr = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r.as_ref(),
            &batch_infos,
            &oe_pairs,
            lambda_scalar,
            params.alpha,
            params.use_dynamic_lambda,
            params.batch_proportion_cutoff,
        );

        z_cos = cosine_normalise(&z_corr);

        // update centroids and distances for next round
        y = update_centroids_from_r(z_cos.as_ref(), r.as_ref());
        dist_mat = compute_cosine_distances(y.as_ref(), z_cos.as_ref());

        // re-initialise R from new distances (diversity applied in next inner loop)
        r = initialise_r_from_dist(dist_mat.as_ref(), &sigma)?;
        oe_pairs = compute_all_diversity_statistics(r.as_ref(), &batch_infos);

        let harmony_obj = *objectives_kmeans.last().unwrap();
        objectives_harmony.push(harmony_obj);

        if verbosity.normal_verbosity() {
            println!("  Harmony objective: {:.4}", harmony_obj);
            println!(
                "   Finished iteration in {:.2?} / Total runtime {:.2?}",
                start_iter.elapsed(),
                start.elapsed()
            );
        }

        if harmony_iter >= 1 {
            let obj_old = objectives_harmony[objectives_harmony.len() - 2];
            let obj_new = objectives_harmony[objectives_harmony.len() - 1];
            let rel_change = (obj_old - obj_new) / obj_old.abs();
            if rel_change < params.epsilon_harmony {
                if verbosity.normal_verbosity() {
                    println!("\nHarmony v2 converged at iteration {}", harmony_iter + 1);
                }
                break;
            }
        }
    }

    if verbosity.normal_verbosity() {
        println!(" Finished Harmony {:.2?}", start.elapsed());
    }

    Ok(HarmonyResult { z_corr, r })
}

/// Run Harmony v2 batch correction.
///
/// The outer loop alternates between diversity-penalised soft clustering
/// (with fixed distances per round) and batch-pruned ridge regression.
///
/// ### Params
///
/// * `pca` - PCA embedding (N x d)
/// * `batch_labels` - one label slice per variable, each of length N
/// * `params` - Harmony v2 hyperparameters
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Corrected PCA embedding (N x d)
pub fn harmony_v2(
    pca: MatRef<f32>,
    batch_labels: &[Vec<usize>],
    params: &HarmonyParamsV2,
    seed: usize,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors> {
    Ok(harmony_v2_with_state(pca, batch_labels, params, seed, verbose)?.z_corr)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::mat;
    use std::slice::from_ref;

    use super::super::harmony::{compute_diversity_statistics, create_batch_info};

    /// With tau zero every level of a variable keeps the raw theta.
    #[test]
    fn test_expand_theta_tau_zero() {
        let labels = vec![0, 0, 1, 1, 1];
        let info = create_batch_info(&labels, 5).unwrap();
        let result = expand_theta(&[2.0], &[info], 10, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        assert_relative_eq!(result[0][0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result[0][1], 2.0, epsilon = 1e-6);
    }

    /// A positive tau damps theta for small levels, so the bigger batch gets more penalty.
    #[test]
    fn test_expand_theta_tau_positive() {
        let labels = vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let info = create_batch_info(&labels, 10).unwrap();
        let result = expand_theta(&[2.0], &[info], 5, 5.0);

        assert!(
            result[0][0] < result[0][1],
            "Larger batch should get higher theta: {} vs {}",
            result[0][0],
            result[0][1]
        );
        assert!(result[0][0] < 2.0);
        assert!(result[0][1] < 2.0);
        assert!(result[0][0] > 0.0);
    }

    /// Each variable gets a theta vector as long as its own level count.
    #[test]
    fn test_expand_theta_multiple_variables() {
        let labels0 = vec![0, 0, 1, 1];
        let labels1 = vec![0, 1, 2, 0];
        let info0 = create_batch_info(&labels0, 4).unwrap();
        let info1 = create_batch_info(&labels1, 4).unwrap();
        let result = expand_theta(&[2.0, 3.0], &[info0, info1], 10, 0.0);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[1].len(), 3);
        assert_relative_eq!(result[0][0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result[1][0], 3.0, epsilon = 1e-6);
    }

    /// Too few objective values to fill the window means not converged, not a panic.
    #[test]
    fn test_check_convergence_too_few_values() {
        assert!(!check_convergence(&[1.0, 2.0], 3, 1e-5));
    }

    /// An objective trace that flattens over the window counts as converged.
    #[test]
    fn test_check_convergence_converged() {
        let vals = vec![100.0, 99.5, 99.0, 98.99, 98.98, 98.97];
        assert!(check_convergence(&vals, 3, 0.01));
    }

    /// A steadily falling objective is not treated as converged.
    #[test]
    fn test_check_convergence_not_converged() {
        let vals = vec![100.0, 90.0, 80.0, 70.0, 60.0, 50.0];
        assert!(!check_convergence(&vals, 3, 0.01));
    }

    /// The stabilised objective returns a finite number on ordinary input.
    #[test]
    fn test_objective_v2_finite_and_positive() {
        let labels = vec![0, 0, 1];
        let info = create_batch_info(&labels, 3).unwrap();
        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![1.0, 1.0]];

        let r = mat![[0.8, 0.7, 0.2], [0.2, 0.3, 0.8]];
        let dist_mat = mat![[0.1, 0.2, 0.9], [0.9, 0.8, 0.1]];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        let obj = compute_objective_v2(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe,
            &sigma,
            &theta_expanded,
            from_ref(&info),
        );

        assert!(obj.is_finite());
        assert!(!obj.is_nan());
    }

    /// A confident, well-separated R scores lower than a maximally uncertain one.
    #[test]
    fn test_objective_v2_decreases_with_better_assignments() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4).unwrap();
        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![1.0, 1.0]];

        let r_uncertain = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]];
        let dist_mat_high = mat![[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]];
        let oe1 = compute_all_diversity_statistics(r_uncertain.as_ref(), from_ref(&info));
        let obj1 = compute_objective_v2(
            r_uncertain.as_ref(),
            dist_mat_high.as_ref(),
            &oe1,
            &sigma,
            &theta_expanded,
            from_ref(&info),
        );

        let r_confident = mat![[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]];
        let dist_mat_low = mat![[0.1, 0.1, 1.0, 1.0], [1.0, 1.0, 0.1, 0.1]];
        let oe2 = compute_all_diversity_statistics(r_confident.as_ref(), from_ref(&info));
        let obj2 = compute_objective_v2(
            r_confident.as_ref(),
            dist_mat_low.as_ref(),
            &oe2,
            &sigma,
            &theta_expanded,
            from_ref(&info),
        );

        assert!(
            obj2 < obj1,
            "Confident R should have lower objective: {} vs {}",
            obj2,
            obj1
        );
    }

    /// With theta zero the cross-entropy term drops out and only the k-means term is left.
    #[test]
    fn test_objective_v2_zero_theta_no_cross_entropy() {
        let labels = vec![0, 1];
        let info = create_batch_info(&labels, 2).unwrap();
        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![0.0, 0.0]];

        let r = mat![[1.0, 0.0], [0.0, 1.0]];
        let dist_mat = mat![[0.1, 0.9], [0.9, 0.1]];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        let obj = compute_objective_v2(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe,
            &sigma,
            &theta_expanded,
            from_ref(&info),
        );

        // with hard assignment and theta=0: only kmeans error, no entropy (r*ln(r) = 0 for r=0,1)
        let expected = 0.2 * 1000.0;
        assert!(
            (obj - expected).abs() < 1.0,
            "Expected ~{}, got {}",
            expected,
            obj
        );
    }

    /// A second covariate contributes its own term rather than being ignored.
    #[test]
    fn test_objective_v2_two_variables() {
        let labels0 = vec![0, 0, 1, 1];
        let labels1 = vec![0, 1, 1, 0]; // asymmetric w.r.t. clusters
        let info0 = create_batch_info(&labels0, 4).unwrap();
        let info1 = create_batch_info(&labels1, 4).unwrap();

        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![1.0, 1.0], vec![1.0, 1.0]];

        let r = mat![[0.8, 0.7, 0.2, 0.3], [0.2, 0.3, 0.8, 0.7]];
        let dist_mat = mat![[0.1, 0.2, 0.9, 0.8], [0.9, 0.8, 0.1, 0.2]];

        let oe = compute_all_diversity_statistics(r.as_ref(), &[info0.clone(), info1.clone()]);

        let obj = compute_objective_v2(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe,
            &sigma,
            &theta_expanded,
            &[info0.clone(), info1.clone()],
        );

        assert!(obj.is_finite());

        let oe_single = compute_all_diversity_statistics(r.as_ref(), std::slice::from_ref(&info0));
        let obj_single = compute_objective_v2(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe_single,
            &sigma,
            &[vec![1.0, 1.0]],
            std::slice::from_ref(&info0),
        );

        assert!(
            (obj - obj_single).abs() > 1e-6,
            "Two variables should differ from one: {} vs {}",
            obj,
            obj_single,
        );
    }

    /// Regression: the v1 cross-entropy blew up as O_kb approached zero; v2 stays finite.
    #[test]
    fn test_objective_v2_stabilised_near_zero_o() {
        // Scenario where v1 formula would be unstable: O_kb near zero
        let labels = vec![0, 0, 0, 0, 1];
        let info = create_batch_info(&labels, 5).unwrap();
        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![2.0, 2.0]];

        // Almost all mass in cluster 0 for batch 0 cells, nearly none for batch 1
        let r = mat![
            [0.99, 0.99, 0.99, 0.99, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.99],
        ];
        let dist_mat = mat![
            [0.01, 0.01, 0.01, 0.01, 0.99],
            [0.99, 0.99, 0.99, 0.99, 0.01],
        ];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        let obj = compute_objective_v2(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe,
            &sigma,
            &theta_expanded,
            from_ref(&info),
        );

        assert!(obj.is_finite(), "v2 objective should be stable: {}", obj);
        assert!(!obj.is_nan());
    }

    /// Columns stay normalised and the incrementally tracked O matches a fresh recomputation.
    #[test]
    fn test_update_r_v2_properties() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4).unwrap();
        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![1.0, 1.0]];

        let r_init = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]];
        let dist_mat = mat![[0.1, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1]];
        let oe_init = compute_all_diversity_statistics(r_init.as_ref(), from_ref(&info));
        let scale_dist = compute_scaled_distances(dist_mat.as_ref(), &sigma).unwrap();

        let (r_new, oe_new) = update_r_with_diversity_v2(
            scale_dist.as_ref(),
            &theta_expanded,
            from_ref(&info),
            0.5,
            42,
            r_init.as_ref(),
            &oe_init,
        );

        // Columns are a distribution over clusters.
        for cell_idx in 0..4 {
            let col_sum: f32 = (0..2).map(|k| r_new[(k, cell_idx)]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-5,
                "Column {} sum: {}",
                cell_idx,
                col_sum
            );
        }

        // Each cell moves towards its nearer centroid.
        assert!(r_new[(0, 0)] > 0.5);
        assert!(r_new[(0, 1)] > 0.5);
        assert!(r_new[(1, 2)] > 0.5);
        assert!(r_new[(1, 3)] > 0.5);

        // The incrementally tracked O matches a recomputation from R.
        let oe_check = compute_all_diversity_statistics(r_new.as_ref(), from_ref(&info));
        let OEPair { o: o_new, .. } = &oe_new[0];
        let OEPair { o: o_check, .. } = &oe_check[0];

        for k in 0..2 {
            for b in 0..2 {
                assert!(
                    (o_new[(k, b)] - o_check[(k, b)]).abs() < 1e-4,
                    "O mismatch at [{},{}]: {} vs {}",
                    k,
                    b,
                    o_new[(k, b)],
                    o_check[(k, b)]
                );
            }
        }
    }

    /// With theta zero the update follows the scaled distances alone.
    #[test]
    fn test_update_r_v2_no_diversity_penalty() {
        let labels = vec![0, 0];
        let info = create_batch_info(&labels, 2).unwrap();
        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![0.0]];

        let r_init = mat![[0.5, 0.5], [0.5, 0.5]];
        let dist_mat = mat![[0.1, 0.9], [0.9, 0.1]];
        let oe_init = compute_all_diversity_statistics(r_init.as_ref(), from_ref(&info));
        let scale_dist = compute_scaled_distances(dist_mat.as_ref(), &sigma).unwrap();

        let (r_new, _) = update_r_with_diversity_v2(
            scale_dist.as_ref(),
            &theta_expanded,
            from_ref(&info),
            1.0,
            42,
            r_init.as_ref(),
            &oe_init,
        );

        assert!(r_new[(0, 0)] > 0.6);
        assert!(r_new[(1, 1)] > 0.6);
    }

    /// With two covariates both tracked O blocks stay consistent with the updated R.
    #[test]
    fn test_update_r_v2_two_variables() {
        let labels0 = vec![0, 0, 1, 1];
        let labels1 = vec![0, 1, 0, 1];
        let info0 = create_batch_info(&labels0, 4).unwrap();
        let info1 = create_batch_info(&labels1, 4).unwrap();
        let sigma = vec![1.0, 1.0];
        let theta_expanded = vec![vec![1.0, 1.0], vec![1.0, 1.0]];

        let r_init = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]];
        let dist_mat = mat![[0.1, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1]];
        let oe_init =
            compute_all_diversity_statistics(r_init.as_ref(), &[info0.clone(), info1.clone()]);
        let scale_dist = compute_scaled_distances(dist_mat.as_ref(), &sigma).unwrap();

        let (r_new, oe_new) = update_r_with_diversity_v2(
            scale_dist.as_ref(),
            &theta_expanded,
            &[info0.clone(), info1.clone()],
            0.5,
            42,
            r_init.as_ref(),
            &oe_init,
        );

        for cell_idx in 0..4 {
            let col_sum: f32 = (0..2).map(|k| r_new[(k, cell_idx)]).sum();
            assert!((col_sum - 1.0).abs() < 1e-5);
        }

        assert_eq!(oe_new.len(), 2);
        for (var_idx, info) in [&info0, &info1].iter().enumerate() {
            let OEPair { o: o_new, .. } = &oe_new[var_idx];
            let OEPair { o: o_check, .. } = compute_diversity_statistics(r_new.as_ref(), info);
            for k in 0..2 {
                for b in 0..info.n_levels {
                    assert!(
                        (o_new[(k, b)] - o_check[(k, b)]).abs() < 1e-4,
                        "Var {} O mismatch at [{},{}]",
                        var_idx,
                        k,
                        b
                    );
                }
            }
        }
    }

    /// The fast arrowhead solve agrees with a general LU solve on arrowhead input.
    #[test]
    fn test_arrowhead_matches_lu() {
        // Build a design_cov that has arrowhead structure:
        // first row/col is dense, rest is diagonal
        let p = 4;
        let d = 3;
        let mut design_cov = Mat::<f32>::zeros(p, p);
        design_cov[(0, 0)] = 10.0;
        for i in 1..p {
            design_cov[(0, i)] = 1.0 + i as f32 * 0.5;
            design_cov[(i, 0)] = design_cov[(0, i)];
            design_cov[(i, i)] = 5.0 + i as f32;
        }

        let phi_z = Mat::from_fn(p, d, |i, j| (i * d + j) as f32 * 0.1 + 1.0);

        let w_arrow = solve_arrowhead(&design_cov, &phi_z).expect("should succeed");
        let w_lu = solve_lu(&design_cov, &phi_z);

        for i in 0..p {
            for j in 0..d {
                assert!(
                    (w_arrow[(i, j)] - w_lu[(i, j)]).abs() < 1e-3,
                    "Mismatch at [{},{}]: arrow={}, lu={}",
                    i,
                    j,
                    w_arrow[(i, j)],
                    w_lu[(i, j)]
                );
            }
        }
    }

    /// A zero on the diagonal makes the arrowhead solve degenerate, so it declines rather than divides.
    #[test]
    fn test_arrowhead_degenerate_returns_none() {
        let p = 3;
        let d = 2;
        let mut design_cov = Mat::<f32>::zeros(p, p);
        design_cov[(0, 0)] = 1.0;
        // diagonal entry at [1,1] is 0 -> degenerate
        design_cov[(2, 2)] = 1.0;

        let phi_z = Mat::from_fn(p, d, |i, j| (i + j) as f32);
        assert!(solve_arrowhead(&design_cov, &phi_z).is_none());
    }

    /// A variable left with one surviving level after pruning has nothing to correct.
    #[test]
    fn test_ridge_v2_batch_pruning_skips_single_level() {
        // If only one level passes the cutoff for a variable, that variable
        // should be skipped entirely (nothing to correct).
        let labels = vec![0, 0, 0, 0, 1]; // level 1 has only 1 cell
        let info = create_batch_info(&labels, 5).unwrap();

        let z_orig = mat![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [5.0, 0.0],];

        // Nearly all mass assigned to cluster 0 for all cells
        let r = mat![
            [0.99, 0.99, 0.99, 0.99, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.99],
        ];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        // With very high cutoff, level 1 in cluster 0 gets pruned (O[0,1]/N_1 = 0.01/1 = 0.01)
        // With cutoff 0.5, only level 0 passes -> single level -> skip
        let z_corr = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            &oe,
            0.01,
            0.2,
            false,
            0.5,
        );

        // Cluster 0 should have no correction because only one level qualifies
        for i in 0..4 {
            assert!(
                (z_corr[(i, 0)] - z_orig[(i, 0)]).abs() < 0.1,
                "Cell {} should be mostly unchanged: {} vs {}",
                i,
                z_corr[(i, 0)],
                z_orig[(i, 0)]
            );
        }
    }

    /// Soft cluster memberships still shrink the batch effect.
    #[test]
    fn test_ridge_v2_soft_assignments() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4).unwrap();

        let z_orig = mat![[1.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.0, 0.0]];
        let r = mat![[0.8, 0.9, 0.1, 0.2], [0.2, 0.1, 0.9, 0.8]];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        let z_corr = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            &oe,
            0.01,
            0.2,
            false,
            1e-5,
        );

        let orig_diff = ((z_orig[(2, 0)] + z_orig[(3, 0)]) / 2.0
            - (z_orig[(0, 0)] + z_orig[(1, 0)]) / 2.0)
            .abs();
        let corr_diff = ((z_corr[(2, 0)] + z_corr[(3, 0)]) / 2.0
            - (z_corr[(0, 0)] + z_corr[(1, 0)]) / 2.0)
            .abs();

        assert!(
            corr_diff < orig_diff,
            "Soft assignments should still reduce batch effect: {} vs {}",
            orig_diff,
            corr_diff
        );
    }

    /// Dynamic lambda scales with E_kb, so its fit differs from the fixed-lambda one.
    #[test]
    fn test_ridge_v2_dynamic_lambda() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4).unwrap();

        let z_orig = mat![[1.0, 0.1], [1.1, 0.2], [5.0, 0.1], [5.1, 0.2]];
        let r = mat![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        let z_fixed = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            &oe,
            1.0,
            0.2,
            false,
            1e-5,
        );

        let z_dynamic = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            &oe,
            1.0,
            0.2,
            true,
            1e-5,
        );

        // They should differ since dynamic lambda uses alpha * E_kb
        let mut any_diff = false;
        for i in 0..4 {
            for j in 0..2 {
                if (z_fixed[(i, j)] - z_dynamic[(i, j)]).abs() > 1e-4 {
                    any_diff = true;
                }
            }
        }
        assert!(
            any_diff,
            "Dynamic and fixed lambda should produce different results"
        );
    }

    /// Two covariates each get their own effect removed from their own feature.
    #[test]
    fn test_ridge_v2_two_variables() {
        let batch_labels = vec![0, 0, 1, 1, 2, 2];
        let sample_labels = vec![0, 1, 0, 1, 0, 1];
        let info_batch = create_batch_info(&batch_labels, 6).unwrap();
        let info_sample = create_batch_info(&sample_labels, 6).unwrap();

        let z_orig = mat![
            [1.0, 0.0],
            [1.0, 3.0],
            [5.0, 0.0],
            [5.0, 3.0],
            [9.0, 0.0],
            [9.0, 3.0],
        ];

        let r = mat![
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let infos = [info_batch.clone(), info_sample.clone()];
        let oe = compute_all_diversity_statistics(r.as_ref(), &infos);

        let z_corr = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r.as_ref(),
            &infos,
            &oe,
            0.01,
            0.2,
            false,
            1e-5,
        );

        // Batch effect in feature 0 should be reduced
        let batch_means_orig: Vec<f32> = (0..3)
            .map(|b| {
                let cells = &info_batch.batch_indices[b];
                cells.iter().map(|&c| z_orig[(c, 0)]).sum::<f32>() / cells.len() as f32
            })
            .collect();
        let batch_means_corr: Vec<f32> = (0..3)
            .map(|b| {
                let cells = &info_batch.batch_indices[b];
                cells.iter().map(|&c| z_corr[(c, 0)]).sum::<f32>() / cells.len() as f32
            })
            .collect();

        let orig_spread = batch_means_orig[2] - batch_means_orig[0];
        let corr_spread = (batch_means_corr[2] - batch_means_corr[0]).abs();
        assert!(
            corr_spread < orig_spread,
            "Batch effect should be reduced: {} vs {}",
            orig_spread,
            corr_spread
        );

        // Sample effect in feature 1 should be reduced
        let sample_means_orig: Vec<f32> = (0..2)
            .map(|s| {
                let cells = &info_sample.batch_indices[s];
                cells.iter().map(|&c| z_orig[(c, 1)]).sum::<f32>() / cells.len() as f32
            })
            .collect();
        let sample_means_corr: Vec<f32> = (0..2)
            .map(|s| {
                let cells = &info_sample.batch_indices[s];
                cells.iter().map(|&c| z_corr[(c, 1)]).sum::<f32>() / cells.len() as f32
            })
            .collect();

        let orig_sample_diff = (sample_means_orig[1] - sample_means_orig[0]).abs();
        let corr_sample_diff = (sample_means_corr[1] - sample_means_corr[0]).abs();
        assert!(
            corr_sample_diff < orig_sample_diff,
            "Sample effect should be reduced: {} vs {}",
            orig_sample_diff,
            corr_sample_diff
        );
    }

    /// When every level is pruned the embedding comes back untouched.
    #[test]
    fn test_ridge_v2_no_correction_when_all_pruned() {
        let labels = vec![0, 1];
        let info = create_batch_info(&labels, 2).unwrap();

        let z_orig = mat![[1.0, 2.0], [5.0, 6.0]];

        // Uniform assignment
        let r = mat![[0.5, 0.5], [0.5, 0.5]];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        // Cutoff so high that everything gets pruned
        let z_corr = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            &oe,
            1.0,
            0.2,
            false,
            100.0, // absurdly high cutoff
        );

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(z_corr[(i, j)], z_orig[(i, j)], epsilon = 1e-6);
            }
        }
    }
}
