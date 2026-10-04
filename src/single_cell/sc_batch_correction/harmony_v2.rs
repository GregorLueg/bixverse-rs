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
use std::time::Instant;
use thousands::*;

use crate::ml::clustering::k_means::KMeansParamsWrappers;
use crate::prelude::parse_verbosity_level;
use crate::prelude::*;

use super::harmony::{BatchInfo, HarmonyResult, StageTimes, create_batch_infos};
use super::harmony_core::{
    RidgeSettings, Variant, base_error_and_entropy, distances_and_base, kmeans_centroids,
    normalise_rows_into, objective, observed_counts, r_to_mat, ridge_correction, row_major_to_mat,
    to_row_major, update_assignments, window_converged,
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

/// Windowed convergence check, as in R harmony v2.
///
/// Sums the last `window_size` objectives and the same window shifted back by
/// one, and returns true when `|old - new| / |old|` drops below `epsilon`.
///
/// ### Params
///
/// * `objectives` - Objective trace
/// * `window_size` - Number of values per window
/// * `epsilon` - Convergence threshold
///
/// ### Returns
///
/// Whether convergence is reached; false while the trace is shorter than
/// `window_size + 1`
pub fn check_convergence(objectives: &[f32], window_size: usize, epsilon: f32) -> bool {
    window_converged(objectives, window_size, epsilon, true)
}

///////////////////////////
// Ridge solvers (single) //
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

//////////////////
// Harmony (v2) //
//////////////////

/// Run Harmony v2 batch correction.
///
/// Follows R harmony v2 (Patikas et al.): each round refines the soft
/// assignments with the diversity penalty at fixed distances, then applies the
/// batch-pruned ridge correction. The next round's centroids are the
/// normalised ridge intercepts (clusters without a system keep theirs), and
/// the assignments restart from the diversity-free softmax of the new
/// distances. All state is held in flat cell-major buffers, see
/// `harmony_core`.
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
/// The [HarmonyResult]: corrected embedding (N x d) and the final soft
/// assignments (K x N)
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
    let k = params.k;
    let n_vars = batch_labels.len();

    assert!(n_vars >= 1, "At least one batch variable required");

    let batch_infos = create_batch_infos(batch_labels, n)?;

    if verbosity.normal_verbosity() {
        println!(
            "Harmony v2: {} cells, {} dims, {} variable(s), {} clusters",
            n.separate_with_underscores(),
            d,
            n_vars,
            k
        );
        for (v, info) in batch_infos.iter().enumerate() {
            println!("  Variable {}: {} levels", v, info.n_levels);
        }
    }

    let sigma = if params.sigma.len() == 1 {
        vec![params.sigma[0]; k]
    } else {
        assert_eq!(params.sigma.len(), k, "sigma must be length 1 or K");
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

    let theta_expanded = expand_theta(&theta, &batch_infos, k, params.tau);

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

    let ridge = RidgeSettings {
        lambda: params.lambda[0],
        alpha: params.alpha,
        dynamic_lambda: params.use_dynamic_lambda,
        prune_cutoff: Some(params.batch_proportion_cutoff),
    };

    let mut times = StageTimes::default();

    let z_orig = to_row_major(pca);
    let mut z_cos = vec![0.0f32; n * d];
    normalise_rows_into(&z_orig, d, &mut z_cos);
    let mut z_corr = z_orig.clone();

    if verbosity.normal_verbosity() {
        println!(" Initial data preparation done in {:.2?}", start.elapsed());
        println!("Running initial k-means...");
    }

    let mut y = times.time("kmeans_init", || {
        kmeans_centroids(
            &z_cos,
            d,
            k,
            params.kmeans_params,
            seed,
            verbosity.detailed_verbosity(),
        )
    })?;

    let mut log_base = vec![0.0f32; n * k];
    let mut shift = vec![0.0f32; n];
    let mut base = vec![0.0f32; n * k];
    let (mut r, mut o, mut r_sum) = times.time("init_r", || {
        distances_and_base(&z_cos, &y, &sigma, d, &mut log_base, &mut base, &mut shift);
        let (o, r_sum) = observed_counts(&base, &batch_infos, k);
        (base.clone(), o, r_sum)
    });

    let (error, entropy) = base_error_and_entropy(&log_base, &base, &shift, &sigma);
    let initial_obj = objective(
        Variant::V2,
        error,
        entropy,
        &o,
        &r_sum,
        &batch_infos,
        &sigma,
        &theta_expanded,
        n,
    );

    let mut objectives_kmeans: Vec<f32> = vec![initial_obj];
    let mut objectives_harmony: Vec<f32> = vec![initial_obj];

    if verbosity.normal_verbosity() {
        println!("Initial objective: {:.4}", initial_obj);
    }

    for harmony_iter in 0..params.max_iter_harmony {
        if verbosity.normal_verbosity() {
            println!("\n=== Harmony v2 iteration {} ===", harmony_iter + 1);
            println!("  Running k-means clustering...");
        }

        let start_iter = Instant::now();

        // inner loop: refine R with the diversity penalty, distances fixed
        for kmeans_iter in 0..params.max_iter_kmeans {
            let (error, entropy) = times.time("update_r", || {
                update_assignments(
                    Variant::V2,
                    &mut r,
                    &base,
                    &log_base,
                    &shift,
                    &sigma,
                    &theta_expanded,
                    &batch_infos,
                    &mut o,
                    &mut r_sum,
                    params.block_size,
                    (seed + harmony_iter * 1000 + kmeans_iter) as u64,
                )
            });

            let obj = objective(
                Variant::V2,
                error,
                entropy,
                &o,
                &r_sum,
                &batch_infos,
                &sigma,
                &theta_expanded,
                n,
            );
            objectives_kmeans.push(obj);

            if verbosity.detailed_verbosity() {
                println!("  K-means iter {}: obj = {:.4}", kmeans_iter + 1, obj);
            }

            if kmeans_iter > params.window_size
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

        if verbosity.normal_verbosity() {
            println!("  Applying ridge regression correction...");
        }

        let out = times.time("ridge", || {
            ridge_correction(&z_orig, &r, &batch_infos, ridge, k, d, &mut z_corr)
        });

        // next round's centroids are the normalised ridge intercepts
        let mut intercept_norm = vec![0.0f32; k * d];
        normalise_rows_into(&out.intercept, d, &mut intercept_norm);
        for (kk, &solved) in out.solved.iter().enumerate() {
            if solved {
                y[kk * d..(kk + 1) * d].copy_from_slice(&intercept_norm[kk * d..(kk + 1) * d]);
            }
        }

        // cold restart of R from the new distances
        times.time("init_r", || {
            normalise_rows_into(&z_corr, d, &mut z_cos);
            distances_and_base(&z_cos, &y, &sigma, d, &mut log_base, &mut base, &mut shift);
            r.copy_from_slice(&base);
            (o, r_sum) = observed_counts(&r, &batch_infos, k);
        });

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

        let obj_old = objectives_harmony[objectives_harmony.len() - 2];
        let rel_change = (obj_old - harmony_obj) / obj_old.abs();
        if rel_change < params.epsilon_harmony {
            if verbosity.normal_verbosity() {
                println!("\nHarmony v2 converged at iteration {}", harmony_iter + 1);
            }
            break;
        }
    }

    if verbosity.normal_verbosity() {
        println!(" Finished Harmony {:.2?}", start.elapsed());
    }
    if verbosity.detailed_verbosity() {
        times.print();
    }

    Ok(HarmonyResult {
        z_corr: row_major_to_mat(&z_corr, d),
        r: r_to_mat(&r, k),
    })
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

    use super::super::harmony::create_batch_info;

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
        assert!(result[0][0] < result[0][1]);
        assert!(result[0][0] < 2.0);
        assert!(result[0][1] < 2.0);
        assert!(result[0][0] > 0.0);
    }

    /// Each variable gets a theta vector as long as its own level count.
    #[test]
    fn test_expand_theta_multiple_variables() {
        let info0 = create_batch_info(&[0, 0, 1, 1], 4).unwrap();
        let info1 = create_batch_info(&[0, 1, 2, 0], 4).unwrap();
        let result = expand_theta(&[2.0, 3.0], &[info0, info1], 10, 0.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[1].len(), 3);
        assert_relative_eq!(result[1][0], 3.0, epsilon = 1e-6);
    }

    /// Too few objective values to fill the shifted window means not converged.
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

    /// The closed-form arrowhead solve agrees with LU on an arrowhead system.
    #[test]
    fn test_arrowhead_matches_lu() {
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
                assert!((w_arrow[(i, j)] - w_lu[(i, j)]).abs() < 1e-3);
            }
        }
    }

    /// A zero on the diagonal makes the arrowhead solve degenerate, so it declines rather than divides.
    #[test]
    fn test_arrowhead_degenerate_returns_none() {
        let mut design_cov = Mat::<f32>::zeros(3, 3);
        design_cov[(0, 0)] = 1.0;
        design_cov[(2, 2)] = 1.0;
        let phi_z = Mat::from_fn(3, 2, |i, j| (i + j) as f32);
        assert!(solve_arrowhead(&design_cov, &phi_z).is_none());
    }
}
