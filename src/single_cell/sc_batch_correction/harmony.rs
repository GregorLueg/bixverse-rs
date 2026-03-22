//! Implementation of the Harmony batch correction, please see Korsunsky, et
//! al., Nat Methods, 2019

use ann_search_rs::{utils::dist::Dist, utils::k_means_utils::train_centroids};
use faer::linalg::solvers::PartialPivLu;
use faer::{Mat, MatRef, linalg::solvers::DenseSolveCore};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use thousands::*;

use crate::single_cell::sc_batch_correction::batch_utils::cosine_normalise;
use crate::utils::matrix_utils::{flat_row_major_to_mat, mat_to_flat_row_major};

////////////
// Params //
////////////

/// Parameters for Harmony batch correction.
pub struct HarmonyParams {
    /// Number of clusters
    pub k: usize,
    /// Per-cluster diversity weights (length 1 or K)
    pub sigma: Vec<f32>,
    /// Per-variable diversity penalties (length 1 or n_variables)
    pub theta: Vec<f32>,
    /// Ridge penalty (length 1, broadcast to all design matrix columns)
    pub lambda: Vec<f32>,
    /// Fraction of cells to update per block (0.0-1.0)
    pub block_size: f32,
    /// Maximum k-means iterations per Harmony round
    pub max_iter_kmeans: usize,
    /// Maximum Harmony outer iterations
    pub max_iter_harmony: usize,
    /// K-means clustering convergence threshold
    pub epsilon_kmeans: f32,
    /// Harmony convergence threshold
    pub epsilon_harmony: f32,
    /// Window size for convergence checking
    pub window_size: usize,
}

/// Default implementation for HarmonyParams
impl Default for HarmonyParams {
    fn default() -> Self {
        Self {
            k: 10,
            sigma: vec![0.1],
            theta: vec![2.0],
            lambda: vec![1.0],
            block_size: 0.05,
            max_iter_kmeans: 20,
            max_iter_harmony: 10,
            epsilon_kmeans: 1e-5,
            epsilon_harmony: 1e-4,
            window_size: 3,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Observed and expected cluster-batch assignment counts for one batch
/// variable.
pub struct OEPair {
    /// Observed counts (K x B): sum of soft assignments per cluster per level
    pub o: Mat<f32>,
    /// Expected counts (K x B): expected assignments under uniform mixing
    pub e: Mat<f32>,
}

/// Batch information for a single categorical variable.
///
/// Holds the mapping from cells to levels, level frequencies, and
/// cell index lists per level for one batch variable (e.g. "sample",
/// "technology", "donor").
#[derive(Debug, Clone)]
pub struct BatchInfo {
    /// Cell indices per level (length n_levels)
    pub batch_indices: Vec<Vec<usize>>,
    /// Level frequencies (length n_levels)
    pub pr_b: Vec<f32>,
    /// Number of distinct levels
    pub n_levels: usize,
    /// For each cell, its level in this variable (length N)
    pub cell_to_level: Vec<usize>,
}

/// Create batch information from cell-level labels for a single variable.
///
/// ### Params
///
/// * `labels` - level assignment per cell (length N), values in 0..n_levels
/// * `n_cells` - number of cells
///
/// ### Returns
///
/// `BatchInfo` with level frequencies, cell indices per level, and
/// reverse lookup
pub fn create_batch_info(labels: &[usize], n_cells: usize) -> BatchInfo {
    assert_eq!(labels.len(), n_cells, "labels length must match n_cells");

    let n_levels = labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    let mut batch_indices: Vec<Vec<usize>> = vec![Vec::new(); n_levels];
    for (cell_idx, &level) in labels.iter().enumerate() {
        batch_indices[level].push(cell_idx);
    }

    let pr_b: Vec<f32> = batch_indices
        .iter()
        .map(|cells| cells.len() as f32 / n_cells as f32)
        .collect();

    BatchInfo {
        batch_indices,
        pr_b,
        n_levels,
        cell_to_level: labels.to_vec(),
    }
}

/// Create batch information for multiple categorical variables.
///
/// ### Params
///
/// * `all_labels` - one label slice per variable, each of length N
/// * `n_cells` - number of cells
///
/// ### Returns
///
/// Vec of `BatchInfo`, one per variable
pub fn create_batch_infos(all_labels: &[Vec<usize>], n_cells: usize) -> Vec<BatchInfo> {
    all_labels
        .iter()
        .map(|labels| create_batch_info(labels, n_cells))
        .collect()
}

/// Run k-means clustering on cosine-normalised data.
///
/// Integrates with ann_search_rs k-means implementation which expects flat
/// row-major layout for cache efficiency.
///
/// ### Params
///
/// * `data_cos` - Cosine-normalised data (N x d)
/// * `k` - Number of clusters
/// * `max_iter` - Maximum k-means iterations
/// * `seed` - Random seed
/// * `verbose` - Print progress
///
/// ### Returns
///
/// Cluster centroids (K x d), cosine-normalised
pub fn run_kmeans_cosine(
    data_cos: MatRef<f32>,
    k: usize,
    max_iter: usize,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let n = data_cos.nrows();
    let d = data_cos.ncols();

    if verbose {
        println!("Running k-means: {} cells, {} dims, {} clusters", n, d, k);
    }

    let data_flat = mat_to_flat_row_major(data_cos);
    let centroids_flat =
        train_centroids(&data_flat, d, n, k, &Dist::Cosine, max_iter, seed, verbose);
    let centroids = flat_row_major_to_mat(&centroids_flat, k, d);

    cosine_normalise(&centroids)
}

/// Compute cosine distances between centroids and data.
///
/// For cosine-normalised vectors: dist = 2 * (1 - dot_product)
///
/// ### Params
///
/// * `centroids` - Cluster centroids (K x d), must be cosine-normalised
/// * `data_cos` - Data matrix (N x d), must be cosine-normalised
///
/// ### Returns
///
/// Distance matrix (K x N)
pub fn compute_cosine_distances(centroids: MatRef<f32>, data_cos: MatRef<f32>) -> Mat<f32> {
    let k = centroids.nrows();
    let n = data_cos.nrows();
    let dot_products = centroids * data_cos.transpose();
    Mat::from_fn(k, n, |i, j| 2.0 * (1.0 - dot_products[(i, j)]))
}

/// Initialise soft cluster assignments from distances.
///
/// Converts distances to probabilistic cluster assignments using exponential
/// decay weighted by per-cluster sigma values. Each cell's assignments are
/// normalised to sum to 1.
///
/// ### Params
///
/// * `dist_mat` - Distance matrix (K x N)
/// * `sigma` - Per-cluster diversity weights (length K)
///
/// ### Returns
///
/// Soft assignment matrix R (K x N), columns sum to 1
pub fn initialise_r_from_dist(dist_mat: MatRef<f32>, sigma: &[f32]) -> Mat<f32> {
    let k = dist_mat.nrows();
    let n = dist_mat.ncols();
    assert_eq!(sigma.len(), k, "sigma length must match number of clusters");

    let columns: Vec<_> = (0..n)
        .into_par_iter()
        .map(|cell_idx| {
            let mut col_sum = 0.0f32;
            let mut col = vec![0.0f32; k];

            for cluster_idx in 0..k {
                let dist = dist_mat[(cluster_idx, cell_idx)];
                let val = (-dist / sigma[cluster_idx]).exp();
                col[cluster_idx] = val;
                col_sum += val;
            }

            for cluster_idx in 0..k {
                col[cluster_idx] /= col_sum;
            }

            col
        })
        .collect();

    Mat::from_fn(k, n, |i, j| columns[j][i])
}

/// Compute observed and expected diversity statistics for one variable.
///
/// `O[k,b] = sum of R[k,n]` for all cells n at level b
/// `E[k,b] = R_k_total * pr_b[b]`
///
/// ### Params
///
/// * `r` - Soft assignments (K x N)
/// * `info` - Batch information for one variable
///
/// ### Returns
///
/// `OEPair` of (O: K x B, E: K x B)
pub fn compute_diversity_statistics(r: MatRef<f32>, info: &BatchInfo) -> OEPair {
    let k = r.nrows();
    let b = info.n_levels;

    let mut o = Mat::zeros(k, b);

    for (level_idx, cell_indices) in info.batch_indices.iter().enumerate() {
        for &cell_idx in cell_indices {
            for cluster_idx in 0..k {
                o[(cluster_idx, level_idx)] += r[(cluster_idx, cell_idx)];
            }
        }
    }

    let mut row_sums = vec![0.0f32; k];
    for cluster_idx in 0..k {
        for level_idx in 0..b {
            row_sums[cluster_idx] += o[(cluster_idx, level_idx)];
        }
    }

    let mut e = Mat::zeros(k, b);
    for cluster_idx in 0..k {
        for level_idx in 0..b {
            e[(cluster_idx, level_idx)] = row_sums[cluster_idx] * info.pr_b[level_idx];
        }
    }

    OEPair { o, e }
}

/// Compute diversity statistics for all variables.
///
/// ### Params
///
/// * `r` - Soft assignments (K x N)
/// * `batch_infos` - Batch information per variable
///
/// ### Returns
///
/// Vec of `OEPair`, one per variable
pub fn compute_all_diversity_statistics(r: MatRef<f32>, batch_infos: &[BatchInfo]) -> Vec<OEPair> {
    batch_infos
        .iter()
        .map(|info| compute_diversity_statistics(r, info))
        .collect()
}

/// Update cluster centroids using soft assignments.
///
/// Computes weighted mean: Y = normalise(R * Z_cos)
///
/// ### Params
///
/// * `z_cos` - Cosine-normalised data (N x d)
/// * `r` - Soft assignments (K x N)
///
/// ### Returns
///
/// Updated centroids (K x d), cosine-normalised
pub fn update_centroids_from_r(z_cos: MatRef<f32>, r: MatRef<f32>) -> Mat<f32> {
    assert_eq!(r.ncols(), z_cos.nrows(), "R columns must match Z_cos rows");
    let y = r * z_cos;
    cosine_normalise(&y)
}

/// Compute Harmony objective function for convergence checking.
///
/// Objective = kmeans_error + entropy + cross_entropy, where the
/// cross-entropy term sums over all batch variables.
///
/// ### Params
///
/// * `r` - Soft assignments (K x N)
/// * `dist_mat` - Distance matrix (K x N)
/// * `oe_pairs` - Observed/expected pairs per variable
/// * `sigma` - Per-cluster weights (length K)
/// * `theta` - Per-variable penalties (length n_variables)
/// * `batch_infos` - Batch information per variable
///
/// ### Returns
///
/// Objective value (lower is better)
pub fn compute_objective(
    r: MatRef<f32>,
    dist_mat: MatRef<f32>,
    oe_pairs: &[OEPair],
    sigma: &[f32],
    theta: &[f32],
    batch_infos: &[BatchInfo],
) -> f32 {
    let k = r.nrows();
    let n = r.ncols();
    let n_vars = batch_infos.len();

    assert_eq!(dist_mat.nrows(), k);
    assert_eq!(dist_mat.ncols(), n);
    assert_eq!(sigma.len(), k);
    assert_eq!(theta.len(), n_vars);
    assert_eq!(oe_pairs.len(), n_vars);

    let norm_const = 2000.0 / n as f32;

    // K-means error + entropy
    let mut kmeans_error = 0.0f32;
    let mut entropy = 0.0f32;

    for cell_idx in 0..n {
        for cluster_idx in 0..k {
            let r_val = r[(cluster_idx, cell_idx)];
            kmeans_error += r_val * dist_mat[(cluster_idx, cell_idx)];
            if r_val > 0.0 {
                entropy += r_val * r_val.ln() * sigma[cluster_idx];
            }
        }
    }

    // Cross-entropy: sum over all variables
    let mut cross_entropy = 0.0f32;

    for (var_idx, info) in batch_infos.iter().enumerate() {
        let OEPair { o, e } = &oe_pairs[var_idx];
        let b = info.n_levels;
        let theta_v = theta[var_idx];

        // Precompute log-ratio table for this variable
        let mut log_ratio = vec![0.0f32; k * b];
        for cluster_idx in 0..k {
            for level_idx in 0..b {
                let o_val = o[(cluster_idx, level_idx)];
                let e_val = e[(cluster_idx, level_idx)];
                if e_val > 0.0 {
                    log_ratio[cluster_idx * b + level_idx] = ((o_val + e_val) / e_val).ln();
                }
            }
        }

        for (level_idx, cell_indices) in info.batch_indices.iter().enumerate() {
            for &cell_idx in cell_indices {
                for cluster_idx in 0..k {
                    let r_val = r[(cluster_idx, cell_idx)];
                    cross_entropy += r_val
                        * sigma[cluster_idx]
                        * theta_v
                        * log_ratio[cluster_idx * b + level_idx];
                }
            }
        }
    }

    (kmeans_error + entropy + cross_entropy) * norm_const
}

/// Update soft assignments with diversity penalties across all batch variables.
///
/// The diversity penalty is a product over variables:
///
/// `R[k,n] proportional to exp(-dist[k,n]/sigma[k]) * product_v (E_v[k, b_v(n)] / (O_v[k, b_v(n)] + E_v[k, b_v(n)]))^theta_v`
///
/// Uses block-wise shuffling to efficiently update O and E statistics.
///
/// ### Params
///
/// * `dist_mat` - Distance matrix (K x N)
/// * `sigma` - Per-cluster diversity weights (length K)
/// * `theta` - Per-variable diversity penalties (length n_variables)
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
pub fn update_r_with_diversity(
    dist_mat: MatRef<f32>,
    sigma: &[f32],
    theta: &[f32],
    batch_infos: &[BatchInfo],
    block_size: f32,
    seed: usize,
    r_init: MatRef<f32>,
    oe_init: &[OEPair],
) -> (Mat<f32>, Vec<OEPair>) {
    let k = dist_mat.nrows();
    let n = dist_mat.ncols();
    let n_vars = batch_infos.len();

    assert_eq!(sigma.len(), k);
    assert_eq!(theta.len(), n_vars);
    assert_eq!(oe_init.len(), n_vars);

    // Precompute base scaled distances: exp(-dist / sigma), column-normalised
    let mut scale_dist = Mat::zeros(k, n);
    for cell_idx in 0..n {
        let mut col_sum = 0.0f32;
        for cluster_idx in 0..k {
            let val = (-dist_mat[(cluster_idx, cell_idx)] / sigma[cluster_idx]).exp();
            scale_dist[(cluster_idx, cell_idx)] = val;
            col_sum += val;
        }
        if col_sum > 0.0 {
            for cluster_idx in 0..k {
                scale_dist[(cluster_idx, cell_idx)] /= col_sum;
            }
        }
    }

    // Shuffled update order
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

    // Block-wise updates
    let n_blocks = (1.0 / block_size).ceil() as usize;
    let cells_per_block = (n as f32 * block_size) as usize;

    for block_idx in 0..n_blocks {
        let idx_min = block_idx * cells_per_block;
        let idx_max = ((block_idx + 1) * cells_per_block).min(n);

        // Step 1: remove block cells from O and E for all variables
        for &cell_idx in &update_order[idx_min..idx_max] {
            for var_idx in 0..n_vars {
                let level = batch_infos[var_idx].cell_to_level[cell_idx];
                let OEPair { o, e } = &mut oe[var_idx];
                let pr_b = &batch_infos[var_idx].pr_b;
                let b = batch_infos[var_idx].n_levels;

                for cluster_idx in 0..k {
                    let r_val = r[(cluster_idx, cell_idx)];
                    o[(cluster_idx, level)] -= r_val;
                    for b_idx in 0..b {
                        e[(cluster_idx, b_idx)] -= r_val * pr_b[b_idx];
                    }
                }
            }
        }

        // Step 2: recompute R for block cells with diversity penalty
        for &cell_idx in &update_order[idx_min..idx_max] {
            let mut new_col_sum = 0.0f32;

            for cluster_idx in 0..k {
                let base = scale_dist[(cluster_idx, cell_idx)];

                // Product of penalties across all variables
                let mut penalty = 1.0f32;
                for var_idx in 0..n_vars {
                    let level = batch_infos[var_idx].cell_to_level[cell_idx];
                    let OEPair { o, e } = &oe[var_idx];
                    let theta_v = theta[var_idx];

                    let o_val = o[(cluster_idx, level)];
                    let e_val = e[(cluster_idx, level)];
                    if o_val + e_val > 0.0 {
                        penalty *= (e_val / (o_val + e_val)).powf(theta_v);
                    }
                }

                let new_r = base * penalty;
                r[(cluster_idx, cell_idx)] = new_r;
                new_col_sum += new_r;
            }

            if new_col_sum > 0.0 {
                for cluster_idx in 0..k {
                    r[(cluster_idx, cell_idx)] /= new_col_sum;
                }
            }
        }

        // Step 3: add block cells back to O and E for all variables
        for &cell_idx in &update_order[idx_min..idx_max] {
            for var_idx in 0..n_vars {
                let level = batch_infos[var_idx].cell_to_level[cell_idx];
                let OEPair { o, e } = &mut oe[var_idx];
                let pr_b = &batch_infos[var_idx].pr_b;
                let b = batch_infos[var_idx].n_levels;

                for cluster_idx in 0..k {
                    let r_val = r[(cluster_idx, cell_idx)];
                    o[(cluster_idx, level)] += r_val;
                    for b_idx in 0..b {
                        e[(cluster_idx, b_idx)] += r_val * pr_b[b_idx];
                    }
                }
            }
        }
    }

    (r, oe)
}

/// Apply per-cluster ridge regression to remove batch effects from multiple
/// variables jointly.
///
/// For each cluster k, constructs a joint design matrix with an intercept
/// and deviation columns for each non-reference level of each variable:
///
///   Phi = [intercept | var0_level1 ... var0_levelB0-1 | var1_level1 ... ]
///
/// The design matrix has P = 1 + sum_v (B_v - 1) rows. Ridge regression
/// estimates batch effects, which are subtracted from the data weighted
/// by soft cluster assignments.
///
/// ### Params
///
/// * `z_orig` - Original data (N x d)
/// * `r` - Soft assignments (K x N)
/// * `batch_infos` - Batch information per variable
/// * `lambda` - Ridge penalty (scalar, applied to all diagonal entries)
///
/// ### Returns
///
/// Corrected data (N x d)
pub fn ridge_regression_correction(
    z_orig: MatRef<f32>,
    r: MatRef<f32>,
    batch_infos: &[BatchInfo],
    lambda: f32,
) -> Mat<f32> {
    let n = z_orig.nrows();
    let d = z_orig.ncols();
    let k = r.nrows();
    let n_vars = batch_infos.len();

    assert_eq!(r.ncols(), n);

    // Compute design matrix dimensions and column offsets.
    // Column 0 = intercept.
    // For variable v, columns offset_v .. offset_v + (B_v - 2) correspond
    // to levels 1 .. B_v - 1 (level 0 is the reference).
    let mut offsets = Vec::with_capacity(n_vars);
    let mut col = 1usize;
    for info in batch_infos {
        offsets.push(col);
        col += info.n_levels - 1;
    }
    let p = col; // total design matrix rows

    let mut z_corr = z_orig.to_owned();

    for cluster_idx in 0..k {
        // For each cell, determine which design columns are active and
        // accumulate design_cov (P x P) and phi_z (P x d).
        let mut design_cov = Mat::<f32>::zeros(p, p);
        let mut phi_z = Mat::<f32>::zeros(p, d);

        for cell_idx in 0..n {
            let r_val = r[(cluster_idx, cell_idx)];
            let r2 = r_val * r_val;

            // Determine active columns for this cell
            // Column 0 (intercept) is always active
            let mut active_cols: Vec<usize> = Vec::with_capacity(1 + n_vars);
            active_cols.push(0);

            for var_idx in 0..n_vars {
                let level = batch_infos[var_idx].cell_to_level[cell_idx];
                if level >= 1 {
                    active_cols.push(offsets[var_idx] + level - 1);
                }
            }

            // Accumulate phi_z
            for &c in &active_cols {
                for feat in 0..d {
                    phi_z[(c, feat)] += r_val * z_orig[(cell_idx, feat)];
                }
            }

            // Accumulate design_cov (symmetric outer product)
            for (i, &ci) in active_cols.iter().enumerate() {
                for &cj in &active_cols[i..] {
                    design_cov[(ci, cj)] += r2;
                    if ci != cj {
                        design_cov[(cj, ci)] += r2;
                    }
                }
            }
        }

        // Add ridge penalty
        for i in 0..p {
            design_cov[(i, i)] += lambda;
        }

        // Solve: W = design_cov^{-1} * phi_z
        let partial_piv_lu: PartialPivLu<f32> = design_cov.partial_piv_lu();
        let inv_cov = partial_piv_lu.inverse();
        let w = &inv_cov * &phi_z;

        // Apply correction: subtract deviation contributions (columns 1..P-1)
        // For each cell, subtract R[k,n] * W[c, :] for each active
        // non-intercept column c.
        for cell_idx in 0..n {
            let r_val = r[(cluster_idx, cell_idx)];

            for var_idx in 0..n_vars {
                let level = batch_infos[var_idx].cell_to_level[cell_idx];
                if level >= 1 {
                    let c = offsets[var_idx] + level - 1;
                    for feat in 0..d {
                        z_corr[(cell_idx, feat)] -= r_val * w[(c, feat)];
                    }
                }
            }
        }
    }

    z_corr
}

/////////////
// Harmony //
/////////////

/// Check convergence using windowed moving average.
///
/// ### Params
///
/// * `objectives` - Objective values
/// * `window_size` - Window size
/// * `epsilon` - Convergence threshold
///
/// ### Returns
///
/// Whether convergence is reached
fn check_convergence(objectives: &[f32], window_size: usize, epsilon: f32) -> bool {
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

/// Harmony state
struct HarmonyState {
    z_orig: Mat<f32>,
    z_cos: Mat<f32>,
    z_corr: Mat<f32>,
    y: Mat<f32>,
    r: Mat<f32>,
    oe_pairs: Vec<OEPair>,
    objectives_kmeans: Vec<f32>,
    objectives_harmony: Vec<f32>,
}

/// Run Harmony batch correction with one or more batch variables.
///
/// Each element of `batch_labels` is a slice of length N giving the level
/// assignments for one categorical variable. For example, to correct for
/// both sample and technology:
///
/// ```ignore
/// let sample_labels = vec![0, 0, 1, 1, 2, 2];
/// let tech_labels   = vec![0, 1, 0, 1, 0, 1];
/// let corrected = harmony(
///     pca.as_ref(),
///     &[&sample_labels, &tech_labels],
///     &params,
///     42,
///     true,
/// );
/// ```
///
/// ### Params
///
/// * `pca` - PCA embedding (N x d)
/// * `batch_labels` - one label slice per variable, each of length N
/// * `params` - Harmony hyperparameters
/// * `seed` - Random seed
/// * `verbose` - Print progress
///
/// ### Returns
///
/// Corrected PCA embedding (N x d)
pub fn harmony(
    pca: MatRef<f32>,
    batch_labels: &[Vec<usize>],
    params: &HarmonyParams,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let n = pca.nrows();
    let d = pca.ncols();
    let n_vars = batch_labels.len();

    assert!(n_vars >= 1, "At least one batch variable required");

    let batch_infos = create_batch_infos(batch_labels, n);

    if verbose {
        println!(
            "Harmony: {} cells, {} dims, {} variable(s), {} clusters",
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

    let lambda_scalar = params.lambda[0];

    let z_orig = pca.to_owned();
    let z_cos = cosine_normalise(&z_orig);

    if verbose {
        println!("Running initial k-means...");
    }

    let y = run_kmeans_cosine(
        z_cos.as_ref(),
        params.k,
        params.max_iter_kmeans,
        seed,
        verbose,
    );

    let dist_mat = compute_cosine_distances(y.as_ref(), z_cos.as_ref());
    let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma);
    let oe_pairs = compute_all_diversity_statistics(r.as_ref(), &batch_infos);

    let initial_obj = compute_objective(
        r.as_ref(),
        dist_mat.as_ref(),
        &oe_pairs,
        &sigma,
        &theta,
        &batch_infos,
    );

    let mut state = HarmonyState {
        z_orig,
        z_cos,
        z_corr: pca.to_owned(),
        y,
        r,
        oe_pairs,
        objectives_kmeans: vec![initial_obj],
        objectives_harmony: vec![initial_obj],
    };

    if verbose {
        println!("Initial objective: {:.4}", initial_obj);
    }

    for harmony_iter in 0..params.max_iter_harmony {
        if verbose {
            println!("\n=== Harmony iteration {} ===", harmony_iter + 1);
        }

        for kmeans_iter in 0..params.max_iter_kmeans {
            state.y = update_centroids_from_r(state.z_cos.as_ref(), state.r.as_ref());

            let dist_mat = compute_cosine_distances(state.y.as_ref(), state.z_cos.as_ref());

            let (r_new, oe_new) = update_r_with_diversity(
                dist_mat.as_ref(),
                &sigma,
                &theta,
                &batch_infos,
                params.block_size,
                seed + harmony_iter * 1000 + kmeans_iter,
                state.r.as_ref(),
                &state.oe_pairs,
            );

            state.r = r_new;
            state.oe_pairs = oe_new;

            let obj = compute_objective(
                state.r.as_ref(),
                dist_mat.as_ref(),
                &state.oe_pairs,
                &sigma,
                &theta,
                &batch_infos,
            );

            state.objectives_kmeans.push(obj);

            if verbose && kmeans_iter % 5 == 0 {
                println!("  K-means iter {}: obj = {:.4}", kmeans_iter + 1, obj);
            }

            if kmeans_iter >= params.window_size {
                let converged = check_convergence(
                    &state.objectives_kmeans,
                    params.window_size,
                    params.epsilon_kmeans,
                );
                if converged {
                    if verbose {
                        println!("  K-means converged at iteration {}", kmeans_iter + 1);
                    }
                    break;
                }
            }
        }

        if verbose {
            println!("  Applying ridge regression correction...");
        }

        state.z_corr = ridge_regression_correction(
            state.z_orig.as_ref(),
            state.r.as_ref(),
            &batch_infos,
            lambda_scalar,
        );

        state.z_cos = cosine_normalise(&state.z_corr);

        let harmony_obj = *state.objectives_kmeans.last().unwrap();
        state.objectives_harmony.push(harmony_obj);

        if verbose {
            println!("  Harmony objective: {:.4}", harmony_obj);
        }

        if harmony_iter >= 1 {
            let obj_old = state.objectives_harmony[harmony_iter];
            let obj_new = state.objectives_harmony[harmony_iter + 1];
            let rel_change = (obj_old - obj_new).abs() / obj_old.abs();

            if rel_change < params.epsilon_harmony {
                if verbose {
                    println!("\nHarmony converged at iteration {}", harmony_iter + 1);
                    println!("  Final objective: {:.4}", obj_new);
                }
                break;
            }
        }
    }

    state.z_corr
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

    #[test]
    fn test_create_batch_info() {
        let labels = vec![0, 0, 1, 1, 2, 0];
        let info = create_batch_info(&labels, 6);

        assert_eq!(info.n_levels, 3);
        assert!((info.pr_b[0] - 0.5).abs() < 1e-6);
        assert!((info.pr_b[1] - 0.333333).abs() < 1e-4);
        assert!((info.pr_b[2] - 0.166666).abs() < 1e-4);
        assert_eq!(info.batch_indices[0], vec![0, 1, 5]);
        assert_eq!(info.batch_indices[1], vec![2, 3]);
        assert_eq!(info.batch_indices[2], vec![4]);
        assert_eq!(info.cell_to_level, labels);
    }

    #[test]
    fn test_create_batch_info_single_level() {
        let labels = vec![0, 0, 0];
        let info = create_batch_info(&labels, 3);

        assert_eq!(info.n_levels, 1);
        assert_eq!(info.pr_b, vec![1.0]);
        assert_eq!(info.batch_indices[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_create_batch_infos_multiple() {
        let var0 = vec![0, 0, 1, 1];
        let var1 = vec![0, 1, 0, 1];
        let infos = create_batch_infos(&[var0, var1], 4);

        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].n_levels, 2);
        assert_eq!(infos[1].n_levels, 2);
        assert_eq!(infos[0].batch_indices[0], vec![0, 1]);
        assert_eq!(infos[0].batch_indices[1], vec![2, 3]);
        assert_eq!(infos[1].batch_indices[0], vec![0, 2]);
        assert_eq!(infos[1].batch_indices[1], vec![1, 3]);
    }

    #[test]
    fn test_run_kmeans_cosine_basic() {
        let mut data = Vec::new();
        for _ in 0..10 {
            data.push(vec![0.9, 0.1, 0.0]);
        }
        for _ in 0..10 {
            data.push(vec![0.1, 0.9, 0.0]);
        }

        let mat = Mat::from_fn(20, 3, |i, j| data[i][j]);
        let mat_cos = cosine_normalise(&mat);

        let centroids = run_kmeans_cosine(mat_cos.as_ref(), 2, 25, 42, false);

        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 3);

        for k in 0..2 {
            let norm: f32 = (0..3)
                .map(|j| centroids[(k, j)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_compute_cosine_distances_identical() {
        let centroids = Mat::from_fn(2, 3, |i, j| match i {
            0 => 1.0 / 3.0f32.sqrt(),
            _ => {
                if j == 0 {
                    1.0
                } else {
                    0.0
                }
            }
        });
        let data = centroids.clone();
        let dist = compute_cosine_distances(centroids.as_ref(), data.as_ref());
        for k in 0..2 {
            assert_relative_eq!(dist[(k, k)], 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_compute_cosine_distances_orthogonal() {
        let centroids = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let data = centroids.clone();
        let dist = compute_cosine_distances(centroids.as_ref(), data.as_ref());
        assert_relative_eq!(dist[(0, 1)], 2.0, epsilon = 1e-5);
        assert_relative_eq!(dist[(1, 0)], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_initialise_r_from_distances() {
        let dist_data = [0.0, 2.0, 4.0, 4.0, 2.0, 0.0];
        let dist_mat = Mat::from_fn(2, 3, |i, j| dist_data[i * 3 + j]);
        let sigma = vec![1.0, 1.0];

        let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma);

        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 3);

        for col in 0..3 {
            let col_sum: f32 = (0..2).map(|row| r[(row, col)]).sum();
            assert_relative_eq!(col_sum, 1.0, epsilon = 1e-6);
        }

        assert!(r[(0, 0)] > 0.9);
        assert!(r[(1, 0)] < 0.1);
        assert!(r[(0, 2)] < 0.1);
        assert!(r[(1, 2)] > 0.9);
        assert_relative_eq!(r[(0, 1)], 0.5, epsilon = 1e-6);
        assert_relative_eq!(r[(1, 1)], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_initialise_r_different_sigmas() {
        let dist_data = [1.0, 1.0, 1.0, 1.0];
        let dist_mat = Mat::from_fn(2, 2, |i, j| dist_data[i * 2 + j]);
        let sigma = vec![0.5, 2.0];

        let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma);
        assert!(r[(1, 0)] > r[(0, 0)]);
    }

    #[test]
    fn test_compute_diversity_statistics_simple() {
        let labels = vec![0, 0, 1, 1, 1, 2];
        let info = create_batch_info(&labels, 6);

        let r = mat![
            [0.8, 0.7, 0.6, 0.9, 0.5, 0.3],
            [0.2, 0.3, 0.4, 0.1, 0.5, 0.7],
        ];

        let OEPair { o, e } = compute_diversity_statistics(r.as_ref(), &info);

        assert_eq!(o.nrows(), 2);
        assert_eq!(o.ncols(), 3);

        assert!((o[(0, 0)] - 1.5).abs() < 1e-6);
        assert!((o[(0, 1)] - 2.0).abs() < 1e-6);
        assert!((o[(0, 2)] - 0.3).abs() < 1e-6);
        assert!((o[(1, 0)] - 0.5).abs() < 1e-6);
        assert!((o[(1, 1)] - 1.0).abs() < 1e-6);
        assert!((o[(1, 2)] - 0.7).abs() < 1e-6);

        let row_sum_0 = 3.8f32;
        let row_sum_1 = 2.2f32;

        for cluster_idx in 0..2 {
            let row_sum = if cluster_idx == 0 {
                row_sum_0
            } else {
                row_sum_1
            };
            for level_idx in 0..3 {
                let expected = row_sum * info.pr_b[level_idx];
                assert!(
                    (e[(cluster_idx, level_idx)] - expected).abs() < 1e-5,
                    "E[{},{}] = {}, expected {}",
                    cluster_idx,
                    level_idx,
                    e[(cluster_idx, level_idx)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_diversity_statistics_properties() {
        let labels = vec![0, 0, 0, 1, 1, 2, 2, 2, 2];
        let info = create_batch_info(&labels, 9);

        let r = mat![
            [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.5, 0.6],
            [0.3, 0.2, 0.4, 0.2, 0.5, 0.1, 0.6, 0.3, 0.3],
            [0.2, 0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1],
        ];

        let OEPair { o, e } = compute_diversity_statistics(r.as_ref(), &info);

        // Property 1: O row sums equal R row sums
        for cluster_idx in 0..3 {
            let o_sum: f32 = (0..3).map(|b| o[(cluster_idx, b)]).sum();
            let r_row_sum: f32 = (0..9).map(|n| r[(cluster_idx, n)]).sum();
            assert!(
                (o_sum - r_row_sum).abs() < 1e-5,
                "Cluster {}: O sum ({}) != R row sum ({})",
                cluster_idx,
                o_sum,
                r_row_sum
            );
        }

        // Property 2: E row sums equal R row sums
        for cluster_idx in 0..3 {
            let e_sum: f32 = (0..3).map(|b| e[(cluster_idx, b)]).sum();
            let r_row_sum: f32 = (0..9).map(|n| r[(cluster_idx, n)]).sum();
            assert!(
                (e_sum - r_row_sum).abs() < 1e-5,
                "Cluster {}: E sum ({}) != R row sum ({})",
                cluster_idx,
                e_sum,
                r_row_sum
            );
        }

        // Property 3: E proportional to pr_b
        for cluster_idx in 0..3 {
            let r_row_sum: f32 = (0..9).map(|n| r[(cluster_idx, n)]).sum();
            for level_idx in 0..3 {
                let expected = r_row_sum * info.pr_b[level_idx];
                assert!(
                    (e[(cluster_idx, level_idx)] - expected).abs() < 1e-5,
                    "E[{},{}] = {}, expected {}",
                    cluster_idx,
                    level_idx,
                    e[(cluster_idx, level_idx)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_update_centroids_simple() {
        let norm_3 = (0.5f32 * 0.5 + 0.5 * 0.5).sqrt();
        let z_cos_norm = mat![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5 / norm_3, 0.5 / norm_3, 0.0],
        ];

        let r = mat![[0.9, 0.1, 0.1, 0.8], [0.1, 0.9, 0.9, 0.2],];

        let y = update_centroids_from_r(z_cos_norm.as_ref(), r.as_ref());

        assert_eq!(y.nrows(), 2);
        assert_eq!(y.ncols(), 3);

        for k in 0..2 {
            let norm = (y[(k, 0)].powi(2) + y[(k, 1)].powi(2) + y[(k, 2)].powi(2)).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Cluster {} not unit length: {}",
                k,
                norm
            );
        }

        assert!(y[(0, 0)] > y[(0, 1)]);
        assert!(y[(1, 1)] > y[(1, 0)]);
    }

    #[test]
    fn test_update_centroids_hard_assignment() {
        let z_cos = mat![[1.0, 0.0], [0.0, 1.0], [0.707, 0.707],];

        let r = mat![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0],];

        let y = update_centroids_from_r(z_cos.as_ref(), r.as_ref());

        assert!((y[(1, 0)] - 0.0).abs() < 1e-6);
        assert!((y[(1, 1)] - 1.0).abs() < 1e-6);

        let norm = (y[(0, 0)].powi(2) + y[(0, 1)].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(y[(0, 0)] > y[(0, 1)]);
    }

    #[test]
    fn test_compute_objective_decreases() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4);
        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0];
        let r_uncertain = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]];
        let dist_mat_high = mat![[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]];
        let oe1 = compute_all_diversity_statistics(r_uncertain.as_ref(), from_ref(&info));
        let obj1 = compute_objective(
            r_uncertain.as_ref(),
            dist_mat_high.as_ref(),
            &oe1[..],
            &sigma,
            &theta,
            from_ref(&info),
        );
        let r_confident = mat![[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]];
        let dist_mat_low = mat![[0.1, 0.1, 1.0, 1.0], [1.0, 1.0, 0.1, 0.1]];
        let oe2 = compute_all_diversity_statistics(r_confident.as_ref(), from_ref(&info));
        let obj2 = compute_objective(
            r_confident.as_ref(),
            dist_mat_low.as_ref(),
            &oe2[..],
            &sigma,
            &theta,
            from_ref(&info),
        );
        assert!(
            obj2 < obj1,
            "Confident R should have lower objective: {} vs {}",
            obj2,
            obj1
        );
    }

    #[test]
    fn test_compute_objective_components() {
        let labels = vec![0, 0, 1];
        let info = create_batch_info(&labels, 3);
        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0];

        let r = mat![[0.8, 0.7, 0.2], [0.2, 0.3, 0.8]];
        let dist_mat = mat![[0.1, 0.2, 0.9], [0.9, 0.8, 0.1]];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        let obj = compute_objective(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe[..],
            &sigma,
            &theta,
            from_ref(&info),
        );

        assert!(obj.is_finite());
        assert!(obj > 0.0);
        assert!(obj < 10000.0, "Objective seems too large: {}", obj);
    }

    #[test]
    fn test_objective_zero_entropy() {
        let labels = vec![0, 1];
        let info = create_batch_info(&labels, 2);
        let sigma = vec![1.0, 1.0];
        let theta = vec![0.0]; // no diversity penalty

        let r = mat![[1.0, 0.0], [0.0, 1.0]];
        let dist_mat = mat![[0.1, 0.9], [0.9, 0.1]];
        let oe = compute_all_diversity_statistics(r.as_ref(), from_ref(&info));

        let obj = compute_objective(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe[..],
            &sigma,
            &theta,
            from_ref(&info),
        );

        let expected = 0.2 * 1000.0;
        assert!(
            (obj - expected).abs() < 1.0,
            "Objective should be ~200: got {}",
            obj
        );
    }

    #[test]
    fn test_update_r_basic() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4);
        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0];

        let r_init = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]];
        let dist_mat = mat![[0.1, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1]];
        let oe_init = compute_all_diversity_statistics(r_init.as_ref(), from_ref(&info));

        let (r_new, oe_new) = update_r_with_diversity(
            dist_mat.as_ref(),
            &sigma,
            &theta,
            from_ref(&info),
            0.5,
            42,
            r_init.as_ref(),
            &oe_init[..],
        );

        for cell_idx in 0..4 {
            let col_sum: f32 = (0..2).map(|k| r_new[(k, cell_idx)]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-5,
                "Column {} sum: {}",
                cell_idx,
                col_sum
            );
        }

        assert!(r_new[(0, 0)] > 0.5);
        assert!(r_new[(0, 1)] > 0.5);
        assert!(r_new[(1, 2)] > 0.5);
        assert!(r_new[(1, 3)] > 0.5);

        // O and E should be consistent with new R
        let oe_check = compute_all_diversity_statistics(r_new.as_ref(), &[info]);
        let OEPair { o: o_new, e: e_new } = &oe_new[0];
        let OEPair {
            o: o_check,
            e: e_check,
        } = &oe_check[0];

        for cluster_idx in 0..2 {
            for level_idx in 0..2 {
                assert!(
                    (o_new[(cluster_idx, level_idx)] - o_check[(cluster_idx, level_idx)]).abs()
                        < 1e-4,
                    "O mismatch at [{},{}]",
                    cluster_idx,
                    level_idx,
                );
                assert!(
                    (e_new[(cluster_idx, level_idx)] - e_check[(cluster_idx, level_idx)]).abs()
                        < 1e-4,
                    "E mismatch at [{},{}]",
                    cluster_idx,
                    level_idx,
                );
            }
        }
    }

    #[test]
    fn test_update_r_no_diversity_penalty() {
        let labels = vec![0, 0];
        let info = create_batch_info(&labels, 2);
        let sigma = vec![1.0, 1.0];
        let theta = vec![0.0];

        let r_init = mat![[0.5, 0.5], [0.5, 0.5]];
        let dist_mat = mat![[0.1, 0.9], [0.9, 0.1]];
        let oe_init = compute_all_diversity_statistics(r_init.as_ref(), from_ref(&info));

        let (r_new, _) = update_r_with_diversity(
            dist_mat.as_ref(),
            &sigma,
            &theta,
            from_ref(&info),
            1.0,
            42,
            r_init.as_ref(),
            &oe_init[..],
        );

        assert!(r_new[(0, 0)] > 0.6);
        assert!(r_new[(1, 1)] > 0.6);
    }

    #[test]
    fn test_update_r_diversity_correction() {
        let labels = vec![0, 0, 0, 1];
        let info = create_batch_info(&labels, 4);
        let sigma = vec![1.0, 1.0];
        let theta = vec![2.0]; // strong penalty

        let r_init = mat![[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]];
        let dist_mat = mat![[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8]];
        let oe_init = compute_all_diversity_statistics(r_init.as_ref(), from_ref(&info));

        let (r_new, _) = update_r_with_diversity(
            dist_mat.as_ref(),
            &sigma,
            &theta,
            from_ref(&info),
            0.5,
            42,
            r_init.as_ref(),
            &oe_init[..],
        );

        for cell_idx in 0..4 {
            let col_sum: f32 = (0..2).map(|k| r_new[(k, cell_idx)]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-5,
                "Column {} sum: {}",
                cell_idx,
                col_sum
            );
        }
    }

    #[test]
    fn test_update_r_two_variables() {
        let labels0 = vec![0, 0, 1, 1];
        let labels1 = vec![0, 1, 0, 1];
        let info0 = create_batch_info(&labels0, 4);
        let info1 = create_batch_info(&labels1, 4);

        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0, 1.0];

        let r_init = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]];
        let dist_mat = mat![[0.1, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1]];

        let oe_init =
            compute_all_diversity_statistics(r_init.as_ref(), &[info0.clone(), info1.clone()]);

        let (r_new, oe_new) = update_r_with_diversity(
            dist_mat.as_ref(),
            &sigma,
            &theta,
            &[info0.clone(), info1.clone()],
            0.5,
            42,
            r_init.as_ref(),
            &oe_init,
        );

        // Columns should still sum to 1
        for cell_idx in 0..4 {
            let col_sum: f32 = (0..2).map(|k| r_new[(k, cell_idx)]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-5,
                "Column {} sum: {}",
                cell_idx,
                col_sum
            );
        }

        // O and E should match recomputation for both variables
        assert_eq!(oe_new.len(), 2);
        for (var_idx, info) in [&info0, &info1].iter().enumerate() {
            let OEPair { o: o_new, e: e_new } = &oe_new[var_idx];
            let OEPair {
                o: o_check,
                e: e_check,
            } = compute_diversity_statistics(r_new.as_ref(), info);

            for cluster_idx in 0..2 {
                for level_idx in 0..info.n_levels {
                    assert!(
                        (o_new[(cluster_idx, level_idx)] - o_check[(cluster_idx, level_idx)]).abs()
                            < 1e-4,
                        "Var {} O mismatch at [{},{}]",
                        var_idx,
                        cluster_idx,
                        level_idx,
                    );
                    assert!(
                        (e_new[(cluster_idx, level_idx)] - e_check[(cluster_idx, level_idx)]).abs()
                            < 1e-4,
                        "Var {} E mismatch at [{},{}]",
                        var_idx,
                        cluster_idx,
                        level_idx,
                    );
                }
            }
        }
    }

    #[test]
    fn test_ridge_regression_basic() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4);

        // Batch effect in feature 0
        let z_orig = mat![[1.0, 0.1], [1.1, 0.2], [5.0, 0.1], [5.1, 0.2],];

        let r = mat![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]];

        let z_corr = ridge_regression_correction(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            0.01,
        );

        let batch0_mean_orig = (z_orig[(0, 0)] + z_orig[(1, 0)]) / 2.0;
        let batch1_mean_orig = (z_orig[(2, 0)] + z_orig[(3, 0)]) / 2.0;
        let orig_diff = (batch1_mean_orig - batch0_mean_orig).abs();

        let batch0_mean_corr = (z_corr[(0, 0)] + z_corr[(1, 0)]) / 2.0;
        let batch1_mean_corr = (z_corr[(2, 0)] + z_corr[(3, 0)]) / 2.0;
        let corr_diff = (batch1_mean_corr - batch0_mean_corr).abs();

        assert!(
            corr_diff < orig_diff,
            "Batch effect should be reduced: orig_diff={}, corr_diff={}",
            orig_diff,
            corr_diff
        );

        let feature1_change: f32 = (0..4)
            .map(|i| (z_corr[(i, 1)] - z_orig[(i, 1)]).abs())
            .sum::<f32>()
            / 4.0;

        assert!(
            feature1_change < 0.5,
            "Feature without batch effect should change little: {}",
            feature1_change
        );
    }

    #[test]
    fn test_ridge_regression_no_correction_needed() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4);

        let z_orig = mat![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
        let r = mat![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]];

        let z_corr = ridge_regression_correction(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            0.1,
        );

        for i in 0..4 {
            for j in 0..2 {
                assert!(
                    (z_corr[(i, j)] - z_orig[(i, j)]).abs() < 0.5,
                    "Cell {}, feature {}: changed from {} to {}",
                    i,
                    j,
                    z_orig[(i, j)],
                    z_corr[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_ridge_regression_soft_assignments() {
        let labels = vec![0, 0, 1, 1];
        let info = create_batch_info(&labels, 4);

        let z_orig = mat![[1.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.0, 0.0]];
        let r = mat![[0.8, 0.9, 0.1, 0.2], [0.2, 0.1, 0.9, 0.8],];

        let z_corr = ridge_regression_correction(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            0.01,
        );

        let batch0_mean_orig = (z_orig[(0, 0)] + z_orig[(1, 0)]) / 2.0;
        let batch1_mean_orig = (z_orig[(2, 0)] + z_orig[(3, 0)]) / 2.0;
        let orig_diff = (batch1_mean_orig - batch0_mean_orig).abs();

        let batch0_mean_corr = (z_corr[(0, 0)] + z_corr[(1, 0)]) / 2.0;
        let batch1_mean_corr = (z_corr[(2, 0)] + z_corr[(3, 0)]) / 2.0;
        let corr_diff = (batch1_mean_corr - batch0_mean_corr).abs();

        assert!(
            corr_diff < orig_diff,
            "Batch effect should be reduced with soft assignments: orig_diff={}, corr_diff={}",
            orig_diff,
            corr_diff
        );
    }

    #[test]
    fn test_ridge_regression_preserves_dimensions() {
        let labels = vec![0, 1, 2];
        let info = create_batch_info(&labels, 3);

        let z_orig = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let r = mat![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]];

        let z_corr = ridge_regression_correction(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info),
            0.1,
        );

        assert_eq!(z_corr.nrows(), z_orig.nrows());
        assert_eq!(z_corr.ncols(), z_orig.ncols());
    }

    #[test]
    fn test_ridge_regression_two_variables() {
        let batch_labels = vec![0, 0, 1, 1, 2, 2];
        let sample_labels = vec![0, 1, 0, 1, 0, 1];
        let info_batch = create_batch_info(&batch_labels, 6);
        let info_sample = create_batch_info(&sample_labels, 6);

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

        let z_corr = ridge_regression_correction(
            z_orig.as_ref(),
            r.as_ref(),
            &[info_batch.clone(), info_sample.clone()],
            0.01,
        );

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
            "Batch effect in feature 0 should be reduced: orig={}, corr={}",
            orig_spread,
            corr_spread
        );

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
            "Sample effect in feature 1 should be reduced: orig={}, corr={}",
            orig_sample_diff,
            corr_sample_diff
        );
    }

    #[test]
    fn test_ridge_regression_two_vars_design_matrix_size() {
        let batch_labels = vec![0, 0, 1, 1];
        let sample_labels = vec![0, 1, 0, 1];
        let info_batch = create_batch_info(&batch_labels, 4);
        let info_sample = create_batch_info(&sample_labels, 4);

        let z_orig = mat![[1.0, 0.0], [1.0, 5.0], [5.0, 0.0], [5.0, 5.0],];

        let r = mat![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]];

        let z_one_var = ridge_regression_correction(
            z_orig.as_ref(),
            r.as_ref(),
            std::slice::from_ref(&info_batch),
            0.01,
        );

        let z_two_vars = ridge_regression_correction(
            z_orig.as_ref(),
            r.as_ref(),
            &[info_batch.clone(), info_sample.clone()],
            0.01,
        );

        let mut any_diff = false;
        for i in 0..4 {
            for j in 0..2 {
                if (z_one_var[(i, j)] - z_two_vars[(i, j)]).abs() > 1e-4 {
                    any_diff = true;
                }
            }
        }
        assert!(
            any_diff,
            "Two-variable correction should differ from single-variable"
        );

        let sample_diff_one: f32 = (0..4)
            .step_by(2)
            .map(|i| (z_one_var[(i, 1)] - z_one_var[(i + 1, 1)]).abs())
            .sum::<f32>();
        let sample_diff_two: f32 = (0..4)
            .step_by(2)
            .map(|i| (z_two_vars[(i, 1)] - z_two_vars[(i + 1, 1)]).abs())
            .sum::<f32>();

        assert!(
            sample_diff_two < sample_diff_one,
            "Two-variable correction should reduce sample effect more: one_var={}, two_var={}",
            sample_diff_one,
            sample_diff_two,
        );
    }

    #[test]
    fn test_objective_two_variables() {
        let labels0 = vec![0, 0, 1, 1];
        let labels1 = vec![0, 1, 0, 1];
        let info0 = create_batch_info(&labels0, 4);
        let info1 = create_batch_info(&labels1, 4);

        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0, 1.0];

        let r = mat![[0.8, 0.7, 0.2, 0.3], [0.2, 0.3, 0.8, 0.7]];
        let dist_mat = mat![[0.1, 0.2, 0.9, 0.8], [0.9, 0.8, 0.1, 0.2]];

        let oe = compute_all_diversity_statistics(r.as_ref(), &[info0.clone(), info1.clone()]);

        let obj = compute_objective(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe,
            &sigma,
            &theta,
            &[info0.clone(), info1.clone()],
        );

        assert!(obj.is_finite());

        let oe_single = compute_all_diversity_statistics(r.as_ref(), std::slice::from_ref(&info0));
        let obj_single = compute_objective(
            r.as_ref(),
            dist_mat.as_ref(),
            &oe_single,
            &sigma,
            &[1.0],
            std::slice::from_ref(&info0),
        );

        assert!(
            obj > obj_single,
            "Two variables should give higher objective than one: {} vs {}",
            obj,
            obj_single,
        );
    }
}
