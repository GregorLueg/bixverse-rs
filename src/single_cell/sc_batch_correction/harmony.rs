//! Implementation of the Harmony batch correction, please see Korsunsky, et
//! al., Nat Methods, 2019

use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::ml::clustering::k_means::KMeansParamsWrappers;
use crate::prelude::parse_verbosity_level;
use crate::prelude::*;

use super::batch_utils::par_for_each_col_mut;
use super::harmony_core::{
    RidgeSettings, Variant, base_error_and_entropy, centroids_from_r, distances_and_base,
    kmeans_centroids, normalise_rows_into, objective, observed_counts, r_to_mat, ridge_correction,
    row_major_to_mat, to_row_major, update_assignments, window_converged,
};

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
    /// Maximum iterations for the k-means clustering
    pub max_iter_kmeans: usize,
    /// Maximum Harmony outer iterations
    pub max_iter_harmony: usize,
    /// K-means clustering convergence threshold
    pub epsilon_kmeans: f32,
    /// Harmony convergence threshold
    pub epsilon_harmony: f32,
    /// Window size for convergence checking
    pub window_size: usize,
    /// K-mean parameters, see [KMeansParamsWrappers]
    pub kmeans_params: KMeansParamsWrappers,
}

/// Default implementation for HarmonyParams
impl Default for HarmonyParams {
    fn default() -> Self {
        Self {
            k: 100,
            sigma: vec![0.1],
            theta: vec![2.0],
            lambda: vec![1.0],
            block_size: 0.05,
            max_iter_kmeans: 20,
            max_iter_harmony: 10,
            epsilon_kmeans: 1e-3,
            epsilon_harmony: 1e-2,
            window_size: 3,
            kmeans_params: KMeansParamsWrappers::new(30, None, None),
        }
    }
}

/////////////
// Helpers //
/////////////

/// Wall-clock accumulator per named stage, printed at detailed verbosity.
#[derive(Default)]
pub(crate) struct StageTimes {
    /// Stage name and total time spent in it, in first-seen order
    stages: Vec<(&'static str, std::time::Duration)>,
}

impl StageTimes {
    /// Run `f` and add its wall-clock time to stage `name`.
    ///
    /// ### Params
    ///
    /// * `name` - Stage label
    /// * `f` - The work to time
    ///
    /// ### Returns
    ///
    /// Whatever `f` returns
    pub(crate) fn time<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed();
        match self.stages.iter_mut().find(|(n, _)| *n == name) {
            Some((_, acc)) => *acc += dt,
            None => self.stages.push((name, dt)),
        }
        out
    }

    /// Print one line per stage with its share of the summed stage time.
    pub(crate) fn print(&self) {
        let total: f64 = self.stages.iter().map(|(_, d)| d.as_secs_f64()).sum();
        println!(" Stage times:");
        for (name, d) in &self.stages {
            println!(
                "  {:<14} {:>9.3} s  {:>5.1}%",
                name,
                d.as_secs_f64(),
                100.0 * d.as_secs_f64() / total
            );
        }
    }
}

/// Harmony results (batch-corrected embedding + soft assignments)
pub struct HarmonyResult {
    /// Corrected embedding
    pub z_corr: Mat<f32>,
    /// Soft assignments
    pub r: Mat<f32>,
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
pub fn create_batch_info(labels: &[usize], n_cells: usize) -> Result<BatchInfo, BixverseErrors> {
    if labels.len() != n_cells {
        return Err(BixverseErrors::NumberLabelsNotEqualSampleNumber {
            label_length: labels.len(),
            n_samples: n_cells,
        });
    }

    let n_levels = labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    let mut batch_indices: Vec<Vec<usize>> = vec![Vec::new(); n_levels];
    for (cell_idx, &level) in labels.iter().enumerate() {
        batch_indices[level].push(cell_idx);
    }

    let pr_b: Vec<f32> = batch_indices
        .iter()
        .map(|cells| cells.len() as f32 / n_cells as f32)
        .collect();

    Ok(BatchInfo {
        batch_indices,
        pr_b,
        n_levels,
        cell_to_level: labels.to_vec(),
    })
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
pub fn create_batch_infos(
    all_labels: &[Vec<usize>],
    n_cells: usize,
) -> Result<Vec<BatchInfo>, BixverseErrors> {
    all_labels
        .iter()
        .map(|labels| create_batch_info(labels, n_cells))
        .collect()
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
    let mut dist = centroids * data_cos.transpose();
    par_for_each_col_mut(&mut dist, |_, col| {
        col.iter_mut().for_each(|x| *x = 2.0 * (1.0 - *x));
    });
    dist
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
pub fn initialise_r_from_dist(
    dist_mat: MatRef<f32>,
    sigma: &[f32],
) -> Result<Mat<f32>, BixverseErrors> {
    let k = dist_mat.nrows();
    let n = dist_mat.ncols();
    if sigma.len() != k {
        return Err(BixverseErrors::HarmonySigmaLengthUnequalCluster);
    }

    // pre-allocation trick also here...
    let mut flat = vec![0.0f32; n * k];
    flat.par_chunks_mut(k).enumerate().for_each(|(cell, col)| {
        let mut col_sum = 0.0f32;
        for cluster in 0..k {
            let val = (-dist_mat[(cluster, cell)] / sigma[cluster]).exp();
            col[cluster] = val;
            col_sum += val;
        }
        for v in col.iter_mut() {
            *v /= col_sum;
        }
    });

    Ok(MatRef::from_column_major_slice(&flat, k, n).to_owned())
}

/////////////
// Harmony //
/////////////

/// Run Harmony batch correction with one or more batch variables.
///
/// Follows R harmony 1.2.0 (Korsunsky et al.): every clustering iteration
/// recomputes the centroids from the soft assignments, the distances, and the
/// diversity-penalised assignments; each round ends with the ridge correction
/// over the full one-hot design of every variable (intercept unpenalised).
/// All state is held in flat cell-major buffers, see `harmony_core`.
///
/// ### Params
///
/// * `pca` - PCA embedding (N x d)
/// * `batch_labels` - one label slice per variable, each of length N
/// * `params` - Harmony hyperparameters
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Returns the [HarmonyResult]
pub fn harmony_with_state(
    pca: MatRef<f32>,
    batch_labels: &[Vec<usize>],
    params: &HarmonyParams,
    seed: usize,
    verbose: usize,
) -> Result<HarmonyResult, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start = Instant::now();

    let n = pca.nrows();
    let d = pca.ncols();
    let k = params.k;
    let n_vars = batch_labels.len();

    assert!(n_vars >= 1, "At least one batch variable required");

    let batch_infos = create_batch_infos(batch_labels, n)?;

    if verbosity.normal_verbosity() {
        println!(
            "Harmony: {} cells, {} dims, {} variable(s), {} clusters",
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
    let theta_levels: Vec<Vec<f32>> = batch_infos
        .iter()
        .zip(&theta)
        .map(|(info, &t)| vec![t; info.n_levels])
        .collect();

    let ridge = RidgeSettings {
        lambda: params.lambda[0],
        alpha: 0.0,
        dynamic_lambda: false,
        prune_cutoff: None,
    };

    let mut times = StageTimes::default();

    let z_orig = to_row_major(pca);
    let mut z_cos = vec![0.0f32; n * d];
    normalise_rows_into(&z_orig, d, &mut z_cos);
    let mut z_corr = z_orig.clone();

    if verbosity.normal_verbosity() {
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
        Variant::V1,
        error,
        entropy,
        &o,
        &r_sum,
        &batch_infos,
        &sigma,
        &theta_levels,
        n,
    );
    let mut objectives_kmeans = vec![initial_obj];
    let mut objectives_harmony = vec![initial_obj];

    if verbosity.normal_verbosity() {
        println!(" Initial data preparation done in {:.2?}", start.elapsed());
        println!("Initial objective: {:.4}", initial_obj);
    }

    for harmony_iter in 0..params.max_iter_harmony {
        if verbosity.normal_verbosity() {
            println!("\n=== Harmony iteration {} ===", harmony_iter + 1);
            println!("  Running k-means clustering...");
        }

        let start_iter = Instant::now();

        for kmeans_iter in 0..params.max_iter_kmeans {
            times.time("centroids", || centroids_from_r(&r, &z_cos, k, d, &mut y));
            times.time("distances", || {
                distances_and_base(&z_cos, &y, &sigma, d, &mut log_base, &mut base, &mut shift)
            });

            let (error, entropy) = times.time("update_r", || {
                update_assignments(
                    Variant::V1,
                    &mut r,
                    &base,
                    &log_base,
                    &shift,
                    &sigma,
                    &theta_levels,
                    &batch_infos,
                    &mut o,
                    &mut r_sum,
                    params.block_size,
                    (seed + harmony_iter * 1000 + kmeans_iter) as u64,
                )
            });

            let obj = objective(
                Variant::V1,
                error,
                entropy,
                &o,
                &r_sum,
                &batch_infos,
                &sigma,
                &theta_levels,
                n,
            );
            objectives_kmeans.push(obj);

            if verbosity.detailed_verbosity() && kmeans_iter % 5 == 0 {
                println!("  K-means iter {}: obj = {:.4}", kmeans_iter + 1, obj);
            }

            if kmeans_iter > params.window_size
                && window_converged(
                    &objectives_kmeans,
                    params.window_size,
                    params.epsilon_kmeans,
                    false,
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

        times.time("ridge", || {
            ridge_correction(&z_orig, &r, &batch_infos, ridge, k, d, &mut z_corr)
        });
        times.time("normalise", || normalise_rows_into(&z_corr, d, &mut z_cos));

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
        if (obj_old - harmony_obj) / obj_old.abs() < params.epsilon_harmony {
            if verbosity.normal_verbosity() {
                println!("\nHarmony converged at iteration {}", harmony_iter + 1);
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

/// Run Harmony batch correction with one or more batch variables.
///
/// Each element of `batch_labels` is a slice of length N giving the level
/// assignments for one categorical variable. For example, to correct for
/// both sample and technology:
///
/// ### Params
///
/// * `pca` - PCA embedding (N x d)
/// * `batch_labels` - one label slice per variable, each of length N
/// * `params` - Harmony hyperparameters
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Corrected PCA embedding (N x d)
pub fn harmony(
    pca: MatRef<f32>,
    batch_labels: &[Vec<usize>],
    params: &HarmonyParams,
    seed: usize,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors> {
    Ok(harmony_with_state(pca, batch_labels, params, seed, verbose)?.z_corr)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_create_batch_info() {
        let labels = vec![0, 0, 1, 1, 2, 0];
        let info = create_batch_info(&labels, 6).unwrap();

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
    fn test_create_batch_infos_multiple() {
        let var0 = vec![0, 0, 1, 1];
        let var1 = vec![0, 1, 0, 1];
        let infos = create_batch_infos(&[var0, var1], 4).unwrap();

        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].n_levels, 2);
        assert_eq!(infos[1].n_levels, 2);
        assert_eq!(infos[0].batch_indices[0], vec![0, 1]);
        assert_eq!(infos[0].batch_indices[1], vec![2, 3]);
        assert_eq!(infos[1].batch_indices[0], vec![0, 2]);
        assert_eq!(infos[1].batch_indices[1], vec![1, 3]);
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

        let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma).unwrap();

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

        let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma).unwrap();
        assert!(r[(1, 0)] > r[(0, 0)]);
    }
}
