//! GPU implementation of Harmony v2 for a single batch covariate: full-batch
//! Jacobi assignment updates and the arrowhead ridge correction, with the
//! per-cluster systems solved on the host by the shared CPU solver.

#![allow(missing_docs)]

use ann_search_rs::gpu::k_means_gpu::{KMeansGpuParams, k_means_clusters_gpu};
use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use faer::{Mat, MatRef};
use std::time::Instant;

use crate::gpu::linalg::spmm::launch_dense_column_sum;
use crate::gpu::sc_gpu::kernels::harmony_kernels::*;
use crate::gpu::{WORKGROUP_64, WORKGROUP_512};
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::harmony::{BatchInfo, create_batch_infos};
use crate::single_cell::sc_batch_correction::harmony_core::{
    HARMONY_KMEANS_ITERS, RidgeSettings, normalise_rows_into, row_major_to_mat, solve_ridge,
    to_row_major,
};
use crate::single_cell::sc_batch_correction::harmony_v2::{check_convergence, expand_theta};

////////////
// Consts //
////////////

/// Cells per run for the per-level reductions. Each `(run, cluster)`
/// workgroup of the weighted sum loops over at most this many cells, so the
/// work per workgroup is bounded however unbalanced the levels are.
pub const HARMONY_RUN_CELLS: usize = 1024;

///////////////
// LevelRuns //
///////////////

/// Cells sorted by level, and the run structure over them.
struct LevelRuns {
    /// Original cell index of each sorted position (length n)
    order: Vec<usize>,
    /// Level of each sorted position (length n)
    sorted_labels: Vec<usize>,
    /// Sorted-position offsets of each run (length n_runs + 1)
    run_bounds: Vec<u32>,
    /// Run offsets of each level (length b + 1)
    level_runs: Vec<u32>,
}

/// Sort cells by level and cut every level into runs of at most
/// [`HARMONY_RUN_CELLS`] cells.
///
/// ### Params
///
/// * `info` - Batch information for the single covariate
///
/// ### Returns
///
/// The [`LevelRuns`]
fn level_runs(info: &BatchInfo) -> LevelRuns {
    let order: Vec<usize> = info.batch_indices.concat();
    let sorted_labels: Vec<usize> = order.iter().map(|&c| info.cell_to_level[c]).collect();
    let mut run_bounds = vec![0u32];
    let mut level_runs = vec![0u32];
    let mut pos = 0usize;
    for cells in &info.batch_indices {
        let mut left = cells.len();
        while left > 0 {
            let take = left.min(HARMONY_RUN_CELLS);
            pos += take;
            left -= take;
            run_bounds.push(pos as u32);
        }
        level_runs.push((run_bounds.len() - 1) as u32);
    }
    LevelRuns {
        order,
        sorted_labels,
        run_bounds,
        level_runs,
    }
}

////////////
// Params //
////////////

/// Parameters for GPU Harmony (version 2, single covariate, arrowhead,
/// full-batch Jacobi). Mirrors `HarmonyParamsV2` minus `block_size` (the Jacobi
/// update has no blocks), with the GPU k-means parameters.
pub struct HarmonyParamsV2Gpu {
    /// Number of clusters
    pub k: usize,
    /// Per-cluster diversity weights (length 1 or K)
    pub sigma: Vec<f32>,
    /// Per-variable diversity penalties (length 1 or n_variables)
    pub theta: Vec<f32>,
    /// Ridge penalty (length 1)
    pub lambda: Vec<f32>,
    /// Maximum diversity-refinement (Jacobi) sweeps per Harmony round
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
    /// GPU k-means parameters; `None` uses the k-means defaults rather than
    /// the Harmony default of `HARMONY_KMEANS_ITERS` iterations
    pub kmeans_params: Option<KMeansGpuParams>,
}

/// Default implementation
impl Default for HarmonyParamsV2Gpu {
    fn default() -> Self {
        Self {
            k: 100,
            sigma: vec![0.1],
            theta: vec![2.0],
            lambda: vec![1.0],
            max_iter_kmeans: 4,
            max_iter_harmony: 10,
            epsilon_kmeans: 1e-3,
            epsilon_harmony: 1e-2,
            window_size: 3,
            alpha: 0.2,
            tau: 0.0,
            batch_proportion_cutoff: 1e-5,
            use_dynamic_lambda: true,
            kmeans_params: Some(KMeansGpuParams::new(
                HARMONY_KMEANS_ITERS,
                None,
                true,
                false,
            )),
        }
    }
}

/////////
// GPU //
/////////

/// Harmony v2 objective (single covariate), reduced to a host scalar.
///
/// Launches `objective_partials`, reads the `OBJ_BLOCKS` partials back, sums
/// them, and applies the `2000 / N` constant. Pinned to `f32`: Harmony is
/// f32-only and the result is a host scalar. The host sum and the GPU tree
/// reduction differ in order from the CPU's parallel reduction, so a CPU
/// comparison needs a tolerance rather than an exact match.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `r` - Soft assignments `[n, k]`
/// * `dist` - Distance matrix `[n, k]`
/// * `o` - Observed counts `[b, k]`
/// * `r_sum` - Per-cluster totals `[k]`
/// * `sigma` - Per-cluster weights `[k]`
/// * `theta` - Per-level theta for the single covariate `[b]`
/// * `pr_b` - Level frequencies `[b]`
/// * `cell_to_level` - Level index per cell `[n]`
/// * `n` - Cell count
/// * `k` - Cluster count
///
/// ### Errors
///
/// * `CubeclUtils` if the partials allocation busts the per-binding size
///   limit, or the read-back fails on the device.
#[allow(clippy::too_many_arguments)]
pub fn compute_objective_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    r: &GpuTensor<R, f32>,
    dist: &GpuTensor<R, f32>,
    o: &GpuTensor<R, f32>,
    r_sum: &GpuTensor<R, f32>,
    sigma: &GpuTensor<R, f32>,
    theta: &GpuTensor<R, f32>,
    pr_b: &GpuTensor<R, f32>,
    cell_to_level: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
) -> Result<f32, BixverseErrors> {
    let partials = GpuTensor::<R, f32>::empty(vec![WORKGROUP_512 as usize], client)?;

    launch_objective_partials(
        r,
        dist,
        o,
        r_sum,
        sigma,
        theta,
        pr_b,
        cell_to_level,
        &partials,
        n,
        k,
        client,
    );

    let host = partials.clone().read(client)?;
    let sum: f32 = host.iter().sum();

    Ok(sum * (2000.0 / n as f32))
}

///////////////////////////
// Out-of-place row norm //
///////////////////////////

/// Out-of-place row L2-normalisation: `dst[row, :] = src[row, :] / ||src[row,
/// :]||`, or zeros if the norm is below `1e-8`. Identical to
/// `row_l2_normalise` but preserves `src`; used for `z_cos =
/// normalise(z_corr)` in the driver, where `z_corr` is the returned value and
/// must survive.
///
/// ### Params
///
/// * `src` - Input matrix `[n_rows, dim]` row-major
/// * `dst` - Output matrix `[n_rows, dim]` row-major
/// * `n_rows` - Number of rows
/// * `dim` - Row length (comptime)
#[cube(launch_unchecked)]
pub fn row_l2_normalise_into<F: Float>(
    src: &Tensor<F>,
    dst: &mut Tensor<F>,
    n_rows: u32,
    #[comptime] dim: usize,
) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if row >= n_rows {
        terminate!();
    }
    if UNIT_POS_X == 0u32 {
        let base = row as usize * dim;

        let mut acc = F::new(0.0_f32);
        for e in 0..dim {
            let v = src[base + e];
            acc += v * v;
        }
        let norm = F::sqrt(acc);

        if norm > F::new(1e-8) {
            for e in 0..dim {
                dst[base + e] = src[base + e] / norm;
            }
        } else {
            for e in 0..dim {
                dst[base + e] = F::new(0.0_f32);
            }
        }
    }
}

/// Dispatch [`fn@row_l2_normalise_into`]. One workgroup per row.
///
/// The out-of-place sibling of [`launch_row_l2_normalise`], for the ridge
/// correction where the source `z_corr` is the value returned to the caller and
/// so cannot be normalised in place.
///
/// ### Params
///
/// * `src` - Input matrix `[n_rows, dim]` row-major, left untouched
/// * `dst` - Output matrix `[n_rows, dim]` row-major
/// * `n_rows` - Number of rows
/// * `dim` - Row length
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `dst` holds the row-normalised copy. `CubeclUtils` if the grid is
/// over the device's cube-count limit.
pub fn launch_row_l2_normalise_into<R, F>(
    src: &GpuTensor<R, F>,
    dst: &GpuTensor<R, F>,
    n_rows: usize,
    dim: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n_rows as u32, &limits)?;

    unsafe {
        row_l2_normalise_into::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_64),
            src.clone().into_tensor_arg(),
            dst.clone().into_tensor_arg(),
            n_rows as u32,
            dim,
        );
    }

    Ok(())
}

//////////
// Main //
//////////

/// Per-level observed counts `O` `[b, k]` and per-cluster totals `r_sum` of
/// the current assignments.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `r` - Soft assignments `[n, k]`, cells sorted by level
/// * `run_bounds` - Run cell offsets `[n_runs + 1]`
/// * `level_runs` - Run offsets per level `[b + 1]`
/// * `partial` - Scratch `[n_runs, k]`
/// * `o` - Output observed counts `[b, k]`
/// * `r_sum` - Output per-cluster totals `[k]`
/// * `n` - Cell count
/// * `n_runs` - Number of runs
/// * `b` - Number of levels
/// * `k` - Cluster count
///
/// ### Errors
///
/// * `CubeclUtils` if a dispatch grid is over the device limit.
#[allow(clippy::too_many_arguments)]
fn observed_counts_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    r: &GpuTensor<R, f32>,
    run_bounds: &GpuTensor<R, u32>,
    level_runs: &GpuTensor<R, u32>,
    partial: &GpuTensor<R, f32>,
    o: &GpuTensor<R, f32>,
    r_sum: &GpuTensor<R, f32>,
    n: usize,
    n_runs: usize,
    b: usize,
    k: usize,
) -> Result<(), BixverseErrors> {
    launch_run_sums(r, run_bounds, partial, n_runs, k, client)?;
    launch_reduce_runs(partial, level_runs, o, b, k, client)?;
    launch_dense_column_sum(r, r_sum, n, k, client)?;
    Ok(())
}

/// GPU Harmony v2 (single covariate).
///
/// Mirrors the CPU `harmony_v2`: per round, refine R with the diversity
/// penalty over `max_iter_kmeans` full-batch Jacobi sweeps at fixed distances,
/// apply the batch-pruned ridge correction, take the next centroids from the
/// normalised ridge intercepts, and restart R from the new distances.
///
/// Cells are sorted by batch level once on upload, so every level is a
/// contiguous range cut into runs of at most [`HARMONY_RUN_CELLS`]; the per-level
/// sums are per-run partials plus a small reduction, with no index gather and
/// bounded work per workgroup. The Jacobi sweep has no visiting order, so the
/// sort changes nothing but summation order; the output is unsorted on
/// readback. The per-cluster ridge systems are solved on the host in f64 by
/// `harmony_core::solve_ridge`, the same solver as the CPU path. Pinned to
/// `f32`.
///
/// ### Params
///
/// * `pca` - PCA embedding `[n, d]`
/// * `batch_labels` - One label slice of length `n`; exactly one variable is
///   supported on the GPU path
/// * `params` - GPU Harmony v2 hyperparameters
/// * `seed` - Random seed for the initial k-means
/// * `device` - CubeCL runtime device
/// * `verbose` - Whether to print progress
///
/// ### Returns
///
/// Corrected PCA embedding `[n, d]`.
///
/// ### Errors
///
/// * Propagates k-means, read-back and device-limit (`CubeclUtils`) errors.
pub fn harmony_v2_gpu<R: Runtime>(
    pca: MatRef<f32>,
    batch_labels: &[Vec<usize>],
    params: &HarmonyParamsV2Gpu,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors>
where
    R::Device: Clone,
{
    let verbosity = parse_verbosity_level(verbose);
    let n = pca.nrows();
    let d = pca.ncols();
    let k = params.k;
    let start = Instant::now();

    // early return upon error
    if batch_labels.len() != 1 {
        return Err(BixverseErrors::GpuHarmonySupportsSingleCovariateOnly);
    }

    // sort cells by level; everything below works in sorted order
    let runs = level_runs(&create_batch_infos(batch_labels, n)?[0]);
    let batch_infos = create_batch_infos(std::slice::from_ref(&runs.sorted_labels), n)?;
    let info = &batch_infos[0];
    let b = info.n_levels;
    let n_runs = runs.run_bounds.len() - 1;

    let sigma = if params.sigma.len() == 1 {
        vec![params.sigma[0]; k]
    } else {
        assert_eq!(params.sigma.len(), k, "sigma must be length 1 or K");
        params.sigma.clone()
    };

    let theta = if params.theta.len() == 1 {
        vec![params.theta[0]; 1]
    } else {
        assert_eq!(
            params.theta.len(),
            1,
            "theta must be length 1 (single covariate)"
        );
        params.theta.clone()
    };
    let theta_expanded = expand_theta(&theta, &batch_infos, k, params.tau);
    let theta_levels = &theta_expanded[0];

    let ridge = RidgeSettings {
        lambda: params.lambda[0],
        alpha: params.alpha,
        dynamic_lambda: params.use_dynamic_lambda,
        prune_cutoff: Some(params.batch_proportion_cutoff),
    };

    // host prep: sorted original and cosine-normalised embeddings
    let pca_rm = to_row_major(pca);
    let mut z_orig = vec![0.0f32; n * d];
    for (i, &c) in runs.order.iter().enumerate() {
        z_orig[i * d..(i + 1) * d].copy_from_slice(&pca_rm[c * d..(c + 1) * d]);
    }
    let mut z_cos_host = vec![0.0f32; n * d];
    normalise_rows_into(&z_orig, d, &mut z_cos_host);

    // initial centroids via GPU k-means
    if verbosity.normal_verbosity() {
        println!("GPU Harmony v2: running initial k-means");
    }
    let (y_mat, _) = k_means_clusters_gpu::<f32, R>(
        (z_cos_host.as_slice(), n, d),
        "cosine",
        k,
        params.kmeans_params,
        seed,
        device.clone(),
        verbosity.detailed_verbosity(),
    )?;
    let mut y_host = vec![0.0f32; k * d];
    normalise_rows_into(&to_row_major(y_mat.as_ref()), d, &mut y_host);

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start.elapsed());
    }

    let client = R::client(&device);

    // resident buffers
    let z_orig_gpu = GpuTensor::<R, f32>::from_slice(&z_orig, vec![n, d], &client)?;
    let z_cos_gpu = GpuTensor::<R, f32>::from_slice(&z_cos_host, vec![n, d], &client)?;
    let mut y_gpu = GpuTensor::<R, f32>::from_slice(&y_host, vec![k, d], &client)?;
    let sigma_gpu = GpuTensor::<R, f32>::from_slice(&sigma, vec![k], &client)?;
    let theta_gpu = GpuTensor::<R, f32>::from_slice(theta_levels, vec![b], &client)?;
    let pr_b_gpu = GpuTensor::<R, f32>::from_slice(&info.pr_b, vec![b], &client)?;
    let cell_to_level_u32: Vec<u32> = runs.sorted_labels.iter().map(|&l| l as u32).collect();
    let cell_to_level_gpu = GpuTensor::<R, u32>::from_slice(&cell_to_level_u32, vec![n], &client)?;
    let run_bounds_gpu =
        GpuTensor::<R, u32>::from_slice(&runs.run_bounds, vec![n_runs + 1], &client)?;
    let level_runs_gpu = GpuTensor::<R, u32>::from_slice(&runs.level_runs, vec![b + 1], &client)?;

    // working buffers
    let dist_gpu = GpuTensor::<R, f32>::empty(vec![n, k], &client)?;
    let scale_dist_gpu = GpuTensor::<R, f32>::empty(vec![n, k], &client)?;
    let r_gpu = GpuTensor::<R, f32>::empty(vec![n, k], &client)?;
    let o_gpu = GpuTensor::<R, f32>::empty(vec![b, k], &client)?;
    let r_sum_gpu = GpuTensor::<R, f32>::empty(vec![k], &client)?;
    let z_corr_gpu = GpuTensor::<R, f32>::empty(vec![n, d], &client)?;
    let partial_o_gpu = GpuTensor::<R, f32>::empty(vec![n_runs, k], &client)?;
    let partial_s_gpu = GpuTensor::<R, f32>::empty(vec![n_runs * k * d], &client)?;
    let s_gpu = GpuTensor::<R, f32>::empty(vec![b * k * d], &client)?;

    let stats = |r: &GpuTensor<R, f32>| {
        observed_counts_gpu(
            &client,
            r,
            &run_bounds_gpu,
            &level_runs_gpu,
            &partial_o_gpu,
            &o_gpu,
            &r_sum_gpu,
            n,
            n_runs,
            b,
            k,
        )
    };
    let objective = || {
        compute_objective_gpu(
            &client,
            &r_gpu,
            &dist_gpu,
            &o_gpu,
            &r_sum_gpu,
            &sigma_gpu,
            &theta_gpu,
            &pr_b_gpu,
            &cell_to_level_gpu,
            n,
            k,
        )
    };

    // initial distances, R, and statistics
    launch_cosine_distances(&y_gpu, &z_cos_gpu, &dist_gpu, n, k, d, &client)?;
    launch_scale_exp_normalise(&dist_gpu, &sigma_gpu, &r_gpu, n, k, &client)?;
    stats(&r_gpu)?;

    let initial_obj = objective()?;
    let mut objectives_kmeans: Vec<f32> = vec![initial_obj];
    let mut objectives_harmony: Vec<f32> = vec![initial_obj];

    if verbosity.normal_verbosity() {
        println!("GPU Harmony v2: initial objective {:.4}", initial_obj);
    }

    for harmony_iter in 0..params.max_iter_harmony {
        if verbosity.normal_verbosity() {
            println!("\n=== Harmony GPU v2 iteration {} ===", harmony_iter + 1);
            println!("  Running k-means clustering...");
        }

        let start_iter = Instant::now();

        // base assignments, fixed across the inner loop
        launch_scale_exp_normalise(&dist_gpu, &sigma_gpu, &scale_dist_gpu, n, k, &client)?;

        for kmeans_iter in 0..params.max_iter_kmeans {
            // Jacobi sweep: reads O/r_sum from the previous sweep, writes R in place
            launch_jacobi_r_update(
                &scale_dist_gpu,
                &o_gpu,
                &r_sum_gpu,
                &theta_gpu,
                &pr_b_gpu,
                &cell_to_level_gpu,
                &r_gpu,
                n,
                k,
                &client,
            )?;
            stats(&r_gpu)?;

            objectives_kmeans.push(objective()?);

            if kmeans_iter > params.window_size
                && check_convergence(
                    &objectives_kmeans,
                    params.window_size,
                    params.epsilon_kmeans,
                )
            {
                break;
            }
        }

        if verbosity.normal_verbosity() {
            println!("  Applying ridge regression correction...");
        }

        // per-level sums on the device, systems solved on the host in f64
        launch_run_weighted_sums(
            &r_gpu,
            &z_orig_gpu,
            &run_bounds_gpu,
            &partial_s_gpu,
            n_runs,
            k,
            d,
            &client,
        )?;
        launch_reduce_runs(&partial_s_gpu, &level_runs_gpu, &s_gpu, b, k * d, &client)?;
        let s_host = s_gpu.clone().read(&client)?;
        let o_host = o_gpu.clone().read(&client)?;
        let sums = [(
            s_host.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
            o_host.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
        )];
        let out = solve_ridge(&sums, &[], &batch_infos, ridge, k, d);

        let c_gpu = GpuTensor::<R, f32>::from_slice(&out.corr[0], vec![b * k * d], &client)?;
        launch_ridge_subtract(
            &z_orig_gpu,
            &r_gpu,
            &c_gpu,
            &cell_to_level_gpu,
            &z_corr_gpu,
            n,
            k,
            d,
            &client,
        )?;

        // next centroids are the normalised ridge intercepts
        let mut intercept_norm = vec![0.0f32; k * d];
        normalise_rows_into(&out.intercept, d, &mut intercept_norm);
        for (kk, &solved) in out.solved.iter().enumerate() {
            if solved {
                y_host[kk * d..(kk + 1) * d].copy_from_slice(&intercept_norm[kk * d..(kk + 1) * d]);
            }
        }
        y_gpu = GpuTensor::<R, f32>::from_slice(&y_host, vec![k, d], &client)?;

        // z_cos = normalise(z_corr) (out-of-place: z_corr is the returned value)
        launch_row_l2_normalise_into(&z_corr_gpu, &z_cos_gpu, n, d, &client)?;
        launch_cosine_distances(&y_gpu, &z_cos_gpu, &dist_gpu, n, k, d, &client)?;

        // cold restart of R and statistics from the new distances
        launch_scale_exp_normalise(&dist_gpu, &sigma_gpu, &r_gpu, n, k, &client)?;
        stats(&r_gpu)?;

        let harmony_obj = *objectives_kmeans.last().unwrap();
        objectives_harmony.push(harmony_obj);

        if verbosity.normal_verbosity() {
            println!(
                "  GPU Harmony v2: iteration {} objective {:.4}",
                harmony_iter + 1,
                harmony_obj
            );
            println!(
                "   Finished iteration in {:.2?} / Total runtime {:.2?}",
                start_iter.elapsed(),
                start.elapsed()
            );
        }

        let obj_old = objectives_harmony[objectives_harmony.len() - 2];
        if (obj_old - harmony_obj) / obj_old.abs() < params.epsilon_harmony {
            if verbosity.normal_verbosity() {
                println!(
                    " GPU Harmony v2 converged at iteration {}",
                    harmony_iter + 1
                );
            }
            break;
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "Finished the GPU-accelerated Harmony (version 2) in {:.2?}",
            start.elapsed()
        )
    }

    // back to the caller's cell order
    let z_sorted = z_corr_gpu.clone().read(&client)?;
    let mut z_out = vec![0.0f32; n * d];
    for (i, &c) in runs.order.iter().enumerate() {
        z_out[c * d..(c + 1) * d].copy_from_slice(&z_sorted[i * d..(i + 1) * d]);
    }
    Ok(row_major_to_mat(&z_out, d))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests_harmony_gpu {
    use super::*;
    use approx::assert_relative_eq;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    use crate::single_cell::sc_batch_correction::harmony::create_batch_info;
    use crate::single_cell::sc_batch_correction::harmony_core::{Variant, objective};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Sorting groups each level's cells contiguously and cuts runs at the
    /// run length, with empty levels owning no runs.
    #[test]
    fn test_level_runs_structure() {
        let n = 2 * HARMONY_RUN_CELLS + 5;
        // level 0: every third cell, level 2: the rest, level 1 empty
        let labels: Vec<usize> = (0..n).map(|c| if c % 3 == 0 { 0 } else { 2 }).collect();
        let info = create_batch_info(&labels, n).unwrap();
        let runs = level_runs(&info);

        let n0 = labels.iter().filter(|&&l| l == 0).count();
        assert!(runs.sorted_labels[..n0].iter().all(|&l| l == 0));
        assert!(runs.sorted_labels[n0..].iter().all(|&l| l == 2));
        assert_eq!(*runs.run_bounds.last().unwrap() as usize, n);
        assert_eq!(runs.level_runs.len(), 4);
        assert_eq!(
            runs.level_runs[1], runs.level_runs[2],
            "empty level owns no runs"
        );
        for w in runs.run_bounds.windows(2) {
            assert!((w[1] - w[0]) as usize <= HARMONY_RUN_CELLS);
        }
        let mut seen = runs.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>());
    }

    /// Parity gate for the objective that drives the convergence check.
    #[test]
    fn test_compute_objective_gpu_matches_cpu_v2() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k) = (16, 4);
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let info = create_batch_info(&labels, n).unwrap();
        let b = info.n_levels;

        let mut r = vec![0.0f32; n * k];
        for c in 0..n {
            let row: Vec<f32> = (0..k)
                .map(|cl| ((c * 11 + cl * 7 + 1) % 13) as f32 + 0.1)
                .collect();
            let s: f32 = row.iter().sum();
            for cl in 0..k {
                r[c * k + cl] = row[cl] / s;
            }
        }
        let dist: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 + 3) % 17) as f32 * 0.05)
            .collect();
        let sigma = vec![0.1f32; k];
        let theta_levels = vec![vec![1.0f32; b]];

        let mut o = vec![0.0f32; b * k];
        let mut r_sum = vec![0.0f32; k];
        let (mut err, mut ent) = (0.0f64, 0.0f64);
        for c in 0..n {
            for cl in 0..k {
                let v = r[c * k + cl];
                o[info.cell_to_level[c] * k + cl] += v;
                r_sum[cl] += v;
                err += (v * dist[c * k + cl]) as f64;
                ent += (sigma[cl] * v * v.ln()) as f64;
            }
        }
        let cpu_obj = objective(
            Variant::V2,
            err,
            ent,
            std::slice::from_ref(&o),
            &r_sum,
            std::slice::from_ref(&info),
            &sigma,
            &theta_levels,
            n,
        );

        let up = |x: &[f32], shape: Vec<usize>| {
            GpuTensor::<WgpuRuntime, f32>::from_slice(x, shape, &client).unwrap()
        };
        let ctl: Vec<u32> = info.cell_to_level.iter().map(|&l| l as u32).collect();
        let ctl_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&ctl, vec![n], &client).unwrap();
        let gpu_obj = compute_objective_gpu::<WgpuRuntime>(
            &client,
            &up(&r, vec![n, k]),
            &up(&dist, vec![n, k]),
            &up(&o, vec![b, k]),
            &up(&r_sum, vec![k]),
            &up(&sigma, vec![k]),
            &up(&theta_levels[0], vec![b]),
            &up(&info.pr_b, vec![b]),
            &ctl_gpu,
            n,
            k,
        )
        .unwrap();

        assert_relative_eq!(gpu_obj, cpu_obj, epsilon = 1e-2);
    }

    /// The out-of-place row norm matches the host and leaves the source alone.
    #[test]
    fn test_row_l2_normalise_into_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, dim) = (5, 6);
        let src: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let src_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&src, vec![n, dim], &client).unwrap();
        let dst_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, dim], &client).unwrap();

        launch_row_l2_normalise_into(&src_gpu, &dst_gpu, n, dim, &client).unwrap();
        let dst = dst_gpu.read(&client).unwrap();

        for row in 0..n {
            let base = row * dim;
            let norm: f32 = (0..dim).map(|e| src[base + e].powi(2)).sum::<f32>().sqrt();
            for e in 0..dim {
                let want = if norm > 1e-8 {
                    src[base + e] / norm
                } else {
                    0.0
                };
                assert_relative_eq!(dst[base + e], want, epsilon = 1e-5);
            }
        }
        assert_eq!(src_gpu.read(&client).unwrap(), src);
    }

    /// End-to-end on two shifted, interleaved batches: the output is in the
    /// caller's cell order (the sort is undone) and the shift shrinks.
    #[test]
    fn test_harmony_v2_gpu_removes_interleaved_shift() {
        let Some(device) = try_device() else { return };

        let (n, d) = (60, 5);
        let mut data = vec![0.0f32; n * d];
        let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
        for i in 0..n {
            for j in 0..d {
                let base = ((i * 7 + j * 3) % 11) as f32 * 0.1 + 0.2;
                let shift = if j == 0 && labels[i] == 1 { 3.0 } else { 0.0 };
                data[i * d + j] = base + shift;
            }
        }
        let pca = row_major_to_mat(&data, d);

        let params = HarmonyParamsV2Gpu {
            k: 3,
            max_iter_kmeans: 2,
            max_iter_harmony: 3,
            ..HarmonyParamsV2Gpu::default()
        };
        let result = harmony_v2_gpu::<WgpuRuntime>(
            pca.as_ref(),
            std::slice::from_ref(&labels),
            &params,
            42,
            device,
            0,
        )
        .expect("harmony_v2_gpu should succeed");

        assert_eq!((result.nrows(), result.ncols()), (n, d));
        assert!((0..n).all(|i| (0..d).all(|j| result[(i, j)].is_finite())));

        let gap = |m: &dyn Fn(usize) -> f32| {
            let mean = |lvl: usize| {
                let v: Vec<f32> = (0..n).filter(|&i| labels[i] == lvl).map(m).collect();
                v.iter().sum::<f32>() / v.len() as f32
            };
            (mean(1) - mean(0)).abs()
        };
        let before = gap(&|i| data[i * d]);
        let after = gap(&|i| result[(i, 0)]);
        assert!(after < 0.5 * before, "shift {before} -> {after}");
    }
}
