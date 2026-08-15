//! GPU implementation of the Harmony v2 version. Only supports the arrowhead
//! version, i.e., single batch effect to regress out.

#![allow(missing_docs)]

use ann_search_rs::gpu::k_means_gpu::{
    KMeansGpuParams, build_csr_gpu_privatised, k_means_clusters_gpu,
};
use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use cubek::matmul::definition::MatmulPrecision;
use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::time::Instant;

use crate::gpu::linalg::cholesky_gpu::dense_gemm;
use crate::gpu::linalg::spmm::launch_dense_column_sum;
use crate::gpu::sc_gpu::kernels::harmony_kernels::*;
use crate::gpu::{WORKGROUP_64, WORKGROUP_512};
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::cosine_normalise;
use crate::single_cell::sc_batch_correction::harmony::{BatchInfo, create_batch_infos};
use crate::single_cell::sc_batch_correction::harmony_v2::{
    check_convergence, expand_theta, solve_arrowhead, solve_lu,
};

///////////
// Types //
///////////

/// Arrowhead solutions
type Solution = (Vec<usize>, Mat<f32>);

///////////
// Utils //
///////////

/// Flatten a column-major faer matrix into a row-major `Vec<f32>`.
///
/// ### Params
///
/// * `m` - The matrix to flatten
///
/// ### Returns
///
/// The row-major flattened data
fn to_row_major(m: MatRef<f32>) -> Vec<f32> {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = m[(i, j)];
        }
    }
    out
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
    /// Number of workgroups for the privatised CSR histogram. Use the same
    /// value as the GPU k-means driver.
    pub csr_cube_count: usize,
    /// GPU k-means parameters; `None` uses the k-means defaults.
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
            use_dynamic_lambda: false,
            csr_cube_count: 256,
            kmeans_params: None,
        }
    }
}

///////////////////
// Orchestrators //
///////////////////

/////////
// CPU //
/////////

/// Host-side arrowhead ridge solve (single covariate), producing the
/// per-`(cluster, level)` correction tensor `C` `[k, b, d]`.
///
/// Replaces the cell-iteration phase 1 of `ridge_regression_correction_v2`:
/// the arrowhead `design` is built from `O` and `r_sum`, and `phi_z` from the
/// weighted segmented sums `S`. Each cluster prunes low-occupancy levels,
/// assembles its `p x p` arrowhead system and `p x d` right-hand side, and
/// solves via `solve_arrowhead` (falling back to `solve_lu` on a degenerate
/// Schur complement). Clusters with at most one passing level produce no
/// correction. The intercept row of `W` is not copied into `C`.
///
/// ### Params
///
/// * `o_host` - Observed counts `[b * k]`, `o[level * k + cluster]`
/// * `r_sum_host` - Per-cluster totals `[k]`
/// * `s_host` - Weighted segmented sums `[b * k * d]`,
///   `s[(level * k + cluster) * d + feat]`
/// * `batch_info` - Batch information for the single covariate
/// * `k` - Cluster count
/// * `d` - Feature dimension
/// * `lambda` - Fixed ridge penalty (used when `use_dynamic_lambda` is false)
/// * `alpha` - Dynamic lambda multiplier (`lambda_kb = alpha * E_kb`)
/// * `use_dynamic_lambda` - Whether to use dynamic or fixed lambda
/// * `batch_proportion_cutoff` - Minimum `O[k,b] / N_b` to include a level
///
/// ### Returns
///
/// `C` `[k * b * d]` row-major, `c[cluster * b * d + level * d + feat]`; zero
/// for pruned levels and clusters with no correction.
#[allow(clippy::too_many_arguments)]
pub fn solve_ridge_arrowhead_host(
    o_host: &[f32],
    r_sum_host: &[f32],
    s_host: &[f32],
    batch_info: &BatchInfo,
    k: usize,
    d: usize,
    lambda: f32,
    alpha: f32,
    use_dynamic_lambda: bool,
    batch_proportion_cutoff: f32,
) -> Vec<f32> {
    let b = batch_info.n_levels;
    let pr_b = &batch_info.pr_b;

    let solutions: Vec<Option<Solution>> = (0..k)
        .into_par_iter()
        .map(|cluster| {
            // pruning: levels whose mean assignment clears the cutoff
            let mut passing: Vec<usize> = Vec::new();
            for level in 0..b {
                let n_cells = batch_info.batch_indices[level].len();
                if n_cells == 0 {
                    continue;
                }
                let avg_r = o_host[level * k + cluster] / n_cells as f32;
                if avg_r > batch_proportion_cutoff {
                    passing.push(level);
                }
            }
            // single covariate is active only with more than one passing level
            if passing.len() <= 1 {
                return None;
            }

            let p = 1 + passing.len();
            let mut design = Mat::<f32>::zeros(p, p);
            let mut phi_z = Mat::<f32>::zeros(p, d);

            // intercept: all cells. design[0,0] = r_sum; phi_z[0] = sum_b S
            design[(0, 0)] = r_sum_host[cluster];
            for level in 0..b {
                let s_off = (level * k + cluster) * d;
                for feat in 0..d {
                    phi_z[(0, feat)] += s_host[s_off + feat];
                }
            }

            // arrowhead arms: one batch column per passing level
            for (col_off, &level) in passing.iter().enumerate() {
                let cc = 1 + col_off;
                let o_val = o_host[level * k + cluster];
                design[(0, cc)] = o_val;
                design[(cc, 0)] = o_val;
                design[(cc, cc)] = o_val;
                let s_off = (level * k + cluster) * d;
                for feat in 0..d {
                    phi_z[(cc, feat)] = s_host[s_off + feat];
                }
            }

            // ridge penalty on the batch diagonal (intercept unpenalised)
            if use_dynamic_lambda {
                for (col_off, &level) in passing.iter().enumerate() {
                    let e_val = r_sum_host[cluster] * pr_b[level];
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

    let mut c = vec![0.0f32; k * b * d];
    for (cluster, sol) in solutions.into_iter().enumerate() {
        if let Some((passing, w)) = sol {
            for (col_off, level) in passing.into_iter().enumerate() {
                let row = 1 + col_off;
                let c_off = cluster * b * d + level * d;
                for feat in 0..d {
                    c[c_off + feat] = w[(row, feat)];
                }
            }
        }
    }

    c
}

/////////
// GPU //
/////////

/// Centroid update: `Y = normalise(R * z_cos)`.
///
/// `r` is stored `[n, k]`, so the GEMM reads it transposed as logical `[k, n]`
/// and contracts against `z_cos` `[n, dim]` to produce `y_out` `[k, dim]`,
/// which is then row L2-normalised in place.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `r` - Soft assignments `[n, k]`
/// * `z_cos` - Cosine-normalised data `[n, dim]`
/// * `y_out` - Output centroids `[k, dim]`, overwritten
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `dim` - Embedding dimension
///
/// ### Errors
///
/// * `GpuMatmul` if the GEMM dispatch fails.
/// * `CubeclUtils` if the row-normalise grid is over the device limit.
pub fn update_centroids_from_r_gpu<R, T, MP>(
    client: &ComputeClient<R>,
    r: &GpuTensor<R, T>,
    z_cos: &GpuTensor<R, T>,
    y_out: &GpuTensor<R, T>,
    n: usize,
    k: usize,
    dim: usize,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement,
    MP: MatmulPrecision,
{
    dense_gemm::<R, MP>(
        r.handle(),
        [k, n],
        true, // storage [n, k] read as logical [k, n]
        z_cos.handle(),
        [n, dim],
        false,
        y_out.handle(),
        [k, dim],
        None,
        client,
    )?;

    launch_row_l2_normalise(y_out, k, dim, client)?;

    Ok(())
}

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

/// Ridge correction (single covariate, arrowhead), end to end.
///
/// Computes the weighted segmented sums `S` on the GPU, reads `S`, `O`, and
/// `r_sum` back, solves the per-cluster arrowhead systems on the host, uploads
/// the correction tensor `C`, and subtracts it. `o` and `r_sum` are the
/// statistics of the current `r` (from the last Jacobi sweep). Pinned to `f32`
/// for the host solve.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `z` - Original data `[n, d]`
/// * `r` - Soft assignments `[n, k]`
/// * `o` - Observed counts `[b, k]`
/// * `r_sum` - Per-cluster totals `[k]`
/// * `all_indices` - Level-CSR cell order `[n]`
/// * `offsets` - Level-CSR offsets `[b + 1]`
/// * `cell_to_level` - Level index per cell `[n]`
/// * `z_corr` - Output corrected data `[n, d]`, overwritten
/// * `batch_info` - Batch information for the single covariate
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `b` - Number of levels
/// * `d` - Feature dimension
/// * `lambda`, `alpha`, `use_dynamic_lambda`, `batch_proportion_cutoff` - Ridge
///   parameters, see [`solve_ridge_arrowhead_host`]
///
/// ### Errors
///
/// * `CubeclUtils` if a dispatch grid or an allocation busts a device limit,
///   or a read-back of `S`, `O` or `r_sum` fails.
#[allow(clippy::too_many_arguments)]
pub fn ridge_correction_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    z: &GpuTensor<R, f32>,
    r: &GpuTensor<R, f32>,
    o: &GpuTensor<R, f32>,
    r_sum: &GpuTensor<R, f32>,
    all_indices: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    cell_to_level: &GpuTensor<R, u32>,
    z_corr: &GpuTensor<R, f32>,
    batch_info: &BatchInfo,
    n: usize,
    k: usize,
    b: usize,
    d: usize,
    lambda: f32,
    alpha: f32,
    use_dynamic_lambda: bool,
    batch_proportion_cutoff: f32,
) -> Result<(), BixverseErrors> {
    let s = GpuTensor::<R, f32>::empty(vec![b * k * d], client)?;
    launch_weighted_segmented_sum(r, z, all_indices, offsets, &s, b, k, d, client)?;

    let s_host = s.clone().read(client)?;
    let o_host = o.clone().read(client)?;
    let r_sum_host = r_sum.clone().read(client)?;

    let c_host = solve_ridge_arrowhead_host(
        &o_host,
        &r_sum_host,
        &s_host,
        batch_info,
        k,
        d,
        lambda,
        alpha,
        use_dynamic_lambda,
        batch_proportion_cutoff,
    );

    let c = GpuTensor::<R, f32>::from_slice(&c_host, vec![k * b * d], client)?;
    launch_ridge_subtract(z, r, &c, cell_to_level, z_corr, n, k, b, d, client)?;

    Ok(())
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

        let mut acc = F::new(0.0);
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
                dst[base + e] = F::new(0.0);
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

/// GPU Harmony v2 (single covariate).
///
/// Outer loop mirrors the CPU `harmony_v2`: per round, compute the fixed base
/// assignments, refine R with the diversity penalty over `max_iter_kmeans`
/// full-batch Jacobi sweeps (rebuilding O and `r_sum` after each), apply the
/// arrowhead ridge correction, then update centroids and distances. Pinned to
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
/// * Propagates k-means, GEMM, read-back and device-limit (`CubeclUtils`)
///   errors.
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

    let batch_infos = create_batch_infos(batch_labels, n)?;
    let info = &batch_infos[0];
    let b = info.n_levels;

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

    let lambda_scalar = params.lambda[0];

    // host prep: original and cosine-normalised embeddings
    let z_orig = pca.to_owned();
    let z_cos_host = cosine_normalise(&z_orig);

    // initial centroids via GPU k-means (returns host centroids + assignments)
    if verbosity.normal_verbosity() {
        println!("GPU Harmony v2: running initial k-means");
    }
    let (y_host, _) = k_means_clusters_gpu::<f32, R>(
        z_cos_host.as_ref(),
        "cosine",
        k,
        params.kmeans_params,
        seed,
        device.clone(),
        verbosity.detailed_verbosity(),
    )?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start.elapsed());
    }

    let client = R::client(&device);

    // upload resident buffers
    if verbosity.detailed_verbosity() {
        println!("GPU Harmony v2: moving data to GPU.");
    }
    let z_orig_gpu =
        GpuTensor::<R, f32>::from_slice(&to_row_major(z_orig.as_ref()), vec![n, d], &client)?;
    let z_cos_gpu =
        GpuTensor::<R, f32>::from_slice(&to_row_major(z_cos_host.as_ref()), vec![n, d], &client)?;
    let y_gpu =
        GpuTensor::<R, f32>::from_slice(&to_row_major(y_host.as_ref()), vec![k, d], &client)?;

    let sigma_gpu = GpuTensor::<R, f32>::from_slice(&sigma, vec![k], &client)?;
    let theta_gpu = GpuTensor::<R, f32>::from_slice(theta_levels, vec![b], &client)?;
    let pr_b_gpu = GpuTensor::<R, f32>::from_slice(&info.pr_b, vec![b], &client)?;

    let cell_to_level_u32: Vec<u32> = info.cell_to_level.iter().map(|&l| l as u32).collect();
    let cell_to_level_gpu = GpuTensor::<R, u32>::from_slice(&cell_to_level_u32, vec![n], &client)?;

    // build the level-CSR once (cell_to_level is static)
    let cc = params.csr_cube_count;
    let privatised_counts =
        GpuTensor::<R, u32>::from_slice(&vec![0u32; cc * b], vec![cc * b], &client)?;
    let counts = GpuTensor::<R, u32>::from_slice(&vec![0u32; b], vec![b], &client)?;
    let offsets = GpuTensor::<R, u32>::from_slice(&vec![0u32; b + 1], vec![b + 1], &client)?;
    let all_indices = GpuTensor::<R, u32>::from_slice(&vec![0u32; n], vec![n], &client)?;
    build_csr_gpu_privatised(
        &cell_to_level_gpu,
        n,
        b,
        cc,
        &privatised_counts,
        &counts,
        &offsets,
        &all_indices,
        &client,
    )?;

    // working buffers
    let dist_gpu = GpuTensor::<R, f32>::empty(vec![n, k], &client)?;
    let scale_dist_gpu = GpuTensor::<R, f32>::empty(vec![n, k], &client)?;
    let r_gpu = GpuTensor::<R, f32>::empty(vec![n, k], &client)?;
    let o_gpu = GpuTensor::<R, f32>::empty(vec![b, k], &client)?;
    let r_sum_gpu = GpuTensor::<R, f32>::empty(vec![k], &client)?;
    let z_corr_gpu = GpuTensor::<R, f32>::empty(vec![n, d], &client)?;

    if verbosity.detailed_verbosity() {
        println!(" ... done in {:.2?}", start.elapsed());
    }

    // initial distances, R, and statistics
    launch_cosine_distances(&y_gpu, &z_cos_gpu, &dist_gpu, n, k, d, &client)?;
    launch_scale_exp_normalise(&dist_gpu, &sigma_gpu, &r_gpu, n, k, &client)?;
    launch_dense_column_sum(&r_gpu, &r_sum_gpu, n, k, &client)?;
    launch_segmented_sum(&r_gpu, &all_indices, &offsets, &o_gpu, b, k, &client)?;

    let initial_obj = compute_objective_gpu(
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
    )?;

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
            launch_dense_column_sum(&r_gpu, &r_sum_gpu, n, k, &client)?;
            launch_segmented_sum(&r_gpu, &all_indices, &offsets, &o_gpu, b, k, &client)?;

            let obj = compute_objective_gpu(
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
            )?;
            objectives_kmeans.push(obj);

            if kmeans_iter >= params.window_size
                && check_convergence(
                    &objectives_kmeans,
                    params.window_size,
                    params.epsilon_kmeans,
                )
            {
                break;
            }
        }

        // ridge regression with batch pruning
        if verbosity.normal_verbosity() {
            println!("  Applying ridge regression correction...");
        }

        // arrowhead ridge correction -> z_corr
        ridge_correction_gpu(
            &client,
            &z_orig_gpu,
            &r_gpu,
            &o_gpu,
            &r_sum_gpu,
            &all_indices,
            &offsets,
            &cell_to_level_gpu,
            &z_corr_gpu,
            info,
            n,
            k,
            b,
            d,
            lambda_scalar,
            params.alpha,
            params.use_dynamic_lambda,
            params.batch_proportion_cutoff,
        )?;

        // z_cos = normalise(z_corr) (out-of-place: z_corr is the returned value)
        launch_row_l2_normalise_into(&z_corr_gpu, &z_cos_gpu, n, d, &client)?;

        // update centroids and distances for the next round
        update_centroids_from_r_gpu::<R, f32, f32>(&client, &r_gpu, &z_cos_gpu, &y_gpu, n, k, d)?;
        launch_cosine_distances(&y_gpu, &z_cos_gpu, &dist_gpu, n, k, d, &client)?;

        // re-initialise R and statistics from the new distances
        launch_scale_exp_normalise(&dist_gpu, &sigma_gpu, &r_gpu, n, k, &client)?;
        launch_dense_column_sum(&r_gpu, &r_sum_gpu, n, k, &client)?;
        launch_segmented_sum(&r_gpu, &all_indices, &offsets, &o_gpu, b, k, &client)?;

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

        if harmony_iter >= 1 {
            let obj_old = objectives_harmony[objectives_harmony.len() - 2];
            let obj_new = objectives_harmony[objectives_harmony.len() - 1];
            let rel_change = (obj_old - obj_new) / obj_old.abs();
            if rel_change < params.epsilon_harmony {
                if verbosity.normal_verbosity() {
                    println!(
                        " GPU Harmony v2 converged at iteration {}",
                        harmony_iter + 1
                    );
                }
                break;
            }
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "Finished the GPU-accelerated Harmony (version 2) in {:.2?}",
            start.elapsed()
        )
    }

    let z_corr_host = z_corr_gpu.clone().read(&client)?;

    Ok(Mat::from_fn(n, d, |i, j| z_corr_host[i * d + j]))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests_harmony_gpu {
    use super::*;
    use approx::assert_relative_eq;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    use crate::single_cell::sc_batch_correction::harmony::{
        compute_diversity_statistics, create_batch_info,
    };
    use crate::single_cell::sc_batch_correction::harmony_v2::{
        compute_objective_v2, ridge_regression_correction_v2,
    };

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    fn make_normalised_row_major(n: usize, dim: usize) -> Vec<f32> {
        let mut data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.1 + 0.05)
            .collect();
        for row in 0..n {
            let base = row * dim;
            let norm: f32 = (0..dim).map(|e| data[base + e].powi(2)).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for e in 0..dim {
                    data[base + e] /= norm;
                }
            }
        }
        data
    }

    // Row-major [n, d] -> faer Mat<f32> with shape [n, d].
    fn row_major_to_mat_nd(data: &[f32], n: usize, d: usize) -> Mat<f32> {
        Mat::<f32>::from_fn(n, d, |i, j| data[i * d + j])
    }

    /// Pins the flattening every GPU upload in this file depends on.
    #[test]
    fn test_to_row_major_roundtrip() {
        let m = Mat::<f32>::from_fn(3, 4, |i, j| (i * 10 + j) as f32);
        let flat = to_row_major(m.as_ref());
        assert_eq!(flat.len(), 12);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(flat[i * 4 + j], m[(i, j)]);
            }
        }
    }

    /// A near-singular arrowhead system must give exactly zero, not NaN.
    #[test]
    fn test_solve_ridge_arrowhead_host_single_passing_returns_zero() {
        let (n, k, b, d) = (4, 2, 2, 3);
        let labels = vec![0usize, 0, 1, 1];
        let info = create_batch_info(&labels, n).unwrap();

        let r_cpu = faer::mat![
            [0.9f32, 0.9, 1e-7, 1e-7],
            [0.1, 0.1, 1.0 - 1e-7, 1.0 - 1e-7],
        ];

        let mut o_host = vec![0.0f32; b * k];
        for cell in 0..n {
            let level = info.cell_to_level[cell];
            for cluster in 0..k {
                o_host[level * k + cluster] += r_cpu[(cluster, cell)];
            }
        }
        let mut r_sum_host = vec![0.0f32; k];
        for cluster in 0..k {
            for cell in 0..n {
                r_sum_host[cluster] += r_cpu[(cluster, cell)];
            }
        }
        let s_host = vec![0.0f32; b * k * d];

        let c = solve_ridge_arrowhead_host(
            &o_host,
            &r_sum_host,
            &s_host,
            &info,
            k,
            d,
            1.0,
            0.2,
            false,
            0.01,
        );

        for v in &c {
            assert_eq!(*v, 0.0);
        }
    }

    /// Pins the GPU-layout packing of O, r_sum and S against the CPU ridge.
    #[test]
    fn test_solve_ridge_arrowhead_host_matches_cpu_ridge() {
        // Single-covariate case: solve_ridge_arrowhead_host must match
        // ridge_regression_correction_v2 (which uses arrowhead internally).
        let (n, k, d) = (12, 3, 4);
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect(); // 3 levels
        let info = create_batch_info(&labels, n).unwrap();
        let b = info.n_levels;

        // R: dense, no near-zero rows, columns sum to 1.
        let r_cols: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut col: Vec<f32> = (0..k)
                    .map(|cl| ((i * 7 + cl * 3 + 1) % 9) as f32 + 0.1)
                    .collect();
                let s: f32 = col.iter().sum();
                for v in col.iter_mut() {
                    *v /= s;
                }
                col
            })
            .collect();
        let r_cpu = Mat::<f32>::from_fn(k, n, |cluster, cell| r_cols[cell][cluster]);

        let z_orig = Mat::<f32>::from_fn(n, d, |i, j| ((i * 5 + j * 11) % 17) as f32 * 0.1);

        let oe = vec![compute_diversity_statistics(r_cpu.as_ref(), &info)];

        // Reference: CPU ridge correction
        let z_corr_cpu = ridge_regression_correction_v2(
            z_orig.as_ref(),
            r_cpu.as_ref(),
            std::slice::from_ref(&info),
            &oe,
            1.0,
            0.2,
            false,
            1e-5,
        );

        // Now: build the arrowhead host inputs in GPU layout and call solve_ridge_arrowhead_host
        // O: [b, k] row-major; r_sum: [k]; S: [b * k * d] row-major.
        let mut o_host = vec![0.0f32; b * k];
        for cell in 0..n {
            let level = info.cell_to_level[cell];
            for cluster in 0..k {
                o_host[level * k + cluster] += r_cpu[(cluster, cell)];
            }
        }
        let r_sum_host: Vec<f32> = (0..k)
            .map(|cluster| (0..n).map(|cell| r_cpu[(cluster, cell)]).sum())
            .collect();
        let mut s_host = vec![0.0f32; b * k * d];
        for cell in 0..n {
            let level = info.cell_to_level[cell];
            for cluster in 0..k {
                let r_val = r_cpu[(cluster, cell)];
                for feat in 0..d {
                    s_host[(level * k + cluster) * d + feat] += r_val * z_orig[(cell, feat)];
                }
            }
        }

        let c = solve_ridge_arrowhead_host(
            &o_host,
            &r_sum_host,
            &s_host,
            &info,
            k,
            d,
            1.0,
            0.2,
            false,
            1e-5,
        );

        // Apply correction on host using ridge_subtract semantics
        let mut z_corr_gpu = vec![0.0f32; n * d];
        for cell in 0..n {
            let level = info.cell_to_level[cell];
            for feat in 0..d {
                let mut acc = z_orig[(cell, feat)];
                for cluster in 0..k {
                    let r_val = r_cpu[(cluster, cell)];
                    acc -= r_val * c[cluster * b * d + level * d + feat];
                }
                z_corr_gpu[cell * d + feat] = acc;
            }
        }

        for cell in 0..n {
            for feat in 0..d {
                assert_relative_eq!(
                    z_corr_gpu[cell * d + feat],
                    z_corr_cpu[(cell, feat)],
                    epsilon = 1e-4
                );
            }
        }
    }

    /// Parity gate for the objective that drives the convergence check.
    #[test]
    fn test_compute_objective_gpu_matches_cpu_v2() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, d) = (16, 4, 6);
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let info = create_batch_info(&labels, n).unwrap();
        let b = info.n_levels;

        // Build a dense, row-normalised R (cols sum to 1 in CPU layout)
        let r_cols: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut col: Vec<f32> = (0..k)
                    .map(|cl| ((i * 11 + cl * 7 + 1) % 13) as f32 + 0.1)
                    .collect();
                let s: f32 = col.iter().sum();
                for v in col.iter_mut() {
                    *v /= s;
                }
                col
            })
            .collect();
        let r_cpu = Mat::<f32>::from_fn(k, n, |cluster, cell| r_cols[cell][cluster]);

        // Distance matrix (random-ish, non-negative)
        let dist_cpu = Mat::<f32>::from_fn(k, n, |cluster, cell| {
            ((cell * 13 + cluster * 5 + 3) % 17) as f32 * 0.05
        });

        let sigma = vec![0.1f32; k];
        let theta_expanded = vec![vec![1.0f32; b]];
        let oe = vec![compute_diversity_statistics(r_cpu.as_ref(), &info)];

        let cpu_obj = compute_objective_v2(
            r_cpu.as_ref(),
            dist_cpu.as_ref(),
            &oe,
            &sigma,
            &theta_expanded,
            std::slice::from_ref(&info),
        );

        // GPU layout: r [n, k], dist [n, k], o [b, k]
        let mut r_host = vec![0.0f32; n * k];
        let mut dist_host = vec![0.0f32; n * k];
        for cell in 0..n {
            for cluster in 0..k {
                r_host[cell * k + cluster] = r_cpu[(cluster, cell)];
                dist_host[cell * k + cluster] = dist_cpu[(cluster, cell)];
            }
        }
        let mut o_host = vec![0.0f32; b * k];
        for cell in 0..n {
            let level = info.cell_to_level[cell];
            for cluster in 0..k {
                o_host[level * k + cluster] += r_cpu[(cluster, cell)];
            }
        }
        let r_sum_host: Vec<f32> = (0..k)
            .map(|cluster| (0..n).map(|cell| r_cpu[(cluster, cell)]).sum())
            .collect();
        let cell_to_level_u32: Vec<u32> = info.cell_to_level.iter().map(|&l| l as u32).collect();

        let r_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&r_host, vec![n, k], &client).unwrap();
        let dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&dist_host, vec![n, k], &client).unwrap();
        let o_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&o_host, vec![b, k], &client).unwrap();
        let r_sum_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&r_sum_host, vec![k], &client).unwrap();
        let sigma_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&sigma, vec![k], &client).unwrap();
        let theta_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&theta_expanded[0], vec![b], &client)
                .unwrap();
        let pr_b_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&info.pr_b, vec![b], &client).unwrap();
        let ctl_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&cell_to_level_u32, vec![n], &client)
                .unwrap();

        let gpu_obj = compute_objective_gpu::<WgpuRuntime>(
            &client, &r_gpu, &dist_gpu, &o_gpu, &r_sum_gpu, &sigma_gpu, &theta_gpu, &pr_b_gpu,
            &ctl_gpu, n, k,
        )
        .unwrap();

        // Tolerance: f32 tree reduction vs CPU parallel reduction can diverge a bit.
        assert_relative_eq!(gpu_obj, cpu_obj, epsilon = 1e-2);
        let _ = d; // d unused here
    }

    /// The centroid kernel must emit unit-norm rows pointing like host R*Z.
    #[test]
    fn test_update_centroids_from_r_gpu_basic() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, d) = (8, 3, 4);
        let z_cos = make_normalised_row_major(n, d);
        let r_host: Vec<f32> = (0..n)
            .flat_map(|cell| {
                let raw: Vec<f32> = (0..k)
                    .map(|cl| ((cell * 5 + cl + 1) % 7) as f32 + 0.2)
                    .collect();
                let s: f32 = raw.iter().sum();
                raw.into_iter().map(move |v| v / s)
            })
            .collect();

        let r_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&r_host, vec![n, k], &client).unwrap();
        let z_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&z_cos, vec![n, d], &client).unwrap();
        let y_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![k, d], &client).unwrap();

        update_centroids_from_r_gpu::<WgpuRuntime, f32, f32>(
            &client, &r_gpu, &z_gpu, &y_gpu, n, k, d,
        )
        .unwrap();

        let y_host = y_gpu.read(&client).unwrap();

        // Each centroid row has unit L2 norm (or is zero if it collapsed).
        for cluster in 0..k {
            let base = cluster * d;
            let norm: f32 = (0..d).map(|e| y_host[base + e].powi(2)).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4 || norm < 1e-6,
                "centroid {} has norm {}",
                cluster,
                norm
            );
        }

        // Compare direction against CPU R*Z then normalise.
        let mut y_cpu = vec![0.0f32; k * d];
        for cell in 0..n {
            for cluster in 0..k {
                let r_val = r_host[cell * k + cluster];
                for feat in 0..d {
                    y_cpu[cluster * d + feat] += r_val * z_cos[cell * d + feat];
                }
            }
        }
        for cluster in 0..k {
            let base = cluster * d;
            let norm: f32 = (0..d).map(|e| y_cpu[base + e].powi(2)).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for e in 0..d {
                    y_cpu[base + e] /= norm;
                }
            }
        }
        for i in 0..k * d {
            assert_relative_eq!(y_host[i], y_cpu[i], epsilon = 1e-3);
        }
    }

    /// Must match the host, zero a zero-norm row, and leave the source alone.
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
            if norm > 1e-8 {
                for e in 0..dim {
                    assert_relative_eq!(dst[base + e], src[base + e] / norm, epsilon = 1e-5);
                }
            } else {
                for e in 0..dim {
                    assert_eq!(dst[base + e], 0.0);
                }
            }
        }

        // src is unmodified
        let src_back = src_gpu.read(&client).unwrap();
        for i in 0..n * dim {
            assert_eq!(src_back[i], src[i]);
        }
    }

    /// End-to-end on two shifted batches: right shape, all entries finite.
    #[test]
    fn test_harmony_v2_gpu_smoke() {
        let Some(device) = try_device() else { return };

        let (n, d) = (40, 5);
        // Two batches with a deliberate shift in feature 0
        let mut data = vec![0.0f32; n * d];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let batch = if i < n / 2 { 0 } else { 1 };
            labels[i] = batch;
            for j in 0..d {
                let base = ((i * 7 + j * 3) % 11) as f32 * 0.1;
                let shift = if j == 0 && batch == 1 { 3.0 } else { 0.0 };
                data[i * d + j] = base + shift;
            }
        }
        let pca = row_major_to_mat_nd(&data, n, d);

        let params = HarmonyParamsV2Gpu {
            k: 4,
            sigma: vec![0.1],
            theta: vec![2.0],
            lambda: vec![1.0],
            max_iter_kmeans: 2,
            max_iter_harmony: 2,
            window_size: 3,
            csr_cube_count: 64,
            ..HarmonyParamsV2Gpu::default()
        };

        let result = harmony_v2_gpu::<WgpuRuntime>(pca.as_ref(), &[labels], &params, 42, device, 0)
            .expect("harmony_v2_gpu should succeed");

        assert_eq!(result.nrows(), n);
        assert_eq!(result.ncols(), d);
        for i in 0..n {
            for j in 0..d {
                assert!(result[(i, j)].is_finite(), "non-finite at ({}, {})", i, j);
            }
        }
    }
}
