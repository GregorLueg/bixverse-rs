//! GPU kernels for Harmony v2 (single-covariate, arrowhead path).
//!
//! Layout convention: every K-per-cell matrix (`dist`, `scale_dist`, `R`) is
//! stored `[N, K]` row-major on the GPU, transposed relative to the CPU's
//! `[K, N]`. Each per-cell kernel then reads K contiguous values, and
//! `R * z_cos` is expressed as a `dense_gemm` with the left operand
//! transposed (storage `[N, K]` read as logical `[K, N]`).

#![allow(missing_docs)]

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;
use cubecl::prelude::*;

use crate::gpu::*;

/////////////
// Kernels //
/////////////

// TODO: check if SharedMemory would be more performant ... ?

/// Cosine distances between centroids and cells: `dist[n, k] = 2 * (1 - dot)`.
///
/// Both `centroids` and `data_cos` must be cosine-normalised, so the dot
/// product is the cosine similarity. One workgroup per cell; threads stride
/// over the K clusters. `data_cos[n, :]` is reread from global per cluster
/// (contiguous, L1-cached) rather than staged in shared memory.
///
/// ### Params
///
/// * `centroids` - Centroids `[k, dim]` row-major, cosine-normalised
/// * `data_cos` - Cells `[n, dim]` row-major, cosine-normalised
/// * `dist` - Output `[n, k]` row-major
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `dim` - Embedding dimension (comptime)
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> cell index
/// * `UNIT_POS_X` -> stride offset over clusters
#[cube(launch_unchecked)]
fn cosine_distances<F: Float>(
    centroids: &Tensor<F>,
    data_cos: &Tensor<F>,
    dist: &mut Tensor<F>,
    n: u32,
    k: u32,
    #[comptime] dim: usize,
    #[comptime] wg_size: u32,
) {
    let cell = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cell >= n {
        terminate!();
    }

    let z_base = cell as usize * dim;

    let mut cluster = UNIT_POS_X;
    while cluster < k {
        let y_base = cluster as usize * dim;
        let mut dot = F::new(0.0);
        for e in 0..dim {
            dot += data_cos[z_base + e] * centroids[y_base + e];
        }
        dist[cell as usize * k as usize + cluster as usize] = F::new(2.0) * (F::new(1.0) - dot);
        cluster += wg_size;
    }
}

/// In-place row L2-normalisation: each row scaled to unit norm, or zeroed if
/// its norm is below `1e-8`.
///
/// One workgroup per row; only thread 0 is active (mirrors `centroid_norms_l2`,
/// `dim` is small). Serves both the centroid update (`Y`, `[k, dim]`) and the
/// data cosine-normalisation (`z`, `[n, dim]`).
///
/// ### Params
///
/// * `data` - Matrix `[n_rows, dim]` row-major, normalised in place
/// * `n_rows` - Number of rows
/// * `dim` - Row length (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> row index
#[cube(launch_unchecked)]
fn row_l2_normalise<F: Float>(data: &mut Tensor<F>, n_rows: u32, #[comptime] dim: usize) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if row >= n_rows {
        terminate!();
    }
    if UNIT_POS_X == 0u32 {
        let base = row as usize * dim;

        let mut acc = F::new(0.0);
        for e in 0..dim {
            let v = data[base + e];
            acc += v * v;
        }
        let norm = F::sqrt(acc);

        if norm > F::new(1e-8) {
            for e in 0..dim {
                data[base + e] = data[base + e] / norm;
            }
        } else {
            for e in 0..dim {
                data[base + e] = F::new(0.0);
            }
        }
    }
}

/// Column-normalised `exp(-dist / sigma)` per cell, serving both
/// `compute_scaled_distances` and `initialise_r_from_dist` (identical ops).
///
/// One workgroup per cell; threads stride over the K clusters. Pass 1
/// accumulates `sum_k exp(-dist[n,k] / sigma[k])` via a workgroup tree
/// reduction; pass 2 recomputes each `exp` and writes the normalised value.
/// The recompute avoids a cross-thread global read-after-write, since
/// `sync_cube` orders shared but not necessarily storage memory. A cell whose
/// terms all underflow to zero is written as zeros (matches the CPU guard).
///
/// ### Params
///
/// * `dist` - Distance matrix `[n, k]` row-major
/// * `sigma` - Per-cluster weights `[k]`
/// * `out` - Output `[n, k]` row-major (scale_dist or initial R)
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `wg_size` - Workgroup size (comptime, power of two)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> cell index
/// * `UNIT_POS_X` -> stride offset over clusters
#[cube(launch_unchecked)]
fn scale_exp_normalise<F: Float>(
    dist: &Tensor<F>,
    sigma: &Tensor<F>,
    out: &mut Tensor<F>,
    n: u32,
    k: u32,
    #[comptime] wg_size: u32,
) {
    let cell = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cell >= n {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let base = cell as usize * k as usize;

    // pass 1: partial sum of exp(-dist / sigma) over this thread's clusters
    let mut acc = F::new(0.0);
    let mut kk = tx;
    while kk < k {
        let arg = (F::new(0.0) - dist[base + kk as usize]) / sigma[kk as usize];
        acc += F::exp(arg);
        kk += wg_size;
    }

    // tree reduction (manually unrolled for WORKGROUP_128)
    let mut shared = SharedMemory::<F>::new(WORKGROUP_128 as usize);
    shared[tx as usize] = acc;
    sync_cube();
    if tx < 64u32 {
        let v = shared[(tx + 64u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 32u32 {
        let v = shared[(tx + 32u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 16u32 {
        let v = shared[(tx + 16u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 8u32 {
        let v = shared[(tx + 8u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 4u32 {
        let v = shared[(tx + 4u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 2u32 {
        let v = shared[(tx + 2u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 1u32 {
        let v = shared[(tx + 1u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();

    let total = shared[0];

    // pass 2: recompute exp and normalise, or write zeros if all underflowed
    let mut kk2 = tx;
    if total > F::new(0.0) {
        while kk2 < k {
            let idx = base + kk2 as usize;
            let arg = (F::new(0.0) - dist[idx]) / sigma[kk2 as usize];
            out[idx] = F::exp(arg) / total;
            kk2 += wg_size;
        }
    } else {
        while kk2 < k {
            out[base + kk2 as usize] = F::new(0.0);
            kk2 += wg_size;
        }
    }
}

/// Observed counts via segmented sum of R's rows grouped by level:
/// `O[b, :] = sum_{cells at level b} R[cell, :]`.
///
/// One workgroup per level; thread `tx` owns clusters `tx, tx + wg, ...` and
/// sums each over the level's segment. Mirrors `segmented_centroid_update` but
/// accumulates a plain sum (not a mean) and writes zeros for empty levels
/// rather than skipping them, matching the CPU `O[k,b]` semantics. Per-segment
/// accumulation is naive serial, as in the centroid update; the CSR route is
/// chosen for the large-B regime, so segments are short.
///
/// Expected counts E are not materialised: every consumer derives
/// `E[k,b] = r_sum[k] * pr_b[b]` on the fly from `r_sum` (a `dense_column_sum`
/// of R) and the uploaded `pr_b`.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]` row-major (R)
/// * `all_indices` - Cell indices in CSR order `[n]` from
///   `build_csr_gpu_privatised` with level labels
/// * `offsets` - Exclusive prefix sums `[b + 1]`; level `l` occupies
///   `all_indices[offsets[l]..offsets[l + 1]]`
/// * `o` - Output observed counts `[b, k]` row-major, written in place
/// * `b` - Number of levels
/// * `k` - Number of clusters (feature dim, comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> level index
/// * `UNIT_POS_X` -> cluster stride offset within the level's O row
#[cube(launch_unchecked)]
fn segmented_sum<F: Float>(
    r: &Tensor<F>,
    all_indices: &Tensor<u32>,
    offsets: &Tensor<u32>,
    o: &mut Tensor<F>,
    b: u32,
    #[comptime] k: usize,
) {
    let level = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if level >= b {
        terminate!();
    }

    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    let seg_start = offsets[level as usize];
    let seg_end = offsets[(level + 1u32) as usize];
    let count = seg_end - seg_start;

    let o_base = level as usize * k;

    let mut e = tx;
    while e < k {
        let mut acc = F::new(0.0);
        let mut p = 0u32;
        while p < count {
            let global = all_indices[(seg_start + p) as usize];
            acc += r[global as usize * k + e];
            p += 1u32;
        }
        o[o_base + e] = acc;
        e += wg;
    }
}

/// Per-workgroup partial sums of the Harmony v2 objective (single covariate).
///
/// Objective = `(kmeans_error + entropy + cross_entropy) * 2000 / N`. This
/// kernel emits the unscaled per-workgroup partials of the inner sum over
/// cells; the host sums them and applies `2000 / N`. A fixed grid of
/// `OBJ_BLOCKS` workgroups grid-strides over cells, accumulating
///
/// `sum_k [ r * dist + (r > 0) r ln(r) sigma_k + r sigma_k theta_l ln_ratio ]`
///
/// per cell, where `ln_ratio = ln((O + E + 1) / (2E + 1))` and
/// `E = r_sum[k] * pr_b[level]` is derived inline. `log_ratio` is recomputed
/// per cell rather than tabulated; the objective is a convergence check, not
/// hot-path.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]` row-major
/// * `dist` - Distance matrix `[n, k]` row-major
/// * `o` - Observed counts `[b, k]` row-major
/// * `r_sum` - Per-cluster totals `[k]` (column sums of R)
/// * `sigma` - Per-cluster weights `[k]`
/// * `theta` - Per-level theta for the single covariate `[b]`
/// * `pr_b` - Level frequencies `[b]`
/// * `cell_to_level` - Level index per cell `[n]`
/// * `partials` - Output partial sums `[OBJ_BLOCKS]`, one per workgroup
/// * `n` - Cell count
/// * `k` - Cluster count (comptime)
/// * `wg_size` - Workgroup size (comptime, power of two)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> workgroup / partial index (1D grid)
/// * `CUBE_POS_X * wg_size + UNIT_POS_X` -> first cell, then grid-stride
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn objective_partials<F: Float>(
    r: &Tensor<F>,
    dist: &Tensor<F>,
    o: &Tensor<F>,
    r_sum: &Tensor<F>,
    sigma: &Tensor<F>,
    theta: &Tensor<F>,
    pr_b: &Tensor<F>,
    cell_to_level: &Tensor<u32>,
    partials: &mut Tensor<F>,
    n: u32,
    #[comptime] k: usize,
    #[comptime] wg_size: u32,
) {
    let tx = UNIT_POS_X;
    let wid = CUBE_POS_X;
    let stride = CUBE_COUNT_X * wg_size;

    let mut acc = F::new(0.0);
    let mut cell = wid * wg_size + tx;
    while cell < n {
        let level = cell_to_level[cell as usize] as usize;
        let theta_l = theta[level];
        let pr = pr_b[level];
        let r_base = cell as usize * k;
        let o_base = level * k;

        for cl in 0..k {
            let r_val = r[r_base + cl];
            acc += r_val * dist[r_base + cl];
            if r_val > F::new(0.0) {
                acc += r_val * F::ln(r_val) * sigma[cl];
            }
            let e_val = r_sum[cl] * pr;
            let ln_ratio =
                F::ln((o[o_base + cl] + e_val + F::new(1.0)) / (F::new(2.0) * e_val + F::new(1.0)));
            acc += r_val * sigma[cl] * theta_l * ln_ratio;
        }

        cell += stride;
    }

    // tree reduction within workgroup (manully unrolled for 128)
    let mut shared = SharedMemory::<F>::new(WORKGROUP_128 as usize);
    shared[tx as usize] = acc;
    sync_cube();
    if tx < 64u32 {
        let v = shared[(tx + 64u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 32u32 {
        let v = shared[(tx + 32u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 16u32 {
        let v = shared[(tx + 16u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 8u32 {
        let v = shared[(tx + 8u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 4u32 {
        let v = shared[(tx + 4u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 2u32 {
        let v = shared[(tx + 2u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 1u32 {
        let v = shared[(tx + 1u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();

    if tx == 0u32 {
        partials[wid as usize] = shared[0];
    }
}

/// Full-batch Jacobi R-update (single covariate).
///
/// One workgroup per cell; threads stride over the K clusters. Pass 1
/// accumulates `sum_k scale_dist[n,k] * penalty` via a workgroup tree
/// reduction, where `penalty = ((2E + 1) / (O[k,level] + E + 1))^theta_l` and
/// `E = r_sum[k] * pr_b[level]`. Pass 2 recomputes and normalises. O and
/// `r_sum` are read from the previous sweep (no block removal); this differs
/// from the CPU block update by design. The kernel never reads R, so writing
/// R in place is safe. The penalty is recomputed in pass 2 to avoid a
/// cross-thread global read-after-write (`sync_cube` orders shared, not
/// necessarily storage); this costs a second `powf` per element.
///
/// ### Params
///
/// * `scale_dist` - Column-normalised base assignments `[n, k]`, fixed per round
/// * `o` - Observed counts `[b, k]` from the previous sweep
/// * `r_sum` - Per-cluster totals `[k]` from the previous sweep
/// * `theta` - Per-level theta for the single covariate `[b]`
/// * `pr_b` - Level frequencies `[b]`
/// * `cell_to_level` - Level index per cell `[n]`
/// * `r_out` - Output R `[n, k]`, written in place
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `wg_size` - Workgroup size (comptime, power of two)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> cell index
/// * `UNIT_POS_X` -> stride offset over clusters
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn jacobi_r_update<F: Float>(
    scale_dist: &Tensor<F>,
    o: &Tensor<F>,
    r_sum: &Tensor<F>,
    theta: &Tensor<F>,
    pr_b: &Tensor<F>,
    cell_to_level: &Tensor<u32>,
    r_out: &mut Tensor<F>,
    n: u32,
    k: u32,
    #[comptime] wg_size: u32,
) {
    let cell = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cell >= n {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let level = cell_to_level[cell as usize] as usize;
    let theta_l = theta[level];
    let pr = pr_b[level];
    let cell_base = cell as usize * k as usize;
    let o_base = level * k as usize;

    // pass 1: partial sum of base * penalty over this thread's clusters
    let mut acc = F::new(0.0);
    let mut kk = tx;
    while kk < k {
        let e_val = r_sum[kk as usize] * pr;
        let ratio =
            (F::new(2.0) * e_val + F::new(1.0)) / (o[o_base + kk as usize] + e_val + F::new(1.0));
        let penalty = F::powf(ratio, theta_l);
        acc += scale_dist[cell_base + kk as usize] * penalty;
        kk += wg_size;
    }

    // tree reduction (unrolled for RUPD_WG = 128)
    let mut shared = SharedMemory::<F>::new(WORKGROUP_128 as usize);
    shared[tx as usize] = acc;
    sync_cube();
    if tx < 64u32 {
        let v = shared[(tx + 64u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 32u32 {
        let v = shared[(tx + 32u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 16u32 {
        let v = shared[(tx + 16u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 8u32 {
        let v = shared[(tx + 8u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 4u32 {
        let v = shared[(tx + 4u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 2u32 {
        let v = shared[(tx + 2u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();
    if tx < 1u32 {
        let v = shared[(tx + 1u32) as usize];
        shared[tx as usize] += v;
    }
    sync_cube();

    let total = shared[0];

    // pass 2: recompute and normalise, or write zeros if the column collapsed
    let mut kk2 = tx;
    if total > F::new(0.0) {
        while kk2 < k {
            let e_val = r_sum[kk2 as usize] * pr;
            let ratio = (F::new(2.0) * e_val + F::new(1.0))
                / (o[o_base + kk2 as usize] + e_val + F::new(1.0));
            let penalty = F::powf(ratio, theta_l);
            let val = scale_dist[cell_base + kk2 as usize] * penalty;
            r_out[cell_base + kk2 as usize] = val / total;
            kk2 += wg_size;
        }
    } else {
        while kk2 < k {
            r_out[cell_base + kk2 as usize] = F::new(0.0);
            kk2 += wg_size;
        }
    }
}

/// Weighted segmented sum for the ridge `phi_z`: `S[b, k, :] = sum_{cells at
/// level b} R[cell, k] * z[cell, :]`.
///
/// One workgroup per `(level, cluster)` pair; threads stride over the feature
/// dimension `d`. Output `S` is `[b * k, d]` row-major, indexed by
/// `(level * k + cluster) * d + feat`. `R[cell, k]` is read once per cell and
/// broadcast across the feature threads (L1); `z[cell, :]` reads are
/// coalesced across `feat`.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]` row-major
/// * `z` - Original (uncorrected) data `[n, d]` row-major
/// * `all_indices` - Cell indices in CSR order `[n]` from
///   `build_csr_gpu_privatised`
/// * `offsets` - Exclusive prefix sums `[b + 1]`
/// * `s` - Output `[b * k, d]` row-major
/// * `b` - Number of levels
/// * `k` - Cluster count
/// * `d` - Feature dimension (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> `level * k + cluster`
/// * `UNIT_POS_X` -> feature stride offset
#[cube(launch_unchecked)]
fn weighted_segmented_sum<F: Float>(
    r: &Tensor<F>,
    z: &Tensor<F>,
    all_indices: &Tensor<u32>,
    offsets: &Tensor<u32>,
    s: &mut Tensor<F>,
    b: u32,
    k: u32,
    #[comptime] d: usize,
) {
    let wid = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if wid >= b * k {
        terminate!();
    }

    let level = wid / k;
    let cluster = wid % k;

    let seg_start = offsets[level as usize];
    let seg_end = offsets[(level + 1u32) as usize];
    let count = seg_end - seg_start;

    let s_base = wid as usize * d;

    let mut feat = UNIT_POS_X as usize;
    let wg = WORKGROUP_128 as usize;
    while feat < d {
        let mut acc = F::new(0.0);
        let mut p = 0u32;
        while p < count {
            let cell = all_indices[(seg_start + p) as usize] as usize;
            acc += r[cell * k as usize + cluster as usize] * z[cell * d + feat];
            p += 1u32;
        }
        s[s_base + feat] = acc;
        feat += wg;
    }
}

/// Ridge correction subtract: `z_corr[cell, :] = z[cell, :] - sum_k R[cell, k]
/// * C[k, level, :]`.
///
/// One workgroup per cell; threads stride over the feature dimension `d`. `C`
/// is the per-`(cluster, level)` correction `[k, b, d]` row-major (intercept
/// already excluded; pruned entries are zero). The intercept is never
/// subtracted because it is not stored in `C`.
///
/// ### Params
///
/// * `z` - Original data `[n, d]` row-major
/// * `r` - Soft assignments `[n, k]` row-major
/// * `c` - Correction tensor `[k, b, d]` row-major
/// * `cell_to_level` - Level index per cell `[n]`
/// * `z_corr` - Output corrected data `[n, d]` row-major
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `b` - Number of levels
/// * `d` - Feature dimension (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> cell index
/// * `UNIT_POS_X` -> feature stride offset
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ridge_subtract<F: Float>(
    z: &Tensor<F>,
    r: &Tensor<F>,
    c: &Tensor<F>,
    cell_to_level: &Tensor<u32>,
    z_corr: &mut Tensor<F>,
    n: u32,
    k: u32,
    b: u32,
    #[comptime] d: usize,
) {
    let cell = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cell >= n {
        terminate!();
    }

    let level = cell_to_level[cell as usize] as usize;
    let z_base = cell as usize * d;
    let r_base = cell as usize * k as usize;
    let c_level_off = level * d;
    let bd = b as usize * d;

    let mut feat = UNIT_POS_X as usize;
    let wg = WORKGROUP_128 as usize;
    while feat < d {
        let mut acc = z[z_base + feat];
        let mut cluster = 0u32;
        while cluster < k {
            let r_val = r[r_base + cluster as usize];
            acc -= r_val * c[cluster as usize * bd + c_level_off + feat];
            cluster += 1u32;
        }
        z_corr[z_base + feat] = acc;
        feat += wg;
    }
}

///////////////
// Launchers //
///////////////

// Dispatch [`cosine_distances`]. One workgroup per cell.
pub fn launch_cosine_distances<R, F>(
    centroids: &GpuTensor<R, F>,
    data_cos: &GpuTensor<R, F>,
    dist: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    dim: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(n as u32);

    unsafe {
        cosine_distances::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            centroids.clone().into_tensor_arg(),
            data_cos.clone().into_tensor_arg(),
            dist.clone().into_tensor_arg(),
            n as u32,
            k as u32,
            dim,
            WORKGROUP_128,
        );
    }
}

/// Dispatch `row_l2_normalise`. One workgroup per row.
pub fn launch_row_l2_normalise<R, F>(
    data: &GpuTensor<R, F>,
    n_rows: usize,
    dim: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(n_rows as u32);

    unsafe {
        row_l2_normalise::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_64),
            data.clone().into_tensor_arg(),
            n_rows as u32,
            dim,
        );
    }
}

/// Dispatch `scale_exp_normalise`. One workgroup per cell.
pub fn launch_scale_exp_normalise<R, F>(
    dist: &GpuTensor<R, F>,
    sigma: &GpuTensor<R, F>,
    out: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(n as u32);

    unsafe {
        scale_exp_normalise::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            dist.clone().into_tensor_arg(),
            sigma.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            n as u32,
            k as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch `segmented_sum`. One workgroup per level. The level-CSR
/// (`all_indices`, `offsets`) is built once via `build_csr_gpu_privatised`
/// with `cell_to_level` as labels and `b` as the cluster count.
pub fn launch_segmented_sum<R, F>(
    r: &GpuTensor<R, F>,
    all_indices: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    o: &GpuTensor<R, F>,
    b: usize,
    k: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(b as u32);

    unsafe {
        segmented_sum::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            r.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            o.clone().into_tensor_arg(),
            b as u32,
            k,
        );
    }
}

/// Dispatch `objective_partials`.
#[allow(clippy::too_many_arguments)]
pub fn launch_objective_partials<R, F>(
    r: &GpuTensor<R, F>,
    dist: &GpuTensor<R, F>,
    o: &GpuTensor<R, F>,
    r_sum: &GpuTensor<R, F>,
    sigma: &GpuTensor<R, F>,
    theta: &GpuTensor<R, F>,
    pr_b: &GpuTensor<R, F>,
    cell_to_level: &GpuTensor<R, u32>,
    partials: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    unsafe {
        objective_partials::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(WORKGROUP_512, 1, 1),
            CubeDim::new_1d(WORKGROUP_128),
            r.clone().into_tensor_arg(),
            dist.clone().into_tensor_arg(),
            o.clone().into_tensor_arg(),
            r_sum.clone().into_tensor_arg(),
            sigma.clone().into_tensor_arg(),
            theta.clone().into_tensor_arg(),
            pr_b.clone().into_tensor_arg(),
            cell_to_level.clone().into_tensor_arg(),
            partials.clone().into_tensor_arg(),
            n as u32,
            k,
            WORKGROUP_128,
        );
    }
}

/// Dispatch `jacobi_r_update`. One workgroup per cell. R is written in place.
#[allow(clippy::too_many_arguments)]
pub fn launch_jacobi_r_update<R, F>(
    scale_dist: &GpuTensor<R, F>,
    o: &GpuTensor<R, F>,
    r_sum: &GpuTensor<R, F>,
    theta: &GpuTensor<R, F>,
    pr_b: &GpuTensor<R, F>,
    cell_to_level: &GpuTensor<R, u32>,
    r_out: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(n as u32);

    unsafe {
        jacobi_r_update::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            scale_dist.clone().into_tensor_arg(),
            o.clone().into_tensor_arg(),
            r_sum.clone().into_tensor_arg(),
            theta.clone().into_tensor_arg(),
            pr_b.clone().into_tensor_arg(),
            cell_to_level.clone().into_tensor_arg(),
            r_out.clone().into_tensor_arg(),
            n as u32,
            k as u32,
            WORKGROUP_128,
        );
    }
}

/// Dispatch `weighted_segmented_sum`. One workgroup per `(level, cluster)`
/// pair; `s` must have length `b * k * d`.
#[allow(clippy::too_many_arguments)]
pub fn launch_weighted_segmented_sum<R, F>(
    r: &GpuTensor<R, F>,
    z: &GpuTensor<R, F>,
    all_indices: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    s: &GpuTensor<R, F>,
    b: usize,
    k: usize,
    d: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d((b * k) as u32);

    unsafe {
        weighted_segmented_sum::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            r.clone().into_tensor_arg(),
            z.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            s.clone().into_tensor_arg(),
            b as u32,
            k as u32,
            d,
        );
    }
}

/// Dispatch `ridge_subtract`. One workgroup per cell.
#[allow(clippy::too_many_arguments)]
pub fn launch_ridge_subtract<R, F>(
    z: &GpuTensor<R, F>,
    r: &GpuTensor<R, F>,
    c: &GpuTensor<R, F>,
    cell_to_level: &GpuTensor<R, u32>,
    z_corr: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    b: usize,
    d: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (gx, gy) = grid_2d(n as u32);

    unsafe {
        ridge_subtract::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            z.clone().into_tensor_arg(),
            r.clone().into_tensor_arg(),
            c.clone().into_tensor_arg(),
            cell_to_level.clone().into_tensor_arg(),
            z_corr.clone().into_tensor_arg(),
            n as u32,
            k as u32,
            b as u32,
            d,
        );
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests_harmony_kernels {
    use super::*;
    use ann_search_rs::gpu::tensor::GpuTensor;
    use approx::assert_relative_eq;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    fn l2_normalise_rows(data: &mut [f32], n: usize, dim: usize) {
        for row in 0..n {
            let base = row * dim;
            let norm: f32 = (0..dim).map(|e| data[base + e].powi(2)).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for e in 0..dim {
                    data[base + e] /= norm;
                }
            }
        }
    }

    fn cpu_cosine_distances(
        centroids: &[f32],
        data_cos: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * k];
        for cell in 0..n {
            for cluster in 0..k {
                let mut dot = 0.0f32;
                for e in 0..dim {
                    dot += centroids[cluster * dim + e] * data_cos[cell * dim + e];
                }
                out[cell * k + cluster] = 2.0 * (1.0 - dot);
            }
        }
        out
    }

    fn cpu_scale_exp_normalise(dist: &[f32], sigma: &[f32], n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * k];
        for cell in 0..n {
            let base = cell * k;
            let mut total = 0.0f32;
            let mut tmp = vec![0.0f32; k];
            for cluster in 0..k {
                let v = (-dist[base + cluster] / sigma[cluster]).exp();
                tmp[cluster] = v;
                total += v;
            }
            if total > 0.0 {
                for cluster in 0..k {
                    out[base + cluster] = tmp[cluster] / total;
                }
            }
        }
        out
    }

    fn cpu_segmented_sum(
        r: &[f32],
        cell_to_level: &[u32],
        n: usize,
        k: usize,
        b: usize,
    ) -> Vec<f32> {
        let mut o = vec![0.0f32; b * k];
        for cell in 0..n {
            let level = cell_to_level[cell] as usize;
            for cluster in 0..k {
                o[level * k + cluster] += r[cell * k + cluster];
            }
        }
        o
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_jacobi_r_update(
        scale_dist: &[f32],
        o: &[f32],
        r_sum: &[f32],
        theta: &[f32],
        pr_b: &[f32],
        cell_to_level: &[u32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * k];
        for cell in 0..n {
            let level = cell_to_level[cell] as usize;
            let theta_l = theta[level];
            let pr = pr_b[level];
            let base = cell * k;
            let o_base = level * k;
            let mut tmp = vec![0.0f32; k];
            let mut total = 0.0f32;
            for cluster in 0..k {
                let e_val = r_sum[cluster] * pr;
                let ratio = (2.0 * e_val + 1.0) / (o[o_base + cluster] + e_val + 1.0);
                let penalty = ratio.powf(theta_l);
                let v = scale_dist[base + cluster] * penalty;
                tmp[cluster] = v;
                total += v;
            }
            if total > 0.0 {
                for cluster in 0..k {
                    out[base + cluster] = tmp[cluster] / total;
                }
            }
        }
        out
    }

    fn cpu_weighted_segmented_sum(
        r: &[f32],
        z: &[f32],
        cell_to_level: &[u32],
        n: usize,
        k: usize,
        b: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut s = vec![0.0f32; b * k * d];
        for cell in 0..n {
            let level = cell_to_level[cell] as usize;
            for cluster in 0..k {
                let r_val = r[cell * k + cluster];
                for feat in 0..d {
                    s[(level * k + cluster) * d + feat] += r_val * z[cell * d + feat];
                }
            }
        }
        s
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_ridge_subtract(
        z: &[f32],
        r: &[f32],
        c: &[f32],
        cell_to_level: &[u32],
        n: usize,
        k: usize,
        b: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * d];
        for cell in 0..n {
            let level = cell_to_level[cell] as usize;
            for feat in 0..d {
                let mut acc = z[cell * d + feat];
                for cluster in 0..k {
                    let r_val = r[cell * k + cluster];
                    acc -= r_val * c[cluster * b * d + level * d + feat];
                }
                out[cell * d + feat] = acc;
            }
        }
        out
    }

    // Build the level CSR by hand on the host, sorted by level for determinism.
    fn cpu_build_level_csr(cell_to_level: &[u32], n: usize, b: usize) -> (Vec<u32>, Vec<u32>) {
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); b];
        for cell in 0..n {
            buckets[cell_to_level[cell] as usize].push(cell as u32);
        }
        let mut offsets = vec![0u32; b + 1];
        for level in 0..b {
            offsets[level + 1] = offsets[level] + buckets[level].len() as u32;
        }
        let mut all_indices = Vec::with_capacity(n);
        for bucket in buckets {
            all_indices.extend(bucket);
        }
        (all_indices, offsets)
    }

    #[test]
    fn test_cosine_distances_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, dim) = (12, 4, 8);
        let mut centroids: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 + 0.05)
            .collect();
        let mut data_cos: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.1 + 0.05)
            .collect();
        l2_normalise_rows(&mut centroids, k, dim);
        l2_normalise_rows(&mut data_cos, n, dim);

        let cent_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&centroids, vec![k, dim], &client);
        let data_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data_cos, vec![n, dim], &client);
        let dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client);

        launch_cosine_distances(&cent_gpu, &data_gpu, &dist_gpu, n, k, dim, &client);

        let got = dist_gpu.read(&client).unwrap();
        let want = cpu_cosine_distances(&centroids, &data_cos, n, k, dim);

        for i in 0..n * k {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_row_l2_normalise_basic_and_zero() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, dim) = (4, 6);
        let mut data: Vec<f32> = vec![
            3.0, 4.0, 0.0, 0.0, 0.0, 0.0, // norm 5
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // norm sqrt(6)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // zero row
            -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, // norm 2
        ];
        let original = data.clone();

        let data_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        launch_row_l2_normalise(&data_gpu, n, dim, &client);
        let got = data_gpu.read(&client).unwrap();

        // Reference: normalise the same data on CPU
        for row in 0..n {
            let base = row * dim;
            let norm: f32 = (0..dim)
                .map(|e| original[base + e].powi(2))
                .sum::<f32>()
                .sqrt();
            if norm > 1e-8 {
                for e in 0..dim {
                    data[base + e] = original[base + e] / norm;
                }
            } else {
                for e in 0..dim {
                    data[base + e] = 0.0;
                }
            }
        }

        for i in 0..n * dim {
            assert_relative_eq!(got[i], data[i], epsilon = 1e-6);
        }

        // Sanity: zero row is exactly zero
        for e in 0..dim {
            assert_eq!(got[2 * dim + e], 0.0);
        }
    }

    #[test]
    fn test_scale_exp_normalise_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k) = (16, 6);
        let dist: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 + 7) % 23) as f32 * 0.05)
            .collect();
        let sigma: Vec<f32> = (0..k).map(|i| 0.1 + i as f32 * 0.02).collect();

        let dist_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&dist, vec![n, k], &client);
        let sigma_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&sigma, vec![k], &client);
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client);

        launch_scale_exp_normalise(&dist_gpu, &sigma_gpu, &out_gpu, n, k, &client);

        let got = out_gpu.read(&client).unwrap();
        let want = cpu_scale_exp_normalise(&dist, &sigma, n, k);

        for i in 0..n * k {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-5);
        }

        // Rows sum to 1 (within tolerance)
        for cell in 0..n {
            let s: f32 = (0..k).map(|cl| got[cell * k + cl]).sum();
            assert_relative_eq!(s, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_segmented_sum_with_empty_level() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, b) = (10, 4, 3);
        // Levels: 0 has 4 cells, 1 is empty, 2 has 6 cells
        let cell_to_level: Vec<u32> = vec![0, 2, 0, 2, 0, 2, 0, 2, 2, 2];
        let r: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 + 1) % 11) as f32 * 0.1 + 0.01)
            .collect();

        let r_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r, vec![n, k], &client);

        let (cpu_idx, cpu_off) = cpu_build_level_csr(&cell_to_level, n, b);
        let all_indices_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cpu_idx, vec![n], &client);
        let offsets_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cpu_off, vec![b + 1], &client);

        let o_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![b, k], &client);
        launch_segmented_sum(
            &r_gpu,
            &all_indices_gpu,
            &offsets_gpu,
            &o_gpu,
            b,
            k,
            &client,
        );

        let got = o_gpu.read(&client).unwrap();
        let want = cpu_segmented_sum(&r, &cell_to_level, n, k, b);

        for i in 0..b * k {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-5);
        }

        // Empty level 1: all zeros
        for cluster in 0..k {
            assert_eq!(got[k + cluster], 0.0);
        }
    }

    #[test]
    fn test_jacobi_r_update_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, b) = (12, 5, 3);
        let scale_dist: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 + 7) % 19) as f32 * 0.05 + 0.01)
            .collect();
        let cell_to_level: Vec<u32> = (0..n as u32).map(|i| i % b as u32).collect();
        // O has shape [b, k]
        let o: Vec<f32> = (0..b * k)
            .map(|i| ((i * 5 + 3) % 11) as f32 * 0.1 + 0.05)
            .collect();
        let r_sum: Vec<f32> = (0..k).map(|i| 1.0 + i as f32 * 0.5).collect();
        let theta: Vec<f32> = (0..b).map(|i| 1.0 + i as f32 * 0.3).collect();
        let pr_b: Vec<f32> = (0..b).map(|i| 0.2 + i as f32 * 0.1).collect();

        let scale_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&scale_dist, vec![n, k], &client);
        let o_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&o, vec![b, k], &client);
        let r_sum_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r_sum, vec![k], &client);
        let theta_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&theta, vec![b], &client);
        let pr_b_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&pr_b, vec![b], &client);
        let ctl_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cell_to_level, vec![n], &client);
        let r_out_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client);

        launch_jacobi_r_update(
            &scale_gpu, &o_gpu, &r_sum_gpu, &theta_gpu, &pr_b_gpu, &ctl_gpu, &r_out_gpu, n, k,
            &client,
        );

        let got = r_out_gpu.read(&client).unwrap();
        let want =
            cpu_jacobi_r_update(&scale_dist, &o, &r_sum, &theta, &pr_b, &cell_to_level, n, k);

        for i in 0..n * k {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-4);
        }

        // Each cell sums to 1
        for cell in 0..n {
            let s: f32 = (0..k).map(|cl| got[cell * k + cl]).sum();
            assert_relative_eq!(s, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_jacobi_r_update_zero_scale_dist_row_writes_zeros() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, b) = (3, 4, 2);
        let mut scale_dist = vec![0.1f32; n * k];
        // cell 1 has an all-zero base
        for cluster in 0..k {
            scale_dist[k + cluster] = 0.0;
        }
        let cell_to_level: Vec<u32> = vec![0, 1, 0];
        let o = vec![0.5f32; b * k];
        let r_sum = vec![1.0f32; k];
        let theta = vec![1.0f32; b];
        let pr_b = vec![0.5f32; b];

        let scale_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&scale_dist, vec![n, k], &client);
        let o_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&o, vec![b, k], &client);
        let r_sum_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r_sum, vec![k], &client);
        let theta_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&theta, vec![b], &client);
        let pr_b_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&pr_b, vec![b], &client);
        let ctl_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cell_to_level, vec![n], &client);
        let r_out_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client);

        launch_jacobi_r_update(
            &scale_gpu, &o_gpu, &r_sum_gpu, &theta_gpu, &pr_b_gpu, &ctl_gpu, &r_out_gpu, n, k,
            &client,
        );

        let got = r_out_gpu.read(&client).unwrap();
        for cluster in 0..k {
            assert_eq!(got[k + cluster], 0.0);
        }
    }

    #[test]
    fn test_weighted_segmented_sum_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, b, d) = (10, 3, 3, 5);
        let cell_to_level: Vec<u32> = vec![0, 2, 0, 1, 2, 1, 0, 2, 2, 1];
        let r: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 + 1) % 11) as f32 * 0.1 + 0.05)
            .collect();
        let z: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.13 - 1.0).collect();

        let r_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r, vec![n, k], &client);
        let z_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&z, vec![n, d], &client);

        let (cpu_idx, cpu_off) = cpu_build_level_csr(&cell_to_level, n, b);
        let idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cpu_idx, vec![n], &client);
        let off_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cpu_off, vec![b + 1], &client);

        let s_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![b * k * d], &client);
        launch_weighted_segmented_sum(&r_gpu, &z_gpu, &idx_gpu, &off_gpu, &s_gpu, b, k, d, &client);

        let got = s_gpu.read(&client).unwrap();
        let want = cpu_weighted_segmented_sum(&r, &z, &cell_to_level, n, k, b, d);

        for i in 0..b * k * d {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_ridge_subtract_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k, b, d) = (8, 3, 2, 4);
        let cell_to_level: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let z: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1).collect();
        let r: Vec<f32> = (0..n * k)
            .map(|i| ((i * 5 + 3) % 7) as f32 * 0.1 + 0.05)
            .collect();
        let c: Vec<f32> = (0..k * b * d)
            .map(|i| ((i * 11 + 1) % 13) as f32 * 0.05)
            .collect();

        let z_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&z, vec![n, d], &client);
        let r_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r, vec![n, k], &client);
        let c_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&c, vec![k * b * d], &client);
        let ctl_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cell_to_level, vec![n], &client);
        let z_corr_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, d], &client);

        launch_ridge_subtract(
            &z_gpu,
            &r_gpu,
            &c_gpu,
            &ctl_gpu,
            &z_corr_gpu,
            n,
            k,
            b,
            d,
            &client,
        );

        let got = z_corr_gpu.read(&client).unwrap();
        let want = cpu_ridge_subtract(&z, &r, &c, &cell_to_level, n, k, b, d);

        for i in 0..n * d {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-5);
        }
    }
}
