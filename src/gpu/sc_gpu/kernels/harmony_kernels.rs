//! GPU kernels for Harmony v2 (single-covariate, arrowhead path).
//!
//! Layout convention: every K-per-cell matrix (`dist`, `scale_dist`, `R`) is
//! stored `[N, K]` row-major, the same cell-major layout as the CPU core, so
//! each per-cell kernel reads K contiguous values. Cells are sorted by batch
//! level, so every per-level reduction is a set of contiguous runs.

#![allow(missing_docs)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use crate::errors::BixverseErrors;
use crate::gpu::linalg::cholesky_gpu::dense_gemm;
use crate::gpu::*;

/////////////
// Kernels //
/////////////

/// In-place cosine distance from a cosine similarity: `x = 2 * (1 - x)`.
///
/// Epilogue of [`launch_cosine_distances`], whose GEMM writes the dot products
/// of unit-norm rows into the output buffer.
///
/// ### Params
///
/// * `x` - Similarities `[len]`, overwritten with distances
/// * `len` - Number of elements
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X` ->
///   element index
#[cube(launch_unchecked)]
fn similarity_to_cosine_distance<F: Float>(x: &mut Tensor<F>, len: u32) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X;
    if idx >= len {
        terminate!();
    }
    x[idx as usize] = F::new(2.0_f32) * (F::new(1.0_f32) - x[idx as usize]);
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

        let mut acc = F::new(0.0_f32);
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
                data[base + e] = F::new(0.0_f32);
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
    let mut acc = F::new(0.0_f32);
    let mut kk = tx;
    while kk < k {
        let arg = (F::new(0.0_f32) - dist[base + kk as usize]) / sigma[kk as usize];
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
    if total > F::new(0.0_f32) {
        while kk2 < k {
            let idx = base + kk2 as usize;
            let arg = (F::new(0.0_f32) - dist[idx]) / sigma[kk2 as usize];
            out[idx] = F::exp(arg) / total;
            kk2 += wg_size;
        }
    } else {
        while kk2 < k {
            out[base + kk2 as usize] = F::new(0.0_f32);
            kk2 += wg_size;
        }
    }
}

/// Per-run column sums of R: `partial[run, :] = sum_{cells of run} R[cell, :]`.
///
/// Cells are stored sorted by level, and each run is a contiguous range of at
/// most `HARMONY_RUN_CELLS` cells of one level, so the reads are plain strided
/// rows with no index gather. One workgroup per run; thread `tx` owns
/// clusters `tx, tx + wg, ...`, so a warp reads consecutive addresses of each
/// row. [`fn@reduce_runs`] folds the partials into the per-level O.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]` row-major, cells sorted by level
/// * `run_bounds` - Cell offsets `[n_runs + 1]`; run `j` is
///   `run_bounds[j]..run_bounds[j + 1]`
/// * `partial` - Output `[n_runs, k]` row-major
/// * `n_runs` - Number of runs
/// * `k` - Cluster count (comptime)
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> run index
/// * `UNIT_POS_X` -> cluster stride offset
#[cube(launch_unchecked)]
fn run_sums<F: Float>(
    r: &Tensor<F>,
    run_bounds: &Tensor<u32>,
    partial: &mut Tensor<F>,
    n_runs: u32,
    #[comptime] k: usize,
    #[comptime] wg_size: u32,
) {
    let run = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if run >= n_runs {
        terminate!();
    }
    let start = run_bounds[run as usize] as usize;
    let end = run_bounds[(run + 1u32) as usize] as usize;

    let mut kk = UNIT_POS_X as usize;
    while kk < k {
        let mut acc = F::new(0.0_f32);
        let mut c = start;
        while c < end {
            acc += r[c * k + kk];
            c += 1;
        }
        partial[run as usize * k + kk] = acc;
        kk += wg_size as usize;
    }
}

/// Per-run weighted sums for the ridge right-hand side:
/// `partial[run, k, :] = sum_{cells of run} R[cell, k] * z[cell, :]`.
///
/// One workgroup per `(run, cluster)`; threads stride over the feature
/// dimension. `R[cell, k]` is a broadcast read, `z[cell, :]` is coalesced
/// across the threads. Runs are contiguous (cells sorted by level), so there
/// is no index gather, and every workgroup does at most `HARMONY_RUN_CELLS`
/// iterations however unbalanced the levels are.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]` row-major, cells sorted by level
/// * `z` - Original embedding `[n, d]` row-major, same order
/// * `run_bounds` - Cell offsets `[n_runs + 1]`
/// * `partial` - Output `[n_runs, k, d]` row-major
/// * `n_runs` - Number of runs
/// * `k` - Cluster count
/// * `d` - Feature dimension (comptime)
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> `run * k + cluster`
/// * `UNIT_POS_X` -> feature stride offset
#[cube(launch_unchecked)]
fn run_weighted_sums<F: Float>(
    r: &Tensor<F>,
    z: &Tensor<F>,
    run_bounds: &Tensor<u32>,
    partial: &mut Tensor<F>,
    n_runs: u32,
    k: u32,
    #[comptime] d: usize,
    #[comptime] wg_size: u32,
) {
    let wid = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if wid >= n_runs * k {
        terminate!();
    }
    let run = wid / k;
    let cluster = (wid % k) as usize;
    let start = run_bounds[run as usize] as usize;
    let end = run_bounds[(run + 1u32) as usize] as usize;
    let ku = k as usize;

    let mut f = UNIT_POS_X as usize;
    while f < d {
        let mut acc = F::new(0.0_f32);
        let mut c = start;
        while c < end {
            acc += r[c * ku + cluster] * z[c * d + f];
            c += 1;
        }
        partial[wid as usize * d + f] = acc;
        f += wg_size as usize;
    }
}

/// Fold per-run partials into per-level totals:
/// `out[level, w] = sum_{runs of level} partial[run, w]`.
///
/// One thread per output element; each loops over its level's runs, of which
/// there are few (a level of `m` cells has `ceil(m / HARMONY_RUN_CELLS)`).
/// Empty levels have no runs and are written as zero.
///
/// ### Params
///
/// * `partial` - Per-run partials `[n_runs, width]` row-major
/// * `level_runs` - Run offsets per level `[b + 1]`
/// * `out` - Output `[b, width]` row-major
/// * `b` - Number of levels
/// * `width` - Row length of `partial` and `out`
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X` ->
///   flat output index
#[cube(launch_unchecked)]
fn reduce_runs<F: Float>(
    partial: &Tensor<F>,
    level_runs: &Tensor<u32>,
    out: &mut Tensor<F>,
    b: u32,
    width: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X;
    if idx >= b * width {
        terminate!();
    }
    let level = idx / width;
    let w = (idx % width) as usize;
    let first = level_runs[level as usize];
    let last = level_runs[(level + 1u32) as usize];

    let mut acc = F::new(0.0_f32);
    let mut run = first;
    while run < last {
        acc += partial[run as usize * width as usize + w];
        run += 1u32;
    }
    out[idx as usize] = acc;
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

    let mut acc = F::new(0.0_f32);
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
            if r_val > F::new(0.0_f32) {
                acc += r_val * F::ln(r_val) * sigma[cl];
            }
            let e_val = r_sum[cl] * pr;
            let ln_ratio = F::ln(
                (o[o_base + cl] + e_val + F::new(1.0_f32))
                    / (F::new(2.0_f32) * e_val + F::new(1.0_f32)),
            );
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
/// * `scale_dist` - Column-normalised base assignments `[n, k]`, fixed per
///   round
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
    let mut acc = F::new(0.0_f32);
    let mut kk = tx;
    while kk < k {
        let e_val = r_sum[kk as usize] * pr;
        let ratio = (F::new(2.0_f32) * e_val + F::new(1.0_f32))
            / (o[o_base + kk as usize] + e_val + F::new(1.0_f32));
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
    if total > F::new(0.0_f32) {
        while kk2 < k {
            let e_val = r_sum[kk2 as usize] * pr;
            let ratio = (F::new(2.0_f32) * e_val + F::new(1.0_f32))
                / (o[o_base + kk2 as usize] + e_val + F::new(1.0_f32));
            let penalty = F::powf(ratio, theta_l);
            let val = scale_dist[cell_base + kk2 as usize] * penalty;
            r_out[cell_base + kk2 as usize] = val / total;
            kk2 += wg_size;
        }
    } else {
        while kk2 < k {
            r_out[cell_base + kk2 as usize] = F::new(0.0_f32);
            kk2 += wg_size;
        }
    }
}

/// Ridge correction subtract: `z_corr[cell, :] = z[cell, :] - sum_k R[cell, k]
/// * C[level, k, :]`.
///
/// One workgroup per cell; threads stride over the feature dimension `d`. `C`
/// is the per-`(level, cluster)` correction `[b, k, d]` row-major, the layout
/// `harmony_core::solve_ridge` produces (intercept excluded; pruned entries
/// are zero). The intercept is never subtracted because it is not stored in
/// `C`.
///
/// ### Params
///
/// * `z` - Original data `[n, d]` row-major
/// * `r` - Soft assignments `[n, k]` row-major
/// * `c` - Correction tensor `[b, k, d]` row-major
/// * `cell_to_level` - Level index per cell `[n]`
/// * `z_corr` - Output corrected data `[n, d]` row-major
/// * `n` - Cell count
/// * `k` - Cluster count
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
    #[comptime] d: usize,
) {
    let cell = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cell >= n {
        terminate!();
    }

    let level = cell_to_level[cell as usize] as usize;
    let z_base = cell as usize * d;
    let r_base = cell as usize * k as usize;
    let c_base = level * k as usize * d;

    let mut feat = UNIT_POS_X as usize;
    let wg = WORKGROUP_128 as usize;
    while feat < d {
        let mut acc = z[z_base + feat];
        let mut cluster = 0u32;
        while cluster < k {
            let r_val = r[r_base + cluster as usize];
            acc -= r_val * c[c_base + cluster as usize * d + feat];
            cluster += 1u32;
        }
        z_corr[z_base + feat] = acc;
        feat += wg;
    }
}

///////////////
// Launchers //
///////////////

/// Cosine distances between centroids and cells, `dist[n, k] = 2 * (1 - z_n .
/// y_k)`, as one library GEMM (`z_cos [n, dim] x Y^T`) and an elementwise
/// epilogue. Short reduction, large output: the regime the cubek matmul is
/// good at, unlike a per-cell kernel whose threads each stream a centroid row.
/// Pinned to `f32` like the rest of GPU Harmony.
///
/// ### Params
///
/// * `centroids` - Centroids `[k, dim]` row-major, cosine-normalised
/// * `data_cos` - Cells `[n, dim]` row-major, cosine-normalised
/// * `dist` - Output distances `[n, k]` row-major
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `dim` - Embedding dimension
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `dist` is written in place. `GpuMatmul` if the GEMM dispatch
/// fails, `CubeclUtils` if the epilogue grid is over the device limit.
pub fn launch_cosine_distances<R: Runtime>(
    centroids: &GpuTensor<R, f32>,
    data_cos: &GpuTensor<R, f32>,
    dist: &GpuTensor<R, f32>,
    n: usize,
    k: usize,
    dim: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors> {
    dense_gemm::<R, f32>(
        data_cos.handle(),
        [n, dim],
        false,
        centroids.handle(),
        [dim, k],
        true, // storage [k, dim] read as logical [dim, k]
        dist.handle(),
        [n, k],
        None,
        client,
    )?;

    let limits = GpuLimits::from_client(client);
    let len = n * k;
    let (gx, gy) = grid_2d(len.div_ceil(WORKGROUP_256 as usize) as u32, &limits)?;
    unsafe {
        similarity_to_cosine_distance::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_256),
            dist.clone().into_tensor_arg(),
            len as u32,
        );
    }

    Ok(())
}

/// Dispatch `row_l2_normalise`. One workgroup per row.
///
/// ### Params
///
/// * `data` - Matrix `[n_rows, dim]` row-major, normalised in place
/// * `n_rows` - Number of rows
/// * `dim` - Row length
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `data` is normalised in place. `CubeclUtils` if the grid is over
/// the device's cube-count limit.
pub fn launch_row_l2_normalise<R, F>(
    data: &GpuTensor<R, F>,
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
        row_l2_normalise::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_64),
            data.clone().into_tensor_arg(),
            n_rows as u32,
            dim,
        );
    }

    Ok(())
}

/// Dispatch `scale_exp_normalise`. One workgroup per cell.
///
/// ### Params
///
/// * `dist` - Distance matrix `[n, k]` row-major
/// * `sigma` - Per-cluster weights `[k]`
/// * `out` - Output `[n, k]` row-major, either `scale_dist` or the initial `R`
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `out` is written in place. `CubeclUtils` if the grid is over the
/// device's cube-count limit.
pub fn launch_scale_exp_normalise<R, F>(
    dist: &GpuTensor<R, F>,
    sigma: &GpuTensor<R, F>,
    out: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n as u32, &limits)?;

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

    Ok(())
}

/// Dispatch [`fn@run_sums`]. One workgroup per run.
///
/// ### Params
///
/// * `r` - Assignment matrix `[n, k]` row-major, cells sorted by level
/// * `run_bounds` - Cell offsets `[n_runs + 1]`
/// * `partial` - Output `[n_runs, k]` row-major
/// * `n_runs` - Number of runs
/// * `k` - Cluster count
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `partial` is written in place. `CubeclUtils` if the grid is over
/// the device's cube-count limit.
pub fn launch_run_sums<R, F>(
    r: &GpuTensor<R, F>,
    run_bounds: &GpuTensor<R, u32>,
    partial: &GpuTensor<R, F>,
    n_runs: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n_runs as u32, &limits)?;

    unsafe {
        run_sums::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            r.clone().into_tensor_arg(),
            run_bounds.clone().into_tensor_arg(),
            partial.clone().into_tensor_arg(),
            n_runs as u32,
            k,
            WORKGROUP_128,
        );
    }

    Ok(())
}

/// Dispatch [`fn@run_weighted_sums`]. One workgroup per `(run, cluster)`.
///
/// ### Params
///
/// * `r` - Assignment matrix `[n, k]` row-major, cells sorted by level
/// * `z` - Embedding `[n, d]` row-major, same order
/// * `run_bounds` - Cell offsets `[n_runs + 1]`
/// * `partial` - Output `[n_runs, k, d]` row-major
/// * `n_runs` - Number of runs
/// * `k` - Cluster count
/// * `d` - Embedding dimension
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `partial` is written in place. `CubeclUtils` if the grid is over
/// the device's cube-count limit.
#[allow(clippy::too_many_arguments)]
pub fn launch_run_weighted_sums<R, F>(
    r: &GpuTensor<R, F>,
    z: &GpuTensor<R, F>,
    run_bounds: &GpuTensor<R, u32>,
    partial: &GpuTensor<R, F>,
    n_runs: usize,
    k: usize,
    d: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((n_runs * k) as u32, &limits)?;

    unsafe {
        run_weighted_sums::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_64),
            r.clone().into_tensor_arg(),
            z.clone().into_tensor_arg(),
            run_bounds.clone().into_tensor_arg(),
            partial.clone().into_tensor_arg(),
            n_runs as u32,
            k as u32,
            d,
            WORKGROUP_64,
        );
    }

    Ok(())
}

/// Dispatch [`fn@reduce_runs`]. One thread per output element.
///
/// ### Params
///
/// * `partial` - Per-run partials `[n_runs, width]` row-major
/// * `level_runs` - Run offsets per level `[b + 1]`
/// * `out` - Output `[b, width]` row-major
/// * `b` - Number of levels
/// * `width` - Row length
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `out` is written in place. `CubeclUtils` if the grid is over the
/// device's cube-count limit.
pub fn launch_reduce_runs<R, F>(
    partial: &GpuTensor<R, F>,
    level_runs: &GpuTensor<R, u32>,
    out: &GpuTensor<R, F>,
    b: usize,
    width: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let cubes = (b * width).div_ceil(WORKGROUP_256 as usize);
    let (gx, gy) = grid_2d(cubes as u32, &limits)?;

    unsafe {
        reduce_runs::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_256),
            partial.clone().into_tensor_arg(),
            level_runs.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            b as u32,
            width as u32,
        );
    }

    Ok(())
}

/// Dispatch `objective_partials`.
///
/// The only launcher here with a fixed grid (`WORKGROUP_512` cubes), so it has
/// no geometry to validate and stays infallible.
///
/// ### Params
///
/// * `r` - Assignment matrix `[n, k]` row-major
/// * `dist` - Distance matrix `[n, k]` row-major
/// * `o` - Observed counts `[b, k]` row-major
/// * `r_sum` - Per-cluster totals `[k]`
/// * `sigma` - Per-cluster weights `[k]`
/// * `theta` - Per-level diversity penalties `[b]`
/// * `pr_b` - Level proportions `[b]`
/// * `cell_to_level` - Level index per cell `[n]`
/// * `partials` - Output per-workgroup partial sums `[WORKGROUP_512]`
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// Nothing; `partials` is written in place and the caller sums it on the host.
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

/// Dispatch `jacobi_r_update`. One workgroup per cell.
///
/// ### Params
///
/// * `scale_dist` - Scaled distances `[n, k]` row-major, fixed across the sweep
/// * `o` - Observed counts `[b, k]` row-major, from the previous sweep
/// * `r_sum` - Per-cluster totals `[k]`, from the previous sweep
/// * `theta` - Per-level diversity penalties `[b]`
/// * `pr_b` - Level proportions `[b]`
/// * `cell_to_level` - Level index per cell `[n]`
/// * `r_out` - Output assignments `[n, k]` row-major
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `r_out` is written in place. `CubeclUtils` if the grid is over the
/// device's cube-count limit.
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
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n as u32, &limits)?;

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

    Ok(())
}

/// Dispatch `ridge_subtract`. One workgroup per cell.
///
/// ### Params
///
/// * `z` - Embedding `[n, d]` row-major
/// * `r` - Assignment matrix `[n, k]` row-major
/// * `c` - Per-`(level, cluster)` corrections `[b, k, d]` row-major
/// * `cell_to_level` - Level index per cell `[n]`
/// * `z_corr` - Output corrected embedding `[n, d]` row-major
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `d` - Embedding dimension
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`; `z_corr` is written in place. `CubeclUtils` if the grid is over
/// the device's cube-count limit.
#[allow(clippy::too_many_arguments)]
pub fn launch_ridge_subtract<R, F>(
    z: &GpuTensor<R, F>,
    r: &GpuTensor<R, F>,
    c: &GpuTensor<R, F>,
    cell_to_level: &GpuTensor<R, u32>,
    z_corr: &GpuTensor<R, F>,
    n: usize,
    k: usize,
    d: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n as u32, &limits)?;

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
            d,
        );
    }

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests_harmony_kernels {
    use super::*;
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

    #[allow(clippy::too_many_arguments)]
    fn cpu_ridge_subtract(
        z: &[f32],
        r: &[f32],
        c: &[f32],
        cell_to_level: &[u32],
        n: usize,
        k: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * d];
        for cell in 0..n {
            let level = cell_to_level[cell] as usize;
            for feat in 0..d {
                let mut acc = z[cell * d + feat];
                for cluster in 0..k {
                    let r_val = r[cell * k + cluster];
                    acc -= r_val * c[(level * k + cluster) * d + feat];
                }
                out[cell * d + feat] = acc;
            }
        }
        out
    }

    // Build the level CSR by hand on the host, sorted by level for determinism.

    /// Cosine distance kernel against the host, on pre-normalised rows.
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

        let cent_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&centroids, vec![k, dim], &client).unwrap();
        let data_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data_cos, vec![n, dim], &client).unwrap();
        let dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client).unwrap();

        launch_cosine_distances(&cent_gpu, &data_gpu, &dist_gpu, n, k, dim, &client).unwrap();

        let got = dist_gpu.read(&client).unwrap();
        let want = cpu_cosine_distances(&centroids, &data_cos, n, k, dim);

        for i in 0..n * k {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-5);
        }
    }

    /// In-place normalisation against the host; a zero row must give zeros.
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

        let data_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        launch_row_l2_normalise(&data_gpu, n, dim, &client).unwrap();
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

    /// Sigma scaling, exp and normalisation: must match, rows must sum to 1.
    #[test]
    fn test_scale_exp_normalise_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, k) = (16, 6);
        let dist: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 + 7) % 23) as f32 * 0.05)
            .collect();
        let sigma: Vec<f32> = (0..k).map(|i| 0.1 + i as f32 * 0.02).collect();

        let dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&dist, vec![n, k], &client).unwrap();
        let sigma_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&sigma, vec![k], &client).unwrap();
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client).unwrap();

        launch_scale_exp_normalise(&dist_gpu, &sigma_gpu, &out_gpu, n, k, &client).unwrap();

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

    /// Diversity-penalised R update against the host; rows still sum to 1.
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

        let scale_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&scale_dist, vec![n, k], &client).unwrap();
        let o_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&o, vec![b, k], &client).unwrap();
        let r_sum_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&r_sum, vec![k], &client).unwrap();
        let theta_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&theta, vec![b], &client).unwrap();
        let pr_b_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&pr_b, vec![b], &client).unwrap();
        let ctl_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&cell_to_level, vec![n], &client).unwrap();
        let r_out_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client).unwrap();

        launch_jacobi_r_update(
            &scale_gpu, &o_gpu, &r_sum_gpu, &theta_gpu, &pr_b_gpu, &ctl_gpu, &r_out_gpu, n, k,
            &client,
        )
        .unwrap();

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

    /// An all-zero distance row would divide by zero: it must write zeros.
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

        let scale_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&scale_dist, vec![n, k], &client).unwrap();
        let o_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&o, vec![b, k], &client).unwrap();
        let r_sum_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&r_sum, vec![k], &client).unwrap();
        let theta_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&theta, vec![b], &client).unwrap();
        let pr_b_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&pr_b, vec![b], &client).unwrap();
        let ctl_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&cell_to_level, vec![n], &client).unwrap();
        let r_out_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client).unwrap();

        launch_jacobi_r_update(
            &scale_gpu, &o_gpu, &r_sum_gpu, &theta_gpu, &pr_b_gpu, &ctl_gpu, &r_out_gpu, n, k,
            &client,
        )
        .unwrap();

        let got = r_out_gpu.read(&client).unwrap();
        for cluster in 0..k {
            assert_eq!(got[k + cluster], 0.0);
        }
    }

    /// Correction subtraction on the host, pinning C's `[b, k, d]` stride.
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

        let z_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&z, vec![n, d], &client).unwrap();
        let r_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r, vec![n, k], &client).unwrap();
        let c_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&c, vec![k * b * d], &client).unwrap();
        let ctl_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&cell_to_level, vec![n], &client).unwrap();
        let z_corr_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, d], &client).unwrap();

        launch_ridge_subtract(
            &z_gpu,
            &r_gpu,
            &c_gpu,
            &ctl_gpu,
            &z_corr_gpu,
            n,
            k,
            d,
            &client,
        )
        .unwrap();

        let got = z_corr_gpu.read(&client).unwrap();
        let want = cpu_ridge_subtract(&z, &r, &c, &cell_to_level, n, k, d);

        for i in 0..n * d {
            assert_relative_eq!(got[i], want[i], epsilon = 1e-5);
        }
    }

    /// Runs and levels for cells sorted by level, splitting each level into
    /// runs of at most `max_run` cells.
    fn cpu_runs(level_sizes: &[usize], max_run: usize) -> (Vec<u32>, Vec<u32>) {
        let (mut bounds, mut level_runs) = (vec![0u32], vec![0u32]);
        let mut pos = 0usize;
        for &m in level_sizes {
            let mut left = m;
            while left > 0 {
                let take = left.min(max_run);
                pos += take;
                left -= take;
                bounds.push(pos as u32);
            }
            level_runs.push((bounds.len() - 1) as u32);
        }
        (bounds, level_runs)
    }

    /// Per-level O from run partials, with an empty level and a level split
    /// over several runs.
    #[test]
    fn test_run_sums_reduce_with_empty_and_split_levels() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let level_sizes = [5usize, 0, 9];
        let (n, k, b) = (14usize, 4usize, 3usize);
        let (bounds, level_runs) = cpu_runs(&level_sizes, 4);
        let n_runs = bounds.len() - 1;
        let r: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 + 1) % 11) as f32 * 0.1)
            .collect();

        let r_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r, vec![n, k], &client).unwrap();
        let b_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&bounds, vec![n_runs + 1], &client).unwrap();
        let lr_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&level_runs, vec![b + 1], &client).unwrap();
        let partial = GpuTensor::<WgpuRuntime, f32>::empty(vec![n_runs, k], &client).unwrap();
        let o =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![-1.0f32; b * k], vec![b, k], &client)
                .unwrap();

        launch_run_sums(&r_gpu, &b_gpu, &partial, n_runs, k, &client).unwrap();
        launch_reduce_runs(&partial, &lr_gpu, &o, b, k, &client).unwrap();
        let got = o.read(&client).unwrap();

        let mut start = 0usize;
        for (level, &m) in level_sizes.iter().enumerate() {
            for kk in 0..k {
                let want: f32 = (start..start + m).map(|c| r[c * k + kk]).sum();
                assert_relative_eq!(got[level * k + kk], want, epsilon = 1e-5);
            }
            start += m;
        }
    }

    /// Per-level weighted sums S from run partials, in the `[b, k, d]` layout.
    #[test]
    fn test_run_weighted_sums_reduce_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let level_sizes = [3usize, 7, 0, 2];
        let (n, k, b, d) = (12usize, 3usize, 4usize, 5usize);
        let (bounds, level_runs) = cpu_runs(&level_sizes, 3);
        let n_runs = bounds.len() - 1;
        let r: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 + 1) % 11) as f32 * 0.1 + 0.05)
            .collect();
        let z: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.13 - 1.0).collect();

        let r_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&r, vec![n, k], &client).unwrap();
        let z_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&z, vec![n, d], &client).unwrap();
        let b_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&bounds, vec![n_runs + 1], &client).unwrap();
        let lr_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&level_runs, vec![b + 1], &client).unwrap();
        let partial = GpuTensor::<WgpuRuntime, f32>::empty(vec![n_runs * k * d], &client).unwrap();
        let s = GpuTensor::<WgpuRuntime, f32>::empty(vec![b * k * d], &client).unwrap();

        launch_run_weighted_sums(&r_gpu, &z_gpu, &b_gpu, &partial, n_runs, k, d, &client).unwrap();
        launch_reduce_runs(&partial, &lr_gpu, &s, b, k * d, &client).unwrap();
        let got = s.read(&client).unwrap();

        let mut start = 0usize;
        for (level, &m) in level_sizes.iter().enumerate() {
            for kk in 0..k {
                for f in 0..d {
                    let want: f32 = (start..start + m)
                        .map(|c| r[c * k + kk] * z[c * d + f])
                        .sum();
                    assert_relative_eq!(got[(level * k + kk) * d + f], want, epsilon = 1e-4);
                }
            }
            start += m;
        }
    }
}
