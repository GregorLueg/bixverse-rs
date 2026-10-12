//! Harmony v2 on MLX for a single batch covariate.
//!
//! Mirrors the cubecl `harmony_v2_gpu`: full-batch damped Jacobi assignment
//! updates and the arrowhead ridge correction, with the per-cluster systems
//! solved on the host in f64 by the shared CPU solver. Cells are sorted by
//! batch level once on upload, so every level is a contiguous row range:
//! the ridge right-hand side `S_b = R_bᵀ Z_b` and the correction
//! `Z_b - R_b C_b` are one MLX GEMM per level over row slices, cell axis
//! outermost in both operands. The per-cell softmax, the Jacobi update, the
//! row norm and the objective are custom Metal kernels with one SIMD group
//! per cell and `simd_sum` reductions. The initial k-means is the CPU one, on
//! the unsorted cells, so it starts from the same centroids as the CPU v2.
//! f32 only.

use faer::{Mat, MatRef};
use std::time::Instant;

use crate::mlx::ffi::{
    Array, MLX_FLOAT32, MetalKernel, OutputSpec, Stream, clear_cache, eval_all,
    install_error_handler,
};
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::harmony::{BatchInfo, StageTimes, create_batch_infos};
use crate::single_cell::sc_batch_correction::harmony_core::{
    RidgeSettings, Variant, kmeans_centroids, normalise_rows_into, objective, row_major_to_mat,
    solve_ridge, to_row_major,
};
use crate::single_cell::sc_batch_correction::harmony_v2::{
    HarmonyParamsV2, check_convergence, expand_theta,
};

////////////
// Consts //
////////////

/// Cells per run for the per-level observed counts. Every `run_sums`
/// threadgroup walks at most this many cells, however unbalanced the levels.
/// Same as the cubecl `HARMONY_RUN_CELLS`.
pub const MLX_HARMONY_RUN_CELLS: usize = 1024;

/// Lanes per SIMD group on Apple GPUs.
const SIMD_WIDTH: usize = 32;

/// Threads per threadgroup of the one-SIMD-group-per-cell kernels, so
/// `CELL_TG / SIMD_WIDTH` cells per threadgroup.
const CELL_TG: usize = 256;

/// Threads per threadgroup of `run_sums`; threads stride the clusters.
const RUN_TG: usize = 128;

/// Threads per threadgroup of `level_reduce`; one thread per cluster.
const REDUCE_TG: usize = 64;

/// Threadgroups of the grid-strided objective, i.e. the number of partials
/// the host sums.
const OBJ_GROUPS: usize = 256;

/// Upper bound on SIMD groups per threadgroup (1024 / 32), sizes the
/// threadgroup scratch of the objective.
const MAX_SIMD_GROUPS: usize = 32;

/// Weight of the fresh assignments in each Jacobi update, the rest is the
/// previous sweep's R. Same value and reasoning as the cubecl
/// `R_UPDATE_DAMPING`: undamped, every cell reacts to the same stale O at
/// once and the objective oscillates with period two.
const R_UPDATE_DAMPING: f32 = 0.5;

/// Row norm below which a row is written as zeros, as on the CPU.
const NORM_EPS: f32 = 1e-8;

/////////////////////
// Kernel sources //
/////////////////////

/// Per-cell softmax `exp(-dist / sigma) / sum`, shifted by the per-cell
/// maximum logit (as the CPU `distances_and_base`). Serves the base
/// assignments and the cold-started R, which are the same array.
///
/// Inputs: `dist` `[n, K]`, `sigma` `[K]`, `dims` `[2]` u32 (`n`, `b`).
/// Output: `out` `[n, K]`.
const EXP_NORMALISE_SOURCE: &str = r#"
    uint lane = thread_index_in_simdgroup;
    uint cell = threadgroup_position_in_grid.x * simdgroups_per_threadgroup
        + simdgroup_index_in_threadgroup;
    if (cell >= dims[0]) return;
    size_t rb = size_t(cell) * K;
    float m = -FLT_MAX;
    for (uint j = lane; j < uint(K); j += SIMD_W) m = max(m, -dist[rb + j] / sigma[j]);
    m = simd_max(m);
    float s = 0.0f;
    for (uint j = lane; j < uint(K); j += SIMD_W) s += exp(-dist[rb + j] / sigma[j] - m);
    float inv = 1.0f / simd_sum(s);
    for (uint j = lane; j < uint(K); j += SIMD_W) out[rb + j] = exp(-dist[rb + j] / sigma[j] - m) * inv;
"#;

/// Damped Jacobi update of one cell: `r_out = DAMP * base * pen / sum +
/// (1 - DAMP) * r`, with the penalty table of the cell's level. A cell whose
/// weights all vanish is written as zeros, as in the cubecl kernel.
///
/// Inputs: `base` `[n, K]`, `pen` `[b, K]`, `level` `[n]` u32, `r` `[n, K]`,
/// `dims` `[2]` u32. Output: `r_out` `[n, K]`.
const JACOBI_SOURCE: &str = r#"
    uint lane = thread_index_in_simdgroup;
    uint cell = threadgroup_position_in_grid.x * simdgroups_per_threadgroup
        + simdgroup_index_in_threadgroup;
    if (cell >= dims[0]) return;
    size_t rb = size_t(cell) * K;
    size_t pb = size_t(level[cell]) * K;
    float s = 0.0f;
    for (uint j = lane; j < uint(K); j += SIMD_W) s += base[rb + j] * pen[pb + j];
    s = simd_sum(s);
    if (s > 0.0f) {
        float inv = DAMP / s;
        for (uint j = lane; j < uint(K); j += SIMD_W) {
            r_out[rb + j] = base[rb + j] * pen[pb + j] * inv + (1.0f - DAMP) * r[rb + j];
        }
    } else {
        for (uint j = lane; j < uint(K); j += SIMD_W) r_out[rb + j] = 0.0f;
    }
"#;

/// Per-run column sums of R. One threadgroup per run, threads stride the
/// clusters so each row read is coalesced.
///
/// Inputs: `r` `[n, K]`, `run_bounds` `[n_runs + 1]` u32. Output: `partial`
/// `[n_runs, K]`.
const RUN_SUMS_SOURCE: &str = r#"
    uint run = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;
    uint start = run_bounds[run];
    uint end = run_bounds[run + 1];
    for (uint j = t; j < uint(K); j += tpg) {
        float acc = 0.0f;
        for (uint c = start; c < end; c++) acc += r[size_t(c) * K + j];
        partial[size_t(run) * K + j] = acc;
    }
"#;

/// Fold the run partials into the observed counts `O [b, K]`, the cluster
/// totals `r_sum [K]` (sum of O over levels, as on the CPU) and the v2
/// penalty table `pen [b, K] = ((2E + 1) / (O + E + 1))^theta_l` with
/// `E = r_sum * pr_b`. One thread per cluster.
///
/// Inputs: `partial` `[n_runs, K]`, `level_runs` `[b + 1]` u32, `theta`
/// `[b]`, `pr_b` `[b]`, `dims` `[2]` u32. Outputs: `o`, `r_sum`, `pen`.
const LEVEL_REDUCE_SOURCE: &str = r#"
    uint j = thread_position_in_grid.x;
    if (j >= uint(K)) return;
    uint b = dims[1];
    float tot = 0.0f;
    for (uint l = 0; l < b; l++) {
        float acc = 0.0f;
        for (uint run = level_runs[l]; run < level_runs[l + 1]; run++) {
            acc += partial[size_t(run) * K + j];
        }
        o[l * K + j] = acc;
        tot += acc;
    }
    r_sum[j] = tot;
    for (uint l = 0; l < b; l++) {
        float e = tot * pr_b[l];
        pen[l * K + j] = pow((2.0f * e + 1.0f) / (o[l * K + j] + e + 1.0f), theta[l]);
    }
"#;

/// Grid-strided partials of the k-means error and entropy,
/// `sum r * dist + sigma r ln r` (`r > 0`). One SIMD group per cell, lanes
/// stride the clusters; `simd_sum` and a threadgroup scratch give one partial
/// per threadgroup. The cross-entropy term is added on the host from O.
///
/// Inputs: `r` `[n, K]`, `dist` `[n, K]`, `sigma` `[K]`, `dims` `[2]` u32.
/// Output: `partials` `[OBJ_GROUPS]`.
const OBJECTIVE_SOURCE: &str = r#"
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint nsg = simdgroups_per_threadgroup;
    uint tg = threadgroup_position_in_grid.x;
    threadgroup float sh[MAX_SIMD_GROUPS];
    float acc = 0.0f;
    for (uint cell = tg * nsg + sg; cell < dims[0]; cell += OBJ_GROUPS * nsg) {
        size_t rb = size_t(cell) * K;
        for (uint j = lane; j < uint(K); j += SIMD_W) {
            float v = r[rb + j];
            acc += v * dist[rb + j];
            if (v > 0.0f) acc += sigma[j] * v * log(v);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) sh[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_position_in_threadgroup.x == 0) {
        float t = 0.0f;
        for (uint i = 0; i < nsg; i++) t += sh[i];
        partials[tg] = t;
    }
"#;

/// Out-of-place row L2 normalisation, rows with norm below `NORM_EPS` become
/// zero. One SIMD group per row, lanes stride the `D` features.
///
/// Inputs: `src` `[n, D]`, `dims` `[2]` u32. Output: `dst` `[n, D]`.
const ROW_NORMALISE_SOURCE: &str = r#"
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x * simdgroups_per_threadgroup
        + simdgroup_index_in_threadgroup;
    if (row >= dims[0]) return;
    size_t rb = size_t(row) * D;
    float ss = 0.0f;
    for (uint j = lane; j < uint(D); j += SIMD_W) {
        float v = src[rb + j];
        ss += v * v;
    }
    float nrm = sqrt(simd_sum(ss));
    float inv = nrm > NORM_EPS ? 1.0f / nrm : 0.0f;
    for (uint j = lane; j < uint(D); j += SIMD_W) dst[rb + j] = src[rb + j] * inv;
"#;

/////////////
// Helpers //
/////////////

/// Cast a size that has passed [`check_dims`] to an MLX shape or grid int.
///
/// ### Params
///
/// * `x` - The size
///
/// ### Returns
///
/// `x` as `i32`
fn to_i32(x: usize) -> i32 {
    i32::try_from(x).expect("sizes are checked against i32::MAX on entry")
}

/// Reject sizes whose launch grid would overflow an MLX grid int.
///
/// ### Params
///
/// * `n` - Cell count
/// * `k` - Cluster count
/// * `d` - Embedding dimension
///
/// ### Returns
///
/// `Ok(())`, or `InvalidArgument` naming the limit
fn check_dims(n: usize, k: usize, d: usize) -> Result<(), BixverseErrors> {
    let max = i32::MAX as usize;
    let cells_per_tg = CELL_TG / SIMD_WIDTH;
    if n.div_ceil(cells_per_tg) * CELL_TG > max || k > max || d > max {
        return Err(BixverseErrors::InvalidArgument(format!(
            "Harmony MLX: {n} cells exceed the MLX grid range ({} cells)",
            max / SIMD_WIDTH
        )));
    }
    Ok(())
}

/// Copy rows of a row-major `[n, width]` buffer into the order `order`.
///
/// ### Params
///
/// * `src` - Row-major source
/// * `order` - Source row of each output row
/// * `width` - Row length
///
/// ### Returns
///
/// The permuted rows, row-major
fn permute_rows(src: &[f32], order: &[usize], width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(order.len() * width);
    for &c in order {
        out.extend_from_slice(&src[c * width..(c + 1) * width]);
    }
    out
}

/// Undo [`permute_rows`]: row `i` of `sorted` goes back to row `order[i]`.
///
/// ### Params
///
/// * `sorted` - Row-major rows in sorted order
/// * `order` - Original row of each sorted row
/// * `width` - Row length
///
/// ### Returns
///
/// The rows in the original order
fn unpermute_rows(sorted: &[f32], order: &[usize], width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; sorted.len()];
    for (i, &c) in order.iter().enumerate() {
        out[c * width..(c + 1) * width].copy_from_slice(&sorted[i * width..(i + 1) * width]);
    }
    out
}

/// Evaluate an array and copy it to the host.
///
/// ### Params
///
/// * `x` - Array to read
///
/// ### Returns
///
/// The values, row-major
fn read_host(x: &Array) -> Result<Vec<f32>, BixverseErrors> {
    eval_all(&[x], false)?;
    Ok(x.as_f32()?.to_vec())
}

/////////////////
// LevelLayout //
/////////////////

/// Cells sorted by level, the contiguous level ranges and the run structure
/// over them.
struct LevelLayout {
    /// Original cell index of each sorted position (length n)
    order: Vec<usize>,
    /// Level of each sorted position (length n)
    level: Vec<u32>,
    /// Sorted-position offsets of each level (length b + 1)
    level_offsets: Vec<usize>,
    /// Sorted-position offsets of each run (length n_runs + 1)
    run_bounds: Vec<u32>,
    /// Run offsets of each level (length b + 1)
    level_runs: Vec<u32>,
}

/// Sort cells by level and cut every level into runs of at most
/// [`MLX_HARMONY_RUN_CELLS`]. Empty levels own no rows and no runs.
///
/// ### Params
///
/// * `info` - Batch information for the single covariate
///
/// ### Returns
///
/// The [`LevelLayout`]
fn level_layout(info: &BatchInfo) -> LevelLayout {
    let order: Vec<usize> = info.batch_indices.concat();
    let level: Vec<u32> = order
        .iter()
        .map(|&c| info.cell_to_level[c] as u32)
        .collect();
    let mut level_offsets = vec![0usize];
    let mut run_bounds = vec![0u32];
    let mut level_runs = vec![0u32];
    let mut pos = 0usize;
    for cells in &info.batch_indices {
        let mut left = cells.len();
        while left > 0 {
            let take = left.min(MLX_HARMONY_RUN_CELLS);
            pos += take;
            left -= take;
            run_bounds.push(pos as u32);
        }
        level_offsets.push(pos);
        level_runs.push((run_bounds.len() - 1) as u32);
    }
    LevelLayout {
        order,
        level,
        level_offsets,
        run_bounds,
        level_runs,
    }
}

/////////////
// Kernels //
/////////////

/// The custom kernels of one run, registered once.
struct HarmonyKernels {
    /// Per-cell softmax, see [`EXP_NORMALISE_SOURCE`]
    exp_normalise: MetalKernel,
    /// Damped Jacobi update, see [`JACOBI_SOURCE`]
    jacobi: MetalKernel,
    /// Per-run column sums, see [`RUN_SUMS_SOURCE`]
    run_sums: MetalKernel,
    /// Per-level fold plus penalty table, see [`LEVEL_REDUCE_SOURCE`]
    level_reduce: MetalKernel,
    /// Objective partials, see [`OBJECTIVE_SOURCE`]
    objective: MetalKernel,
    /// Row L2 normalisation, see [`ROW_NORMALISE_SOURCE`]
    row_normalise: MetalKernel,
}

impl HarmonyKernels {
    /// Register every kernel with the shared constants header.
    ///
    /// ### Returns
    ///
    /// The kernel handles
    fn new() -> Self {
        let header = format!(
            "constant constexpr uint SIMD_W = {SIMD_WIDTH};\n\
             constant constexpr uint OBJ_GROUPS = {OBJ_GROUPS};\n\
             constant constexpr uint MAX_SIMD_GROUPS = {MAX_SIMD_GROUPS};\n\
             constant constexpr float DAMP = {R_UPDATE_DAMPING:e}f;\n\
             constant constexpr float NORM_EPS = {NORM_EPS:e}f;\n"
        );
        let k = |name: &str, inputs: &[&str], outputs: &[&str], src: &str| {
            MetalKernel::new(name, inputs, outputs, &header, src)
        };
        Self {
            exp_normalise: k(
                "bixverse_harmony_exp_normalise",
                &["dist", "sigma", "dims"],
                &["out"],
                EXP_NORMALISE_SOURCE,
            ),
            jacobi: k(
                "bixverse_harmony_jacobi",
                &["base", "pen", "level", "r", "dims"],
                &["r_out"],
                JACOBI_SOURCE,
            ),
            run_sums: k(
                "bixverse_harmony_run_sums",
                &["r", "run_bounds"],
                &["partial"],
                RUN_SUMS_SOURCE,
            ),
            level_reduce: k(
                "bixverse_harmony_level_reduce",
                &["partial", "level_runs", "theta", "pr_b", "dims"],
                &["o", "r_sum", "pen"],
                LEVEL_REDUCE_SOURCE,
            ),
            objective: k(
                "bixverse_harmony_objective",
                &["r", "dist", "sigma", "dims"],
                &["partials"],
                OBJECTIVE_SOURCE,
            ),
            row_normalise: k(
                "bixverse_harmony_row_normalise",
                &["src", "dims"],
                &["dst"],
                ROW_NORMALISE_SOURCE,
            ),
        }
    }
}

////////////
// Assign //
////////////

/// Assignments and their per-level statistics after one sweep.
struct Assign {
    /// Soft assignments `[n, k]`; `None` means R equals the round's base
    /// assignments (cold start)
    r: Option<Array>,
    /// Observed counts `[b, k]`
    o: Array,
    /// Per-cluster totals `[k]`
    r_sum: Array,
    /// Penalty table for the next sweep `[b, k]`
    pen: Array,
    /// Objective partials of `r`, if queued
    partials: Option<Array>,
}

////////////////
// HarmonyMlx //
////////////////

/// Per-run constants on the device plus the kernels and the stream. MLX
/// streams are thread affine, so this is `!Send`.
struct HarmonyMlx<'a> {
    /// Batch information of the single covariate, original cell order
    info: &'a BatchInfo,
    /// Sorted layout of the cells
    layout: LevelLayout,
    /// Per-cluster sigma (length k)
    sigma: Vec<f32>,
    /// Per-level theta (length b)
    theta: Vec<f32>,
    /// `sigma` on the device
    sigma_arr: Array,
    /// `theta` on the device
    theta_arr: Array,
    /// Level frequencies on the device `[b]`
    pr_b_arr: Array,
    /// `[n, b]` as u32
    dims: Array,
    /// Level of each sorted cell `[n]` u32
    level_arr: Array,
    /// Run bounds `[n_runs + 1]` u32
    run_bounds_arr: Array,
    /// Run offsets per level `[b + 1]` u32
    level_runs_arr: Array,
    /// The custom kernels
    kernels: HarmonyKernels,
    /// Stream every op runs on
    s: Stream,
    /// Cell count
    n: usize,
    /// Cluster count
    k: usize,
    /// Embedding dimension
    d: usize,
    /// Number of levels
    b: usize,
    /// Number of runs
    n_runs: usize,
}

impl<'a> HarmonyMlx<'a> {
    /// Sort the cells by level and upload the per-run constants.
    ///
    /// ### Params
    ///
    /// * `info` - Batch information of the single covariate
    /// * `sigma` - Per-cluster sigma (length k)
    /// * `theta` - Per-level theta (length b)
    /// * `d` - Embedding dimension
    ///
    /// ### Returns
    ///
    /// The context, or `InvalidArgument` if the sizes exceed the MLX grid
    fn new(
        info: &'a BatchInfo,
        sigma: Vec<f32>,
        theta: Vec<f32>,
        d: usize,
    ) -> Result<Self, BixverseErrors> {
        let n = info.cell_to_level.len();
        let k = sigma.len();
        check_dims(n, k, d)?;
        install_error_handler();
        let layout = level_layout(info);
        let b = info.n_levels;
        let n_runs = layout.run_bounds.len() - 1;
        Ok(Self {
            sigma_arr: Array::from_f32(&sigma, &[to_i32(k)]),
            theta_arr: Array::from_f32(&theta, &[to_i32(b)]),
            pr_b_arr: Array::from_f32(&info.pr_b, &[to_i32(b)]),
            dims: Array::from_u32(&[n as u32, b as u32], &[2]),
            level_arr: Array::from_u32(&layout.level, &[to_i32(n)]),
            run_bounds_arr: Array::from_u32(&layout.run_bounds, &[to_i32(n_runs + 1)]),
            level_runs_arr: Array::from_u32(&layout.level_runs, &[to_i32(b + 1)]),
            kernels: HarmonyKernels::new(),
            s: Stream::default_gpu(),
            info,
            layout,
            sigma,
            theta,
            n,
            k,
            d,
            b,
            n_runs,
        })
    }

    /// Launch grid of the one-SIMD-group-per-row kernels.
    ///
    /// ### Returns
    ///
    /// Total threads along x
    fn cell_grid(&self) -> [i32; 3] {
        let cells_per_tg = CELL_TG / SIMD_WIDTH;
        [to_i32(self.n.div_ceil(cells_per_tg) * CELL_TG), 1, 1]
    }

    /// Cosine distances `2 (1 - z_cos Yᵀ)` as one fused `addmm`.
    ///
    /// ### Params
    ///
    /// * `z_cos` - Cosine-normalised cells `[n, d]`
    /// * `y` - Cosine-normalised centroids `[k, d]`
    ///
    /// ### Returns
    ///
    /// The (lazy) distances `[n, k]`
    fn cosine_distances(&self, z_cos: &Array, y: &Array) -> Result<Array, BixverseErrors> {
        let two = Array::from_f32(&[2.0], &[]);
        two.addmm(z_cos, &y.transpose(&self.s)?, -2.0, 1.0, &self.s)
    }

    /// Per-cell softmax of the distances, see [`EXP_NORMALISE_SOURCE`].
    ///
    /// ### Params
    ///
    /// * `dist` - Distances `[n, k]`
    ///
    /// ### Returns
    ///
    /// The (lazy) assignments `[n, k]`
    fn exp_normalise(&self, dist: &Array) -> Result<Array, BixverseErrors> {
        let mut out = self.kernels.exp_normalise.apply(
            &[dist, &self.sigma_arr, &self.dims],
            &[OutputSpec {
                shape: &[to_i32(self.n), to_i32(self.k)],
                dtype: MLX_FLOAT32,
            }],
            self.cell_grid(),
            [to_i32(CELL_TG), 1, 1],
            &[("K", to_i32(self.k))],
            &self.s,
        )?;
        Ok(out.remove(0))
    }

    /// One damped Jacobi sweep, see [`JACOBI_SOURCE`].
    ///
    /// ### Params
    ///
    /// * `base` - Base assignments of the round `[n, k]`
    /// * `pen` - Penalty table of the previous sweep `[b, k]`
    /// * `r` - Previous sweep's assignments `[n, k]`
    ///
    /// ### Returns
    ///
    /// The (lazy) updated assignments `[n, k]`
    fn jacobi(&self, base: &Array, pen: &Array, r: &Array) -> Result<Array, BixverseErrors> {
        let mut out = self.kernels.jacobi.apply(
            &[base, pen, &self.level_arr, r, &self.dims],
            &[OutputSpec {
                shape: &[to_i32(self.n), to_i32(self.k)],
                dtype: MLX_FLOAT32,
            }],
            self.cell_grid(),
            [to_i32(CELL_TG), 1, 1],
            &[("K", to_i32(self.k))],
            &self.s,
        )?;
        Ok(out.remove(0))
    }

    /// Observed counts, cluster totals and the next penalty table of `r`.
    ///
    /// ### Params
    ///
    /// * `r` - Assignments `[n, k]`, cells sorted by level
    ///
    /// ### Returns
    ///
    /// The (lazy) `(O [b, k], r_sum [k], pen [b, k])`
    fn stats(&self, r: &Array) -> Result<(Array, Array, Array), BixverseErrors> {
        let (b, k) = (to_i32(self.b), to_i32(self.k));
        let partial = self
            .kernels
            .run_sums
            .apply(
                &[r, &self.run_bounds_arr],
                &[OutputSpec {
                    shape: &[to_i32(self.n_runs), k],
                    dtype: MLX_FLOAT32,
                }],
                [to_i32(self.n_runs * RUN_TG), 1, 1],
                [to_i32(RUN_TG), 1, 1],
                &[("K", k)],
                &self.s,
            )?
            .remove(0);
        let mut out = self.kernels.level_reduce.apply(
            &[
                &partial,
                &self.level_runs_arr,
                &self.theta_arr,
                &self.pr_b_arr,
                &self.dims,
            ],
            &[
                OutputSpec {
                    shape: &[b, k],
                    dtype: MLX_FLOAT32,
                },
                OutputSpec {
                    shape: &[k],
                    dtype: MLX_FLOAT32,
                },
                OutputSpec {
                    shape: &[b, k],
                    dtype: MLX_FLOAT32,
                },
            ],
            [to_i32(self.k.div_ceil(REDUCE_TG) * REDUCE_TG), 1, 1],
            [to_i32(REDUCE_TG), 1, 1],
            &[("K", k)],
            &self.s,
        )?;
        let pen = out.remove(2);
        let r_sum = out.remove(1);
        Ok((out.remove(0), r_sum, pen))
    }

    /// Queue the objective partials of `r`, see [`OBJECTIVE_SOURCE`].
    ///
    /// ### Params
    ///
    /// * `r` - Assignments `[n, k]`
    /// * `dist` - Distances `[n, k]`
    ///
    /// ### Returns
    ///
    /// The (lazy) partials `[OBJ_GROUPS]`
    fn objective_partials(&self, r: &Array, dist: &Array) -> Result<Array, BixverseErrors> {
        let mut out = self.kernels.objective.apply(
            &[r, dist, &self.sigma_arr, &self.dims],
            &[OutputSpec {
                shape: &[to_i32(OBJ_GROUPS)],
                dtype: MLX_FLOAT32,
            }],
            [to_i32(OBJ_GROUPS * CELL_TG), 1, 1],
            [to_i32(CELL_TG), 1, 1],
            &[("K", to_i32(self.k))],
            &self.s,
        )?;
        Ok(out.remove(0))
    }

    /// Block on the partials and the statistics, and finish the objective on
    /// the host in f64 with the CPU cross-entropy over the O table.
    ///
    /// ### Params
    ///
    /// * `partials` - Objective partials `[OBJ_GROUPS]`
    /// * `o` - Observed counts `[b, k]`
    /// * `r_sum` - Per-cluster totals `[k]`
    ///
    /// ### Returns
    ///
    /// The Harmony v2 objective
    fn finish_objective(
        &self,
        partials: &Array,
        o: &Array,
        r_sum: &Array,
    ) -> Result<f32, BixverseErrors> {
        eval_all(&[partials, o, r_sum], false)?;
        let err_ent: f64 = partials.as_f32()?.iter().map(|&x| x as f64).sum();
        Ok(objective(
            Variant::V2,
            err_ent,
            0.0,
            &[o.as_f32()?.to_vec()],
            r_sum.as_f32()?,
            std::slice::from_ref(self.info),
            &self.sigma,
            std::slice::from_ref(&self.theta),
            self.n,
        ))
    }

    /// Out-of-place row L2 normalisation, see [`ROW_NORMALISE_SOURCE`].
    ///
    /// ### Params
    ///
    /// * `src` - Rows `[n, d]`
    ///
    /// ### Returns
    ///
    /// The (lazy) normalised rows `[n, d]`
    fn row_normalise(&self, src: &Array) -> Result<Array, BixverseErrors> {
        let mut out = self.kernels.row_normalise.apply(
            &[src, &self.dims],
            &[OutputSpec {
                shape: &[to_i32(self.n), to_i32(self.d)],
                dtype: MLX_FLOAT32,
            }],
            self.cell_grid(),
            [to_i32(CELL_TG), 1, 1],
            &[("D", to_i32(self.d))],
            &self.s,
        )?;
        Ok(out.remove(0))
    }

    /// Rows `start..stop` of a sorted `[n, width]` array.
    ///
    /// ### Params
    ///
    /// * `x` - Array `[n, width]`
    /// * `start` - First row
    /// * `stop` - One past the last row
    /// * `width` - Row length
    ///
    /// ### Returns
    ///
    /// The (lazy) row slice
    fn rows(
        &self,
        x: &Array,
        start: usize,
        stop: usize,
        width: usize,
    ) -> Result<Array, BixverseErrors> {
        x.slice(&[to_i32(start), 0], &[to_i32(stop), to_i32(width)], &self.s)
    }

    /// Ridge right-hand sides `S_b = R_bᵀ Z_b` per level, one GEMM per
    /// non-empty level over contiguous row slices, read back in f64.
    ///
    /// ### Params
    ///
    /// * `r` - Assignments `[n, k]`, sorted
    /// * `z` - Original embedding `[n, d]`, sorted
    ///
    /// ### Returns
    ///
    /// `S` `[b, k, d]` row-major, zero for empty levels
    fn level_weighted_sums(&self, r: &Array, z: &Array) -> Result<Vec<f64>, BixverseErrors> {
        let (k, d) = (self.k, self.d);
        let offs = &self.layout.level_offsets;
        let mut sums: Vec<(usize, Array)> = Vec::with_capacity(self.b);
        for l in 0..self.b {
            if offs[l] == offs[l + 1] {
                continue;
            }
            let r_b = self.rows(r, offs[l], offs[l + 1], k)?;
            let z_b = self.rows(z, offs[l], offs[l + 1], d)?;
            sums.push((l, r_b.transpose(&self.s)?.matmul(&z_b, &self.s)?));
        }
        let refs: Vec<&Array> = sums.iter().map(|(_, a)| a).collect();
        eval_all(&refs, false)?;
        let mut out = vec![0.0f64; self.b * k * d];
        for (l, a) in &sums {
            for (o, &x) in out[l * k * d..(l + 1) * k * d].iter_mut().zip(a.as_f32()?) {
                *o = x as f64;
            }
        }
        Ok(out)
    }

    /// Ridge correction `Z_b - R_b C_b` per non-empty level as one fused
    /// `addmm` each, joined back into `[n, d]`.
    ///
    /// ### Params
    ///
    /// * `z` - Original embedding `[n, d]`, sorted
    /// * `r` - Assignments `[n, k]`, sorted
    /// * `corr` - Corrections `[b, k, d]` row-major from `solve_ridge`
    ///
    /// ### Returns
    ///
    /// The (lazy) corrected embedding `[n, d]`, sorted
    fn ridge_subtract(&self, z: &Array, r: &Array, corr: &[f32]) -> Result<Array, BixverseErrors> {
        let (k, d) = (self.k, self.d);
        let offs = &self.layout.level_offsets;
        let mut pieces: Vec<Array> = Vec::with_capacity(self.b);
        for l in 0..self.b {
            if offs[l] == offs[l + 1] {
                continue;
            }
            let c_b = Array::from_f32(&corr[l * k * d..(l + 1) * k * d], &[to_i32(k), to_i32(d)]);
            let r_b = self.rows(r, offs[l], offs[l + 1], k)?;
            let z_b = self.rows(z, offs[l], offs[l + 1], d)?;
            pieces.push(z_b.addmm(&r_b, &c_b, -1.0, 1.0, &self.s)?);
        }
        let refs: Vec<&Array> = pieces.iter().collect();
        Array::concatenate(&refs, 0, &self.s)
    }

    /// Fresh assignments and statistics from the distances (cold start).
    ///
    /// ### Params
    ///
    /// * `dist` - Distances `[n, k]`
    ///
    /// ### Returns
    ///
    /// The round's base assignments and the cold-start [`Assign`]
    fn cold_start(&self, dist: &Array) -> Result<(Array, Assign), BixverseErrors> {
        let base = self.exp_normalise(dist)?;
        let (o, r_sum, pen) = self.stats(&base)?;
        Ok((
            base,
            Assign {
                r: None,
                o,
                r_sum,
                pen,
                partials: None,
            },
        ))
    }

    /// Up to `params.max_iter_kmeans` Jacobi sweeps with the windowed
    /// convergence check. One sweep stays in flight: sweep `t` is submitted
    /// before the host blocks on the objective of `t - 1`; on convergence at
    /// `t - 1` the speculative sweep is dropped, so the result is the same as
    /// a synchronous loop.
    ///
    /// ### Params
    ///
    /// * `base` - Base assignments of the round `[n, k]`
    /// * `dist` - Distances `[n, k]`
    /// * `start` - State at the start of the round
    /// * `params` - Harmony parameters
    /// * `objectives` - Objective trace, one entry pushed per kept sweep
    ///
    /// ### Returns
    ///
    /// The state after the last kept sweep
    fn sweeps(
        &self,
        base: &Array,
        dist: &Array,
        start: Assign,
        params: &HarmonyParamsV2,
        objectives: &mut Vec<f32>,
    ) -> Result<Assign, BixverseErrors> {
        let mut cur = start;
        let mut flight: Option<Assign> = None;
        for it in 0..params.max_iter_kmeans {
            let src = flight.as_ref().unwrap_or(&cur);
            let r = self.jacobi(base, &src.pen, src.r.as_ref().unwrap_or(base))?;
            let (o, r_sum, pen) = self.stats(&r)?;
            let partials = self.objective_partials(&r, dist)?;
            eval_all(&[&r, &o, &r_sum, &pen, &partials], true)?;
            let next = Assign {
                r: Some(r),
                o,
                r_sum,
                pen,
                partials: Some(partials),
            };
            if let Some(prev) = flight.replace(next) {
                let partials = prev.partials.as_ref().expect("queued with the sweep");
                objectives.push(self.finish_objective(partials, &prev.o, &prev.r_sum)?);
                cur = prev;
                if it - 1 > params.window_size
                    && check_convergence(objectives, params.window_size, params.epsilon_kmeans)
                {
                    return Ok(cur);
                }
            }
        }
        if let Some(last) = flight {
            let partials = last.partials.as_ref().expect("queued with the sweep");
            objectives.push(self.finish_objective(partials, &last.o, &last.r_sum)?);
            cur = last;
        }
        Ok(cur)
    }
}

//////////
// Main //
//////////

/// Harmony v2 on MLX (single covariate).
///
/// Mirrors `harmony_v2_gpu`: per round, refine R with the diversity penalty
/// over `max_iter_kmeans` damped full-batch Jacobi sweeps at fixed distances,
/// apply the batch-pruned ridge correction, take the next centroids from the
/// normalised ridge intercepts, and restart R from the new distances. Reuses
/// [`HarmonyParamsV2`]; `block_size` is ignored, since the Jacobi update has
/// no blocks. The initial k-means is the CPU one. At detailed verbosity the
/// device is synchronised at every stage boundary and the stage split is
/// printed; that costs a few syncs per round, so time totals at verbosity 0.
///
/// ### Params
///
/// * `pca` - PCA embedding `[n, d]`
/// * `batch_labels` - One label slice of length `n`; exactly one variable is
///   supported
/// * `params` - Harmony v2 hyperparameters
/// * `seed` - Random seed for the initial k-means
/// * `verbose` - `0` silent, `1` progress, `2` progress plus stage split
///
/// ### Returns
///
/// Corrected PCA embedding `[n, d]`
///
/// ### Errors
///
/// * `MlxHarmonySupportsSingleCovariateOnly` for more than one variable
/// * `InvalidArgument` if `n` exceeds the MLX grid range
/// * Propagates k-means, batch-label and `MlxError` errors
pub fn harmony_v2_mlx(
    pca: MatRef<f32>,
    batch_labels: &[Vec<usize>],
    params: &HarmonyParamsV2,
    seed: usize,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let split = verbosity.detailed_verbosity();
    let (n, d, k) = (pca.nrows(), pca.ncols(), params.k);
    let start = Instant::now();

    if batch_labels.len() != 1 {
        return Err(BixverseErrors::MlxHarmonySupportsSingleCovariateOnly);
    }
    let infos = create_batch_infos(batch_labels, n)?;

    let sigma = if params.sigma.len() == 1 {
        vec![params.sigma[0]; k]
    } else {
        assert_eq!(params.sigma.len(), k, "sigma must be length 1 or K");
        params.sigma.clone()
    };
    assert_eq!(
        params.theta.len(),
        1,
        "theta must be length 1 (single covariate)"
    );
    let theta_levels = expand_theta(&params.theta, &infos, k, params.tau).remove(0);
    let ridge = RidgeSettings {
        lambda: params.lambda[0],
        alpha: params.alpha,
        dynamic_lambda: params.use_dynamic_lambda,
        prune_cutoff: Some(params.batch_proportion_cutoff),
    };

    let mut times = StageTimes::default();
    let ctx = HarmonyMlx::new(&infos[0], sigma, theta_levels, d)?;
    let order = &ctx.layout.order;
    let sync = |arrays: &[&Array]| {
        if split {
            eval_all(arrays, false)
        } else {
            Ok(())
        }
    };

    // k-means on the unsorted cells, as the CPU v2, then sort
    let pca_rm = to_row_major(pca);
    let mut z_cos_host = vec![0.0f32; n * d];
    normalise_rows_into(&pca_rm, d, &mut z_cos_host);
    if verbosity.normal_verbosity() {
        println!("MLX Harmony v2: running initial k-means");
    }
    let mut y_host = times.time("kmeans_init", || {
        kmeans_centroids(&z_cos_host, d, k, params.kmeans_params, seed, split)
    })?;
    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start.elapsed());
    }

    let (z_orig, mut dist, mut base, mut state) = times.time("init", || {
        let z_orig = Array::from_f32(&permute_rows(&pca_rm, order, d), &[to_i32(n), to_i32(d)]);
        let z_cos = Array::from_f32(
            &permute_rows(&z_cos_host, order, d),
            &[to_i32(n), to_i32(d)],
        );
        let y = Array::from_f32(&y_host, &[to_i32(k), to_i32(d)]);
        let dist = ctx.cosine_distances(&z_cos, &y)?;
        let (base, state) = ctx.cold_start(&dist)?;
        sync(&[&dist, &base, &state.o, &state.r_sum, &state.pen])?;
        Ok::<_, BixverseErrors>((z_orig, dist, base, state))
    })?;

    let initial_obj = times.time("objective", || {
        let partials = ctx.objective_partials(&base, &dist)?;
        ctx.finish_objective(&partials, &state.o, &state.r_sum)
    })?;
    let mut objectives_kmeans: Vec<f32> = vec![initial_obj];
    let mut objectives_harmony: Vec<f32> = vec![initial_obj];
    let mut z_corr: Option<Array> = None;

    if verbosity.normal_verbosity() {
        println!("MLX Harmony v2: initial objective {:.4}", initial_obj);
    }

    for harmony_iter in 0..params.max_iter_harmony {
        if verbosity.normal_verbosity() {
            println!("\n=== Harmony MLX v2 iteration {} ===", harmony_iter + 1);
            println!("  Running k-means clustering...");
        }
        let start_iter = Instant::now();

        state = times.time("sweeps", || {
            ctx.sweeps(&base, &dist, state, params, &mut objectives_kmeans)
        })?;

        if verbosity.normal_verbosity() {
            println!("  Applying ridge regression correction...");
        }

        let r = state.r.as_ref().unwrap_or(&base);
        let sums = times.time("ridge_sums", || {
            let s = ctx.level_weighted_sums(r, &z_orig)?;
            let o: Vec<f64> = read_host(&state.o)?.iter().map(|&x| x as f64).collect();
            Ok::<_, BixverseErrors>([(s, o)])
        })?;
        let out = times.time("ridge_solve", || {
            solve_ridge(&sums, &[], &infos, ridge, k, d)
        });
        let corrected = times.time("ridge_subtract", || {
            let z = ctx.ridge_subtract(&z_orig, r, &out.corr[0])?;
            sync(&[&z])?;
            Ok::<_, BixverseErrors>(z)
        })?;

        // next centroids are the normalised ridge intercepts
        let mut intercept_norm = vec![0.0f32; k * d];
        normalise_rows_into(&out.intercept, d, &mut intercept_norm);
        for (kk, &solved) in out.solved.iter().enumerate() {
            if solved {
                y_host[kk * d..(kk + 1) * d].copy_from_slice(&intercept_norm[kk * d..(kk + 1) * d]);
            }
        }

        (dist, base, state) = times.time("restart", || {
            let y = Array::from_f32(&y_host, &[to_i32(k), to_i32(d)]);
            let z_cos = ctx.row_normalise(&corrected)?;
            let dist = ctx.cosine_distances(&z_cos, &y)?;
            let (base, state) = ctx.cold_start(&dist)?;
            sync(&[&dist, &base, &state.o, &state.r_sum, &state.pen])?;
            Ok::<_, BixverseErrors>((dist, base, state))
        })?;
        z_corr = Some(corrected);

        let harmony_obj = *objectives_kmeans
            .last()
            .expect("trace starts with the initial objective");
        objectives_harmony.push(harmony_obj);

        if verbosity.normal_verbosity() {
            println!(
                "  MLX Harmony v2: iteration {} objective {:.4}",
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
                    " MLX Harmony v2 converged at iteration {}",
                    harmony_iter + 1
                );
            }
            break;
        }
    }

    // back to the caller's cell order
    let z_sorted = read_host(z_corr.as_ref().unwrap_or(&z_orig))?;
    let z_out = unpermute_rows(&z_sorted, order, d);
    drop((z_corr, z_orig, dist, base, state));
    clear_cache()?;

    if verbosity.normal_verbosity() {
        println!(
            "Finished the MLX Harmony (version 2) in {:.2?}",
            start.elapsed()
        );
    }
    if split {
        times.print();
    }
    Ok(row_major_to_mat(&z_out, d))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use crate::single_cell::sc_batch_correction::harmony::create_batch_info;
    use crate::single_cell::sc_batch_correction::harmony_core::ridge_correction;

    /// Deterministic values in `[0, 1)` from an index.
    fn val(i: usize, a: usize, m: usize) -> f32 {
        ((i * a + 7) % m) as f32 / m as f32
    }

    /// Row-stochastic `[n, k]` assignments.
    fn assignments(n: usize, k: usize) -> Vec<f32> {
        let mut r = vec![0.0f32; n * k];
        for c in 0..n {
            let row: Vec<f32> = (0..k).map(|j| val(c * k + j, 11, 13) + 0.1).collect();
            let s: f32 = row.iter().sum();
            for j in 0..k {
                r[c * k + j] = row[j] / s;
            }
        }
        r
    }

    /// Unsorted labels with a single-cell level 1, an empty level 2 and two
    /// large levels 0 and 3.
    fn awkward_labels(n: usize) -> Vec<usize> {
        let mut labels: Vec<usize> = (0..n).map(|i| if i % 3 == 0 { 3 } else { 0 }).collect();
        labels[n / 2] = 1;
        labels
    }

    /// Largest absolute difference, asserting it is within `tol`.
    fn assert_close(got: &[f32], want: &[f32], tol: f32) -> f32 {
        assert_eq!(got.len(), want.len());
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max <= tol, "max diff {max} > {tol}");
        max
    }

    /// Context with flat sigma and theta for `labels`.
    fn context(info: &BatchInfo, k: usize, d: usize) -> HarmonyMlx<'_> {
        HarmonyMlx::new(info, vec![0.1; k], vec![2.0; info.n_levels], d).unwrap()
    }

    /// Parity gate for the objective that drives the convergence check.
    #[test]
    fn test_compute_objective_mlx_matches_cpu_v2() {
        let (n, k) = (16, 4);
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let info = create_batch_info(&labels, n).unwrap();
        let b = info.n_levels;
        let r = assignments(n, k);
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

        let ctx = HarmonyMlx::new(&info, sigma, theta_levels[0].clone(), 1).unwrap();
        let partials = ctx
            .objective_partials(
                &Array::from_f32(&r, &[n as i32, k as i32]),
                &Array::from_f32(&dist, &[n as i32, k as i32]),
            )
            .unwrap();
        let mlx_obj = ctx
            .finish_objective(
                &partials,
                &Array::from_f32(&o, &[b as i32, k as i32]),
                &Array::from_f32(&r_sum, &[k as i32]),
            )
            .unwrap();
        println!("objective: mlx {mlx_obj}, cpu {cpu_obj}");
        assert_relative_eq!(mlx_obj, cpu_obj, epsilon = 1e-2);
    }

    /// The out-of-place row norm matches the host, including a zero row.
    #[test]
    fn test_row_l2_normalise_into_matches_cpu() {
        let (n, dim) = (5, 6);
        let mut src: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
        src[2 * dim..3 * dim].fill(0.0);
        let info = create_batch_info(&vec![0; n], n).unwrap();
        let ctx = context(&info, 2, dim);
        let dst = read_host(
            &ctx.row_normalise(&Array::from_f32(&src, &[n as i32, dim as i32]))
                .unwrap(),
        )
        .unwrap();
        let mut want = vec![0.0f32; n * dim];
        normalise_rows_into(&src, dim, &mut want);
        assert_close(&dst, &want, 1e-5);
    }

    /// Softmax, observed counts, penalty table and one Jacobi sweep against
    /// the host, over levels that span several runs.
    #[test]
    fn test_assignment_kernels_match_cpu() {
        let (n, k) = (2 * MLX_HARMONY_RUN_CELLS + 37, 5);
        let labels = awkward_labels(n);
        let info = create_batch_info(&labels, n).unwrap();
        let ctx = context(&info, k, 1);
        let b = info.n_levels;
        let order = &ctx.layout.order;

        let dist: Vec<f32> = (0..n * k).map(|i| 4.0 * val(i, 13, 17)).collect();
        let dist_s = permute_rows(&dist, order, k);
        let dist_arr = Array::from_f32(&dist_s, &[n as i32, k as i32]);
        let base = ctx.exp_normalise(&dist_arr).unwrap();
        let (o, r_sum, pen) = ctx.stats(&base).unwrap();
        let r_prev_host = permute_rows(&assignments(n, k), order, k);
        let r_prev = Array::from_f32(&r_prev_host, &[n as i32, k as i32]);
        let r_new = ctx.jacobi(&base, &pen, &r_prev).unwrap();

        // host reference, sorted order
        let mut base_h = vec![0.0f32; n * k];
        for c in 0..n {
            let row = &dist_s[c * k..(c + 1) * k];
            let w: Vec<f64> = row.iter().map(|&x| (-(x as f64) / 0.1).exp()).collect();
            let s: f64 = w.iter().sum();
            for j in 0..k {
                base_h[c * k + j] = (w[j] / s) as f32;
            }
        }
        let mut o_acc = vec![0.0f64; b * k];
        for c in 0..n {
            let l = ctx.layout.level[c] as usize;
            for j in 0..k {
                o_acc[l * k + j] += base_h[c * k + j] as f64;
            }
        }
        let o_h: Vec<f32> = o_acc.iter().map(|&x| x as f32).collect();
        let rs_h: Vec<f32> = (0..k)
            .map(|j| (0..b).map(|l| o_h[l * k + j]).sum())
            .collect();
        let mut pen_h = vec![0.0f32; b * k];
        for l in 0..b {
            for j in 0..k {
                let e = rs_h[j] * info.pr_b[l];
                pen_h[l * k + j] = ((2.0 * e + 1.0) / (o_h[l * k + j] + e + 1.0)).powf(2.0);
            }
        }
        let mut r_h = vec![0.0f32; n * k];
        for c in 0..n {
            let l = ctx.layout.level[c] as usize;
            let s: f32 = (0..k).map(|j| base_h[c * k + j] * pen_h[l * k + j]).sum();
            for j in 0..k {
                let v = base_h[c * k + j] * pen_h[l * k + j] / s;
                r_h[c * k + j] =
                    R_UPDATE_DAMPING * v + (1.0 - R_UPDATE_DAMPING) * r_prev_host[c * k + j];
            }
        }

        let d_base = assert_close(&read_host(&base).unwrap(), &base_h, 1e-5);
        let d_o = assert_close(&read_host(&o).unwrap(), &o_h, 1e-2);
        let d_rs = assert_close(&read_host(&r_sum).unwrap(), &rs_h, 1e-2);
        let d_pen = assert_close(&read_host(&pen).unwrap(), &pen_h, 1e-4);
        let d_r = assert_close(&read_host(&r_new).unwrap(), &r_h, 1e-5);
        println!("max diffs: base {d_base}, o {d_o}, r_sum {d_rs}, pen {d_pen}, r {d_r}");
    }

    /// Per-level `R_bᵀ Z_b` against a naive f64 sum on the unsorted cells,
    /// with a single-cell level and an empty level.
    #[test]
    fn test_level_weighted_sums_match_cpu() {
        let (n, k, d) = (301, 4, 7);
        let labels = awkward_labels(n);
        let info = create_batch_info(&labels, n).unwrap();
        let ctx = context(&info, k, d);
        let order = &ctx.layout.order;
        let r = assignments(n, k);
        let z: Vec<f32> = (0..n * d).map(|i| val(i, 7, 23) - 0.5).collect();

        let got = ctx
            .level_weighted_sums(
                &Array::from_f32(&permute_rows(&r, order, k), &[n as i32, k as i32]),
                &Array::from_f32(&permute_rows(&z, order, d), &[n as i32, d as i32]),
            )
            .unwrap();

        let mut want = vec![0.0f64; info.n_levels * k * d];
        for c in 0..n {
            let l = labels[c];
            for j in 0..k {
                for f in 0..d {
                    want[(l * k + j) * d + f] += (r[c * k + j] * z[c * d + f]) as f64;
                }
            }
        }
        let got: Vec<f32> = got.iter().map(|&x| x as f32).collect();
        let want: Vec<f32> = want.iter().map(|&x| x as f32).collect();
        let max = assert_close(&got, &want, 1e-4);
        println!("weighted sums max diff {max}");
    }

    /// The per-level `addmm` correction matches the CPU ridge correction on
    /// unsorted labels with a single-cell level and an empty level.
    #[test]
    fn test_ridge_subtract_matches_cpu() {
        let (n, k, d) = (301, 4, 7);
        let labels = awkward_labels(n);
        let info = create_batch_info(&labels, n).unwrap();
        let ctx = context(&info, k, d);
        let order = &ctx.layout.order;
        let r = assignments(n, k);
        let z: Vec<f32> = (0..n * d).map(|i| val(i, 7, 23) - 0.5).collect();

        let settings = RidgeSettings {
            lambda: 1.0,
            alpha: 0.2,
            dynamic_lambda: true,
            prune_cutoff: None,
        };
        let mut z_corr = vec![0.0f32; n * d];
        let out = ridge_correction(
            &z,
            &r,
            std::slice::from_ref(&info),
            settings,
            k,
            d,
            &mut z_corr,
        );
        assert!(out.corr[0].iter().any(|&x| x != 0.0));

        let got = ctx
            .ridge_subtract(
                &Array::from_f32(&permute_rows(&z, order, d), &[n as i32, d as i32]),
                &Array::from_f32(&permute_rows(&r, order, k), &[n as i32, k as i32]),
                &out.corr[0],
            )
            .unwrap();
        let got = unpermute_rows(&read_host(&got).unwrap(), order, d);
        assert!(z_corr.iter().zip(&z).any(|(a, b)| (a - b).abs() > 1e-3));
        let max = assert_close(&got, &z_corr, 1e-4);
        println!("ridge subtract max diff {max}");
    }

    /// More than one covariate is rejected.
    #[test]
    fn test_harmony_v2_mlx_rejects_two_covariates() {
        let pca = Mat::<f32>::zeros(4, 2);
        let labels = vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]];
        let res = harmony_v2_mlx(pca.as_ref(), &labels, &HarmonyParamsV2::default(), 1, 0);
        assert!(matches!(
            res,
            Err(BixverseErrors::MlxHarmonySupportsSingleCovariateOnly)
        ));
    }

    /// End-to-end on two shifted, interleaved batches: the output is in the
    /// caller's cell order (the sort is undone) and the shift shrinks.
    #[test]
    fn test_harmony_v2_mlx_removes_interleaved_shift() {
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

        let params = HarmonyParamsV2 {
            k: 3,
            max_iter_kmeans: 2,
            max_iter_harmony: 3,
            ..HarmonyParamsV2::default()
        };
        let result = harmony_v2_mlx(pca.as_ref(), std::slice::from_ref(&labels), &params, 42, 0)
            .expect("harmony_v2_mlx should succeed");

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
