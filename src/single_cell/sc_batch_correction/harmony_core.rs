//! Shared numerical core of Harmony v1 and v2 on flat buffers.
//!
//! Every per-cell matrix is cell-major: `R`, `dist` and the base assignments
//! are `[n, k]` (a cell's K values contiguous), embeddings are `[n, d]`
//! row-major. Observed counts are `[n_levels, k]` per batch variable. The
//! variants differ only in the diversity penalty, the cross-entropy term, the
//! ridge pruning and the ridge penalty, all of which are parameters here.
//!
//! Work over cells is split into runs of at most [`RUN_CELLS`] cells of one
//! level (after Korsunsky's harmonypy C++ backend), so the per-level sums and
//! the correction are each one small GEMM per run.

use faer::linalg::solvers::Solve;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use ann_search_rs::{utils::dist::Dist, utils::k_means_utils::train_centroids};

use crate::ml::clustering::k_means::KMeansParamsWrappers;
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::harmony::BatchInfo;
use crate::utils::gemm::gemm;

///////////
// Const //
///////////

/// Cells per task for per-cell loops and per-level runs. Large enough to
/// amortise scheduling and to give the run GEMMs a decent inner dimension,
/// small enough that the gathered `R` and `Z` blocks stay in L2.
pub(crate) const RUN_CELLS: usize = 1024;

/// Lloyd iterations of the initial k-means. R harmony 2.0.5 runs 10
/// (`kmeans_centers` in `src/utils.cpp`). Against 30 on 829k cells x 50 PCs it
/// halved `kmeans_init` with a marginally lower final objective in both v1 and
/// v2, and corrected embeddings within 3% relative Frobenius.
pub(crate) const HARMONY_KMEANS_ITERS: usize = 10;

/// Cells per task in the block assignment update. Blocks are 5% of the data
/// by default, so this is smaller than [`RUN_CELLS`] to keep every thread busy
/// on small data sets.
const BLOCK_TASK_CELLS: usize = 256;

/// How many cells ahead the scattered block pass touches the `base` row it
/// will read. Its rows are far apart in memory, so the hardware prefetcher
/// cannot follow; anything from 2 to 16 measured the same (v2, 829k cells,
/// k = 100), about 12% off `update_r`.
const PREFETCH_CELLS: usize = 4;

/// f32 values per 128-byte cache line on Apple Silicon. On a 64-byte line
/// machine this touches every other line, which still starts the stream.
const LINE_F32: usize = 32;

/// Independent running maxima in [`lane_max`].
const MAX_LANES: usize = 8;

/// Pivot threshold below which the arrowhead Schur complement is treated as
/// degenerate and the solve falls back to LU.
const ARROWHEAD_EPS: f64 = 1e-10;

///////////
// Types //
///////////

/// Which Harmony variant's penalty and objective to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Variant {
    /// Korsunsky et al. 2019: penalty `(E / (O + E))^theta`, cross-entropy
    /// `log((O + E) / E)`
    V1,
    /// Patikas et al. 2026: penalty `((2E + 1) / (O + E + 1))^theta`,
    /// cross-entropy `log((O + E + 1) / (2E + 1))`
    V2,
}

/// Ridge settings for one correction step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RidgeSettings {
    /// Fixed ridge penalty on every batch column (the intercept is never
    /// penalised)
    pub lambda: f32,
    /// Dynamic lambda multiplier, `lambda_kb = alpha * E_kb`
    pub alpha: f32,
    /// Use the dynamic lambda instead of `lambda`
    pub dynamic_lambda: bool,
    /// Minimum `O[k, b] / N_b` for a level to enter cluster k's system; `None`
    /// keeps every level (v1)
    pub prune_cutoff: Option<f32>,
}

/// Output of one ridge solve.
pub(crate) struct RidgeOutput {
    /// Intercept row per cluster `[k, d]` row-major; meaningful only where
    /// `solved[k]`
    pub intercept: Vec<f32>,
    /// Whether cluster k had a system to solve
    pub solved: Vec<bool>,
    /// Correction per variable `[n_levels, k, d]`; zero where a level is not
    /// in the cluster's system
    pub corr: Vec<Vec<f32>>,
}

/// Raw pointer that may be shared across rayon tasks writing disjoint ranges.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);

// SAFETY: only used for writes to disjoint index ranges, see call sites.
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

/////////////
// Helpers //
/////////////

/// Split every level's cells into runs of at most [`RUN_CELLS`].
///
/// ### Params
///
/// * `info` - Batch information for one variable
///
/// ### Returns
///
/// `(level, cells)` per run, borrowing from `info.batch_indices`
fn level_runs(info: &BatchInfo) -> Vec<(usize, &[usize])> {
    info.batch_indices
        .iter()
        .enumerate()
        .flat_map(|(level, cells)| cells.chunks(RUN_CELLS).map(move |c| (level, c)))
        .collect()
}

/// Load one value per cache line of `row` and discard it, so the lines are in
/// flight before the row is needed. A software prefetch on stable Rust: the
/// loads have no dependants, so the core does not wait on them.
///
/// ### Params
///
/// * `row` - The row to pull into cache
#[inline(always)]
fn touch_row(row: &[f32]) {
    for x in row.iter().step_by(LINE_F32) {
        std::hint::black_box(*x);
    }
}

/// Copy the K-vectors of `cells` from a cell-major `[n, k]` matrix into a
/// column-major `[k, m]` block.
///
/// ### Params
///
/// * `src` - Cell-major `[n, k]` matrix
/// * `cells` - Cells to gather
/// * `k` - Row length
/// * `dst` - Output, resized to `k * cells.len()`
fn gather_rows(src: &[f32], cells: &[usize], k: usize, dst: &mut Vec<f32>) {
    dst.clear();
    for &c in cells {
        dst.extend_from_slice(&src[c * k..(c + 1) * k]);
    }
}

/// Normalise every row of a row-major `[n, d]` matrix to unit L2 norm, into
/// `out`. Rows with norm below `1e-8` become zero.
///
/// ### Params
///
/// * `src` - Row-major `[n, d]` input
/// * `d` - Row length
/// * `out` - Row-major `[n, d]` output
pub(crate) fn normalise_rows_into(src: &[f32], d: usize, out: &mut [f32]) {
    out.par_chunks_mut(d * RUN_CELLS)
        .zip(src.par_chunks(d * RUN_CELLS))
        .for_each(|(o, s)| {
            for (orow, srow) in o.chunks_exact_mut(d).zip(s.chunks_exact(d)) {
                let norm = srow.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    let inv = 1.0 / norm;
                    for (a, b) in orow.iter_mut().zip(srow) {
                        *a = b * inv;
                    }
                } else {
                    orow.fill(0.0);
                }
            }
        });
}

///////////////////////////
// Distances and softmax //
///////////////////////////

/// Maximum of a slice over [`MAX_LANES`] independent running maxima.
///
/// A single running max is a serial chain of compare latencies, k long per
/// cell; independent lanes let it vectorise and overlap.
///
/// ### Params
///
/// * `x` - Values, assumed free of NaN
///
/// ### Returns
///
/// The largest value, `-inf` for an empty slice
#[inline]
fn lane_max(x: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; MAX_LANES];
    let mut chunks = x.chunks_exact(MAX_LANES);
    for c in &mut chunks {
        for j in 0..MAX_LANES {
            acc[j] = if c[j] > acc[j] { c[j] } else { acc[j] };
        }
    }
    let mut m = chunks
        .remainder()
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a });
    for a in acc {
        m = if a > m { a } else { m };
    }
    m
}

/// Diversity-free soft assignments from cosine distances, in log and linear
/// form.
///
/// Per run of cells, one GEMM gives `Y z_j` for every cluster. The logits
/// `-2 (1 - Y z_j) / sigma_k` are shifted by their per-cell log-sum-exp, so
/// `log_base` holds the log of the normalised assignment and `shift` the
/// constant that recovers the logit: `dist_k = -sigma_k (log_base_k +
/// shift)`. Storing the logs lets the update compute the entropy and the
/// k-means error without a `ln` or a distance array per element.
///
/// ### Params
///
/// * `z_cos` - Cosine-normalised cells `[n, d]` row-major
/// * `y` - Cosine-normalised centroids `[k, d]` row-major
/// * `sigma` - Per-cluster softness (length k)
/// * `d` - Embedding dimension
/// * `log_base` - Output log assignments `[n, k]`
/// * `base` - Output assignments `[n, k]`, each cell sums to 1
/// * `shift` - Output per-cell log-sum-exp of the logits (length n)
pub(crate) fn distances_and_base(
    z_cos: &[f32],
    y: &[f32],
    sigma: &[f32],
    d: usize,
    log_base: &mut [f32],
    base: &mut [f32],
    shift: &mut [f32],
) {
    let k = sigma.len();
    let y_ref = MatRef::from_row_major_slice(y, k, d);
    let inv_sigma: Vec<f32> = sigma.iter().map(|s| 1.0 / s).collect();

    log_base
        .par_chunks_mut(k * RUN_CELLS)
        .zip(base.par_chunks_mut(k * RUN_CELLS))
        .zip(shift.par_chunks_mut(RUN_CELLS))
        .zip(z_cos.par_chunks(d * RUN_CELLS))
        .for_each(|(((lb_c, base_c), shift_c), z_c)| {
            let m = z_c.len() / d;
            let z_ref = MatRef::from_row_major_slice(z_c, m, d);
            gemm(
                MatMut::from_column_major_slice_mut(lb_c, k, m),
                Accum::Replace,
                y_ref,
                z_ref.transpose(),
                1.0,
                Par::Seq,
            );
            for ((lb, bc), sh) in lb_c
                .chunks_exact_mut(k)
                .zip(base_c.chunks_exact_mut(k))
                .zip(shift_c.iter_mut())
            {
                for (l, is) in lb.iter_mut().zip(&inv_sigma) {
                    *l = -2.0 * (1.0 - *l) * is;
                }
                let max_logit = lane_max(lb);
                let total = f32::bxv_exp_shift_sum(lb, max_logit, bc);
                let inv = 1.0 / total;
                bc.iter_mut().for_each(|b| *b *= inv);
                *sh = max_logit + total.ln();
                lb.iter_mut().for_each(|l| *l -= *sh);
            }
        });
}

////////////////
// Statistics //
////////////////

/// Observed counts per variable and per-cluster totals from scratch.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]`
/// * `infos` - Batch information per variable
/// * `k` - Cluster count
///
/// ### Returns
///
/// `(O per variable [n_levels, k], r_sum [k])`
pub(crate) fn observed_counts(
    r: &[f32],
    infos: &[BatchInfo],
    k: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let o: Vec<Vec<f32>> = infos
        .iter()
        .map(|info| {
            let runs = level_runs(info);
            let partials: Vec<(usize, Vec<f32>)> = runs
                .par_iter()
                .map(|&(level, cells)| {
                    let mut acc = vec![0.0f32; k];
                    for &c in cells {
                        for (a, v) in acc.iter_mut().zip(&r[c * k..(c + 1) * k]) {
                            *a += v;
                        }
                    }
                    (level, acc)
                })
                .collect();
            let mut o = vec![0.0f32; info.n_levels * k];
            for (level, acc) in partials {
                for (a, v) in o[level * k..(level + 1) * k].iter_mut().zip(acc) {
                    *a += v;
                }
            }
            o
        })
        .collect();

    let mut r_sum = vec![0.0f32; k];
    for level in 0..infos[0].n_levels {
        for (a, v) in r_sum.iter_mut().zip(&o[0][level * k..(level + 1) * k]) {
            *a += v;
        }
    }
    (o, r_sum)
}

/// K-means error `sum r * dist` and entropy `sum sigma_k r ln r` of the
/// diversity-free assignments, i.e. `r = base`, accumulated in f64.
///
/// ### Params
///
/// * `log_base` - Log assignments `[n, k]`, see [`distances_and_base`]
/// * `base` - Assignments `[n, k]`
/// * `shift` - Per-cell log-sum-exp (length n)
/// * `sigma` - Per-cluster softness (length k)
///
/// ### Returns
///
/// `(error, entropy)`
pub(crate) fn base_error_and_entropy(
    log_base: &[f32],
    base: &[f32],
    shift: &[f32],
    sigma: &[f32],
) -> (f64, f64) {
    let k = sigma.len();
    base.par_chunks(k * RUN_CELLS)
        .zip(log_base.par_chunks(k * RUN_CELLS))
        .zip(shift.par_chunks(RUN_CELLS))
        .map(|((bc, lc), sc)| {
            let (mut err, mut ent) = (0.0f64, 0.0f64);
            for ((b, l), &sh) in bc.chunks_exact(k).zip(lc.chunks_exact(k)).zip(sc) {
                let (mut sw, mut swl) = (0.0f32, 0.0f32);
                for kk in 0..k {
                    let w = sigma[kk] * b[kk];
                    sw += w;
                    swl += w * l[kk];
                }
                // dist = -sigma (l + shift); ln r = l
                err += (-(swl + sh * sw)) as f64;
                ent += swl as f64;
            }
            (err, ent)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// Harmony objective from the error and entropy sums and the O/E tables.
///
/// The cross-entropy `sum_cells sum_k r sigma_k theta_l log_ratio[k, l]`
/// collapses to `sum_{k, l} sigma_k theta_l log_ratio[k, l] O[k, l]`, since
/// `O[k, l]` is exactly the sum of `r[., k]` over the cells of level `l`.
///
/// ### Params
///
/// * `variant` - Which log ratio to use
/// * `error` - `sum r * dist`
/// * `entropy` - `sum sigma r ln r`
/// * `o` - Observed counts per variable `[n_levels, k]`
/// * `r_sum` - Per-cluster totals (length k)
/// * `infos` - Batch information per variable
/// * `sigma` - Per-cluster softness (length k)
/// * `theta` - Per-level theta per variable
/// * `n` - Cell count
///
/// ### Returns
///
/// `(error + entropy + cross_entropy) * 2000 / n`
#[allow(clippy::too_many_arguments)]
pub(crate) fn objective(
    variant: Variant,
    error: f64,
    entropy: f64,
    o: &[Vec<f32>],
    r_sum: &[f32],
    infos: &[BatchInfo],
    sigma: &[f32],
    theta: &[Vec<f32>],
    n: usize,
) -> f32 {
    let k = sigma.len();
    let mut cross = 0.0f64;
    for (v, info) in infos.iter().enumerate() {
        for level in 0..info.n_levels {
            let pr = info.pr_b[level];
            let th = theta[v][level];
            for kk in 0..k {
                let o_val = o[v][level * k + kk];
                let e_val = r_sum[kk] * pr;
                let lr = match variant {
                    Variant::V1 if e_val > 0.0 => ((o_val + e_val) / e_val).ln(),
                    Variant::V1 => 0.0,
                    Variant::V2 => ((o_val + e_val + 1.0) / (2.0 * e_val + 1.0)).ln(),
                };
                cross += (sigma[kk] * th * lr * o_val) as f64;
            }
        }
    }
    ((error + entropy + cross) * 2000.0 / n as f64) as f32
}

///////////////////////
// Assignment update //
///////////////////////

/// Diversity penalty table for one variable and its log, `[n_levels, k]`
/// each.
///
/// ### Params
///
/// * `variant` - Penalty form
/// * `o` - Observed counts `[n_levels, k]` with the block removed
/// * `r_sum` - Per-cluster totals with the block removed
/// * `info` - Batch information
/// * `theta` - Per-level theta
/// * `pen` - Output penalties `[n_levels, k]`
/// * `log_pen` - Output log penalties `[n_levels, k]`; `-inf` where the
///   penalty is zero
fn penalty_table(
    variant: Variant,
    o: &[f32],
    r_sum: &[f32],
    info: &BatchInfo,
    theta: &[f32],
    pen: &mut [f32],
    log_pen: &mut [f32],
) {
    let k = r_sum.len();
    for level in 0..info.n_levels {
        let pr = info.pr_b[level];
        let th = theta[level];
        for kk in 0..k {
            let i = level * k + kk;
            let o_val = o[i];
            let e_val = r_sum[kk] * pr;
            let lp = match variant {
                Variant::V1 if o_val + e_val > 0.0 => th * (e_val / (o_val + e_val)).ln(),
                Variant::V1 => 0.0,
                Variant::V2 => th * ((2.0 * e_val + 1.0) / (o_val + e_val + 1.0)).ln(),
            };
            // a zero penalty (v1, E = 0) would give 0 * -inf in the entropy
            log_pen[i] = lp.max(f32::MIN);
            pen[i] = lp.exp();
        }
    }
}

/// Add the K-vector `x` into every variable's level table and `r_sum` of a
/// flat accumulator laid out as one block of [`all_block_sums`].
///
/// ### Params
///
/// * `acc` - Accumulator
/// * `x` - The cell's assignments (length k)
/// * `cell` - Cell index
/// * `infos` - Batch information per variable
/// * `offsets` - Start of each variable's table in `acc`
#[inline]
fn accumulate_cell(
    acc: &mut [f32],
    x: &[f32],
    cell: usize,
    infos: &[BatchInfo],
    offsets: &[usize],
) {
    let k = x.len();
    for (v, info) in infos.iter().enumerate() {
        let off = offsets[v] + info.cell_to_level[cell] * k;
        for (a, y) in acc[off..off + k].iter_mut().zip(x) {
            *a += y;
        }
    }
    let len = acc.len();
    for (a, y) in acc[len - k..].iter_mut().zip(x) {
        *a += y;
    }
}

/// Sum the assignments of every block into its own per-variable level tables
/// and per-cluster total, in one sequential pass over the cells.
///
/// A block's cells are not touched before its turn in
/// [`update_assignments`], so these are exactly the sums each block removes.
/// Walking cells in memory order rather than block by block avoids a
/// scattered row gather per block.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]`
/// * `block_of` - Block index per cell (length n)
/// * `n_blocks` - Number of blocks
/// * `infos` - Batch information per variable
/// * `offsets` - Start of each variable's table in one block's accumulator
/// * `k` - Cluster count
///
/// ### Returns
///
/// `[n_blocks, len]` flat, each block laid out as every variable's
/// `[n_levels, k]` table back to back, then `r_sum` (length k)
fn all_block_sums(
    r: &[f32],
    block_of: &[u32],
    n_blocks: usize,
    infos: &[BatchInfo],
    offsets: &[usize],
    k: usize,
) -> Vec<f32> {
    let len = offsets[infos.len()] + k;
    r.par_chunks(k * RUN_CELLS)
        .zip(block_of.par_chunks(RUN_CELLS))
        .enumerate()
        .fold(
            || vec![0.0f32; n_blocks * len],
            |mut acc, (chunk, (rc, bc))| {
                let c0 = chunk * RUN_CELLS;
                for (i, (row, &b)) in rc.chunks_exact(k).zip(bc).enumerate() {
                    let off = b as usize * len;
                    accumulate_cell(&mut acc[off..off + len], row, c0 + i, infos, offsets);
                }
                acc
            },
        )
        .reduce(|| vec![0.0f32; n_blocks * len], add_into)
}

/// Element-wise `a += b`, returning `a`.
///
/// ### Params
///
/// * `a` - Accumulator
/// * `b` - Addend, same length
///
/// ### Returns
///
/// `a`
fn add_into(mut a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
    a
}

/// Apply a flat block-sum accumulator to O and `r_sum` with sign `sign`.
///
/// ### Params
///
/// * `sums` - One block of [`all_block_sums`], or the same layout
/// * `offsets` - Variable table offsets
/// * `o` - Observed counts per variable
/// * `r_sum` - Per-cluster totals
/// * `sign` - `1.0` to add, `-1.0` to remove
fn apply_sums(sums: &[f32], offsets: &[usize], o: &mut [Vec<f32>], r_sum: &mut [f32], sign: f32) {
    for (v, ov) in o.iter_mut().enumerate() {
        for (a, x) in ov.iter_mut().zip(&sums[offsets[v]..offsets[v + 1]]) {
            *a += sign * x;
        }
    }
    for (a, x) in r_sum.iter_mut().zip(&sums[offsets[o.len()]..]) {
        *a += sign * x;
    }
}

/////////////////
// CellScratch //
/////////////////

/// Per-task state of the block reassignment.
struct CellScratch {
    /// Level sums of the new assignments, laid out as one block of
    /// [`all_block_sums`]; empty in the write-back pass
    sums: Vec<f32>,
    /// The cell's combined penalty (multi-variable only)
    pen: Vec<f32>,
    /// The cell's combined log penalty (multi-variable only)
    lp: Vec<f32>,
    /// The cell's penalised assignments
    x: Vec<f32>,
    /// `sigma * r` for the cell
    w: Vec<f32>,
}

impl CellScratch {
    /// Zeroed state.
    ///
    /// ### Params
    ///
    /// * `len` - Length of the level-sum accumulator
    /// * `k` - Cluster count
    ///
    /// ### Returns
    ///
    /// Initialised self
    fn new(len: usize, k: usize) -> Self {
        Self {
            sums: vec![0.0; len],
            pen: vec![0.0; k],
            lp: vec![0.0; k],
            x: vec![0.0; k],
            w: vec![0.0; k],
        }
    }
}

/// A cell's penalty and log penalty, multiplied (added) over all variables.
///
/// ### Params
///
/// * `c` - Cell index
/// * `pen` - One block's penalty tables, every variable back to back
/// * `log_pen` - The matching log penalty tables
/// * `infos` - Batch information per variable
/// * `offsets` - Start of each variable's table
/// * `pen_s` - Scratch for the combined penalty (length k), used with more
///   than one variable
/// * `lp_s` - Scratch for the combined log penalty (length k)
///
/// ### Returns
///
/// `(penalty, log penalty)`, each length k
#[inline]
fn cell_penalty<'a>(
    c: usize,
    pen: &'a [f32],
    log_pen: &'a [f32],
    infos: &[BatchInfo],
    offsets: &[usize],
    pen_s: &'a mut [f32],
    lp_s: &'a mut [f32],
) -> (&'a [f32], &'a [f32]) {
    let k = pen_s.len();
    let off0 = offsets[0] + infos[0].cell_to_level[c] * k;
    if infos.len() == 1 {
        return (&pen[off0..off0 + k], &log_pen[off0..off0 + k]);
    }
    pen_s.copy_from_slice(&pen[off0..off0 + k]);
    lp_s.copy_from_slice(&log_pen[off0..off0 + k]);
    for (v, info) in infos.iter().enumerate().skip(1) {
        let off = offsets[v] + info.cell_to_level[c] * k;
        for kk in 0..k {
            pen_s[kk] *= pen[off + kk];
            lp_s[kk] += log_pen[off + kk];
        }
    }
    (pen_s, lp_s)
}

/// `out = base * pen` element-wise, returning the sum of `out`.
///
/// ### Params
///
/// * `base` - The cell's diversity-free assignments
/// * `pen` - The cell's combined penalty
/// * `out` - Output, same length
///
/// ### Returns
///
/// The normaliser `sum(base * pen)`
#[inline]
fn penalised(base: &[f32], pen: &[f32], out: &mut [f32]) -> f32 {
    for ((x, b), p) in out.iter_mut().zip(base).zip(pen) {
        *x = b * p;
    }
    f32::bxv_sum(out)
}

/// Block-wise diversity-penalised update of all soft assignments.
///
/// Cells are visited in a seeded random order in `ceil(1 / block_size)`
/// blocks; the last block takes every remaining cell. Per block: remove the
/// block from O and `r_sum`, build each variable's `[n_levels, k]` penalty
/// table once (`O`, `E` are frozen within a block), reassign every block cell
/// in parallel as `base * prod_v penalty_v[level_v]` normalised, then add the
/// block back.
///
/// A block's cells are scattered across memory, and that gather is the cost.
/// So the block loop reads only `base` and accumulates level sums, without
/// touching `r`: a cell's new assignment depends only on its block's penalty
/// tables. Every block's removal sums come from one pass in memory order up
/// front, and a final pass in memory order recomputes and writes `r` from the
/// stored tables. The k-means error and entropy come from the log tables in
/// that pass, with no per-element `ln`: `ln r = log_base + log_pen - ln T` and
/// `dist = -sigma (log_base + shift)`.
///
/// ### Params
///
/// * `variant` - Penalty form
/// * `r` - Soft assignments `[n, k]`, updated in place
/// * `base` - Diversity-free assignments `[n, k]`
/// * `log_base` - Their logs `[n, k]`
/// * `shift` - Per-cell log-sum-exp (length n)
/// * `sigma` - Per-cluster softness (length k)
/// * `theta` - Per-level theta per variable
/// * `infos` - Batch information per variable
/// * `o` - Observed counts per variable, kept in sync with `r`
/// * `r_sum` - Per-cluster totals, kept in sync with `r`
/// * `block_size` - Fraction of cells per block
/// * `seed` - Shuffle seed
///
/// ### Returns
///
/// `(error, entropy)` of the updated assignments
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_assignments(
    variant: Variant,
    r: &mut [f32],
    base: &[f32],
    log_base: &[f32],
    shift: &[f32],
    sigma: &[f32],
    theta: &[Vec<f32>],
    infos: &[BatchInfo],
    o: &mut [Vec<f32>],
    r_sum: &mut [f32],
    block_size: f32,
    seed: u64,
) -> (f64, f64) {
    let k = sigma.len();
    let n = r.len() / k;
    let n_vars = infos.len();

    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));

    let n_blocks = (1.0 / block_size).ceil() as usize;
    let cells_per_block = ((n as f32 * block_size) as usize).max(1);
    let bounds: Vec<(usize, usize)> = (0..n_blocks)
        .map(|b| b * cells_per_block)
        .take_while(|&lo| lo < n)
        .enumerate()
        .map(|(b, lo)| {
            let hi = if b == n_blocks - 1 {
                n
            } else {
                (lo + cells_per_block).min(n)
            };
            (lo, hi)
        })
        .collect();
    // only block membership is random: bucket the cells by block in memory
    // order, so every block's row reads are monotone
    let mut block_of = vec![0u32; n];
    for (bi, &(lo, hi)) in bounds.iter().enumerate() {
        for &c in &order[lo..hi] {
            block_of[c] = bi as u32;
        }
    }
    let mut cursor: Vec<usize> = bounds.iter().map(|&(lo, _)| lo).collect();
    for (c, &bi) in block_of.iter().enumerate() {
        order[cursor[bi as usize]] = c;
        cursor[bi as usize] += 1;
    }
    let blocks: Vec<&[usize]> = bounds.iter().map(|&(lo, hi)| &order[lo..hi]).collect();

    let mut offsets = Vec::with_capacity(n_vars + 1);
    offsets.push(0);
    for info in infos {
        offsets.push(offsets.last().unwrap() + info.n_levels * k);
    }
    let len = offsets[n_vars] + k;
    let tab = offsets[n_vars];
    let n_blk = blocks.len();
    // every block's penalty tables, kept for the write-back pass
    let mut pen = vec![0.0f32; n_blk * tab];
    let mut log_pen = vec![0.0f32; n_blk * tab];

    let removed_all = all_block_sums(r, &block_of, n_blk, infos, &offsets, k);

    for (b, block) in blocks.iter().enumerate() {
        apply_sums(
            &removed_all[b * len..(b + 1) * len],
            &offsets,
            o,
            r_sum,
            -1.0,
        );
        let pen_b = &mut pen[b * tab..(b + 1) * tab];
        let log_pen_b = &mut log_pen[b * tab..(b + 1) * tab];
        for (v, info) in infos.iter().enumerate() {
            let range = offsets[v]..offsets[v + 1];
            penalty_table(
                variant,
                &o[v],
                r_sum,
                info,
                &theta[v],
                &mut pen_b[range.clone()],
                &mut log_pen_b[range],
            );
        }

        let (pen_b, log_pen_b) = (&*pen_b, &*log_pen_b);
        let added = block
            .par_chunks(BLOCK_TASK_CELLS)
            .fold(
                || CellScratch::new(len, k),
                |mut st, chunk| {
                    for (i, &c) in chunk.iter().enumerate() {
                        if let Some(&ahead) = chunk.get(i + PREFETCH_CELLS) {
                            touch_row(&base[ahead * k..(ahead + 1) * k]);
                        }
                        let bc = &base[c * k..(c + 1) * k];
                        let (pen_c, _) = cell_penalty(
                            c,
                            pen_b,
                            log_pen_b,
                            infos,
                            &offsets,
                            &mut st.pen,
                            &mut st.lp,
                        );
                        let total = penalised(bc, pen_c, &mut st.x);
                        if total <= 0.0 {
                            continue;
                        }
                        let inv = 1.0 / total;
                        st.x.iter_mut().for_each(|x| *x *= inv);
                        accumulate_cell(&mut st.sums, &st.x, c, infos, &offsets);
                    }
                    st
                },
            )
            .map(|st| st.sums)
            .reduce(|| vec![0.0f32; len], add_into);
        apply_sums(&added, &offsets, o, r_sum, 1.0);
    }

    // write-back in memory order: recompute each cell's assignment from its
    // block's tables, bit-identical to the one summed above
    let (error, entropy) = r
        .par_chunks_mut(k * RUN_CELLS)
        .zip(block_of.par_chunks(RUN_CELLS))
        .enumerate()
        .fold(
            || (CellScratch::new(0, k), 0.0f64, 0.0f64),
            |(mut st, mut err, mut ent), (chunk, (rc, bc))| {
                let c0 = chunk * RUN_CELLS;
                for (i, (row, &bi)) in rc.chunks_exact_mut(k).zip(bc).enumerate() {
                    let c = c0 + i;
                    let bi = bi as usize;
                    let (pen_b, log_pen_b) = (
                        &pen[bi * tab..(bi + 1) * tab],
                        &log_pen[bi * tab..(bi + 1) * tab],
                    );
                    let (pen_c, lp_c) = cell_penalty(
                        c,
                        pen_b,
                        log_pen_b,
                        infos,
                        &offsets,
                        &mut st.pen,
                        &mut st.lp,
                    );
                    let total = penalised(&base[c * k..(c + 1) * k], pen_c, row);
                    if total <= 0.0 {
                        continue;
                    }
                    let inv = 1.0 / total;
                    for ((x, w), s) in row.iter_mut().zip(st.w.iter_mut()).zip(sigma) {
                        *x *= inv;
                        *w = s * *x;
                    }
                    // ln r = lc + lp - ln(total), dist = -sigma (lc + shift)
                    let lc = &log_base[c * k..(c + 1) * k];
                    let sw = f32::bxv_sum(&st.w);
                    let swb = f32::bxv_dot_simd(&st.w, lc);
                    let swp = f32::bxv_dot_simd(&st.w, lp_c);
                    ent += (swb + swp - total.ln() * sw) as f64;
                    err += (-(swb + shift[c] * sw)) as f64;
                }
                (st, err, ent)
            },
        )
        .map(|(_, e, h)| (e, h))
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    (error, entropy)
}

//////////////////////
// Ridge correction //
//////////////////////

/// Per-level weighted sums for one variable: `S[l, k, :] = sum_{cells of l}
/// r[c, k] z[c, :]` and `O[l, k] = sum_{cells of l} r[c, k]`, accumulated in
/// f64. One gathered GEMM per run.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]`
/// * `z` - Original embedding `[n, d]` row-major
/// * `info` - Batch information
/// * `k` - Cluster count
/// * `d` - Embedding dimension
///
/// ### Returns
///
/// `(S [n_levels, k, d], O [n_levels, k])`
fn level_sums(r: &[f32], z: &[f32], info: &BatchInfo, k: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let runs = level_runs(info);
    let partials: Vec<(usize, Vec<f32>, Vec<f32>)> = runs
        .par_iter()
        .map_init(
            || (Vec::new(), Vec::new()),
            |(r_buf, z_buf), &(level, cells)| {
                let m = cells.len();
                gather_rows(r, cells, k, r_buf);
                gather_rows(z, cells, d, z_buf);
                let mut s = vec![0.0f32; k * d];
                gemm(
                    MatMut::from_row_major_slice_mut(&mut s, k, d),
                    Accum::Replace,
                    MatRef::from_column_major_slice(r_buf, k, m),
                    MatRef::from_row_major_slice(z_buf, m, d),
                    1.0,
                    Par::Seq,
                );
                let mut o = vec![0.0f32; k];
                for rc in r_buf.chunks_exact(k) {
                    for (a, x) in o.iter_mut().zip(rc) {
                        *a += x;
                    }
                }
                (level, s, o)
            },
        )
        .collect();

    let mut s = vec![0.0f64; info.n_levels * k * d];
    let mut o = vec![0.0f64; info.n_levels * k];
    for (level, sp, op) in partials {
        for (a, x) in s[level * k * d..(level + 1) * k * d].iter_mut().zip(sp) {
            *a += x as f64;
        }
        for (a, x) in o[level * k..(level + 1) * k].iter_mut().zip(op) {
            *a += x as f64;
        }
    }
    (s, o)
}

/// Overlap counts between two variables: `P[(la, lb), k] = sum r[c, k]` over
/// the cells at level `la` of variable `a` and `lb` of variable `b`.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]`
/// * `a` - First variable
/// * `b` - Second variable
/// * `k` - Cluster count
///
/// ### Returns
///
/// `P` as `[n_levels_a * n_levels_b, k]`, f64
fn pair_overlaps(r: &[f32], a: &BatchInfo, b: &BatchInfo, k: usize) -> Vec<f64> {
    let len = a.n_levels * b.n_levels * k;
    let n = r.len() / k;
    let chunk = n.div_ceil(4 * rayon::current_num_threads()).max(RUN_CELLS);
    (0..n)
        .into_par_iter()
        .with_min_len(chunk)
        .fold(
            || vec![0.0f64; len],
            |mut acc, c| {
                let off = (a.cell_to_level[c] * b.n_levels + b.cell_to_level[c]) * k;
                for (x, v) in acc[off..off + k].iter_mut().zip(&r[c * k..(c + 1) * k]) {
                    *x += *v as f64;
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; len],
            |mut x, y| {
                for (p, q) in x.iter_mut().zip(y) {
                    *p += q;
                }
                x
            },
        )
}

/// Solve an arrowhead system `A W = B` in f64, where `A` has a dense first
/// row/column and is otherwise diagonal.
///
/// ### Params
///
/// * `a` - The `p x p` system
/// * `rhs` - Right-hand side `p x d`
///
/// ### Returns
///
/// `Some(W)` or `None` if a pivot is degenerate
pub(crate) fn solve_arrowhead_f64(a: MatRef<f64>, rhs: MatRef<f64>) -> Option<Mat<f64>> {
    let p = a.nrows();
    let d = rhs.ncols();
    // eliminate the diagonal block: s = a00 - sum c_i^2 / D_i
    let mut inv_diag = vec![0.0f64; p];
    let mut schur = a[(0, 0)];
    for i in 1..p {
        if a[(i, i)].abs() < ARROWHEAD_EPS {
            return None;
        }
        inv_diag[i] = 1.0 / a[(i, i)];
        schur -= a[(0, i)] * a[(0, i)] * inv_diag[i];
    }
    if schur.abs() < ARROWHEAD_EPS {
        return None;
    }
    let mut w = Mat::<f64>::zeros(p, d);
    for f in 0..d {
        let mut b0 = rhs[(0, f)];
        for i in 1..p {
            b0 -= a[(0, i)] * inv_diag[i] * rhs[(i, f)];
        }
        let w0 = b0 / schur;
        w[(0, f)] = w0;
        for i in 1..p {
            w[(i, f)] = (rhs[(i, f)] - a[(0, i)] * w0) * inv_diag[i];
        }
    }
    Some(w)
}

/// Solve a symmetric positive definite system in f64 via Cholesky, falling
/// back to partial-pivot LU.
///
/// ### Params
///
/// * `a` - The `p x p` system
/// * `rhs` - Right-hand side `p x d`
///
/// ### Returns
///
/// `W` (`p x d`)
fn solve_spd_f64(a: MatRef<f64>, rhs: MatRef<f64>) -> Mat<f64> {
    match a.llt(Side::Lower) {
        Ok(llt) => llt.solve(rhs),
        Err(_) => a.partial_piv_lu().solve(rhs),
    }
}

/// Assemble and solve every cluster's ridge normal equations from the
/// per-level tables alone:
///
/// `[ r_sum_k   O_k^T            ] + diag(0, lambda)`
/// `[ O_k       diag(O_k) + P_k   ]`
///
/// with the intercept unpenalised and the right-hand side built from the
/// weighted sums. A cluster with one active variable is an arrowhead system;
/// otherwise it is solved by Cholesky. Solves are in f64.
///
/// ### Params
///
/// * `sums` - Per variable `(S [n_levels, k, d], O [n_levels, k])`
/// * `pairs` - Overlap counts `P [(la, lb), k]` per variable pair `(a, b)`,
///   `a < b`; empty for one variable
/// * `infos` - Batch information per variable
/// * `settings` - Ridge penalties and pruning
/// * `k` - Cluster count
/// * `d` - Embedding dimension
///
/// ### Returns
///
/// Intercepts, solved flags and the per-variable correction tables
pub(crate) fn solve_ridge(
    sums: &[(Vec<f64>, Vec<f64>)],
    pairs: &[((usize, usize), Vec<f64>)],
    infos: &[BatchInfo],
    settings: RidgeSettings,
    k: usize,
    d: usize,
) -> RidgeOutput {
    let r_sum: Vec<f64> = (0..k)
        .map(|kk| (0..infos[0].n_levels).map(|l| sums[0].1[l * k + kk]).sum())
        .collect();

    // per cluster: column of each (var, level), or usize::MAX if not in the
    // system, and the solution
    type Solution = (Vec<Vec<usize>>, Mat<f64>);
    let solutions: Vec<Option<Solution>> = (0..k)
        .into_par_iter()
        .map(|kk| {
            let mut cols: Vec<Vec<usize>> =
                infos.iter().map(|i| vec![usize::MAX; i.n_levels]).collect();
            let mut col_map: Vec<(usize, usize)> = Vec::new();
            let mut n_active = 0usize;
            for (v, info) in infos.iter().enumerate() {
                let passing: Vec<usize> = (0..info.n_levels)
                    .filter(|&l| {
                        let n_l = info.batch_indices[l].len();
                        match settings.prune_cutoff {
                            None => n_l > 0,
                            Some(cut) => n_l > 0 && sums[v].1[l * k + kk] / n_l as f64 > cut as f64,
                        }
                    })
                    .collect();
                if passing.len() > 1 || (settings.prune_cutoff.is_none() && !passing.is_empty()) {
                    n_active += 1;
                    for l in passing {
                        cols[v][l] = 1 + col_map.len();
                        col_map.push((v, l));
                    }
                }
            }
            if col_map.is_empty() {
                return None;
            }

            let p = 1 + col_map.len();
            let mut a = Mat::<f64>::zeros(p, p);
            let mut rhs = Mat::<f64>::zeros(p, d);
            a[(0, 0)] = r_sum[kk];
            for l in 0..infos[0].n_levels {
                let base = (l * k + kk) * d;
                for f in 0..d {
                    rhs[(0, f)] += sums[0].0[base + f];
                }
            }
            for (ci, &(v, l)) in col_map.iter().enumerate() {
                let c = ci + 1;
                let o_val = sums[v].1[l * k + kk];
                let penalty = if settings.dynamic_lambda {
                    (settings.alpha as f64) * r_sum[kk] * infos[v].pr_b[l] as f64
                } else {
                    settings.lambda as f64
                };
                a[(0, c)] = o_val;
                a[(c, 0)] = o_val;
                a[(c, c)] = o_val + penalty;
                let base = (l * k + kk) * d;
                for f in 0..d {
                    rhs[(c, f)] = sums[v].0[base + f];
                }
            }
            for ((va, vb), p_ab) in pairs {
                let (ia, ib) = (&infos[*va], &infos[*vb]);
                for la in 0..ia.n_levels {
                    let ca = cols[*va][la];
                    if ca == usize::MAX {
                        continue;
                    }
                    for lb in 0..ib.n_levels {
                        let cb = cols[*vb][lb];
                        if cb == usize::MAX {
                            continue;
                        }
                        let ov = p_ab[(la * ib.n_levels + lb) * k + kk];
                        a[(ca, cb)] += ov;
                        a[(cb, ca)] += ov;
                    }
                }
            }

            let w = if n_active == 1 {
                solve_arrowhead_f64(a.as_ref(), rhs.as_ref())
                    .unwrap_or_else(|| solve_spd_f64(a.as_ref(), rhs.as_ref()))
            } else {
                solve_spd_f64(a.as_ref(), rhs.as_ref())
            };
            Some((cols, w))
        })
        .collect();

    // correction tables C_v [n_levels, k, d], zero where a level is not in the
    // cluster's system, plus the intercepts
    let mut intercept = vec![0.0f32; k * d];
    let mut solved = vec![false; k];
    let mut corr: Vec<Vec<f32>> = infos
        .iter()
        .map(|i| vec![0.0f32; i.n_levels * k * d])
        .collect();
    for (kk, sol) in solutions.iter().enumerate() {
        let Some((cols, w)) = sol else { continue };
        solved[kk] = true;
        for f in 0..d {
            intercept[kk * d + f] = w[(0, f)] as f32;
        }
        for (v, vcols) in cols.iter().enumerate() {
            for (l, &c) in vcols.iter().enumerate() {
                if c == usize::MAX {
                    continue;
                }
                let base = (l * k + kk) * d;
                for f in 0..d {
                    corr[v][base + f] = w[(c, f)] as f32;
                }
            }
        }
    }

    RidgeOutput {
        intercept,
        solved,
        corr,
    }
}

/// Batch-pruned, multi-variable ridge correction in one pass over the cells.
///
/// Builds every variable's per-level weighted sums (and, with more than one
/// variable, every pair's overlap counts) in one sweep, solves every cluster
/// with [`solve_ridge`], and applies `z_corr = z - sum_v R^T C_v[level_v]` as
/// one GEMM per run, variable by variable.
///
/// ### Params
///
/// * `z` - Original embedding `[n, d]` row-major
/// * `r` - Soft assignments `[n, k]`
/// * `infos` - Batch information per variable
/// * `settings` - Ridge penalties and pruning
/// * `k` - Cluster count
/// * `d` - Embedding dimension
/// * `z_corr` - Output corrected embedding `[n, d]` row-major
///
/// ### Returns
///
/// The ridge solution, see [`RidgeOutput`]
pub(crate) fn ridge_correction(
    z: &[f32],
    r: &[f32],
    infos: &[BatchInfo],
    settings: RidgeSettings,
    k: usize,
    d: usize,
    z_corr: &mut [f32],
) -> RidgeOutput {
    let n_vars = infos.len();
    let sums: Vec<(Vec<f64>, Vec<f64>)> = infos
        .iter()
        .map(|info| level_sums(r, z, info, k, d))
        .collect();

    let pairs: Vec<((usize, usize), Vec<f64>)> = if n_vars > 1 {
        (0..n_vars)
            .flat_map(|a| ((a + 1)..n_vars).map(move |b| (a, b)))
            .map(|(a, b)| ((a, b), pair_overlaps(r, &infos[a], &infos[b], k)))
            .collect()
    } else {
        Vec::new()
    };

    let out = solve_ridge(&sums, &pairs, infos, settings, k, d);

    z_corr.copy_from_slice(z);
    let zc_ptr = SyncPtr(z_corr.as_mut_ptr());
    for (v, info) in infos.iter().enumerate() {
        let runs = level_runs(info);
        let c_v = &out.corr[v];
        runs.par_iter().for_each_init(
            || (Vec::new(), Vec::new()),
            |(r_buf, delta), &(level, cells)| {
                let ptr = zc_ptr;
                let m = cells.len();
                gather_rows(r, cells, k, r_buf);
                delta.clear();
                delta.resize(m * d, 0.0f32);
                gemm(
                    MatMut::from_row_major_slice_mut(delta, m, d),
                    Accum::Replace,
                    MatRef::from_column_major_slice(r_buf, k, m).transpose(),
                    MatRef::from_row_major_slice(&c_v[level * k * d..(level + 1) * k * d], k, d),
                    1.0,
                    Par::Seq,
                );
                for (i, &c) in cells.iter().enumerate() {
                    // SAFETY: within one variable every cell is in exactly one
                    // run, so rows are written by one task only
                    let row = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(c * d), d) };
                    for (x, dv) in row.iter_mut().zip(&delta[i * d..(i + 1) * d]) {
                        *x -= dv;
                    }
                }
            },
        );
    }

    out
}

/// Centroids `normalise(R^T z_cos)`, `[k, d]` row-major.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]`
/// * `z_cos` - Cosine-normalised cells `[n, d]` row-major
/// * `k` - Cluster count
/// * `d` - Embedding dimension
/// * `y` - Output centroids `[k, d]` row-major
pub(crate) fn centroids_from_r(r: &[f32], z_cos: &[f32], k: usize, d: usize, y: &mut [f32]) {
    let n = r.len() / k;
    let mut raw = vec![0.0f32; k * d];
    gemm(
        MatMut::from_row_major_slice_mut(&mut raw, k, d),
        Accum::Replace,
        MatRef::from_column_major_slice(r, k, n),
        MatRef::from_row_major_slice(z_cos, n, d),
        1.0,
        crate::utils::faer_parallelism(),
    );
    normalise_rows_into(&raw, d, y);
}

/// Flatten a column-major faer matrix into row-major.
///
/// ### Params
///
/// * `m` - Matrix
///
/// ### Returns
///
/// Row-major copy
pub(crate) fn to_row_major(m: MatRef<f32>) -> Vec<f32> {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut out = vec![0.0f32; rows * cols];
    out.par_chunks_mut(cols).enumerate().for_each(|(i, row)| {
        for (j, x) in row.iter_mut().enumerate() {
            *x = m[(i, j)];
        }
    });
    out
}

/// Initial centroids: cosine k-means on the normalised cells, row-normalised.
///
/// ### Params
///
/// * `z_cos` - Cosine-normalised cells `[n, d]` row-major
/// * `d` - Embedding dimension
/// * `k` - Cluster count
/// * `params` - k-means parameters
/// * `seed` - Random seed
/// * `verbose` - Print k-means progress
///
/// ### Returns
///
/// Centroids `[k, d]` row-major, unit norm
pub(crate) fn kmeans_centroids(
    z_cos: &[f32],
    d: usize,
    k: usize,
    params: KMeansParamsWrappers,
    seed: usize,
    verbose: bool,
) -> Result<Vec<f32>, BixverseErrors> {
    let n = z_cos.len() / d;
    let raw = train_centroids(
        z_cos,
        d,
        n,
        k,
        &Dist::Cosine,
        Some(params.get_data()),
        seed,
        verbose,
    )?;
    let mut y = vec![0.0f32; k * d];
    normalise_rows_into(&raw, d, &mut y);
    Ok(y)
}

/// Windowed convergence test of the clustering objective, as in R harmony:
/// the sum of the last `window` objectives against the same window shifted
/// back by one.
///
/// ### Params
///
/// * `objectives` - Objective trace
/// * `window` - Window length
/// * `epsilon` - Relative tolerance
/// * `abs_change` - Compare `|old - new|` (v2) rather than `old - new` (v1,
///   where an increase also counts as converged)
///
/// ### Returns
///
/// Whether the relative change is below `epsilon`
pub(crate) fn window_converged(
    objectives: &[f32],
    window: usize,
    epsilon: f32,
    abs_change: bool,
) -> bool {
    let n = objectives.len();
    if n < window + 1 {
        return false;
    }
    let old: f32 = (0..window).map(|i| objectives[n - 2 - i]).sum();
    let new: f32 = (0..window).map(|i| objectives[n - 1 - i]).sum();
    let change = if abs_change {
        (old - new).abs()
    } else {
        old - new
    };
    change / old.abs() < epsilon
}

/// Soft assignments as a `[k, n]` faer matrix, the layout `HarmonyResult`
/// exposes.
///
/// ### Params
///
/// * `r` - Soft assignments `[n, k]`
/// * `k` - Cluster count
///
/// ### Returns
///
/// `R` (`k x n`)
pub(crate) fn r_to_mat(r: &[f32], k: usize) -> Mat<f32> {
    MatRef::from_column_major_slice(r, k, r.len() / k).to_owned()
}

/// Row-major `[n, d]` buffer as an `n x d` faer matrix.
///
/// ### Params
///
/// * `z` - Row-major data
/// * `d` - Row length
///
/// ### Returns
///
/// The matrix
pub(crate) fn row_major_to_mat(z: &[f32], d: usize) -> Mat<f32> {
    MatRef::from_row_major_slice(z, z.len() / d, d).to_owned()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_batch_correction::harmony::create_batch_infos;
    use approx::assert_relative_eq;
    use rand::Rng;

    /// Random unit-norm rows.
    fn unit_rows(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let raw: Vec<f32> = (0..n * d).map(|_| rng.random::<f32>() - 0.5).collect();
        let mut out = vec![0.0; n * d];
        normalise_rows_into(&raw, d, &mut out);
        out
    }

    /// Random soft assignments, each cell summing to 1.
    fn random_r(n: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut r: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() + 0.01).collect();
        for row in r.chunks_exact_mut(k) {
            let s: f32 = row.iter().sum();
            row.iter_mut().for_each(|x| *x /= s);
        }
        r
    }

    /// Base assignments sum to one, their logs match, and `shift` recovers the
    /// cosine distance.
    #[test]
    fn test_distances_and_base_consistent() {
        let (n, d, k) = (37, 5, 4);
        let z = unit_rows(n, d, 1);
        let y = unit_rows(k, d, 2);
        let sigma = vec![0.1, 0.2, 0.3, 0.05];
        let (mut lb, mut b, mut sh) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        distances_and_base(&z, &y, &sigma, d, &mut lb, &mut b, &mut sh);
        for c in 0..n {
            let row = &b[c * k..(c + 1) * k];
            assert_relative_eq!(row.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
            for kk in 0..k {
                assert_relative_eq!(lb[c * k + kk].exp(), row[kk], epsilon = 1e-5);
                let dot: f32 = (0..d).map(|f| z[c * d + f] * y[kk * d + f]).sum();
                let dist = -sigma[kk] * (lb[c * k + kk] + sh[c]);
                assert_relative_eq!(dist, 2.0 * (1.0 - dot), epsilon = 1e-4);
            }
        }
    }

    /// A very small sigma does not underflow a cell to an all-zero row.
    #[test]
    fn test_distances_and_base_small_sigma() {
        let (n, d, k) = (8, 3, 3);
        let z = unit_rows(n, d, 3);
        let y = unit_rows(k, d, 4);
        let sigma = vec![1e-3; k];
        let (mut lb, mut b, mut sh) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        distances_and_base(&z, &y, &sigma, d, &mut lb, &mut b, &mut sh);
        for row in b.chunks_exact(k) {
            assert_relative_eq!(row.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
        }
    }

    /// Brute-force error and entropy of `r` against `dist`.
    fn brute_error_entropy(r: &[f32], dist: &[f32], sigma: &[f32]) -> (f64, f64) {
        let k = sigma.len();
        let (mut e, mut h) = (0.0f64, 0.0f64);
        for (i, &v) in r.iter().enumerate() {
            e += (v * dist[i]) as f64;
            if v > 0.0 {
                h += (sigma[i % k] * v * v.ln()) as f64;
            }
        }
        (e, h)
    }

    /// Recover distances from the log tables.
    fn dist_from(lb: &[f32], sh: &[f32], sigma: &[f32]) -> Vec<f32> {
        let k = sigma.len();
        lb.iter()
            .enumerate()
            .map(|(i, l)| -sigma[i % k] * (l + sh[i / k]))
            .collect()
    }

    /// The log-table error and entropy agree with a direct computation.
    #[test]
    fn test_base_error_entropy_matches_brute_force() {
        let (n, d, k) = (50, 6, 5);
        let z = unit_rows(n, d, 5);
        let y = unit_rows(k, d, 6);
        let sigma = vec![0.1; k];
        let (mut lb, mut b, mut sh) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        distances_and_base(&z, &y, &sigma, d, &mut lb, &mut b, &mut sh);
        let (e, h) = base_error_and_entropy(&lb, &b, &sh, &sigma);
        let (eb, hb) = brute_error_entropy(&b, &dist_from(&lb, &sh, &sigma), &sigma);
        assert_relative_eq!(e, eb, max_relative = 1e-4);
        assert_relative_eq!(h, hb, max_relative = 1e-4);
    }

    /// After an update every cell sums to one, O and r_sum are in sync with R,
    /// every cell was touched (n not divisible into blocks), and the returned
    /// error and entropy match the final R.
    #[test]
    fn test_update_assignments_invariants() {
        for variant in [Variant::V1, Variant::V2] {
            let (n, d, k) = (203, 6, 7);
            let labels = vec![
                (0..n).map(|c| c % 3).collect::<Vec<_>>(),
                (0..n).map(|c| (c / 5) % 2).collect::<Vec<_>>(),
            ];
            let infos = create_batch_infos(&labels, n).unwrap();
            let theta: Vec<Vec<f32>> = infos.iter().map(|i| vec![2.0; i.n_levels]).collect();
            let z = unit_rows(n, d, 7);
            let y = unit_rows(k, d, 8);
            let sigma = vec![0.1; k];
            let (mut lb, mut b, mut sh) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
            distances_and_base(&z, &y, &sigma, d, &mut lb, &mut b, &mut sh);

            // start away from base so a skipped cell would show
            let mut r = random_r(n, k, 9);
            let (mut o, mut r_sum) = observed_counts(&r, &infos, k);
            let (e, h) = update_assignments(
                variant, &mut r, &b, &lb, &sh, &sigma, &theta, &infos, &mut o, &mut r_sum, 0.07, 11,
            );

            for row in r.chunks_exact(k) {
                assert_relative_eq!(row.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
            }
            let (o_ref, r_sum_ref) = observed_counts(&r, &infos, k);
            for (a, b) in o.iter().flatten().zip(o_ref.iter().flatten()) {
                assert_relative_eq!(*a, *b, epsilon = 1e-3);
            }
            for (a, b) in r_sum.iter().zip(&r_sum_ref) {
                assert_relative_eq!(*a, *b, epsilon = 1e-3);
            }
            let (eb, hb) = brute_error_entropy(&r, &dist_from(&lb, &sh, &sigma), &sigma);
            assert_relative_eq!(e, eb, max_relative = 1e-4);
            assert_relative_eq!(h, hb, max_relative = 1e-4);
        }
    }

    /// With theta zero the update returns the diversity-free assignments.
    #[test]
    fn test_update_assignments_theta_zero_is_base() {
        let (n, d, k) = (64, 4, 3);
        let labels = vec![(0..n).map(|c| c % 2).collect::<Vec<_>>()];
        let infos = create_batch_infos(&labels, n).unwrap();
        let theta = vec![vec![0.0; 2]];
        let z = unit_rows(n, d, 12);
        let y = unit_rows(k, d, 13);
        let sigma = vec![0.2; k];
        let (mut lb, mut b, mut sh) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        distances_and_base(&z, &y, &sigma, d, &mut lb, &mut b, &mut sh);
        let mut r = random_r(n, k, 14);
        let (mut o, mut r_sum) = observed_counts(&r, &infos, k);
        update_assignments(
            Variant::V2,
            &mut r,
            &b,
            &lb,
            &sh,
            &sigma,
            &theta,
            &infos,
            &mut o,
            &mut r_sum,
            0.05,
            1,
        );
        for (a, b) in r.iter().zip(&b) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    /// The closed-form cross-entropy equals the per-cell sum.
    #[test]
    fn test_objective_cross_entropy_closed_form() {
        let (n, k) = (40, 4);
        let labels = vec![(0..n).map(|c| c % 3).collect::<Vec<_>>()];
        let infos = create_batch_infos(&labels, n).unwrap();
        let theta = vec![vec![1.0, 2.0, 3.0]];
        let sigma = vec![0.1, 0.2, 0.1, 0.3];
        let r = random_r(n, k, 15);
        let (o, r_sum) = observed_counts(&r, &infos, k);
        for variant in [Variant::V1, Variant::V2] {
            let got = objective(variant, 0.0, 0.0, &o, &r_sum, &infos, &sigma, &theta, n);
            let mut want = 0.0f64;
            for c in 0..n {
                let l = infos[0].cell_to_level[c];
                for kk in 0..k {
                    let (ov, ev) = (o[0][l * k + kk], r_sum[kk] * infos[0].pr_b[l]);
                    let lr = match variant {
                        Variant::V1 => ((ov + ev) / ev).ln(),
                        Variant::V2 => ((ov + ev + 1.0) / (2.0 * ev + 1.0)).ln(),
                    };
                    want += (r[c * k + kk] * sigma[kk] * theta[0][l] * lr) as f64;
                }
            }
            assert_relative_eq!(got as f64, want * 2000.0 / n as f64, max_relative = 1e-4);
        }
    }

    /// Dense reference ridge correction: build each cluster's design over the
    /// kept columns from scratch and solve it in f64.
    fn brute_ridge(
        z: &[f32],
        r: &[f32],
        infos: &[BatchInfo],
        settings: RidgeSettings,
        k: usize,
        d: usize,
    ) -> Vec<f32> {
        let n = z.len() / d;
        let mut out: Vec<f64> = z.iter().map(|&x| x as f64).collect();
        for kk in 0..k {
            // kept (var, level) columns
            let mut cols: Vec<(usize, usize)> = Vec::new();
            for (v, info) in infos.iter().enumerate() {
                let o: Vec<f64> = (0..info.n_levels)
                    .map(|l| {
                        info.batch_indices[l]
                            .iter()
                            .map(|&c| r[c * k + kk] as f64)
                            .sum()
                    })
                    .collect();
                let passing: Vec<usize> = (0..info.n_levels)
                    .filter(|&l| match settings.prune_cutoff {
                        None => true,
                        Some(cut) => o[l] / info.batch_indices[l].len() as f64 > cut as f64,
                    })
                    .collect();
                if passing.len() > 1 || settings.prune_cutoff.is_none() {
                    cols.extend(passing.into_iter().map(|l| (v, l)));
                }
            }
            if cols.is_empty() {
                continue;
            }
            let p = 1 + cols.len();
            let phi = |c: usize, j: usize| -> f64 {
                if j == 0 {
                    1.0
                } else {
                    let (v, l) = cols[j - 1];
                    (infos[v].cell_to_level[c] == l) as u8 as f64
                }
            };
            let mut a = Mat::<f64>::zeros(p, p);
            let mut rhs = Mat::<f64>::zeros(p, d);
            for c in 0..n {
                let w = r[c * k + kk] as f64;
                for i in 0..p {
                    let pi = phi(c, i);
                    if pi == 0.0 {
                        continue;
                    }
                    for j in 0..p {
                        a[(i, j)] += w * pi * phi(c, j);
                    }
                    for f in 0..d {
                        rhs[(i, f)] += w * pi * z[c * d + f] as f64;
                    }
                }
            }
            for i in 1..p {
                a[(i, i)] += settings.lambda as f64;
            }
            let wmat = a.partial_piv_lu().solve(&rhs);
            for c in 0..n {
                let w = r[c * k + kk] as f64;
                for j in 1..p {
                    let pj = phi(c, j);
                    if pj == 0.0 {
                        continue;
                    }
                    for f in 0..d {
                        out[c * d + f] -= w * wmat[(j, f)];
                    }
                }
            }
        }
        out.into_iter().map(|x| x as f32).collect()
    }

    /// One-pass ridge correction matches the dense reference, single variable,
    /// with every level kept (v1) and with pruning (v2).
    #[test]
    fn test_ridge_single_variable_matches_dense() {
        let (n, d, k) = (90, 4, 3);
        let labels = vec![(0..n).map(|c| c % 3).collect::<Vec<_>>()];
        let infos = create_batch_infos(&labels, n).unwrap();
        let mut rng = StdRng::seed_from_u64(16);
        let z: Vec<f32> = (0..n * d)
            .map(|_| rng.random::<f32>() * 4.0 - 2.0)
            .collect();
        let mut r = random_r(n, k, 17);
        // cluster 2 has essentially no cells of level 0, so it is pruned there
        for c in infos[0].batch_indices[0].iter() {
            r[c * k + 2] = 0.0;
        }
        for settings in [
            RidgeSettings {
                lambda: 1.0,
                alpha: 0.0,
                dynamic_lambda: false,
                prune_cutoff: None,
            },
            RidgeSettings {
                lambda: 1.0,
                alpha: 0.0,
                dynamic_lambda: false,
                prune_cutoff: Some(1e-5),
            },
        ] {
            let mut got = vec![0.0; n * d];
            ridge_correction(&z, &r, &infos, settings, k, d, &mut got);
            let want = brute_ridge(&z, &r, &infos, settings, k, d);
            for (a, b) in got.iter().zip(&want) {
                assert_relative_eq!(*a, *b, epsilon = 1e-4);
            }
        }
    }

    /// One-pass ridge correction matches the dense reference with two crossed
    /// variables, which exercises the overlap counts and the Cholesky path.
    #[test]
    fn test_ridge_two_variables_matches_dense() {
        let (n, d, k) = (120, 3, 4);
        let labels = vec![
            (0..n).map(|c| c % 4).collect::<Vec<_>>(),
            (0..n).map(|c| (c / 7) % 3).collect::<Vec<_>>(),
        ];
        let infos = create_batch_infos(&labels, n).unwrap();
        let mut rng = StdRng::seed_from_u64(18);
        let z: Vec<f32> = (0..n * d)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let r = random_r(n, k, 19);
        for prune in [None, Some(1e-5)] {
            let settings = RidgeSettings {
                lambda: 0.5,
                alpha: 0.0,
                dynamic_lambda: false,
                prune_cutoff: prune,
            };
            let mut got = vec![0.0; n * d];
            ridge_correction(&z, &r, &infos, settings, k, d, &mut got);
            let want = brute_ridge(&z, &r, &infos, settings, k, d);
            for (a, b) in got.iter().zip(&want) {
                assert_relative_eq!(*a, *b, epsilon = 1e-4);
            }
        }
    }

    /// The f64 arrowhead solve agrees with Cholesky on an arrowhead system.
    #[test]
    fn test_arrowhead_f64_matches_cholesky() {
        let p = 5;
        let mut a = Mat::<f64>::zeros(p, p);
        a[(0, 0)] = 20.0;
        for i in 1..p {
            a[(0, i)] = 1.0 + i as f64;
            a[(i, 0)] = a[(0, i)];
            a[(i, i)] = 6.0 + i as f64;
        }
        let rhs = Mat::<f64>::from_fn(p, 3, |i, j| (i * 3 + j) as f64 * 0.3 - 1.0);
        let w1 = solve_arrowhead_f64(a.as_ref(), rhs.as_ref()).unwrap();
        let w2 = solve_spd_f64(a.as_ref(), rhs.as_ref());
        for i in 0..p {
            for j in 0..3 {
                assert_relative_eq!(w1[(i, j)], w2[(i, j)], epsilon = 1e-10);
            }
        }
    }

    /// The window test compares windows shifted by one, and v1 counts an
    /// objective increase as converged while v2 does not.
    #[test]
    fn test_window_converged_semantics() {
        assert!(!window_converged(&[1.0, 2.0, 3.0], 3, 1e-3, true));
        let rising = [100.0, 100.0, 100.0, 120.0];
        assert!(window_converged(&rising, 3, 1e-3, false));
        assert!(!window_converged(&rising, 3, 1e-3, true));
    }
}
