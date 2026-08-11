//! Gene expression trends along a Palantir trajectory.
//!
//! Two pieces. [select_branch_cells] turns fate probabilities into per-branch
//! cell sets, and [compute_gene_trends] fits a smooth curve per branch with the
//! landmark Gaussian process in [crate::ml::gp].
//!
//! Gene trends take whatever expression matrix the caller hands over. Raw,
//! log-normalised or MAGIC-imputed is their decision; nothing here imputes, and
//! this module deliberately does not depend on
//! [crate::single_cell::sc_processing::magic].
//!
//! ### What the defaults do
//!
//! Palantir's pseudotime is min-max scaled to `[0, 1]`, and the reference fits
//! with a Matérn-5/2 length scale of `1.0` and a noise standard deviation of
//! `1.0`. The length scale therefore spans the entire domain and the noise sits
//! at roughly the signal scale of log-normalised expression, so the posterior
//! is prior-dominated: it will flatten genuine transient structure and resolve
//! almost any gene into a smooth monotone or single-peaked curve. That is a
//! presentation choice, not an inference. Anyone asking whether a bump is real
//! needs to shorten `length_scale` and look again.
//!
//! ### References
//!
//! Setty, et al., Nat. Biotechnol., 2019.
//! Otto, Zhou, et al., bioRxiv, 2023 (mellon, the estimator the reference uses).

use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::core::math::matrix_helpers::subset_rows;
use crate::core::math::vector_helpers::quantile_sorted;
use crate::ml::gp::{LandmarkGpParams, fit_landmark_gp};
use crate::prelude::*;
use crate::single_cell::sc_trajectory::palantir::PalantirResult;

////////////
// Consts //
////////////

/// Grid points per branch, and the bucket count for the branch selection.
///
/// `PSEUDOTIME_RES` in the reference (`palantir/presults.py`).
pub const PSEUDOTIME_RES: usize = 500;

/// Fewest cells a branch can be fitted from.
///
/// The reference has no such guard and will happily fit a 500-point grid to a
/// single cell, which returns the prior dressed up as a trend.
const GENE_TRENDS_MIN_CELLS: usize = 3;

/// Fewest grid points a trend can be evaluated on.
///
/// The grid doubles as the Gaussian process landmark set, which needs at least
/// two distinct points. Rejecting anything below this rather than clamping it
/// keeps [GeneTrendsResult::trends] exactly `resolution` rows tall, which the
/// caller is entitled to rely on.
const GENE_TRENDS_MIN_RESOLUTION: usize = 2;

//////////////////////
// Branch selection //
//////////////////////

/// Parameters for [select_branch_cells].
#[derive(Clone, Copy, Debug)]
pub struct BranchSelectionParams {
    /// Upper-tail quantile of the fate probability. Reference: `1e-2`.
    pub q: f64,
    /// Slack subtracted from the threshold before comparing. Reference: `1e-2`.
    pub eps: f64,
    /// Buckets the pseudotime axis is split into, capped at the cell count.
    /// Reference: [PSEUDOTIME_RES].
    pub resolution: usize,
}

impl BranchSelectionParams {
    /// Construct the parameter set.
    ///
    /// ### Params
    ///
    /// * `q` - Upper-tail quantile of the fate probability.
    /// * `eps` - Slack subtracted from the threshold.
    /// * `resolution` - Buckets along the pseudotime axis.
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(q: f64, eps: f64, resolution: usize) -> Self {
        Self { q, eps, resolution }
    }
}

impl Default for BranchSelectionParams {
    fn default() -> Self {
        Self {
            q: 1e-2,
            eps: 1e-2,
            resolution: PSEUDOTIME_RES,
        }
    }
}

/// Derive per-branch cell sets from fate probabilities.
///
/// The threshold schedule is an **expanding prefix**, not a sliding window, and
/// getting that wrong is the easiest way to produce plausible nonsense. Cells
/// are sorted by pseudotime; for bucket `i` spanning `[l, r)` the threshold is
/// the `1 - q` quantile of *every* cell up to `r`, including the bucket itself,
/// and it is written back onto rows `l..r`. Any leftover tail takes the
/// quantile over all cells. The result is then made monotone non-decreasing
/// down the pseudotime axis, so a fate's bar can only rise, and a cell joins a
/// branch when its probability exceeds `threshold - eps`.
///
/// Deviations from `palantir/presults.py:597-638`, all deliberate:
///
/// * Pseudotime ties break on ascending cell index, because the sort here is
///   stable where `np.argsort` defaults to unstable quicksort. It only shifts
///   tied cells between adjacent buckets.
/// * Non-finite pseudotime and a length disagreement are errors; the reference
///   checks neither.
///
/// `PalantirResult::branch_probs` is thresholded without renormalisation, so
/// its rows need not sum to one. That is fine here: the quantile is taken per
/// fate, never across them.
///
/// ### Params
///
/// * `pseudotime` - Pseudotime per cell.
/// * `branch_probs` - Fate probabilities, cells by fates. `NaN` is replaced by
///   `1 / n_fates`, matching the reference.
/// * `params` - Selection parameters.
///
/// ### Returns
///
/// One cell-index vector per fate, ascending, with the column order of
/// `branch_probs` and therefore of `PalantirResult::terminal_states`.
pub fn select_branch_cells(
    pseudotime: &[f32],
    branch_probs: MatRef<'_, f32>,
    params: &BranchSelectionParams,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let n = pseudotime.len();
    let n_fates = branch_probs.ncols();

    if n == 0 || n_fates == 0 {
        return Err(BixverseErrors::GpEmptyInput {
            name: "branch selection input",
        });
    }
    if branch_probs.nrows() != n {
        return Err(BixverseErrors::GeneTrendsShapeMismatch {
            n_cells: branch_probs.nrows(),
            n_pseudotime: n,
        });
    }
    if pseudotime.iter().any(|v| !v.is_finite()) {
        return Err(BixverseErrors::GpNonFiniteInput { name: "pseudotime" });
    }

    // Pseudotime order, ties on ascending cell index.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pseudotime[a].total_cmp(&pseudotime[b]));

    let resolution = params.resolution.clamp(1, n);
    let step = (n / resolution).max(1);
    let n_steps = n / step;
    let uniform = 1.0 / n_fates as f64;
    let upper = 1.0 - params.q;

    let masks: Vec<Vec<usize>> = (0..n_fates)
        .into_par_iter()
        .map(|fate| {
            let sorted_probs: Vec<f64> = order
                .iter()
                .map(|&cell| {
                    let v = branch_probs[(cell, fate)] as f64;
                    if v.is_nan() { uniform } else { v }
                })
                .collect();

            // The prefix is kept sorted by merging each bucket in, which is
            // linear per bucket. Re-sorting or re-partitioning the whole prefix
            // at every one of the `n_steps` checkpoints would not be.
            let mut prefix: Vec<f64> = Vec::with_capacity(n);
            let mut merged: Vec<f64> = Vec::with_capacity(n);
            let mut bucket: Vec<f64> = Vec::with_capacity(step);
            let mut thresholds = vec![0.0_f64; n];

            let mut r = 0usize;
            for i in 0..n_steps {
                let (l, hi) = (i * step, (i + 1) * step);
                r = hi;

                bucket.clear();
                bucket.extend_from_slice(&sorted_probs[l..hi]);
                bucket.sort_by(|a, b| a.total_cmp(b));

                merge_sorted(&prefix, &bucket, &mut merged);
                std::mem::swap(&mut prefix, &mut merged);

                let threshold = quantile_sorted(&prefix, upper);
                thresholds[l..hi].fill(threshold);
            }

            // The `n % step` leftover takes the quantile over every cell.
            if r < n {
                bucket.clear();
                bucket.extend_from_slice(&sorted_probs[r..]);
                bucket.sort_by(|a, b| a.total_cmp(b));
                merge_sorted(&prefix, &bucket, &mut merged);
                thresholds[r..].fill(quantile_sorted(&merged, upper));
            }

            // Monotone non-decreasing down the pseudotime axis.
            let mut running = f64::NEG_INFINITY;
            for t in thresholds.iter_mut() {
                running = running.max(*t);
                *t = running;
            }

            let mut cells: Vec<usize> = (0..n)
                .filter(|&pos| thresholds[pos] - params.eps < sorted_probs[pos])
                .map(|pos| order[pos])
                .collect();
            cells.sort_unstable();
            cells
        })
        .collect();

    Ok(masks)
}

/////////////////
// Gene trends //
/////////////////

/// How a branch's cells enter its fit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BranchWeighting {
    /// Every selected cell counts equally. What the reference's current path
    /// does once [select_branch_cells] has thresholded the probabilities.
    #[default]
    HardMask,
    /// Selected cells are weighted by their fate probability.
    ///
    /// Closer to the legacy GAM path, and the more defensible treatment: a cell
    /// sitting at probability 0.6 is not a member of the branch. Combine it
    /// with a selection of every cell for the legacy path's *weighting*, or
    /// with [select_branch_cells] to threshold first and soften afterwards.
    ///
    /// The weighting matches `compute_gene_trends_legacy`; the estimator does
    /// not. That path fits a four-spline pyGAM, this one a Matérn-5/2 landmark
    /// GP, so the curves will not agree numerically.
    FateProbability,
}

/// Parameters for [compute_gene_trends].
#[derive(Clone, Copy, Debug)]
pub struct GeneTrendsParams {
    /// Grid points per branch. Reference: [PSEUDOTIME_RES].
    pub resolution: usize,
    /// How selected cells are weighted.
    pub weighting: BranchWeighting,
    /// Gaussian process hyperparameters, carrying the reference defaults.
    pub gp: LandmarkGpParams,
}

impl GeneTrendsParams {
    /// Construct the parameter set.
    ///
    /// ### Params
    ///
    /// * `resolution` - Grid points per branch.
    /// * `weighting` - How selected cells are weighted.
    /// * `gp` - Gaussian process hyperparameters.
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(resolution: usize, weighting: BranchWeighting, gp: LandmarkGpParams) -> Self {
        Self {
            resolution,
            weighting,
            gp,
        }
    }
}

impl Default for GeneTrendsParams {
    fn default() -> Self {
        Self {
            resolution: PSEUDOTIME_RES,
            weighting: BranchWeighting::default(),
            gp: LandmarkGpParams::default(),
        }
    }
}

/// Fitted trends, one entry per branch.
#[derive(Clone, Debug)]
pub struct GeneTrendsResult<T> {
    /// Per branch, `resolution` by `n_genes`.
    ///
    /// The reference transposes this to genes by grid for its AnnData `varm`;
    /// do that at the R boundary, not here.
    pub trends: Vec<Mat<T>>,
    /// Per branch, the pseudotime grid the trend is evaluated on, spanning that
    /// branch's own minimum to maximum.
    pub grids: Vec<Vec<f64>>,
    /// Cells that went into each branch's fit.
    pub n_cells: Vec<usize>,
    /// Jitter each branch's landmark Cholesky settled on. Above the requested
    /// value means the grid Gram was singular there and the curve is flatter
    /// than the hyperparameters asked for.
    pub jitter_used: Vec<f64>,
}

/// Fit a smooth expression trend per branch.
///
/// Branches are fitted sequentially: there are only a handful of them, each fit
/// already saturates the machine through the Gaussian process's own
/// parallelism, and going one at a time keeps peak memory at a single branch's
/// expression submatrix.
///
/// The grid stays at `resolution` points even when a branch has fewer cells
/// than that, matching the reference. That is structurally fine, since the
/// landmark Cholesky depends only on the grid and the `+ I` keeps the posterior
/// factorisation conditioned; the curve simply reverts toward the prior. Note
/// the reference's own asymmetry here: [select_branch_cells] *does* clamp its
/// bucket count to the cell count.
///
/// ### Params
///
/// * `expression` - Expression matrix, all cells by genes, aligned with
///   `pseudotime`. Raw, log-normalised or imputed, the caller's choice.
/// * `pseudotime` - Pseudotime per cell.
/// * `branch_cells` - Cell indices per branch, e.g. from [select_branch_cells].
/// * `branch_probs` - Fate probabilities, cells by branches. Required only for
///   [BranchWeighting::FateProbability].
/// * `params` - Fit parameters.
///
/// ### Returns
///
/// The per-branch trends, or the matching [BixverseErrors] variant if the
/// inputs disagree in shape, a branch is empty or too small, or a branch spans
/// zero pseudotime.
pub fn compute_gene_trends<T>(
    expression: MatRef<'_, T>,
    pseudotime: &[f32],
    branch_cells: &[Vec<usize>],
    branch_probs: Option<MatRef<'_, f32>>,
    params: &GeneTrendsParams,
) -> Result<GeneTrendsResult<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let n = pseudotime.len();

    if expression.nrows() != n {
        return Err(BixverseErrors::GeneTrendsShapeMismatch {
            n_cells: expression.nrows(),
            n_pseudotime: n,
        });
    }
    if pseudotime.iter().any(|v| !v.is_finite()) {
        return Err(BixverseErrors::GpNonFiniteInput { name: "pseudotime" });
    }
    if params.resolution < GENE_TRENDS_MIN_RESOLUTION {
        return Err(BixverseErrors::GeneTrendsResolutionTooLow {
            resolution: params.resolution,
            minimum: GENE_TRENDS_MIN_RESOLUTION,
        });
    }
    if params.weighting == BranchWeighting::FateProbability {
        // Each failure gets its own variant: "you did not supply the
        // probabilities" and "the ones you supplied are the wrong shape" send
        // the caller to entirely different places.
        let Some(probs) = branch_probs else {
            return Err(BixverseErrors::GeneTrendsMissingFateProbabilities);
        };
        if probs.nrows() != n {
            return Err(BixverseErrors::GeneTrendsFateShapeMismatch {
                n_rows: probs.nrows(),
                n_cells: n,
            });
        }
        // Equality, not `>=`. Branch `i` takes column `i` positionally, so a
        // wider matrix would silently weight every cell by the wrong fate.
        if probs.ncols() != branch_cells.len() {
            return Err(BixverseErrors::GeneTrendsFateColumnMismatch {
                n_columns: probs.ncols(),
                n_branches: branch_cells.len(),
            });
        }
    }

    let mut trends = Vec::with_capacity(branch_cells.len());
    let mut grids = Vec::with_capacity(branch_cells.len());
    let mut n_cells = Vec::with_capacity(branch_cells.len());
    let mut jitter_used = Vec::with_capacity(branch_cells.len());

    for (branch, cells) in branch_cells.iter().enumerate() {
        if cells.is_empty() {
            return Err(BixverseErrors::GeneTrendsBranchEmpty { branch });
        }
        if cells.len() < GENE_TRENDS_MIN_CELLS {
            return Err(BixverseErrors::GeneTrendsBranchTooFewCells {
                branch,
                n_cells: cells.len(),
                minimum: GENE_TRENDS_MIN_CELLS,
            });
        }
        if let Some(&bad) = cells.iter().find(|&&c| c >= n) {
            return Err(BixverseErrors::GeneTrendsCellOutOfRange {
                branch,
                cell: bad,
                n_cells: n,
            });
        }

        let pt: Vec<f64> = cells.iter().map(|&c| pseudotime[c] as f64).collect();
        let (lo, hi) = pt
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        // Both are finite: pseudotime was checked above and the branch is
        // non-empty, so neither fold identity survives.
        if hi <= lo {
            return Err(BixverseErrors::GeneTrendsDegeneratePseudotime { branch });
        }

        let grid = linspace(lo, hi, params.resolution);
        let sub = subset_rows(expression, cells);

        let weights: Option<Vec<f64>> = match params.weighting {
            BranchWeighting::HardMask => None,
            BranchWeighting::FateProbability => {
                let probs = branch_probs.expect("checked above");
                let w: Vec<f64> = cells
                    .iter()
                    .map(|&c| {
                        let v = probs[(c, branch)] as f64;
                        if v.is_finite() { v.max(0.0) } else { 0.0 }
                    })
                    .collect();
                // The GP drops zero-weight observations, so an all-zero branch
                // would come back as the flat prior rather than as a failure.
                if w.iter().all(|v| *v == 0.0) {
                    return Err(BixverseErrors::GeneTrendsZeroWeightBranch { branch });
                }
                Some(w)
            }
        };

        let fit = fit_landmark_gp(&pt, sub.as_ref(), &grid, weights.as_deref(), &params.gp)?;

        trends.push(fit.predict_on_landmarks::<T>());
        jitter_used.push(fit.jitter_used);
        grids.push(grid);
        n_cells.push(cells.len());
    }

    Ok(GeneTrendsResult {
        trends,
        grids,
        n_cells,
        jitter_used,
    })
}

/// Select branch cells from a finished Palantir run and fit their trends.
///
/// ### Params
///
/// * `expression` - Expression matrix, all cells by genes.
/// * `palantir` - A finished Palantir run.
/// * `selection` - Branch selection parameters.
/// * `params` - Fit parameters.
///
/// ### Returns
///
/// The per-branch trends, with the branch order of
/// `PalantirResult::terminal_states`.
pub fn compute_gene_trends_palantir<T>(
    expression: MatRef<'_, T>,
    palantir: &PalantirResult,
    selection: &BranchSelectionParams,
    params: &GeneTrendsParams,
) -> Result<GeneTrendsResult<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let branch_cells = select_branch_cells(
        &palantir.pseudotime,
        palantir.branch_probs.as_ref(),
        selection,
    )?;

    compute_gene_trends(
        expression,
        &palantir.pseudotime,
        &branch_cells,
        Some(palantir.branch_probs.as_ref()),
        params,
    )
}

/////////////
// Helpers //
/////////////

/// Merge two sorted slices into `out`, which is cleared first.
///
/// ### Params
///
/// * `a` - First sorted slice.
/// * `b` - Second sorted slice.
/// * `out` - Destination, overwritten with the merged sequence.
fn merge_sorted(a: &[f64], b: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.reserve(a.len() + b.len());

    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// Evenly spaced points across `[lo, hi]`, both endpoints pinned exactly.
///
/// Pinning the last point matters: accumulating `lo + step * i` leaves the end
/// short by a rounding error, and the grid doubles as the landmark set, so a
/// drifting endpoint would put a landmark outside the data.
///
/// ### Params
///
/// * `lo` - First value.
/// * `hi` - Last value.
/// * `n` - Number of points, at least one.
///
/// ### Returns
///
/// The grid.
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let step = (hi - lo) / (n - 1) as f64;
    let mut out: Vec<f64> = (0..n).map(|i| lo + step * i as f64).collect();
    out[n - 1] = hi;
    out
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Pinned against `palantir.presults.select_branch_cells` (Palantir 1.4.5).
    ///
    /// `n = 12`, 2 fates, pseudotime and fate 0 both rising as `i / 11` with
    /// fate 1 as the mirror. Both parameter sets came out of a run of the
    /// reference against this exact fixture rather than out of a second
    /// implementation of the schedule here, which is the point: a test that
    /// re-derives the thresholds shares any misreading with the code it checks.
    #[test]
    fn test_select_branch_cells_matches_reference() {
        let n = 12usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let probs = Mat::<f32>::from_fn(n, 2, |i, j| {
            let t = i as f32 / (n - 1) as f32;
            if j == 0 { t } else { 1.0 - t }
        });

        // Reference at q = 0.25, eps = 0. Fate 1 decays fast enough that the
        // ratcheting threshold cuts every one of its cells.
        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams {
                q: 0.25,
                eps: 0.0,
                resolution: PSEUDOTIME_RES,
            },
        )
        .unwrap();
        assert_eq!(masks[0], (1..12).collect::<Vec<usize>>());
        assert_eq!(masks[1], Vec::<usize>::new());

        // Reference at the library defaults. The `eps` slack pulls the whole of
        // fate 0 back in and leaves fate 1 with only its first cell.
        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams::default(),
        )
        .unwrap();
        assert_eq!(masks[0], (0..12).collect::<Vec<usize>>());
        assert_eq!(masks[1], vec![0]);
    }

    /// The cumulative maximum, pinned so that deleting it fails.
    ///
    /// Every other fixture in this module gives an identical answer with and
    /// without the running maximum, because their thresholds happen to be
    /// nondecreasing already. This one is built the other way round: a lone
    /// `0.95` at the earliest pseudotime makes the size-one first prefix return
    /// `0.95` exactly, and the accumulate then floors every later threshold at
    /// it. Without the accumulate the threshold decays as the prefix fills with
    /// `0.10`s, and the late `0.939` slips through.
    ///
    /// Reference answers at the library defaults: `[0]` with the running
    /// maximum, `[0, 19]` without. Margins are around `1e-3` either way, far
    /// above `f32` noise.
    #[test]
    fn test_select_branch_cells_cumulative_maximum_is_load_bearing() {
        let n = 20usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let probs = Mat::<f32>::from_fn(n, 1, |i, _| match i {
            0 => 0.95,
            19 => 0.939,
            _ => 0.10,
        });

        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams::default(),
        )
        .unwrap();

        assert_eq!(
            masks[0],
            vec![0],
            "cell 19 was kept, which is what a missing cumulative maximum looks like"
        );
    }

    /// The `step > 1` regime, pinned against the reference.
    ///
    /// `n = 11` at `resolution = 5` gives `step = 2`, `n_steps = 5` and a
    /// one-row tail. Thresholds are block-constant across each pair, which is
    /// what strips the even indices out; the doc on
    /// `test_select_branch_cells_threshold_is_monotone_at_unit_step` explains
    /// why that does not happen when the bucket size is one.
    ///
    /// Reference thresholds, for anyone re-deriving this:
    /// `[0.099, 0.099, 0.297, 0.297, 0.495, 0.495, 0.693, 0.693, 0.891, 0.891,
    /// 0.99]`.
    #[test]
    fn test_select_branch_cells_step_above_one() {
        let n = 11usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let probs = Mat::<f32>::from_fn(n, 1, |i, _| i as f32 / (n - 1) as f32);

        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams {
                resolution: 5,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(masks[0], vec![1, 3, 5, 7, 9, 10]);
    }

    /// The production regime, pinned against the reference.
    ///
    /// `n = 1001` at the default resolution gives `step = 2`, 500 buckets and a
    /// one-row tail, which is the arithmetic every real dataset above a thousand
    /// cells goes through and which no other fixture here reaches. The signal
    /// oscillates so the selection breaks into three disjoint runs rather than a
    /// single prefix.
    #[test]
    fn test_select_branch_cells_production_regime() {
        let n = 1001usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let probs = Mat::<f32>::from_fn(n, 1, |i, _| 0.5 + 0.4 * (i as f64 * 0.02).sin() as f32);

        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams::default(),
        )
        .unwrap();

        let want: Vec<usize> = (0..=89).chain(382..=403).chain(696..=718).collect();
        assert_eq!(masks[0].len(), 135);
        assert_eq!(masks[0], want);
    }

    /// The rising threshold is what cuts a decaying fate loose, and the shape
    /// of that asymmetry is worth pinning down because it surprises people.
    ///
    /// **When the bucket size is one**, a fate whose probability only ever
    /// rises keeps every cell: the prefix quantile tracks the prefix maximum,
    /// which is the current cell, so the comparison reduces to `p - eps < p`.
    /// Only a fate that falls behind its own running maximum loses anything.
    ///
    /// That qualifier matters. `step = n / min(500, n)` is one only below 1000
    /// cells, so on any real dataset the bucket spans several cells, the
    /// threshold is taken once per bucket, and a rising fate does lose its
    /// within-bucket stragglers. `test_select_branch_cells_step_above_one`
    /// covers that regime.
    #[test]
    fn test_select_branch_cells_threshold_is_monotone_at_unit_step() {
        let n = 60usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        // Fate 0 climbs, fate 1 falls away entirely.
        let probs = Mat::<f32>::from_fn(n, 2, |i, j| {
            let t = i as f32 / (n - 1) as f32;
            if j == 0 { t } else { 1.0 - t }
        });

        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams::default(),
        )
        .unwrap();

        // The rising fate keeps everything.
        assert_eq!(masks[0].len(), n);

        // The falling one keeps its early cells and drops its late ones.
        assert!(masks[1].contains(&0));
        assert!(!masks[1].contains(&(n - 1)));
        // And the cut is a single crossing, since the threshold only rises.
        let last_kept = *masks[1].last().unwrap();
        assert_eq!(masks[1], (0..=last_kept).collect::<Vec<usize>>());
    }

    /// Pseudotime order, not storage order, is what the schedule follows.
    ///
    /// Pseudotime runs backwards against the cell index here and the fate
    /// probability runs with it, so in *pseudotime* order the probability
    /// decays and the late cells get cut. Read in storage order the same
    /// probabilities are ascending, which would keep everything, so the two
    /// readings give visibly different answers.
    #[test]
    fn test_select_branch_cells_respects_pseudotime_order() {
        let n = 30usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| (n - 1 - i) as f32).collect();
        let probs = Mat::<f32>::from_fn(n, 1, |i, _| i as f32 / (n - 1) as f32);

        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams::default(),
        )
        .unwrap();

        // Cell n-1 sits earliest in pseudotime and carries the highest
        // probability, so it survives; cell 0 is latest and lowest.
        assert!(masks[0].contains(&(n - 1)));
        assert!(!masks[0].contains(&0));
        assert!(
            masks[0].len() < n,
            "storage order would have kept everything"
        );
        // Output must be ascending cell index, not pseudotime order.
        assert!(masks[0].windows(2).all(|w| w[0] < w[1]));
    }

    /// `NaN` probabilities take the uniform value rather than poisoning the
    /// quantile, matching the reference.
    #[test]
    fn test_select_branch_cells_fills_nan_probabilities() {
        let n = 12usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32).collect();
        // Only fate 0 carries the NaN, so the `1 / n_fates` value is
        // observable rather than cancelling out across both columns.
        let probs = Mat::<f32>::from_fn(n, 2, |i, j| match (i, j) {
            (5, 0) => f32::NAN,
            (_, 0) => 0.1,
            _ => 0.9,
        });

        let masks = select_branch_cells(
            &pseudotime,
            probs.as_ref(),
            &BranchSelectionParams::default(),
        )
        .unwrap();
        assert_eq!(masks.len(), 2);

        // The single filled 0.5 drags the running threshold up as it enters the
        // prefix, so the filled cell is kept and the flat 0.1 tail behind it is
        // cut. Filling with 0.0 instead inverts this exactly: the cell is
        // dropped and its successors kept. Dropping the handling altogether
        // gives a NaN threshold. All three are distinguishable here.
        assert_eq!(masks[0], vec![0, 1, 2, 3, 4, 5]);

        // Fate 1 is flat and finite, so its threshold never ratchets and
        // nothing is cut.
        assert_eq!(masks[1].len(), n);
    }

    #[test]
    fn test_select_branch_cells_rejects_bad_inputs() {
        let probs = Mat::<f32>::from_fn(4, 2, |_, _| 0.5);
        let params = BranchSelectionParams::default();

        assert!(matches!(
            select_branch_cells(&[], probs.as_ref(), &params),
            Err(BixverseErrors::GpEmptyInput { .. })
        ));
        assert!(matches!(
            select_branch_cells(&[0.0, 1.0], probs.as_ref(), &params),
            Err(BixverseErrors::GeneTrendsShapeMismatch { .. })
        ));
        assert!(matches!(
            select_branch_cells(&[0.0, 1.0, f32::NAN, 3.0], probs.as_ref(), &params),
            Err(BixverseErrors::GpNonFiniteInput { .. })
        ));
    }

    #[test]
    fn test_merge_sorted() {
        let mut out = Vec::new();
        merge_sorted(&[1.0, 3.0, 5.0], &[2.0, 4.0], &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        merge_sorted(&[], &[2.0, 4.0], &mut out);
        assert_eq!(out, vec![2.0, 4.0]);

        merge_sorted(&[1.0], &[], &mut out);
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn test_linspace_pins_both_endpoints() {
        let g = linspace(0.3, 2.7, 5);
        assert_eq!(g.len(), 5);
        assert_relative_eq!(g[0], 0.3, epsilon = 1e-15);
        assert_relative_eq!(g[4], 2.7, epsilon = 1e-15);
        assert!(g.windows(2).all(|w| w[0] < w[1]));
    }

    ///////////////////
    // Trend fitting //
    ///////////////////

    /// A gene peaking mid-trajectory must give a trend whose maximum lands in
    /// the right stretch of pseudotime.
    #[test]
    fn test_gene_trends_recovers_a_peak() {
        let n = 300usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();

        // Gene 0 peaks at t = 0.3, gene 1 rises monotonically.
        let expression = Mat::<f32>::from_fn(n, 2, |i, j| {
            let t = pseudotime[i] as f64;
            if j == 0 {
                (-((t - 0.3) / 0.1).powi(2)).exp() as f32
            } else {
                t as f32
            }
        });

        let cells: Vec<Vec<usize>> = vec![(0..n).collect()];
        let params = GeneTrendsParams {
            resolution: 100,
            gp: LandmarkGpParams {
                length_scale: 0.08,
                sigma: 0.05,
                ..Default::default()
            },
            ..Default::default()
        };

        let res =
            compute_gene_trends(expression.as_ref(), &pseudotime, &cells, None, &params).unwrap();

        assert_eq!(res.trends.len(), 1);
        assert_eq!(res.trends[0].nrows(), 100);
        assert_eq!(res.trends[0].ncols(), 2);
        assert_eq!(res.n_cells, vec![n]);

        // Peak of gene 0.
        let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
        for i in 0..100 {
            if res.trends[0][(i, 0)] > best_v {
                best_v = res.trends[0][(i, 0)];
                best = i;
            }
        }
        let peak_t = res.grids[0][best];
        assert!(
            (peak_t - 0.3).abs() < 0.06,
            "peak landed at t={}, expected near 0.3",
            peak_t
        );

        // Gene 1 must still be monotone increasing over the grid.
        for i in 1..100 {
            assert!(
                res.trends[0][(i, 1)] >= res.trends[0][(i - 1, 1)] - 1e-3,
                "monotone gene dipped at grid point {}",
                i
            );
        }
    }

    /// Two branches must be fitted independently, so a gene switched on in only
    /// one of them must not bleed into the other.
    #[test]
    fn test_gene_trends_separates_branches() {
        let n = 200usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        // First half is branch 0, second half branch 1, with an overlap.
        let branch_a: Vec<usize> = (0..130).collect();
        let branch_b: Vec<usize> = (70..n).collect();

        // Gene 0 is on only in the cells unique to branch b.
        let expression = Mat::<f32>::from_fn(n, 1, |i, _| if i >= 130 { 1.0 } else { 0.0 });

        let params = GeneTrendsParams {
            resolution: 50,
            gp: LandmarkGpParams {
                length_scale: 0.1,
                sigma: 0.05,
                ..Default::default()
            },
            ..Default::default()
        };

        let res = compute_gene_trends(
            expression.as_ref(),
            &pseudotime,
            &[branch_a, branch_b],
            None,
            &params,
        )
        .unwrap();

        // Branch 0 never sees an expressing cell, so its trend stays near zero.
        let max_a = (0..50).fold(f32::NEG_INFINITY, |m, i| m.max(res.trends[0][(i, 0)]));
        // Branch 1 ends high.
        let end_b = res.trends[1][(49, 0)];

        assert!(
            max_a < 0.2,
            "branch 0 picked up signal it never saw: {max_a}"
        );
        assert!(end_b > 0.5, "branch 1 lost its signal: {end_b}");
    }

    /// Fate-probability weighting must actually change the fit, and must demand
    /// the probabilities it needs.
    #[test]
    fn test_gene_trends_fate_probability_weighting() {
        let n = 120usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let expression = Mat::<f32>::from_fn(n, 1, |i, _| if i >= 60 { 1.0 } else { 0.0 });
        // Weight collapses over the second half, so the late cells barely count.
        let probs = Mat::<f32>::from_fn(n, 1, |i, _| if i >= 60 { 0.01 } else { 1.0 });
        let cells: Vec<Vec<usize>> = vec![(0..n).collect()];

        let base = GeneTrendsParams {
            resolution: 40,
            gp: LandmarkGpParams {
                length_scale: 0.1,
                sigma: 0.05,
                ..Default::default()
            },
            ..Default::default()
        };

        let hard =
            compute_gene_trends(expression.as_ref(), &pseudotime, &cells, None, &base).unwrap();
        let weighted = compute_gene_trends(
            expression.as_ref(),
            &pseudotime,
            &cells,
            Some(probs.as_ref()),
            &GeneTrendsParams {
                weighting: BranchWeighting::FateProbability,
                ..base
            },
        )
        .unwrap();

        // Down-weighting the expressing cells has to pull the end of the curve
        // down relative to the unweighted fit.
        assert!(
            weighted.trends[0][(39, 0)] < hard.trends[0][(39, 0)],
            "weighting did not change the fit"
        );

        // And it must refuse to run without the probabilities.
        assert!(matches!(
            compute_gene_trends(
                expression.as_ref(),
                &pseudotime,
                &cells,
                None,
                &GeneTrendsParams {
                    weighting: BranchWeighting::FateProbability,
                    ..base
                }
            ),
            Err(BixverseErrors::GeneTrendsMissingFateProbabilities)
        ));
    }

    #[test]
    fn test_gene_trends_rejects_bad_branches() {
        let n = 20usize;
        let pseudotime: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expression = Mat::<f32>::from_fn(n, 1, |i, _| i as f32);
        let params = GeneTrendsParams {
            resolution: 10,
            ..Default::default()
        };

        assert!(matches!(
            compute_gene_trends(
                expression.as_ref(),
                &pseudotime,
                &[Vec::new()],
                None,
                &params
            ),
            Err(BixverseErrors::GeneTrendsBranchEmpty { branch: 0 })
        ));

        assert!(matches!(
            compute_gene_trends(
                expression.as_ref(),
                &pseudotime,
                &[vec![0, 1]],
                None,
                &params
            ),
            Err(BixverseErrors::GeneTrendsBranchTooFewCells { branch: 0, .. })
        ));

        // Zero pseudotime span within the branch.
        let flat = vec![1.0_f32; n];
        assert!(matches!(
            compute_gene_trends(
                expression.as_ref(),
                &flat,
                &[(0..n).collect()],
                None,
                &params
            ),
            Err(BixverseErrors::GeneTrendsDegeneratePseudotime { branch: 0 })
        ));

        // Expression rows against pseudotime length.
        assert!(matches!(
            compute_gene_trends(
                expression.as_ref(),
                &pseudotime[..5],
                &[(0..5).collect()],
                None,
                &params
            ),
            Err(BixverseErrors::GeneTrendsShapeMismatch { .. })
        ));
    }
}
