//! Thresholding per-cell regulon activity into on/off calls.
//!
//! The final SCENIC step: each regulon's AUC scores across cells get one
//! threshold, and a cell is on when its score sits strictly above it. Follows
//! pySCENIC rather than AUCell. AUCell fits six candidate thresholds and then
//! lets `minimumDens` override all of them whenever a density trough exists, so
//! the extra candidates rarely change the answer.
//!
//! Two differences from pySCENIC worth knowing. The bimodality test is the
//! 2-component versus 1-component BIC comparison, not Hartigan's dip, because
//! the mixture has to be fitted anyway to bound the trough search. And the
//! initialisation is deterministic (quartiles), so there is no seed and repeat
//! runs give identical thresholds.
//!
//! Scores stay `f32` to match the AUCell output, every accumulator is `f64`.

use rayon::prelude::*;

use crate::core::math::vector_helpers::quantile_sorted;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Grid resolution for the trough search between the two component means.
/// The search interval is narrow by construction, so 512 points resolve it
/// far below the noise in the underlying kernel density estimate.
const KDE_GRID: usize = 512;

/// Number of histogram bins the scores are collapsed into before the kernel
/// density estimate. Evaluating the estimate against the raw scores is
/// `O(n_cells * n_grid)`, which does not scale past a few tens of thousands of
/// cells. Binning first makes it `O(n_cells + n_bins * n_grid)` at an error
/// well under one bin width.
const KDE_BINS: usize = 512;

/// Iteration cap for the mixture EM. The fit is one-dimensional with two
/// components, so it converges in tens of iterations; this only guards
/// pathological input.
const EM_MAX_ITER: usize = 200;

/// Relative change in the log likelihood below which the EM stops.
const EM_TOL: f64 = 1e-8;

/// Variance floor for a mixture component. Without it a component can collapse
/// onto a single repeated score and drive the likelihood to infinity.
const MIN_VARIANCE: f64 = 1e-12;

/// Multiple of the standard deviation used for the unimodal threshold, as in
/// pySCENIC's `derive_threshold`.
const UNIMODAL_SD_MULTIPLE: f64 = 2.0;

////////////
// Params //
////////////

/// Parameters for the regulon binarisation
#[derive(Clone, Copy, Debug)]
pub struct BinariseParams {
    /// Multiplier on the Silverman bandwidth of the kernel density estimate.
    /// Values above one smooth away shallow troughs, which is what AUCell does
    /// with `density(auc, adjust = 2)`.
    pub bw_adjust: f64,
    /// Number of points at which the density is evaluated between the two
    /// component means.
    pub n_grid: usize,
    /// Number of histogram bins used to approximate the density.
    pub n_bins: usize,
}

impl Default for BinariseParams {
    fn default() -> Self {
        Self {
            bw_adjust: 1.0,
            n_grid: KDE_GRID,
            n_bins: KDE_BINS,
        }
    }
}

/// Per-regulon thresholds derived from an AUC matrix
///
/// The on/off calls themselves are a strict `score > threshold` comparison,
/// left to the caller so the thresholds can be inspected or overridden first.
/// SCENIC does the same: it writes the thresholds to an editable file between
/// scoring and assignment.
#[derive(Clone, Debug)]
pub struct RegulonThresholds {
    /// One threshold per regulon, in input order.
    pub thresholds: Vec<f64>,
    /// Whether the regulon's scores were called bimodal. When false the
    /// threshold is the unimodal `mean + 2 * sd` fallback.
    pub bimodal: Vec<bool>,
}

/////////////
// Helpers //
/////////////

/// Fitted 1D two-component Gaussian mixture
///
/// Only the means and the likelihood leave the fit: the means bound the trough
/// search, the likelihood feeds the BIC comparison. The kernel density
/// estimate, not the mixture, is what actually places the threshold.
#[derive(Clone, Copy, Debug)]
struct TwoComponent {
    /// Component means.
    mu: [f64; 2],
    /// Log likelihood at convergence.
    log_lik: f64,
}

/// Gaussian density at a point
///
/// ### Params
///
/// * `x` - Where to evaluate.
/// * `mu` - Mean.
/// * `sd` - Standard deviation, assumed strictly positive.
///
/// ### Returns
///
/// The density value.
#[inline(always)]
fn gaussian_pdf(x: f64, mu: f64, sd: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    let z = (x - mu) / sd;
    INV_SQRT_2PI / sd * (-0.5 * z * z).exp()
}

/// Bayesian information criterion, lower is better
///
/// ### Params
///
/// * `log_lik` - Log likelihood of the fit.
/// * `n_params` - Number of free parameters.
/// * `n` - Number of observations.
///
/// ### Returns
///
/// The BIC.
#[inline]
fn bic(log_lik: f64, n_params: usize, n: usize) -> f64 {
    n_params as f64 * (n as f64).ln() - 2.0 * log_lik
}

/// Fit a two-component Gaussian mixture by EM
///
/// Initialised deterministically at the lower and upper quartile, so the fit
/// needs no seed and is reproducible. Returns `None` when the data cannot
/// support two components, which is the caller's cue to treat the regulon as
/// unimodal.
///
/// ### Params
///
/// * `x` - The scores.
/// * `sorted` - The same scores, sorted ascending, for the initialisation.
/// * `mean` - Overall mean.
/// * `variance` - Overall variance.
///
/// ### Returns
///
/// The fitted mixture, or `None` if it degenerated.
fn fit_two_component(x: &[f64], sorted: &[f64], mean: f64, variance: f64) -> Option<TwoComponent> {
    let n = x.len();
    if n < 4 || variance <= MIN_VARIANCE {
        return None;
    }

    let mut mu = [quantile_sorted(sorted, 0.25), quantile_sorted(sorted, 0.75)];
    if (mu[1] - mu[0]).abs() < f64::EPSILON {
        // A heavily zero-inflated regulon can put both quartiles on the same
        // value, so fall back to straddling the mean
        let sd = variance.sqrt();
        mu = [mean - sd, mean + sd];
    }
    let mut var = [variance, variance];
    let mut weight = [0.5_f64, 0.5];

    let mut resp = vec![0.0_f64; n];
    let mut log_lik = f64::NEG_INFINITY;

    for _ in 0..EM_MAX_ITER {
        // E step, accumulating the log likelihood as we go
        let mut new_log_lik = 0.0;
        let sd = [var[0].sqrt(), var[1].sqrt()];
        for (i, &xi) in x.iter().enumerate() {
            let d0 = weight[0] * gaussian_pdf(xi, mu[0], sd[0]);
            let d1 = weight[1] * gaussian_pdf(xi, mu[1], sd[1]);
            let total = d0 + d1;
            if total <= f64::MIN_POSITIVE {
                // Point sits impossibly far from both components
                resp[i] = 0.5;
                continue;
            }
            resp[i] = d0 / total;
            new_log_lik += total.ln();
        }

        if !new_log_lik.is_finite() {
            return None;
        }

        // M step
        let n0: f64 = resp.iter().sum();
        let n1 = n as f64 - n0;
        if n0 < 1.0 || n1 < 1.0 {
            // One component has emptied out, so there is no mixture to find
            return None;
        }

        let mut mean0 = 0.0;
        let mut mean1 = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            mean0 += resp[i] * xi;
            mean1 += (1.0 - resp[i]) * xi;
        }
        mu = [mean0 / n0, mean1 / n1];

        let mut var0 = 0.0;
        let mut var1 = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            let d0 = xi - mu[0];
            let d1 = xi - mu[1];
            var0 += resp[i] * d0 * d0;
            var1 += (1.0 - resp[i]) * d1 * d1;
        }
        var = [(var0 / n0).max(MIN_VARIANCE), (var1 / n1).max(MIN_VARIANCE)];
        weight = [n0 / n as f64, n1 / n as f64];

        if (new_log_lik - log_lik).abs() < EM_TOL * new_log_lik.abs().max(1.0) {
            log_lik = new_log_lik;
            break;
        }
        log_lik = new_log_lik;
    }

    if !log_lik.is_finite() {
        return None;
    }

    Some(TwoComponent { mu, log_lik })
}

/// Silverman's rule of thumb bandwidth, R's `bw.nrd0`
///
/// ### Params
///
/// * `sorted` - Ascending scores.
/// * `sd` - Standard deviation of the scores.
///
/// ### Returns
///
/// The bandwidth, always strictly positive.
fn silverman_bandwidth(sorted: &[f64], sd: f64) -> f64 {
    let n = sorted.len();
    let iqr = quantile_sorted(sorted, 0.75) - quantile_sorted(sorted, 0.25);
    let spread = if iqr > 0.0 { sd.min(iqr / 1.349) } else { sd };
    let bw = 0.9 * spread * (n as f64).powf(-0.2);
    if bw > 0.0 {
        bw
    } else {
        // Degenerate input; any positive width keeps the density finite
        f64::EPSILON.sqrt()
    }
}

/// Locate the density trough between the two component means
///
/// Bins the scores once, then evaluates a Gaussian kernel density estimate on
/// a grid spanning the two means. A minimum landing on either end of that grid
/// means the density is monotone across the interval, i.e. there is no real
/// trough, and `None` is returned so the caller falls back to the unimodal
/// threshold. This is the cheap stand-in for AUCell's check that the peak
/// beyond the trough is not vanishingly small.
///
/// ### Params
///
/// * `x` - The scores.
/// * `lo` - Lower bound of the search, the smaller component mean.
/// * `hi` - Upper bound of the search, the larger component mean.
/// * `bandwidth` - Kernel bandwidth.
/// * `params` - Grid and binning resolution.
///
/// ### Returns
///
/// The score at the trough, or `None` if the interval holds no interior
/// minimum.
fn find_density_trough(
    x: &[f64],
    lo: f64,
    hi: f64,
    bandwidth: f64,
    params: &BinariseParams,
) -> Option<f64> {
    if hi <= lo || params.n_grid < 3 {
        return None;
    }

    let (data_min, data_max) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    if data_max <= data_min {
        return None;
    }

    // Collapse to a histogram so the grid evaluation does not scale with cells
    let n_bins = params.n_bins.max(2);
    let bin_width = (data_max - data_min) / n_bins as f64;
    let mut counts = vec![0.0_f64; n_bins];
    for &v in x {
        let idx = (((v - data_min) / bin_width) as usize).min(n_bins - 1);
        counts[idx] += 1.0;
    }

    let inv_2h2 = 1.0 / (2.0 * bandwidth * bandwidth);
    let step = (hi - lo) / (params.n_grid - 1) as f64;

    let mut best_idx = 0_usize;
    let mut best_dens = f64::INFINITY;
    for g in 0..params.n_grid {
        let point = lo + step * g as f64;
        let mut dens = 0.0;
        for (b, &count) in counts.iter().enumerate() {
            if count == 0.0 {
                continue;
            }
            let centre = data_min + bin_width * (b as f64 + 0.5);
            let d = point - centre;
            dens += count * (-d * d * inv_2h2).exp();
        }
        if dens < best_dens {
            best_dens = dens;
            best_idx = g;
        }
    }

    // Monotone across the interval, so no genuine separation between the modes
    if best_idx == 0 || best_idx == params.n_grid - 1 {
        return None;
    }

    Some(lo + step * best_idx as f64)
}

/// Derive the threshold for one regulon
///
/// ### Params
///
/// * `row` - The regulon's AUC scores across cells.
/// * `params` - Binarisation parameters.
///
/// ### Returns
///
/// The threshold and whether the scores were called bimodal.
fn threshold_one_row(row: &[f32], params: &BinariseParams) -> (f64, bool) {
    let n = row.len();
    if n == 0 {
        return (f64::INFINITY, false);
    }

    let x: Vec<f64> = row.iter().map(|&v| v as f64).collect();
    let mean = x.iter().sum::<f64>() / n as f64;

    if n < 2 {
        return (mean, false);
    }

    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let sd = variance.sqrt();
    let unimodal = mean + UNIMODAL_SD_MULTIPLE * sd;

    if variance <= MIN_VARIANCE {
        // Every cell scores the same, so nothing can be called on
        return (unimodal, false);
    }

    let mut sorted = x.clone();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));

    let Some(fit) = fit_two_component(&x, &sorted, mean, variance) else {
        return (unimodal, false);
    };

    // Single Gaussian log likelihood, for the BIC comparison
    let log_lik_1: f64 = x.iter().map(|&v| gaussian_pdf(v, mean, sd).ln()).sum();
    if !log_lik_1.is_finite() {
        return (unimodal, false);
    }

    // 5 free parameters for the mixture (two means, two sds, one weight)
    // against 2 for the single Gaussian
    if bic(fit.log_lik, 5, n) > bic(log_lik_1, 2, n) {
        return (unimodal, false);
    }

    let lo = fit.mu[0].min(fit.mu[1]);
    let hi = fit.mu[0].max(fit.mu[1]);
    let bandwidth = silverman_bandwidth(&sorted, sd) * params.bw_adjust;

    match find_density_trough(&x, lo, hi, bandwidth, params) {
        Some(trough) => (trough, true),
        None => (unimodal, false),
    }
}

//////////
// Main //
//////////

/// Derive one on/off threshold per regulon from an AUC matrix
///
/// A cell counts as on when its score is strictly greater than the regulon's
/// threshold, matching `AUCell_assignCells` and pySCENIC's `binarize`. The
/// comparison itself is left to the caller.
///
/// Parallel over regulons; each row's fit is sequential.
///
/// ### Params
///
/// * `rows` - One row of per-cell scores per regulon, the layout returned by
///   [crate::single_cell::sc_analysis::dge_pathway_scores::calculate_aucell].
/// * `params` - Optional parameters, see [BinariseParams].
///
/// ### Returns
///
/// The thresholds and the bimodality calls, in input order.
///
/// ### References
///
/// Aibar, et al., Nat Methods, 2017
pub fn derive_regulon_thresholds(
    rows: &[Vec<f32>],
    params: Option<BinariseParams>,
) -> Result<RegulonThresholds, BixverseErrors> {
    if rows.is_empty() {
        return Err(BixverseErrors::InvalidArgument(
            "The AUC matrix has no rows".to_string(),
        ));
    }

    let params = params.unwrap_or_default();

    if params.n_grid < 3 || params.n_bins < 2 {
        return Err(BixverseErrors::InvalidArgument(
            "n_grid must be at least 3 and n_bins at least 2".to_string(),
        ));
    }

    let per_row: Vec<(f64, bool)> = rows
        .par_iter()
        .map(|row| threshold_one_row(row, &params))
        .collect();

    let (thresholds, bimodal) = per_row.into_iter().unzip();

    Ok(RegulonThresholds {
        thresholds,
        bimodal,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well separated modes at 0.1 and 0.6, 200 cells each.
    fn bimodal_row() -> Vec<f32> {
        let mut out = Vec::with_capacity(400);
        for i in 0..200 {
            out.push(0.1 + (i as f32 - 100.0) * 0.0002);
        }
        for i in 0..200 {
            out.push(0.6 + (i as f32 - 100.0) * 0.0002);
        }
        out
    }

    /// A single mode centred on 0.3.
    fn unimodal_row() -> Vec<f32> {
        (0..400)
            .map(|i| 0.3 + (i as f32 - 200.0) * 0.0005)
            .collect()
    }

    #[test]
    fn test_threshold_splits_two_clean_modes() {
        let row = bimodal_row();
        let res = derive_regulon_thresholds(std::slice::from_ref(&row), None).unwrap();
        assert!(res.bimodal[0]);
        // The trough has to sit between the two modes
        assert!(res.thresholds[0] > 0.15 && res.thresholds[0] < 0.55);
        let on = row
            .iter()
            .filter(|&&v| v as f64 > res.thresholds[0])
            .count();
        assert_eq!(on, 200);
    }

    #[test]
    fn test_threshold_falls_back_on_unimodal() {
        let row = unimodal_row();
        let res = derive_regulon_thresholds(std::slice::from_ref(&row), None).unwrap();
        assert!(!res.bimodal[0]);

        let n = row.len() as f64;
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n;
        let sd = (row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        approx::assert_relative_eq!(res.thresholds[0], mean + 2.0 * sd, epsilon = 1e-9);
    }

    #[test]
    fn test_threshold_is_reproducible() {
        let rows = vec![bimodal_row(), unimodal_row()];
        let first = derive_regulon_thresholds(&rows, None).unwrap();
        let second = derive_regulon_thresholds(&rows, None).unwrap();
        assert_eq!(first.thresholds, second.thresholds);
        assert_eq!(first.bimodal, second.bimodal);
    }

    #[test]
    fn test_threshold_constant_row_calls_nothing_on() {
        let row = vec![0.42_f32; 100];
        let res = derive_regulon_thresholds(std::slice::from_ref(&row), None).unwrap();
        assert!(!res.bimodal[0]);
        // Strictly greater, so a flat row can never switch on
        assert!(row.iter().all(|&v| (v as f64) <= res.thresholds[0]));
    }

    #[test]
    fn test_threshold_rejects_empty_input() {
        assert!(derive_regulon_thresholds(&[], None).is_err());
    }

    #[test]
    fn test_two_component_recovers_known_means() {
        let row = bimodal_row();
        let x: Vec<f64> = row.iter().map(|&v| v as f64).collect();
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let mut sorted = x.clone();
        sorted.sort_unstable_by(|a, b| a.total_cmp(b));

        let fit = fit_two_component(&x, &sorted, mean, variance).unwrap();
        let lo = fit.mu[0].min(fit.mu[1]);
        let hi = fit.mu[0].max(fit.mu[1]);
        approx::assert_relative_eq!(lo, 0.1, epsilon = 1e-3);
        approx::assert_relative_eq!(hi, 0.6, epsilon = 1e-3);
    }
}
