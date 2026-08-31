//! blitzGSEA: gene set enrichment with a gamma-approximated null.
//!
//! Permutation GSEA spends everything it has on the null. blitzGSEA replaces
//! the per-pathway permutations with a calibration step: draw random gene sets
//! at a spread of anchor sizes, fit a gamma to the resulting enrichment scores
//! at each anchor, and smooth the fitted parameters across sizes. Scoring a
//! pathway is then a single gamma tail evaluation, whatever its size, and the
//! calibration is reusable for every library run against the same signature.
//!
//! ### Where this differs from the reference implementation
//!
//! * The enrichment score comes from [`calc_gsea_stats`], which touches only
//!   the `k` hits rather than running a cumulative sum over all `n` genes. The
//!   reference does a full-length numpy `cumsum` per permutation, and the
//!   calibration loop is essentially the whole runtime.
//! * Tail probabilities go through [`gamma_sf`] rather than `1 - cdf`. The
//!   reference escalates to 50-digit `mpmath` whenever the cdf saturates at one;
//!   the upper regularised incomplete gamma hands back the same number with no
//!   cancellation and no arbitrary-precision arithmetic.
//! * The anchor grid is log-spaced over the whole usable size range rather
//!   than evenly spaced up to the library's largest gene set. The gamma
//!   parameters are a property of the signature, not of the library, so a null
//!   built this way serves Reactome and GO alike from one calibration, and
//!   puts about half its knots below a hundred without needing a hardcoded
//!   list of small sizes to patch the bottom.
//! * Calibration is reproducible from the seed. The reference seeds Python's
//!   `random` but samples through `numpy.random`, so its per-anchor draws are
//!   not seed-controlled and its multiprocessing workers share a parent state.
//! * The reference nudges a negative-tail probability off exactly one half by
//!   subtracting the gamma cdf from it. That branch fires only when `pos_ratio`
//!   is below a half and the score is weakly negative, where the p-value is
//!   already close to one, and it has no positive-tail counterpart. Dropped as
//!   a discontinuous patch rather than a derivation; the p-value reported here
//!   is the larger of the two, so the divergence only ever makes a
//!   non-significant result more non-significant.
//!
//! ### References
//!
//! Lachmann et al., Bioinformatics, 2022

use rayon::prelude::*;

use crate::core::base::loess::{LoessRegression, LoessSurface};
use crate::core::math::distributions::{gamma_cdf, gamma_sf, norm_ppf};
use crate::core::math::stats::{calc_fdr, fit_gamma_mle, ks_test_1samp};
use crate::core::math::vector_helpers::interp_log_linear_at;
use crate::enrichment::gsea::{calc_gsea_stats, create_random_gs_indices};
use crate::prelude::*;

////////////
// Consts //
////////////

/// Default permutations drawn per anchor size.
///
/// The reference's own default. Enough for the split-tail fit to clear
/// [`BLITZ_MIN_TAIL_COUNT`] on both sides.
const BLITZ_DEFAULT_PERMUTATIONS: usize = 2000;

/// Default number of log-spaced anchors requested.
///
/// Collapses to roughly 35 distinct sizes on a twenty thousand gene signature,
/// about half of them below one hundred.
const BLITZ_DEFAULT_ANCHORS: usize = 40;

/// Default random seed.
const BLITZ_DEFAULT_SEED: u64 = 42;

/// Fraction of the signature the anchor grid runs up to.
///
/// The grid has to cover every gene set that will ever be scored, but it must
/// not run to the end of the signature: at `k` near `n` there is barely a miss
/// left and the enrichment score degenerates, which gives the gamma fit nothing
/// to work with. Half the universe is far above any curated library (ten
/// thousand genes for a twenty thousand gene signature) and well clear of that
/// degeneracy.
const BLITZ_ANCHOR_TOP_FRACTION: f64 = 0.5;

/// Fewest anchors a usable calibration can be built from.
///
/// One anchor makes the null size-independent, and the size it would land on is
/// `k = 1`, whose enrichment score is purely positional and looks nothing like
/// any real gene set's. Silently returning that for every pathway is worse than
/// refusing.
const BLITZ_MIN_ANCHORS: usize = 2;

/// Fewest permutations a gamma can be fitted to.
///
/// Not a quality threshold, just the point below which the fit is undefined: one
/// observation has no spread on the log scale and zero has nothing to fit at
/// all. Both otherwise surface as a confusing error from deep inside the fitter.
const BLITZ_MIN_PERMUTATIONS: usize = 2;

/// Minimum count in each tail before the two tails are fitted separately.
///
/// A gamma fitted to fewer observations than this is noise, and the whole point
/// of splitting the tails is to capture a real asymmetry rather than sampling
/// variation.
const BLITZ_MIN_TAIL_COUNT: usize = 250;

/// Permutation count below which the tails are pooled regardless.
///
/// With fewer draws than this neither tail can clear
/// [`BLITZ_MIN_TAIL_COUNT`] often enough for the split to be stable.
const BLITZ_MIN_PERMUTATIONS_SPLIT: usize = 1000;

/// Loess span for the shape parameter across anchor sizes.
///
/// Measured, not inherited. Against a 40000-permutation reference curve, over
/// the sizes real gene sets occupy, smoothing the default 2000-permutation fits
/// leaves a worst-case error of 3.2% at this span, against 7.0% unsmoothed and
/// 10.6% at the reference implementation's 0.6. The reference's spans suit its
/// denser linear grid; 0.6 of a 35-anchor log grid is 21 anchors spanning three
/// decades, which smooths straight through the curve rather than along it.
const BLITZ_SPAN_SHAPE: f64 = 0.25;

/// Loess span for the scale parameter across anchor sizes.
///
/// The scale falls off sharply at small sizes and a wide span flattens exactly
/// the region that matters most: on the same measurement as
/// [`BLITZ_SPAN_SHAPE`], 6.3% here against 58% at a span of 0.6. This is the one
/// span the reference already had narrow enough to keep.
const BLITZ_SPAN_SCALE: f64 = 0.15;

/// Loess span for the positive-score ratio across anchor sizes.
///
/// The ratio sits near a half at every size, so the span barely matters: every
/// value from 0.15 to 0.6 lands between 1.5% and 1.9% on the same measurement.
const BLITZ_SPAN_RATIO: f64 = 0.5;

/// Bisquare reweighting passes applied to every anchor smoother.
///
/// A single anchor whose gamma fit went badly would otherwise drag the curve
/// through its neighbours. Three is statsmodels' `lowess` default and what the
/// reference implementation therefore runs.
const BLITZ_LOESS_ROBUSTNESS_ITERS: usize = 3;

/// Local polynomial degree for the anchor smoothers.
const BLITZ_LOESS_DEGREE: usize = 1;

/// Floor applied to a one-tailed probability before it reaches [`norm_ppf`].
///
/// A gamma tail can underflow to exactly zero, and the normal quantile of zero
/// is negative infinity. Clamping here caps the reported NES near 38 rather
/// than handing back an infinity that no downstream table can hold.
const BLITZ_MIN_TAIL_PROB: f64 = f64::MIN_POSITIVE;

/// Largest a one-tailed probability can be before the two-sided p-value is one.
const BLITZ_MAX_TAIL_PROB: f64 = 0.5;

////////////
// Params //
////////////

/// Parameters controlling a blitzGSEA run
#[derive(Clone, Copy, Debug)]
pub struct BlitzGseaParams {
    /// Random gene sets drawn per anchor size during calibration
    pub permutations: usize,
    /// Log-spaced anchor sizes requested. Sizes that collide after rounding are
    /// collapsed, so the grid is usually a little smaller than this
    pub anchors: usize,
    /// Force a single pooled gamma for both tails rather than fitting them
    /// separately. Forced on anyway when either tail is too thin
    pub symmetric: bool,
    /// Centre the signature on its mean before taking absolute values. The
    /// enrichment score is not otherwise invariant to an offset
    pub centre: bool,
    /// Run the Kolmogorov-Smirnov goodness-of-fit diagnostic at every anchor.
    /// Costs a sort per anchor and only produces a warning
    pub ks_test: bool,
    /// Random seed
    pub seed: u64,
}

impl Default for BlitzGseaParams {
    fn default() -> Self {
        Self::new(None, None, None, None, None, None)
    }
}

impl BlitzGseaParams {
    /// Generate a new set of blitzGSEA parameters
    ///
    /// Every field is optional and `None` takes the default, so a caller
    /// supplies only what it wants to change. [`Default`] is this with nothing
    /// supplied, which keeps the two from drifting apart.
    ///
    /// ### Params
    ///
    /// * `permutations` - Random gene sets per anchor size. Defaults to
    ///   [`BLITZ_DEFAULT_PERMUTATIONS`]
    /// * `anchors` - Log-spaced anchor sizes requested. Defaults to
    ///   [`BLITZ_DEFAULT_ANCHORS`]
    /// * `symmetric` - Pool both tails into one gamma. Defaults to `false`,
    ///   though a low permutation count forces it on regardless
    /// * `centre` - Centre the signature before scoring. Defaults to `true`
    /// * `ks_test` - Run the goodness-of-fit diagnostic. Defaults to `true`
    /// * `seed` - Random seed. Defaults to [`BLITZ_DEFAULT_SEED`]
    ///
    /// ### Returns
    ///
    /// The initialised parameter structure.
    pub fn new(
        permutations: Option<usize>,
        anchors: Option<usize>,
        symmetric: Option<bool>,
        centre: Option<bool>,
        ks_test: Option<bool>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            permutations: permutations.unwrap_or(BLITZ_DEFAULT_PERMUTATIONS),
            anchors: anchors.unwrap_or(BLITZ_DEFAULT_ANCHORS),
            symmetric: symmetric.unwrap_or(false),
            centre: centre.unwrap_or(true),
            ks_test: ks_test.unwrap_or(true),
            seed: seed.unwrap_or(BLITZ_DEFAULT_SEED),
        }
    }
}

//////////////
// Anchors //
//////////////

/// Gamma parameters fitted to the null enrichment scores at one anchor size
#[derive(Clone, Copy, Debug)]
struct AnchorFit {
    /// The set size this anchor was fitted at
    size: f64,
    /// Shape of the gamma fitted to the positive scores
    shape_pos: f64,
    /// Scale of the gamma fitted to the positive scores
    scale_pos: f64,
    /// Shape of the gamma fitted to the negated negative scores
    shape_neg: f64,
    /// Scale of the gamma fitted to the negated negative scores
    scale_neg: f64,
    /// Fraction of non-zero null scores that were positive
    pos_ratio: f64,
    /// Goodness-of-fit p-value for the positive tail, 1.0 when disabled
    ks_pos: f64,
    /// Goodness-of-fit p-value for the negative tail, 1.0 when disabled
    ks_neg: f64,
}

/// The interpolated gamma parameters governing one gene set size
#[derive(Clone, Copy, Debug)]
pub struct BlitzGseaTail {
    /// Shape of the positive-tail gamma
    pub shape_pos: f64,
    /// Scale of the positive-tail gamma
    pub scale_pos: f64,
    /// Shape of the negative-tail gamma
    pub shape_neg: f64,
    /// Scale of the negative-tail gamma
    pub scale_neg: f64,
    /// Fraction of the null mass sitting above zero, in `[0, 1]`
    pub pos_ratio: f64,
}

/////////////////
// Null model //
/////////////////

/// The calibrated null: smoothed gamma parameters over a grid of anchor sizes.
///
/// This is the whole reusable product of a calibration run. It is plain numeric
/// data of the order of a kilobyte, holds no handles and no interior state, so
/// it round-trips through any serialisation the caller likes rather than
/// needing to be kept alive behind a pointer.
#[derive(Clone, Debug)]
pub struct BlitzGseaNull {
    /// Anchor set sizes, strictly ascending
    pub anchor_sizes: Vec<f64>,
    /// Smoothed positive-tail shape at each anchor
    pub shape_pos: Vec<f64>,
    /// Smoothed positive-tail scale at each anchor
    pub scale_pos: Vec<f64>,
    /// Smoothed negative-tail shape at each anchor
    pub shape_neg: Vec<f64>,
    /// Smoothed negative-tail scale at each anchor
    pub scale_neg: Vec<f64>,
    /// Smoothed fraction of positive null scores at each anchor
    pub pos_ratio: Vec<f64>,
    /// Mean goodness-of-fit p-value across anchors for the positive tail
    pub ks_pos: f64,
    /// Mean goodness-of-fit p-value across anchors for the negative tail
    pub ks_neg: f64,
    /// Whether the signature was centred before the null was drawn. Scoring has
    /// to make the same choice or the enrichment scores land on a scale the null
    /// was never fitted to
    pub centred: bool,
}

impl BlitzGseaNull {
    /// Interpolate the gamma parameters for a given gene set size
    ///
    /// In log space, because that is the axis the anchors are spaced on and the
    /// axis the parameters were smoothed against. Interpolating linearly across
    /// a geometrically widening grid would give the largest gene sets the
    /// coarsest resolution, which is where the segments are widest.
    ///
    /// Extrapolates past both ends, and clamps the ratio into `[0, 1]`, which
    /// the smoother and the extrapolation can each step outside. A size below
    /// one is clamped up: a gene set of no genes has no null.
    ///
    /// ### Params
    ///
    /// * `size` - The gene set size to evaluate at
    ///
    /// ### Returns
    ///
    /// The interpolated [`BlitzGseaTail`] for that size.
    pub fn tail_at(&self, size: f64) -> BlitzGseaTail {
        let size = size.max(1.0);
        let at = |y: &[f64]| interp_log_linear_at(&self.anchor_sizes, y, size);

        BlitzGseaTail {
            shape_pos: at(&self.shape_pos),
            scale_pos: at(&self.scale_pos),
            shape_neg: at(&self.shape_neg),
            scale_neg: at(&self.scale_neg),
            pos_ratio: at(&self.pos_ratio).clamp(0.0, 1.0),
        }
    }

    /// Check the invariants [`tail_at`](Self::tail_at) relies on
    ///
    /// Every parameter vector is read in lockstep with the anchor grid, so a
    /// short one would index out of bounds. The fields are public and the null
    /// round-trips through R, so neither construction path is guaranteed to have
    /// gone through [`calibrate_null`].
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`BixverseErrors::InvalidArgument`] naming the bad field.
    pub fn validate(&self) -> Result<(), BixverseErrors> {
        let n = self.anchor_sizes.len();
        if n == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "The blitzGSEA null model has an empty anchor grid.".to_string(),
            ));
        }

        for (name, values) in [
            ("shape_pos", &self.shape_pos),
            ("scale_pos", &self.scale_pos),
            ("shape_neg", &self.shape_neg),
            ("scale_neg", &self.scale_neg),
            ("pos_ratio", &self.pos_ratio),
        ] {
            if values.len() != n {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "The blitzGSEA null model has {} values for '{name}' but {n} anchor sizes.",
                    values.len()
                )));
            }
        }

        Ok(())
    }
}

/////////////
// Results //
/////////////

/// One pathway's scores, before they are transposed into [`BlitzGseaResults`]
///
/// The scoring fan-out has to hand five values back per pathway and rayon wants
/// one item, so they travel together rather than as an anonymous tuple.
#[derive(Clone, Debug)]
struct ScoredPathway {
    /// Enrichment score
    es: f64,
    /// Normalised enrichment score
    nes: f64,
    /// Two-sided p-value
    pval: f64,
    /// Gene set size
    size: usize,
    /// Index positions of the leading edge genes
    leading_edge: Vec<i32>,
}

/// Per-pathway results of a blitzGSEA run
#[derive(Clone, Debug)]
pub struct BlitzGseaResults {
    /// Enrichment score
    pub es: Vec<f64>,
    /// Normalised enrichment score, the signed normal quantile of the one-sided
    /// tail probability
    pub nes: Vec<f64>,
    /// Two-sided p-value from the gamma approximation
    pub pvals: Vec<f64>,
    /// Sidak-adjusted p-value
    pub sidak: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value
    pub fdr: Vec<f64>,
    /// Gene set size after intersecting with the signature
    pub size: Vec<usize>,
    /// Index positions of the leading edge genes, in the indexing of the input
    pub leading_edge: Vec<Vec<i32>>,
}

/////////////
// Helpers //
/////////////

/// Centre a signature on its mean, or pass it through untouched.
///
/// ### Params
///
/// * `stats` - The signature values
/// * `centre` - Whether to subtract the mean
///
/// ### Returns
///
/// The signature, centred or not.
fn centre_stats<T: BixverseFloat>(stats: &[T], centre: bool) -> Vec<T> {
    if !centre || stats.is_empty() {
        return stats.to_vec();
    }

    let n = T::from_usize(stats.len()).unwrap();
    let mean = stats.iter().fold(T::zero(), |acc, &x| acc + x) / n;

    stats.iter().map(|&x| x - mean).collect()
}

/// Build the anchor grid for the calibration.
///
/// Log-spaced from one gene up to [`BLITZ_ANCHOR_TOP_FRACTION`] of the
/// signature. Log spacing is what makes the grid library-independent: the gamma
/// parameters move fastest at small `k` and barely at all at large `k`, so
/// equal steps in `ln k` put knots where the curvature is, and one grid covers a
/// five hundred gene Reactome set and a five thousand gene GO set equally well.
///
/// The reference instead spaces evenly up to the library's largest set and then
/// bolts on a fixed list of seventeen small sizes to patch the bottom. That ties
/// the null to the library and still spends most of its anchors on the flat end
/// of the curve. Forty log-spaced anchors over a twenty thousand gene signature
/// land seventeen below one hundred on their own, and stay dense in the hundreds
/// where the hardcoded list stops entirely.
///
/// Duplicates after rounding are collapsed, so asking for forty anchors returns
/// slightly fewer. That is the grid saturating the small end, not a problem.
///
/// ### Params
///
/// * `n_genes` - Length of the signature
/// * `anchors` - Number of log-spaced anchors requested
///
/// ### Returns
///
/// Strictly ascending, deduplicated anchor sizes, all in `1..n_genes`.
fn anchor_grid(n_genes: usize, anchors: usize) -> Vec<usize> {
    if n_genes < 2 {
        return Vec::new();
    }

    let top = ((n_genes as f64 * BLITZ_ANCHOR_TOP_FRACTION) as usize).clamp(1, n_genes - 1);
    if anchors <= 1 || top == 1 {
        return vec![1];
    }

    let step = (top as f64).ln() / (anchors - 1) as f64;
    let mut sizes: Vec<usize> = (0..anchors)
        .map(|i| ((step * i as f64).exp().round() as usize).clamp(1, top))
        .collect();

    sizes.sort_unstable();
    sizes.dedup();

    sizes
}

/// Enrichment scores of `permutations` random gene sets of a fixed size.
///
/// Each draw is an independent uniform subset, unlike
/// [`crate::enrichment::gsea::calc_gsea_stat_traditional_batch`], which shares
/// one pool of permutations across every size it scores. Sharing is cheaper but
/// correlates the anchors, and the anchors are precisely what gets smoothed
/// here, so the correlation would show up as structure in the fitted curve.
///
/// ### Params
///
/// * `stats` - The signature values, sorted descending
/// * `size` - Gene set size to sample at
/// * `permutations` - Number of random sets to draw
/// * `seed` - Random seed for this anchor
///
/// ### Returns
///
/// The enrichment scores, one per draw.
fn null_scores<T: BixverseFloat>(
    stats: &[T],
    size: usize,
    permutations: usize,
    seed: u64,
) -> Vec<f64> {
    let samples = create_random_gs_indices(permutations, size, stats.len(), seed, false);

    // Nested inside the per-anchor fan-out on purpose. Anchor cost spans three
    // orders of magnitude between the smallest and largest set size, so the
    // outer loop alone would leave most threads idle behind the slowest anchor.
    samples
        .into_par_iter()
        .map(|mut indices| {
            // `calc_gsea_stats` counts misses from each hit's position, which
            // only works on an ascending index list
            indices.sort_unstable();
            let indices: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
            let stats_res = calc_gsea_stats(stats, &indices, T::one(), false, false, false);
            stats_res.es.to_f64().unwrap_or(0.0)
        })
        .collect()
}

/// Fit gamma tails to one anchor's null enrichment scores.
///
/// Both tails are fitted on strictly positive data, the negative one after
/// negation, because the gamma has support on the positive half line. Zeros are
/// dropped rather than nudged: they carry no information about either tail.
///
/// The tails are pooled into one fit when the caller asks for it or when either
/// side holds fewer than [`BLITZ_MIN_TAIL_COUNT`] scores, since a gamma fitted
/// to less than that is describing sampling noise.
///
/// ### Params
///
/// * `scores` - Null enrichment scores at this anchor
/// * `size` - The anchor set size, carried through to the fit
/// * `symmetric` - Force the pooled fit
/// * `ks_test` - Run the goodness-of-fit diagnostic
///
/// ### Returns
///
/// The [`AnchorFit`], or [`BixverseErrors::InvalidArgument`] when a tail holds
/// nothing to fit.
fn fit_anchor(
    scores: &[f64],
    size: usize,
    symmetric: bool,
    ks_test: bool,
) -> Result<AnchorFit, BixverseErrors> {
    let positive: Vec<f64> = scores.iter().copied().filter(|&s| s > 0.0).collect();
    let negative: Vec<f64> = scores.iter().filter(|&&s| s < 0.0).map(|&s| -s).collect();

    let n_pos = positive.len();
    let n_neg = negative.len();
    if n_pos + n_neg == 0 {
        return Err(BixverseErrors::InvalidArgument(format!(
            "Every null enrichment score at anchor size {size} was exactly zero."
        )));
    }

    let pooled = symmetric || n_pos < BLITZ_MIN_TAIL_COUNT || n_neg < BLITZ_MIN_TAIL_COUNT;
    let pos_ratio = n_pos as f64 / (n_pos + n_neg) as f64;

    let ks_of = |sample: &[f64], shape: f64, scale: f64| -> Result<f64, BixverseErrors> {
        if !ks_test {
            return Ok(1.0);
        }
        Ok(ks_test_1samp(sample, |x| gamma_cdf(x, shape, scale).unwrap_or(0.0))?.pval)
    };

    if pooled {
        let mut both = positive;
        both.extend_from_slice(&negative);

        let (shape, scale) = fit_gamma_mle(&both)?;
        let ks = ks_of(&both, shape, scale)?;

        return Ok(AnchorFit {
            size: size as f64,
            shape_pos: shape,
            scale_pos: scale,
            shape_neg: shape,
            scale_neg: scale,
            pos_ratio,
            ks_pos: ks,
            ks_neg: ks,
        });
    }

    let (shape_pos, scale_pos) = fit_gamma_mle(&positive)?;
    let (shape_neg, scale_neg) = fit_gamma_mle(&negative)?;

    Ok(AnchorFit {
        size: size as f64,
        shape_pos,
        scale_pos,
        shape_neg,
        scale_neg,
        pos_ratio,
        ks_pos: ks_of(&positive, shape_pos, scale_pos)?,
        ks_neg: ks_of(&negative, shape_neg, scale_neg)?,
    })
}

/// Robust loess smoothing of one fitted parameter across the anchor sizes.
///
/// The x-axis is `ln(size)`, not size. Loess picks its neighbourhood by nearest
/// neighbours, so the spacing does not change *which* anchors are in it, but the
/// local polynomial is fitted against x itself: on a geometric grid a
/// neighbourhood at the top spans a decade where one at the bottom spans a
/// handful of genes, the tricube weights then collapse onto the largest few
/// anchors, and a local linear fit to a convex curve at that boundary reads low.
/// Smoothing on the axis the grid is uniform in removes the asymmetry.
///
/// [`LoessSurface::Direct`] rather than the interpolating surface: an anchor
/// grid is tens of points, far below the size at which the spline pays for
/// itself, and `LoessRegression` would fall back to the direct fit anyway.
///
/// ### Params
///
/// * `log_sizes` - Natural logs of the anchor sizes, ascending
/// * `values` - The fitted parameter at each anchor
/// * `span` - Loess span
///
/// ### Returns
///
/// The smoothed values, in the order of `log_sizes`.
fn smooth_anchors(log_sizes: &[f64], values: &[f64], span: f64) -> Vec<f64> {
    LoessRegression::with_options(
        span,
        BLITZ_LOESS_DEGREE,
        LoessSurface::Direct,
        BLITZ_LOESS_ROBUSTNESS_ITERS,
    )
    .fit(log_sizes, values)
    .fitted_vals
}

/// Sidak-adjusted p-values for a family of `m` tests.
///
/// `1 - (1 - p)^m`, formed as `-expm1(m ln1p(-p))` so that a small `p` keeps
/// every digit it had instead of being annihilated against one.
///
/// ### Params
///
/// * `pvals` - The raw p-values
///
/// ### Returns
///
/// The adjusted p-values, in the input order.
fn sidak(pvals: &[f64]) -> Vec<f64> {
    let m = pvals.len() as f64;

    pvals
        .iter()
        .map(|&p| (-((m * (-p).ln_1p()).exp_m1())).clamp(0.0, 1.0))
        .collect()
}

/// Two-sided p-value and normalised enrichment score for one pathway.
///
/// The reference computes `1 - min(cdf * ratio + 1 - ratio, 1)` for a positive
/// score, which is algebraically `ratio * (1 - cdf)`, and the mirrored
/// expression for a negative one. Evaluating the survival function directly
/// removes the subtraction from one, and with it the whole reason the reference
/// escalates to arbitrary precision.
///
/// The NES is the normal quantile of that one-sided probability, signed by the
/// direction of the enrichment score.
///
/// ### Params
///
/// * `es` - The observed enrichment score
/// * `tail` - Interpolated gamma parameters for this gene set size
///
/// ### Returns
///
/// `(p-value, NES)`, or [`BixverseErrors::InvalidArgument`] when the
/// interpolated gamma parameters are not usable.
///
/// Public so the parity suite can gate the tail mapping on its own, without
/// having to reverse an enrichment score out of a gene set first.
pub fn pval_and_nes(es: f64, tail: &BlitzGseaTail) -> Result<(f64, f64), BixverseErrors> {
    let one_sided = if es > 0.0 {
        tail.pos_ratio * gamma_sf(es, tail.shape_pos, tail.scale_pos)?
    } else {
        (1.0 - tail.pos_ratio) * gamma_sf(-es, tail.shape_neg, tail.scale_neg)?
    };

    let one_sided = one_sided.min(BLITZ_MAX_TAIL_PROB);
    let quantile = norm_ppf(one_sided.max(BLITZ_MIN_TAIL_PROB))?;

    // `norm_ppf` of a probability at or below a half is non-positive, so the
    // sign flip is what puts a strong positive score at a large positive NES.
    let nes = if es > 0.0 { -quantile } else { quantile };

    Ok((2.0 * one_sided, nes))
}

/////////////////
// Calibration //
/////////////////

/// Calibrate the gamma null for a signature.
///
/// Draws `params.permutations` random gene sets at each anchor size, fits gamma
/// tails to the resulting enrichment scores, then smooths each fitted parameter
/// across sizes. The result depends only on the signature and the parameters,
/// so it can be cached and reused for every library scored against that
/// signature.
///
/// The signature must already be sorted descending; centring, if requested,
/// happens here because the enrichment score is not invariant to an offset.
///
/// Nothing about the library enters here. One call per signature serves every
/// library scored against it, whatever their size ranges.
///
/// ### Params
///
/// * `stats` - The signature values, sorted descending
/// * `params` - Run parameters
///
/// ### Returns
///
/// The calibrated [`BlitzGseaNull`], or [`BixverseErrors::InvalidArgument`] for
/// a signature too short to sample from or an anchor whose fit failed.
pub fn calibrate_null<T: BixverseFloat>(
    stats: &[T],
    params: &BlitzGseaParams,
) -> Result<BlitzGseaNull, BixverseErrors> {
    if params.anchors < BLITZ_MIN_ANCHORS {
        return Err(BixverseErrors::InvalidArgument(format!(
            "blitzGSEA needs at least {BLITZ_MIN_ANCHORS} anchors to build a size-dependent \
             null; got {}.",
            params.anchors
        )));
    }
    if params.permutations < BLITZ_MIN_PERMUTATIONS {
        return Err(BixverseErrors::InvalidArgument(format!(
            "blitzGSEA needs at least {BLITZ_MIN_PERMUTATIONS} permutations per anchor to fit \
             a gamma; got {}.",
            params.permutations
        )));
    }

    let sizes = anchor_grid(stats.len(), params.anchors);
    if sizes.is_empty() {
        return Err(BixverseErrors::InvalidArgument(format!(
            "A signature of {} genes is too short to calibrate against; a gene set needs at \
             least one hit and one miss.",
            stats.len()
        )));
    }

    let centred = centre_stats(stats, params.centre);
    let symmetric = params.symmetric || params.permutations < BLITZ_MIN_PERMUTATIONS_SPLIT;

    let fits: Vec<AnchorFit> = sizes
        .par_iter()
        .map(|&size| {
            // Offsetting by the size keeps every anchor on its own stream while
            // staying a pure function of the caller's seed
            let seed = params.seed.wrapping_add(size as u64);
            let scores = null_scores(&centred, size, params.permutations, seed);
            fit_anchor(&scores, size, symmetric, params.ks_test)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let anchor_sizes: Vec<f64> = fits.iter().map(|f| f.size).collect();
    // The smoother works on the axis the grid is uniform in; the null stores
    // the sizes themselves, which is what a reader and the R layer want.
    let log_sizes: Vec<f64> = anchor_sizes.iter().map(|size| size.ln()).collect();
    let column =
        |extract: fn(&AnchorFit) -> f64| -> Vec<f64> { fits.iter().map(extract).collect() };

    let n = fits.len() as f64;
    let mean_of =
        |extract: fn(&AnchorFit) -> f64| -> f64 { fits.iter().map(extract).sum::<f64>() / n };

    Ok(BlitzGseaNull {
        shape_pos: smooth_anchors(&log_sizes, &column(|f| f.shape_pos), BLITZ_SPAN_SHAPE),
        scale_pos: smooth_anchors(&log_sizes, &column(|f| f.scale_pos), BLITZ_SPAN_SCALE),
        shape_neg: smooth_anchors(&log_sizes, &column(|f| f.shape_neg), BLITZ_SPAN_SHAPE),
        scale_neg: smooth_anchors(&log_sizes, &column(|f| f.scale_neg), BLITZ_SPAN_SCALE),
        pos_ratio: smooth_anchors(&log_sizes, &column(|f| f.pos_ratio), BLITZ_SPAN_RATIO),
        ks_pos: mean_of(|f| f.ks_pos),
        ks_neg: mean_of(|f| f.ks_neg),
        centred: params.centre,
        anchor_sizes,
    })
}

//////////////
// Scoring //
//////////////

/// Score gene sets against a calibrated null.
///
/// Each pathway costs one enrichment score plus one gamma tail evaluation. The
/// pathways are expected to have been filtered to the caller's size bounds and
/// intersected with the signature already, which is where the R layer does it.
///
/// ### Params
///
/// Index lists are sorted and deduplicated here rather than being trusted.
/// [`calc_gsea_stats`] counts the misses before each hit from that hit's
/// position, so an unsorted or repeated index silently returns a different
/// enrichment score rather than failing: `[1, 5, 9, 40, 100]` and
/// `[100, 5, 40, 1, 9]` are the same gene set and used to score 0.952 and 0.997.
///
/// ### Params
///
/// * `stats` - The signature values, sorted descending
/// * `pathways` - Index positions of each gene set's members, in the indexing
///   `one_indexed` selects. Order and duplicates do not matter
/// * `null` - The calibrated null from [`calibrate_null`]
/// * `params` - Run parameters. Only `centre` is read here, and it has to match
///   what the calibration used, which is checked against the null
/// * `one_indexed` - Whether `pathways` carries R's one-based indices
///
/// ### Returns
///
/// The [`BlitzGseaResults`], or [`BixverseErrors::InvalidArgument`] for an empty
/// gene set, a null that disagrees with `params.centre`, a malformed null, or a
/// bad gamma parameter.
pub fn blitzgsea_score<T: BixverseFloat>(
    stats: &[T],
    pathways: &[Vec<i32>],
    null: &BlitzGseaNull,
    params: &BlitzGseaParams,
    one_indexed: bool,
) -> Result<BlitzGseaResults, BixverseErrors> {
    null.validate()?;

    if null.centred != params.centre {
        return Err(BixverseErrors::InvalidArgument(format!(
            "The blitzGSEA null was calibrated with centre = {} but scoring was asked for \
             centre = {}. The enrichment scores would sit on a scale the null was never \
             fitted to.",
            null.centred, params.centre
        )));
    }

    let centred = centre_stats(stats, params.centre);

    let scored: Vec<ScoredPathway> = pathways
        .par_iter()
        .map(|indices| {
            // `calc_gsea_stats` derives each hit's miss count from its position,
            // so it needs an ascending list with no repeats. Cheap next to the
            // scan, and a silently wrong score is worse than the sort.
            let mut indices = indices.clone();
            indices.sort_unstable();
            indices.dedup();

            if indices.is_empty() {
                return Err(BixverseErrors::InvalidArgument(
                    "A gene set has no genes left after intersecting with the signature; \
                     filter it out before scoring."
                        .to_string(),
                ));
            }

            let gsea_stats =
                calc_gsea_stats(&centred, &indices, T::one(), true, false, one_indexed);
            let es = gsea_stats.es.to_f64().unwrap_or(0.0);

            let tail = null.tail_at(indices.len() as f64);
            let (pval, nes) = pval_and_nes(es, &tail)?;

            Ok(ScoredPathway {
                es,
                nes,
                pval,
                size: indices.len(),
                leading_edge: gsea_stats.leading_edge,
            })
        })
        .collect::<Result<Vec<_>, BixverseErrors>>()?;

    let mut es = Vec::with_capacity(scored.len());
    let mut nes = Vec::with_capacity(scored.len());
    let mut pvals = Vec::with_capacity(scored.len());
    let mut size = Vec::with_capacity(scored.len());
    let mut leading_edge = Vec::with_capacity(scored.len());

    for pathway in scored {
        es.push(pathway.es);
        nes.push(pathway.nes);
        pvals.push(pathway.pval);
        size.push(pathway.size);
        leading_edge.push(pathway.leading_edge);
    }

    Ok(BlitzGseaResults {
        sidak: sidak(&pvals),
        fdr: calc_fdr(&pvals),
        es,
        nes,
        pvals,
        size,
        leading_edge,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A descending signature with real spread on both sides of zero, which is
    /// the shape every gamma fit here assumes.
    fn signature(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 3.0 - 6.0 * (i as f64) / (n as f64 - 1.0))
            .collect()
    }

    fn small_params() -> BlitzGseaParams {
        BlitzGseaParams {
            permutations: 200,
            anchors: 8,
            ..Default::default()
        }
    }

    /////////////
    // Anchors //
    /////////////

    #[test]
    fn test_anchor_grid_is_sorted_and_unique() {
        let grid = anchor_grid(10_000, 40);

        assert!(grid.windows(2).all(|w| w[0] < w[1]));
        assert!(grid.iter().all(|&size| (1..10_000).contains(&size)));
        assert_eq!(grid[0], 1);
    }

    /// Log spacing is there to concentrate anchors where the gamma parameters
    /// actually move. Half of them below a hundred is the property that buys.
    #[test]
    fn test_anchor_grid_concentrates_on_the_small_end() {
        let grid = anchor_grid(20_000, 40);

        let below_100 = grid.iter().filter(|&&size| size < 100).count();

        assert!(below_100 >= 15, "only {below_100} anchors below 100");
        // at least two fifths of the grid, against one in forty for a linear one
        assert!(
            below_100 * 5 >= grid.len() * 2,
            "{below_100} of {} anchors below 100 is underweighted",
            grid.len()
        );
    }

    /// The grid stops well short of the signature, where a gene set has no
    /// misses left and the enrichment score degenerates.
    #[test]
    fn test_anchor_grid_stops_short_of_the_signature() {
        let grid = anchor_grid(10_000, 40);
        let top = *grid.iter().max().unwrap();

        assert!(top <= 5_000, "top anchor {top} runs too far");
        assert!(
            top >= 4_000,
            "top anchor {top} does not cover real libraries"
        );
    }

    /// The same grid has to serve libraries with very different size ranges,
    /// which is what makes one calibration reusable.
    #[test]
    fn test_anchor_grid_does_not_depend_on_the_library() {
        // Reactome tops out near 500, GO runs to several thousand. Both are
        // interior to the same grid.
        let grid = anchor_grid(20_000, 40);

        assert!(grid.iter().any(|&size| size > 500));
        assert!(grid.iter().any(|&size| size > 5_000));
    }

    #[test]
    fn test_anchor_grid_degenerate_signatures() {
        assert!(anchor_grid(1, 40).is_empty());
        assert!(anchor_grid(0, 40).is_empty());
        assert_eq!(anchor_grid(3, 40), vec![1]);
        assert_eq!(anchor_grid(10_000, 1), vec![1]);
    }

    //////////////
    // Helpers //
    //////////////

    #[test]
    fn test_centre_stats() {
        let stats = [3.0, 1.0, -1.0, 1.0];

        let centred = centre_stats(&stats, true);
        let sum: f64 = centred.iter().sum();

        assert_relative_eq!(sum, 0.0, epsilon = 1e-12);
        assert_eq!(centre_stats(&stats, false), stats.to_vec());
    }

    #[test]
    fn test_sidak_matches_the_closed_form() {
        let pvals = [0.01, 0.2, 0.5];
        let m = pvals.len() as f64;

        let adjusted = sidak(&pvals);

        for (&p, &adj) in pvals.iter().zip(adjusted.iter()) {
            assert_relative_eq!(adj, 1.0 - (1.0 - p).powf(m), max_relative = 1e-12);
        }
    }

    /// The `expm1`/`ln1p` form is the point: the naive expression rounds a tiny
    /// p-value straight to `m * p` at best and to zero at worst.
    #[test]
    fn test_sidak_keeps_tiny_pvalues() {
        let adjusted = sidak(&[1e-300, 0.5]);

        assert_relative_eq!(adjusted[0], 2e-300, max_relative = 1e-10);
        assert!(adjusted[0] > 0.0);
    }

    #[test]
    fn test_sidak_stays_in_range() {
        let adjusted = sidak(&[0.0, 1.0, 0.9, 0.9, 0.9]);

        assert!(adjusted.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert_eq!(adjusted[0], 0.0);
        assert_eq!(adjusted[1], 1.0);
    }

    ///////////////////
    // Tail and NES //
    ///////////////////

    fn balanced_tail() -> BlitzGseaTail {
        BlitzGseaTail {
            shape_pos: 4.0,
            scale_pos: 0.05,
            shape_neg: 4.0,
            scale_neg: 0.05,
            pos_ratio: 0.5,
        }
    }

    /// A positive score has to give a positive NES and a negative one a
    /// negative NES. Getting this backwards is the easy mistake, because the
    /// reference computes an inverse survival function and then negates it.
    #[test]
    fn test_nes_follows_the_sign_of_the_score() {
        let tail = balanced_tail();

        let (p_pos, nes_pos) = pval_and_nes(0.6, &tail).unwrap();
        let (p_neg, nes_neg) = pval_and_nes(-0.6, &tail).unwrap();

        assert!(nes_pos > 0.0, "got {nes_pos}");
        assert!(nes_neg < 0.0, "got {nes_neg}");
        // a symmetric null must treat the two mirror scores identically
        assert_relative_eq!(p_pos, p_neg, max_relative = 1e-12);
        assert_relative_eq!(nes_pos, -nes_neg, max_relative = 1e-12);
    }

    #[test]
    fn test_stronger_scores_give_smaller_pvalues() {
        let tail = balanced_tail();

        let (weak, nes_weak) = pval_and_nes(0.25, &tail).unwrap();
        let (strong, nes_strong) = pval_and_nes(0.8, &tail).unwrap();

        assert!(strong < weak);
        assert!(nes_strong > nes_weak);
    }

    /// A score deep enough to underflow the gamma tail must still come back
    /// with a finite NES rather than an infinity.
    #[test]
    fn test_extreme_score_stays_finite() {
        let (pval, nes) = pval_and_nes(50.0, &balanced_tail()).unwrap();

        assert!(nes.is_finite() && nes > 30.0, "got {nes}");
        assert!(pval >= 0.0);
    }

    /// A score at the very centre of the null exhausts the clamp, giving no
    /// evidence in either direction.
    #[test]
    fn test_uninformative_score_clamps_to_one() {
        let tail = BlitzGseaTail {
            pos_ratio: 1.0,
            ..balanced_tail()
        };

        let (pval, nes) = pval_and_nes(1e-9, &tail).unwrap();

        assert_relative_eq!(pval, 1.0, max_relative = 1e-12);
        assert_relative_eq!(nes, 0.0, epsilon = 1e-12);
    }

    /// The ratio splits the two-sided mass, so a null with no negative scores
    /// leaves a positive score twice as significant.
    #[test]
    fn test_pos_ratio_scales_the_tail() {
        let balanced = pval_and_nes(0.5, &balanced_tail()).unwrap().0;
        let all_positive = pval_and_nes(
            0.5,
            &BlitzGseaTail {
                pos_ratio: 1.0,
                ..balanced_tail()
            },
        )
        .unwrap()
        .0;

        assert_relative_eq!(all_positive, 2.0 * balanced, max_relative = 1e-12);
    }

    #[test]
    fn test_tail_at_interpolates_and_clamps_the_ratio() {
        let null = BlitzGseaNull {
            anchor_sizes: vec![10.0, 20.0],
            shape_pos: vec![1.0, 3.0],
            scale_pos: vec![0.1, 0.3],
            shape_neg: vec![2.0, 4.0],
            scale_neg: vec![0.2, 0.4],
            // the smoother can push the ratio outside [0, 1]
            pos_ratio: vec![-0.2, 1.4],
            ks_pos: 1.0,
            ks_neg: 1.0,
            centred: true,
        };

        // log space: 15 sits at ln(1.5) / ln(2) of the way from 10 to 20
        let t = (15.0f64.ln() - 10.0f64.ln()) / (20.0f64.ln() - 10.0f64.ln());
        let tail = null.tail_at(15.0);

        assert_relative_eq!(tail.shape_pos, 1.0 + 2.0 * t, max_relative = 1e-12);
        assert_relative_eq!(tail.scale_neg, 0.2 + 0.2 * t, max_relative = 1e-12);
        assert!((0.0..=1.0).contains(&tail.pos_ratio));
        assert!((0.0..=1.0).contains(&null.tail_at(0.0).pos_ratio));
        assert!((0.0..=1.0).contains(&null.tail_at(100.0).pos_ratio));
    }

    /// `new` and `Default` have to agree, since one routes through the other.
    #[test]
    fn test_params_new_matches_default() {
        let defaults = BlitzGseaParams::default();
        let explicit = BlitzGseaParams::new(None, None, None, None, None, None);

        assert_eq!(defaults.permutations, explicit.permutations);
        assert_eq!(defaults.anchors, explicit.anchors);
        assert_eq!(defaults.symmetric, explicit.symmetric);
        assert_eq!(defaults.centre, explicit.centre);
        assert_eq!(defaults.ks_test, explicit.ks_test);
        assert_eq!(defaults.seed, explicit.seed);
    }

    /// A supplied field overrides its default and the rest stay put.
    #[test]
    fn test_params_new_overrides_only_what_is_given() {
        let params = BlitzGseaParams::new(Some(500), None, Some(true), None, None, Some(7));

        assert_eq!(params.permutations, 500);
        assert_eq!(params.anchors, BLITZ_DEFAULT_ANCHORS);
        assert!(params.symmetric);
        assert!(params.centre);
        assert!(params.ks_test);
        assert_eq!(params.seed, 7);
    }

    /////////////////
    // Calibration //
    /////////////////

    #[test]
    fn test_calibration_shapes_and_ranges() {
        let stats = signature(2_000);

        let null = calibrate_null(&stats, &small_params()).unwrap();

        let n = null.anchor_sizes.len();
        assert!(n > 5);
        assert_eq!(null.shape_pos.len(), n);
        assert_eq!(null.scale_pos.len(), n);
        assert_eq!(null.shape_neg.len(), n);
        assert_eq!(null.scale_neg.len(), n);
        assert_eq!(null.pos_ratio.len(), n);

        assert!(null.shape_pos.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(null.scale_pos.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(null.anchor_sizes.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_calibration_is_reproducible() {
        let stats = signature(1_500);
        let params = small_params();

        let first = calibrate_null(&stats, &params).unwrap();
        let second = calibrate_null(&stats, &params).unwrap();

        for (a, b) in first.shape_pos.iter().zip(second.shape_pos.iter()) {
            assert_relative_eq!(*a, *b, max_relative = 1e-15);
        }
        assert_relative_eq!(first.ks_pos, second.ks_pos, max_relative = 1e-15);
    }

    /// Below `BLITZ_MIN_PERMUTATIONS_SPLIT` the tails are pooled whatever the
    /// caller asked for, so both sides come back with identical parameters.
    #[test]
    fn test_low_permutation_count_forces_pooled_tails() {
        let stats = signature(1_000);
        let params = BlitzGseaParams {
            permutations: 100,
            anchors: 6,
            symmetric: false,
            ..Default::default()
        };

        let null = calibrate_null(&stats, &params).unwrap();

        for (pos, neg) in null.shape_pos.iter().zip(null.shape_neg.iter()) {
            assert_relative_eq!(*pos, *neg, max_relative = 1e-12);
        }
    }

    /// A one-gene signature leaves no set size with at least one hit and one
    /// miss, so there is nothing to calibrate against.
    #[test]
    fn test_calibration_rejects_an_impossible_grid() {
        let stats = vec![1.0_f64];

        assert!(calibrate_null(&stats, &small_params()).is_err());
    }

    //////////////
    // Scoring //
    //////////////

    #[test]
    fn test_enriched_set_beats_a_spread_out_one() {
        let stats = signature(2_000);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        // 50 genes off the very top of the ranking
        let top: Vec<i32> = (0..50).collect();
        // 50 genes spread evenly across the whole ranking
        let spread: Vec<i32> = (0..50).map(|i| i * 40).collect();

        let res = blitzgsea_score(&stats, &[top, spread], &null, &params, false).unwrap();

        assert!(res.es[0] > 0.5, "top set scored {}", res.es[0]);
        assert!(res.es[0] > res.es[1]);
        assert!(res.pvals[0] < res.pvals[1]);
        assert!(res.nes[0] > res.nes[1]);
    }

    /// A set off the bottom of the ranking has to come back negative on both
    /// the score and the NES.
    #[test]
    fn test_depleted_set_scores_negative() {
        let stats = signature(2_000);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let bottom: Vec<i32> = (1_950..2_000).collect();

        let res = blitzgsea_score(&stats, &[bottom], &null, &params, false).unwrap();

        assert!(res.es[0] < -0.5, "got {}", res.es[0]);
        assert!(res.nes[0] < 0.0);
        assert!(!res.leading_edge[0].is_empty());
    }

    #[test]
    fn test_scoring_result_columns_line_up() {
        let stats = signature(1_200);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let pathways: Vec<Vec<i32>> = vec![(0..20).collect(), (100..140).collect()];

        let res = blitzgsea_score(&stats, &pathways, &null, &params, false).unwrap();

        assert_eq!(res.es.len(), 2);
        assert_eq!(res.nes.len(), 2);
        assert_eq!(res.pvals.len(), 2);
        assert_eq!(res.sidak.len(), 2);
        assert_eq!(res.fdr.len(), 2);
        assert_eq!(res.size, vec![20, 40]);
        assert_eq!(res.leading_edge.len(), 2);

        assert!(res.pvals.iter().all(|p| (0.0..=1.0).contains(p)));
        assert!(res.fdr.iter().zip(res.pvals.iter()).all(|(f, p)| f >= p));
        assert!(res.nes.iter().all(|n| n.is_finite()));
    }

    /// One-based indices from R must land on the same genes as zero-based ones.
    #[test]
    fn test_one_indexed_matches_zero_indexed() {
        let stats = signature(1_000);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let zero: Vec<i32> = (0..30).collect();
        let one: Vec<i32> = (1..=30).collect();

        let from_zero = blitzgsea_score(&stats, &[zero], &null, &params, false).unwrap();
        let from_one = blitzgsea_score(&stats, &[one], &null, &params, true).unwrap();

        assert_relative_eq!(from_zero.es[0], from_one.es[0], max_relative = 1e-12);
    }

    #[test]
    fn test_empty_pathway_list() {
        let stats = signature(500);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let res = blitzgsea_score(&stats, &[], &null, &params, false).unwrap();

        assert!(res.es.is_empty());
        assert!(res.fdr.is_empty());
    }

    /////////////////////
    // Input hygiene //
    /////////////////////

    /// A gene set with nothing left after intersecting with the signature used
    /// to reach `array_max` on an empty slice and panic, which aborts the R
    /// session rather than returning.
    #[test]
    fn test_empty_pathway_errors_rather_than_panicking() {
        let stats = signature(500);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let res = blitzgsea_score(&stats, &[vec![]], &null, &params, false);

        assert!(res.is_err());
    }

    /// `calc_gsea_stats` counts each hit's misses from its position, so an
    /// unsorted list silently scores differently. The same genes in any order
    /// have to give the same answer.
    #[test]
    fn test_unsorted_indices_score_the_same() {
        let stats = signature(1_000);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let sorted = vec![1, 5, 9, 40, 100];
        let shuffled = vec![100, 5, 40, 1, 9];

        let a = blitzgsea_score(&stats, &[sorted], &null, &params, false).unwrap();
        let b = blitzgsea_score(&stats, &[shuffled], &null, &params, false).unwrap();

        assert_relative_eq!(a.es[0], b.es[0], max_relative = 1e-12);
        assert_eq!(a.size[0], b.size[0]);
    }

    /// A repeated gene is one gene. Left in, it inflates both the score and the
    /// reported set size.
    #[test]
    fn test_duplicate_indices_are_collapsed() {
        let stats = signature(1_000);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let once = blitzgsea_score(&stats, &[vec![1, 5, 9]], &null, &params, false).unwrap();
        let twice =
            blitzgsea_score(&stats, &[vec![1, 1, 5, 5, 9, 9]], &null, &params, false).unwrap();

        assert_eq!(twice.size[0], 3);
        assert_relative_eq!(once.es[0], twice.es[0], max_relative = 1e-12);
    }

    /// Scoring against a null calibrated with the other centring choice puts the
    /// enrichment scores on a scale the null was never fitted to.
    #[test]
    fn test_centring_mismatch_is_rejected() {
        let stats = signature(500);
        let calibrated = BlitzGseaParams {
            centre: true,
            ..small_params()
        };
        let null = calibrate_null(&stats, &calibrated).unwrap();

        let scoring = BlitzGseaParams {
            centre: false,
            ..small_params()
        };
        let res = blitzgsea_score(&stats, &[vec![1, 2, 3]], &null, &scoring, false);

        assert!(res.is_err());
    }

    /// The parameter vectors are read in lockstep with the anchor grid, and the
    /// fields are public, so a short one has to be caught before it indexes.
    #[test]
    fn test_malformed_null_is_rejected() {
        let stats = signature(500);
        let params = small_params();
        let mut null = calibrate_null(&stats, &params).unwrap();

        assert!(null.validate().is_ok());

        null.shape_pos.pop();
        assert!(null.validate().is_err());
        assert!(blitzgsea_score(&stats, &[vec![1, 2, 3]], &null, &params, false).is_err());
    }

    /// One anchor gives a size-independent null pinned to `k = 1`, whose score
    /// is purely positional. Silently returning that is worse than refusing.
    #[test]
    fn test_too_few_anchors_is_rejected() {
        let stats = signature(1_000);

        for anchors in [0, 1] {
            let params = BlitzGseaParams {
                anchors,
                ..small_params()
            };
            assert!(
                calibrate_null(&stats, &params).is_err(),
                "anchors = {anchors} should be rejected"
            );
        }
    }

    /// Zero permutations leaves nothing to fit and one leaves no spread; both
    /// otherwise surface as a confusing error from inside the gamma fitter.
    #[test]
    fn test_too_few_permutations_is_rejected() {
        let stats = signature(1_000);

        for permutations in [0, 1] {
            let params = BlitzGseaParams {
                permutations,
                ..small_params()
            };
            assert!(
                calibrate_null(&stats, &params).is_err(),
                "permutations = {permutations} should be rejected"
            );
        }
    }

    /// The ratio endpoints drive the one-sided probability to exactly zero,
    /// which is the clamp path rather than the gamma-underflow one.
    #[test]
    fn test_ratio_endpoints_stay_finite() {
        let all_negative = BlitzGseaTail {
            pos_ratio: 0.0,
            ..balanced_tail()
        };
        let all_positive = BlitzGseaTail {
            pos_ratio: 1.0,
            ..balanced_tail()
        };

        let (p_pos, nes_pos) = pval_and_nes(0.5, &all_negative).unwrap();
        let (p_neg, nes_neg) = pval_and_nes(-0.5, &all_positive).unwrap();

        assert_eq!(p_pos, 0.0);
        assert_eq!(p_neg, 0.0);
        assert!(nes_pos.is_finite() && nes_pos > 30.0);
        assert!(nes_neg.is_finite() && nes_neg < -30.0);
    }

    /// A gene set larger than the top anchor is extrapolated. The parameters
    /// have to stay positive, or `gamma_sf` errors and takes the whole batch
    /// down with it.
    #[test]
    fn test_extrapolated_tail_stays_usable() {
        let stats = signature(2_000);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let top = *null
            .anchor_sizes
            .last()
            .expect("calibration returns anchors");

        for size in [top + 1.0, top * 1.5, stats.len() as f64 - 1.0] {
            let tail = null.tail_at(size);
            assert!(
                tail.shape_pos > 0.0,
                "shape at {size} is {}",
                tail.shape_pos
            );
            assert!(
                tail.scale_pos > 0.0,
                "scale at {size} is {}",
                tail.scale_pos
            );
            assert!(tail.shape_neg > 0.0 && tail.scale_neg > 0.0);
            assert!(pval_and_nes(0.3, &tail).is_ok());
        }
    }

    /// A size below one has no null; it is clamped rather than taking `ln` of
    /// zero and extrapolating to nonsense.
    #[test]
    fn test_tail_at_clamps_sizes_below_one() {
        let stats = signature(500);
        let params = small_params();
        let null = calibrate_null(&stats, &params).unwrap();

        let one = null.tail_at(1.0);
        for size in [0.0, -5.0, 0.5] {
            let tail = null.tail_at(size);
            assert_relative_eq!(tail.shape_pos, one.shape_pos, max_relative = 1e-12);
        }
    }
}
