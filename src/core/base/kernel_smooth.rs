//! Kernel density bandwidths and Nadaraya-Watson smoothing, matching R.
//!
//! Three R routines that scTransform's regularisation step leans on and that
//! this crate did not have: `bw.nrd`, `bw.SJ` and `ksmooth(kernel = "normal")`.
//! All three are ports rather than reimplementations, down to R's binning
//! quirks, because the point is that a regularised model fitted here lands
//! where the R one lands.
//!
//! Everything is `f64`. These run on gene-length vectors, a few thousand
//! elements, so the memory argument for narrower floats does not apply, and
//! `bw.SJ` in particular sums fourth and sixth derivatives of a Gaussian
//! kernel over pair counts, which `f32` would shred.

use crate::core::math::optimise::{UNIROOT_MAXIT, zeroin};
use crate::core::math::vector_helpers::{quantile, standard_deviation};
use crate::errors::BixverseErrors;

////////////
// Consts //
////////////

/// Number of bins R's `bw.SJ` uses for the pair-distance counts.
///
/// R's `nb` argument, default `1000L`. It also decides which counting path is
/// taken: binned when `n > nb / 2`, exact pairwise below that.
const BW_NB: usize = 1000;

/// Squared-distance cutoff in R's `bw_phi4` / `bw_phi6`.
///
/// `exp(-delta / 2)` has underflowed to zero long before this, so the loop
/// breaks rather than walking the remaining bins.
const BW_DELMAX: f64 = 1000.0;

/// `sqrt(2 * pi)`, the Gaussian normalising constant in `bw_phi4` / `bw_phi6`.
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Denominator R's `bw.nrd` divides the interquartile range by.
///
/// Note this is `1.34`, where `bw.SJ` uses `1.349`. The discrepancy is R's, not
/// a typo here.
const BW_NRD_IQR_SCALE: f64 = 1.34;

/// Denominator R's `bw.SJ` divides the interquartile range by.
const BW_SJ_IQR_SCALE: f64 = 1.349;

/// Interval-widening factor in `bw.SJ`'s bracket search.
const BW_SJ_BRACKET_MULT: f64 = 1.2;

/// Iteration cap on `bw.SJ`'s bracket search, R's `itry > 99L`.
const BW_SJ_BRACKET_MAX_TRY: usize = 99;

/// Bandwidth rescaling for `ksmooth(kernel = "normal")`.
///
/// R documents its `bandwidth` as the width of the kernel's interquartile
/// range, so the Gaussian standard deviation is this multiple of it.
const KSMOOTH_NORMAL_SCALE: f64 = 0.370_650_6;

/// Kernel support cutoff in `ksmooth(kernel = "normal")`, in rescaled
/// bandwidths.
const KSMOOTH_NORMAL_CUTOFF: f64 = 4.0;

///////////////////////
// Simple bandwidths //
///////////////////////

/// Scott's rule-of-thumb kernel bandwidth, R's `bw.nrd`.
///
/// `1.06 * min(sd, IQR / 1.34) * n^(-1/5)`. This is what `density(bw = 'nrd')`
/// uses.
///
/// ### Params
///
/// * `x` - The sample.
///
/// ### Returns
///
/// The bandwidth, or [`BixverseErrors::BandwidthTooFewPoints`] when fewer than
/// two points are supplied.
pub fn bw_nrd(x: &[f64]) -> Result<f64, BixverseErrors> {
    if x.len() < 2 {
        return Err(BixverseErrors::BandwidthTooFewPoints { found: x.len() });
    }
    let n = x.len() as f64;
    let iqr = quantile(x, 0.75_f64) - quantile(x, 0.25_f64);
    let scale = standard_deviation(x).min(iqr / BW_NRD_IQR_SCALE);

    Ok(1.06 * scale * n.powf(-0.2))
}

///////////
// bw.SJ //
///////////

/// Binned pair-distance counts, R's `stats:::bw_pair_cnts`.
///
/// Returns the bin width and, at index `i`, the number of unordered pairs whose
/// bin indices differ by `i`. Two paths, picked exactly as R picks them.
///
/// The binned path reproduces R's `trunc(abs(x) / d) * sign(x)` literally,
/// including the fact that it truncates towards zero rather than binning off
/// the minimum. That makes the bin edges asymmetric around zero, which matters
/// here because `log10` gene means straddle it. It is R's behaviour and the
/// reference values come from R.
///
/// ### Params
///
/// * `x` - The sample.
/// * `nb` - Number of bins.
///
/// ### Returns
///
/// `(bin_width, counts)` with `counts.len() == nb`.
fn bw_pair_cnts(x: &[f64], nb: usize) -> (f64, Vec<f64>) {
    let n = x.len();
    let (min, max) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let d = (max - min) * 1.01 / nb as f64;
    let mut cnt = vec![0.0_f64; nb];

    if n > nb / 2 {
        // R: xx <- trunc(abs(x) / d) * sign(x); xx <- xx - min(xx) + 1
        let raw: Vec<f64> = x
            .iter()
            .map(|&v| (v.abs() / d).trunc() * v.signum())
            .collect();
        let raw_min = raw.iter().cloned().fold(f64::INFINITY, f64::min);

        // R: tabulate(xx, nb) drops anything outside 1..=nb.
        let mut bins = vec![0.0_f64; nb];
        for &r in &raw {
            let idx = r - raw_min + 1.0;
            if idx >= 1.0 && idx <= nb as f64 {
                bins[idx as usize - 1] += 1.0;
            }
        }

        // R: C_bw_den_binned, pair counts by bin separation.
        for ii in 0..nb {
            let w = bins[ii];
            cnt[0] += w * (w - 1.0) / 2.0;
            for jj in 0..ii {
                cnt[ii - jj] += w * bins[jj];
            }
        }
    } else {
        // R: C_bw_den, exact pairwise over the same bin grid. Note the C cast
        // truncates towards zero, so this is not `floor` for negative values.
        for i in 1..n {
            let ii = (x[i] / d) as i64;
            for j in 0..i {
                let jj = (x[j] / d) as i64;
                let iij = (ii - jj).unsigned_abs() as usize;
                if iij < nb {
                    cnt[iij] += 1.0;
                }
            }
        }
    }

    (d, cnt)
}

/// Binned estimate of the integrated squared fourth derivative, R's `bw_phi4`.
///
/// ### Params
///
/// * `n` - Sample size.
/// * `d` - Bin width from [`bw_pair_cnts`].
/// * `cnt` - Pair counts from [`bw_pair_cnts`].
/// * `h` - Pilot bandwidth.
///
/// ### Returns
///
/// The functional estimate.
fn bw_phi4(n: usize, d: f64, cnt: &[f64], h: f64) -> f64 {
    let mut sum = 0.0;
    for (i, &c) in cnt.iter().enumerate() {
        let delta = (i as f64 * d / h).powi(2);
        if delta >= BW_DELMAX {
            break;
        }
        sum += (-delta / 2.0).exp() * (delta * delta - 6.0 * delta + 3.0) * c;
    }
    let n_f = n as f64;
    // Doubled for the unordered pairs, plus the n diagonal terms at delta = 0.
    let sum = 2.0 * sum + n_f * 3.0;

    sum / (n_f * (n_f - 1.0) * h.powi(5) * SQRT_2PI)
}

/// Binned estimate of the integrated squared sixth derivative, R's `bw_phi6`.
///
/// ### Params
///
/// * `n` - Sample size.
/// * `d` - Bin width from [`bw_pair_cnts`].
/// * `cnt` - Pair counts from [`bw_pair_cnts`].
/// * `h` - Pilot bandwidth.
///
/// ### Returns
///
/// The functional estimate.
fn bw_phi6(n: usize, d: f64, cnt: &[f64], h: f64) -> f64 {
    let mut sum = 0.0;
    for (i, &c) in cnt.iter().enumerate() {
        let delta = (i as f64 * d / h).powi(2);
        if delta >= BW_DELMAX {
            break;
        }
        sum += (-delta / 2.0).exp()
            * (delta * delta * delta - 15.0 * delta * delta + 45.0 * delta - 15.0)
            * c;
    }
    let n_f = n as f64;
    // The diagonal term of the sixth-derivative kernel is -15, not 3.
    let sum = 2.0 * sum - 15.0 * n_f;

    sum / (n_f * (n_f - 1.0) * h.powi(7) * SQRT_2PI)
}

/// Sheather-Jones solve-the-equation kernel bandwidth, R's `bw.SJ`.
///
/// A port of `stats::bw.SJ(x, method = "ste")` with R's defaults: `nb = 1000`,
/// `lower = 0.1 * hmax`, `upper = hmax`, `tol = 0.1 * lower`. The bracket is
/// widened alternately upwards and downwards, as R does, before handing the
/// root to [`zeroin`].
///
/// ### Params
///
/// * `x` - The sample.
///
/// ### Returns
///
/// The bandwidth, or a [`BixverseErrors`] when the sample is too small, too
/// sparse for the pilot functionals, or no root can be bracketed.
///
/// ### References
///
/// Sheather & Jones, Journal of the Royal Statistical Society B, 1991
pub fn bw_sj(x: &[f64]) -> Result<f64, BixverseErrors> {
    let n = x.len();
    if n < 2 {
        return Err(BixverseErrors::BandwidthTooFewPoints { found: n });
    }

    let (d, cnt) = bw_pair_cnts(x, BW_NB);

    let iqr = quantile(x, 0.75_f64) - quantile(x, 0.25_f64);
    let scale = standard_deviation(x).min(iqr / BW_SJ_IQR_SCALE);
    let n_f = n as f64;

    let a = 1.24 * scale * n_f.powf(-1.0 / 7.0);
    let b = 1.23 * scale * n_f.powf(-1.0 / 9.0);
    let c1 = 1.0 / (2.0 * std::f64::consts::PI.sqrt() * n_f);

    let sd_h = |h: f64| bw_phi4(n, d, &cnt, h);
    let td = -bw_phi6(n, d, &cnt, b);

    if !td.is_finite() || td <= 0.0 {
        return Err(BixverseErrors::BandwidthTooSparse { stage: "TD" });
    }

    let alph2 = 1.357 * (sd_h(a) / td).powf(1.0 / 7.0);
    if !alph2.is_finite() {
        return Err(BixverseErrors::BandwidthTooSparse { stage: "alph2" });
    }

    let f_sd = |h: f64| (c1 / sd_h(alph2 * h.powf(5.0 / 7.0))).powf(0.2) - h;

    let hmax = 1.144 * scale * n_f.powf(-0.2);
    let (mut lower, mut upper) = (0.1 * hmax, hmax);
    let tol = 0.1 * lower;

    let mut itry = 1_usize;
    while f_sd(lower) * f_sd(upper) > 0.0 {
        if itry > BW_SJ_BRACKET_MAX_TRY {
            return Err(BixverseErrors::BandwidthNoBracket);
        }
        if itry % 2 == 1 {
            upper *= BW_SJ_BRACKET_MULT;
        } else {
            lower /= BW_SJ_BRACKET_MULT;
        }
        itry += 1;
    }

    zeroin(
        lower,
        upper,
        f_sd(lower),
        f_sd(upper),
        f_sd,
        tol,
        UNIROOT_MAXIT,
    )
    .ok_or(BixverseErrors::BandwidthNoBracket)
}

//////////////
// ksmooth //
//////////////

/// Nadaraya-Watson kernel regression with a Gaussian kernel, R's
/// `ksmooth(kernel = "normal")`.
///
/// `bandwidth` is the kernel's interquartile width, as R documents it, so the
/// Gaussian standard deviation used internally is `bandwidth` times
/// [`KSMOOTH_NORMAL_SCALE`]. Points further than [`KSMOOTH_NORMAL_CUTOFF`]
/// rescaled bandwidths away are dropped, which is what keeps the sweep linear
/// rather than quadratic.
///
/// Both `x` and `x_points` are sorted internally, and the returned values are
/// in **sorted `x_points` order**, matching R. Callers that need the original
/// ordering back must reorder, exactly as scTransform's `model_pars_fit[o, i]`
/// assignment does.
///
/// ### Params
///
/// * `x` - Predictor values.
/// * `y` - Response values, same length as `x`.
/// * `x_points` - Points to evaluate at.
/// * `bandwidth` - Kernel interquartile width.
///
/// ### Returns
///
/// `(sorted_x_points, fitted)`, where an entry of `fitted` is `f64::NAN` when
/// no data point fell inside the kernel support, or
/// [`BixverseErrors::LengthMismatch`] when `x` and `y` disagree.
pub fn ksmooth_normal(
    x: &[f64],
    y: &[f64],
    x_points: &[f64],
    bandwidth: f64,
) -> Result<(Vec<f64>, Vec<f64>), BixverseErrors> {
    if x.len() != y.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "y",
            expected: x.len(),
            found: y.len(),
        });
    }

    let mut ord: Vec<usize> = (0..x.len()).collect();
    ord.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    let xs: Vec<f64> = ord.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = ord.iter().map(|&i| y[i]).collect();

    let mut xp = x_points.to_vec();
    xp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let bw = bandwidth * KSMOOTH_NORMAL_SCALE;
    let cutoff = KSMOOTH_NORMAL_CUTOFF * bw;

    let n = xs.len();
    let mut out = Vec::with_capacity(xp.len());

    // `imin` only ever advances, so the whole sweep is linear in `n + np`
    // rather than quadratic. This is why both inputs have to be sorted.
    let mut imin = 0_usize;
    if let Some(&first) = xp.first() {
        while imin < n && xs[imin] < first - cutoff {
            imin += 1;
        }
    }

    for &x0 in &xp {
        let (mut num, mut den) = (0.0, 0.0);
        // The C original assigns to `imin` inside the loop it is the bound of,
        // which only ever affects the next evaluation point. Kept as a separate
        // variable so that is obvious rather than looking like a live bound.
        let mut next_imin = imin;
        for i in imin..n {
            if xs[i] < x0 - cutoff {
                next_imin = i;
            } else {
                if xs[i] > x0 + cutoff {
                    break;
                }
                let u = (xs[i] - x0).abs() / bw;
                let w = (-0.5 * u * u).exp();
                num += w * ys[i];
                den += w;
            }
        }
        imin = next_imin;
        out.push(if den > 0.0 { num / den } else { f64::NAN });
    }

    Ok((xp, out))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Numerical Recipes LCG, the same one `dev/gen_edger_fixtures.R` uses.
    ///
    /// Reference samples are rebuilt on both sides rather than crossing as
    /// text, so nothing here depends on float formatting.
    struct Lcg(u64);

    impl Lcg {
        fn new() -> Self {
            Lcg(20_260_101)
        }

        fn next_unif(&mut self) -> f64 {
            self.0 = (1_664_525_u64
                .wrapping_mul(self.0)
                .wrapping_add(1_013_904_223))
                % 4_294_967_296;
            self.0 as f64 / 4_294_967_296.0
        }

        fn unifs(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.next_unif()).collect()
        }
    }

    /// The binned `bw.SJ` case, n = 2000 over a sign-straddling range.
    fn sample_x1(lcg: &mut Lcg) -> Vec<f64> {
        lcg.unifs(2000).iter().map(|u| u * 6.0 - 2.5).collect()
    }

    /// The exact-pairwise `bw.SJ` case, n = 400 < nb / 2.
    fn sample_x2(lcg: &mut Lcg) -> Vec<f64> {
        lcg.unifs(400).iter().map(|u| u * 3.0 - 1.0).collect()
    }

    /// Deterministic grid, for `ksmooth`.
    fn sample_x3() -> Vec<f64> {
        (0..137).map(|i| -2.0 + 5.0 * i as f64 / 136.0).collect()
    }

    /// The LCG has to agree with R's before any bandwidth comparison means
    /// anything, so pin the sample itself first.
    #[test]
    fn test_lcg_sample_matches_r() {
        let mut lcg = Lcg::new();
        let x1 = sample_x1(&mut lcg);
        let x2 = sample_x2(&mut lcg);

        assert_relative_eq!(x1[0], -1.975_693_717_598_915, max_relative = 1e-15);
        assert_relative_eq!(x1[1999], -0.218_262_553_680_688_14, max_relative = 1e-15);
        assert_relative_eq!(
            x1.iter().sum::<f64>(),
            973.903_402_473_777_5,
            max_relative = 1e-13
        );
        assert_relative_eq!(x2[0], 1.219_621_244_817_972_2, max_relative = 1e-15);
        assert_relative_eq!(x2[399], 1.157_794_128_870_591_5, max_relative = 1e-15);
        assert_relative_eq!(
            x2.iter().sum::<f64>(),
            227.239_331_772_550_94,
            max_relative = 1e-13
        );
    }

    /// R: bw.nrd(x) on all three samples.
    #[test]
    fn test_bw_nrd_matches_r() {
        let mut lcg = Lcg::new();
        let x1 = sample_x1(&mut lcg);
        let x2 = sample_x2(&mut lcg);
        let x3 = sample_x3();

        assert_relative_eq!(
            bw_nrd(&x1).unwrap(),
            0.402_879_637_405_405_4,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            bw_nrd(&x2).unwrap(),
            0.273_081_477_025_925_6,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            bw_nrd(&x3).unwrap(),
            0.578_231_905_756_761_5,
            max_relative = 1e-12
        );
    }

    /// R: bw.SJ(x). `x1` takes the binned path (n > nb / 2), `x2` and `x3` the
    /// exact pairwise one, so both arms of [`bw_pair_cnts`] are covered.
    #[test]
    fn test_bw_sj_matches_r() {
        let mut lcg = Lcg::new();
        let x1 = sample_x1(&mut lcg);
        let x2 = sample_x2(&mut lcg);
        let x3 = sample_x3();

        assert_relative_eq!(
            bw_sj(&x1).unwrap(),
            0.206_287_864_703_460_47,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            bw_sj(&x2).unwrap(),
            0.173_954_641_288_164_04,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            bw_sj(&x3).unwrap(),
            0.485_858_100_468_439_2,
            max_relative = 1e-10
        );
    }

    /// R: ksmooth(x3, y3, kernel = "normal", bandwidth = 0.8, x.points = xp).
    #[test]
    fn test_ksmooth_normal_matches_r() {
        let x3 = sample_x3();
        let y3: Vec<f64> = x3
            .iter()
            .map(|&x| x.sin() + 0.1 * (5.0 * x).cos())
            .collect();
        let xp: Vec<f64> = (0..11).map(|i| -1.5 + 4.0 * i as f64 / 10.0).collect();

        let expected = [
            -0.946_236_559_236_919,
            -0.829_228_441_342_057_6,
            -0.647_733_021_991_770_6,
            -0.280_470_089_304_166_7,
            0.124_776_709_423_778_93,
            0.432_132_628_371_082_9,
            0.742_643_732_345_198,
            0.954_687_679_698_398_2,
            0.929_003_922_551_888_9,
            0.811_078_648_675_108_5,
            0.634_815_071_281_249,
        ];

        let (_, got) = ksmooth_normal(&x3, &y3, &xp, 0.8).unwrap();
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*g, *e, max_relative = 1e-12);
        }
    }

    /// Unsorted `x_points` come back sorted, as R's `ksmooth` returns them.
    /// The caller is responsible for reordering, which is what scTransform's
    /// `model_pars_fit[o, i]` assignment does.
    #[test]
    fn test_ksmooth_normal_sorts_x_points() {
        let x3 = sample_x3();
        let y3: Vec<f64> = x3
            .iter()
            .map(|&x| x.sin() + 0.1 * (5.0 * x).cos())
            .collect();

        let (xs, got) = ksmooth_normal(&x3, &y3, &[2.0, -1.0, 0.5], 0.5).unwrap();

        assert_eq!(xs, vec![-1.0, 0.5, 2.0]);
        let expected = [
            -0.808_685_303_690_512,
            0.419_113_609_237_200_04,
            0.839_211_048_088_081_7,
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*g, *e, max_relative = 1e-12);
        }
    }

    /// Points with no data inside the kernel support come back as NaN, not as
    /// a silent zero.
    #[test]
    fn test_ksmooth_normal_empty_support_is_nan() {
        let x = [0.0, 0.1, 0.2];
        let y = [1.0, 2.0, 3.0];

        let (_, got) = ksmooth_normal(&x, &y, &[100.0], 0.01).unwrap();

        assert!(got[0].is_nan());
    }

    #[test]
    fn test_bandwidths_reject_short_input() {
        assert!(matches!(
            bw_nrd(&[1.0]),
            Err(BixverseErrors::BandwidthTooFewPoints { found: 1 })
        ));
        assert!(matches!(
            bw_sj(&[1.0]),
            Err(BixverseErrors::BandwidthTooFewPoints { found: 1 })
        ));
    }

    #[test]
    fn test_ksmooth_normal_rejects_length_mismatch() {
        assert!(matches!(
            ksmooth_normal(&[1.0, 2.0], &[1.0], &[1.5], 0.5),
            Err(BixverseErrors::LengthMismatch { .. })
        ));
    }
}
