//! Distribution tails: normal, gamma, chi-squared, Student's t and F.
//!
//! Plain functions rather than distribution objects, because every caller here
//! wants a tail probability and there is nothing worth carrying state for.
//!
//! ### Survival functions, not `1 - cdf`
//!
//! `1 - cdf` is exact only down to about 1e-16 and returns a flat `0.0` below
//! that, so `chisq_sf(200, 1)` would come back as zero instead of 2.09e-45.
//! Each `_sf` here goes through the complement directly: the upper regularised
//! incomplete gamma for the normal and chi-squared, and the regularised
//! incomplete beta with its arguments swapped for t and F. None of them
//! subtracts from one. `statrs`'s own `FisherSnedecor::sf` does, which is why
//! `f_sf` is here rather than a call into it.
//!
//! Ported from `edge-rs` (`src/numeric/dist.rs`), trimmed to the tails this
//! crate actually reads.

use statrs::function::beta::beta_reg;
use statrs::function::erf::erfc_inv;
use statrs::function::gamma::ln_gamma;

use crate::errors::BixverseErrors;

////////////
// Consts //
////////////

/// Relative convergence tolerance for the incomplete gamma recurrences. Both
/// reach it in a few tens of terms over the range used here.
const INC_GAMMA_EPS: f64 = f64::EPSILON;

/// Iteration budget for the incomplete gamma recurrences.
///
/// The series needs roughly `sqrt(a)` terms near the mean, so this covers
/// shapes into the millions. It is a runaway guard, not a working limit: on
/// exhaustion the best available partial sum is returned rather than an error,
/// because every caller here is a tail probability that is already converged to
/// well past `f64` resolution by then.
const INC_GAMMA_MAX_ITER: usize = 10_000;

/// Floor for the modified Lentz continued fraction, guarding a zero pivot.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
const LENTZ_TINY: f64 = 1e-300;

/// Crossover on `x^2` between the two incomplete beta forms of the t tails.
///
/// The mass inside `(0, |x|)` and the mass beyond `|x|` are equal at the upper
/// quartile, which runs from 1.0 at `df = 1` down to 0.4549 as `df` grows. Take
/// the inner form below this and the outer form above it and the quantity being
/// evaluated directly is always the smaller of the two, so the other one is
/// recovered from `0.5 - .` while it is still the larger. Sitting between the
/// two extremes, 0.7 leaves neither branch worse than a factor of about two
/// from balanced for any `df`.
const T_INNER_OUTER_SWITCH: f64 = 0.7;

////////////////
// Validation //
////////////////

/// Checks that a degrees-of-freedom-like parameter is finite and positive.
///
/// ### Params
///
/// * `name` - Parameter name, used verbatim in the error message
/// * `value` - Value supplied by the caller
///
/// ### Returns
///
/// `Ok(())`, or [`BixverseErrors::InvalidArgument`] naming the parameter and
/// value.
fn check_positive(name: &str, value: f64) -> Result<(), BixverseErrors> {
    if !(value.is_finite() && value > 0.0) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "'{name}' must be finite and strictly positive; got {value}."
        )));
    }
    Ok(())
}

//////////////////////////////////
// Regularised incomplete gamma //
//////////////////////////////////

/// The shared prefactor `exp(a ln x - x - ln Gamma(a))` of both branches.
///
/// Formed in logs so it underflows cleanly to zero in the far tail rather than
/// overflowing on `x^a` first.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, strictly positive
///
/// ### Returns
///
/// The prefactor, or 0.0 where the exponent underflows.
fn inc_gamma_prefactor(a: f64, x: f64) -> f64 {
    (a * x.ln() - x - ln_gamma(a)).exp()
}

/// Regularised lower incomplete gamma `P(a, x)`.
///
/// Series below `x = a + 1`, complement of the continued fraction above it,
/// which is where each converges fastest and neither cancels.
///
/// Unlike `statrs::function::gamma::gamma_lr` there is no cutoff at small `x`:
/// the series prefactor is evaluated in logs, so `P(0.3, 1e-30)` comes back as
/// 1e-9 rather than zero.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, non-negative
///
/// ### Returns
///
/// `P(a, x)` in `[0, 1]`.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
fn reg_gamma_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cont_frac(a, x)
    }
}

/// Regularised upper incomplete gamma `Q(a, x)`.
///
/// The continued fraction above `x = a + 1`, evaluated directly, which is what
/// keeps `chisq_sf(1000, 1)` at 1.8e-219 rather than zero.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, non-negative
///
/// ### Returns
///
/// `Q(a, x) = 1 - P(a, x)` in `[0, 1]`.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
fn reg_gamma_upper(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cont_frac(a, x)
    }
}

/// Series representation of `P(a, x)`, for `x < a + 1`.
///
/// `P(a, x) = exp(a ln x - x - ln Gamma(a)) * sum_{n>=0} x^n / (a (a+1)...(a+n))`.
/// Every term is positive, so there is nothing to cancel.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, strictly positive and below `a + 1`
///
/// ### Returns
///
/// `P(a, x)`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut term = 1.0 / a;
    let mut sum = term;
    for _ in 0..INC_GAMMA_MAX_ITER {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * INC_GAMMA_EPS {
            break;
        }
    }
    sum * inc_gamma_prefactor(a, x)
}

/// Continued fraction representation of `Q(a, x)`, for `x >= a + 1`.
///
/// Evaluated by modified Lentz, which is stable where the naive recurrence
/// overflows.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, at least `a + 1`
///
/// ### Returns
///
/// `Q(a, x)`.
fn gamma_cont_frac(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / LENTZ_TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=INC_GAMMA_MAX_ITER {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < LENTZ_TINY {
            d = LENTZ_TINY;
        }
        c = b + an / c;
        if c.abs() < LENTZ_TINY {
            c = LENTZ_TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= INC_GAMMA_EPS {
            break;
        }
    }
    h * inc_gamma_prefactor(a, x)
}

////////////
// Normal //
////////////

/// Standard normal survival function.
///
/// The mirror of [`norm_cdf`], never `1 - norm_cdf(x)`. The difference shows
/// from about `x = 8` onwards, where the tail drops below `f64` resolution
/// near one; `norm_sf(37)` is 5.7e-300 and the subtraction gives a flat zero.
///
/// ### Params
///
/// * `x` - Quantile
///
/// ### Returns
///
/// `P(Z > x)`, accurate down to roughly 1e-308.
pub fn norm_sf(x: f64) -> f64 {
    let h = 0.5 * x * x;
    if x > 0.0 {
        0.5 * reg_gamma_upper(0.5, h)
    } else {
        0.5 * (1.0 + reg_gamma_lower(0.5, h))
    }
}

/// Standard normal CDF.
///
/// The reflection of [norm_sf], which is where the accurate branch lives. Never
/// `1 - norm_sf(x)`: that subtraction is exactly what this module exists to
/// avoid.
///
/// ### Params
///
/// * `x` - Quantile
///
/// ### Returns
///
/// `P(Z <= x)`. Total, so no `Result`: every `f64` is in the domain, and `NaN`
/// propagates.
pub fn norm_cdf(x: f64) -> f64 {
    norm_sf(-x)
}

/// Standard normal quantile function.
///
/// `-sqrt(2) * erfc_inv(2p)`, so a tiny `p` is handled in the tail branch of
/// the inverse rather than by inverting a CDF value that has already rounded
/// to zero.
///
/// ### Params
///
/// * `p` - Probability in `[0, 1]`
///
/// ### Returns
///
/// `z` with `P(Z <= z) = p`. `p = 0` gives `-inf` and `p = 1` gives `+inf`.
/// Anything outside `[0, 1]` is [`BixverseErrors::InvalidArgument`].
pub fn norm_ppf(p: f64) -> Result<f64, BixverseErrors> {
    if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "'p' must be a probability in [0, 1]; got {p}."
        )));
    }
    Ok(-std::f64::consts::SQRT_2 * erfc_inv(2.0 * p))
}

///////////
// Gamma //
///////////

/// Gamma cumulative distribution function, shape and scale parametrisation.
///
/// `P(k, x / theta)`. No location parameter: every caller here fits with the
/// location pinned at zero.
///
/// ### Params
///
/// * `x` - Argument. Negative values give 0.0
/// * `shape` - Shape `k`, finite and strictly positive
/// * `scale` - Scale `theta`, finite and strictly positive
///
/// ### Returns
///
/// `P(X <= x)`, or [`BixverseErrors::InvalidArgument`] for a bad parameter.
pub fn gamma_cdf(x: f64, shape: f64, scale: f64) -> Result<f64, BixverseErrors> {
    check_positive("shape", shape)?;
    check_positive("scale", scale)?;
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x.is_infinite() {
        return Ok(1.0);
    }
    Ok(reg_gamma_lower(shape, x / scale))
}

/// Gamma survival function, shape and scale parametrisation.
///
/// The upper regularised incomplete gamma directly, which is the whole reason
/// this exists: a two-tailed p-value built from `1 - gamma_cdf` saturates at
/// zero once the cdf reaches 1, and reference implementations reach for
/// arbitrary-precision arithmetic to get past it. The continued fraction hands
/// back the tail with no cancellation and no bignum.
///
/// ### Params
///
/// * `x` - Argument. Negative values give 1.0
/// * `shape` - Shape `k`, finite and strictly positive
/// * `scale` - Scale `theta`, finite and strictly positive
///
/// ### Returns
///
/// `P(X > x)`, or [`BixverseErrors::InvalidArgument`] for a bad parameter.
pub fn gamma_sf(x: f64, shape: f64, scale: f64) -> Result<f64, BixverseErrors> {
    check_positive("shape", shape)?;
    check_positive("scale", scale)?;
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(reg_gamma_upper(shape, x / scale))
}

/////////////////
// Chi-squared //
/////////////////

/// Chi-squared survival function.
///
/// The upper regularised incomplete gamma `Q(df/2, x/2)` directly, so the far
/// tail stays accurate: `chisq_sf(200, 1)` is 2.09e-45, where `1 - cdf` returns
/// a flat zero. Fisher's method reads exactly this tail.
///
/// ### Params
///
/// * `x` - Test statistic, non-negative. Negative values give 1.0.
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(X > x)`, or [`BixverseErrors::InvalidArgument`] for a non-positive `df`.
pub fn chisq_sf(x: f64, df: f64) -> Result<f64, BixverseErrors> {
    check_positive("df", df)?;
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(reg_gamma_upper(0.5 * df, 0.5 * x))
}

/////////////////
// Student's t //
/////////////////

/// The two halves of the t distribution's mass either side of `|x|`.
///
/// With `h = df / (df + x^2)` and `z = x^2 / (df + x^2)`, the outer tail beyond
/// `|x|` is `I(h; df/2, 1/2) / 2` and the mass between zero and `|x|` is
/// `I(z; 1/2, df/2) / 2`. Whichever of the two is the smaller is the one
/// evaluated, per [`T_INNER_OUTER_SWITCH`], and the larger is recovered by
/// subtraction, where it can absorb the loss.
///
/// Both halves of this matter. Forming the inner mass as `0.5 - outer` for a
/// large `|x|` is the obvious blunder, but the reverse costs just as much: `h`
/// at `x = 1e-4, df = 100` is `1 - 1e-10`, whose complement carries only six
/// significant digits, and the tail that comes out of it is wrong in the
/// eleventh.
///
/// ### Params
///
/// * `x` - Quantile
/// * `df` - Degrees of freedom, assumed already validated
///
/// ### Returns
///
/// `(inner, outer)`, the mass in `(0, |x|)` and the mass beyond `|x|`. They sum
/// to a half.
fn t_half_masses(x: f64, df: f64) -> (f64, f64) {
    let x2 = x * x;
    if x2 < T_INNER_OUTER_SWITCH {
        let inner = 0.5 * beta_reg(0.5, 0.5 * df, x2 / (df + x2));
        (inner, 0.5 - inner)
    } else {
        let outer = 0.5 * beta_reg(0.5 * df, 0.5, df / (df + x2));
        (0.5 - outer, outer)
    }
}

/// Student's t survival function.
///
/// For `x > 0` this is the incomplete beta itself, never `1 - cdf`, which is
/// what keeps `t_sf(50, 100)` at 7.24e-73 instead of zero.
///
/// ### Params
///
/// * `x` - Quantile
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(T > x)`, or [`BixverseErrors::InvalidArgument`] for a non-positive `df`.
pub fn t_sf(x: f64, df: f64) -> Result<f64, BixverseErrors> {
    check_positive("df", df)?;
    if x.is_infinite() {
        return Ok(if x > 0.0 { 0.0 } else { 1.0 });
    }
    let (inner, outer) = t_half_masses(x, df);
    Ok(if x <= 0.0 { 0.5 + inner } else { outer })
}

/// Student's t CDF.
///
/// The reflection of [t_sf], for the same reason [norm_cdf] is.
///
/// ### Params
///
/// * `x` - Quantile
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(T <= x)`, or [BixverseErrors::InvalidArgument] for a non-positive `df`.
pub fn t_cdf(x: f64, df: f64) -> Result<f64, BixverseErrors> {
    t_sf(-x, df)
}

/// Two-sided Student's t p-value.
///
/// `2 * t_sf(|x|, df)`, which is the Wald p-value every fixed-effect
/// coefficient in [`crate::core::math::mixed_model`] reports. Taking the
/// absolute value first keeps the evaluation in the tail that has not
/// cancelled.
///
/// ### Params
///
/// * `x` - t statistic
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(|T| > |x|)`, capped at 1.0, or [`BixverseErrors::InvalidArgument`] for a
/// non-positive `df`.
pub fn t_pval_two_sided(x: f64, df: f64) -> Result<f64, BixverseErrors> {
    Ok((2.0 * t_sf(x.abs(), df)?).min(1.0))
}

///////
// F //
///////

/// F survival function.
///
/// `I(df2 / (df1 x + df2); df2/2, df1/2)`, formed from the ratio directly
/// rather than as `1 - df1 x / (df1 x + df2)`. That subtraction is where
/// `statrs`'s own `FisherSnedecor::sf` loses the tail.
///
/// ### Params
///
/// * `x` - Test statistic, non-negative. Negative values give 1.0.
/// * `df1` - Numerator degrees of freedom, finite and strictly positive
/// * `df2` - Denominator degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(F > x)`, or [`BixverseErrors::InvalidArgument`] for a non-positive `df`.
pub fn f_sf(x: f64, df1: f64, df2: f64) -> Result<f64, BixverseErrors> {
    check_positive("df1", df1)?;
    check_positive("df2", df2)?;
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(beta_reg(0.5 * df2, 0.5 * df1, df2 / (df1 * x + df2)))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_norm_cdf_matches_r_pnorm() {
        // R: pnorm(c(-3, -1, 0, 1, 3))
        assert_relative_eq!(norm_cdf(-3.0), 0.001349898031630095, max_relative = 1e-12);
        assert_relative_eq!(norm_cdf(-1.0), 0.1586552539314571, max_relative = 1e-12);
        assert_relative_eq!(norm_cdf(0.0), 0.5, max_relative = 1e-15);
        assert_relative_eq!(norm_cdf(1.0), 0.8413447460685429, max_relative = 1e-12);
        assert_relative_eq!(norm_cdf(3.0), 0.9986501019683699, max_relative = 1e-12);
    }

    #[test]
    fn test_norm_sf_survives_the_far_tail() {
        // R: pnorm(37, lower.tail = FALSE) -> 5.7255712225245771e-300.
        assert_relative_eq!(
            norm_sf(37.0),
            5.725_571_222_524_577e-300,
            max_relative = 1e-8
        );
        assert!(1.0 - norm_cdf(37.0) == 0.0, "the naive form should flatten");
    }

    #[test]
    fn test_norm_ppf_round_trips() {
        for p in [1e-10, 1e-3, 0.25, 0.5, 0.975, 1.0 - 1e-9] {
            let z = norm_ppf(p).unwrap();
            assert_relative_eq!(norm_cdf(z), p, max_relative = 1e-9);
        }
        assert!(norm_ppf(-0.1).is_err());
        assert!(norm_ppf(1.1).is_err());
    }

    #[test]
    fn test_chisq_sf_matches_r_pchisq() {
        // R: pchisq(c(0.5, 3.84, 20), df = c(1, 1, 4), lower.tail = FALSE)
        assert_relative_eq!(
            chisq_sf(0.5, 1.0).unwrap(),
            0.4795001221869535,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            chisq_sf(3.84, 1.0).unwrap(),
            0.05004352124870519,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            chisq_sf(20.0, 4.0).unwrap(),
            0.000_499_399_227_387_333_6,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_chisq_sf_far_tail() {
        // R: pchisq(200, 1, lower.tail = FALSE) -> 2.0884875837625449e-45
        assert_relative_eq!(
            chisq_sf(200.0, 1.0).unwrap(),
            2.088_487_583_762_545e-45,
            max_relative = 1e-8
        );
        assert!(chisq_sf(5.0, 0.0).is_err());
        assert_eq!(chisq_sf(-1.0, 1.0).unwrap(), 1.0);
    }

    #[test]
    fn test_t_cdf_matches_r_pt() {
        // R: pt(c(-2, -0.5, 0, 0.5, 2), df = 10)
        assert_relative_eq!(
            t_cdf(-2.0, 10.0).unwrap(),
            0.036694017385370196,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            t_cdf(-0.5, 10.0).unwrap(),
            0.31394680287148646,
            max_relative = 1e-12
        );
        assert_relative_eq!(t_cdf(0.0, 10.0).unwrap(), 0.5, max_relative = 1e-14);
        assert_relative_eq!(
            t_cdf(0.5, 10.0).unwrap(),
            0.686_053_197_128_513_5,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            t_cdf(2.0, 10.0).unwrap(),
            0.963_305_982_614_629_7,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_t_sf_far_tail() {
        // R: pt(50, 100, lower.tail = FALSE) -> 7.2360818398806696e-73
        assert_relative_eq!(
            t_sf(50.0, 100.0).unwrap(),
            7.236_081_839_880_67e-73,
            max_relative = 1e-6
        );
        assert!(t_sf(1.0, 0.0).is_err());
    }

    #[test]
    fn test_t_sf_mirrors_t_cdf() {
        // Both branches of T_INNER_OUTER_SWITCH, either side of zero.
        for &x in &[-3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0] {
            for &df in &[1.0, 5.0, 300.0] {
                let total = t_cdf(x, df).unwrap() + t_sf(x, df).unwrap();
                assert_relative_eq!(total, 1.0, max_relative = 1e-14);
            }
        }
    }

    #[test]
    fn test_t_pval_two_sided() {
        // R: 2 * pt(2.5, 12, lower.tail = FALSE) -> 0.027915399571325213
        assert_relative_eq!(
            t_pval_two_sided(2.5, 12.0).unwrap(),
            0.027915399571325213,
            max_relative = 1e-6
        );
        // Sign does not matter.
        assert_relative_eq!(
            t_pval_two_sided(-2.5, 12.0).unwrap(),
            t_pval_two_sided(2.5, 12.0).unwrap(),
            max_relative = 1e-15
        );
        assert_eq!(t_pval_two_sided(0.0, 12.0).unwrap(), 1.0);
    }

    #[test]
    fn test_f_sf_matches_r_pf() {
        // R: pf(c(1, 4.96, 100), df1 = c(1, 2, 3), df2 = c(10, 10, 10),
        //       lower.tail = FALSE)
        assert_relative_eq!(
            f_sf(1.0, 1.0, 10.0).unwrap(),
            0.340_893_132_302_060_1,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            f_sf(4.96, 2.0, 10.0).unwrap(),
            0.031882570564059055,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            f_sf(100.0, 3.0, 10.0).unwrap(),
            9.327_525_286_716_801e-8,
            max_relative = 1e-6
        );
    }

    #[test]
    fn test_f_sf_edges() {
        assert_eq!(f_sf(-1.0, 1.0, 1.0).unwrap(), 1.0);
        assert_eq!(f_sf(f64::INFINITY, 1.0, 1.0).unwrap(), 0.0);
        assert!(f_sf(1.0, 0.0, 1.0).is_err());
        assert!(f_sf(1.0, 1.0, -2.0).is_err());
    }

    ///////////
    // Gamma //
    ///////////

    #[test]
    fn test_gamma_cdf_matches_r() {
        // R: pgamma(x, shape = k, scale = theta)
        let cases = [
            (1.0, 2.0, 3.0, 0.044_624_919_234_947_685),
            (0.5, 0.3, 2.0, 0.695_545_214_656_659_5),
            (10.0, 5.0, 1.0, 0.970_747_311_923_038_9),
            (2.5, 7.5, 0.4, 0.359_143_513_422_833_35),
        ];

        for (x, shape, scale, expected) in cases {
            assert_relative_eq!(
                gamma_cdf(x, shape, scale).unwrap(),
                expected,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_gamma_sf_matches_r() {
        // R: pgamma(x, shape = k, scale = theta, lower.tail = FALSE)
        let cases = [
            (1.0, 2.0, 3.0, 0.955_375_080_765_052_4),
            (0.5, 0.3, 2.0, 0.304_454_785_343_340_5),
            (10.0, 5.0, 1.0, 0.029_252_688_076_961_08),
            (50.0, 2.0, 3.0, 1.020_735_571_764_047_2e-6),
        ];

        for (x, shape, scale, expected) in cases {
            assert_relative_eq!(
                gamma_sf(x, shape, scale).unwrap(),
                expected,
                max_relative = 1e-11
            );
        }
    }

    /// The whole reason `gamma_sf` exists: `1 - cdf` is a flat zero here.
    #[test]
    fn test_gamma_sf_far_tail() {
        // R: pgamma(200, shape = 2, scale = 1, lower.tail = FALSE)
        let sf = gamma_sf(200.0, 2.0, 1.0).unwrap();

        assert_relative_eq!(sf, 2.781_632_018_740_842_3e-85, max_relative = 1e-9);
        assert_eq!(1.0 - gamma_cdf(200.0, 2.0, 1.0).unwrap(), 0.0);
    }

    #[test]
    fn test_gamma_cdf_sf_complement() {
        for &x in &[0.1, 1.0, 3.0, 7.5] {
            let cdf = gamma_cdf(x, 2.5, 1.5).unwrap();
            let sf = gamma_sf(x, 2.5, 1.5).unwrap();
            assert_relative_eq!(cdf + sf, 1.0, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_gamma_edges() {
        assert_eq!(gamma_cdf(-1.0, 2.0, 1.0).unwrap(), 0.0);
        assert_eq!(gamma_sf(-1.0, 2.0, 1.0).unwrap(), 1.0);
        assert_eq!(gamma_cdf(f64::INFINITY, 2.0, 1.0).unwrap(), 1.0);
        assert_eq!(gamma_sf(f64::INFINITY, 2.0, 1.0).unwrap(), 0.0);
        assert!(gamma_cdf(1.0, 0.0, 1.0).is_err());
        assert!(gamma_sf(1.0, 1.0, -1.0).is_err());
    }
}
