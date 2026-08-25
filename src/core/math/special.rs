//! Special functions that need to stay generic over the float type.
//!
//! `statrs` covers the same ground in `f64` only. That is fine for the tail
//! probabilities in [crate::core::math::distributions], which are evaluated
//! once per test, but not for [digamma]: variational Bayes calls it `k` times
//! per document per inner iteration, and round-tripping an `f32` pipeline
//! through `f64` there costs more than the function itself.

use crate::utils::traits::BixverseFloat;

////////////
// Consts //
////////////

/// Argument above which the asymptotic series for [digamma] is used directly.
///
/// The series error falls off as `x^-(2n)`, so it is the shift that sets the
/// accuracy. The first dropped term of the six-term expansion below is
/// `B_14 / (14 x^14)`, which at `x = 6` is 1e-12 and only reaches `f64`
/// resolution around ten.
const DIGAMMA_ASYMPTOTIC_MIN: f64 = 10.0;

/// Coefficients `B_{2n} / 2n` of the asymptotic expansion of `psi(x)`.
///
/// `psi(x) ~ ln x - 1/(2x) - sum_{n>=1} B_{2n} / (2n x^{2n})`. The Bernoulli
/// numbers alternate in sign from `B_4` on, so these do too; dropping that
/// alternation doubles the second term's contribution and costs five digits
/// near the shift boundary.
const DIGAMMA_SERIES: [f64; 6] = [
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
];

/////////////
// Digamma //
/////////////

/// Digamma function `psi(x) = d/dx ln Gamma(x)`, generic over the float type.
///
/// Recurrence `psi(x) = psi(x + 1) - 1/x` shifts the argument up past
/// `DIGAMMA_ASYMPTOTIC_MIN`, then the asymptotic expansion in
/// `DIGAMMA_SERIES` finishes it. Accurate to a few ulp in `f64` and to `f32`
/// resolution in `f32` across the positive reals.
///
/// Only defined for `x > 0`, which is all the variational Bayes updates ever
/// pass (Dirichlet parameters are strictly positive by construction). A
/// non-positive argument returns `NaN` rather than reflecting, since a caller
/// reaching one has a bug upstream and silently returning a finite value would
/// hide it.
///
/// ### Params
///
/// * `x` - Argument, strictly positive.
///
/// ### Returns
///
/// `psi(x)`, or `NaN` for `x <= 0`.
///
/// ### References
///
/// Abramowitz and Stegun, Handbook of Mathematical Functions, 6.3.5 and 6.3.18
#[inline]
pub fn digamma<F: BixverseFloat>(x: F) -> F {
    if x <= F::zero() {
        return F::nan();
    }

    let min_shift = F::from_f64(DIGAMMA_ASYMPTOTIC_MIN).unwrap();

    // psi(x) = psi(x + 1) - 1/x, applied until the asymptotic series is valid
    let mut acc = F::zero();
    let mut z = x;
    while z < min_shift {
        acc -= z.recip();
        z += F::one();
    }

    let inv = z.recip();
    let inv_sq = inv * inv;

    let mut series = F::zero();
    let mut power = inv_sq;
    for coeff in DIGAMMA_SERIES {
        series += F::from_f64(coeff).unwrap() * power;
        power *= inv_sq;
    }

    acc + z.ln() - F::from_f64(0.5).unwrap() * inv - series
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use statrs::function::gamma::digamma as statrs_digamma;

    #[test]
    fn test_digamma_matches_statrs_f64() {
        let xs = [
            1e-6, 1e-3, 0.1, 0.5, 1.0, 1.5, 2.0, 3.7, 9.999, 10.0, 10.001, 50.0, 1e3, 1e6,
        ];
        for x in xs {
            assert_relative_eq!(digamma(x), statrs_digamma(x), max_relative = 1e-13);
        }
    }

    #[test]
    fn test_digamma_matches_statrs_f32() {
        for x in [0.25_f32, 1.0, 2.5, 7.0, 100.0] {
            assert_relative_eq!(
                digamma(x),
                statrs_digamma(x as f64) as f32,
                max_relative = 1e-6
            );
        }
    }

    /// `psi(1) = -gamma` and `psi(0.5) = -gamma - 2 ln 2`, the two closed forms.
    #[test]
    fn test_digamma_known_values() {
        const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
        assert_relative_eq!(digamma(1.0), -EULER_MASCHERONI, max_relative = 1e-14);
        assert_relative_eq!(
            digamma(0.5),
            -EULER_MASCHERONI - 2.0 * 2.0_f64.ln(),
            max_relative = 1e-14
        );
    }

    /// The defining recurrence, checked across the shift boundary.
    #[test]
    fn test_digamma_recurrence() {
        for x in [0.3_f64, 2.0, 5.5, 9.999, 15.0] {
            assert_relative_eq!(digamma(x + 1.0) - digamma(x), 1.0 / x, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_digamma_non_positive_is_nan() {
        assert!(digamma(0.0_f64).is_nan());
        assert!(digamma(-1.5_f64).is_nan());
    }
}
