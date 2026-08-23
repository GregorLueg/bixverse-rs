//! Scalar optimisation.
//!
//! One function: R's `optimize`, so a profiled likelihood minimised here stops
//! where R stops. That matters when the objective is flat, which a profiled
//! REML criterion near the variance-component boundary very much is.
//!
//! Ported from `edge-rs` (`src/numeric/optimise.rs`).

////////////
// Consts //
////////////

/// Default `tol` of R's `optimize`, `.Machine$double.eps^0.25`, which is
/// exactly `2^-13`.
pub const OPTIMIZE_TOL: f64 = 1.220_703_125e-4;

/// `sqrt(DBL_EPSILON)`, exactly `2^-26`, the relative half of `Brent_fmin`'s
/// convergence test.
const BRENT_SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

/// Golden section step `(3 - sqrt(5)) / 2`, as `Brent_fmin` spells it.
const BRENT_GOLDEN: f64 = 0.381_966_011_250_105_15;

///////////
// Brent //
///////////

/// Minimises a scalar function on a closed interval, as R's `optimize` does.
///
/// A line-by-line port of R's `Brent_fmin`, parabolic interpolation with a
/// golden-section fallback. The convergence test is R's
/// `sqrt(eps) * |x| + tol / 3`, not scipy's `xatol * |x| + xatol / 3`: the two
/// constants cannot be recovered from a single tolerance, so on a flat
/// objective the iterates and therefore the answers diverge. Matching R to
/// better than its own `tol` means stopping where R stops.
///
/// No iteration cap, because the interval is halved at worst every step and the
/// convergence test is on the interval width, so it always terminates.
///
/// ### Params
///
/// * `ax` - Lower end of the interval
/// * `bx` - Upper end of the interval
/// * `f` - Objective
/// * `tol` - Convergence tolerance, R's `optimize(tol = )`. See
///   [`OPTIMIZE_TOL`] for R's default.
///
/// ### Returns
///
/// The minimiser.
///
/// ### References
///
/// Brent, Algorithms for Minimization without Derivatives, 1973, chapter 5
pub fn brent_fmin<F>(ax: f64, bx: f64, mut f: F, tol: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    let (mut a, mut b) = (ax, bx);
    let mut v = a + BRENT_GOLDEN * (b - a);
    let (mut w, mut x) = (v, v);
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;
    let mut fx = f(x);
    let (mut fv, mut fw) = (fx, fx);
    let tol3 = tol / 3.0;

    loop {
        let xm = 0.5 * (a + b);
        let tol1 = BRENT_SQRT_EPS * x.abs() + tol3;
        let t2 = 2.0 * tol1;
        if (x - xm).abs() <= t2 - 0.5 * (b - a) {
            return x;
        }

        let (mut p, mut q, mut r) = (0.0_f64, 0.0_f64, 0.0_f64);
        if e.abs() > tol1 {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }

        if p.abs() >= (0.5 * q * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            e = if x < xm { b - x } else { a - x };
            d = BRENT_GOLDEN * e;
        } else {
            d = p / q;
            let u = x + d;
            // f must not be evaluated too close to either end of the interval.
            if u - a < t2 || b - u < t2 {
                d = if x >= xm { -tol1 } else { tol1 };
            }
        }

        // Nor too close to the incumbent.
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_brent_fmin_matches_r_optimize() {
        // R: optimize(function(x) (x - 1/3)^2, c(0, 1))$minimum
        //    -> 0.33333333333333331
        let got = brent_fmin(0.0, 1.0, |x| (x - 1.0 / 3.0).powi(2), OPTIMIZE_TOL);
        assert_relative_eq!(got, 0.333_333_333_333_333_3, max_relative = 1e-14);
    }

    #[test]
    fn test_brent_fmin_matches_r_on_a_skewed_objective() {
        // R: optimize(function(x) x * sin(4 * x), c(0, 3))$minimum
        //    -> 1.2282971127458993
        let got = brent_fmin(0.0, 3.0, |x| x * (4.0 * x).sin(), OPTIMIZE_TOL);
        assert_relative_eq!(got, 1.2282971127458993, max_relative = 1e-14);
    }

    #[test]
    fn test_brent_fmin_finds_a_boundary_minimum() {
        // Monotone increasing, so the minimiser sits at the lower end and only
        // the convergence test keeps it off it. R: optimize(function(x) x, c(-1, 2))
        //    -> -0.99992424816296976
        let got = brent_fmin(-1.0, 2.0, |x| x, OPTIMIZE_TOL);
        assert_relative_eq!(got, -0.999_924_248_162_969_8, max_relative = 1e-12);
    }

    /// A tighter tolerance lands closer.
    ///
    /// The objective has to be non-quadratic to show it: parabolic
    /// interpolation solves a quadratic exactly on the first step, so both
    /// tolerances return the identical `f64` and the comparison is `0.0 <= 0.0`.
    /// A quartic makes the tolerance the thing that decides where it stops.
    #[test]
    fn test_brent_fmin_tighter_tol_gets_closer() {
        let quartic = |x: f64| (x - 1.0 / 3.0).powi(4);
        let target = 1.0 / 3.0;
        // R: optimize(function(x) (x - 1/3)^4, c(0, 1))$minimum
        let loose = brent_fmin(0.0, 1.0, quartic, OPTIMIZE_TOL);
        assert_relative_eq!(loose, 0.33333944604491594, max_relative = 1e-12);
        // R: the same with tol = 1e-12
        let tight = brent_fmin(0.0, 1.0, quartic, 1e-12);
        assert_relative_eq!(tight, 0.333_333_333_266_562_2, max_relative = 1e-12);

        assert!((tight - target).abs() < (loose - target).abs());
    }
}
