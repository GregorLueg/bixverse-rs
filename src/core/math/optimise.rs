//! Scalar optimisation.
//!
//! R's `optimize` and R's `uniroot`, so a profiled likelihood minimised here
//! stops where R stops and a root found here is the root R finds. That matters
//! when the objective is flat, which a profiled REML criterion near the
//! variance-component boundary very much is, and when the answer feeds a
//! parity fixture.
//!
//! [`brent_fmin`] is ported from `edge-rs` (`src/numeric/optimise.rs`).

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

/// Default `tol` of R's `uniroot`, `.Machine$double.eps^0.25`.
///
/// The same constant as [`OPTIMIZE_TOL`], spelled separately because the two
/// routines are free to disagree and callers read one or the other.
pub const UNIROOT_TOL: f64 = 1.220_703_125e-4;

/// Default `maxiter` of R's `uniroot`.
pub const UNIROOT_MAXIT: usize = 1000;

/// `DBL_EPSILON`, the relative half of `R_zeroin2`'s convergence test.
const ZEROIN_EPS: f64 = f64::EPSILON;

/// Interpolation acceptance factor in `R_zeroin2`.
///
/// A parabolic or secant step is only taken when it lands inside this fraction
/// of the bracketing interval, otherwise the bisection step stands.
const ZEROIN_INTERP_FRAC: f64 = 0.75;

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

////////////
// Zeroin //
////////////

/// Finds a root of a scalar function on a bracketing interval, as R's
/// `uniroot` does.
///
/// A line-by-line port of R's `R_zeroin2`: inverse quadratic interpolation with
/// a secant fallback and bisection whenever the interpolated step leaves the
/// bracket. The convergence test is R's `2 * eps * |b| + tol / 2`.
///
/// The caller supplies `f(ax)` and `f(bx)` because the surrounding code in
/// `bw.SJ` has already evaluated both while widening the search interval, and
/// the objective is not cheap.
///
/// ### Params
///
/// * `ax` - Lower end of the bracket
/// * `bx` - Upper end of the bracket
/// * `fa` - `f(ax)`, already evaluated
/// * `fb` - `f(bx)`, already evaluated
/// * `f` - The objective
/// * `tol` - Convergence tolerance, R's `uniroot(tol = )`. See [`UNIROOT_TOL`]
///   for R's default.
/// * `maxit` - Iteration cap, R's `uniroot(maxiter = )`. See [`UNIROOT_MAXIT`].
///
/// ### Returns
///
/// The root, or `None` when the interval does not bracket one or the iteration
/// cap is hit without converging.
///
/// ### References
///
/// Brent, Algorithms for Minimization without Derivatives, 1973, chapter 4
pub fn zeroin<F>(
    ax: f64,
    bx: f64,
    fa: f64,
    fb: f64,
    mut f: F,
    tol: f64,
    maxit: usize,
) -> Option<f64>
where
    F: FnMut(f64) -> f64,
{
    let (mut a, mut b, mut fa, mut fb) = (ax, bx, fa, fb);

    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    if fa * fb > 0.0 {
        return None;
    }

    // Third point of the bracket. Starts on `a` and thereafter holds whichever
    // of the previous two iterates sits on the opposite side of the root.
    let (mut c, mut fc) = (a, fa);

    for _ in 0..=maxit {
        let prev_step = b - a;

        // Keep `b` as the best approximation so far.
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol_act = 2.0 * ZEROIN_EPS * b.abs() + tol / 2.0;
        let mut new_step = (c - b) / 2.0;

        if new_step.abs() <= tol_act || fb == 0.0 {
            return Some(b);
        }

        // Interpolate only when the previous step was large enough to trust the
        // curvature and `b` really is the better of the two.
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q) = if a == c {
                // Two distinct points only, so linear (secant) interpolation.
                let t1 = fb / fa;
                (cb * t1, 1.0 - t1)
            } else {
                // Three distinct points, so inverse quadratic interpolation.
                let q0 = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                (
                    t2 * (cb * q0 * (q0 - t1) - (b - a) * (t1 - 1.0)),
                    (q0 - 1.0) * (t1 - 1.0) * (t2 - 1.0),
                )
            };

            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }

            if p < (ZEROIN_INTERP_FRAC * cb * q - (tol_act * q).abs() / 2.0)
                && p < (prev_step * q / 2.0).abs()
            {
                new_step = p / q;
            }
        }

        // Never step by less than the tolerance, or the iteration stalls.
        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }

        a = b;
        fa = fb;
        b += new_step;
        fb = f(b);

        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }

    None
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
