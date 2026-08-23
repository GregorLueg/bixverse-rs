//! Linear mixed model with a single random intercept.
//!
//! Fits `y = X beta + Z b + e` with `b ~ N(0, sigma_b^2 I_G)`,
//! `e ~ N(0, sigma^2 I_n)` and one grouping factor, by REML, and reports Wald
//! tests on the fixed effects with Satterthwaite denominator degrees of
//! freedom. That is `lmer(y ~ (1 | g) + ...)` read through `lmerTest`.
//!
//! ### Why it is cheap
//!
//! With one grouping factor the inverse of `H = I + lambda Z Z'` is closed
//! form, `lambda = sigma_b^2 / sigma^2`:
//!
//! ```text
//! H^-1 = I - sum_g [lambda / (1 + n_g lambda)] J_g
//! ```
//!
//! so every generalised least squares quantity is a rank-`G` correction to the
//! ordinary Gram matrices:
//!
//! ```text
//! X'H^-1 X = X'X - sum_g c_g s_g s_g'      s_g = X_g' 1
//! X'H^-1 y = X'y - sum_g c_g s_g t_g       t_g = 1' y_g
//! y'H^-1 y = y'y - sum_g c_g t_g^2         c_g = lambda / (1 + n_g lambda)
//! ```
//!
//! The whole fit therefore depends on the data only through `X'X`, `X'y`,
//! `y'y` and the per-group `(n_g, s_g, t_g)`. Building those is one `O(n p^2)`
//! pass; every subsequent evaluation, and so the entire optimisation, is
//! `O(G p^2)` in the number of *groups*. A caller sweeping many responses or
//! many predictors over the same grouping can build the statistics once and
//! refit for almost nothing.
//!
//! ### Everything is in `f64`
//!
//! [RandomInterceptStats::from_design] accepts any [BixverseFloat] but stores
//! and works in `f64`. The REML criterion is a sum of logs against a residual
//! sum of squares formed by cancellation, and the Satterthwaite step then takes
//! numerical second derivatives of it. There is no `f32` version of this that
//! is worth having.
//!
//! ### References
//!
//! Bates, Mächler, Bolker & Walker, Journal of Statistical Software 67(1),
//! 2015, for the REML formulation. Kuznetsova, Brockhoff & Christensen, Journal
//! of Statistical Software 82(13), 2017, for the Satterthwaite approximation.

use faer::{
    Mat, MatRef,
    linalg::solvers::{DenseSolveCore, PartialPivLu},
};

use crate::core::math::distributions::t_pval_two_sided;
use crate::core::math::linear_algebra::spd_log_det;
use crate::core::math::optimise::brent_fmin;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Upper end of the bracket searched for the variance ratio.
///
/// `lambda` is between-group over within-group variance. `lme4` will happily
/// report ratios in the millions when a grouping absorbs almost all the
/// variation, so this is deliberately generous. Hitting it is still detected
/// and reported rather than silently accepted: see
/// [RandomInterceptFit::bracket_exhausted].
const DEFAULT_LAMBDA_MAX: f64 = 1.0e6;

/// Fraction of the bracket above which the fit is treated as having run out of
/// room, and the inference withheld.
const BRACKET_EDGE: f64 = 0.99;

/// Convergence tolerance handed to Brent for the variance ratio.
///
/// Tighter than R's `optimize` default, because this is `lmer`'s only free
/// parameter and everything downstream inherits its error. Do not tighten
/// further: Brent stops at `sqrt(eps) * |x| + tol / 3`, so past roughly `1e-7`
/// the first term dominates and a smaller tolerance buys iterations rather
/// than digits. At this setting the ratio agrees with `lme4` to eight
/// significant figures and the fixed effects to eleven.
const DEFAULT_LAMBDA_TOL: f64 = 1.0e-8;

/// Below this the variance ratio is treated as exactly zero.
///
/// `lme4` calls the same situation a singular fit. The model has collapsed to
/// ordinary least squares, which is a legitimate answer, not a failure.
const SINGULAR_LAMBDA: f64 = 1.0e-8;

/// Relative step for the numerical gradient of the coefficient variance.
const GRAD_STEP: f64 = 1.0e-5;

/// Relative step for the numerical Hessian of the REML criterion.
///
/// `lmerTest` differentiates its deviance numerically too, and the two agree to
/// seven significant figures on the degrees of freedom at this setting.
const HESSIAN_STEP: f64 = 1.0e-4;

/// Absolute floor on the `theta` finite-difference step.
///
/// `theta` is a ratio of standard deviations, so it is dimensionless and
/// `O(1)`, and this floor is what lets the central difference straddle the
/// `theta = 0` boundary. `sigma` gets no such floor: it carries the units of
/// the response, so flooring its step at an absolute value would make the
/// degrees of freedom depend on the scale of `y`, which they are invariant to.
const THETA_STEP_FLOOR: f64 = 1.0;

/// Floor on the `sigma` step, present only to keep a degenerate fit finite.
const SIGMA_STEP_FLOOR: f64 = 1.0e-300;

////////////////////////////
// RandomInterceptStats //
////////////////////////////

/// Sufficient statistics for a single-random-intercept fit.
///
/// Everything the REML criterion needs, reduced from the data once. See the
/// module documentation for why this is the whole story.
#[derive(Clone, Debug)]
pub struct RandomInterceptStats {
    /// `X'X`, `p x p`.
    pub xtx: Mat<f64>,
    /// `X'y`, length `p`.
    pub xty: Vec<f64>,
    /// `y'y`.
    pub yty: f64,
    /// Number of observations.
    pub n: usize,
    /// Number of fixed-effect columns.
    pub p: usize,
    /// Observations per group. Empty groups are dropped on construction, so
    /// every entry is positive.
    pub group_n: Vec<f64>,
    /// Column sums of `X` within each group, `G x p`.
    pub group_s: Mat<f64>,
    /// Sum of `y` within each group, length `G`.
    pub group_t: Vec<f64>,
}

impl RandomInterceptStats {
    /// Reduces a design and response to the sufficient statistics.
    ///
    /// One `O(n p^2)` pass. Groups with no observations are dropped rather than
    /// counted, so a caller may pass a level set wider than the data.
    ///
    /// ### Params
    ///
    /// * `x` - Design matrix, `n x p`. Include the intercept column yourself.
    /// * `y` - Response, length `n`
    /// * `groups` - Level code per observation, each in `0..n_groups`
    /// * `n_groups` - Number of levels
    ///
    /// ### Returns
    ///
    /// The statistics, or [BixverseErrors::ShapeMismatch] when the arguments
    /// disagree in length and [BixverseErrors::InvalidArgument] for an
    /// out-of-range level code or fewer than two non-empty groups.
    pub fn from_design<T: BixverseFloat>(
        x: MatRef<T>,
        y: &[T],
        groups: &[usize],
        n_groups: usize,
    ) -> Result<Self, BixverseErrors> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (n, p),
                got: (y.len(), p),
            });
        }
        if groups.len() != n {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (n, p),
                got: (groups.len(), p),
            });
        }
        if p == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "the design matrix has no columns.".to_string(),
            ));
        }

        let xf = Mat::<f64>::from_fn(n, p, |i, j| x[(i, j)].to_f64().unwrap_or(f64::NAN));
        let yf: Vec<f64> = y.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();

        // A non-finite value would otherwise reach the optimiser, which reports
        // `INFINITY` at every lambda, falls back to the boundary, and hands back
        // a result labelled as a perfectly ordinary singular fit.
        if yf.iter().any(|v| !v.is_finite())
            || (0..n).any(|i| (0..p).any(|j| !xf[(i, j)].is_finite()))
        {
            return Err(BixverseErrors::InvalidArgument(
                "the design or response contains a non-finite value.".to_string(),
            ));
        }

        let mut counts = vec![0.0_f64; n_groups];
        let mut sums = Mat::<f64>::zeros(n_groups, p);
        let mut tots = vec![0.0_f64; n_groups];
        for (i, &g) in groups.iter().enumerate() {
            if g >= n_groups {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "group code {g} is outside 0..{n_groups}."
                )));
            }
            counts[g] += 1.0;
            tots[g] += yf[i];
            for j in 0..p {
                sums[(g, j)] += xf[(i, j)];
            }
        }

        // Drop empty levels; they contribute nothing and would put a zero in
        // the log-determinant.
        let kept: Vec<usize> = (0..n_groups).filter(|&g| counts[g] > 0.0).collect();
        if kept.len() < 2 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "a random intercept needs at least two non-empty groups; got {}.",
                kept.len()
            )));
        }
        let group_n: Vec<f64> = kept.iter().map(|&g| counts[g]).collect();
        let group_t: Vec<f64> = kept.iter().map(|&g| tots[g]).collect();
        let group_s = Mat::<f64>::from_fn(kept.len(), p, |i, j| sums[(kept[i], j)]);

        let xtx = xf.transpose() * &xf;
        let xty: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| xf[(i, j)] * yf[i]).sum())
            .collect();
        let yty: f64 = yf.iter().map(|v| v * v).sum();

        Ok(Self {
            xtx,
            xty,
            yty,
            n,
            p,
            group_n,
            group_s,
            group_t,
        })
    }

    /// Number of groups.
    ///
    /// ### Returns
    ///
    /// The count of non-empty levels.
    pub fn n_groups(&self) -> usize {
        self.group_n.len()
    }
}

/////////////////
// The GLS core //
/////////////////

/// Generalised least squares quantities at one variance ratio.
#[derive(Clone, Debug)]
struct GlsFit {
    /// `(X'H^-1 X)^-1`.
    a_inv: Mat<f64>,
    /// `log |X'H^-1 X|`.
    log_det_a: f64,
    /// `sum_g log(1 + n_g lambda)`, which is `log |H|`.
    log_det_h: f64,
    /// The GLS coefficients.
    beta: Vec<f64>,
    /// `r' H^-1 r` at those coefficients.
    rss: f64,
}

/// Solves the GLS system at a given variance ratio.
///
/// ### Params
///
/// * `s` - Sufficient statistics
/// * `lambda` - Variance ratio `sigma_b^2 / sigma^2`, non-negative
///
/// ### Returns
///
/// The [GlsFit], or [BixverseErrors::InvalidArgument] when `X'H^-1 X` is
/// numerically singular, which means the design is rank deficient.
fn gls_solve(s: &RandomInterceptStats, lambda: f64) -> Result<GlsFit, BixverseErrors> {
    let p = s.p;
    let g = s.n_groups();

    let mut a = s.xtx.clone();
    let mut b = s.xty.clone();
    let mut yhy = s.yty;
    let mut log_det_h = 0.0;

    for k in 0..g {
        let nk = s.group_n[k];
        let shrink = lambda / (1.0 + nk * lambda);
        log_det_h += (1.0 + nk * lambda).ln();
        if shrink == 0.0 {
            continue;
        }
        let tk = s.group_t[k];
        for i in 0..p {
            let si = s.group_s[(k, i)];
            b[i] -= shrink * si * tk;
            for j in 0..p {
                a[(i, j)] -= shrink * si * s.group_s[(k, j)];
            }
        }
        yhy -= shrink * tk * tk;
    }

    // X'H^-1 X is positive definite for a full-rank design, so a Cholesky both
    // gives the log-determinant and doubles as the rank test. Working in logs
    // keeps it finite for a design that is merely badly scaled.
    let Some(log_det_a) = spd_log_det(a.as_ref()) else {
        return Err(BixverseErrors::InvalidArgument(
            "the fixed-effect design is rank deficient; X'H^-1 X is not positive definite."
                .to_string(),
        ));
    };

    let lu: PartialPivLu<f64> = a.partial_piv_lu();
    let a_inv = lu.inverse();

    let beta: Vec<f64> = (0..p)
        .map(|i| (0..p).map(|j| a_inv[(i, j)] * b[j]).sum())
        .collect();
    // r'H^-1 r = y'H^-1 y - 2 beta' X'H^-1 y + beta' X'H^-1 X beta.
    let cross: f64 = beta.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let quad: f64 = (0..p)
        .map(|i| (0..p).map(|j| beta[i] * a[(i, j)] * beta[j]).sum::<f64>())
        .sum();
    let rss = yhy - 2.0 * cross + quad;

    Ok(GlsFit {
        log_det_a,
        a_inv,
        log_det_h,
        beta,
        rss,
    })
}

/// REML criterion profiled over the residual variance.
///
/// `-2 log L_REML` with `sigma^2` replaced by its conditional maximiser, so the
/// only free parameter left is the variance ratio.
///
/// ### Params
///
/// * `s` - Sufficient statistics
/// * `lambda` - Variance ratio
///
/// ### Returns
///
/// The criterion, or `f64::INFINITY` where the solve fails, so an optimiser
/// walks away from it rather than stopping.
fn profiled_reml(s: &RandomInterceptStats, lambda: f64) -> f64 {
    let Ok(fit) = gls_solve(s, lambda) else {
        return f64::INFINITY;
    };
    let df = (s.n - s.p) as f64;
    if !(fit.rss.is_finite() && fit.rss > 0.0) {
        return f64::INFINITY;
    }
    let sigma_sq = fit.rss / df;
    df * sigma_sq.ln() + fit.log_det_h + fit.log_det_a + df + df * (2.0 * std::f64::consts::PI).ln()
}

/// REML criterion in `lmerTest`'s parametrisation, unprofiled.
///
/// `-2 log L_REML` as a function of `(theta, sigma)`, where
/// `theta = sigma_b / sigma`. Two things about this matter and neither is
/// arbitrary. The Satterthwaite denominator comes from a *numerical* Hessian,
/// which is not invariant to reparametrisation even though the degrees of
/// freedom are, so it has to be taken in the same coordinates `lmerTest` uses.
/// And squaring `theta` makes the criterion even in it, which is what lets the
/// central differences below work unchanged at the `theta = 0` boundary.
///
/// ### Params
///
/// * `s` - Sufficient statistics
/// * `theta` - Ratio of the between-group to the residual standard deviation
/// * `sigma` - Residual standard deviation
///
/// ### Returns
///
/// The criterion, or `f64::INFINITY` where the solve fails.
fn reml_criterion(s: &RandomInterceptStats, theta: f64, sigma: f64) -> f64 {
    let Ok(fit) = gls_solve(s, theta * theta) else {
        return f64::INFINITY;
    };
    let df = (s.n - s.p) as f64;
    let sigma_sq = sigma * sigma;
    if !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        return f64::INFINITY;
    }
    df * sigma_sq.ln()
        + fit.log_det_h
        + fit.log_det_a
        + fit.rss / sigma_sq
        + df * (2.0 * std::f64::consts::PI).ln()
}

/// Variance of one fixed-effect contrast, as a function of `(theta, sigma)`.
///
/// ### Params
///
/// * `s` - Sufficient statistics
/// * `theta` - Standard deviation ratio
/// * `sigma` - Residual standard deviation
/// * `coef` - Index of the coefficient
///
/// ### Returns
///
/// `sigma^2 (X'H^-1 X)^-1_[coef, coef]`, or `f64::NAN` where the solve fails.
fn coefficient_variance(s: &RandomInterceptStats, theta: f64, sigma: f64, coef: usize) -> f64 {
    match gls_solve(s, theta * theta) {
        Ok(fit) => fit.a_inv[(coef, coef)] * sigma * sigma,
        Err(_) => f64::NAN,
    }
}

//////////////////////////
// Params and the result //
//////////////////////////

/// Tuning knobs for [fit_random_intercept].
#[derive(Clone, Copy, Debug)]
pub struct RandomInterceptParams {
    /// Upper end of the bracket searched for the variance ratio.
    pub lambda_max: f64,
    /// Convergence tolerance for the variance ratio.
    pub lambda_tol: f64,
    /// Compute Satterthwaite degrees of freedom. Turning this off leaves `df`
    /// at the residual value `n - p` and skips a numerical Hessian and one
    /// gradient per coefficient, which is worth doing when fitting at scale and
    /// the group count is large enough that the two agree anyway. An exhausted
    /// bracket still yields `NaN` regardless of this flag.
    pub satterthwaite: bool,
}

impl RandomInterceptParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `lambda_max` - Upper end of the variance-ratio bracket
    /// * `lambda_tol` - Convergence tolerance for the variance ratio
    /// * `satterthwaite` - Compute Satterthwaite degrees of freedom
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(lambda_max: f64, lambda_tol: f64, satterthwaite: bool) -> Self {
        Self {
            lambda_max,
            lambda_tol,
            satterthwaite,
        }
    }
}

impl Default for RandomInterceptParams {
    fn default() -> Self {
        Self {
            lambda_max: DEFAULT_LAMBDA_MAX,
            lambda_tol: DEFAULT_LAMBDA_TOL,
            satterthwaite: true,
        }
    }
}

/// A fitted single-random-intercept model.
#[derive(Clone, Debug)]
pub struct RandomInterceptFit {
    /// Fixed-effect coefficients.
    pub beta: Vec<f64>,
    /// Standard errors of the coefficients.
    pub se: Vec<f64>,
    /// Wald t statistics.
    pub t_stat: Vec<f64>,
    /// Denominator degrees of freedom per coefficient. Satterthwaite by
    /// default, `n - p` when it is switched off, and `NaN` when the variance
    /// ratio ran past its bracket, which withholds the inference either way.
    pub df: Vec<f64>,
    /// Two-sided p-values.
    pub p_value: Vec<f64>,
    /// Residual variance.
    pub sigma_sq: f64,
    /// Between-group variance.
    pub sigma_b_sq: f64,
    /// The fitted variance ratio `sigma_b^2 / sigma^2`.
    pub lambda: f64,
    /// Whether the fit landed on the `lambda = 0` boundary. `lme4` calls this
    /// singular. The model has collapsed to ordinary least squares; the
    /// coefficients are still the right answer for the data.
    pub singular: bool,
    /// Whether the search ran out of bracket at the far end.
    ///
    /// The variance ratio is unbounded above, so it can exceed any bracket. The
    /// coefficients and standard errors still converge and are reported, but
    /// `df` and `p_value` come back `NaN`: the Satterthwaite denominator is
    /// built from the asymptotic covariance of the variance parameters, which
    /// is only that at a stationary point, and the edge of the bracket is not
    /// one. Reporting a number from there was wrong by a factor of five in
    /// testing, in the anti-conservative direction.
    pub bracket_exhausted: bool,
}

////////////////
// The fit //
////////////////

/// Fits a single-random-intercept linear mixed model by REML.
///
/// The variance ratio is found by Brent on the profiled criterion, then the
/// coefficients, standard errors and Satterthwaite degrees of freedom follow in
/// closed form. Cost is `O(iterations * G p^2)` given the statistics, with `G`
/// the number of groups and `p` the number of fixed effects: nothing here
/// touches the observations again.
///
/// ### Params
///
/// * `stats` - Sufficient statistics from [RandomInterceptStats::from_design]
/// * `params` - Optional [RandomInterceptParams]; the default otherwise
///
/// ### Returns
///
/// The [RandomInterceptFit], or [BixverseErrors::InvalidArgument] for a rank
/// deficient design or a model with no residual degrees of freedom.
///
/// ### References
///
/// Kuznetsova, Brockhoff & Christensen, Journal of Statistical Software 82(13),
/// 2017
pub fn fit_random_intercept(
    stats: &RandomInterceptStats,
    params: Option<RandomInterceptParams>,
) -> Result<RandomInterceptFit, BixverseErrors> {
    let params = params.unwrap_or_default();
    if stats.n <= stats.p {
        return Err(BixverseErrors::InvalidArgument(format!(
            "the model has {} observations and {} fixed effects, so no residual degrees of \
             freedom.",
            stats.n, stats.p
        )));
    }
    // Checked here rather than only in the constructor: every field is public
    // and callers holding per-group summaries build the struct directly. With
    // one group the random intercept is not identified from the fixed one, and
    // with none this degenerates to ordinary least squares wearing a mixed
    // model's name.
    if stats.n_groups() < 2 {
        return Err(BixverseErrors::InvalidArgument(format!(
            "a random intercept needs at least two groups; got {}.",
            stats.n_groups()
        )));
    }

    let lambda_hat = brent_fmin(
        0.0,
        params.lambda_max,
        |lambda| profiled_reml(stats, lambda),
        params.lambda_tol,
    );
    // Brent never evaluates the ends of the bracket, and the boundary is where
    // a grouping that explains nothing lands. Take it if it wins.
    let lambda = if profiled_reml(stats, 0.0) <= profiled_reml(stats, lambda_hat) {
        0.0
    } else {
        lambda_hat
    };
    let singular = lambda <= SINGULAR_LAMBDA;
    let lambda = if singular { 0.0 } else { lambda };
    // See `RandomInterceptFit::bracket_exhausted`.
    let bracket_exhausted = lambda >= BRACKET_EDGE * params.lambda_max;

    let fit = gls_solve(stats, lambda)?;
    let df_resid = (stats.n - stats.p) as f64;
    let sigma_sq = fit.rss / df_resid;
    let sigma = sigma_sq.sqrt();
    let theta = lambda.sqrt();

    let mut se = Vec::with_capacity(stats.p);
    let mut t_stat = Vec::with_capacity(stats.p);
    let mut df = Vec::with_capacity(stats.p);
    let mut p_value = Vec::with_capacity(stats.p);

    let hessian = if params.satterthwaite && !bracket_exhausted {
        Some(reml_hessian(stats, theta, sigma))
    } else {
        None
    };

    for j in 0..stats.p {
        let variance = fit.a_inv[(j, j)] * sigma_sq;
        let error = variance.max(0.0).sqrt();
        let t = fit.beta[j] / error;
        let dof = match hessian.as_ref() {
            Some(h) => satterthwaite_df(stats, theta, sigma, j, variance, h).unwrap_or(df_resid),
            None if bracket_exhausted => f64::NAN,
            None => df_resid,
        };
        se.push(error);
        t_stat.push(t);
        p_value.push(if t.is_finite() && dof.is_finite() && dof > 0.0 {
            t_pval_two_sided(t, dof)?
        } else {
            f64::NAN
        });
        df.push(dof);
    }

    Ok(RandomInterceptFit {
        beta: fit.beta,
        se,
        t_stat,
        df,
        p_value,
        sigma_sq,
        sigma_b_sq: lambda * sigma_sq,
        lambda,
        singular,
        bracket_exhausted,
    })
}

/// Numerical Hessian of the REML criterion in `(theta, sigma)`.
///
/// Central differences on the four corner points, which is the same scheme
/// `lmerTest` reaches for through `numDeriv`. Symmetrised, because the two
/// mixed partials differ at the level of the truncation error.
///
/// ### Params
///
/// * `s` - Sufficient statistics
/// * `theta` - Fitted standard deviation ratio
/// * `sigma` - Fitted residual standard deviation
///
/// ### Returns
///
/// The `2 x 2` Hessian, row-major.
fn reml_hessian(s: &RandomInterceptStats, theta: f64, sigma: f64) -> [f64; 4] {
    let phi = [theta, sigma];
    let step = [
        HESSIAN_STEP * phi[0].abs().max(THETA_STEP_FLOOR),
        HESSIAN_STEP * phi[1].abs().max(SIGMA_STEP_FLOOR),
    ];
    let eval = |a: f64, b: f64| reml_criterion(s, a, b);
    let at = |d0: f64, d1: f64| eval(phi[0] + d0, phi[1] + d1);

    let mut h = [0.0_f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            let (mut pp, mut pm, mut mp, mut mm) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
            pp[i] += step[i];
            pp[j] += step[j];
            pm[i] += step[i];
            pm[j] -= step[j];
            mp[i] -= step[i];
            mp[j] += step[j];
            mm[i] -= step[i];
            mm[j] -= step[j];
            h[i * 2 + j] = (at(pp[0], pp[1]) - at(pm[0], pm[1]) - at(mp[0], mp[1])
                + at(mm[0], mm[1]))
                / (4.0 * step[i] * step[j]);
        }
    }
    let off = 0.5 * (h[1] + h[2]);
    h[1] = off;
    h[2] = off;
    h
}

/// Satterthwaite denominator degrees of freedom for one coefficient.
///
/// `nu = 2 C^2 / (g' A g)` with `C` the coefficient variance, `g` its gradient
/// in `(theta, sigma)`, and `A = 2 Hess^-1` the asymptotic covariance of the
/// variance parameters.
///
/// At a singular fit this returns the residual degrees of freedom on its own,
/// with no special case: the criterion and the coefficient variance both depend
/// on `theta` only through `theta^2`, so the `theta` component of the gradient
/// vanishes at the boundary and the formula collapses to the ordinary least
/// squares answer.
///
/// ### Params
///
/// * `s` - Sufficient statistics
/// * `theta` - Fitted standard deviation ratio
/// * `sigma` - Fitted residual standard deviation
/// * `coef` - Index of the coefficient
/// * `variance` - Its variance at the fitted parameters
/// * `hessian` - The `2 x 2` REML Hessian, row-major
///
/// ### Returns
///
/// The degrees of freedom, or `None` when the Hessian is singular or the
/// denominator collapses, in which case the caller should fall back.
fn satterthwaite_df(
    s: &RandomInterceptStats,
    theta: f64,
    sigma: f64,
    coef: usize,
    variance: f64,
    hessian: &[f64; 4],
) -> Option<f64> {
    let det = hessian[0] * hessian[3] - hessian[1] * hessian[2];
    if !det.is_finite() || det.abs() < f64::EPSILON {
        return None;
    }
    // A = 2 * Hess^-1, spelled out because it is only 2 x 2.
    let a = [
        2.0 * hessian[3] / det,
        -2.0 * hessian[1] / det,
        -2.0 * hessian[2] / det,
        2.0 * hessian[0] / det,
    ];

    let phi = [theta, sigma];
    let mut grad = [0.0_f64; 2];
    for (i, gi) in grad.iter_mut().enumerate() {
        let floor = if i == 0 {
            THETA_STEP_FLOOR
        } else {
            SIGMA_STEP_FLOOR
        };
        let h = GRAD_STEP * phi[i].abs().max(floor);
        let mut up = phi;
        let mut down = phi;
        up[i] += h;
        down[i] -= h;
        let f_up = coefficient_variance(s, up[0], up[1], coef);
        let f_down = coefficient_variance(s, down[0], down[1], coef);
        *gi = (f_up - f_down) / (2.0 * h);
    }

    let denom =
        grad[0] * (a[0] * grad[0] + a[1] * grad[1]) + grad[1] * (a[2] * grad[0] + a[3] * grad[1]);
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }
    let nu = 2.0 * variance * variance / denom;
    if nu.is_finite() && nu > 0.0 {
        Some(nu)
    } else {
        None
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Every reference value below came out of lme4 3.1-x with lmerTest on this
    // machine, printed with "%.17g".

    /// The DIALOGUE shape: `y` and `q` vary per observation, `x` and `tq` are
    /// group-level and constant within a group, and the groups are unbalanced.
    /// Seven groups of sizes 7, 4, 9, 5, 6, 3, 8.
    struct Unbalanced {
        groups: Vec<usize>,
        x: Vec<f64>,
        q: Vec<f64>,
        tq: Vec<f64>,
        y: Vec<f64>,
    }

    fn unbalanced_fixture() -> Unbalanced {
        let sizes = [7usize, 4, 9, 5, 6, 3, 8];
        let groups: Vec<usize> = sizes
            .iter()
            .enumerate()
            .flat_map(|(g, &n)| std::iter::repeat_n(g, n))
            .collect();
        let x_s = [-0.326, 0.552, -0.675, 0.214, 0.311, 1.174, 0.619];
        let t_s = [-0.113, 0.917, -0.223, 0.526, -0.795, 1.428, -1.467];
        let q = vec![
            -0.237, -0.193, -0.85, 0.058, -0.818, -2.05, -0.164, 0.709, -0.268, -1.464, 0.744,
            -1.41, 0.467, -0.119, 0.467, 0.498, 0.895, 0.279, 1.008, -2.073, 1.19, -0.724, 0.168,
            0.92, -1.672, 0.448, 0.482, 0.758, -2.319, -0.46, -1.105, 0.403, 0.569, -0.706, -0.29,
            -1.484, -1.15, -0.274, 0.578, -1.397, 0.749, -1.051,
        ];
        let y = vec![
            0.259, 0.493, 0.205, -0.184, 0.052, -0.586, 0.157, 3.132, 2.818, 2.013, 2.035, -1.141,
            0.156, 0.86, -0.012, 0.992, 1.018, 2.104, 1.877, -0.491, 0.352, -1.243, 0.654, 0.629,
            0.532, 1.59, 2.183, 2.135, -0.048, 0.467, -0.042, 0.71, 0.026, -0.242, 2.009, 0.925,
            0.14, 1.248, 0.987, 2.105, 2.29, 1.272,
        ];
        let x: Vec<f64> = groups.iter().map(|&g| x_s[g]).collect();
        let tq: Vec<f64> = groups.iter().map(|&g| t_s[g]).collect();
        Unbalanced {
            groups,
            x,
            q,
            tq,
            y,
        }
    }

    /// Assembles a design from an intercept plus the named columns.
    fn design(cols: &[&[f64]]) -> Mat<f64> {
        let n = cols[0].len();
        Mat::from_fn(
            n,
            cols.len() + 1,
            |i, j| {
                if j == 0 { 1.0 } else { cols[j - 1][i] }
            },
        )
    }

    /// The full DIALOGUE-shaped model against
    /// `lmer(y ~ (1 | samples) + x + q + tq)`.
    ///
    /// The interesting part is the degrees of freedom. `x` and `tq` are
    /// group-level, so Satterthwaite charges them roughly the group count;
    /// `q` varies within a group and gets an order of magnitude more. Anything
    /// that reported a single residual dof for all three would be wrong by a
    /// factor of nine on two of them.
    #[test]
    fn test_random_intercept_matches_lmer_full_model() {
        let f = unbalanced_fixture();
        let x = design(&[&f.x, &f.q, &f.tq]);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 7).unwrap();
        let fit = fit_random_intercept(&stats, None).unwrap();

        assert!(!fit.singular);
        assert_relative_eq!(fit.lambda, 2.0532733375608516, max_relative = 1e-6);
        assert_relative_eq!(fit.sigma_sq.sqrt(), 0.6577853719340665, max_relative = 1e-7);
        assert_relative_eq!(
            fit.sigma_b_sq.sqrt(),
            0.9425569392646459,
            max_relative = 1e-6
        );

        let est = [
            0.837_003_209_793_895_5,
            0.480_850_405_566_252_4,
            0.494_990_133_154_51,
            -0.29819131253327946,
        ];
        let se = [
            0.41114768465916934,
            0.690_886_295_671_207_5,
            0.11240202592823408,
            0.42743196063127253,
        ];
        let df = [
            3.6710855587773943,
            3.6751754158502474,
            33.965_389_734_957_95,
            3.751_245_840_530_402,
        ];
        let p = [
            0.11774406230189201,
            0.527_923_812_686_889_6,
            0.00010056269552554244,
            0.526_216_644_969_979_1,
        ];
        for j in 0..4 {
            assert_relative_eq!(fit.beta[j], est[j], max_relative = 1e-7);
            assert_relative_eq!(fit.se[j], se[j], max_relative = 1e-6);
            assert_relative_eq!(fit.df[j], df[j], max_relative = 1e-5);
            assert_relative_eq!(fit.p_value[j], p[j], max_relative = 1e-5);
        }
    }

    /// Only the group-level predictor: `lmer(y ~ (1 | samples) + x)`.
    #[test]
    fn test_random_intercept_matches_lmer_group_level_predictor() {
        let f = unbalanced_fixture();
        let x = design(&[&f.x]);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 7).unwrap();
        let fit = fit_random_intercept(&stats, None).unwrap();

        assert_relative_eq!(fit.lambda, 1.003_725_272_087_444, max_relative = 1e-6);
        assert_relative_eq!(fit.beta[0], 0.749_953_089_282_345_8, max_relative = 1e-7);
        assert_relative_eq!(fit.beta[1], 0.387_648_044_668_823_9, max_relative = 1e-7);
        assert_relative_eq!(fit.se[1], 0.591_762_242_340_748, max_relative = 1e-6);
        assert_relative_eq!(fit.df[1], 4.544_957_359_657_38, max_relative = 1e-5);
        assert_relative_eq!(fit.p_value[1], 0.544_109_067_513_398_2, max_relative = 1e-5);
    }

    /// Only the within-group covariate: `lmer(y ~ (1 | samples) + q)`. This is
    /// the opposite extreme for Satterthwaite, and the dof lands near the
    /// residual count rather than the group count.
    #[test]
    fn test_random_intercept_matches_lmer_within_group_predictor() {
        let f = unbalanced_fixture();
        let x = design(&[&f.q]);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 7).unwrap();
        let fit = fit_random_intercept(&stats, None).unwrap();

        assert_relative_eq!(fit.lambda, 1.579109868473656, max_relative = 1e-6);
        assert_relative_eq!(fit.beta[1], 0.490_443_581_359_349_5, max_relative = 1e-7);
        assert_relative_eq!(fit.se[1], 0.112_131_054_175_273_7, max_relative = 1e-6);
        assert_relative_eq!(fit.df[1], 34.303_696_450_666_13, max_relative = 1e-5);
        assert_relative_eq!(fit.p_value[1], 0.00010799519833758655, max_relative = 1e-4);
    }

    /// Balanced design, six groups of five.
    fn balanced_fixture() -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        let groups: Vec<usize> = (0..6).flat_map(|g| std::iter::repeat_n(g, 5)).collect();
        let x_s = [-0.859, 0.512, -1.75, -0.74, 1.836, 0.597];
        let y = vec![
            0.056, -0.985, 1.037, 1.177, 0.63, 2.577, 0.817, 1.419, 3.018, 1.283, 3.407, 1.125,
            2.492, 2.931, 1.936, 1.861, 1.881, 1.743, 1.391, 1.295, 2.359, 2.216, 1.244, 4.311,
            3.43, 1.275, -0.281, -0.844, 0.103, 0.872,
        ];
        let x: Vec<f64> = groups.iter().map(|&g| x_s[g]).collect();
        (groups, x, y)
    }

    /// The balanced case has a closed-form REML solution, so this gate needs no
    /// reference implementation at all: `sigma^2` is the within-group mean
    /// square and `sigma_b^2` is `(MS_between - MS_within) / n_per_group`.
    #[test]
    fn test_random_intercept_matches_the_balanced_closed_form() {
        let (groups, _x, y) = balanced_fixture();
        let n = y.len();
        let per = 5.0;
        let x = Mat::<f64>::from_fn(n, 1, |_, _| 1.0);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &y, &groups, 6).unwrap();
        let fit = fit_random_intercept(&stats, None).unwrap();

        // One-way mean squares, computed here rather than quoted.
        let grand: f64 = y.iter().sum::<f64>() / n as f64;
        let mut ss_between = 0.0;
        let mut ss_within = 0.0;
        for g in 0..6 {
            let members: Vec<f64> = (0..n).filter(|&i| groups[i] == g).map(|i| y[i]).collect();
            let mean = members.iter().sum::<f64>() / per;
            ss_between += per * (mean - grand).powi(2);
            ss_within += members.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        }
        let ms_between = ss_between / 5.0;
        let ms_within = ss_within / 24.0;

        assert_relative_eq!(fit.sigma_sq, ms_within, max_relative = 1e-8);
        assert_relative_eq!(
            fit.sigma_b_sq,
            (ms_between - ms_within) / per,
            max_relative = 1e-7
        );
        // And the intercept gets the containment dof, G - 1.
        assert_relative_eq!(fit.df[0], 5.0, max_relative = 1e-6);
    }

    /// A group-level predictor on a balanced design gets `G - 2` degrees of
    /// freedom, which is what `lmerTest` reports.
    #[test]
    fn test_random_intercept_balanced_with_a_group_level_predictor() {
        let (groups, x_col, y) = balanced_fixture();
        let x = design(&[&x_col]);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &y, &groups, 6).unwrap();
        let fit = fit_random_intercept(&stats, None).unwrap();

        assert_relative_eq!(fit.beta[0], 1.5336703600487156, max_relative = 1e-7);
        assert_relative_eq!(fit.beta[1], 0.11589643636704265, max_relative = 1e-6);
        assert_relative_eq!(fit.se[1], 0.39246534408354555, max_relative = 1e-6);
        assert_relative_eq!(fit.df[1], 4.0, max_relative = 1e-5);
        assert_relative_eq!(fit.p_value[1], 0.782_455_988_812_368_8, max_relative = 1e-5);
    }

    /// No between-group variance at all: the optimum sits on the boundary,
    /// `lme4` calls the fit singular, and the model collapses to ordinary least
    /// squares with the residual degrees of freedom.
    ///
    /// The dof falls out of the general formula rather than a special case,
    /// because the criterion depends on `theta` only through `theta^2`.
    #[test]
    fn test_random_intercept_singular_fit() {
        let groups: Vec<usize> = (0..5).flat_map(|g| std::iter::repeat_n(g, 6)).collect();
        let x_s = [-0.629, -0.871, 0.046, 0.672, -0.262];
        let y = vec![
            -0.497, 0.279, 0.384, 0.069, -0.578, 1.314, -1.261, -0.727, -0.231, 0.608, 1.022,
            0.944, 0.945, 0.849, -0.728, -0.984, 0.774, -0.494, -0.452, 0.556, 0.564, 0.358,
            -0.984, -0.678, 0.779, 0.148, -0.203, -1.111, 2.31, 0.67,
        ];
        let x_col: Vec<f64> = groups.iter().map(|&g| x_s[g]).collect();
        let x = design(&[&x_col]);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &y, &groups, 5).unwrap();
        let fit = fit_random_intercept(&stats, None).unwrap();

        assert!(fit.singular);
        assert_eq!(fit.lambda, 0.0);
        assert_eq!(fit.sigma_b_sq, 0.0);
        assert_relative_eq!(
            fit.sigma_sq.sqrt(),
            0.854_108_154_752_631_7,
            max_relative = 1e-8
        );
        assert_relative_eq!(fit.beta[0], 0.091_701_676_247_745_51, max_relative = 1e-7);
        assert_relative_eq!(fit.beta[1], -0.14271227850696577, max_relative = 1e-7);
        assert_relative_eq!(fit.se[1], 0.28871102086056694, max_relative = 1e-7);
        // n - p = 30 - 2.
        assert_relative_eq!(fit.df[1], 28.0, max_relative = 1e-6);
        assert_relative_eq!(fit.p_value[1], 0.624_944_411_719_467_6, max_relative = 1e-6);
    }

    /// Switching Satterthwaite off leaves the estimates alone and falls back to
    /// the residual degrees of freedom.
    #[test]
    fn test_random_intercept_without_satterthwaite() {
        let f = unbalanced_fixture();
        let x = design(&[&f.x, &f.q, &f.tq]);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 7).unwrap();
        let with = fit_random_intercept(&stats, None).unwrap();
        let without = fit_random_intercept(
            &stats,
            Some(RandomInterceptParams {
                satterthwaite: false,
                ..Default::default()
            }),
        )
        .unwrap();

        for j in 0..4 {
            assert_eq!(with.beta[j], without.beta[j]);
            assert_eq!(with.se[j], without.se[j]);
            assert_eq!(without.df[j], 38.0);
        }
        // The within-group covariate is the one the two nearly agree on.
        assert!((with.df[2] - 38.0).abs() < 5.0);
        // The group-level ones are nowhere near it.
        assert!(with.df[1] < 5.0);
    }

    /// Empty levels are dropped rather than counted, so widening the level set
    /// changes nothing.
    #[test]
    fn test_random_intercept_ignores_empty_groups() {
        let f = unbalanced_fixture();
        let x = design(&[&f.x, &f.q]);
        let tight = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 7).unwrap();
        let widened = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 20).unwrap();
        assert_eq!(tight.n_groups(), widened.n_groups());
        let a = fit_random_intercept(&tight, None).unwrap();
        let b = fit_random_intercept(&widened, None).unwrap();
        assert_eq!(a.beta, b.beta);
        assert_eq!(a.df, b.df);
    }

    /// The degrees of freedom must not depend on the scale of the response.
    ///
    /// `y -> c*y` leaves `lambda` and every dof mathematically unchanged. This
    /// caught a real bug: flooring the `sigma` finite-difference step at an
    /// absolute 1.0 made the step a 15x perturbation once `sigma` fell below
    /// 1e-4, and the reported dof silently collapsed. DIALOGUE feeds this
    /// arbitrarily scaled projections, so the small-`sigma` regime is normal.
    #[test]
    fn test_random_intercept_df_is_scale_invariant() {
        let f = unbalanced_fixture();
        let x = design(&[&f.x, &f.q, &f.tq]);
        let base = {
            let stats = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 7).unwrap();
            fit_random_intercept(&stats, None).unwrap()
        };

        for scale in [1e-6, 1e-3, 1e3, 1e6] {
            let scaled: Vec<f64> = f.y.iter().map(|v| v * scale).collect();
            let stats =
                RandomInterceptStats::from_design(x.as_ref(), &scaled, &f.groups, 7).unwrap();
            let fit = fit_random_intercept(&stats, None).unwrap();

            assert_relative_eq!(fit.lambda, base.lambda, max_relative = 1e-6);
            for j in 0..4 {
                assert_relative_eq!(fit.df[j], base.df[j], max_relative = 1e-4);
                assert_relative_eq!(fit.p_value[j], base.p_value[j], max_relative = 1e-4);
                // The estimates themselves scale linearly.
                assert_relative_eq!(fit.beta[j], base.beta[j] * scale, max_relative = 1e-8);
            }
        }
    }

    /// A variance ratio past the bracket withholds the inference.
    #[test]
    fn test_random_intercept_flags_an_exhausted_bracket() {
        let groups: Vec<usize> = (0..8).flat_map(|g| std::iter::repeat_n(g, 5)).collect();
        // Group means an order of magnitude apart, jitter at 1e-7.
        let y: Vec<f64> = groups
            .iter()
            .enumerate()
            .map(|(i, &g)| g as f64 + (i % 5) as f64 * 1e-7)
            .collect();
        let x = Mat::<f64>::from_fn(y.len(), 1, |_, _| 1.0);
        let stats = RandomInterceptStats::from_design(x.as_ref(), &y, &groups, 8).unwrap();
        let fit = fit_random_intercept(&stats, None).unwrap();

        assert!(fit.bracket_exhausted, "lambda was {}", fit.lambda);
        assert!(fit.df[0].is_nan());
        assert!(fit.p_value[0].is_nan());
        // The estimate itself is still meaningful and still reported.
        assert!(fit.beta[0].is_finite());
        assert_relative_eq!(fit.beta[0], 3.5, max_relative = 1e-6);
    }

    /// A well-conditioned design must not be called rank deficient just for
    /// being small.
    #[test]
    fn test_random_intercept_accepts_a_small_scale_design() {
        let f = unbalanced_fixture();
        for scale in [1e-7, 1e-9] {
            let scaled_x: Vec<f64> = f.x.iter().map(|v| v * scale).collect();
            let scaled_q: Vec<f64> = f.q.iter().map(|v| v * scale).collect();
            let x = Mat::<f64>::from_fn(f.y.len(), 3, |i, j| match j {
                0 => scale,
                1 => scaled_x[i],
                _ => scaled_q[i],
            });
            let stats = RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 7).unwrap();
            assert!(
                fit_random_intercept(&stats, None).is_ok(),
                "rejected a full-rank design scaled by {scale}"
            );
        }
    }

    #[test]
    fn test_random_intercept_rejects_bad_input() {
        let f = unbalanced_fixture();
        let x = design(&[&f.x]);
        // Response and design of different lengths.
        assert!(RandomInterceptStats::from_design(x.as_ref(), &f.y[..10], &f.groups, 7).is_err());
        // Group code past the declared level count.
        assert!(RandomInterceptStats::from_design(x.as_ref(), &f.y, &f.groups, 3).is_err());
        // A single group leaves nothing for the random intercept to explain.
        let one = vec![0usize; f.y.len()];
        assert!(RandomInterceptStats::from_design(x.as_ref(), &f.y, &one, 1).is_err());
        // A duplicated column is rank deficient.
        let dup = design(&[&f.x, &f.x]);
        let stats = RandomInterceptStats::from_design(dup.as_ref(), &f.y, &f.groups, 7).unwrap();
        assert!(fit_random_intercept(&stats, None).is_err());
        // A non-finite entry must error rather than be reported as a singular
        // fit, which is what it silently looked like before.
        let mut poisoned = f.y.clone();
        poisoned[3] = f64::NAN;
        assert!(RandomInterceptStats::from_design(x.as_ref(), &poisoned, &f.groups, 7).is_err());
        // A single group leaves the random intercept unidentified.
        let single = RandomInterceptStats {
            xtx: Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 4.0 } else { 1.0 }),
            xty: vec![1.0, 2.0],
            yty: 10.0,
            n: 4,
            p: 2,
            group_n: vec![4.0],
            group_s: Mat::<f64>::from_fn(1, 2, |_, _| 1.0),
            group_t: vec![1.0],
        };
        assert!(fit_random_intercept(&single, None).is_err());
    }
}
