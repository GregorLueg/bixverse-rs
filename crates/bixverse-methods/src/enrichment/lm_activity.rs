//! Univariate and multivariate linear model activity scores (ULM, MLM), see
//! Badia-i-Mompel, et al., Bioinform. Adv., 2022 (decoupler).
//!
//! Every observation (a sample, a cell) gets its own fit with the features as
//! the data points: `y = b0 + W b + e`, `W` holding the feature set weights.
//! ULM fits one set at a time, MLM all sets jointly, so features shared
//! between sets are credited to the set that explains them. The design is the
//! same for every observation, so the fits reduce to `Wᵀy`, `Σy` and `Σy²`.
//! Features with `y = 0` add to none of them, so a sparse observation only
//! touches its non-zeros.
//!
//! Sums run in f64 whatever the input: `Σy² - (Σy)² / n` and `yᵀy - βᵀXᵀy`
//! both cancel.

use faer::{Mat, MatRef, Side};
use rayon::prelude::*;

use crate::prelude::*;
use bixverse_core::core::math::distributions::t_pval_two_sided;

/// decoupler's guard in `t = r sqrt(df / ((1 - r) (1 + r)))`. Kept so a
/// perfect correlation gives the same large finite t as the reference.
const ULM_R_EPS: f64 = 2.2e-16;

/// Smallest accepted ratio of the extreme eigenvalues of `XᵀX` before the MLM
/// design counts as rank deficient. `XᵀX` squares the condition number of
/// `X`, so this lets `X` reach a condition number of 1e5. Not tuned.
const MLM_RANK_TOL: f64 = 1e-10;

///////////
// Enums //
///////////

/// Which linear model to fit per observation
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LmMethod {
    /// Univariate linear model: one fit per set and observation.
    #[default]
    Ulm,
    /// Multivariate linear model: one joint fit of all sets per observation.
    Mlm,
}

/// Parse the linear model
///
/// ### Params
///
/// * `s` - String specifying the method.
///
/// ### Returns
///
/// The Option of the `LmMethod`
pub fn parse_lm_method(s: &str) -> Option<LmMethod> {
    match s.to_lowercase().as_str() {
        "ulm" => Some(LmMethod::Ulm),
        "mlm" => Some(LmMethod::Mlm),
        _ => None,
    }
}

/// Which statistic to report as the activity score
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LmStat {
    /// t value of the slope, decoupler's default.
    #[default]
    TValue,
    /// The slope itself.
    Coefficient,
}

/// Parse the reported statistic
///
/// ### Params
///
/// * `s` - String specifying the statistic.
///
/// ### Returns
///
/// The Option of the `LmStat`
pub fn parse_lm_stat(s: &str) -> Option<LmStat> {
    match s.to_lowercase().as_str() {
        "t" | "tval" | "t_value" => Some(LmStat::TValue),
        "coef" | "coefficient" | "beta" => Some(LmStat::Coefficient),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Parameters for the linear model activity scores
#[derive(Clone, Copy, Debug, Default)]
pub struct LmActivityParams {
    /// ULM or MLM.
    pub method: LmMethod,
    /// Statistic reported as the score.
    pub stat: LmStat,
    /// Also return two-sided p-values from the t distribution. Off by default:
    /// they treat the features as independent draws, which they are not, and
    /// on single cell data they double the output.
    pub p_values: bool,
}

impl LmActivityParams {
    /// Construct the parameters.
    ///
    /// ### Params
    ///
    /// * `method` - ULM or MLM.
    /// * `stat` - Statistic reported as the score.
    /// * `p_values` - Also return p-values.
    ///
    /// ### Returns
    ///
    /// The populated [LmActivityParams]
    pub fn new(method: LmMethod, stat: LmStat, p_values: bool) -> Self {
        Self {
            method,
            stat,
            p_values,
        }
    }
}

/////////////
// Results //
/////////////

/// Activity scores, observations x sets
#[derive(Clone, Debug)]
pub struct LmActivityRes<T> {
    /// The scores (t values or slopes).
    pub es: Mat<T>,
    /// Two-sided p-values, if asked for. Not multiple testing corrected.
    pub p_values: Option<Mat<T>>,
}

///////////////////////
// FeatureSetWeights //
///////////////////////

/// Weighted feature sets, stored CSR by feature
///
/// The per-observation pass walks the non-zero features of the observation
/// and needs every set a feature belongs to, hence feature-major.
#[derive(Clone, Debug)]
pub struct FeatureSetWeights {
    /// Size of the feature universe
    pub n_features: usize,
    /// Number of feature sets
    pub n_sets: usize,
    /// Offsets into `set_idx` and `weights`, length `n_features + 1`
    indptr: Vec<usize>,
    /// Set index of each entry
    set_idx: Vec<usize>,
    /// Weight of each entry
    weights: Vec<f64>,
}

impl FeatureSetWeights {
    /// Build the feature sets
    ///
    /// ### Params
    ///
    /// * `sets` - Feature indices (0-based) per set.
    /// * `weights` - Weights per set, aligned with `sets`. `None` gives every
    ///   member weight 1, i.e. plain gene sets.
    /// * `n_features` - Size of the feature universe.
    ///
    /// ### Returns
    ///
    /// The feature sets, or an error for an empty or out of range set, a
    /// repeated feature within a set, a non-finite weight or misaligned
    /// weights.
    pub fn new<T: BixverseFloat>(
        sets: &[Vec<usize>],
        weights: Option<&[Vec<T>]>,
        n_features: usize,
    ) -> Result<Self, MethodsErrors> {
        if sets.is_empty() {
            return Err(MethodsErrors::FeatureSetInvalid {
                set: 0,
                reason: "no feature sets were supplied",
            });
        }
        if let Some(w) = weights
            && w.len() != sets.len()
        {
            return Err(MethodsErrors::FeatureSetInvalid {
                set: w.len().min(sets.len()),
                reason: "weights need one vector per set",
            });
        }

        let mut indptr = vec![0_usize; n_features + 1];
        // last set that used a feature, so a repeat inside a set is O(1)
        let mut last_set = vec![usize::MAX; n_features];
        for (s, idx) in sets.iter().enumerate() {
            let invalid = |reason| MethodsErrors::FeatureSetInvalid { set: s, reason };
            if idx.is_empty() {
                return Err(invalid("it has no features"));
            }
            let w = weights.map(|w| &w[s]);
            if let Some(w) = w {
                if w.len() != idx.len() {
                    return Err(invalid("weights and feature indices differ in length"));
                }
                if w.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("it carries a non-finite weight"));
                }
            }
            for &i in idx {
                if i >= n_features {
                    return Err(MethodsErrors::FeatureSetIndexOutOfRange {
                        set: s,
                        index: i,
                        n_features,
                    });
                }
                if last_set[i] == s {
                    return Err(invalid("it repeats a feature"));
                }
                last_set[i] = s;
                indptr[i + 1] += 1;
            }
        }
        for i in 0..n_features {
            indptr[i + 1] += indptr[i];
        }

        let nnz = indptr[n_features];
        let mut set_idx = vec![0_usize; nnz];
        let mut entry_weights = vec![0.0_f64; nnz];
        let mut cursor = indptr[..n_features].to_vec();
        for (s, idx) in sets.iter().enumerate() {
            for (k, &i) in idx.iter().enumerate() {
                let pos = cursor[i];
                set_idx[pos] = s;
                entry_weights[pos] = weights.map_or(1.0, |w| {
                    w[s][k].to_f64().expect("finite float converts to f64")
                });
                cursor[i] += 1;
            }
        }

        Ok(Self {
            n_features,
            n_sets: sets.len(),
            indptr,
            set_idx,
            weights: entry_weights,
        })
    }

    /// Cross products of one observation with the sets
    ///
    /// ### Params
    ///
    /// * `values` - `(feature, value)` pairs of the observation. Absent
    ///   features count as zero, so a sparse observation passes only its
    ///   non-zeros.
    /// * `xty` - Output buffer of length `n_sets`, overwritten with `Wᵀy`.
    ///
    /// ### Returns
    ///
    /// `(Σy, Σy²)`
    pub fn cross_products(
        &self,
        values: impl Iterator<Item = (usize, f64)>,
        xty: &mut [f64],
    ) -> (f64, f64) {
        xty.fill(0.0);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for (i, y) in values {
            sum += y;
            sum_sq += y * y;
            for k in self.indptr[i]..self.indptr[i + 1] {
                xty[self.set_idx[k]] += y * self.weights[k];
            }
        }
        (sum, sum_sq)
    }
}

////////////////////
// Cross products //
////////////////////

/// Cross products of a batch of observations with the feature sets
#[derive(Clone, Debug)]
pub struct ObsCrossProducts {
    /// `Wᵀy` per observation, row-major observations x sets
    pub xty: Vec<f64>,
    /// `Σy` per observation
    pub sum: Vec<f64>,
    /// `Σy²` per observation
    pub sum_sq: Vec<f64>,
}

impl ObsCrossProducts {
    /// Compute the cross products of a batch of observations, in parallel
    ///
    /// ### Params
    ///
    /// * `net` - The feature sets.
    /// * `n_obs` - Number of observations in the batch.
    /// * `values` - Returns the `(feature, value)` pairs of observation `o`,
    ///   see [FeatureSetWeights::cross_products].
    ///
    /// ### Returns
    ///
    /// The cross products of the batch
    pub fn new<F, I>(net: &FeatureSetWeights, n_obs: usize, values: F) -> Self
    where
        F: Fn(usize) -> I + Sync,
        I: Iterator<Item = (usize, f64)>,
    {
        let mut xty = vec![0.0_f64; n_obs * net.n_sets];
        let (sum, sum_sq): (Vec<f64>, Vec<f64>) = xty
            .par_chunks_mut(net.n_sets)
            .enumerate()
            .map(|(o, row)| net.cross_products(values(o), row))
            .unzip();
        Self { xty, sum, sum_sq }
    }

    /// Number of observations in the batch
    ///
    /// ### Returns
    ///
    /// The number of observations
    pub fn n_obs(&self) -> usize {
        self.sum.len()
    }
}

////////////
// Models //
////////////

/// Per-set statistics for ULM
#[derive(Clone, Debug)]
struct UlmModel {
    /// Number of features
    n: f64,
    /// Residual degrees of freedom, `n - 2`
    df: f64,
    /// Mean weight per set over all features
    w_mean: Vec<f64>,
    /// Centred sum of squares of the weights per set
    w_ss: Vec<f64>,
}

impl UlmModel {
    /// Prepare the per-set statistics
    ///
    /// ### Params
    ///
    /// * `net` - The feature sets.
    ///
    /// ### Returns
    ///
    /// The model, or an error if there are fewer than three features or a set
    /// has the same weight on every feature.
    fn new(net: &FeatureSetWeights) -> Result<Self, MethodsErrors> {
        if net.n_features <= 2 {
            return Err(MethodsErrors::LmActivityDegreesOfFreedom {
                n_features: net.n_features,
                n_params: 2,
            });
        }
        let n = net.n_features as f64;
        let mut w_sum = vec![0.0_f64; net.n_sets];
        let mut w_sq = vec![0.0_f64; net.n_sets];
        for (&s, &w) in net.set_idx.iter().zip(&net.weights) {
            w_sum[s] += w;
            w_sq[s] += w * w;
        }
        let w_mean: Vec<f64> = w_sum.iter().map(|&s| s / n).collect();
        let w_ss: Vec<f64> = w_sq
            .iter()
            .zip(&w_sum)
            .map(|(&sq, &s)| sq - s * s / n)
            .collect();
        if let Some(set) = w_ss.iter().position(|&x| x <= 0.0) {
            return Err(MethodsErrors::FeatureSetInvalid {
                set,
                reason: "its weights are constant across the features",
            });
        }
        Ok(Self {
            n,
            df: n - 2.0,
            w_mean,
            w_ss,
        })
    }

    /// Score one observation
    ///
    /// An observation without variance (an empty cell) scores 0 with p = 1.
    ///
    /// ### Params
    ///
    /// * `xty` - `Wᵀy` of the observation.
    /// * `sum` - `Σy`.
    /// * `sum_sq` - `Σy²`.
    /// * `stat` - Statistic to report.
    /// * `es` - Output scores, length `n_sets`.
    /// * `pv` - Optional output p-values, length `n_sets`.
    fn score_obs(
        &self,
        xty: &[f64],
        sum: f64,
        sum_sq: f64,
        stat: LmStat,
        es: &mut [f64],
        mut pv: Option<&mut [f64]>,
    ) {
        let syy = sum_sq - sum * sum / self.n;
        for j in 0..xty.len() {
            let (score, p) = if syy <= 0.0 {
                (0.0, 1.0)
            } else {
                let sxy = xty[j] - sum * self.w_mean[j];
                let r = (sxy / (syy * self.w_ss[j]).sqrt()).clamp(-1.0, 1.0);
                let t = r * (self.df / ((1.0 - r + ULM_R_EPS) * (1.0 + r + ULM_R_EPS))).sqrt();
                let score = match stat {
                    LmStat::TValue => t,
                    LmStat::Coefficient => sxy / self.w_ss[j],
                };
                let p = if pv.is_some() {
                    t_pval_two_sided(t, self.df).expect("df > 0 checked in UlmModel::new")
                } else {
                    1.0
                };
                (score, p)
            };
            es[j] = score;
            if let Some(pv) = pv.as_deref_mut() {
                pv[j] = p;
            }
        }
    }
}

/// Shared inverse Gram matrix for MLM
#[derive(Clone, Debug)]
struct MlmModel {
    /// `(XᵀX)⁻¹` with `X = [1, W]`, `(n_sets + 1)` squared
    g: Mat<f64>,
    /// Residual degrees of freedom, `n_features - n_sets - 1`
    df: f64,
}

impl MlmModel {
    /// Build and invert `XᵀX`
    ///
    /// `WᵀW` comes from the feature-major storage, one outer product per
    /// feature over the sets it belongs to. The inverse goes through the
    /// symmetric eigendecomposition so the rank check falls out of it.
    ///
    /// ### Params
    ///
    /// * `net` - The feature sets.
    ///
    /// ### Returns
    ///
    /// The model, or an error if no degrees of freedom remain or the design is
    /// rank deficient.
    fn new(net: &FeatureSetWeights) -> Result<Self, MethodsErrors> {
        let p = net.n_sets + 1;
        if net.n_features <= p {
            return Err(MethodsErrors::LmActivityDegreesOfFreedom {
                n_features: net.n_features,
                n_params: p,
            });
        }

        let mut xtx = Mat::<f64>::zeros(p, p);
        xtx[(0, 0)] = net.n_features as f64;
        for i in 0..net.n_features {
            let range = net.indptr[i]..net.indptr[i + 1];
            for a in range.clone() {
                let (sa, wa) = (net.set_idx[a] + 1, net.weights[a]);
                xtx[(0, sa)] += wa;
                xtx[(sa, 0)] += wa;
                for b in range.clone() {
                    xtx[(sa, net.set_idx[b] + 1)] += wa * net.weights[b];
                }
            }
        }

        let eig = xtx
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| MethodsErrors::from(CoreErrors::FaerEigenError))?;
        let lambda = eig.S().column_vector();
        let u = eig.U();
        let (min, max) = lambda
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &l| {
                (lo.min(l), hi.max(l))
            });
        let ratio = min / max;
        if ratio.is_nan() || ratio <= MLM_RANK_TOL {
            return Err(MethodsErrors::LmActivityRankDeficient { ratio });
        }
        let u_scaled = Mat::<f64>::from_fn(p, p, |i, j| u[(i, j)] / lambda[j]);
        let g = &u_scaled * u.transpose();

        Ok(Self {
            g,
            df: (net.n_features - p) as f64,
        })
    }

    /// Score a batch of observations
    ///
    /// `β = G [Σy, Wᵀy]` for the whole batch is one GEMM against the shared
    /// inverse; the residuals then follow from `SSE = yᵀy - βᵀXᵀy`. An
    /// observation without residual variance (an empty cell) scores 0 with
    /// p = 1.
    ///
    /// ### Params
    ///
    /// * `cp` - Cross products of the batch.
    /// * `stat` - Statistic to report.
    /// * `es` - Output scores, row-major observations x sets.
    /// * `pv` - Optional output p-values, same layout.
    fn score_batch(
        &self,
        cp: &ObsCrossProducts,
        stat: LmStat,
        es: &mut [f64],
        pv: Option<&mut [f64]>,
    ) {
        let n_obs = cp.n_obs();
        let n_sets = self.g.nrows() - 1;
        let xty = MatRef::from_row_major_slice(&cp.xty, n_obs, n_sets);
        // the intercept row of G is added per observation below
        let beta_sets = xty * self.g.subrows(1, n_sets);
        let g = &self.g;

        let score_row = |o: usize, es_row: &mut [f64], mut pv_row: Option<&mut [f64]>| {
            let sum = cp.sum[o];
            let xty_row = &cp.xty[o * n_sets..(o + 1) * n_sets];
            let beta = |j: usize| beta_sets[(o, j)] + sum * g[(0, j)];

            let mut fitted = beta(0) * sum;
            for j in 0..n_sets {
                fitted += beta(j + 1) * xty_row[j];
            }
            let sse = cp.sum_sq[o] - fitted;
            let sigma2 = sse / self.df;

            for j in 0..n_sets {
                let (score, p) = if sse <= 0.0 {
                    (0.0, 1.0)
                } else {
                    let b = beta(j + 1);
                    let t = b / (sigma2 * g[(j + 1, j + 1)]).sqrt();
                    let score = match stat {
                        LmStat::TValue => t,
                        LmStat::Coefficient => b,
                    };
                    let p = if pv_row.is_some() {
                        t_pval_two_sided(t, self.df).expect("df > 0 checked in MlmModel::new")
                    } else {
                        1.0
                    };
                    (score, p)
                };
                es_row[j] = score;
                if let Some(pv_row) = pv_row.as_deref_mut() {
                    pv_row[j] = p;
                }
            }
        };

        match pv {
            Some(pv) => es
                .par_chunks_mut(n_sets)
                .zip(pv.par_chunks_mut(n_sets))
                .enumerate()
                .for_each(|(o, (e, p))| score_row(o, e, Some(p))),
            None => es
                .par_chunks_mut(n_sets)
                .enumerate()
                .for_each(|(o, e)| score_row(o, e, None)),
        }
    }
}

/// A prepared ULM or MLM, reusable across batches of observations
#[derive(Clone, Debug)]
pub struct LmScorer {
    /// The prepared model
    model: LmModel,
    /// Statistic to report
    stat: LmStat,
    /// Whether to compute p-values
    p_values: bool,
}

/// The two prepared models behind [LmScorer]
#[derive(Clone, Debug)]
enum LmModel {
    /// Univariate linear model
    Ulm(UlmModel),
    /// Multivariate linear model
    Mlm(MlmModel),
}

impl LmScorer {
    /// Prepare the model for a set of feature sets
    ///
    /// ### Params
    ///
    /// * `net` - The feature sets.
    /// * `params` - Method, statistic and p-value switch.
    ///
    /// ### Returns
    ///
    /// The scorer, or an error if the design cannot be fitted.
    pub fn new(net: &FeatureSetWeights, params: &LmActivityParams) -> Result<Self, MethodsErrors> {
        let model = match params.method {
            LmMethod::Ulm => LmModel::Ulm(UlmModel::new(net)?),
            LmMethod::Mlm => LmModel::Mlm(MlmModel::new(net)?),
        };
        Ok(Self {
            model,
            stat: params.stat,
            p_values: params.p_values,
        })
    }

    /// Score a batch of observations
    ///
    /// ### Params
    ///
    /// * `cp` - Cross products of the batch.
    ///
    /// ### Returns
    ///
    /// Tuple of the scores and the optional p-values, each row-major
    /// observations x sets.
    pub fn score(&self, cp: &ObsCrossProducts) -> (Vec<f64>, Option<Vec<f64>>) {
        let len = cp.xty.len();
        let n_sets = len / cp.n_obs().max(1);
        let mut es = vec![0.0_f64; len];
        let mut pv = self.p_values.then(|| vec![1.0_f64; len]);

        match &self.model {
            LmModel::Ulm(m) => {
                let score_row = |o: usize, e: &mut [f64], p: Option<&mut [f64]>| {
                    let xty = &cp.xty[o * n_sets..(o + 1) * n_sets];
                    m.score_obs(xty, cp.sum[o], cp.sum_sq[o], self.stat, e, p);
                };
                match pv.as_deref_mut() {
                    Some(pv) => es
                        .par_chunks_mut(n_sets)
                        .zip(pv.par_chunks_mut(n_sets))
                        .enumerate()
                        .for_each(|(o, (e, p))| score_row(o, e, Some(p))),
                    None => es
                        .par_chunks_mut(n_sets)
                        .enumerate()
                        .for_each(|(o, e)| score_row(o, e, None)),
                }
            }
            LmModel::Mlm(m) => m.score_batch(cp, self.stat, &mut es, pv.as_deref_mut()),
        }

        (es, pv)
    }
}

//////////
// Main //
//////////

/// ULM or MLM activity scores of a dense matrix
///
/// ### Params
///
/// * `mat` - Features x observations, e.g. log expression of genes x samples
///   or a gene-level statistic (t values of a contrast) as a single column.
/// * `net` - The feature sets over the rows of `mat`.
/// * `params` - Method, statistic and p-value switch. `None` gives ULM t
///   values without p-values.
///
/// ### Returns
///
/// The scores, observations x sets.
pub fn lm_activity<T: BixverseFloat>(
    mat: MatRef<T>,
    net: &FeatureSetWeights,
    params: Option<LmActivityParams>,
) -> Result<LmActivityRes<T>, MethodsErrors> {
    let params = params.unwrap_or_default();
    if mat.nrows() != net.n_features {
        return Err(MethodsErrors::LmActivityShapeMismatch {
            expected: net.n_features,
            got: mat.nrows(),
        });
    }
    let scorer = LmScorer::new(net, &params)?;

    let cp = ObsCrossProducts::new(net, mat.ncols(), |o| {
        mat.col(o)
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f64().expect("float converts to f64")))
    });
    let (es, pv) = scorer.score(&cp);

    let n_sets = net.n_sets;
    let to_mat = |v: Vec<f64>| {
        Mat::from_fn(mat.ncols(), n_sets, |i, j| {
            T::from_f64(v[i * n_sets + j]).expect("f64 converts to float")
        })
    };
    Ok(LmActivityRes {
        es: to_mat(es),
        p_values: pv.map(to_mat),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Small deterministic features x observations matrix
    ///
    /// ### Params
    ///
    /// * `n_features` - Number of rows.
    /// * `n_obs` - Number of columns.
    ///
    /// ### Returns
    ///
    /// The matrix
    fn toy_mat(n_features: usize, n_obs: usize) -> Mat<f64> {
        Mat::from_fn(n_features, n_obs, |i, j| {
            (((i * 7 + j * 13) % 11) as f64) * 0.3 + (i % 3) as f64 * (j as f64 + 1.0) * 0.1
        })
    }

    /// Plain OLS t values via the normal equations on a dense design
    ///
    /// ### Params
    ///
    /// * `x` - Design including the intercept column.
    /// * `y` - Response.
    ///
    /// ### Returns
    ///
    /// t values of all coefficients
    fn ols_t(x: &Mat<f64>, y: &[f64]) -> Vec<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let yv = Mat::from_fn(n, 1, |i, _| y[i]);
        let xtx = x.transpose() * x;
        let inv = xtx.self_adjoint_eigen(Side::Lower).unwrap();
        let lam = inv.S().column_vector();
        let u = inv.U();
        let us = Mat::from_fn(p, p, |i, j| u[(i, j)] / lam[j]);
        let g = &us * u.transpose();
        let beta = &g * (x.transpose() * &yv);
        let resid = &yv - x * &beta;
        let sse: f64 = (0..n).map(|i| resid[(i, 0)] * resid[(i, 0)]).sum();
        let s2 = sse / (n - p) as f64;
        (0..p)
            .map(|j| beta[(j, 0)] / (s2 * g[(j, j)]).sqrt())
            .collect()
    }

    #[test]
    fn test_mlm_matches_dense_ols() {
        let n = 40;
        let mat = toy_mat(n, 3);
        let sets = vec![
            vec![0, 3, 5, 9, 12],
            vec![3, 5, 20, 21, 30, 31],
            vec![1, 2, 39],
        ];
        let weights = vec![
            vec![1.0, -0.5, 2.0, 1.0, 0.3],
            vec![1.0, 1.0, -1.0, 0.5, 0.5, 1.0],
            vec![2.0, 1.0, -1.0],
        ];
        let net = FeatureSetWeights::new(&sets, Some(&weights), n).unwrap();
        let params = LmActivityParams::new(LmMethod::Mlm, LmStat::TValue, false);
        let res = lm_activity(mat.as_ref(), &net, Some(params)).unwrap();

        let x = Mat::from_fn(n, 4, |i, j| {
            if j == 0 {
                return 1.0;
            }
            sets[j - 1]
                .iter()
                .position(|&g| g == i)
                .map_or(0.0, |k| weights[j - 1][k])
        });
        for o in 0..3 {
            let y: Vec<f64> = (0..n).map(|i| mat[(i, o)]).collect();
            let t = ols_t(&x, &y);
            for s in 0..3 {
                assert_relative_eq!(res.es[(o, s)], t[s + 1], max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn test_ulm_equals_single_set_mlm() {
        let n = 30;
        let mat = toy_mat(n, 4);
        let sets = vec![vec![0, 4, 8, 15, 16, 23]];
        let net = FeatureSetWeights::new::<f64>(&sets, None, n).unwrap();
        let ulm = lm_activity(
            mat.as_ref(),
            &net,
            Some(LmActivityParams::new(LmMethod::Ulm, LmStat::TValue, true)),
        )
        .unwrap();
        let mlm = lm_activity(
            mat.as_ref(),
            &net,
            Some(LmActivityParams::new(LmMethod::Mlm, LmStat::TValue, true)),
        )
        .unwrap();
        for o in 0..4 {
            assert_relative_eq!(ulm.es[(o, 0)], mlm.es[(o, 0)], max_relative = 1e-9);
            assert_relative_eq!(
                ulm.p_values.as_ref().unwrap()[(o, 0)],
                mlm.p_values.as_ref().unwrap()[(o, 0)],
                max_relative = 1e-9
            );
        }
    }

    #[test]
    fn test_sparse_input_matches_dense() {
        let n = 50;
        let mat = Mat::from_fn(n, 5, |i, j| {
            if (i + j) % 3 == 0 {
                (i + 2 * j) as f64 * 0.1
            } else {
                0.0
            }
        });
        let sets = vec![vec![0, 3, 6, 9, 10], vec![1, 2, 3, 4, 5, 40]];
        let net = FeatureSetWeights::new::<f64>(&sets, None, n).unwrap();
        for method in [LmMethod::Ulm, LmMethod::Mlm] {
            let params = LmActivityParams::new(method, LmStat::TValue, false);
            let dense = lm_activity(mat.as_ref(), &net, Some(params)).unwrap();
            let scorer = LmScorer::new(&net, &params).unwrap();
            let m = mat.as_ref();
            let cp = ObsCrossProducts::new(&net, 5, |o| {
                (0..n)
                    .filter(move |&i| m[(i, o)] != 0.0)
                    .map(move |i| (i, m[(i, o)]))
            });
            let (es, _) = scorer.score(&cp);
            for o in 0..5 {
                for s in 0..2 {
                    assert_relative_eq!(dense.es[(o, s)], es[o * 2 + s], max_relative = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_empty_observation_scores_zero() {
        let n = 20;
        let mat = Mat::<f64>::zeros(n, 1);
        let net = FeatureSetWeights::new::<f64>(&[vec![0, 1, 2]], None, n).unwrap();
        for method in [LmMethod::Ulm, LmMethod::Mlm] {
            let params = LmActivityParams::new(method, LmStat::TValue, true);
            let res = lm_activity(mat.as_ref(), &net, Some(params)).unwrap();
            assert_eq!(res.es[(0, 0)], 0.0);
            assert_eq!(res.p_values.unwrap()[(0, 0)], 1.0);
        }
    }

    #[test]
    fn test_mlm_rejects_duplicate_sets() {
        let sets = vec![vec![0, 1, 2], vec![0, 1, 2]];
        let net = FeatureSetWeights::new::<f64>(&sets, None, 20).unwrap();
        let params = LmActivityParams::new(LmMethod::Mlm, LmStat::TValue, false);
        assert!(matches!(
            LmScorer::new(&net, &params),
            Err(MethodsErrors::LmActivityRankDeficient { .. })
        ));
    }

    #[test]
    fn test_feature_set_validation() {
        assert!(matches!(
            FeatureSetWeights::new::<f64>(&[vec![0, 25]], None, 20),
            Err(MethodsErrors::FeatureSetIndexOutOfRange { index: 25, .. })
        ));
        assert!(matches!(
            FeatureSetWeights::new::<f64>(&[vec![1], vec![2, 2]], None, 20),
            Err(MethodsErrors::FeatureSetInvalid { set: 1, .. })
        ));
        assert!(matches!(
            FeatureSetWeights::new(&[vec![1, 2]], Some(&[vec![1.0]]), 20),
            Err(MethodsErrors::FeatureSetInvalid { set: 0, .. })
        ));
    }
}
