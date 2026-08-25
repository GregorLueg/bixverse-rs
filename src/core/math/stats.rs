//! Statistical helpers

use faer::{Mat, linalg::solvers::DenseSolveCore};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use statrs::distribution::FisherSnedecor;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;

use crate::core::math::distributions::{chisq_sf, f_sf, norm_sf, t_pval_two_sided};
use crate::core::math::vector_helpers::*;
use crate::prelude::*;

//////////////////
// Effect sizes //
//////////////////

/// A type alias representing effect size results
///
/// ### Fields
///
/// * `0` - The calculated effect sizes
/// * `1` - The corresponding standard errors
pub type EffectSizeRes<T> = (Vec<T>, Vec<T>);

/// Calculate the Hedge's g effect size and its standard error
///
/// ### Params
///
/// * `mean_a` - The mean values of group a.
/// * `mean_b` - The mean values of group b.
/// * `std_a` - The standard deviations of group a.
/// * `std_b` - The standard deviations of group b.
/// * `n_a` - Number of samples in a.
/// * `n_b` - Number of samples in b.
/// * `small_sample_correction` - Apply a small sample correction? Recommended
///   when `n_a` + `n_b` ≤ 35.
///
/// ### Returns
///
/// A tuple with the effect sizes being the first element, and the standard
/// errors the second element.
pub fn hedge_g_effect<T>(
    mean_a: &[T],
    mean_b: &[T],
    std_a: &[T],
    std_b: &[T],
    n_a: usize,
    n_b: usize,
    small_sample_correction: bool,
) -> EffectSizeRes<T>
where
    T: BixverseFloat,
{
    assert_same_len!(mean_a, mean_b, std_a, std_b);

    let n_a_t = T::from_usize(n_a).unwrap();
    let n_b_t = T::from_usize(n_b).unwrap();
    let total_n = T::from_usize(n_a + n_b).unwrap();
    let two = T::from_usize(2).unwrap();
    let three = T::from_usize(3).unwrap();

    let (effect_sizes, standard_errors): (Vec<T>, Vec<T>) = mean_a
        .par_iter()
        .zip(mean_b.par_iter())
        .zip(std_a.par_iter())
        .zip(std_b.par_iter())
        .map(|(((mean_a, mean_b), std_a), std_b)| {
            let pooled_sd = (((n_a_t - T::one()) * std_a.powi(2)
                + (n_b_t - T::one()) * std_b.powi(2))
                / (total_n - two))
                .sqrt();

            let mut effect_size = (*mean_a - *mean_b) / pooled_sd;

            if small_sample_correction {
                let correction_factor = ((total_n - three)
                    / (total_n - T::from_f64(2.25).unwrap()))
                    * ((total_n - two) / total_n).sqrt();
                effect_size = correction_factor * effect_size;
            }

            let standard_error =
                ((total_n / (n_a_t * n_b_t)) + (effect_size.powi(2) / (two * total_n))).sqrt();

            (effect_size, standard_error)
        })
        .unzip();

    (effect_sizes, standard_errors)
}

///////////////////////
// Statistical tests //
///////////////////////

/// Test alternatives for different statistical tests
#[derive(Clone, Debug, Default)]
pub enum TestAlternative {
    /// Two sided test for the Z-score
    #[default]
    TwoSided,
    /// One-sided test for greater than
    Greater,
    /// One-sided test for lesser than
    Less,
}

/// Helper function to get the test alternative
///
/// ### Params
///
/// * `s` - String, type of test to run.
///
/// ### Returns
///
/// Option of the `TestAlternative`
pub fn get_test_alternative(s: &str) -> Option<TestAlternative> {
    match s.to_lowercase().as_str() {
        "twosided" => Some(TestAlternative::TwoSided),
        "greater" => Some(TestAlternative::Greater),
        "less" => Some(TestAlternative::Less),
        _ => None,
    }
}

/// Transform Z-scores into p-values (assuming normality).
///
/// ### Params
///
/// * `z_scores` - The Z scores to transform to p-values
///
/// ### Returns
///
/// The p-value vector based on the Z scores (two sided)
pub fn z_scores_to_pval<T>(z_scores: &[T], test_alternative: &str) -> Vec<T>
where
    T: BixverseFloat,
{
    let test_alternative = get_test_alternative(test_alternative).unwrap_or_default();

    let normal = Normal::new(0.0, 1.0).unwrap();

    let one = T::one();
    let two = T::from_usize(2).unwrap();
    let six = T::from_usize(6).unwrap();

    z_scores
        .iter()
        .map(|&z| match test_alternative {
            TestAlternative::TwoSided => {
                let abs_z = z.abs();
                if abs_z > six {
                    let abs_z_f64 = abs_z.to_f64().unwrap();
                    let pdf = T::from_f64(normal.pdf(abs_z_f64)).unwrap();
                    let p = pdf / abs_z * (one - one / (abs_z * abs_z));
                    two * p
                } else {
                    let abs_z_f64 = abs_z.to_f64().unwrap();
                    let cdf = T::from_f64(normal.cdf(abs_z_f64)).unwrap();
                    two * (one - cdf)
                }
            }
            TestAlternative::Greater => {
                if z > six {
                    let z_f64 = z.to_f64().unwrap();
                    let pdf = T::from_f64(normal.pdf(z_f64)).unwrap();
                    pdf / z * (one - one / (z * z))
                } else {
                    let z_f64 = z.to_f64().unwrap();
                    let cdf = T::from_f64(normal.cdf(z_f64)).unwrap();
                    one - cdf
                }
            }
            TestAlternative::Less => {
                let neg_six = -six;
                if z < neg_six {
                    let abs_z = z.abs();
                    let abs_z_f64 = abs_z.to_f64().unwrap();
                    let pdf = T::from_f64(normal.pdf(abs_z_f64)).unwrap();
                    pdf / abs_z * (one - one / (abs_z * abs_z))
                } else {
                    let z_f64 = z.to_f64().unwrap();
                    T::from_f64(normal.cdf(z_f64)).unwrap()
                }
            }
        })
        .collect()
}

/// Calculate the p-value of a hypergeometric test.
///
/// ### Params
///
/// * `q` - Number of white balls drawn
/// * `m` - Number of white balls in the urn
/// * `n` - Number of black balls in the urn
/// * `k` - Number of balls drawn from the urn
///
/// ### Return
///
/// The p-value of the hypergeometric test
pub fn hypergeom_pval<T>(q: usize, m: usize, n: usize, k: usize) -> T
where
    T: BixverseFloat,
{
    let population = m + n;
    let (n_f, m_f, k_f) = (
        T::from_usize(n).unwrap(),
        T::from_usize(m).unwrap(),
        T::from_usize(k).unwrap(),
    );
    let population_f = T::from_usize(population).unwrap();

    let upper = k.min(m);
    let mut log_probs = Vec::new();

    for i in (q + 1)..=upper {
        let i_f = T::from_usize(i).unwrap();

        // ln_gamma likely only supports f64, so convert
        let log_pmf_f64 = ln_gamma(m_f.to_f64().unwrap() + 1.0)
            - ln_gamma(i_f.to_f64().unwrap() + 1.0)
            - ln_gamma((m_f - i_f).to_f64().unwrap() + 1.0)
            + ln_gamma(n_f.to_f64().unwrap() + 1.0)
            - ln_gamma((k_f - i_f).to_f64().unwrap() + 1.0)
            - ln_gamma((n_f - (k_f - i_f)).to_f64().unwrap() + 1.0)
            - (ln_gamma(population_f.to_f64().unwrap() + 1.0)
                - ln_gamma(k_f.to_f64().unwrap() + 1.0)
                - ln_gamma((population_f - k_f).to_f64().unwrap() + 1.0));

        log_probs.push(T::from_f64(log_pmf_f64).unwrap());
    }

    if log_probs.is_empty() {
        return T::zero();
    }

    let max_log_prob = log_probs
        .iter()
        .cloned()
        .fold(T::neg_infinity(), |a, b| a.max(b));

    let mut sum = T::zero();
    for log_p in log_probs {
        sum += (log_p - max_log_prob).exp();
    }

    sum * max_log_prob.exp()
}

/// Holm-Bonferroni step-down adjustment.
///
/// R's `p.adjust` default, which is worth knowing: code that calls `p.adjust`
/// without naming a method is doing Holm, not Benjamini-Hochberg. Controls the
/// family-wise error rate, so it is markedly more conservative than
/// [calc_fdr].
///
/// ### Params
///
/// * `pvals` - Unadjusted p-values
///
/// ### Returns
///
/// The adjusted p-values, in the input order, each capped at 1.
pub fn p_adjust_holm<T>(pvals: &[T]) -> Vec<T>
where
    T: BixverseFloat,
{
    let n = pvals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        pvals[a]
            .partial_cmp(&pvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let one = T::one();
    let mut out = vec![T::zero(); n];
    let mut running = T::zero();
    for (rank, &idx) in order.iter().enumerate() {
        let scaled = T::from_usize(n - rank).unwrap() * pvals[idx];
        running = running.max(scaled);
        out[idx] = running.min(one);
    }
    out
}

/// Calculate the FDR
///
/// Benjamini-Hochberg. See [p_adjust_holm] for the family-wise alternative,
/// which is what R's `p.adjust` does when no method is named.
///
/// ### Params
///
/// * `pvals` - P-values for which to calculate the FDR
///
/// ### Returns
///
/// The calculated FDRs
pub fn calc_fdr<T>(pvals: &[T]) -> Vec<T>
where
    T: BixverseFloat,
{
    let n = pvals.len();
    if n == 0 {
        return Vec::new();
    }
    let n_t = T::from_usize(n).unwrap();
    let one = T::one();

    let mut indexed_pval: Vec<(usize, T)> =
        pvals.par_iter().enumerate().map(|(i, &x)| (i, x)).collect();

    // Unstable and parallel are both safe here despite the sort deciding ranks.
    indexed_pval
        .par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let adj_pvals_tmp: Vec<T> = indexed_pval
        .par_iter()
        .enumerate()
        .map(|(i, (_, p))| {
            let i_t = T::from_usize(i + 1).unwrap();
            (n_t / i_t) * *p
        })
        .collect();

    let mut current_min = adj_pvals_tmp[n - 1].min(one);
    let mut monotonic_adj = vec![current_min; n];

    for i in (0..n - 1).rev() {
        current_min = current_min.min(adj_pvals_tmp[i]).min(one);
        monotonic_adj[i] = current_min;
    }

    let mut adj_pvals = vec![T::zero(); n];

    for (i, &(original_idx, _)) in indexed_pval.iter().enumerate() {
        adj_pvals[original_idx] = monotonic_adj[i];
    }

    adj_pvals
}

////////////
// MANOVA //
////////////

/// ManovaResults
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ManovaResult<T>
where
    T: BixverseFloat,
{
    /// Between-groups SSCP matrix
    pub sscp_between: Mat<T>,
    /// Within-groups SSCP matrix
    pub sscp_within: Mat<T>,
    /// Total SSCP matrix
    pub sscp_total: Mat<T>,
    /// Degrees of freedom between groups
    pub df_between: usize,
    /// Degrees of freedom within groups
    pub df_within: usize,
    /// Total degrees of freedom
    pub df_total: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Means for each group
    pub group_means: Vec<Vec<T>>,
    /// Overall means
    pub overall_mean: Vec<T>,
}

#[allow(dead_code)]
impl<T> ManovaResult<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Non-zero eigenvalue of E⁻¹H for two-group MANOVA.
    ///
    /// For rank-1 H, the unique non-zero generalised eigenvalue equals
    /// trace(`E^-1 x H`), avoiding `det(E) / det(E+H)` which overflows f64 for
    /// moderately large `p (entries of E scale ~ n^3 / 12` for ranked data).
    ///
    /// ### Returns
    ///
    /// The eigenvalue as `T`, clamped to zero to handle numerical noise.
    fn rank1_eigenvalue(&self) -> T {
        debug_assert_eq!(
            self.df_between, 1,
            "rank1_eigenvalue assumes two-group MANOVA (df_between == 1)"
        );
        let e_inv = self.sscp_within.partial_piv_lu().inverse();
        let prod = &e_inv * &self.sscp_between;
        let trace: T = prod.diagonal().column_vector().iter().copied().sum::<T>();
        // Numerical noise can make this slightly negative for ill-conditioned E.
        trace.max(T::zero())
    }

    /// Wilks' Lambda for two-group MANOVA.
    ///
    /// Λ = 1 / (1 + λ), where λ is the rank-1 eigenvalue of E⁻¹H.
    ///
    /// ### Returns
    ///
    /// Wilks' Λ as `T`.
    pub fn wilks_lambda(&self) -> T {
        let lambda = self.rank1_eigenvalue();
        T::one() / (T::one() + lambda)
    }

    /// Pillai's trace for two-group MANOVA.
    ///
    /// V = λ / (1 + λ), where λ is the rank-1 eigenvalue of `E^-1 x H`.
    ///
    /// ### Returns
    ///
    /// Pillai's V as `T`.
    pub fn pillai_trace(&self) -> T {
        let lambda = self.rank1_eigenvalue();
        lambda / (T::one() + lambda)
    }

    /// Exact F-test for two-group MANOVA (Hotelling's T^2 form).
    ///
    /// F = ((n - p - 1) / p) * λ,   df = (p, n - p - 1)
    ///
    /// Returns (NaN, 1.0) when the inputs are degenerate
    /// (n - p - 1 <= 0, non-finite eigenvalue, etc.).
    fn two_group_f_test(&self) -> (T, T) {
        let lambda = self.rank1_eigenvalue();
        let p = T::from_usize(self.n_vars).unwrap();
        let n = T::from_usize(self.df_within + self.df_between + 1).unwrap();
        let df2 = n - p - T::one();

        if !lambda.is_finite() || df2 <= T::zero() {
            return (T::nan(), T::one());
        }

        let f_stat = (df2 / p) * lambda;

        let df1_f64 = p.to_f64().unwrap();
        let df2_f64 = df2.to_f64().unwrap();
        let f_f64 = f_stat.to_f64().unwrap();

        if !f_f64.is_finite() || f_f64 < 0.0 {
            return (f_stat, T::one());
        }

        let f_dist = match FisherSnedecor::new(df1_f64, df2_f64) {
            Ok(d) => d,
            Err(_) => return (f_stat, T::one()),
        };
        let p_value = T::from_f64(1.0 - f_dist.cdf(f_f64)).unwrap();
        (f_stat, p_value)
    }

    /// F-statistic and p-value derived from Wilks' Lambda.
    ///
    /// For two-group MANOVA this coincides with the Pillai F-test.
    ///
    /// ### Returns
    ///
    /// A tuple of `(f_stat, p_value)` as `(T, T)`.
    pub fn wilks_f_test(&self) -> (T, T) {
        self.two_group_f_test()
    }

    /// F-statistic and p-value derived from Pillai's trace.
    ///
    /// For two-group MANOVA this coincides with the Wilks F-test.
    ///
    /// ### Returns
    ///
    /// A tuple of `(f_stat, p_value)` as `(T, T)`.
    pub fn pillai_f_test(&self) -> (T, T) {
        self.two_group_f_test()
    }
}

/// ManovaSummary
#[derive(Debug)]
pub struct ManovaSummary<T>
where
    T: BixverseFloat,
{
    /// Wilks' lambda value
    pub wilks_lambda: T,
    /// Pillai's trace value
    pub pillai_trace: T,
    /// Degrees of freedom between groups
    pub df_between: usize,
    /// Degrees of freedom within groups
    pub df_within: usize,
    /// F statistic according to Wilk
    pub f_stat_wilk: T,
    /// P-value according to Wilk
    pub p_val_wilk: T,
    /// F statistic according to Pillai
    pub f_stat_pillai: T,
    /// P-value according to Pillai
    pub p_val_pillai: T,
}

impl<T> ManovaSummary<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Get the summary results from a `ManovaRes`
    ///
    /// ### Params
    ///
    /// * `res` - The calculated ManovaResults
    ///
    /// ### Returns
    ///
    /// The `ManovaSummary`.
    pub fn from_manova_res(res: &ManovaResult<T>) -> Self {
        let (f_stat_wilk, p_val_wilk) = res.wilks_f_test();
        let (f_stat_pillai, p_val_pillai) = res.pillai_f_test();

        ManovaSummary {
            wilks_lambda: res.wilks_lambda(),
            pillai_trace: res.pillai_trace(),
            df_between: res.df_between,
            df_within: res.df_within,
            f_stat_wilk,
            p_val_wilk,
            f_stat_pillai,
            p_val_pillai,
        }
    }
}

///////////
// ANOVA //
///////////

/// AnovaSummary (based on MANOVA models)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AnovaSummary<T>
where
    T: BixverseFloat,
{
    /// Variable index
    pub variable_index: usize,
    /// Sum of squares between groups
    pub ss_between: T,
    /// Sum of squares within groups
    pub ss_within: T,
    /// Mean square between groups
    pub ms_between: T,
    /// Mean square within groups
    pub ms_within: T,
    /// F statistic
    pub f_stat: T,
    /// P-value
    pub p_val: T,
}

/// Generates from MANOVA results the AnovaSummary
///
/// ### Params
///
/// * `res` - The MANOVA result to analyse
///
/// ### Returns
///
/// A vector of AnovaSummaries
pub fn summary_aov<T>(res: &ManovaResult<T>) -> Vec<AnovaSummary<T>>
where
    T: BixverseFloat,
{
    let mut aov_res = Vec::with_capacity(res.n_vars);
    let df_between_t = T::from_usize(res.df_between).unwrap();
    let df_within_t = T::from_usize(res.df_within).unwrap();

    let f_dist = FisherSnedecor::new(
        df_between_t.to_f64().unwrap(),
        df_within_t.to_f64().unwrap(),
    )
    .ok();

    for var_idx in 0..res.n_vars {
        let ss_between = res.sscp_between[(var_idx, var_idx)];
        let ss_within = res.sscp_within[(var_idx, var_idx)];
        let ms_between = ss_between / df_between_t;
        let ms_within = ss_within / df_within_t;

        let (f_stat, p_val) = if ms_within <= T::zero() {
            (T::nan(), T::one())
        } else {
            let f = ms_between / ms_within;
            let f_f64 = f.to_f64().unwrap();
            let pv = match (&f_dist, f_f64.is_finite() && f_f64 >= 0.0) {
                (Some(d), true) => T::from_f64(1.0 - d.cdf(f_f64)).unwrap(),
                _ => T::one(),
            };
            (f, pv)
        };

        aov_res.push(AnovaSummary {
            variable_index: var_idx,
            ss_between,
            ss_within,
            ms_between,
            ms_within,
            f_stat,
            p_val,
        });
    }

    aov_res
}

///////////////////
// Probabilities //
///////////////////

/// Implementation of the trigamma function (second derivative of ln(gamma(x)))
///
/// ### Params
///
/// * `x` - The value for which to calculate the trigamma function.
///
/// ### Returns
///
/// The trigamma value for the given input.
pub fn trigamma<T: BixverseFloat>(x: T) -> T {
    let mut x = x;
    let mut result = T::zero();

    if x <= T::from_f64(5.0).unwrap() {
        while x < T::from_f64(5.0).unwrap() {
            result += T::one() / (x * x);
            x += T::one();
        }
    }

    let xx = x * x;
    result += T::one() / x
        + T::one() / (T::from_f64(2.0).unwrap() * xx)
        + T::one() / (T::from_f64(6.0).unwrap() * xx * x);

    let xxx = xx * x;
    result += -T::one() / (T::from_f64(30.0).unwrap() * xxx * x)
        + T::one() / (T::from_f64(42.0).unwrap() * xxx * xx * x)
        - T::one() / (T::from_f64(30.0).unwrap() * xxx * xxx * x);

    result
}

///////////
// Other //
///////////

/// Logit function
///
/// ### Params
///
/// * `p` - Probability value (must be in (0, 1))
///
/// ### Returns
///
/// Log-odds: ln(p / (1-p))
pub fn logit<T>(p: T) -> T
where
    T: BixverseFloat,
{
    (p / (T::one() - p)).ln()
}

/// Inverse logit (sigmoid) function
///
/// ### Params
///
/// * `q` - Log-odds value
///
/// ### Returns
///
/// Probability: exp(q) / (1 + exp(q))
pub fn inv_logit<T>(q: T) -> T
where
    T: BixverseFloat,
{
    q.exp() / (T::one() + q.exp())
}

/////////////////////
// Critical values //
/////////////////////

/// Calculate the critical value using bootstrap resampling
///
/// ### Params
///
/// * `values` - Slice of values to resample from.
/// * `sample_size` - Number of samples to draw in the bootstrap sample.
/// * `alpha` - The significance level for the critical value.
/// * `seed` - Random seed for reproducibility.
///
/// ### Returns
///
/// The critical value at the specified alpha level.
pub fn calculate_critval<T: BixverseFloat>(
    values: &[T],
    sample_size: usize,
    alpha: &T,
    seed: usize,
) -> T {
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut random_sample: Vec<T> = (0..sample_size)
        .map(|_| {
            let index = rng.random_range(0..values.len());
            values[index]
        })
        .collect();
    random_sample.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let index = (*alpha * T::from_usize(random_sample.len()).unwrap())
        .ceil()
        .to_usize()
        .unwrap();
    random_sample[index + 1]
}

//////////////
// Outliers //
//////////////

/// Type of outlier detection for MAD thresholding
#[derive(Clone, Debug, Default)]
pub enum OutlierDirection {
    /// Check if outlier is below OR above the thresholds
    #[default]
    Both,
    /// Check if outlier is below the threshold
    Below,
    /// Check if outlier is above the threshold
    Above,
}

/// Helper function to get the outlier detection method
///
/// ### Params
///
/// * `s` - String, type of test to run.
///
/// ### Returns
///
/// Option of the `OutlierDirection`
pub fn parse_outlier_type(s: &str) -> Option<OutlierDirection> {
    match s.to_lowercase().as_str() {
        "below" => Some(OutlierDirection::Below),
        "above" => Some(OutlierDirection::Above),
        "twosided" => Some(OutlierDirection::Both),
        _ => None,
    }
}

/// MAD outlier detection
///
/// ### Params
///
/// * `x` - Slice of values to check for outliers.
/// * `threshold` - Number of MADs to accept as not being an outlier.
/// * `direction` - Direction to check for outliers (below, above, or both).
///
/// ### Returns
///
/// A tuple with a vector of booleans indicating whether each value in the input
/// is an outlier and the value of applied margin.
pub fn mad_outlier<T>(x: &[T], threshold: T, direction: OutlierDirection) -> (Vec<bool>, T)
where
    T: BixverseFloat,
{
    let median_val = match median(x) {
        Some(m) => m,
        None => return (vec![], T::zero()),
    };
    let mad_val = match mad(x, None) {
        Some(m) => m,
        None => return (vec![], T::zero()),
    };
    let margin = threshold * mad_val;
    let res = x
        .iter()
        .map(|&v| match direction {
            OutlierDirection::Below => v < median_val - margin,
            OutlierDirection::Above => v > median_val + margin,
            OutlierDirection::Both => (v - median_val).abs() > margin,
        })
        .collect::<Vec<bool>>();
    (res, margin)
}

///////////////////
// One-way ANOVA //
///////////////////

/// One-way analysis of variance across an arbitrary number of groups.
///
/// The [ManovaResult] path in this module is two-group only, so this exists for
/// the many-group case: does a feature vary across samples at all. Empty groups
/// are dropped rather than counted, so a caller may pass a level set wider than
/// the data.
///
/// ### Params
///
/// * `values` - Observations
/// * `groups` - Level code per observation, parallel to `values`, each in
///   `0..n_groups`
/// * `n_groups` - Number of levels
///
/// ### Returns
///
/// `(f_statistic, p_value)`
pub fn one_way_anova<T: BixverseFloat>(
    values: &[T],
    groups: &[usize],
    n_groups: usize,
) -> Result<(T, T), BixverseErrors> {
    if values.len() != groups.len() {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (values.len(), 1),
            got: (groups.len(), 1),
        });
    }
    let n = values.len();
    let mut counts = vec![0_usize; n_groups];
    let mut sums = vec![0.0_f64; n_groups];
    for (v, &g) in values.iter().zip(groups.iter()) {
        if g >= n_groups {
            return Err(BixverseErrors::InvalidArgument(format!(
                "group code {g} is outside 0..{n_groups}."
            )));
        }
        counts[g] += 1;
        sums[g] += v.to_f64().unwrap_or(f64::NAN);
    }

    let n_used: usize = counts.iter().filter(|&&c| c > 0).count();
    if n_used < 2 {
        return Err(BixverseErrors::InvalidArgument(
            "one-way ANOVA needs at least two non-empty groups.".to_string(),
        ));
    }
    let df_between = (n_used - 1) as f64;
    let df_within = (n - n_used) as f64;
    if df_within <= 0.0 {
        return Err(BixverseErrors::InvalidArgument(
            "one-way ANOVA has no residual degrees of freedom.".to_string(),
        ));
    }

    let grand_mean: f64 = sums.iter().sum::<f64>() / n as f64;
    let ss_between: f64 = counts
        .iter()
        .zip(sums.iter())
        .filter(|(c, _)| **c > 0)
        .map(|(&c, &s)| {
            let d = s / c as f64 - grand_mean;
            c as f64 * d * d
        })
        .sum();
    let ss_within: f64 = values
        .iter()
        .zip(groups.iter())
        .map(|(v, &g)| {
            let d = v.to_f64().unwrap_or(f64::NAN) - sums[g] / counts[g] as f64;
            d * d
        })
        .sum();

    // Finiteness first: a non-finite sum fails both `> 0.0` tests below and
    // would fall through to the F = 0 arm.
    if !ss_within.is_finite() || !ss_between.is_finite() {
        return Err(BixverseErrors::InvalidArgument(
            "the values contain a non-finite entry.".to_string(),
        ));
    }
    // A feature that is constant within every group but varies between them has
    // an infinite F. R prints Inf and a p of 0, so do the same rather than
    // dividing by zero and returning NaN.
    let f_stat = if ss_within > 0.0 {
        (ss_between / df_between) / (ss_within / df_within)
    } else if ss_between > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    let p = f_sf(f_stat, df_between, df_within)?;

    Ok((
        T::from_f64(f_stat).unwrap_or_else(T::nan),
        T::from_f64(p).unwrap_or_else(T::nan),
    ))
}

/////////////////////////
// Partial correlation //
/////////////////////////

/// First-order partial correlation of `x` and `y` given a single control `z`.
///
/// `r_xy.z = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))`, which for
/// one control variable is what inverting the 3x3 correlation matrix reduces
/// to. The Spearman variant ranks all three vectors first, with average ranks
/// for ties, then runs the same formula.
///
/// The test is `t = r sqrt((n - 3) / (1 - r^2))` on `n - 3` degrees of freedom,
/// two-sided: one degree of freedom is spent on the control on top of the two a
/// plain correlation costs.
///
/// ### Params
///
/// * `x` - First variable
/// * `y` - Second variable
/// * `z` - Control variable
/// * `spearman` - Rank-transform first, which is what DIALOGUE asks for
///
/// ### Returns
///
/// `(estimate, p_value)`
pub fn partial_correlation<T: BixverseFloat + Sync>(
    x: &[T],
    y: &[T],
    z: &[T],
    spearman: bool,
) -> Result<(T, T), BixverseErrors> {
    let n = x.len();
    if y.len() != n {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (n, 3),
            got: (y.len(), 3),
        });
    }
    if z.len() != n {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (n, 3),
            got: (z.len(), 3),
        });
    }
    if n < 4 {
        return Err(BixverseErrors::InvalidArgument(format!(
            "a first-order partial correlation needs at least 4 observations; got {n}."
        )));
    }

    let (xv, yv, zv) = if spearman {
        (rank_vector(x), rank_vector(y), rank_vector(z))
    } else {
        (x.to_vec(), y.to_vec(), z.to_vec())
    };

    let r_xy = pearson_correlation(&xv, &yv).unwrap_or(f64::NAN);
    let r_xz = pearson_correlation(&xv, &zv).unwrap_or(f64::NAN);
    let r_yz = pearson_correlation(&yv, &zv).unwrap_or(f64::NAN);

    let denom = ((1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)).sqrt();
    if !(denom.is_finite() && denom > 0.0) {
        return Ok((T::nan(), T::nan()));
    }
    let r = (r_xy - r_xz * r_yz) / denom;

    let df = (n - 3) as f64;
    // |r| == 1 leaves nothing to test; R reports a p of 0 there.
    let p = if r.abs() >= 1.0 {
        0.0
    } else {
        let t_stat = r * (df / (1.0 - r * r)).sqrt();
        t_pval_two_sided(t_stat, df)?
    };

    Ok((
        T::from_f64(r).unwrap_or_else(T::nan),
        T::from_f64(p).unwrap_or_else(T::nan),
    ))
}

//////////////////////
// Fisher combining //
//////////////////////

/// Combines independent p-values by Fisher's method.
///
/// `-2 sum(log p)` is chi-squared on `2m` degrees of freedom under the joint
/// null. `NaN` entries are skipped, and a single surviving p-value is returned
/// unchanged rather than passed through the chi-squared, which is what
/// DIALOGUE's `get.fisher.p.value` does.
///
/// ### Params
///
/// * `pvals` - p-values in `[0, 1]`. `NaN` entries are ignored.
///
/// ### Returns
///
/// The combined p-value, or `NaN` when nothing survives the `NaN` filter.
/// Upstream returns 0.0 in that case, through `pchisq(0, 0)`; that is an
/// artefact of R's zero-df convention and is not reproduced here, because "no
/// evidence at all" is the one thing a p-value of zero must not mean.
pub fn fisher_combine<T: BixverseFloat>(pvals: &[T]) -> Result<T, BixverseErrors> {
    let kept: Vec<f64> = pvals
        .iter()
        .filter_map(|p| p.to_f64())
        .filter(|p| !p.is_nan())
        .collect();
    match kept.len() {
        0 => Ok(T::nan()),
        1 => Ok(T::from_f64(kept[0]).unwrap_or_else(T::nan)),
        m => {
            let stat: f64 = -2.0 * kept.iter().map(|p| p.ln()).sum::<f64>();
            let p = chisq_sf(stat, 2.0 * m as f64)?;
            Ok(T::from_f64(p).unwrap_or_else(T::nan))
        }
    }
}

///////////////////////
// Wilcoxon rank sum //
///////////////////////

/// One-sided Wilcoxon rank-sum p-value, normal approximation.
///
/// Matches `wilcox.test(x, y, alternative = "greater", exact = FALSE,
/// correct = TRUE)`: continuity correction of a half, and the variance carries
/// the tie correction `sum(t^3 - t) / (N (N - 1))`. Deliberately only the
/// approximation.
///
/// ### Params
///
/// * `x` - First sample, the one tested for being larger
/// * `y` - Second sample
///
/// ### Returns
///
/// `P(X > Y)` under the null, or [BixverseErrors::InvalidArgument] when either
/// sample is empty or the tie correction leaves no variance.
pub fn wilcox_rank_sum_greater_approx<T: BixverseFloat>(
    x: &[T],
    y: &[T],
) -> Result<T, BixverseErrors> {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "the Wilcoxon rank sum needs a non-empty sample on both sides.".to_string(),
        ));
    }
    let total = n + m;

    let mut pooled: Vec<T> = Vec::with_capacity(total);
    pooled.extend_from_slice(x);
    pooled.extend_from_slice(y);
    let ranks = rank_vector(&pooled);

    let rank_sum: f64 = ranks[..n].iter().filter_map(|r| r.to_f64()).sum();
    let w = rank_sum - (n * (n + 1)) as f64 / 2.0;

    // Tie correction, from the multiplicities of the pooled ranks.
    let mut sorted: Vec<f64> = ranks.iter().filter_map(|r| r.to_f64()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut tie_term = 0.0_f64;
    let mut i = 0;
    while i < total {
        let mut j = i + 1;
        while j < total && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        tie_term += t * t * t - t;
        i = j;
    }

    let nm = (n * m) as f64;
    let total_f = total as f64;
    let variance = (nm / 12.0) * ((total_f + 1.0) - tie_term / (total_f * (total_f - 1.0)));
    if !(variance.is_finite() && variance > 0.0) {
        return Err(BixverseErrors::InvalidArgument(
            "the Wilcoxon rank sum has zero variance; every observation is tied.".to_string(),
        ));
    }

    let z = (w - nm / 2.0 - 0.5) / variance.sqrt();
    Ok(T::from_f64(norm_sf(z)).unwrap_or_else(T::nan))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Three unbalanced groups with a real effect, against R's `aov`.
    #[test]
    fn test_one_way_anova_matches_r_aov() {
        // R: set.seed(3); grp sizes 4/6/5
        let values: Vec<f64> = vec![
            -0.962, -0.293, 0.259, -1.152, 1.696, 1.53, 1.585, 2.617, 0.281, 2.767, -1.245, -1.631,
            -1.216, -0.247, -0.348,
        ];
        let groups: Vec<usize> = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let (f_stat, p) = one_way_anova(&values, &groups, 3).unwrap();

        assert_relative_eq!(f_stat, 20.398607696686263, max_relative = 1e-12);
        assert_relative_eq!(p, 0.00013785463402486508, max_relative = 1e-11);
    }

    /// Two groups with no signal, so the p-value sits well away from zero.
    #[test]
    fn test_one_way_anova_null_case() {
        // R: set.seed(9); rnorm(16), 8 per group
        let values: Vec<f64> = vec![
            -0.767, -0.816, -0.142, -0.278, 0.436, -1.187, 1.192, -0.018, -0.248, -0.363, 1.278,
            -0.469, 0.071, -0.266, 1.845, -0.839,
        ];
        let groups: Vec<usize> = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];

        let (f_stat, p) = one_way_anova(&values, &groups, 2).unwrap();

        assert_relative_eq!(f_stat, 0.577_110_410_366_241_4, max_relative = 1e-12);
        assert_relative_eq!(p, 0.4600488747180288, max_relative = 1e-12);
    }

    /// An empty level is dropped rather than counted, so the answer is
    /// unchanged by widening the level set.
    #[test]
    fn test_one_way_anova_ignores_empty_groups() {
        let values: Vec<f64> = vec![1.0, 2.0, 5.0, 6.0];
        let groups: Vec<usize> = vec![0, 0, 2, 2];
        let widened = one_way_anova(&values, &groups, 5).unwrap();
        let tight = one_way_anova(&values, &[0, 0, 1, 1], 2).unwrap();
        assert_relative_eq!(widened.0, tight.0, max_relative = 1e-14);
        assert_relative_eq!(widened.1, tight.1, max_relative = 1e-14);
    }

    /// A non-finite value must error, not be reported as "does not vary".
    ///
    /// `NaN > 0.0` is false, so without an explicit finiteness check control
    /// reaches the constant-input arm and the function returns `F = 0, p = 1`.
    /// The DIALOGUE feature filter *keeps* what varies, so that reading
    /// silently discards the feature instead of flagging it.
    #[test]
    fn test_one_way_anova_rejects_non_finite_values() {
        let groups = vec![0usize, 0, 0, 1, 1, 1];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let values: Vec<f64> = vec![1.0, 2.0, bad, 5.0, 6.0, 7.0];
            assert!(
                one_way_anova::<f64>(&values, &groups, 2).is_err(),
                "accepted {bad} and reported a verdict"
            );
        }
    }

    /// Constant within groups but different between them is an infinite F, and
    /// a p of zero. Constant throughout is F of zero and a p of one.
    #[test]
    fn test_one_way_anova_degenerate_variance_branches() {
        let groups = vec![0usize, 0, 1, 1];

        let separated = vec![1.0, 1.0, 5.0, 5.0];
        let (f_stat, p): (f64, f64) = one_way_anova(&separated, &groups, 2).unwrap();
        assert!(f_stat.is_infinite() && f_stat > 0.0);
        assert_eq!(p, 0.0);

        let constant = vec![3.0, 3.0, 3.0, 3.0];
        let (f_stat, p): (f64, f64) = one_way_anova(&constant, &groups, 2).unwrap();
        assert_eq!(f_stat, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_one_way_anova_rejects_degenerate_input() {
        let values: Vec<f64> = vec![1.0, 2.0, 3.0];
        // Every observation in one group.
        assert!(one_way_anova(&values, &[0, 0, 0], 1).is_err());
        // Length mismatch.
        assert!(one_way_anova(&values, &[0, 1], 2).is_err());
        // One observation per group leaves no residual dof.
        assert!(one_way_anova(&values, &[0, 1, 2], 3).is_err());
    }

    // -- p_adjust_holm --

    /// Against R's bare `p.adjust`, whose default method is Holm.
    #[test]
    fn test_p_adjust_holm_matches_r() {
        // R: p.adjust(c(0.001, 0.02, 0.03, 0.04, 0.5))
        let got: Vec<f64> = p_adjust_holm(&[0.001, 0.02, 0.03, 0.04, 0.5]);
        let expected = [0.005, 0.08, 0.09, 0.09, 0.5];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(g, e, max_relative = 1e-12);
        }
    }

    /// The step-down carry is what makes the sequence non-decreasing: the third
    /// value would be 0.09 on its own but is held up by the second.
    #[test]
    fn test_p_adjust_holm_is_monotone_and_handles_ties() {
        // R: p.adjust(c(0.01, 0.01, 0.04))
        let got: Vec<f64> = p_adjust_holm(&[0.01, 0.01, 0.04]);
        for (g, e) in got.iter().zip([0.03, 0.03, 0.04].iter()) {
            assert_relative_eq!(g, e, max_relative = 1e-12);
        }
        assert!(p_adjust_holm::<f64>(&[]).is_empty());
        // Capped at one, and more conservative than Benjamini-Hochberg.
        let p = [0.2, 0.3, 0.4];
        let holm: Vec<f64> = p_adjust_holm(&p);
        let bh: Vec<f64> = calc_fdr(&p);
        assert!(holm.iter().all(|v| *v <= 1.0));
        assert!(holm.iter().zip(bh.iter()).all(|(h, b)| h >= b));
    }

    /// Fixture shared by the partial correlation tests.
    #[allow(clippy::type_complexity)]
    fn pcor_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // R: set.seed(21); z <- rnorm(30); x <- 0.8 z + noise; y <- 0.6 z + noise
        let x = vec![
            -0.466, 0.173, 1.413, -0.47, 2.738, 0.383, -0.147, -0.7, 0.902, 0.874, -1.788, -0.305,
            -0.468, 0.01, 1.708, 1.331, 0.838, 1.904, -0.397, -1.11, -0.035, 1.094, -0.952, -1.223,
            0.17, 1.205, 0.253, -1.441, -0.02, 0.115,
        ];
        let y = vec![
            0.201, 0.87, -0.214, -1.769, 2.378, 0.736, -1.232, 0.596, -0.193, -0.953, -1.238,
            -0.229, -0.035, -0.948, 0.638, 0.959, 0.51, 1.172, -0.878, -0.078, 0.903, 0.24, -0.782,
            -0.877, -0.096, -0.073, -0.017, -1.358, -0.913, 0.349,
        ];
        let z = vec![
            0.793, 0.522, 1.746, -1.271, 2.197, 0.433, -1.57, -0.935, 0.063, -0.002, -2.277, 0.757,
            -0.548, 0.173, 0.563, 1.512, 0.659, 1.122, -0.785, -0.426, 0.393, 0.037, -1.032,
            -1.265, -0.227, 0.746, 0.333, -1.124, -0.706, -0.728,
        ];
        (x, y, z)
    }

    /// Pearson variant, against `ppcor::pcor.test(x, y, z, method = "pearson")`.
    #[test]
    fn test_partial_correlation_pearson_matches_ppcor() {
        let (x, y, z) = pcor_fixture();
        let (est, p) = partial_correlation(&x, &y, &z, false).unwrap();
        assert_relative_eq!(est, 0.13928951527176736, max_relative = 1e-12);
        assert_relative_eq!(p, 0.47113917629636237, max_relative = 1e-11);
    }

    /// Spearman variant, which is what DIALOGUE uses. Ranks first, then the
    /// same formula.
    #[test]
    fn test_partial_correlation_spearman_matches_ppcor() {
        let (x, y, z) = pcor_fixture();
        let (est, p) = partial_correlation(&x, &y, &z, true).unwrap();
        assert_relative_eq!(est, 0.11596978538690578, max_relative = 1e-12);
        assert_relative_eq!(p, 0.549_124_448_784_373_9, max_relative = 1e-11);
    }

    /// Controlling for a variable that carries the whole association drives the
    /// partial correlation to zero, where the raw correlation is near one.
    #[test]
    fn test_partial_correlation_removes_a_shared_driver() {
        let z: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        // x and y are z plus independent, deterministic wobbles.
        let x: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, v)| v + ((i % 7) as f64) * 0.01)
            .collect();
        let y: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, v)| v + ((i % 5) as f64) * 0.01)
            .collect();

        let raw = pearson_correlation(&x, &y).unwrap();
        let (partial, _) = partial_correlation(&x, &y, &z, false).unwrap();

        assert!(raw > 0.99, "raw correlation was {raw}");
        assert!(partial.abs() < 0.5, "partial correlation was {partial}");
    }

    #[test]
    fn test_partial_correlation_rejects_short_and_ragged_input() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(partial_correlation(&a, &a, &a, false).is_err());
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(partial_correlation(&b, &a, &b, false).is_err());
    }

    // -- fisher_combine --

    /// Four p-values, against `pchisq(-2 sum(log p), 2m, lower.tail = FALSE)`.
    #[test]
    fn test_fisher_combine_matches_r() {
        let p = vec![0.01, 0.2, 0.5, 0.03];
        let got: f64 = fisher_combine(&p).unwrap();
        assert_relative_eq!(got, 0.007_616_871_850_449_079, max_relative = 1e-12);
    }

    /// NaN entries drop out of both the sum and the degrees of freedom.
    #[test]
    fn test_fisher_combine_skips_nan() {
        let p = vec![0.4, f64::NAN, 0.9];
        let got: f64 = fisher_combine(&p).unwrap();
        assert_relative_eq!(got, 0.727_794_449_111_513_4, max_relative = 1e-12);
    }

    /// A single surviving p-value is returned unchanged, not passed through the
    /// chi-squared.
    #[test]
    fn test_fisher_combine_single_value_passes_through() {
        let got: f64 = fisher_combine(&[0.037]).unwrap();
        assert_eq!(got, 0.037);
        let got2: f64 = fisher_combine(&[f64::NAN, 0.037, f64::NAN]).unwrap();
        assert_eq!(got2, 0.037);
    }

    /// Nothing to combine gives NaN, deliberately not the 0.0 that R's
    /// zero-df `pchisq` would produce.
    #[test]
    fn test_fisher_combine_empty_is_nan() {
        let got: f64 = fisher_combine(&[]).unwrap();
        assert!(got.is_nan());
        let got2: f64 = fisher_combine(&[f64::NAN, f64::NAN]).unwrap();
        assert!(got2.is_nan());
    }

    // -- wilcox_rank_sum_greater_approx --

    /// The 99 permutation nulls DIALOGUE's empirical p-value is built on.
    fn wilcox_nulls() -> Vec<f64> {
        vec![
            -0.8409, 1.3844, -1.2555, 0.0701, 1.7114, -0.6029, -0.4722, -0.6354, -0.2858, 0.1381,
            1.2276, -0.8018, -1.0804, -0.1575, -1.0718, -0.139, -0.5973, -2.184, 0.2408, -0.2594,
            0.9005, 0.9419, 1.468, 0.7068, 0.819, -0.2935, 1.4186, 1.4988, -0.6571, -0.8528,
            0.3159, 1.1097, 2.2155, 1.2171, 1.4792, 0.9516, -1.0095, -2.0005, -1.7622, -0.1426,
            1.5501, -0.8024, -0.0746, 1.8957, -0.4566, 0.5622, -0.887, -0.4602, -0.7243, -0.0692,
            1.4632, 0.1877, 1.022, -0.5918, -0.1122, -0.925, 0.7533, -0.1126, -0.0641, 0.2333,
            -1.1366, 0.8548, -0.5784, 0.4964, -0.7601, -0.3414, -2.1023, -0.3017, -1.2724, -0.2797,
            -0.2041, -0.2256, 0.347, 0.0324, 0.4135, -0.1553, 0.9735, 0.1211, 0.1892, -0.5629,
            0.4984, -1.7423, 0.9755, -0.0241, 0.6757, -0.7103, 2.3872, -0.4734, -0.0758, -0.5218,
            0.926, -1.0624, 0.557, 0.9007, 0.9899, 0.3836, -0.3466, -0.5402, -0.1826,
        ]
    }

    /// The floor of DIALOGUE's empirical p-value is 0.045, not 0.01.
    ///
    /// One real value beating all 99 nulls. R takes the normal approximation
    /// here because the null group has 99 members, so the naive
    /// `(#{null >= real} + 1) / 100` is the wrong answer by more than a factor
    /// of four, and the 0.1 threshold that assigns cell types to programmes
    /// sits inside the gap.
    #[test]
    fn test_wilcox_rank_sum_greater_beats_every_null() {
        let nulls = wilcox_nulls();
        let real = nulls.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.1;
        let p: f64 = wilcox_rank_sum_greater_approx(&[real], &nulls).unwrap();
        // R: wilcox.test(real, nulls, alternative = "greater")$p.value
        assert_relative_eq!(p, 0.044_801_589_130_970_6, max_relative = 1e-11);
    }

    /// Sitting at the 90th null, with a tie against it, still clears 0.1.
    #[test]
    fn test_wilcox_rank_sum_greater_at_the_ninetieth_null() {
        let nulls = wilcox_nulls();
        let mut sorted = nulls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let real = sorted[89];
        let p: f64 = wilcox_rank_sum_greater_approx(&[real], &nulls).unwrap();
        assert_relative_eq!(p, 0.085_594_599_535_642_33, max_relative = 1e-11);
        assert!(p < 0.1);
    }

    /// At the median the test is uninformative, as it should be.
    #[test]
    fn test_wilcox_rank_sum_greater_at_the_median() {
        let nulls = wilcox_nulls();
        let mut sorted = nulls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let real = sorted[49];
        let p: f64 = wilcox_rank_sum_greater_approx(&[real], &nulls).unwrap();
        assert_relative_eq!(p, 0.506_909_903_708_803, max_relative = 1e-11);
    }

    /// The general branch, pinned against `wilcox.test(..., exact = FALSE)`.
    #[test]
    fn test_wilcox_rank_sum_greater_small_samples() {
        let a = [3.1, 4.5, 2.2, 6.0];
        let b = [1.0, 2.5, 3.9, 0.4, 5.5];
        let p: f64 = wilcox_rank_sum_greater_approx(&a, &b).unwrap();
        assert_relative_eq!(p, 0.195_633_639_641_319_7, max_relative = 1e-11);
    }

    /// Ties change the variance, and R takes this branch regardless of size.
    #[test]
    fn test_wilcox_rank_sum_greater_with_ties() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 3.0, 4.0, 5.0];
        let p: f64 = wilcox_rank_sum_greater_approx(&a, &b).unwrap();
        assert_relative_eq!(p, 0.947_403_747_439_979_2, max_relative = 1e-11);
    }

    #[test]
    fn test_wilcox_rank_sum_rejects_degenerate_input() {
        let a = [1.0, 2.0];
        let empty: [f64; 0] = [];
        assert!(wilcox_rank_sum_greater_approx(&a, &empty).is_err());
        assert!(wilcox_rank_sum_greater_approx(&empty, &a).is_err());
        // Everything tied leaves no variance to test against.
        assert!(wilcox_rank_sum_greater_approx(&[1.0, 1.0], &[1.0, 1.0]).is_err());
    }

    /// `logit` and `inv_logit` are inverses, with p = 0.5 sitting at zero log-odds.
    #[test]
    fn test_logit_and_inv_logit() {
        let p: f64 = 0.5;
        let log_odds: f64 = logit(p);
        assert!((log_odds - 0.0).abs() < 1e-6); // ln(0.5 / 0.5) = ln(1) = 0.0

        let recovered_p = inv_logit(log_odds);
        assert!((recovered_p - 0.5).abs() < 1e-6);
    }

    /// Two-sided p-values at the two z-scores whose answers are known by heart.
    #[test]
    fn test_z_scores_to_pval() {
        let z_scores: Vec<f64> = vec![0.0, 1.95996398454]; // ~ 95% confidence interval

        let pvals_two_sided = z_scores_to_pval(&z_scores, "twosided");
        // Z = 0 -> p = 1.0
        assert!((pvals_two_sided[0] - 1.0).abs() < 1e-6);
        // Z = 1.96 -> p ≈ 0.05
        assert!((pvals_two_sided[1] - 0.05).abs() < 1e-5);
    }

    /// Heavy ties must not let the sort order leak into the result.
    ///
    /// `calc_fdr` sorts unstably and in parallel, so tied p-values land in an
    /// arbitrary order. The cumulative-min pass is what makes that irrelevant,
    /// and this pins it: every tied input must come back with the same adjusted
    /// value, and repeated runs must agree.
    #[test]
    fn test_calc_fdr_is_invariant_to_tie_order() {
        // one large tie group, one small, and a few distinct values
        let mut pvals: Vec<f64> = vec![0.02; 500];
        pvals.extend(vec![0.5; 50]);
        pvals.extend([0.001, 0.3, 0.7, 0.9, 0.02, 0.5]);

        let first = calc_fdr(&pvals);
        let second = calc_fdr(&pvals);
        assert_eq!(first, second, "calc_fdr is not deterministic");

        // every entry sharing a p-value shares its adjusted value
        for (i, p) in pvals.iter().enumerate() {
            for (j, q) in pvals.iter().enumerate() {
                if p == q {
                    assert_eq!(
                        first[i], first[j],
                        "tied p-values {p} got different FDRs at {i} and {j}"
                    );
                }
            }
        }

        // and a reversed input gives the same multiset of answers
        let mut reversed = pvals.clone();
        reversed.reverse();
        let rev_fdr = calc_fdr(&reversed);
        let mut a = first.clone();
        let mut b = rev_fdr;
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-15, "{x} vs {y}");
        }
    }

    /// The upper-tail p-value must be non-increasing in Z, including across the
    /// switch at `z = 6` from the CDF to the asymptotic tail.
    ///
    /// Hotspot's module threshold search skips whole histogram bins on the
    /// strength of this, so a future tweak to the tail approximation that broke
    /// monotonicity would silently give the wrong threshold rather than fail
    /// anywhere near here.
    #[test]
    fn test_upper_tail_pval_monotone_in_z() {
        let mut zs: Vec<f64> = vec![
            -8.0, -6.0, -1.0, 0.0, 1.0, 3.0, 5.0, 5.9, 5.9999, 6.0, 6.0001, 6.1, 7.0, 10.0, 20.0,
            37.0, 39.0,
        ];
        // and a dense sweep either side of the branch
        for i in 0..200 {
            zs.push(5.5 + i as f64 * 0.005);
        }
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let pvals = z_scores_to_pval(&zs, "greater");
        for w in pvals.windows(2) {
            assert!(
                w[1] <= w[0],
                "upper-tail p-value increased: {} then {}",
                w[0],
                w[1]
            );
        }
    }

    /// Benjamini-Hochberg adjustment stays monotonic and comes back in the input order.
    #[test]
    fn test_calc_fdr() {
        // Classic Benjamini-Hochberg test case
        let pvals: Vec<f64> = vec![0.01, 0.04, 0.03];
        // Sorted: 0.01 (idx 0), 0.03 (idx 2), 0.04 (idx 1)
        // Adjusted tmp: 0.01*(3/1)=0.03, 0.03*(3/2)=0.045, 0.04*(3/3)=0.04
        // Monotonic min backwards: 0.04, min(0.04, 0.045)=0.04, min(0.04, 0.03)=0.03
        // Result: [0.03, 0.04, 0.04]
        let fdr = calc_fdr(&pvals);

        assert!((fdr[0] - 0.03).abs() < 1e-6);
        assert!((fdr[1] - 0.04).abs() < 1e-6);
        assert!((fdr[2] - 0.04).abs() < 1e-6);
    }

    /// Only points beyond the median plus `threshold` MADs count as outliers.
    #[test]
    fn test_mad_outlier() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        // Median is 3.0
        // Deviations: [2.0, 1.0, 0.0, 1.0, 97.0]
        // MAD (median of deviations) is 1.0
        // Threshold = 3.0, so margin = 3.0
        // Acceptable range: 3.0 ± 3.0 = [0.0, 6.0]
        let (outliers, margin) = mad_outlier(&vec, 3.0, OutlierDirection::Both);

        assert!((margin - 3.0).abs() < 1e-6);
        assert_eq!(outliers, vec![false, false, false, false, true]); // 100.0 is an outlier
    }

    /// The alternative and outlier-direction parsers ignore case.
    #[test]
    fn test_parse_helpers() {
        assert!(matches!(
            get_test_alternative("twosided"),
            Some(TestAlternative::TwoSided)
        ));
        assert!(matches!(
            get_test_alternative("GREATER"),
            Some(TestAlternative::Greater)
        ));

        assert!(matches!(
            parse_outlier_type("below"),
            Some(OutlierDirection::Below)
        ));
        assert!(matches!(
            parse_outlier_type("Twosided"),
            Some(OutlierDirection::Both)
        ));
    }

    /// Two-group MANOVA against hand-computed Wilks, Pillai and F statistics.
    #[test]
    fn test_manova_two_group_analytic() {
        // Group 0: rows 0, 1; Group 1: rows 2, 3
        // x = [[1, 4], [2, 1], [5, 7], [6, 3]]
        //
        // Hand-computed:
        //   sscp_within = [[1.0, -3.5], [-3.5, 12.5]]
        //   d = mean_1 - mean_0 = [4.0, 2.5]
        //   c = n1*n2/n = 1
        //   lambda = c * d' * E^-1 * d = 1105
        //   Wilks = 1/1106, Pillai = 1105/1106, Wilks + Pillai = 1
        //   F = ((n - p - 1)/p) * lambda = (1/2)*1105 = 552.5, df = (2, 1)
        let mat = Mat::from_fn(4, 2, |i, j| match (i, j) {
            (0, 0) => 1.0f64,
            (0, 1) => 4.0,
            (1, 0) => 2.0,
            (1, 1) => 1.0,
            (2, 0) => 5.0,
            (2, 1) => 7.0,
            (3, 0) => 6.0,
            (3, 1) => 3.0,
            _ => unreachable!(),
        });
        let res = crate::enrichment::mitch::manova_mitch(mat.as_ref(), &[2, 3]);

        let wilks = res.wilks_lambda();
        let pillai = res.pillai_trace();
        let recovered_lambda = (1.0 - wilks) / wilks;

        assert!((recovered_lambda - 1105.0).abs() < 1e-6);
        assert!((wilks + pillai - 1.0).abs() < 1e-12);
        assert!((wilks - 1.0 / 1106.0).abs() < 1e-9);
        assert!((pillai - 1105.0 / 1106.0).abs() < 1e-9);

        let (f_w, _) = res.wilks_f_test();
        let (f_p, p_p) = res.pillai_f_test();
        assert!((f_w - 552.5).abs() < 1e-6);
        assert!((f_p - 552.5).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&p_p));
    }

    /// Regression: at Mitch-scale magnitudes `det()` overflowed to NaN and panicked downstream.
    #[test]
    fn test_manova_no_overflow_large_p() {
        // Regression test: previously panicked via det() overflow → NaN →
        // FisherSnedecor::cdf XOutOfRange. Mimics Mitch ranked-data magnitudes
        // (n ~ 3000, sscp diagonals ~ n^3/12 ~ 2e9).
        let p = 80;
        let n_total = 2949usize;
        let sigma = (n_total as f64).powi(3) / 12.0;

        let sscp_within = Mat::from_fn(p, p, |i, j| if i == j { sigma } else { sigma * 0.3 });
        // Small between-group signal; H entries ~ sigma * 1e-3
        let sscp_between = Mat::from_fn(
            p,
            p,
            |i, j| {
                if i == j { sigma * 1.5e-3 } else { sigma * 5e-4 }
            },
        );
        let sscp_total = &sscp_within + &sscp_between;

        let res = ManovaResult::<f64> {
            sscp_between,
            sscp_within,
            sscp_total,
            df_between: 1,
            df_within: n_total - 2,
            df_total: n_total - 1,
            n_vars: p,
            group_means: vec![vec![0.0; p], vec![0.0; p]],
            overall_mean: vec![0.0; p],
        };

        let wilks = res.wilks_lambda();
        let pillai = res.pillai_trace();
        let (f_w, p_w) = res.wilks_f_test();
        let (f_p, p_p) = res.pillai_f_test();

        assert!(wilks.is_finite() && wilks > 0.0 && wilks <= 1.0);
        assert!((0.0..1.0).contains(&pillai) && pillai.is_finite());
        assert!((wilks + pillai - 1.0).abs() < 1e-9);
        assert!(f_w.is_finite() && f_w >= 0.0);
        assert!(f_p.is_finite() && f_p >= 0.0);
        assert!((0.0..=1.0).contains(&p_w));
        assert!((0.0..=1.0).contains(&p_p));
        // Wilks and Pillai F-tests are identical for two groups.
        assert!((f_w - f_p).abs() < 1e-9);
        assert!((p_w - p_p).abs() < 1e-9);
    }

    /// Regression: a constant column has no within-group variance and used to panic.
    #[test]
    fn test_summary_aov_constant_column() {
        // ss_within == 0 used to feed NaN into FisherSnedecor and panic.
        let p = 2;
        let sscp_within = Mat::from_fn(p, p, |i, j| {
            if i == 0 && j == 0 {
                0.0
            } else if i == j {
                10.0
            } else {
                0.0
            }
        });
        let sscp_between = Mat::from_fn(p, p, |i, j| if i == j { 5.0 } else { 0.0 });
        let sscp_total = &sscp_within + &sscp_between;

        let res = ManovaResult::<f64> {
            sscp_between,
            sscp_within,
            sscp_total,
            df_between: 1,
            df_within: 10,
            df_total: 11,
            n_vars: p,
            group_means: vec![vec![0.0; p], vec![0.0; p]],
            overall_mean: vec![0.0; p],
        };

        let aov = summary_aov(&res);
        assert_eq!(aov.len(), 2);
        assert!(aov[0].f_stat.is_nan());
        assert!((aov[0].p_val - 1.0).abs() < 1e-12);
        assert!(aov[1].f_stat.is_finite() && aov[1].f_stat > 0.0);
        assert!((0.0..=1.0).contains(&aov[1].p_val));
    }
}
