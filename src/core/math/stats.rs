use faer::{Mat, linalg::solvers::DenseSolveCore};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use statrs::distribution::FisherSnedecor;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;

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
    if q == 0 {
        return T::one();
    }

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

/// Calculate the FDR
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
    let n_t = T::from_usize(n).unwrap();
    let one = T::one();

    let mut indexed_pval: Vec<(usize, T)> =
        pvals.par_iter().enumerate().map(|(i, &x)| (i, x)).collect();

    indexed_pval
        .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

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
///
/// ### Fields
///
/// * `sscp_between` - Between-groups SSCP matrix
/// * `sscp_within` - Within-groups SSCP matrix
/// * `sscp_total` - Total SSCP matrix
/// * `df_between` - Degrees of freedom between groups
/// * `df_within` - Degrees of freedom within groups
/// * `df_total` - Total degrees of freedom
/// * `n_vars` - Number of variables
/// * `group_means` - Means for each group
/// * `overall_mean` - Overall means
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ManovaResult<T>
where
    T: BixverseFloat,
{
    pub sscp_between: Mat<T>,
    pub sscp_within: Mat<T>,
    pub sscp_total: Mat<T>,
    pub df_between: usize,
    pub df_within: usize,
    pub df_total: usize,
    pub n_vars: usize,
    pub group_means: Vec<Vec<T>>,
    pub overall_mean: Vec<T>,
}

#[allow(dead_code)]
impl<T> ManovaResult<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Calculate Wilks' Lambda statistic
    ///
    /// ### Returns
    ///
    /// Function will return the Wilks' Lambda
    pub fn wilks_lambda(&self) -> T {
        let w_plus_b = &self.sscp_within + &self.sscp_between;
        let det_w = self.sscp_within.determinant();
        let det_total = w_plus_b.determinant();
        det_w / det_total
    }

    /// Calculate Pillai's trace statistic
    ///
    /// ### Returns
    ///
    /// Function will return Pillai's trace
    pub fn pillai_trace(&self) -> T {
        let w_plus_b = &self.sscp_within + &self.sscp_between;
        let w_plus_b_pu = w_plus_b.partial_piv_lu();
        let w_plus_in = w_plus_b_pu.inverse();
        let h_times_inv = &self.sscp_between * w_plus_in;

        h_times_inv
            .diagonal()
            .column_vector()
            .iter()
            .copied()
            .sum::<T>()
    }

    /// Get F-statistic and p-value for Wilks' Lambda
    ///
    /// ### Returns
    ///
    /// A tuple with the F statistic and p-value according to Wilks'
    pub fn wilks_f_test(&self) -> (T, T) {
        let lambda = self.wilks_lambda();
        let p = T::from_usize(self.n_vars).unwrap();
        let q = T::from_usize(self.df_between).unwrap();
        let one = T::one();
        let two = T::from_usize(2).unwrap();

        // special case for n_vars == 2
        if self.n_vars == 2 {
            let df1 = two * q;
            let df2 = two * (T::from_usize(self.df_within).unwrap() - one);
            let f_stat = ((one - lambda.sqrt()) / lambda.sqrt()) * (df2 / df1);

            let f_dist = FisherSnedecor::new(df1.to_f64().unwrap(), df2.to_f64().unwrap()).unwrap();
            let p_value = T::from_f64(1.0 - f_dist.cdf(f_stat.to_f64().unwrap())).unwrap();
            return (f_stat, p_value);
        }

        let n = T::from_usize(self.df_within + self.df_between + 1).unwrap();
        let five = T::from_usize(5).unwrap();

        let t = ((p * p + q * q - five).max(T::zero())).sqrt();
        let w = n - (p + q + one) / two;
        let df1 = p * q;
        let df2 = w * t - (p * q - two) / two;

        let lambda_root = if t > one {
            lambda.powf(one / t)
        } else {
            lambda
        };

        let f_stat = ((one - lambda_root) / lambda_root) * (df2 / df1);

        let f_dist = FisherSnedecor::new(df1.to_f64().unwrap(), df2.to_f64().unwrap()).unwrap();
        let p_value = T::from_f64(1.0 - f_dist.cdf(f_stat.to_f64().unwrap())).unwrap();

        (f_stat, p_value)
    }

    /// Get F-statistic and p-value for Pillai's trace
    ///
    /// (Version used in R)
    ///
    /// ### Returns
    ///
    /// A tuple with the F statistic and p-value according to Pillai
    pub fn pillai_f_test(&self) -> (T, T) {
        let pillai = self.pillai_trace();
        let p = T::from_usize(self.n_vars).unwrap();
        let q = T::from_usize(self.df_between).unwrap();
        let n = T::from_usize(self.df_within).unwrap();
        let one = T::one();

        // F approximation for Pillai's trace
        let df1 = p * q;
        let df2 = q * (n - p + one);

        let f_stat = (pillai / (q - pillai)) * (df2 / df1);

        let f_dist = FisherSnedecor::new(df1.to_f64().unwrap(), df2.to_f64().unwrap()).unwrap();
        let p_value = T::from_f64(1.0 - f_dist.cdf(f_stat.to_f64().unwrap())).unwrap();

        (f_stat, p_value)
    }
}

/// ManovaSummary
///
/// ### Fields
///
/// * `wilks_lambda` - Wilks' lambda value
/// * `pillai_trace` - Pillai's trace value
/// * `df_between` - Degrees of freedom between groups
/// * `df_within` - Degrees of freedom within groups
/// * `f_stat_wilk` - F statistic according to Wilk
/// * `p_val_wilk` - P-value according to Wilk
/// * `f_stat_pillai` - F statistic according to Pillai
/// * `p_val_pillai` - P-value according to Pillai
#[derive(Debug)]
#[allow(dead_code)]
pub struct ManovaSummary<T>
where
    T: BixverseFloat,
{
    pub wilks_lambda: T,
    pub pillai_trace: T,
    pub df_between: usize,
    pub df_within: usize,
    pub f_stat_wilk: T,
    pub p_val_wilk: T,
    pub f_stat_pillai: T,
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
///
/// ### Fields
///
/// * `variable_index` - Variable index
/// * `ss_between` - Sum of squares between groups
/// * `ss_within` - Sum of squares within groups
/// * `ms_between` - Mean square between groups
/// * `ms_within` - Mean square within groups
/// * `f_stat` - F statistic
/// * `p_val` - P-value
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AnovaSummary<T>
where
    T: BixverseFloat,
{
    pub variable_index: usize,
    pub ss_between: T,
    pub ss_within: T,
    pub ms_between: T,
    pub ms_within: T,
    pub f_stat: T,
    pub p_val: T,
}

pub fn summary_aov<T>(res: &ManovaResult<T>) -> Vec<AnovaSummary<T>>
where
    T: BixverseFloat,
{
    let mut aov_res = Vec::with_capacity(res.n_vars);

    for var_idx in 0..res.n_vars {
        let ss_between = res.sscp_between[(var_idx, var_idx)];
        let ss_within = res.sscp_within[(var_idx, var_idx)];
        let ms_between = ss_between / T::from_usize(res.df_between).unwrap();
        let ms_within = ss_within / T::from_usize(res.df_within).unwrap();
        let f_stat = ms_between / ms_within;

        let f_dist = FisherSnedecor::new(
            T::from_usize(res.df_between).unwrap().to_f64().unwrap(),
            T::from_usize(res.df_within).unwrap().to_f64().unwrap(),
        )
        .unwrap();
        let pval = T::from_f64(1.0 - f_dist.cdf(f_stat.to_f64().unwrap())).unwrap();

        aov_res.push(AnovaSummary {
            variable_index: var_idx,
            ss_between,
            ss_within,
            ms_between,
            ms_within,
            f_stat,
            p_val: pval,
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
