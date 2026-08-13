//! Implementations of the Gene Set Variation analysis from Hänzelmann, et al.,
//! Bmc Bioinformatics, 2013.

use faer::{ColRef, Mat, MatRef, RowRef};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use statrs::distribution::{DiscreteCDF, Poisson};
use std::cell::RefCell;
use std::time::Instant;

use crate::core::math::{matrix_helpers::rank_matrix_col, vector_helpers::standard_deviation};
use crate::prelude::BixverseFloat;

///////////
// Enums //
///////////

/// Kernel type for GSVA's kernel cumulative distribution function step.
#[derive(Debug, Clone, Copy, Default)]
pub enum GsvaKernel {
    /// Gaussian kernel, for continuous expression data (e.g. log-CPM, microarray)
    #[default]
    Gaussian,
    /// Poisson kernel, for integer count data (e.g. raw RNA-seq counts)
    Poisson,
    /// No kernel; rank expression values directly. Use when data is already
    /// on a comparable scale across genes (pre-normalised, z-scored, etc.)
    None,
}

/// Parse the GSVA kernel to use
///
/// ### Params
///
/// * `s` - The string to parse. One of `"gaussian"`, `"poisson"` or `"none"`.
///
/// ### Returns
///
/// The Option of the [GsvaKernel]
pub fn parse_gsva_kernel(s: &str) -> Option<GsvaKernel> {
    match s.to_lowercase().as_str() {
        "gaussian" => Some(GsvaKernel::Gaussian),
        "poisson" => Some(GsvaKernel::Poisson),
        "none" => Some(GsvaKernel::None),
        _ => None,
    }
}

/////////////
// Globals //
/////////////

/// Scaling factor for Gaussian kernel bandwidth calculation.
///
/// Bandwidth = standard_deviation / SIGMA_FACTOR
const SIGMA_FACTOR: f64 = 4.0;

/// Fixed bandwidth parameter added to lambda in Poisson kernel density estimation.
///
/// Helps prevent zero lambda values and provides numerical stability.
const POISSON_BANDWIDTH: f64 = 0.5;

thread_local! {
    /// Thread-local cache for Poisson distributions to avoid repeated allocation.
    ///
    /// Key: lambda value encoded as u32 (lambda * 10000 for precision)
    ///
    /// Value: Pre-computed Poisson distribution object
    static POISSON_CACHE: RefCell<FxHashMap<u32, Poisson>> = RefCell::new(FxHashMap::default());
}

/// Pre-computed lookup table for Poisson CDF values for common cases.
///
/// Covers lambda values from 0.1 to 10.0 (steps of 0.1) and k values 0 to 50.
static POISSON_LOOKUP: Lazy<Vec<Vec<f64>>> = Lazy::new(|| {
    let mut table = Vec::new();
    // Pre-compute for lambda from 0.1 to 10.0 in 0.1 steps
    for lambda_int in 1..=100 {
        let lambda = lambda_int as f64 / 10.0;
        let poisson = Poisson::new(lambda).unwrap();
        let mut row = Vec::new();
        // Pre-compute CDF for k = 0 to 50 (covers most cases)
        for k in 0..=50 {
            row.push(poisson.cdf(k));
        }
        table.push(row);
    }
    table
});

/// Pre-computed normal CDF values using Abramowitz & Stegun approximation.
///
/// Covers z-scores from -8.0 to 8.0 with 0.001 precision.
static NORMAL_CDF_TABLE: Lazy<Vec<f64>> = Lazy::new(|| {
    (-8000..=8000)
        .map(|i| {
            let z = i as f64 / 1000.0; // -8.0 to 8.0 with 0.001 precision
            let t = 1.0 / (1.0 + 0.2316419 * z.abs());
            let poly = t
                * (0.319381530
                    + t * (-0.356563782
                        + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
            let cdf = 1.0 - 0.39894228 * (-0.5 * z * z).exp() * poly;
            if z >= 0.0 { cdf } else { 1.0 - cdf }
        })
        .collect()
});

/////////////
// Helpers //
/////////////

//////////
// CDFs //
//////////

/// Fast approximate normal CDF using Abramowitz & Stegun approximation
///
/// Inlined for further compiler optimisations. Also, thanks Claude for the idea
///
/// ### Params
///
/// * `x` - The value for which to retrieve the CDF
/// * `mean` - The mean of the normal distribution
/// * `std_dev` - The standard deviation of the normal distribution
///
/// ### Returns
///
/// The cumulative probability P(X <= x) for X ~ N(mean, std_dev²)
#[inline(always)]
fn fast_normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let z = (x - mean) / std_dev;
    if z <= -8.0 {
        return 0.0;
    }
    if z >= 8.0 {
        return 1.0;
    }

    // Lookup table with linear interpolation
    let index = ((z + 8.0) * 1000.0) as usize;
    if index < NORMAL_CDF_TABLE.len() - 1 {
        let frac = ((z + 8.0) * 1000.0) - index as f64;
        NORMAL_CDF_TABLE[index] * (1.0 - frac) + NORMAL_CDF_TABLE[index + 1] * frac
    } else {
        NORMAL_CDF_TABLE[NORMAL_CDF_TABLE.len() - 1]
    }
}

/// Get cached Poisson or create new one
///
/// Inlined for further compiler optimisations and leveraging lookups for
/// speed
///
/// ### Parameters
///
/// * `lambda` - Rate parameter of the Poisson distribution (λ > 0)
/// * `k` - Number of events for which to compute P(X ≤ k)
///
///
#[inline(always)]
fn fast_poisson_cdf(lambda: f64, k: u64) -> f64 {
    // Fast path: use lookup table for common cases
    if (0.1..=10.0).contains(&lambda) && k <= 50 {
        let lambda_idx = ((lambda * 10.0) as usize).saturating_sub(1);
        if lambda_idx < POISSON_LOOKUP.len() {
            return POISSON_LOOKUP[lambda_idx][k as usize];
        }
    }

    // Fallback: thread-local cache
    POISSON_CACHE.with(|cache| {
        let lambda_key = (lambda * 10000.0) as u32; // Higher precision
        let mut cache_map = cache.borrow_mut();
        let poisson = cache_map
            .entry(lambda_key)
            .or_insert_with(|| Poisson::new(lambda).unwrap());
        poisson.cdf(k)
    })
}

////////////////////////
// Density estimation //
////////////////////////

/// Calculate the row-wise kernel density
///
/// Uses unsafe under the hood to be fast...
///
/// ### Params
///
/// * `density_row` - The reference distribution.
/// * `test_row` - Test values to evaluate (same as density_row for GSVA).
/// * `use_gaussian` - If true, use Gaussian kernel; if false, use Poisson kernel
/// * `bandwidth` - Bandwidth parameter (used only for Gaussian kernel)
///
/// ### Returns
///
/// Vector of log-odds transformed KCDF values: -ln((1 - left_tail) / left_tail)
/// where left_tail is the cumulative probability.
fn row_kernel_density<T>(
    density_row: RowRef<T>,
    test_row: RowRef<T>,
    use_gaussian: bool,
    bandwidth: f64,
) -> Vec<T>
where
    T: BixverseFloat,
{
    let n_test = test_row.ncols();
    let n_density = density_row.ncols();
    let mut results = Vec::with_capacity(n_test);
    let density_len_inv = 1.0_f64 / n_density as f64;

    let density_f64: Vec<f64> = density_row.iter().map(|&x| x.to_f64().unwrap()).collect();

    if use_gaussian {
        for test_val in test_row.iter() {
            let test_val_f64 = test_val.to_f64().unwrap();
            let mut left_tail = 0.0_f64;

            let mut i = 0;
            while i + 4 <= n_density {
                unsafe {
                    let d1 = *density_f64.get_unchecked(i);
                    let d2 = *density_f64.get_unchecked(i + 1);
                    let d3 = *density_f64.get_unchecked(i + 2);
                    let d4 = *density_f64.get_unchecked(i + 3);

                    left_tail += fast_normal_cdf(test_val_f64, d1, bandwidth);
                    left_tail += fast_normal_cdf(test_val_f64, d2, bandwidth);
                    left_tail += fast_normal_cdf(test_val_f64, d3, bandwidth);
                    left_tail += fast_normal_cdf(test_val_f64, d4, bandwidth);
                }
                i += 4;
            }

            while i < n_density {
                unsafe {
                    let d = *density_f64.get_unchecked(i);
                    left_tail += fast_normal_cdf(test_val_f64, d, bandwidth);
                }
                i += 1;
            }

            left_tail *= density_len_inv;
            let ratio = (1.0 - left_tail) / left_tail.max(1e-15);
            results.push(T::from_f64(-ratio.ln()).unwrap());
        }
    } else {
        for test_val in test_row.iter() {
            let test_val_u64 = test_val.to_f64().unwrap().max(0.0) as u64;
            let mut left_tail = 0.0_f64;

            let mut i = 0;
            while i + 4 <= n_density {
                unsafe {
                    let d1 = *density_f64.get_unchecked(i);
                    let d2 = *density_f64.get_unchecked(i + 1);
                    let d3 = *density_f64.get_unchecked(i + 2);
                    let d4 = *density_f64.get_unchecked(i + 3);

                    let lambda1 = d1 + POISSON_BANDWIDTH;
                    let lambda2 = d2 + POISSON_BANDWIDTH;
                    let lambda3 = d3 + POISSON_BANDWIDTH;
                    let lambda4 = d4 + POISSON_BANDWIDTH;

                    if lambda1 > 0.0 {
                        left_tail += fast_poisson_cdf(lambda1, test_val_u64);
                    }
                    if lambda2 > 0.0 {
                        left_tail += fast_poisson_cdf(lambda2, test_val_u64);
                    }
                    if lambda3 > 0.0 {
                        left_tail += fast_poisson_cdf(lambda3, test_val_u64);
                    }
                    if lambda4 > 0.0 {
                        left_tail += fast_poisson_cdf(lambda4, test_val_u64);
                    }
                }
                i += 4;
            }

            while i < n_density {
                unsafe {
                    let d = *density_f64.get_unchecked(i);
                    let lambda = d + POISSON_BANDWIDTH;
                    if lambda > 0.0 {
                        left_tail += fast_poisson_cdf(lambda, test_val_u64);
                    }
                }
                i += 1;
            }

            left_tail *= density_len_inv;
            let ratio = (1.0 - left_tail) / left_tail.max(1e-15);
            results.push(T::from_f64(-ratio.ln()).unwrap());
        }
    }

    results
}

/// Matrix kernel density estimation
///
/// Matches the C function `matrix_d()` in kernel_estimation.c from GSVA.
/// Leverages heavy parallelism and unsafe Rust under the hood to be fast!
///
/// ### Params
///
/// * `density_matrix` - Reference expression matrix (genes × samples)
/// * `test_matrix` - Test expression matrix (typically same as density_matrix)
/// * `use_gaussian` - Whether to use Gaussian (true) or Poisson (false) kernel
///
/// ### Returns
///
/// Matrix of log-odds transformed KCDF values with same dimensions as input.
pub fn matrix_kernel_density<T>(
    density_matrix: &MatRef<T>,
    test_matrix: &MatRef<T>,
    use_gaussian: bool,
) -> Mat<T>
where
    T: BixverseFloat,
{
    let (n_genes, _) = density_matrix.shape();
    let (_, n_test_samples) = test_matrix.shape();

    let bandwidths: Vec<f64> = if use_gaussian {
        (0..n_genes)
            .map(|gene_idx| {
                let row_data: Vec<f64> = density_matrix
                    .row(gene_idx)
                    .iter()
                    .map(|&x| x.to_f64().unwrap())
                    .collect();
                (standard_deviation(&row_data) / SIGMA_FACTOR).max(0.001)
            })
            .collect()
    } else {
        vec![0.0; n_genes]
    };

    let results: Vec<Vec<T>> = density_matrix
        .par_row_iter()
        .zip(test_matrix.par_row_iter())
        .zip(bandwidths.par_iter())
        .map(|((dens_row, test_row), bw)| row_kernel_density(dens_row, test_row, use_gaussian, *bw))
        .collect();

    let mut result = Mat::zeros(n_genes, n_test_samples);
    for (gene_idx, row_data) in results.into_iter().enumerate() {
        for (sample_idx, val) in row_data.into_iter().enumerate() {
            result[(gene_idx, sample_idx)] = val;
        }
    }

    result
}

///////////////////////
// Scoring functions //
///////////////////////

/// Compute gene ranking and symmetric rank statistics for enrichment analysis.
///
/// Matches the C function `order_rankstat()` in utils.c
///
/// ### Params
///
/// * `values` - KCDF values for all genes in a single sample
///
/// ### Returns
///
/// A tuple containing with `(Gene indices sorted (descending),
/// rank statistics for each gene (in original order))`
pub fn order_rankstat<T>(values: &[T]) -> (Vec<usize>, Vec<T>)
where
    T: BixverseFloat,
{
    let n = values.len();
    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());

    let mut rank_stats = vec![T::zero(); n];
    let n_t = T::from_usize(n).unwrap();
    let half_n = n_t / T::from_f64(2.0).unwrap();

    for (rank, &original_idx) in indices.iter().enumerate() {
        let rank_t = T::from_usize(rank).unwrap();
        rank_stats[original_idx] = (n_t - rank_t - half_n).abs();
    }

    (indices, rank_stats)
}

/// Perform enrichment score calculation via random walk algorithm.
///
/// ### Params
///
/// * `gene_set_ranks` - Rank positions of genes in the gene set. Must be unique;
///   duplicates double-count the walk weights.
/// * `decreasing_order_indices` - All genes sorted by rank (descending KCDF)
/// * `symmetric_rank_stats` - Rank statistics for weighting genes
/// * `tau` - Weighting exponent (1.0 = equal weights, >1.0 = emphasize extremes)
/// * `n` - Total number of genes
///
/// ### Returns
///
/// Tuple of `(max_positive_score, max_negative_score)`
#[inline]
fn gsva_random_walk<T: BixverseFloat>(
    gene_set_ranks: &[usize],
    decreasing_order_indices: &[usize],
    symmetric_rank_stats: &[T],
    tau: f64,
    n: usize,
) -> (T, T) {
    let mut weights = Vec::with_capacity(gene_set_ranks.len());
    let mut total_in = T::zero();

    for &rank in gene_set_ranks {
        let orig_gene_idx = decreasing_order_indices[rank];
        let stat_val = symmetric_rank_stats[orig_gene_idx];
        let weight = if tau == 1.0 {
            stat_val
        } else {
            T::from_f64(stat_val.to_f64().unwrap().powf(tau)).unwrap()
        };
        weights.push((rank, weight));
        total_in += weight;
    }

    if total_in <= T::zero() {
        return (T::nan(), T::nan());
    }

    weights.sort_unstable_by_key(|&(rank, _)| rank);

    // Saturating: a caller that ignores the uniqueness contract falls into the
    // `total_out <= 0` NaN arm rather than underflowing.
    let total_out = T::from_usize(n.saturating_sub(gene_set_ranks.len())).unwrap();
    if total_out <= T::zero() {
        return (T::nan(), T::nan());
    }

    let total_in_inv = T::one() / total_in;
    let total_out_inv = T::one() / total_out;

    let mut max_pos = T::zero();
    let mut max_neg = T::zero();
    let mut cumulative_in = T::zero();
    let mut cumulative_out = T::zero();
    let mut weight_idx = 0;

    for i in 0..n {
        if weight_idx < weights.len() && weights[weight_idx].0 == i {
            cumulative_in += weights[weight_idx].1;
            weight_idx += 1;
        } else {
            cumulative_out += T::one();
        }

        let walk_stat = cumulative_in * total_in_inv - cumulative_out * total_out_inv;

        if walk_stat > max_pos {
            max_pos = walk_stat;
        }
        if walk_stat < max_neg {
            max_neg = walk_stat;
        }
    }

    (max_pos, max_neg)
}

/// Calculate GSVA enrichment scores for all gene sets in a sample.
///
/// Matches the C function `gsva_score_genesets_R()` in ks_test.c
///
/// ### Params
///
/// * `gene_sets` - Vector of gene sets, each containing gene indices. Repeated
///   indices are de-duplicated.
/// * `decreasing_order_indices` - Genes sorted by KCDF values (descending)
/// * `symmetric_rank_stats` - Rank statistics for weighting genes
/// * `tau` - Weighting exponent (1.0 = equal weights, >1.0 = emphasize extremes)
/// * `max_diff` - Scoring mode: true = difference, false = larger absolute value
/// * `abs_rank` - If max_diff=true: true = pos-neg, false = pos+neg
///
/// ### Returns
///
/// Vector of ES, one per gene set. `NaN` for a set that is empty, matches no
/// gene, carries zero total weight, or covers the whole background.
pub fn gsva_score_genesets<T: BixverseFloat>(
    gene_sets: &[Vec<usize>],
    decreasing_order_indices: &[usize],
    symmetric_rank_stats: &[T],
    tau: f64,
    max_diff: bool,
    abs_rank: bool,
) -> Vec<T> {
    let n = decreasing_order_indices.len();

    let rank_lookup: FxHashMap<usize, usize> = decreasing_order_indices
        .iter()
        .enumerate()
        .map(|(rank, &gene_idx)| (gene_idx, rank))
        .collect();

    gene_sets
        .iter()
        .map(|gene_set| {
            if gene_set.is_empty() {
                return T::nan();
            }

            let mut gene_set_ranks = Vec::with_capacity(gene_set.len());
            for &gene_idx in gene_set {
                if let Some(&rank) = rank_lookup.get(&gene_idx) {
                    gene_set_ranks.push(rank);
                }
            }

            // Nothing upstream promises unique indices, and a repeated gene would
            // both double-count its weight and inflate the in-set size past `n`.
            gene_set_ranks.sort_unstable();
            gene_set_ranks.dedup();

            if gene_set_ranks.is_empty() {
                return T::nan();
            }

            let (walk_stat_pos, walk_stat_neg) = gsva_random_walk(
                &gene_set_ranks,
                decreasing_order_indices,
                symmetric_rank_stats,
                tau,
                n,
            );

            if walk_stat_pos.is_nan() || walk_stat_neg.is_nan() {
                T::nan()
            } else if max_diff {
                if abs_rank {
                    walk_stat_pos - walk_stat_neg
                } else {
                    walk_stat_pos + walk_stat_neg
                }
            } else if walk_stat_pos > walk_stat_neg.abs() {
                walk_stat_pos
            } else {
                walk_stat_neg
            }
        })
        .collect()
}

/// Calculate rank weights for ssGSEA
///
/// Equivalent to R's abs(R)^alpha
///
/// ### Params
///
/// * `ranks` - The ranked matrix
/// * `alpha` - The alpha parameter
///
/// ### Returns
///
/// The rank marix that was powered by alpha
fn calculate_rank_weights<T: BixverseFloat>(ranks: &MatRef<T>, alpha: f64) -> Mat<T> {
    let (n_genes, n_samples) = ranks.shape();
    let mut weights = Mat::zeros(n_genes, n_samples);

    weights
        .par_col_iter_mut()
        .enumerate()
        .for_each(|(col_idx, mut col)| {
            let ranks_col = ranks.col(col_idx);
            for (row_idx, &rank) in ranks_col.iter().enumerate() {
                col[row_idx] = T::from_f64(rank.abs().to_f64().unwrap().powf(alpha)).unwrap();
            }
        });

    weights
}

/// Fast random walk for ssGSEA
///
/// Equivalent to .fastRndWalk in R in the GSVA package
///
/// ### Params
/// * `gene_set_indices` - Original gene indices in the gene set (0-based)
/// * `gene_ranking` - Genes sorted by rank (decreasing order)
/// * `rank_weights` - Rank weights for the sample
/// * `sample_idx` - Current sample index
///
/// ### Returns
///
/// ssGSEA enrichment score for this gene set and sample
#[inline]
fn ssgsea_fast_random_walk<T: BixverseFloat>(
    gene_set_indices: &[usize],
    rank_lookup: &FxHashMap<usize, usize>,
    gene_ranking: &[usize],
    rank_weights: &ColRef<T>,
) -> T {
    let n = gene_ranking.len();
    let k = gene_set_indices.len();

    if k == 0 {
        return T::nan();
    }

    if k < 32 {
        let gene_set: FxHashSet<usize> = gene_set_indices.iter().copied().collect();
        let mut sum_weighted_ranks = T::zero();
        let mut sum_weights = T::zero();
        let mut sum_gene_set_ranks = T::zero();
        let mut found_genes = 0;

        for (rank_pos, &gene_idx) in gene_ranking.iter().enumerate() {
            if gene_set.contains(&gene_idx) {
                let weight = rank_weights[gene_idx];
                let rank_contribution = T::from_usize(n - rank_pos).unwrap();

                sum_weighted_ranks += weight * rank_contribution;
                sum_weights += weight;
                sum_gene_set_ranks += rank_contribution;
                found_genes += 1;

                if found_genes == k {
                    break;
                }
            }
        }

        if found_genes == 0 {
            return T::nan();
        }

        let step_cdf_in_gene_set = sum_weighted_ranks / sum_weights;
        let n_t = T::from_usize(n).unwrap();
        let sum_all_ranks = n_t * (n_t + T::one()) / T::from_f64(2.0).unwrap();
        let step_cdf_out_gene_set =
            (sum_all_ranks - sum_gene_set_ranks) / T::from_usize(n - found_genes).unwrap();

        return step_cdf_in_gene_set - step_cdf_out_gene_set;
    }

    let mut sum_weighted_ranks = T::zero();
    let mut sum_weights = T::zero();
    let mut sum_gene_set_ranks = T::zero();
    let mut found_genes = 0;

    for &gene_idx in gene_set_indices {
        if let Some(&rank_pos) = rank_lookup.get(&gene_idx) {
            let weight = rank_weights[gene_idx];
            let rank_contribution = T::from_usize(n - rank_pos).unwrap();

            sum_weighted_ranks += weight * rank_contribution;
            sum_weights += weight;
            sum_gene_set_ranks += rank_contribution;
            found_genes += 1;
        }
    }

    if found_genes == 0 || sum_weights <= T::zero() {
        return T::nan();
    }

    let step_cdf_in_gene_set = sum_weighted_ranks / sum_weights;
    let n_t = T::from_usize(n).unwrap();
    let sum_all_ranks = n_t * (n_t + T::one()) / T::from_f64(2.0).unwrap();
    let step_cdf_out_gene_set =
        (sum_all_ranks - sum_gene_set_ranks) / T::from_usize(n - found_genes).unwrap();

    step_cdf_in_gene_set - step_cdf_out_gene_set
}

////////////////////
// Main functions //
////////////////////

//////////
// GSVA //
//////////

/// GSVA
///
/// Follows the algorithm described in the paper and implemented in the C code.
/// This function orchestrates the complete GSVA pipeline, from kernel density
/// estimation through enrichment scoring, with extensive parallelization and
/// performance monitoring capabilities.
///
/// ### Params
///
/// * `expression_matrix` - Gene expression data (genes × samples)
/// * `gene_sets` - Vector of gene sets as index vectors
/// * `use_gaussian` - Kernel type: true=Gaussian, false=Poisson
/// * `tau` - Gene weighting exponent (typically 1.0)
/// * `max_diff` - Scoring mode for enrichment calculation
/// * `abs_rank` - Additional scoring parameter
/// * `print_timings` - Enable detailed performance output
///
/// ### Returns
///
/// Matrix of enrichment scores (gene_sets × samples)
pub fn gsva<T: BixverseFloat>(
    expression_matrix: &MatRef<T>,
    gene_sets: &[Vec<usize>],
    kernel: GsvaKernel,
    tau: f64,
    max_diff: bool,
    abs_rank: bool,
    print_timings: bool,
) -> Mat<T> {
    let start_total = Instant::now();
    let (_, n_samples) = expression_matrix.shape();
    let mut result = Mat::zeros(gene_sets.len(), n_samples);

    if print_timings {
        println!(
            "Starting GSVA with {} samples and {} gene sets",
            n_samples,
            gene_sets.len()
        );
    }

    let start_kcdf = Instant::now();
    let kcdf_matrix = match kernel {
        GsvaKernel::Gaussian => matrix_kernel_density(expression_matrix, expression_matrix, true),
        GsvaKernel::Poisson => matrix_kernel_density(expression_matrix, expression_matrix, false),
        GsvaKernel::None => expression_matrix.to_owned(),
    };
    let kcdf_time = start_kcdf.elapsed();

    if print_timings {
        match kernel {
            GsvaKernel::None => println!("Step 1 - KCDF skipped: {:.2?}", kcdf_time),
            _ => println!("Step 1 - Kernel density estimation: {:.2?}", kcdf_time),
        }
    }

    let start_parallel = Instant::now();

    result
        .par_col_iter_mut()
        .enumerate()
        .for_each(|(sample_idx, mut col)| {
            let sample_start = Instant::now();

            let extract_start = Instant::now();
            let sample_kcdf: Vec<T> = kcdf_matrix.col(sample_idx).iter().copied().collect();
            let extract_time = extract_start.elapsed();

            let rank_start = Instant::now();
            let (decreasing_order, rank_stats) = order_rankstat(&sample_kcdf);
            let rank_time = rank_start.elapsed();

            let score_start = Instant::now();
            let scores = gsva_score_genesets(
                gene_sets,
                &decreasing_order,
                &rank_stats,
                tau,
                max_diff,
                abs_rank,
            );
            let score_time = score_start.elapsed();

            for (gene_set_idx, &score) in scores.iter().enumerate() {
                col[gene_set_idx] = score;
            }

            let sample_total = sample_start.elapsed();

            if print_timings
                && (sample_idx < 5 || sample_idx % 100 == 0 || sample_idx >= n_samples - 5)
            {
                println!(
                    "Sample {}: total={:.2?}, extract={:.2?}, rank={:.2?}, score={:.2?}",
                    sample_idx, sample_total, extract_time, rank_time, score_time
                );
            }
        });

    let parallel_time = start_parallel.elapsed();
    let total_time = start_total.elapsed();

    if print_timings {
        println!("Step 2-4 - Parallel processing: {:.2?}", parallel_time);
        println!("Total GSVA time: {:.2?}", total_time);
        println!(
            "Time breakdown: KCDF={:.1?} ({:.1}%), Parallel={:.1?} ({:.1}%)",
            kcdf_time,
            kcdf_time.as_secs_f64() / total_time.as_secs_f64() * 100.0,
            parallel_time,
            parallel_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
    }

    result
}

////////////
// ssGSEA //
////////////

/// ssGSEA
///
/// Translation from the original R code into Rust with performance
/// optimisations
///
/// ### Params
///
/// * `expression_matrix` - Gene expression data (genes × samples)
/// * `gene_sets` - Vector of gene sets as index vectors
/// * `alpha` - The exponent defining the weight of the tail in the random walk
///   performed by ssGSEA.
/// * `normalisation` - Shall the extract score be normalised.
/// * `print_timings` - Enable detailed performance output
///
/// ### Returns
///
/// Matrix of enrichment scores (gene_sets × samples)
pub fn ssgsea<T: BixverseFloat>(
    expression_matrix: &MatRef<T>,
    gene_sets: &[Vec<usize>],
    alpha: f64,
    normalisation: bool,
    print_timings: bool,
) -> Mat<T> {
    let start_total = Instant::now();
    let (n_genes, n_samples) = expression_matrix.shape();

    if print_timings {
        println!(
            "Starting ssGSEA with {} samples and {} gene sets",
            n_samples,
            gene_sets.len()
        );
    }

    let start_ranks = Instant::now();
    let ranks = rank_matrix_col(expression_matrix);
    let ranks_time = start_ranks.elapsed();

    if print_timings {
        println!("Step 1 - Calculating ranks: {:.2?}", ranks_time);
    }

    let start_weights = Instant::now();
    let rank_weights = calculate_rank_weights(&ranks.as_ref(), alpha);
    let weights_time = start_weights.elapsed();

    if print_timings {
        println!("Step 2 - Calculating rank weights: {:.2?}", weights_time);
    }

    let start_scoring = Instant::now();
    let mut result: Mat<T> = Mat::zeros(gene_sets.len(), n_samples);

    result
        .par_col_iter_mut()
        .enumerate()
        .for_each(|(sample_idx, mut col)| {
            let sample_start = Instant::now();

            let ranking_start = Instant::now();
            let sample_ranks: Vec<T> = ranks.col(sample_idx).iter().copied().collect();
            let mut gene_ranking: Vec<usize> = (0..n_genes).collect();

            gene_ranking
                .sort_unstable_by(|&a, &b| sample_ranks[b].partial_cmp(&sample_ranks[a]).unwrap());
            let ranking_time = ranking_start.elapsed();

            let score_start = Instant::now();

            let mut rank_lookup = FxHashMap::with_capacity_and_hasher(n_genes, FxBuildHasher);
            for (rank_pos, &gene_idx) in gene_ranking.iter().enumerate() {
                rank_lookup.insert(gene_idx, rank_pos);
            }

            let rank_weights_col = rank_weights.col(sample_idx);

            let scores: Vec<T> = gene_sets
                .iter()
                .map(|gene_set| {
                    ssgsea_fast_random_walk(
                        gene_set,
                        &rank_lookup,
                        &gene_ranking,
                        &rank_weights_col,
                    )
                })
                .collect();
            let score_end = score_start.elapsed();

            for (gene_set_idx, &score) in scores.iter().enumerate() {
                col[gene_set_idx] = score;
            }

            let sample_total = sample_start.elapsed();

            if print_timings
                && (sample_idx < 5 || sample_idx % 100 == 0 || sample_idx >= n_samples - 5)
            {
                println!(
                    "Sample {}: total={:.2?}, ranking={:.2?}, scoring={:.2?}",
                    sample_idx, sample_total, ranking_time, score_end
                );
            }
        });

    let scoring_time = start_scoring.elapsed();

    if print_timings {
        println!(
            "Step 3 - Calculating enrichment scores: {:.2?}",
            scoring_time
        );
    }

    if normalisation {
        let start_norm = Instant::now();

        let mut min_score = T::infinity();
        let mut max_score = T::neg_infinity();
        let mut has_finite = false;

        for col_idx in 0..result.ncols() {
            for &score in result.col(col_idx).iter() {
                if score.is_finite() {
                    min_score = min_score.min(score);
                    max_score = max_score.max(score);
                    has_finite = true;
                }
            }
        }

        if !has_finite {
            return result;
        }

        let range = max_score - min_score;

        if range > T::zero() && range.is_finite() {
            result.par_col_iter_mut().for_each(|col| {
                for score in col.iter_mut() {
                    if score.is_finite() {
                        *score /= range;
                    }
                }
            });
        }

        let norm_time = start_norm.elapsed();

        if print_timings {
            println!("Step 4 - Normalisation: {:.2?}", norm_time);
        }
    }

    let total_time = start_total.elapsed();

    if print_timings {
        println!("Total ssGSEA time: {:.2?}", total_time);
    }

    result
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Pins the descending order and the symmetric rank statistics for an even
    /// background, where `half_n` is a whole number.
    #[test]
    fn order_rankstat_ranks_descending_and_centres_the_statistics() {
        let values: Vec<f64> = vec![0.1, 0.5, 0.3, 0.9];
        let (order, stats) = order_rankstat(&values);

        assert_eq!(order, vec![3, 1, 2, 0]);
        // |n - rank - n/2| = |2 - rank| at ranks 3, 1, 2, 0 for genes 0..3.
        assert_eq!(stats, vec![1.0, 1.0, 0.0, 2.0]);
    }

    /// Pins the odd-n behaviour: `half_n` is the float `n / 2.0`, so every rank
    /// statistic lands on a half-integer. GSVA's C `utils.c` was not available to
    /// check, so this records what the Rust does rather than a claimed reference.
    #[test]
    fn order_rankstat_half_n_stays_fractional_for_odd_backgrounds() {
        let values: Vec<f64> = vec![0.5, 0.1, 0.4, 0.2, 0.3];
        let (order, stats) = order_rankstat(&values);

        assert_eq!(order, vec![0, 2, 4, 3, 1]);
        // |5 - rank - 2.5| = |2.5 - rank| at ranks 0, 4, 1, 3, 2 for genes 0..4.
        assert_eq!(stats, vec![2.5, 1.5, 1.5, 0.5, 0.5]);
        assert!(stats.iter().all(|s| (s - s.floor() - 0.5).abs() < 1e-15));
    }

    /// Pins the full random walk plus all three scoring modes on a set whose walk
    /// reaches `(pos, neg) = (0.0, -0.5)`.
    #[test]
    fn gsva_score_genesets_covers_all_three_scoring_modes() {
        let order = vec![3_usize, 1, 2, 0];
        let stats: Vec<f64> = vec![1.0, 1.0, 0.0, 2.0];
        let gene_sets = vec![vec![1_usize, 0]];

        // Walk over ranks 0..4 with in-ranks {1, 3}, total_in = 2, total_out = 2:
        // -0.5, 0.0, -0.5, 0.0 => max_pos 0.0, max_neg -0.5.
        let max_diff_abs = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, true, true);
        let max_diff_sum = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, true, false);
        let largest_abs = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, false, false);

        assert_relative_eq!(max_diff_abs[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(max_diff_sum[0], -0.5, epsilon = 1e-12);
        assert_relative_eq!(largest_abs[0], -0.5, epsilon = 1e-12);
    }

    /// Pins the other arm of `max_diff = false`, where the positive excursion
    /// beats the negative one and is returned instead.
    #[test]
    fn gsva_score_genesets_returns_the_positive_walk_when_it_dominates() {
        let order = vec![3_usize, 1, 2, 0];
        let stats: Vec<f64> = vec![1.0, 1.0, 0.0, 2.0];
        // Gene 3 sits at rank 0, so the walk starts at 1.0 and decays to 0.0.
        let gene_sets = vec![vec![3_usize]];

        let largest_abs = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, false, false);
        let max_diff_abs = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, true, true);

        assert_relative_eq!(largest_abs[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(max_diff_abs[0], 1.0, epsilon = 1e-12);
    }

    /// Pins the four NaN paths: empty set, no gene matched, zero total weight and
    /// zero genes outside the set.
    #[test]
    fn gsva_score_genesets_returns_nan_on_every_degenerate_set() {
        let order = vec![3_usize, 1, 2, 0];
        let stats: Vec<f64> = vec![1.0, 1.0, 0.0, 2.0];
        let gene_sets = vec![
            vec![],           // empty gene set
            vec![99_usize],   // no index present in the ranking
            vec![2_usize],    // rank statistic is exactly zero => total_in == 0
            vec![0, 1, 2, 3], // whole background => total_out == 0
        ];

        let scores = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, true, true);

        assert!(scores.iter().all(|s| s.is_nan()), "got {scores:?}");
    }

    /// Regression: repeated gene indices used to double-count the walk weights and
    /// underflow `n - gene_set_ranks.len()` once the set outgrew the background.
    #[test]
    fn gsva_score_genesets_deduplicates_repeated_gene_indices() {
        let order = vec![3_usize, 1, 2, 0];
        let stats: Vec<f64> = vec![1.0, 1.0, 0.0, 2.0];
        let gene_sets = vec![
            vec![1_usize, 0],          // reference, scores 0.5
            vec![1_usize, 0, 1, 0],    // duplicated up to n, used to give total_out == 0
            vec![1_usize, 0, 1, 0, 1], // longer than n, used to panic on the subtraction
        ];

        let scores = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, true, true);

        assert_relative_eq!(scores[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(scores[1], 0.5, epsilon = 1e-12);
        assert_relative_eq!(scores[2], 0.5, epsilon = 1e-12);
    }

    /// Regression: a gene set that still covers the whole background after
    /// de-duplication has no genes outside it, so it must stay `NaN`.
    #[test]
    fn gsva_score_genesets_returns_nan_for_a_duplicated_full_background_set() {
        let order = vec![3_usize, 1, 2, 0];
        let stats: Vec<f64> = vec![1.0, 1.0, 0.0, 2.0];
        let gene_sets = vec![vec![0_usize, 1, 2, 3, 3, 0]];

        let scores = gsva_score_genesets(&gene_sets, &order, &stats, 1.0, true, true);

        assert!(scores[0].is_nan(), "got {scores:?}");
    }

    /// Pins the kernel string parsing, including the case folding and the `None`
    /// returned for anything unrecognised.
    #[test]
    fn parse_gsva_kernel_round_trips_the_three_kernel_names() {
        assert!(matches!(
            parse_gsva_kernel("gaussian"),
            Some(GsvaKernel::Gaussian)
        ));
        assert!(matches!(
            parse_gsva_kernel("Poisson"),
            Some(GsvaKernel::Poisson)
        ));
        assert!(matches!(parse_gsva_kernel("NONE"), Some(GsvaKernel::None)));
        assert!(parse_gsva_kernel("normal").is_none());
        assert!(parse_gsva_kernel("").is_none());
    }
}
