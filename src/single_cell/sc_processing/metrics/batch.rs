//! Batch-mixing metrics.
//!
//! Everything here scores how well an integration has erased the batch
//! covariate: kBET, batch silhouette width, iLISI and principal component
//! regression. See Büttner et al., Nat. Methods, 2019 and Luecken et al.,
//! Nat. Methods, 2022.

use faer::MatRef;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use thousands::*;

use super::shared::{
    LisiResult, SilhouetteResult, dense_labels, lisi, lisi_weighted, silhouette_width,
};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Re-label a generic label error as a batch error.
///
/// The shared engines are label agnostic, but the batch entry points predate
/// them and downstream code matches on
/// [`BixverseErrors::NeedAtLeastTwoBatches`], so the batch-specific wording is
/// preserved here.
///
/// ### Params
///
/// * `err` - Error raised by one of the shared engines.
///
/// ### Returns
///
/// The batch-flavoured error, or `err` untouched.
fn as_batch_error(err: BixverseErrors) -> BixverseErrors {
    match err {
        BixverseErrors::NeedAtLeastTwoLabels { n_labels } => {
            BixverseErrors::NeedAtLeastTwoBatches {
                n_batches: n_labels,
            }
        }
        other => other,
    }
}

//////////
// kBET //
//////////

/// Results from the kBET calculation
pub struct KbetResult {
    /// Per-cell p-values from the chi-square test
    pub p_values: Vec<f64>,
    /// Per-cell chi-square statistics
    pub chi_square_stats: Vec<f64>,
    /// Mean chi-square statistic (effect size measure, independent of k)
    pub mean_chi_square: f64,
    /// Median chi-square statistic (robust to outliers)
    pub median_chi_square: f64,
}

/// Calculate kBET-based mixing scores on kNN data
///
/// Uses Pearson's chi-square with Yates' continuity correction for the
/// two-batch case (DoF = 1).
///
/// ### Params
///
/// * `knn_data` - KNN data. Outer vector represents the cells, inner vector
///   the neighbour indices.
/// * `batches` - Vector indicating the batch of each cell.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A `KbetResult` with per-cell p-values, chi-square statistics, and summary
/// measures.
pub fn kbet(
    knn_data: &[Vec<usize>],
    batches: &[usize],
    verbose: bool,
) -> Result<KbetResult, BixverseErrors> {
    let mut batch_counts = FxHashMap::default();
    for &batch in batches {
        *batch_counts.entry(batch).or_insert(0usize) += 1;
    }
    let total = batches.len() as f64;
    let batch_ids: Vec<usize> = batch_counts.keys().copied().collect();
    let n_batches = batch_ids.len();

    if n_batches == 1 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    let dof = (n_batches - 1) as f64;
    let use_yates = n_batches == 2;
    let n = knn_data.len();

    if verbose {
        println!("Running kBET on {} samples", n.separate_with_underscores())
    }

    let chi_sq_dist = ChiSquared::new(dof).unwrap();
    let counter = Arc::new(AtomicUsize::new(0));

    let results: Vec<(f64, f64)> = knn_data
        .par_iter()
        .map(|neighbours| {
            let k = neighbours.len() as f64;
            let mut neighbours_count = FxHashMap::default();
            for &neighbour_idx in neighbours {
                *neighbours_count
                    .entry(batches[neighbour_idx])
                    .or_insert(0usize) += 1;
            }

            let mut chi_square = 0.0;
            for &batch_id in &batch_ids {
                let expected = k * (batch_counts[&batch_id] as f64 / total);
                let observed = *neighbours_count.get(&batch_id).unwrap_or(&0) as f64;
                let diff = if use_yates {
                    (observed - expected).abs() - 0.5
                } else {
                    observed - expected
                };
                chi_square += diff * diff / expected;
            }

            if verbose {
                let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_multiple_of(100_000) {
                    println!(
                        " kBET: processed {} / {} cells.",
                        count.separate_with_underscores(),
                        n.separate_with_underscores()
                    );
                }
            }

            let p_value = 1.0 - chi_sq_dist.cdf(chi_square);
            (chi_square, p_value)
        })
        .collect();

    let chi_square_stats: Vec<f64> = results.iter().map(|(c, _)| *c).collect();
    let p_values: Vec<f64> = results.iter().map(|(_, p)| *p).collect();

    let mean_chi_square = chi_square_stats.iter().sum::<f64>() / chi_square_stats.len() as f64;

    let mut sorted_chi = chi_square_stats.clone();
    sorted_chi.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_chi_square = if sorted_chi.len().is_multiple_of(2) {
        (sorted_chi[sorted_chi.len() / 2 - 1] + sorted_chi[sorted_chi.len() / 2]) / 2.0
    } else {
        sorted_chi[sorted_chi.len() / 2]
    };

    Ok(KbetResult {
        p_values,
        chi_square_stats,
        mean_chi_square,
        median_chi_square,
    })
}

///////////////////////////
// BatchSilhouetteScores //
///////////////////////////

/// Results from batch silhouette width calculation
pub struct BatchSilhouetteResult {
    /// Per-cell silhouette scores in [-1, 1]
    pub per_cell: Vec<f32>,
    /// Mean silhouette width (closer to 0 = better mixing)
    pub mean_asw: f32,
    /// Median silhouette width
    pub median_asw: f32,
}

/// Compute batch average silhouette width on an embedding
///
/// For each cell, computes:
///   a = mean distance to cells of same batch
///   b = mean distance to cells of nearest other batch
///   s = (b - a) / max(a, b)
///
/// Values near 0 indicate good mixing, near 1 indicates separation.
///
/// ### Params
///
/// * `embedding` - Low-dimensional embedding (N x d)
/// * `batch_labels` - Batch assignment per cell (length N)
/// * `subsample` - Optional max cells to use. If Some and N exceeds this,
///   a random subsample is taken.
/// * `seed` - Random seed for subsampling
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// `BatchSilhouetteResult` with per-cell and summary scores
pub fn batch_silhouette_width(
    embedding: MatRef<f32>,
    batch_labels: &[usize],
    subsample: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Result<BatchSilhouetteResult, BixverseErrors> {
    let SilhouetteResult {
        per_cell,
        mean,
        median,
    } = silhouette_width(embedding, batch_labels, subsample, seed, verbose)
        .map_err(as_batch_error)?;

    Ok(BatchSilhouetteResult {
        per_cell,
        mean_asw: mean,
        median_asw: median,
    })
}

///////////
// iLISI //
///////////

/// Compute Local Inverse Simpson's Index on batch labels
///
/// The uniformly weighted variant, see [`lisi`]. Call
/// [`LisiResult::ilisi_norm`] on the result for the `[0, 1]` rescaling that is
/// comparable across datasets.
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per cell
/// * `batch_labels` - Batch assignment per cell (length N)
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// `LisiResult` with per-cell scores and summaries
pub fn batch_lisi(
    knn_indices: &[Vec<usize>],
    batch_labels: &[usize],
    verbose: bool,
) -> Result<LisiResult, BixverseErrors> {
    lisi(knn_indices, batch_labels, verbose).map_err(as_batch_error)
}

/// Compute the kernel-weighted Local Inverse Simpson's Index on batch labels
///
/// iLISI as published, see [`lisi_weighted`]. Prefer this over [`batch_lisi`]
/// whenever distances are to hand, since the score is then insensitive to the
/// choice of `k`.
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per cell
/// * `knn_dists` - True distances to those neighbours
/// * `batch_labels` - Batch assignment per cell (length N)
/// * `perplexity` - Target effective neighbourhood size. Defaults to
///   [`super::shared::LISI_PERPLEXITY`].
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// `LisiResult` with per-cell scores and summaries
pub fn batch_lisi_weighted(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    batch_labels: &[usize],
    perplexity: Option<f64>,
    verbose: bool,
) -> Result<LisiResult, BixverseErrors> {
    lisi_weighted(knn_indices, knn_dists, batch_labels, perplexity, verbose).map_err(as_batch_error)
}

/////////
// PCR //
/////////

/// Results from a principal component regression against a batch covariate
pub struct PcrResult {
    /// Share of the total variance carried by each embedding dimension
    pub var_explained: Vec<f64>,
    /// R^2 of the batch covariate against each embedding dimension
    pub r_squared: Vec<f64>,
    /// Variance-weighted sum of `r_squared`, in `[0, 1]`. The share of the
    /// embedding's variance attributable to batch
    pub pcr: f64,
}

/// Variance attributable to a batch covariate across an embedding
///
/// For each dimension, the R^2 of a regression on the batch indicator, weighted
/// by that dimension's share of the total variance and summed. Because the
/// covariate is categorical, the regression is exactly a one-way ANOVA, so no
/// design matrix is built and no system is solved: R^2 is the ratio of
/// between-group to total sum of squares.
///
/// The absolute value carries little meaning on its own. Report
/// [`pcr_comparison`] of a pre- and post-integration embedding instead.
///
/// ### Params
///
/// * `embedding` - Embedding, cells x dimensions. Pass a PCA embedding: the
///   per-column variance is taken as the explained-variance weight, which only
///   holds for uncorrelated components.
/// * `batches` - Batch assignment per cell, one per row of `embedding`.
///
/// ### Returns
///
/// A [`PcrResult`]. Sums are accumulated in `f64` regardless of `T`, since the
/// centred sums of squares over millions of cells cancel badly in `f32`.
///
/// ### References
///
/// Büttner et al., Nat. Methods, 2019
pub fn pcr<T>(embedding: MatRef<T>, batches: &[usize]) -> Result<PcrResult, BixverseErrors>
where
    T: BixverseFloat + Send + Sync,
{
    let n = embedding.nrows();
    let d = embedding.ncols();
    let (batches, n_batches) = dense_labels(n, batches).map_err(as_batch_error)?;

    // (between-group SS, total SS) per dimension
    let sums: Vec<(f64, f64)> = (0..d)
        .into_par_iter()
        .map(|j| {
            let mut group_sum = vec![0.0f64; n_batches];
            let mut group_count = vec![0usize; n_batches];
            let mut total_sum = 0.0f64;

            for i in 0..n {
                let x = embedding[(i, j)].to_f64().unwrap_or(0.0);
                group_sum[batches[i]] += x;
                group_count[batches[i]] += 1;
                total_sum += x;
            }

            let grand_mean = total_sum / n as f64;

            let ss_between: f64 = group_sum
                .iter()
                .zip(group_count.iter())
                .filter(|&(_, &c)| c > 0)
                .map(|(&s, &c)| {
                    let diff = s / c as f64 - grand_mean;
                    c as f64 * diff * diff
                })
                .sum();

            let ss_total: f64 = (0..n)
                .map(|i| {
                    let diff = embedding[(i, j)].to_f64().unwrap_or(0.0) - grand_mean;
                    diff * diff
                })
                .sum();

            (ss_between, ss_total)
        })
        .collect();

    let ss_grand: f64 = sums.iter().map(|(_, t)| *t).sum();

    let var_explained: Vec<f64> = if ss_grand > 0.0 {
        sums.iter().map(|(_, t)| t / ss_grand).collect()
    } else {
        vec![0.0; d]
    };

    let r_squared: Vec<f64> = sums
        .iter()
        .map(|&(b, t)| if t > 0.0 { (b / t).min(1.0) } else { 0.0 })
        .collect();

    let pcr = var_explained
        .iter()
        .zip(r_squared.iter())
        .map(|(w, r)| w * r)
        .sum();

    Ok(PcrResult {
        var_explained,
        r_squared,
        pcr,
    })
}

/// Relative reduction in batch-attributable variance after integration
///
/// `(pre - post) / pre`, the comparison score of Luecken et al. 1 means the
/// batch signal was removed entirely, 0 that nothing changed, negative that
/// integration made the batch effect worse.
///
/// ### Params
///
/// * `pre` - PCR of the unintegrated embedding.
/// * `post` - PCR of the integrated embedding.
///
/// ### Returns
///
/// The relative reduction, or `0.0` if the unintegrated embedding carried no
/// batch variance to begin with.
pub fn pcr_comparison(pre: &PcrResult, post: &PcrResult) -> f64 {
    if pre.pcr <= 0.0 {
        return 0.0;
    }
    (pre.pcr - post.pcr) / pre.pcr
}

///////////
// Tests //
///////////

#[cfg(test)]
mod batch_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A dimension that is exactly its group mean is fully explained by batch.
    #[test]
    fn test_pcr_pure_batch_signal_is_one() {
        let data: Vec<f32> = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 6, 1);

        let res = pcr(embd, &[0, 0, 0, 1, 1, 1]).unwrap();

        assert_relative_eq!(res.pcr, 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.r_squared[0], 1.0, epsilon = 1e-6);
    }

    /// Group means identical to the grand mean leave no between-group sum of
    /// squares, whatever the within-group spread.
    #[test]
    fn test_pcr_no_batch_signal_is_zero() {
        let data: Vec<f32> = vec![1.0, -1.0, 0.0, 0.0, 1.0, -1.0];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 6, 1);

        let res = pcr(embd, &[0, 0, 0, 1, 1, 1]).unwrap();

        assert_relative_eq!(res.pcr, 0.0, epsilon = 1e-6);
    }

    /// The per-dimension R^2 is weighted by that dimension's variance share:
    /// column 0 carries 6 of the 10 units of variance and all of the batch
    /// signal, column 1 the other 4 and none of it.
    #[test]
    fn test_pcr_weights_by_variance_share() {
        let data: Vec<f32> = vec![
            1.0, 1.0, 1.0, -1.0, 1.0, 0.0, -1.0, 0.0, -1.0, 1.0, -1.0, -1.0,
        ];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 6, 2);

        let res = pcr(embd, &[0, 0, 0, 1, 1, 1]).unwrap();

        assert_relative_eq!(res.r_squared[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.r_squared[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(res.var_explained[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(res.pcr, 0.6, epsilon = 1e-6);
    }

    #[test]
    fn test_pcr_comparison_reports_relative_reduction() {
        let pre = PcrResult {
            var_explained: vec![1.0],
            r_squared: vec![0.8],
            pcr: 0.8,
        };
        let post = PcrResult {
            var_explained: vec![1.0],
            r_squared: vec![0.2],
            pcr: 0.2,
        };

        assert_relative_eq!(pcr_comparison(&pre, &post), 0.75, epsilon = 1e-12);
        assert_relative_eq!(pcr_comparison(&pre, &pre), 0.0, epsilon = 1e-12);
        assert!(pcr_comparison(&post, &pre) < 0.0);
    }

    /// The shared engines raise a label error; the batch entry points must
    /// still report the batch-flavoured one downstream matches on.
    #[test]
    fn test_batch_entry_points_report_batch_error() {
        let knn = vec![vec![1], vec![0]];
        assert!(matches!(
            batch_lisi(&knn, &[7, 7], false),
            Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches: 1 })
        ));

        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 2, 2);
        assert!(matches!(
            batch_silhouette_width(embd, &[7, 7], None, 1, false),
            Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches: 1 })
        ));
    }

    /// Perfectly mixed neighbourhoods sit at the floor of kBET's statistic,
    /// batch-pure ones well above it. Yates' correction keeps the floor off
    /// zero, so the comparison is between the two rather than against zero.
    #[test]
    fn test_kbet_separates_mixed_from_unmixed() {
        let batches: Vec<usize> = (0..8).map(|i| i % 2).collect();
        let mixed: Vec<Vec<usize>> = (0..8)
            .map(|i| (1..=4).map(|o| (i + o) % 8).collect())
            .collect();
        let split: Vec<Vec<usize>> = (0..8)
            .map(|i| (0..4).map(|o| (i % 2) + 2 * o).collect())
            .collect();

        let a = kbet(&mixed, &batches, false).unwrap();
        let b = kbet(&split, &batches, false).unwrap();

        assert_relative_eq!(a.mean_chi_square, 0.25, epsilon = 1e-9);
        assert_relative_eq!(b.mean_chi_square, 2.25, epsilon = 1e-9);
        assert_eq!(a.p_values.len(), 8);
    }
}
