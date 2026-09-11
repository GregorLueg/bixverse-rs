//! Label-generic engines shared by the batch-mixing and biology-conservation
//! metrics.
//!
//! Both families ask the same two questions of a labelling, only with opposite
//! expectations: how many distinct labels sit in a cell's neighbourhood (LISI)
//! and how far the labelled groups sit apart in an embedding (silhouette).
//! One implementation of each lives here so that a batch metric and its
//! biology counterpart cannot drift apart numerically.

use ann_search_rs::utils::dist::euclidean_distance_static;
use faer::MatRef;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use thousands::*;

use crate::core::math::vector_helpers::median;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Default perplexity for the kernel-weighted LISI.
///
/// Korsunsky et al., Nat. Methods, 2019 calibrate each cell's kernel bandwidth
/// to an effective neighbourhood size of 30, which is what makes the weighted
/// score far less sensitive to `k` than a plain neighbour count.
pub const LISI_PERPLEXITY: f64 = 30.0;

/// Bisection steps for the per-cell bandwidth search.
///
/// The entropy is monotone in `beta`, so 50 halvings pin it down well below
/// the tolerance below for any neighbourhood that is not degenerate.
const LISI_BETA_STEPS: usize = 50;

/// Absolute tolerance on the log-perplexity during the bandwidth search.
const LISI_ENTROPY_TOL: f64 = 1e-5;

/// Cells between progress reports in the verbose paths.
const PROGRESS_INTERVAL: usize = 100_000;

/////////////
// Helpers //
/////////////

/// Compact an arbitrary label vector onto `0..n_labels` and validate it.
///
/// Label codes arriving from R need not be dense or zero based once a dataset
/// has been subset, so every metric here remaps first rather than sizing its
/// count buffers off `max + 1`. First occurrence order decides the new codes,
/// which keeps the mapping deterministic without a sort.
///
/// ### Params
///
/// * `n_cells` - Number of cells the metric is being computed over.
/// * `labels` - Label code per cell.
///
/// ### Returns
///
/// `(dense labels, n_labels)`, or an error if the length does not match
/// `n_cells` or fewer than two distinct labels are present.
pub(super) fn dense_labels(
    n_cells: usize,
    labels: &[usize],
) -> Result<(Vec<usize>, usize), BixverseErrors> {
    if labels.len() != n_cells {
        return Err(BixverseErrors::MetricLabelLengthMismatch {
            n_cells,
            n_labels: labels.len(),
        });
    }

    let mut remap: FxHashMap<usize, usize> = FxHashMap::default();
    let dense: Vec<usize> = labels
        .iter()
        .map(|&l| {
            let next = remap.len();
            *remap.entry(l).or_insert(next)
        })
        .collect();

    let n_labels = remap.len();
    if n_labels < 2 {
        return Err(BixverseErrors::NeedAtLeastTwoLabels { n_labels });
    }

    Ok((dense, n_labels))
}

/// Mean and median of a score vector.
///
/// ### Params
///
/// * `scores` - Per-cell scores.
///
/// ### Returns
///
/// `(mean, median)`. Both are `0.0` for an empty slice.
pub(super) fn summarise(scores: &[f32]) -> (f32, f32) {
    if scores.is_empty() {
        return (0.0, 0.0);
    }
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    (mean, median(scores).unwrap_or(0.0))
}

/// Report progress every [`PROGRESS_INTERVAL`] cells.
///
/// ### Params
///
/// * `counter` - Shared cell counter.
/// * `name` - Name of the calling metric.
/// * `n` - Total number of cells.
fn report_progress(counter: &AtomicUsize, name: &str, n: usize) {
    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_multiple_of(PROGRESS_INTERVAL) {
        println!(
            " {}: processed {} / {} cells.",
            name,
            count.separate_with_underscores(),
            n.separate_with_underscores()
        );
    }
}

//////////
// LISI //
//////////

/// Results from a LISI calculation
#[derive(Clone, Debug)]
pub struct LisiResult {
    /// Per-cell LISI scores, in `[1, n_labels]`
    pub per_cell: Vec<f32>,
    /// Mean LISI across all cells
    pub mean_lisi: f32,
    /// Median LISI across all cells
    pub median_lisi: f32,
    /// Number of distinct labels the score was computed against
    pub n_labels: usize,
}

impl LisiResult {
    /// iLISI rescaled onto `[0, 1]`, where 1 is perfect batch mixing.
    ///
    /// `(median - 1) / (n_labels - 1)`, the rescaling used by Luecken et al.,
    /// Nat. Methods, 2022 so that scores stay comparable across datasets with
    /// different batch counts.
    ///
    /// ### Returns
    ///
    /// The normalised score. Use this when the labels are batches.
    pub fn ilisi_norm(&self) -> f32 {
        if self.n_labels < 2 {
            return 0.0;
        }
        (self.median_lisi - 1.0) / (self.n_labels - 1) as f32
    }

    /// cLISI rescaled onto `[0, 1]`, where 1 is perfect label separation.
    ///
    /// `(n_labels - median) / (n_labels - 1)`, the mirror image of
    /// [`LisiResult::ilisi_norm`], so that 1 is the good end for both.
    ///
    /// ### Returns
    ///
    /// The normalised score. Use this when the labels are cell types.
    pub fn clisi_norm(&self) -> f32 {
        if self.n_labels < 2 {
            return 0.0;
        }
        (self.n_labels as f32 - self.median_lisi) / (self.n_labels - 1) as f32
    }
}

/// Local Inverse Simpson's Index over uniformly weighted neighbours
///
/// For each cell, the effective number of labels in its neighbourhood:
///
///   LISI = 1 / sum(p_l^2)
///
/// where `p_l` is the proportion of neighbours carrying label `l`. Counting
/// neighbours uniformly makes the score depend on `k`, so treat it as a
/// relative measure across methods at fixed `k` rather than an absolute
/// number. [`lisi_weighted`] is the `k`-insensitive variant.
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per cell.
/// * `labels` - Label per cell, one per row of `knn_indices`.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A [`LisiResult`] with per-cell scores and summaries.
///
/// ### References
///
/// Korsunsky et al., Nat. Methods, 2019
pub fn lisi(
    knn_indices: &[Vec<usize>],
    labels: &[usize],
    verbose: bool,
) -> Result<LisiResult, BixverseErrors> {
    let n = knn_indices.len();
    let (labels, n_labels) = dense_labels(n, labels)?;

    if verbose {
        println!(
            "Running LISI calculations on {} samples.",
            n.separate_with_underscores()
        )
    }

    let counter = Arc::new(AtomicUsize::new(0));

    let per_cell: Vec<f32> = knn_indices
        .par_iter()
        .map(|neighbours| {
            let k = neighbours.len() as f32;
            let mut counts = vec![0u32; n_labels];

            for &j in neighbours {
                counts[labels[j]] += 1;
            }

            let simpson: f32 = counts
                .iter()
                .map(|&c| {
                    let p = c as f32 / k;
                    p * p
                })
                .sum();

            if verbose {
                report_progress(&counter, "LISI calculations", n);
            }

            1.0 / simpson
        })
        .collect();

    let (mean_lisi, median_lisi) = summarise(&per_cell);

    Ok(LisiResult {
        per_cell,
        mean_lisi,
        median_lisi,
        n_labels,
    })
}

/// Shannon entropy of the Gaussian kernel weights at a given precision
///
/// The `Hbeta` step of the t-SNE perplexity calibration that LISI inherits,
/// except that it acts on distances rather than squared distances, following
/// the reference LISI implementation. `weights` is overwritten with the
/// normalised weights.
///
/// ### Params
///
/// * `dists` - Distances to the neighbours of one cell.
/// * `beta` - Kernel precision, i.e. `1 / (2 * sigma^2)`.
/// * `weights` - Scratch buffer, same length as `dists`.
///
/// ### Returns
///
/// The entropy in nats. A neighbourhood whose weights all underflow falls back
/// to uniform weights and the corresponding maximal entropy.
fn hbeta(dists: &[f32], beta: f64, weights: &mut [f64]) -> f64 {
    let mut sum = 0.0;
    for (w, &d) in weights.iter_mut().zip(dists) {
        *w = (-(d as f64) * beta).exp();
        sum += *w;
    }

    if sum <= f64::MIN_POSITIVE {
        let uniform = 1.0 / weights.len() as f64;
        weights.iter_mut().for_each(|w| *w = uniform);
        return (weights.len() as f64).ln();
    }

    let mut weighted_dist = 0.0;
    for (w, &d) in weights.iter_mut().zip(dists) {
        weighted_dist += (d as f64) * *w;
        *w /= sum;
    }

    sum.ln() + beta * weighted_dist / sum
}

/// Calibrate the kernel bandwidth of one neighbourhood to a target perplexity
///
/// Bisection on `beta`, which the entropy is monotone in, with the usual
/// doubling and halving while a bound is still unbounded.
///
/// ### Params
///
/// * `dists` - Distances to the neighbours of one cell.
/// * `log_u` - Target log-perplexity, i.e. the target entropy in nats.
/// * `weights` - Scratch buffer, same length as `dists`. Holds the calibrated
///   normalised weights on return.
fn calibrate_weights(dists: &[f32], log_u: f64, weights: &mut [f64]) {
    let mut beta = 1.0f64;
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::INFINITY;
    let mut entropy = hbeta(dists, beta, weights);

    for _ in 0..LISI_BETA_STEPS {
        if (entropy - log_u).abs() < LISI_ENTROPY_TOL {
            break;
        }
        if entropy > log_u {
            lower = beta;
            beta = if upper.is_infinite() {
                beta * 2.0
            } else {
                (beta + upper) / 2.0
            };
        } else {
            upper = beta;
            beta = if lower.is_infinite() {
                beta / 2.0
            } else {
                (beta + lower) / 2.0
            };
        }
        entropy = hbeta(dists, beta, weights);
    }
}

/// Local Inverse Simpson's Index over kernel-weighted neighbours
///
/// The score of Korsunsky et al. proper: each cell's neighbours are weighted by
/// a Gaussian kernel whose bandwidth is calibrated to `perplexity`, and the
/// Simpson index is taken over the weighted label proportions. Because the
/// bandwidth absorbs the neighbourhood scale, the result barely moves when `k`
/// changes, which is what makes it comparable across datasets.
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per cell.
/// * `knn_dists` - Distances to those neighbours, same shape as
///   `knn_indices`. These must be true distances, see
///   [`crate::single_cell::sc_processing::knn::to_true_distances`].
/// * `labels` - Label per cell, one per row of `knn_indices`.
/// * `perplexity` - Target effective neighbourhood size. Defaults to
///   [`LISI_PERPLEXITY`]. Values above `k` are clamped to `k`, since a
///   neighbourhood cannot be more diffuse than uniform.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A [`LisiResult`] with per-cell scores and summaries.
///
/// ### References
///
/// Korsunsky et al., Nat. Methods, 2019
pub fn lisi_weighted(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    labels: &[usize],
    perplexity: Option<f64>,
    verbose: bool,
) -> Result<LisiResult, BixverseErrors> {
    let n = knn_indices.len();
    if knn_dists.len() != n {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (n, 0),
            got: (knn_dists.len(), 0),
        });
    }
    for (idx, dist) in knn_indices.iter().zip(knn_dists.iter()) {
        if idx.len() != dist.len() {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (n, idx.len()),
                got: (n, dist.len()),
            });
        }
    }

    let (labels, n_labels) = dense_labels(n, labels)?;
    let perplexity = perplexity.unwrap_or(LISI_PERPLEXITY);

    if verbose {
        println!(
            "Running weighted LISI calculations on {} samples.",
            n.separate_with_underscores()
        )
    }

    let counter = Arc::new(AtomicUsize::new(0));

    let per_cell: Vec<f32> = knn_indices
        .par_iter()
        .zip(knn_dists.par_iter())
        .map(|(neighbours, dists)| {
            let k = neighbours.len();
            if k == 0 {
                return 1.0;
            }

            // A perplexity above k has no attainable bandwidth: the entropy
            // saturates at ln(k) and the search runs away towards beta = 0.
            let log_u = perplexity.min(k as f64).ln();
            let mut weights = vec![0.0f64; k];
            calibrate_weights(dists, log_u, &mut weights);

            let mut label_mass = vec![0.0f64; n_labels];
            for (&j, &w) in neighbours.iter().zip(weights.iter()) {
                label_mass[labels[j]] += w;
            }

            let simpson: f64 = label_mass.iter().map(|&p| p * p).sum();

            if verbose {
                report_progress(&counter, "Weighted LISI calculations", n);
            }

            if simpson > 0.0 {
                (1.0 / simpson) as f32
            } else {
                1.0
            }
        })
        .collect();

    let (mean_lisi, median_lisi) = summarise(&per_cell);

    Ok(LisiResult {
        per_cell,
        mean_lisi,
        median_lisi,
        n_labels,
    })
}

////////////////
// Silhouette //
////////////////

/// Results from a silhouette width calculation
#[derive(Clone, Debug)]
pub struct SilhouetteResult {
    /// Per-cell silhouette scores in `[-1, 1]`
    pub per_cell: Vec<f32>,
    /// Mean silhouette width
    pub mean: f32,
    /// Median silhouette width
    pub median: f32,
}

/// Silhouette width of a labelling on an embedding
///
/// For each cell:
///   a = mean distance to cells carrying the same label
///   b = mean distance to cells of the nearest other label
///   s = (b - a) / max(a, b)
///
/// The pass is O(n^2) in the number of cells kept, hence `subsample`.
///
/// ### Params
///
/// * `embedding` - Low-dimensional embedding, cells x dimensions.
/// * `labels` - Label per cell, one per row of `embedding`.
/// * `subsample` - Optional cap on the number of cells used. If `Some` and the
///   embedding is larger, a random subsample is taken.
/// * `seed` - Random seed for the subsampling.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A [`SilhouetteResult`] with per-cell and summary scores. The label check
/// runs on the subsample, so a label lost to subsampling is an error rather
/// than a silently smaller `n_labels`.
pub fn silhouette_width(
    embedding: MatRef<f32>,
    labels: &[usize],
    subsample: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Result<SilhouetteResult, BixverseErrors> {
    let n = embedding.nrows();
    let d = embedding.ncols();
    if labels.len() != n {
        return Err(BixverseErrors::MetricLabelLengthMismatch {
            n_cells: n,
            n_labels: labels.len(),
        });
    }

    let indices: Vec<usize> = match subsample {
        Some(max_n) if n > max_n => {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            idx.truncate(max_n);
            idx.sort_unstable();
            idx
        }
        _ => (0..n).collect(),
    };

    let n_sub = indices.len();
    let sub_labels: Vec<usize> = indices.iter().map(|&i| labels[i]).collect();
    let (sub_labels, n_labels) = dense_labels(n_sub, &sub_labels)?;

    if verbose {
        println!(
            "Running silhouette calculations on {} samples.",
            n_sub.separate_with_underscores()
        )
    }

    // pre-extract rows as contiguous slices for SIMD
    let rows: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| (0..d).map(|j| embedding[(i, j)]).collect())
        .collect();

    let counter = Arc::new(AtomicUsize::new(0));

    let per_cell: Vec<f32> = (0..n_sub)
        .into_par_iter()
        .map(|ii| {
            let l_i = sub_labels[ii];
            let mut label_sum = vec![0.0f32; n_labels];
            let mut label_count = vec![0u32; n_labels];

            for jj in 0..n_sub {
                if ii == jj {
                    continue;
                }
                let dist = euclidean_distance_static(&rows[ii], &rows[jj]).sqrt();
                label_sum[sub_labels[jj]] += dist;
                label_count[sub_labels[jj]] += 1;
            }

            let a = if label_count[l_i] > 0 {
                label_sum[l_i] / label_count[l_i] as f32
            } else {
                0.0
            };

            let mut b = f32::INFINITY;
            for label_idx in 0..n_labels {
                if label_idx == l_i || label_count[label_idx] == 0 {
                    continue;
                }
                let mean_dist = label_sum[label_idx] / label_count[label_idx] as f32;
                if mean_dist < b {
                    b = mean_dist;
                }
            }

            if verbose {
                report_progress(&counter, "Silhouette calculations", n_sub);
            }

            let max_ab = a.max(b);
            if max_ab > 0.0 { (b - a) / max_ab } else { 0.0 }
        })
        .collect();

    let (mean, median) = summarise(&per_cell);

    Ok(SilhouetteResult {
        per_cell,
        mean,
        median,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod shared_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two batches whose neighbourhoods never cross: the effective number of
    /// labels per cell is exactly one.
    #[test]
    fn test_lisi_fully_separated_is_one() {
        let knn = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![4, 5],
            vec![3, 5],
            vec![3, 4],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];

        let res = lisi(&knn, &labels, false).unwrap();

        assert_relative_eq!(res.median_lisi, 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.ilisi_norm(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(res.clisi_norm(), 1.0, epsilon = 1e-6);
    }

    /// Every neighbourhood split evenly across both batches: the score hits
    /// the ceiling of `n_labels`.
    #[test]
    fn test_lisi_perfectly_mixed_hits_n_labels() {
        let knn = vec![vec![1, 2], vec![0, 3], vec![3, 0], vec![2, 1]];
        let labels = vec![0, 1, 0, 1];

        let res = lisi(&knn, &labels, false).unwrap();

        assert_relative_eq!(res.median_lisi, 2.0, epsilon = 1e-6);
        assert_relative_eq!(res.ilisi_norm(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.clisi_norm(), 0.0, epsilon = 1e-6);
    }

    /// Label codes with gaps must not size the count buffers off `max + 1`.
    #[test]
    fn test_lisi_handles_sparse_label_codes() {
        let knn = vec![vec![1, 2], vec![0, 3], vec![3, 0], vec![2, 1]];
        let sparse = vec![17, 4, 17, 4];
        let dense = vec![0, 1, 0, 1];

        let got = lisi(&knn, &sparse, false).unwrap();
        let want = lisi(&knn, &dense, false).unwrap();

        assert_eq!(got.n_labels, 2);
        assert_eq!(got.per_cell, want.per_cell);
    }

    #[test]
    fn test_lisi_single_label_errors() {
        let knn = vec![vec![1], vec![0]];
        assert!(matches!(
            lisi(&knn, &[3, 3], false),
            Err(BixverseErrors::NeedAtLeastTwoLabels { n_labels: 1 })
        ));
    }

    #[test]
    fn test_lisi_label_length_mismatch_errors() {
        let knn = vec![vec![1], vec![0]];
        assert!(matches!(
            lisi(&knn, &[0, 1, 0], false),
            Err(BixverseErrors::MetricLabelLengthMismatch {
                n_cells: 2,
                n_labels: 3
            })
        ));
    }

    /// With all neighbours equidistant the kernel weights are uniform whatever
    /// the bandwidth, so the weighted score must collapse onto the unweighted
    /// one. This is the one input where the two are analytically identical.
    #[test]
    fn test_lisi_weighted_matches_uniform_on_equal_distances() {
        let knn = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];
        let dists = vec![vec![1.0f32; 3]; 4];
        let labels = vec![0, 0, 1, 1];

        let weighted = lisi_weighted(&knn, &dists, &labels, None, false).unwrap();
        let uniform = lisi(&knn, &labels, false).unwrap();

        for (a, b) in weighted.per_cell.iter().zip(uniform.per_cell.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    /// One close same-label neighbour and two far other-label ones: the kernel
    /// discounts the far pair, so the weighted score sits below the count-based
    /// one, which cannot see the distances at all.
    #[test]
    fn test_lisi_weighted_discounts_distant_neighbours() {
        let knn = vec![vec![1, 2, 3], vec![0, 2, 3], vec![3, 0, 1], vec![2, 0, 1]];
        let dists = vec![vec![0.1f32, 10.0, 10.0]; 4];
        let labels = vec![0, 0, 1, 1];

        let weighted = lisi_weighted(&knn, &dists, &labels, Some(2.0), false).unwrap();
        let uniform = lisi(&knn, &labels, false).unwrap();

        assert!(weighted.median_lisi < uniform.median_lisi);
        assert!(weighted.median_lisi >= 1.0);
    }

    #[test]
    fn test_lisi_weighted_rejects_ragged_distances() {
        let knn = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let dists = vec![vec![1.0f32, 1.0], vec![1.0], vec![1.0, 1.0]];
        assert!(matches!(
            lisi_weighted(&knn, &dists, &[0, 1, 0], None, false),
            Err(BixverseErrors::ShapeMismatch { .. })
        ));
    }

    /// Two tight, far-apart clusters: the silhouette saturates near 1.
    #[test]
    fn test_silhouette_separated_labels_near_one() {
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 20.0, 20.0, 20.1, 20.0, 20.0, 20.1,
        ];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 6, 2);
        let labels = vec![0, 0, 0, 1, 1, 1];

        let res = silhouette_width(embd, &labels, None, 42, false).unwrap();

        assert!(res.mean > 0.99, "mean silhouette was {}", res.mean);
    }

    /// Labels alternating along one line: no separation, so the score stays
    /// near zero. It lands slightly negative because interleaving puts the
    /// nearest other-label cell closer than the mean same-label one.
    #[test]
    fn test_silhouette_interleaved_labels_near_zero() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 6, 2);
        let labels = vec![0, 1, 0, 1, 0, 1];

        let res = silhouette_width(embd, &labels, None, 42, false).unwrap();

        assert!(res.mean < 0.0, "mean silhouette was {}", res.mean);
        assert!(res.mean.abs() < 0.25, "mean silhouette was {}", res.mean);
    }

    /// Subsampling must be reproducible for a fixed seed.
    #[test]
    fn test_silhouette_subsample_is_seeded() {
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let embd = MatRef::<f32>::from_row_major_slice(&data, 20, 2);
        let labels: Vec<usize> = (0..20).map(|i| i % 2).collect();

        let a = silhouette_width(embd, &labels, Some(10), 7, false).unwrap();
        let b = silhouette_width(embd, &labels, Some(10), 7, false).unwrap();

        assert_eq!(a.per_cell.len(), 10);
        assert_eq!(a.per_cell, b.per_cell);
    }
}
