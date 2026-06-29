//! Permutation testing for perturbation distances.
//!
//! See Peidli et al., Nat Methods, 2024.

use faer::Accum;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::{Mat, MatRef};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::core::math::matrix_helpers::{stack_rows, subset_rows};
use crate::prelude::*;
use crate::utils::faer_parallelism;

use super::pert_dist::{PertDistance, edistance_two_matrices};

////////////
// Consts //
////////////

/// Reasonable sub sample size
const N_SUB_SAMPLE: usize = 500;

///////////////////
// Enums/Structs //
///////////////////

/// Strategy for permutation testing.
#[derive(Debug, Clone, Copy)]
pub enum PermutationPath {
    /// Materialise pooled N x N distance matrix once. O(N^2) memory; cheap
    /// per-permutation cost.
    Fast,
    /// Recompute distances per permutation via GEMM. O(Nd) memory; expensive
    /// per-permutation cost.
    Slow,
    /// Cap each group at `max_per_group` cells (uniform random subsample),
    /// then run the fast path.
    Subsample {
        /// Maximum number of samples per group
        max_per_group: usize,
    },
}

/// Default implementation for [PermutationPath]
impl Default for PermutationPath {
    fn default() -> Self {
        PermutationPath::Subsample {
            max_per_group: N_SUB_SAMPLE,
        }
    }
}

/// Parse the permutation path to take
///
/// ### Params
///
/// * `s` - String to parse. One of `"fast"`, `"slow"`, or `"subsample"`.
/// * `max_per_group` - Optional usize. To how many max to subsample. If not
///   provided, defaults to [N_SUB_SAMPLE].
///
/// ### Returns
///
/// Option of [PermutationPath]
pub fn parse_perm_path(s: &str, max_per_group: Option<usize>) -> Option<PermutationPath> {
    let max_per_group = max_per_group.unwrap_or(N_SUB_SAMPLE);

    match s.to_lowercase().as_str() {
        "fast" => Some(PermutationPath::Fast),
        "slow" => Some(PermutationPath::Slow),
        "subsample" => Some(PermutationPath::Subsample { max_per_group }),
        _ => None,
    }
}

/// Result of a single permutation test.
#[derive(Debug, Clone)]
pub struct PermutationTestResult<T> {
    /// Observed E-distance on the full (or subsampled) groups.
    pub observed: T,
    /// p-value: count of null distances strictly greater than `observed`,
    /// clipped to at least 1, divided by `n_perms`.
    pub pvalue: T,
}

/////////////
// Helpers //
/////////////

/// Subsample indices
///
/// ### Params
///
/// * `n` - Number of samples
/// * `max` - The maximum number of samples
/// * `rng` - The random number generator
///
/// ### Returns
///
/// The (potentially) subsampled indices
fn subsample_indices(n: usize, max: usize, rng: &mut StdRng) -> Vec<usize> {
    if n <= max {
        (0..n).collect()
    } else {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(rng);
        idx.truncate(max);
        idx.sort_unstable();
        idx
    }
}

/// Compute pooled cell-cell distance matrix (N x N, symmetric, zero diagonal).
///
/// ### Params
///
/// * `pooled` - The pooled matrix
/// * `dist` - The [PertDistance] to calculate
///
/// ### Returns
///
/// The N x N matrix. This can become large in memory
fn pooled_distance_matrix<T: BixverseFloat>(pooled: MatRef<T>, dist: PertDistance) -> Mat<T> {
    let n = pooled.nrows();

    let mut gram = Mat::<T>::zeros(n, n);
    triangular_matmul(
        &mut gram,
        BlockStructure::TriangularLower,
        Accum::Replace,
        pooled,
        BlockStructure::Rectangular,
        pooled.transpose(),
        BlockStructure::Rectangular,
        T::one(),
        faer_parallelism(),
    );

    let norms_sq: Vec<T> = (0..n).map(|i| *gram.get(i, i)).collect();
    let two = T::from_f64(2.0).unwrap();

    let mut flat = vec![T::zero(); n * n];
    flat.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (hi, lo) = if i > j { (i, j) } else { (j, i) };
            let d_sq = (norms_sq[i] + norms_sq[j] - two * *gram.get(hi, lo)).max(T::zero());
            row[j] = match dist {
                PertDistance::Euclidean => d_sq.sqrt(),
                PertDistance::SquaredEuclidean => d_sq,
            };
        }
    });

    Mat::from_fn(n, n, |i, j| flat[i * n + j])
}

/// E-distance from a precomputed pooled N x N distance matrix.
///
/// `mask[i] = true` puts cell i in group X; otherwise Y. Reused across
/// permutations: distances are invariant, only the partition changes.
///
/// ### Params
///
/// * `d` - The N x N distance matrix
/// * `mask` - The boolean mask for the permutations
///
/// ### Returns
///
/// The pairwise edistance given the mask
pub fn edistance_from_pairwise<T: BixverseFloat>(d: MatRef<T>, mask: &[bool]) -> T {
    let n = d.nrows();
    assert_eq!(n, d.ncols());
    assert_eq!(n, mask.len());

    let n_x = mask.iter().filter(|&&b| b).count();
    let n_y = n - n_x;

    // Sums count each unordered pair twice ((i,j) and (j,i)).
    let (sum_xx, sum_yy, sum_xy) = (0..n)
        .into_par_iter()
        .map(|i| {
            let mi = mask[i];
            let mut sxx = T::zero();
            let mut syy = T::zero();
            let mut sxy = T::zero();
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dij = *d.get(i, j);
                match (mi, mask[j]) {
                    (true, true) => sxx += dij,
                    (false, false) => syy += dij,
                    _ => sxy += dij,
                }
            }
            (sxx, syy, sxy)
        })
        .reduce(
            || (T::zero(), T::zero(), T::zero()),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    let two = T::from_f64(2.0).unwrap();
    let n_x_t = T::from_usize(n_x).unwrap();
    let n_y_t = T::from_usize(n_y).unwrap();

    let within_x = if n_x < 2 {
        T::zero()
    } else {
        sum_xx / (n_x_t * (n_x_t - T::one()))
    };
    let within_y = if n_y < 2 {
        T::zero()
    } else {
        sum_yy / (n_y_t * (n_y_t - T::one()))
    };
    let between = if n_x == 0 || n_y == 0 {
        T::zero()
    } else {
        sum_xy / (two * n_x_t * n_y_t)
    };

    two * between - within_x - within_y
}

//////////
// Main //
//////////

/// Permutation test between two row-matrix groups.
///
/// ### Params
///
/// * `x`, `y` - Samples-as-rows matrices for each group.
/// * `dist` - Cell-cell distance variant.
/// * `path` - Memory/compute trade-off, see [`PermutationPath`].
/// * `n_perms` - Number of label permutations to draw.
/// * `seed` - RNG seed.
///
/// ### Results
///
/// The [PermutationTestResult]
pub fn permutation_test_two_groups<T>(
    x: MatRef<T>,
    y: MatRef<T>,
    dist: PertDistance,
    path: PermutationPath,
    n_perms: usize,
    seed: u64,
) -> Result<PermutationTestResult<T>, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    if n_perms == 0usize {
        return Err(BixverseErrors::MustBePositive("n_perms".into()));
    }
    if x.ncols() != y.ncols() {
        return Err(BixverseErrors::NonMatchingFeatureDim {
            dim_x: x.ncols(),
            dim_y: y.ncols(),
        });
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // subsample collapses to fast path after sampling.
    let (x_owned, y_owned, use_fast) = match path {
        PermutationPath::Subsample { max_per_group } => {
            let idx_x = subsample_indices(x.nrows(), max_per_group, &mut rng);
            let idx_y = subsample_indices(y.nrows(), max_per_group, &mut rng);
            (
                Some(subset_rows(x, &idx_x)),
                Some(subset_rows(y, &idx_y)),
                true,
            )
        }
        PermutationPath::Fast => (None, None, true),
        PermutationPath::Slow => (None, None, false),
    };

    let x_ref = x_owned.as_ref().map(|m| m.as_ref()).unwrap_or(x);
    let y_ref = y_owned.as_ref().map(|m| m.as_ref()).unwrap_or(y);

    let n_x = x_ref.nrows();
    let n_total = n_x + y_ref.nrows();

    let observed = edistance_two_matrices(x_ref, y_ref, dist)?;

    let mut initial_mask = vec![false; n_total];
    for m in initial_mask.iter_mut().take(n_x) {
        *m = true;
    }

    // masks generated sequentially (StdRng !Sync), then consumed in parallel.
    let masks: Vec<Vec<bool>> = (0..n_perms)
        .map(|_| {
            let mut m = initial_mask.clone();
            m.shuffle(&mut rng);
            m
        })
        .collect();

    let pooled = stack_rows(x_ref, y_ref);

    let null_dist: Vec<T> = if use_fast {
        let d_mat = pooled_distance_matrix(pooled.as_ref(), dist);
        masks
            .par_iter()
            .map(|m| edistance_from_pairwise(d_mat.as_ref(), m))
            .collect()
    } else {
        masks
            .par_iter()
            .map(|m| {
                let xs: Vec<usize> = m
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| if b { Some(i) } else { None })
                    .collect();
                let ys: Vec<usize> = m
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| if !b { Some(i) } else { None })
                    .collect();
                let x_p = subset_rows(pooled.as_ref(), &xs);
                let y_p = subset_rows(pooled.as_ref(), &ys);
                let res = edistance_two_matrices(x_p.as_ref(), y_p.as_ref(), dist)?;

                Ok(res)
            })
            .collect::<Result<Vec<_>, BixverseErrors>>()?
    };

    let count = null_dist.iter().filter(|&&d| d > observed).count().max(1);
    let pvalue = T::from_usize(count).unwrap() / T::from_usize(n_perms).unwrap();

    Ok(PermutationTestResult { observed, pvalue })
}

/// One-sided permutation tests: each non-reference group against the reference.
///
/// The reference group's entry has `observed = 0` and `pvalue = 1` (axiom).
///
/// ### Params
///
/// * `embedding` - The embedding matrix of shape samples x features
/// * `labels` - The label vector
/// * `reference` - The reference id
/// * `dist` - The [PertDistance] to use
/// * `path` - The [PermutationPath] to use
/// * `n_perms` - The number of permutations to run for
/// * `seed` - The random seed
///
/// ### Returns
///
/// A Vec of [PermutationTestResult]
#[allow(clippy::too_many_arguments)]
pub fn permutation_test_onesided<T>(
    embedding: MatRef<T>,
    labels: &[usize],
    reference: usize,
    dist: PertDistance,
    path: PermutationPath,
    n_perms: usize,
    seed: u64,
) -> Result<Vec<PermutationTestResult<T>>, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    if embedding.nrows() != labels.len() {
        return Err(BixverseErrors::NumberLabelsNotEqualSampleNumber {
            label_length: labels.len(),
            n_samples: embedding.nrows(),
        });
    }
    let n_groups = labels.iter().max().map(|&x| x + 1).unwrap_or(0);
    if reference >= n_groups {
        return Err(BixverseErrors::ReferenceOutOfRange);
    }

    let d = embedding.ncols();
    let mut indices: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (i, &g) in labels.iter().enumerate() {
        indices[g].push(i);
    }
    let groups: Vec<Mat<T>> = indices
        .into_iter()
        .map(|rows| Mat::from_fn(rows.len(), d, |i, j| *embedding.get(rows[i], j)))
        .collect();

    let res = (0..n_groups)
        .map(|g| {
            if g == reference {
                Ok(PermutationTestResult {
                    observed: T::zero(),
                    pvalue: T::one(),
                })
            } else {
                let res = permutation_test_two_groups(
                    groups[g].as_ref(),
                    groups[reference].as_ref(),
                    dist,
                    path,
                    n_perms,
                    seed.wrapping_add(g as u64),
                )?;

                Ok(res)
            }
        })
        .collect::<Result<Vec<_>, BixverseErrors>>()?;

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::base::pert_dist::edistance_two_matrices;

    const EPS: f64 = 1e-10;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn parse_path_variants() {
        assert!(matches!(
            parse_perm_path("fast", Some(100)),
            Some(PermutationPath::Fast)
        ));
        assert!(matches!(
            parse_perm_path("SLOW", Some(100)),
            Some(PermutationPath::Slow)
        ));
        match parse_perm_path("subsample", Some(250)) {
            Some(PermutationPath::Subsample { max_per_group }) => assert_eq!(max_per_group, 250),
            _ => panic!("expected subsample variant"),
        }
        assert!(parse_perm_path("nope", Some(0)).is_none());
    }

    #[test]
    fn subsample_keeps_all_when_le_max() {
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(subsample_indices(5, 10, &mut rng), vec![0, 1, 2, 3, 4]);
        assert_eq!(subsample_indices(5, 5, &mut rng), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn subsample_truncates_sorted_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let idx = subsample_indices(100, 10, &mut rng);
        assert_eq!(idx.len(), 10);
        for w in idx.windows(2) {
            assert!(w[0] < w[1]);
        }
        assert!(*idx.last().unwrap() < 100);
    }

    #[test]
    fn edist_from_pairwise_matches_direct() {
        let x = Mat::<f64>::from_fn(4, 3, |i, j| (i as f64 - 1.0) * (j as f64 + 1.0));
        let y = Mat::<f64>::from_fn(3, 3, |i, j| (i as f64 + 2.0) * (j as f64 - 0.5));
        let pooled = stack_rows(x.as_ref(), y.as_ref());
        let d_mat = pooled_distance_matrix(pooled.as_ref(), PertDistance::Euclidean);

        let mut mask = vec![false; 7];
        for m in mask.iter_mut().take(4) {
            *m = true;
        }
        let e_mask = edistance_from_pairwise(d_mat.as_ref(), &mask);
        let e_direct =
            edistance_two_matrices(x.as_ref(), y.as_ref(), PertDistance::Euclidean).unwrap();
        assert!(close(e_mask, e_direct));
    }

    #[test]
    fn fast_and_slow_agree() {
        // Same seed -> same masks -> two paths must agree up to fp precision.
        let x = Mat::<f64>::from_fn(6, 4, |i, j| (i * 3 + j) as f64);
        let y = Mat::<f64>::from_fn(7, 4, |i, j| (i + j * 2) as f64 + 0.5);
        let fast = permutation_test_two_groups(
            x.as_ref(),
            y.as_ref(),
            PertDistance::Euclidean,
            PermutationPath::Fast,
            32,
            7,
        )
        .unwrap();
        let slow = permutation_test_two_groups(
            x.as_ref(),
            y.as_ref(),
            PertDistance::Euclidean,
            PermutationPath::Slow,
            32,
            7,
        )
        .unwrap();
        assert!(close(fast.observed, slow.observed));
        assert!(close(fast.pvalue, slow.pvalue));
    }

    #[test]
    fn observed_matches_edistance() {
        let x = Mat::<f64>::from_fn(4, 3, |i, j| (i + 2 * j) as f64);
        let y = Mat::<f64>::from_fn(4, 3, |i, j| (i + 2 * j) as f64 + 1.0);
        let direct =
            edistance_two_matrices(x.as_ref(), y.as_ref(), PertDistance::Euclidean).unwrap();
        let perm = permutation_test_two_groups(
            x.as_ref(),
            y.as_ref(),
            PertDistance::Euclidean,
            PermutationPath::Fast,
            5,
            0,
        )
        .unwrap();
        assert!(close(direct, perm.observed));
    }

    #[test]
    fn zero_perms_errors() {
        let x = Mat::<f64>::zeros(2, 2);
        let y = Mat::<f64>::zeros(2, 2);
        assert!(matches!(
            permutation_test_two_groups(
                x.as_ref(),
                y.as_ref(),
                PertDistance::Euclidean,
                PermutationPath::Fast,
                0,
                0,
            ),
            Err(BixverseErrors::MustBePositive(_))
        ));
    }

    #[test]
    fn dim_mismatch_errors() {
        let x = Mat::<f64>::zeros(3, 2);
        let y = Mat::<f64>::zeros(3, 3);
        assert!(matches!(
            permutation_test_two_groups(
                x.as_ref(),
                y.as_ref(),
                PertDistance::Euclidean,
                PermutationPath::Fast,
                10,
                0,
            ),
            Err(BixverseErrors::NonMatchingFeatureDim { .. })
        ));
    }

    #[test]
    fn pvalue_in_range() {
        let x = Mat::<f64>::from_fn(4, 2, |i, j| (i + j) as f64);
        let y = Mat::<f64>::from_fn(4, 2, |i, j| (i + j) as f64 + 10.0);
        let r = permutation_test_two_groups(
            x.as_ref(),
            y.as_ref(),
            PertDistance::Euclidean,
            PermutationPath::Fast,
            50,
            0,
        )
        .unwrap();
        assert!(r.pvalue >= 1.0 / 50.0 - 1e-12);
        assert!(r.pvalue <= 1.0 + 1e-12);
    }

    #[test]
    fn onesided_reference_axiom() {
        let emb = Mat::<f64>::from_fn(9, 2, |i, j| ((i + 1) * (j + 1)) as f64);
        let labels = vec![0_usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let res = permutation_test_onesided(
            emb.as_ref(),
            &labels,
            1,
            PertDistance::Euclidean,
            PermutationPath::Fast,
            16,
            0,
        )
        .unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(res[1].observed, 0.0);
        assert_eq!(res[1].pvalue, 1.0);
    }
}
