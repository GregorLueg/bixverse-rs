//! K-means clustering methods. Heavy re-use of highly optimised
//! methods implemented in [`ann-search-rs`](https://crates.io/crates/ann-search-rs).

use ann_search_rs::prelude::*;
use ann_search_rs::utils::{k_means_utils::*, matrix_to_flat};
use faer::{Mat, MatRef};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::fmt::Debug;

use crate::prelude::*;

use ann_search_rs::prelude::KMeansTrainingParams;

////////////////////
// Enums / Params //
////////////////////

#[derive(Clone, Debug, Copy, Default)]
/// Type of K means clustering to run
pub enum KMeansType {
    /// Standard version - will use SIMD acceleration and if dimensionality is
    /// big enough, GEMM assignment.
    #[default]
    StandardKMeans,
    /// Mini-batch version of k-means based on Sculley, 2010
    MiniBatchKMeans,
}

/// Function to parse the type of k-means to use
///
/// ### Params
///
/// * `s` - String to parse
///
/// ### Returns
///
/// Enum of the k-means clustering to use
pub fn parse_k_means(s: &str) -> Option<KMeansType> {
    match s.to_lowercase().as_str() {
        "standard" | "kmeans" | "k-means" => Some(KMeansType::StandardKMeans),
        "minibatch" | "mini-batch" => Some(KMeansType::MiniBatchKMeans),
        _ => None,
    }
}

/// Wrapper around the [KMeansTrainingParams]
///
/// ### Fields
///
/// `0` - The [KMeansTrainingParams].
#[derive(Clone, Copy)]
pub struct KMeansParamsWrappers(KMeansTrainingParams);

impl KMeansParamsWrappers {
    /// Generate a wrapped instance around the k-means clustering
    ///
    /// ### Params
    ///
    /// * `iters` - Number of iterations to run the clustering algorithm for
    /// * `init` - Optional specification of the [KMeansInit] for better
    ///   control
    /// * `path` - The Lloyd's path you want to use, see [LloydPath] for better
    ///   control
    ///
    /// ### Returns
    ///
    /// [KMeansTrainingParams]
    pub fn new(iters: usize, init: Option<KMeansInit>, path: Option<LloydPath>) -> Self {
        let internal_params = KMeansTrainingParams::new(iters, init, path);

        Self(internal_params)
    }

    /// Returns the parameters and consumes self
    ///
    /// ### Returns
    ///
    /// The [KMeansTrainingParams]
    pub fn get_data(self) -> KMeansTrainingParams {
        self.0
    }
}

/// Default implementation for [KMeansParamsWrappers]
impl Default for KMeansParamsWrappers {
    fn default() -> Self {
        Self::new(100, None, None)
    }
}

/// Parse the initialisation parameters
///
/// ### Params
///
/// * `s` String to parse
///
/// ### Returns
///
/// The Option of the [KMeansInit]
pub fn parse_kmeans_init(s: &str) -> Option<KMeansInit> {
    match s.to_lowercase().as_str() {
        "random" => Some(KMeansInit::Random),
        "parallel" | "plusplus" => Some(KMeansInit::KMeansParallel),
        _ => None,
    }
}

/// Parse the method details
///
/// ### Params
///
/// * `gemm` - Shall GEMM acceleration be used. This option will use faer under
///   the hood and can accelerate the calculations in high dimensional data.
/// * `hamerly` - Shall Hamerly's algorithm be used. Can provide strong
///   acceleration with large N and/or large number of centroids.
///
/// ### Returns
///
/// The [LloydPath]
pub fn parse_kmean_path(gemm: bool, hamerly: bool) -> LloydPath {
    match (gemm, hamerly) {
        (true, false) => LloydPath::GemmLloyd,
        (true, true) => LloydPath::HamerlyGemm,
        (false, true) => LloydPath::HamerlySimd,
        (false, false) => LloydPath::ParallelLloyd,
    }
}

/// Debug implementation
impl std::fmt::Debug for KMeansParamsWrappers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KMeansParamsWrapper")
            .field("iters", &self.0.iters)
            .field("init", &self.0.init)
            .field("path", &self.0.path)
            .field("balanced", &self.0.balanced)
            .finish()
    }
}

/////////////
// Helpers //
/////////////

/// Recompute centroid norms for the assignment path
///
/// Computes the norm representation expected by `gemm_assign_full` and
/// `direct_assign`: squared L2 norm (`||c||^2`) for Euclidean, or L2
/// norm (`||c||`) for Cosine.
///
/// ### Params
///
/// * `centroids` - All centroids, flattened row-major (`k * dim` elements)
/// * `dim` - Embedding dimensionality
/// * `k` - Number of centroids
/// * `metric` - Distance metric
///
/// ### Returns
///
/// `Vec<T>` of length `k` containing the per-centroid norm values.
pub fn recompute_centroid_norms<T>(centroids: &[T], dim: usize, k: usize, metric: &Dist) -> Vec<T>
where
    T: BixverseFloat + SimdDistance,
{
    (0..k)
        .map(|c| {
            let cent = &centroids[c * dim..(c + 1) * dim];
            match metric {
                Dist::SquaredEuclidean => T::dot_simd(cent, cent),
                Dist::Cosine => T::calculate_l2_norm(cent),
                Dist::Manhattan => unreachable!(),
            }
        })
        .collect()
}

/// Compute per-vector L2 norms, or return an empty vec for Euclidean
///
/// For Cosine distance, computes `||x||` for each vector (needed by the
/// centroid distance and pre-normalised assignment paths). For Euclidean,
/// returns an empty `Vec` since the assignment path computes squared
/// norms internally.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major (`n * dim` elements)
/// * `dim` - Embedding dimensionality
/// * `n` - Number of vectors
/// * `metric` - Distance metric
///
/// ### Returns
///
/// `Vec<T>` of length `n` (Cosine) or length `0` (Euclidean).
pub fn compute_data_norms<T>(data: &[T], dim: usize, n: usize, metric: &Dist) -> Vec<T>
where
    T: BixverseFloat + SimdDistance,
{
    match metric {
        Dist::Cosine => (0..n)
            .into_par_iter()
            .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
            .collect(),
        Dist::SquaredEuclidean => Vec::new(),
        Dist::Manhattan => unreachable!(),
    }
}

////////////////////////
// Mini batch k-means //
////////////////////////

/// Mini-batch k-means (Sculley 2010)
///
/// Trains centroids using random mini-batches with a decaying learning
/// rate (`eta = 1 / count[c]`). Each iteration samples `batch_size` vectors
/// without replacement, assigns them to the nearest centroid, and applies an
/// incremental mean update. A full assignment pass seeds the per-centroid
/// counts after initialisation to stabilise early updates. Convergence is
/// checked via maximum centroid drift.
///
/// After training, a final full-dataset assignment pass produces the returned
/// assignments.
///
/// ### Params
///
/// * `data` - Training vectors, flattened row-major (`n * dim` elements)
/// * `dist` - The distance metric to use. One of `"euclidean"` or `"cosine"`.
///   Weird strings will default to `"euclidean"` (under the hood squared
///   euclidean)
/// * `max_iters` - Maximum number of mini-batch iterations
/// * `batch_size` - Number of vectors sampled per iteration. Clamped to
///   `n / 2` if larger.
/// * `drift_threshold` - Below which centroid drift the algorithm is seen as
///   converged.
/// * `lr_alpha` - Learning rate alpha decay. The original paper used `1.0`.
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print convergence diagnostics
///
/// ### Returns
///
/// Tuple of (centroids, assignments)
#[allow(clippy::too_many_arguments)]
pub fn train_centroids_minibatch<T>(
    data: MatRef<T>,
    dist: &str,
    n_centroids: usize,
    max_iters: usize,
    batch_size: usize,
    drift_threshold: T,
    lr_alpha: T,
    seed: usize,
    verbose: bool,
) -> (Mat<T>, Vec<usize>)
where
    T: BixverseFloat + SimdDistance,
{
    let (data, n, dim) = matrix_to_flat(data);
    let dist = parse_ann_dist(dist).unwrap_or_default();

    let batch_size = batch_size.min(n);

    // precompute all data norms once
    let data_norms: Vec<T> = match dist {
        Dist::SquaredEuclidean => (0..n)
            .into_par_iter()
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                T::dot_simd(v, v)
            })
            .collect(),
        Dist::Cosine => (0..n)
            .into_par_iter()
            .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
            .collect(),
        Dist::Manhattan => unreachable!(),
    };

    // initialise centroids
    let mut centroids = if n_centroids > 200 {
        if verbose {
            println!("  Initialising centroids via fast random selection");
        }
        fast_random_init(&data, dim, n, n_centroids, seed)
    } else {
        if verbose {
            println!("  Initialising centroids via k-means||");
        }
        let init_norms: Vec<T> = match dist {
            Dist::SquaredEuclidean => (0..n)
                .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
                .collect(),
            Dist::Cosine => data_norms.clone(),
            Dist::Manhattan => unreachable!(),
        };
        kmeans_parallel_init(&data, &init_norms, dim, n, n_centroids, &dist, seed)
    };

    let mut centroid_norms = recompute_centroid_norms(&centroids, dim, n_centroids, &dist);

    // seed counts cheaply -> no full assignment pass
    let init_count = (batch_size / n_centroids).max(1);
    let mut counts = vec![init_count; n_centroids];

    let mut old_centroids = vec![T::zero(); n_centroids * dim];

    // pre-shuffle for epoch-based iteration: avoids per-iteration RNG + gather
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut shuffled_indices: Vec<usize> = (0..n).collect();
    shuffled_indices.shuffle(&mut rng);
    let mut epoch_offset = 0usize;

    // scratch buffers
    let mut batch_data = Vec::with_capacity(batch_size * dim);
    let mut batch_norms = Vec::with_capacity(batch_size);

    let num_threads = rayon::current_num_threads();

    if verbose {
        println!(
            "  Running mini-batch iterations (batch_size={}, lr_alpha={:.2})",
            batch_size, lr_alpha
        );
    }

    for iter in 0..max_iters {
        // epoch-based sampling: take next chunk, re-shuffle on wrap
        if epoch_offset + batch_size > n {
            shuffled_indices.shuffle(&mut rng);
            epoch_offset = 0;
        }
        let batch_indices = &shuffled_indices[epoch_offset..epoch_offset + batch_size];
        epoch_offset += batch_size;

        // gather batch
        batch_data.clear();
        batch_norms.clear();
        for &idx in batch_indices {
            batch_data.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
            batch_norms.push(data_norms[idx]);
        }

        // assign batch
        let batch_assignments = assign_all_parallel(
            &batch_data,
            &batch_norms,
            dim,
            batch_size,
            &centroids,
            &centroid_norms,
            n_centroids,
            &dist,
        );

        // parallel fold-reduce -> per-cluster sums and counts
        let chunk_size = (batch_size + num_threads - 1) / num_threads.max(1);
        let (batch_sums, batch_counts) = batch_assignments
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, assign_chunk)| {
                let mut local_sums = vec![T::zero(); n_centroids * dim];
                let mut local_counts = vec![0usize; n_centroids];
                let start = chunk_idx * chunk_size;
                let data_chunk = &batch_data[start * dim..(start + assign_chunk.len()) * dim];

                for (i, &c) in assign_chunk.iter().enumerate() {
                    local_counts[c] += 1;
                    let vec = &data_chunk[i * dim..(i + 1) * dim];
                    let offset = c * dim;
                    T::add_assign_simd(&mut local_sums[offset..offset + dim], vec);
                }
                (local_sums, local_counts)
            })
            .reduce(
                || {
                    (
                        vec![T::zero(); n_centroids * dim],
                        vec![0usize; n_centroids],
                    )
                },
                |(mut s1, mut c1), (s2, c2)| {
                    T::add_assign_simd(&mut s1, &s2);
                    for i in 0..c1.len() {
                        c1[i] += c2[i];
                    }
                    (s1, c1)
                },
            );

        // collect touched centroids
        let touched: Vec<usize> = (0..n_centroids).filter(|&c| batch_counts[c] > 0).collect();

        // save old centroids (only need them for drift, full copy is cheap)
        old_centroids.copy_from_slice(&centroids);

        // apply incremental updates to touched centroids only
        for &c in &touched {
            let m = batch_counts[c];
            counts[c] += m;
            let count_f = T::from(counts[c]).unwrap();
            let eta = T::from(m).unwrap() / count_f.powf(lr_alpha);
            // clamp eta to [0, 1] powf with alpha < 1 can push it above 1
            let eta = if eta > T::one() { T::one() } else { eta };
            let one_minus_eta = T::one() - eta;
            let inv_m = T::one() / T::from(m).unwrap();
            let offset = c * dim;
            for d in 0..dim {
                let batch_mean = batch_sums[offset + d] * inv_m;
                centroids[offset + d] = centroids[offset + d] * one_minus_eta + batch_mean * eta;
            }
        }

        // recompute norms only for touched centroids
        for &c in &touched {
            let cent = &centroids[c * dim..(c + 1) * dim];
            centroid_norms[c] = match dist {
                Dist::SquaredEuclidean => T::dot_simd(cent, cent),
                Dist::Cosine => T::calculate_l2_norm(cent),
                Dist::Manhattan => unreachable!(),
            };
        }

        // drift check only on touched centroids
        let max_drift = touched
            .iter()
            .map(|&c| {
                let old = &old_centroids[c * dim..(c + 1) * dim];
                let new_c = &centroids[c * dim..(c + 1) * dim];
                euclidean_distance_static(old, new_c)
            })
            .fold(T::zero(), T::max);

        if max_drift <= drift_threshold {
            if verbose {
                println!("    Converged at iteration {}", iter + 1);
            }
            break;
        }

        if verbose && (iter + 1) % 50 == 0 {
            println!(
                "    Iteration {}, max centroid drift: {:.6}, touched: {}/{}",
                iter + 1,
                max_drift.to_f64().unwrap(),
                touched.len(),
                n_centroids,
            );
        }
    }

    // final full assignment
    let assignments = assign_all_parallel(
        &data,
        &data_norms,
        dim,
        n,
        &centroids,
        &centroid_norms,
        n_centroids,
        &dist,
    );

    let centroid_mat = Mat::from_fn(n_centroids, dim, |i, j| centroids[i * dim + j]);

    (centroid_mat, assignments)
}

/////////////
// k-means //
/////////////

/// Generate k-means clusters for subsequent kNN search
///
/// ### Params
///
/// * `data` - The data for which to run the k-means clustering
/// * `dist` - The distance metric to use. One of `"euclidean"` or `"cosine"`.
///   Weird strings will default to `"euclidean"` (under the hood squared
///   euclidean)
/// * `n_centroids` - The number of centroids to identify
/// * `n_iters` - Number of iterations for the k-means clustering
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls the verbosity of the function
///
/// ### Returns
///
/// A tuple of `(centroid matrix -> shape n_centoids x dim, assignments of
/// points to cells)`
pub fn k_means_clusters<T>(
    data: MatRef<T>,
    dist: &str,
    n_centroids: usize,
    kmeans_params: Option<KMeansParamsWrappers>,
    seed: usize,
    verbose: bool,
) -> Result<(Mat<T>, Vec<usize>), AnnSearchErrors>
where
    T: BixverseFloat + AnnSearchFloat,
{
    let kmeans_params = kmeans_params.unwrap_or_default();
    let (data_flat, n, dim) = matrix_to_flat(data);
    let dist = parse_ann_dist(dist).unwrap_or_else(|| {
        println!(
            "Unknown string provided ({:?}). Defaulting to Squared Euclidean",
            dist
        );
        Dist::default()
    });

    let norms = if dist == Dist::Cosine {
        (0..n)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                T::calculate_l2_norm(&data_flat[start..end])
            })
            .collect()
    } else {
        Vec::new()
    };

    let centroids = train_centroids(
        &data_flat,
        dim,
        n,
        n_centroids,
        &dist,
        Some(kmeans_params.0),
        seed,
        verbose,
    )?;

    let centroids_norm = if dist == Dist::Cosine {
        (0..n_centroids)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                T::calculate_l2_norm(&centroids[start..end])
            })
            .collect()
    } else {
        Vec::new()
    };

    let assignments = assign_all_parallel(
        &data_flat,
        &norms,
        dim,
        n,
        &centroids,
        &centroids_norm,
        n_centroids,
        &dist,
    );

    let centroid_mat = Mat::from_fn(n_centroids, dim, |i, j| centroids[i * dim + j]);

    Ok((centroid_mat, assignments))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::prelude::*;
    use rand::rngs::StdRng;

    fn make_two_blobs(n_per: usize, dim: usize, separation: f32, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut buf = vec![0.0f32; 2 * n_per * dim];
        for i in 0..(2 * n_per) {
            let centre = if i < n_per { 0.0 } else { separation };
            for j in 0..dim {
                buf[i * dim + j] = centre + rng.random_range(-0.3..0.3);
            }
        }
        Mat::from_fn(2 * n_per, dim, |i, j| buf[i * dim + j])
    }

    /// Each row of the returned centroid Mat must have ALL dimensions near the
    /// same blob centre. A row/col scramble in the flat -> Mat conversion would
    /// mix values across blobs within a single row, so per-dim checks catch it.
    fn assert_centroid_rows_pure(centroids: &Mat<f32>, separation: f32, tol: f32) {
        for r in 0..centroids.nrows() {
            let target = if (centroids[(r, 0)] - 0.0).abs() < separation / 2.0 {
                0.0
            } else {
                separation
            };
            for c in 0..centroids.ncols() {
                let v = centroids[(r, c)];
                assert!(
                    (v - target).abs() < tol,
                    "centroid[{}][{}] = {}, expected near {}",
                    r,
                    c,
                    v,
                    target
                );
            }
        }
    }

    fn assert_assignments_split_blobs(assignments: &[usize], n_per: usize) {
        let blob_a = &assignments[..n_per];
        let blob_b = &assignments[n_per..];
        let pure_a = blob_a.iter().filter(|&&c| c == blob_a[0]).count();
        let pure_b = blob_b.iter().filter(|&&c| c == blob_b[0]).count();
        assert!(
            pure_a as f32 / n_per as f32 > 0.95,
            "blob A purity {}",
            pure_a
        );
        assert!(
            pure_b as f32 / n_per as f32 > 0.95,
            "blob B purity {}",
            pure_b
        );
        assert_ne!(blob_a[0], blob_b[0]);
    }

    /// Every accepted spelling and casing maps to its variant; anything else is None.
    #[test]
    fn parse_k_means_known_strings() {
        assert!(matches!(
            parse_k_means("standard"),
            Some(KMeansType::StandardKMeans)
        ));
        assert!(matches!(
            parse_k_means("k-means"),
            Some(KMeansType::StandardKMeans)
        ));
        assert!(matches!(
            parse_k_means("STANDARD"),
            Some(KMeansType::StandardKMeans)
        ));
        assert!(matches!(
            parse_k_means("minibatch"),
            Some(KMeansType::MiniBatchKMeans)
        ));
        assert!(matches!(
            parse_k_means("mini-batch"),
            Some(KMeansType::MiniBatchKMeans)
        ));
        assert!(parse_k_means("nonsense").is_none());
    }

    /// Two separated blobs give one centroid row each, in row-per-centroid layout.
    #[test]
    fn standard_centroid_rows_match_blob_centres() {
        let n_per = 200;
        let dim = 5;
        let separation = 5.0f32;
        let data = make_two_blobs(n_per, dim, separation, 42);

        let (centroids, assignments) =
            k_means_clusters(data.as_ref(), "euclidean", 2, None, 0, false).unwrap();

        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), dim);
        assert_eq!(assignments.len(), 2 * n_per);

        assert_centroid_rows_pure(&centroids, separation, 0.5);
        assert_assignments_split_blobs(&assignments, n_per);
    }

    /// The same, for the mini-batch path, which has its own update and layout code.
    #[test]
    fn minibatch_centroid_rows_match_blob_centres() {
        let n_per = 200;
        let dim = 5;
        let separation = 5.0f32;
        let data = make_two_blobs(n_per, dim, separation, 42);

        let (centroids, assignments) =
            train_centroids_minibatch(data.as_ref(), "euclidean", 2, 200, 64, 1e-4, 0.75, 0, false);

        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), dim);
        assert_eq!(assignments.len(), 2 * n_per);

        assert_centroid_rows_pure(&centroids, separation, 0.7);
        assert_assignments_split_blobs(&assignments, n_per);
    }

    /// Ties the centroid rows back to the assignments rather than the blob geometry.
    #[test]
    fn standard_centroid_row_is_mean_of_assigned_points() {
        // Independent check that the Mat row r really is the mean of points
        // assigned to centroid r. Catches both layout scrambles and silent
        // off-by-one errors.
        let n_per = 200;
        let dim = 5;
        let data = make_two_blobs(n_per, dim, 5.0, 42);

        let (centroids, assignments) =
            k_means_clusters(data.as_ref(), "euclidean", 2, None, 0, false).unwrap();

        for r in 0..centroids.nrows() {
            let mut sum = vec![0.0f32; dim];
            let mut count = 0usize;
            for i in 0..(2 * n_per) {
                if assignments[i] == r {
                    for d in 0..dim {
                        sum[d] += data[(i, d)];
                    }
                    count += 1;
                }
            }
            assert!(count > 0, "centroid {} has no assignments", r);
            for d in 0..dim {
                let mean = sum[d] / count as f32;
                let cent = centroids[(r, d)];
                assert!(
                    (mean - cent).abs() < 0.05,
                    "centroid[{}][{}] = {}, mean of assigned = {}",
                    r,
                    d,
                    cent,
                    mean
                );
            }
        }
    }

    /// The seed must fix the initialisation, so two runs agree to the last centroid.
    #[test]
    fn standard_deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);

        let (c1, a1) = k_means_clusters(data.as_ref(), "euclidean", 5, None, 7, false).unwrap();
        let (c2, a2) = k_means_clusters(data.as_ref(), "euclidean", 5, None, 7, false).unwrap();

        assert_eq!(a1, a2);
        for i in 0..c1.nrows() {
            for j in 0..c1.ncols() {
                assert!((c1[(i, j)] - c2[(i, j)]).abs() < 1e-6);
            }
        }
    }

    /// The seed must also fix the batch sampler and its shuffles, not just the init.
    #[test]
    fn minibatch_deterministic_with_same_seed() {
        let data = make_two_blobs(100, 5, 5.0, 42);

        let (c1, a1) =
            train_centroids_minibatch(data.as_ref(), "euclidean", 5, 200, 64, 1e-4, 0.75, 7, false);
        let (c2, a2) =
            train_centroids_minibatch(data.as_ref(), "euclidean", 5, 200, 64, 1e-4, 0.75, 7, false);

        assert_eq!(a1, a2);
        for i in 0..c1.nrows() {
            for j in 0..c1.ncols() {
                assert!((c1[(i, j)] - c2[(i, j)]).abs() < 1e-6);
            }
        }
    }

    /// Both variants must honour the metric rather than falling back to Euclidean.
    #[test]
    fn cosine_metric_groups_by_direction_not_magnitude() {
        // Two rays with very different magnitudes along each. Cosine must
        // group by direction, which also exercises the branch where centroid
        // norms are ||c|| rather than ||c||^2.
        let dirs = [[1.0f32, 0.2, 0.0, 0.0, 0.0], [0.2, 1.0, 0.0, 0.0, 0.0]];
        let data = Mat::from_fn(200, 5, |i, j| {
            let ray = if i < 100 { 0 } else { 1 };
            let magnitude = 1.0 + (i % 100) as f32;
            dirs[ray][j] * magnitude
        });

        let (c, a) = k_means_clusters(data.as_ref(), "cosine", 2, None, 0, false).unwrap();
        assert_eq!(c.nrows(), 2);
        assert_assignments_split_blobs(&a, 100);

        let (c, a) =
            train_centroids_minibatch(data.as_ref(), "cosine", 2, 200, 64, 1e-4, 0.75, 0, false);
        assert_eq!(c.nrows(), 2);
        assert_assignments_split_blobs(&a, 100);
    }

    /// The Euclidean assignment path expects `||c||^2`, not the plain L2 norm.
    #[test]
    fn recompute_centroid_norms_euclidean_is_squared() {
        // Three centroids of dim 4: norms should be sum of squares for euclidean.
        let centroids: Vec<f32> = vec![
            1.0, 2.0, 2.0, 0.0, // ||.||^2 = 9
            3.0, 0.0, 4.0, 0.0, // ||.||^2 = 25
            1.0, 1.0, 1.0, 1.0, // ||.||^2 = 4
        ];
        let norms = recompute_centroid_norms(&centroids, 4, 3, &Dist::SquaredEuclidean);
        assert!((norms[0] - 9.0).abs() < 1e-5);
        assert!((norms[1] - 25.0).abs() < 1e-5);
        assert!((norms[2] - 4.0).abs() < 1e-5);
    }

    /// The cosine path wants the plain L2 norm, the other half of that contract.
    #[test]
    fn recompute_centroid_norms_cosine_is_l2() {
        let centroids: Vec<f32> = vec![
            3.0, 4.0, 0.0, 0.0, // ||.|| = 5
            1.0, 1.0, 1.0, 1.0, // ||.|| = 2
        ];
        let norms = recompute_centroid_norms(&centroids, 4, 2, &Dist::Cosine);
        assert!((norms[0] - 5.0).abs() < 1e-5);
        assert!((norms[1] - 2.0).abs() < 1e-5);
    }
}
