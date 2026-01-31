use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::core::base::cors_similarity::*;
use crate::core::math::matrix_helpers::*;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Create an affinity matrix from a distance matrix
///
/// Applies a scaled exponential similarity kernel based on K-nearest neighbors.
/// The kernel uses an adaptive sigma parameter calculated from the average
/// distance to K nearest neighbors. Sounds fancy, doesn't it... ?
///
/// ### Params
///
/// * `dist` - The distance matrix.
/// * `k` - Number of neighbours to consider
/// * `mu` - Controls the Gaussian kernel strength, i.e., sigma parameter
///
/// ### Returns
///
/// The resulting affinity matrix.
fn affinity_from_distance<T>(dist: &MatRef<T>, k: usize, mu: T) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = dist.nrows();

    // compute average distance to K nearest neighbors for each sample (parallelized)
    let knn_avg_dist: Vec<T> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut distances: Vec<T> = (0..n)
                .filter(|&j| i != j)
                .map(|j| *dist.get(i, j))
                .collect();

            distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let k_actual = k.min(distances.len());
            let dist_sum: T = distances[..k_actual].iter().cloned().sum();
            dist_sum / T::from_usize(k_actual).unwrap()
        })
        .collect();

    // apply gaussian kernel with scaled sigma
    let mut affinity = Mat::zeros(n, n);

    let three = T::from_f32(3.0).unwrap();
    let two = T::from_f32(2.0).unwrap();

    for i in 0..n {
        affinity[(i, i)] = T::one();
        for j in (i + 1)..n {
            let dij = *dist.get(i, j);
            let sigma = mu * (knn_avg_dist[i] + knn_avg_dist[j] + dij) / three;

            if sigma < T::epsilon() {
                affinity[(i, j)] = T::zero();
                affinity[(j, i)] = T::zero();
            } else {
                let weight = (-dij.powi(2) / (two * sigma.powi(2))).exp();
                affinity[(i, j)] = weight;
                affinity[(j, i)] = weight;
            }
        }
    }

    affinity
}

/// KNN thresholding for the matrix
///
/// For each sample, keeps only the K most similar neighbors and normalises
/// their weights to sum to 1. Used to emphasise local structure.
///
/// ### Params
///
/// * `mat` - The similarity matrix to threshold
/// * `k` - Number of nearest neighbors to retain per sample
///
/// ### Returns
///
/// A sparse matrix where each row contains at most K non-zero entries.
fn knn_threshold<T>(mat: &MatRef<T>, k: usize) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = mat.nrows();
    let mut result = Mat::zeros(n, n);

    let rows: Vec<Vec<(usize, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut heap = BinaryHeap::with_capacity(k + 1);

            for j in 0..n {
                if i != j {
                    let sim = *mat.get(i, j);
                    heap.push((OrderedFloat(sim), j));
                    if heap.len() > k {
                        heap.pop();
                    }
                }
            }

            let neighbors: Vec<(usize, T)> = heap
                .into_iter()
                .map(|elem| (elem.1, elem.0.get_value()))
                .collect();

            let knn_sum: T = neighbors.iter().map(|(_, v)| *v).sum();

            neighbors
                .into_iter()
                .map(|(idx, val)| (idx, val / knn_sum))
                .collect()
        })
        .collect();

    for (i, neighbors) in rows.iter().enumerate() {
        for &(j, normalized_val) in neighbors {
            result[(i, j)] = normalized_val;
        }
    }

    result
}

/// B0 normalisation for SNF update step
///
/// Applies modified normalisation where diagonal elements are set to 0.5 and
/// off-diagonal elements are scaled by alpha. Increases stability during the
/// fusion process
///
/// ### Params
///
/// * `mat` - The matrix to normalise.
/// * `alpha` - Normalisation factor for off-diagonal elements (typically 1.0)
///
/// ### Returns
///
/// The normalised matrix with diagonal set to 0.5
fn b0_normalise<T>(mat: &MatRef<T>, alpha: T) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = mat.nrows();
    let normalised = normalise_rows_l1(mat);

    let half = T::from_f32(0.5).unwrap();
    let two = T::from_f32(2.0).unwrap();

    Mat::from_fn(n, n, |i, j| {
        if i == j {
            half
        } else {
            *normalised.get(i, j) / (two * alpha)
        }
    })
}

/// Calculate Gower distance between columns
///
/// I need this one, as I cannot just transpose the other one.
///
/// ### Params
///
/// * `mat` - The data matrix (features × samples)
/// * `is_cat` - Boolean vector indicating which rows (features) are categorical
/// * `ranges` - Optional pre-computed ranges for continuous variables
///
/// ### Returns
///
/// The Gower distance matrix with values in [0, 1]
pub fn snf_gower_dist<T>(mat: &MatRef<T>, is_cat: &[bool], ranges: Option<&[T]>) -> Mat<T>
where
    T: BixverseFloat,
{
    let (nrows, ncols) = mat.shape();

    assert!(
        is_cat.len() != nrows,
        "The categorical vector length {} doesn't match features {}",
        is_cat.len(),
        nrows
    );

    // compute ranges in parallel
    let computed_ranges: Vec<T> = if let Some(r) = ranges {
        assert!(
            r.len() != nrows,
            "The range vector length {} doesn't match features {}",
            r.len(),
            nrows
        );

        r.to_vec()
    } else {
        (0..nrows)
            .into_par_iter()
            .map(|i| {
                if is_cat[i] {
                    T::one()
                } else {
                    let mut min_val = T::infinity();
                    let mut max_val = T::neg_infinity();
                    for j in 0..ncols {
                        let val = *mat.get(i, j);
                        min_val = min_val.min(val);
                        max_val = max_val.max(val);
                    }
                    let range = max_val - min_val;
                    if range < T::epsilon() {
                        T::one()
                    } else {
                        range
                    }
                }
            })
            .collect()
    };

    let mut res = Mat::zeros(ncols, ncols);

    let pairs: Vec<(usize, usize)> = (0..ncols)
        .flat_map(|i| ((i + 1)..ncols).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut total_dist = T::zero();

            for k in 0..nrows {
                let val_i = *mat.get(k, i);
                let val_j = *mat.get(k, j);

                let dist = if is_cat[k] {
                    if (val_i - val_j).abs() < T::epsilon() {
                        T::zero()
                    } else {
                        T::one()
                    }
                } else {
                    (val_i - val_j).abs() / computed_ranges[k]
                };

                total_dist += dist;
            }

            (i, j, total_dist / T::from_usize(nrows).unwrap())
        })
        .collect();

    for (i, j, dist) in results {
        res[(i, j)] = dist;
        res[(j, i)] = dist;
    }

    res
}

////////////////////
// Main functions //
////////////////////

/// Generates an affinity matrix for SNF on continuous values
///
/// ### Params
///
/// * `data` - The underlying data. Assumes the orientation features x samples!
/// * `distance_type` - One of the implemented distances.
/// * `k` - Number of neighbours to consider
/// * `mu` - Controls the Gaussian kernel strength
/// * `normalise` - Shall the data be normalised prior to distance calculation
///
/// ### Returns
///
/// The affinity matrix based on continuous values
pub fn make_affinity_continuous<T>(
    data: &MatRef<T>,
    distance_type: &str,
    k: usize,
    mu: T,
    normalise: bool,
) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let dist_type = parse_distance_type(distance_type).unwrap_or_default();

    let normalised_data = if normalise {
        scale_matrix_col(data, true)
    } else {
        data.to_owned()
    };

    let dist_mat = match dist_type {
        DistanceType::L1Norm => column_pairwise_l1_norm(&normalised_data.as_ref()),
        DistanceType::L2Norm => column_pairwise_l2_norm(&normalised_data.as_ref()),
        DistanceType::Cosine => column_pairwise_cosine_dist(&normalised_data.as_ref()),
        DistanceType::Canberra => column_pairwise_canberra_dist(&normalised_data.as_ref()),
    };

    affinity_from_distance(&dist_mat.as_ref(), k, mu)
}

/// Generates an affinity matrix for SNF on mixed feature types
///
/// ### Params
///
/// * `data` - The underlying data. Assumes the orientation features x samples!
/// * `is_cat` - Which of the features are categorical.
/// * `k` - Number of neighbours to consider
/// * `mu` - Controls the Gaussian kernel strength
///
/// ### Returns
///
/// The affinity matrix based on Gower distance
pub fn make_affinity_mixed<T>(data: &MatRef<T>, is_cat: &[bool], k: usize, mu: T) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let dist_mat = snf_gower_dist(data, is_cat, None);
    affinity_from_distance(&dist_mat.as_ref(), k, mu)
}

/// Generates an affinity matrix for SNF on categorical features
///
/// ### Params
///
/// * `data` - The underlying data. Assumes the orientation features x samples!
/// * `k` - Number of neighbours to consider
/// * `mu` - Controls the Gaussian kernel strength
///
/// ### Returns
///
/// The affinity matrix based on Gower distance
pub fn make_affinity_categorical<T>(data: &MatRef<i32>, k: usize, mu: T) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let dist_mat = column_pairwise_hamming_cat(data);
    affinity_from_distance(&dist_mat.as_ref(), k, mu)
}

/// Run similarity network fusion
///
/// Fuses multiple affinity matrices representing different data types into
/// a unified similarity network. The algorithm iteratively updates each
/// affinity matrix by diffusing information through local neighborhoods while
/// incorporating global structure from other modalities.
///
/// ### Params
///
/// * `aff_mats` - Slice of matrix references representing the individual
///   affinity matrices. The dimensions need to be the same and they need to
///   be symmetric.
/// * `k` - Number of neighbours to consider.
/// * `t` - Number of iterations to run the algorithm for.
/// * `alpha` - Normalisation hyperparameter controlling fusion strength
///   (typically 1.0)
///
/// ### Returns
///
/// The adjcaceny matrix of the finally fused network.
pub fn snf<T>(aff_mats: &[MatRef<T>], k: usize, t: usize, alpha: T) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    assert!(
        !aff_mats.is_empty(),
        "At least one affinity matrix required"
    );

    let n = aff_mats[0].nrows();

    for (i, mat) in aff_mats.iter().enumerate() {
        assert_symmetric_mat!(mat);
        assert!(
            mat.nrows() == n,
            "Matrix {} has different size: {} x {} (expected {} x {})",
            i,
            mat.nrows(),
            mat.ncols(),
            n,
            n
        );
    }

    let m = aff_mats.len();

    let mut aff = aff_mats
        .par_iter()
        .map(|mat| normalise_rows_l1(&mat.as_ref()))
        .collect::<Vec<Mat<T>>>();

    let wk = aff
        .par_iter()
        .map(|mat_i| knn_threshold(&mat_i.as_ref(), k))
        .collect::<Vec<Mat<T>>>();

    for _ in 0..t {
        for v in 0..m {
            // compute sum of all P matrices except the current one
            let mut p_sum: Mat<T> = Mat::zeros(n, n);
            for (idx, mat) in aff.iter().enumerate() {
                if idx != v {
                    p_sum += mat;
                }
            }
            p_sum = &p_sum / (m - 1) as f64;

            let fused = &wk[v] * &p_sum * wk[v].transpose();
            aff[v] = b0_normalise(&fused.as_ref(), alpha)
        }
    }

    // final fusion - mean across all matrices
    let mut w: Mat<T> = aff
        .par_iter()
        .cloned()
        .reduce(|| Mat::zeros(n, n), |acc, mat| acc + mat);

    w = &w / m as f64;

    w = normalise_rows_l1(&w.as_ref());

    w = (&w + w.transpose()) / 2.0;

    for i in 0..n {
        w[(i, i)] = T::one();
    }

    w
}
