//! Similarity network fusion implementation based on Wang, et al., Nat Methods,
//! 2014

use ann_search_rs::prelude::SimdDistance;
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

    // the distance matrix is symmetric, so column i is row i, contiguous
    let knn_avg_dist: Vec<T> = (0..n)
        .into_par_iter()
        .map_init(Vec::new, |distances: &mut Vec<T>, i| {
            distances.clear();
            distances.extend(
                dist.col(i)
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &d)| d),
            );

            let k_actual = k.min(distances.len());
            if k_actual < distances.len() {
                distances.select_nth_unstable_by(k_actual, |a, b| a.partial_cmp(b).unwrap());
            }
            distances[..k_actual].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let dist_sum: T = distances[..k_actual].iter().cloned().sum();
            dist_sum / T::from_usize(k_actual).unwrap()
        })
        .collect();

    // apply gaussian kernel with scaled sigma
    let mut affinity = Mat::zeros(n, n);

    let three = T::from_f32(3.0).unwrap();
    let two = T::from_f32(2.0).unwrap();

    affinity
        .par_col_iter_mut()
        .enumerate()
        .for_each(|(j, mut col)| {
            for i in 0..n {
                if i == j {
                    col[i] = T::one();
                    continue;
                }
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                let dij = dist[(i, j)];
                let sigma = mu * (knn_avg_dist[lo] + knn_avg_dist[hi] + dij) / three;

                col[i] = if sigma < T::epsilon() {
                    T::zero()
                } else {
                    (-dij.powi(2) / (two * sigma.powi(2))).exp()
                };
            }
        });

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
/// Per row, the retained `(column, normalised weight)` pairs, sorted by column.
fn knn_threshold<T>(mat: &MatRef<T>, k: usize) -> Vec<Vec<(usize, T)>>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = mat.nrows();

    // the matrix is not symmetric after row normalisation; the transpose makes
    // row i of `mat` a contiguous column
    let mat_t = mat.transpose().to_owned();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut heap = BinaryHeap::with_capacity(k + 1);

            for (j, &sim) in mat_t.col(i).iter().enumerate() {
                if i == j {
                    continue;
                }
                let entry = (RevOrderedFloat(sim), j);
                if heap.len() < k {
                    heap.push(entry);
                } else if heap.peek().is_some_and(|top| entry < *top) {
                    heap.pop();
                    heap.push(entry);
                }
            }

            let mut neighbors: Vec<(usize, T)> = heap
                .into_iter()
                .map(|elem| (elem.1, elem.0.get_value()))
                .collect();
            neighbors.sort_unstable_by_key(|&(j, _)| j);

            let knn_sum: T = neighbors.iter().map(|(_, v)| *v).sum();

            neighbors
                .into_iter()
                .map(|(idx, val)| (idx, val / knn_sum))
                .collect()
        })
        .collect()
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
        is_cat.len() == nrows,
        "The categorical vector length {} doesn't match features {}",
        is_cat.len(),
        nrows
    );

    // compute ranges in parallel
    let computed_ranges: Vec<T> = if let Some(r) = ranges {
        assert!(
            r.len() == nrows,
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

    // continuous features first, so both inner loops run over contiguous
    // slices with no per-feature branch
    let order: Vec<usize> = (0..nrows)
        .filter(|&k| !is_cat[k])
        .chain((0..nrows).filter(|&k| is_cat[k]))
        .collect();
    let n_cont = is_cat.iter().filter(|&&c| !c).count();
    let ranges_ordered: Vec<T> = order.iter().map(|&k| computed_ranges[k]).collect();
    let ordered = Mat::from_fn(nrows, ncols, |r, c| mat[(order[r], c)]);
    let n_features = T::from_usize(nrows).unwrap();

    let mut res = Mat::zeros(ncols, ncols);

    res.par_col_iter_mut().enumerate().for_each(|(j, mut col)| {
        let col_j = ordered.col_as_slice(j);
        for i in 0..j {
            let col_i = ordered.col_as_slice(i);
            let mut total_dist = T::zero();

            for ((&a, &b), &range) in col_i[..n_cont]
                .iter()
                .zip(&col_j[..n_cont])
                .zip(&ranges_ordered[..n_cont])
            {
                total_dist += (a - b).abs() / range;
            }
            for (&a, &b) in col_i[n_cont..].iter().zip(&col_j[n_cont..]) {
                if (a - b).abs() >= T::epsilon() {
                    total_dist += T::one();
                }
            }

            col[i] = total_dist / n_features;
        }
    });

    for j in 0..ncols {
        for i in 0..j {
            res[(j, i)] = res[(i, j)];
        }
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
    T: BixverseFloat + std::iter::Sum + SimdDistance,
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
        DistanceType::Correlation => {
            let cor = column_pairwise_cor(&normalised_data.as_ref(), false);
            Mat::from_fn(cor.nrows(), cor.ncols(), |i, j| T::one() - cor[(i, j)])
        }
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
        .collect::<Vec<Vec<Vec<(usize, T)>>>>();

    // scratch reused across all updates
    let mut p_sum: Mat<T> = Mat::zeros(n, n);
    let mut wk_p: Mat<T> = Mat::zeros(n, n);
    let mut fused: Mat<T> = Mat::zeros(n, n);
    let inv_others = T::from_f64((m - 1) as f64).unwrap().recip();

    for _ in 0..t {
        for v in 0..m {
            // mean of all P matrices except the current one
            p_sum.par_col_iter_mut().enumerate().for_each(|(c, col)| {
                let col = col.try_as_col_major_mut().unwrap().as_slice_mut();
                col.fill(T::zero());
                for (idx, mat) in aff.iter().enumerate() {
                    if idx != v {
                        for (o, &x) in col.iter_mut().zip(mat.col_as_slice(c)) {
                            *o += x;
                        }
                    }
                }
                for o in col.iter_mut() {
                    *o *= inv_others;
                }
            });

            // wk[v] has k non-zeros per row, so both products are sparse
            let rows = &wk[v];
            wk_p.par_col_iter_mut().enumerate().for_each(|(c, col)| {
                let col = col.try_as_col_major_mut().unwrap().as_slice_mut();
                let p_col = p_sum.col_as_slice(c);
                for (o, row) in col.iter_mut().zip(rows) {
                    let mut acc = T::zero();
                    for &(j, val) in row {
                        acc += val * p_col[j];
                    }
                    *o = acc;
                }
            });
            fused.par_col_iter_mut().enumerate().for_each(|(l, col)| {
                let col = col.try_as_col_major_mut().unwrap().as_slice_mut();
                col.fill(T::zero());
                for &(j, val) in &rows[l] {
                    for (o, &x) in col.iter_mut().zip(wk_p.col_as_slice(j)) {
                        *o += val * x;
                    }
                }
            });

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
