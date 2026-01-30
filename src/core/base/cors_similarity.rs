use faer::{Mat, MatRef, Scale};
use rayon::prelude::*;

use crate::core::base::info::*;
use crate::core::math::matrix_helpers::*;
use crate::prelude::*;

///////////////////////
// Column matrix ops //
///////////////////////

/// Calculate the co-variance between columns of a matrix
///
/// ### Params
///
/// * `mat` - Calculates the co-variance between columns of a matrix
///
/// ### Returns
///
/// The resulting co-variance matrix.
pub fn column_pairwise_cov<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let n_rows = mat.nrows();
    let centered = scale_matrix_col(mat, false);

    (centered.transpose() * &centered) / (n_rows - 1) as f64
}

/// Calculate the cosine similarity between columns of a matrix
///
/// ### Params
///
/// * `mat` - Calculates the cosine similarity between columns of a matrix
///
/// ### Returns
///
/// The resulting cosine similarity matrix
pub fn column_pairwise_cos<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let normalised = normalise_matrix_col_l2(mat);

    normalised.transpose() * &normalised
}

/// Calculate the correlation matrix
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the correlation matrix. Assumes
///   that features are columns.
/// * `spearman` - Shall Spearman correlation be used.
///
/// ### Returns
///
/// The resulting correlation matrix.
pub fn column_pairwise_cor<T>(mat: &MatRef<T>, spearman: bool) -> Mat<T>
where
    T: BixverseFloat,
{
    let mat = if spearman {
        rank_matrix_col(mat)
    } else {
        mat.to_owned()
    };
    let scaled = scale_matrix_col(&mat.as_ref(), true);
    let nrow = T::from_usize(scaled.nrows()).unwrap();
    Scale(T::one() / (nrow - T::one())) * (scaled.transpose() * &scaled)
}

/// Calculates the correlation between two matrices
///
/// The two matrices need to have the same number of rows, otherwise the function
/// panics
///
/// ### Params
///
/// * `mat_a` - The first matrix.
/// * `mat_b` - The second matrix.
/// * `spearman` - Shall Spearman correlation be used.
///
/// ### Returns
///
/// The resulting correlation between the samples of the two matrices
pub fn cor_two_matrices<T>(mat_a: &MatRef<T>, mat_b: &MatRef<T>, spearman: bool) -> Mat<T>
where
    T: BixverseFloat,
{
    assert_nrows!(mat_a, mat_b);

    let nrow = T::from_usize(mat_a.nrows()).unwrap();

    let mat_a = if spearman {
        rank_matrix_col(mat_a)
    } else {
        mat_a.to_owned()
    };

    let mat_b = if spearman {
        rank_matrix_col(mat_b)
    } else {
        mat_b.to_owned()
    };

    let mat_a = scale_matrix_col(&mat_a.as_ref(), true);
    let mat_b = scale_matrix_col(&mat_b.as_ref(), true);

    mat_a.transpose() * &mat_b * Scale(T::one() / (nrow - T::one()))
}

/// Calculate the correlation matrix from the co-variance matrix
///
/// ### Params
///
/// * `mat` - The co-variance matrix
///
/// ### Returns
///
/// The resulting correlation matrix.
pub fn cov2cor<T>(mat: MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    assert_symmetric_mat!(mat);
    let n = mat.nrows();
    let mut result = mat.to_owned();
    let inv_sqrt_diag: Vec<T> = (0..n).map(|i| T::one() / mat.get(i, i).sqrt()).collect();
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = *mat.get(i, j) * inv_sqrt_diag[i] * inv_sqrt_diag[j];
        }
    }
    result
}

////////////////////////
// Mutual information //
////////////////////////

/// Calculate the mutual information matrix
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the column-wise mutual
///   information
/// * `n_bins` - Optional number of bins to use. Will default to `sqrt(nrows)`
///   if nothing is provided.
/// * `normalised` - Shall the normalised mutual information be calculated via
///   joint entropy normalisation.
/// * `strategy` - String specifying if equal frequency or equal width binning
///   should be used.
///
/// ### Returns
///
/// The resulting mutual information matrix.
pub fn column_mutual_information<T>(
    mat: &MatRef<T>,
    n_bins: Option<usize>,
    normalised: bool,
    strategy: &str,
) -> Mat<T>
where
    T: BixverseFloat,
{
    let bin_strategy = parse_bin_strategy_type(strategy).unwrap_or_default();
    let binned_mat = bin_matrix_cols(mat, n_bins, bin_strategy);

    let n_cols = binned_mat.ncols();
    let pairs: Vec<(usize, usize)> = (0..n_cols)
        .flat_map(|i| (i + 1..n_cols).map(move |j| (i, j)))
        .collect();
    let mi_vals: Vec<((usize, usize), T)> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let mi = calculate_mi(binned_mat.col(i), binned_mat.col(j), n_bins);
            let nmi = if normalised {
                let joint_entropy =
                    calculate_joint_entropy(binned_mat.col(i), binned_mat.col(j), n_bins);
                mi / joint_entropy
            } else {
                mi
            };
            ((i, j), nmi)
        })
        .collect();
    let entropy: Vec<T> = (0..n_cols)
        .into_par_iter()
        .map(|i| {
            if normalised {
                T::zero()
            } else {
                calculate_entropy(binned_mat.col(i), n_bins)
            }
        })
        .collect();
    let mut mi_matrix = Mat::zeros(n_cols, n_cols);
    for ((i, j), mi_val) in mi_vals {
        mi_matrix[(i, j)] = mi_val;
        mi_matrix[(j, i)] = mi_val;
    }
    for i in 0..n_cols {
        mi_matrix[(i, i)] = entropy[i];
    }
    mi_matrix
}

///////////////
// Distances //
///////////////

/// Distance type enum
#[derive(Debug, Clone, Default)]
pub enum DistanceType {
    /// L2 norm, i.e., Euclidean distance
    #[default]
    L2Norm,
    /// L1 norm, i.e., Manhattan distance
    L1Norm,
    /// Cosine distance
    Cosine,
    /// Canberra distance
    Canberra,
}

/// Parsing the distance type
///
/// ### Params
///
/// * `s` - string defining the distance type
///
/// ### Returns
///
/// The `DistanceType`.
pub fn parse_distance_type(s: &str) -> Option<DistanceType> {
    match s.to_lowercase().as_str() {
        "euclidean" => Some(DistanceType::L2Norm),
        "manhattan" => Some(DistanceType::L1Norm),
        "canberra" => Some(DistanceType::Canberra),
        "cosine" => Some(DistanceType::Cosine),
        _ => None,
    }
}

/// Calculate the cosine distance between columns
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the cosine distance.
///
/// ### Returns
///
/// The resulting cosine distance matrix
pub fn column_pairwise_cosine_dist<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let cosine_sim = column_pairwise_cos(mat);
    let ncols = cosine_sim.ncols();
    let mut res: Mat<T> = Mat::zeros(ncols, ncols);
    for i in 0..ncols {
        for j in 0..ncols {
            if i != j {
                res[(i, j)] = T::one() - cosine_sim.get(i, j).abs();
            }
        }
    }
    res
}

/// Calculate L2 Norm (Euclidean distance) between columns
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the L2 norm (Euclidean distance)
///   pairwise between all columns.
///
/// ### Returns
///
/// The distance matrix based on the L2 Norm.
pub fn column_pairwise_l2_norm<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let ncols = mat.ncols();

    let gram = mat.transpose() * mat;
    let mut col_norms_square = vec![T::zero(); ncols];

    for i in 0..ncols {
        col_norms_square[i] = *gram.get(i, i);
    }

    let mut res: Mat<T> = Mat::zeros(ncols, ncols);
    let two = T::from_f64(2.0).unwrap();

    for i in 0..ncols {
        for j in 0..ncols {
            if i == j {
                res[(i, j)] = T::zero();
            } else {
                let dist_sq = col_norms_square[i] + col_norms_square[j] - two * *gram.get(i, j);
                let dist = dist_sq.max(T::zero()).sqrt();
                res[(i, j)] = dist;
            }
        }
    }

    res
}

/// Calculate L1 Norm (Manhatten distance) between columns
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the L1 norm (Manhatten distance)
///   pairwise between all columns.
///
/// ### Returns
///
/// The distance matrix based on the L1 Norm.
pub fn column_pairwise_l1_norm<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let (nrows, ncols) = mat.shape();
    let mut res: Mat<T> = Mat::zeros(ncols, ncols);

    let pairs: Vec<(usize, usize)> = (0..ncols)
        .flat_map(|i| ((i + 1)..ncols).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let col_i = mat.col(i);
            let col_j = mat.col(j);

            let mut sum_abs = T::zero();

            // manual unrolling in elements of 4
            // could implement SIMD instructions for better performance
            let mut k = 0_usize;

            while k + 3 < nrows {
                unsafe {
                    sum_abs += (*col_i.get_unchecked(k) - *col_j.get_unchecked(k)).abs();
                    sum_abs += (*col_i.get_unchecked(k + 1) - *col_j.get_unchecked(k + 1)).abs();
                    sum_abs += (*col_i.get_unchecked(k + 2) - *col_j.get_unchecked(k + 2)).abs();
                    sum_abs += (*col_i.get_unchecked(k + 3) - *col_j.get_unchecked(k + 3)).abs();
                }
                k += 4;
            }

            // remaining elements
            while k < nrows {
                unsafe {
                    sum_abs += (*col_i.get_unchecked(k) - *col_j.get_unchecked(k)).abs();
                }
                k += 1;
            }

            (i, j, sum_abs)
        })
        .collect();

    for (i, j, dist) in results {
        res[(i, j)] = dist;
        res[(j, i)] = dist;
    }

    res
}

/// Calculate Canberra distance between columns
///
/// ### Params
///
/// * `mat` - The matrix for which to calculate the Canberra distance pairwise
///   between all columns.
///
/// ### Returns
///
/// The Canberra distance matrix between all columns.
pub fn column_pairwise_canberra_dist<T>(mat: &MatRef<T>) -> Mat<T>
where
    T: BixverseFloat,
{
    let (nrows, ncols) = mat.shape();
    let mut res: Mat<T> = Mat::zeros(ncols, ncols);

    let pairs: Vec<(usize, usize)> = (0..ncols)
        .flat_map(|i| ((i + 1)..ncols).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let col_i = mat.col(i);
            let col_j = mat.col(j);

            let mut sum_canberra = T::zero();
            for k in 0..nrows {
                let val_i = *col_i.get(k);
                let val_j = *col_j.get(k);

                let abs_i = val_i.abs();
                let abs_j = val_j.abs();
                let denom = abs_i + abs_j;

                if denom > T::from_f32(f32::EPSILON).unwrap() {
                    sum_canberra += (val_i - val_j).abs() / denom;
                }
            }

            (i, j, sum_canberra)
        })
        .collect();

    for (i, j, dist) in results {
        res[(i, j)] = dist;
        res[(j, i)] = dist;
    }

    res
}

////////////////////////////////////////
// Binary and other distance measures //
////////////////////////////////////////

/// Calculate the pointwise mutual information for a boolean matrix
/// representation
///
/// This function takes in a representation of binary values (`true`, `false`)
/// and calculations the column-wise pointwise mutual information.
///
/// ### Params
///
/// * `x` - A slice of boolean vectors representing the data. The outer vector
///   represents the columns.
/// * `normalise` - Shall the normalised pointwise mutual information be
///   calculated
///
/// ### Returns
///
/// The similarity matrix with (normalised) pointwise mutual information scores
pub fn calc_pmi<T>(x: &[Vec<bool>], normalise: bool) -> Mat<T>
where
    T: BixverseFloat,
{
    let n = x.len();
    let mut sim_mat: Mat<T> = Mat::zeros(n, n);

    let p_values: Vec<T> = x
        .par_iter()
        .map(|col| {
            let sum = col.iter().map(|&x| x as usize).sum::<usize>();
            T::from_usize(sum).unwrap() / T::from_usize(col.len()).unwrap()
        })
        .collect();

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let results: Vec<((usize, usize), T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let col_a = &x[i];
            let col_b = &x[j];
            let p_x = p_values[i];
            let p_y = p_values[j];
            let sum = col_a
                .iter()
                .zip(col_b.iter())
                .map(|(&a, &b)| if a & b { 1 } else { 0 })
                .sum::<usize>();
            let p_xy = T::from_usize(sum).unwrap() / T::from_usize(col_a.len()).unwrap();

            let value = if p_x == T::zero() || p_y == T::zero() || p_xy == T::zero() {
                T::neg_infinity()
            } else {
                let pmi = (p_xy / (p_x * p_y)).log2();
                if normalise { pmi / (-p_xy.log2()) } else { pmi }
            };
            ((i, j), value)
        })
        .collect();

    for ((i, j), value) in results {
        sim_mat[(i, j)] = value;
        sim_mat[(j, i)] = value;
    }

    for i in 0..n {
        if normalise {
            sim_mat[(i, i)] = T::one();
        } else {
            let p_i = p_values[i];
            if p_i > T::zero() {
                sim_mat[(i, i)] = -p_i.log2();
            } else {
                sim_mat[(i, i)] = T::infinity();
            }
        }
    }

    sim_mat
}

/// Calculate Hamming distance between columns
///
/// ### Params
///
/// * `mat` - Integer matrix where categorical values are encoded as integers
///
/// ### Returns
///
/// The Hamming distance matrix with values in [0, 1]
pub fn column_pairwise_hamming_cat<T>(mat: &MatRef<i32>) -> Mat<T>
where
    T: BixverseFloat,
{
    let (nrows, ncols) = mat.shape();
    let mut res = Mat::zeros(ncols, ncols);
    let pairs: Vec<(usize, usize)> = (0..ncols)
        .flat_map(|i| ((i + 1)..ncols).map(move |j| (i, j)))
        .collect();
    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut mismatches = 0;
            for k in 0..nrows {
                if mat.get(k, i) != mat.get(k, j) {
                    mismatches += 1;
                }
            }
            let dist = T::from_usize(mismatches).unwrap() / T::from_usize(nrows).unwrap();
            (i, j, dist)
        })
        .collect();
    for (i, j, dist) in results {
        res[(i, j)] = dist;
        res[(j, i)] = dist;
    }
    res
}

/// Calculate Gower distance between rows (samples) for mixed data types
///
/// Gower distance handles mixed continuous and categorical data by:
/// - Continuous: normalised Manhattan distance |x_i - x_j| / range
/// - Categorical: simple mismatch (0 if same, 1 if different)
///
/// ### Params
///
/// * `mat` - The data matrix (samples × features)
/// * `is_cat` - Boolean vector indicating which columns are categorical
/// * `ranges` - Optional pre-computed ranges for continuous variables. If None,
///   computed from data as max - min for each column.
///
/// ### Returns
///
/// The Gower distance matrix with values in [0, 1]
pub fn row_pairwise_gower<T>(mat: &MatRef<T>, is_cat: &[bool], ranges: Option<&[T]>) -> Mat<T>
where
    T: BixverseFloat,
{
    let (nrow, ncol) = mat.shape();
    assert_eq!(
        is_cat.len(),
        ncol,
        "is_categorical length {} doesn't match features {}",
        is_cat.len(),
        ncol
    );

    let computed_ranges: Vec<T> = if let Some(r) = ranges {
        assert_eq!(
            r.len(),
            ncol,
            "ranges length {} doesn't match features {}",
            r.len(),
            ncol
        );
        r.to_vec()
    } else {
        (0..ncol)
            .into_par_iter()
            .map(|j| {
                if is_cat[j] {
                    T::one()
                } else {
                    let mut min_val = T::infinity();
                    let mut max_val = T::neg_infinity();
                    for i in 0..nrow {
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

    let pairs: Vec<(usize, usize)> = (0..nrow)
        .flat_map(|i| ((i + 1)..nrow).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut total_dist = T::zero();
            for k in 0..ncol {
                let val_i = *mat.get(i, k);
                let val_j = *mat.get(j, k);
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
            (i, j, total_dist / T::from_usize(ncol).unwrap())
        })
        .collect();

    let mut res = Mat::zeros(nrow, nrow);
    for (i, j, dist) in results {
        res[(i, j)] = dist;
        res[(j, i)] = dist;
    }
    res
}

//////////////////////////////////
// Topological overlap measures //
//////////////////////////////////
