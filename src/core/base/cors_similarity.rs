//! Contains correlation, co-variance, distance calculations and similarity
//! types.

use faer::{Mat, MatRef, Scale};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::borrow::Borrow;
use std::hash::Hash;

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
        "euclidean" | "l2" => Some(DistanceType::L2Norm),
        "manhattan" | "l1" => Some(DistanceType::L1Norm),
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
/// * `mat` - The matrix for which to calculate the L1 norm (Manhattan distance)
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
pub fn column_pairwise_gower<T>(mat: &MatRef<T>, is_cat: &[bool], ranges: Option<&[T]>) -> Mat<T>
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

    let pairs: Vec<(usize, usize)> = (0..ncol)
        .flat_map(|i| ((i + 1)..ncol).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(col_i, col_j)| {
            let mut total_dist = T::zero();
            for row in 0..nrow {
                let val_i = *mat.get(row, col_i);
                let val_j = *mat.get(row, col_j);

                let dist = if is_cat[col_i] && is_cat[col_j] {
                    if (val_i - val_j).abs() < T::epsilon() {
                        T::zero()
                    } else {
                        T::one()
                    }
                } else if !is_cat[col_i] && !is_cat[col_j] {
                    (val_i - val_j).abs() / computed_ranges[col_i].max(computed_ranges[col_j])
                } else {
                    T::one()
                };
                total_dist += dist;
            }
            (col_i, col_j, total_dist / T::from_usize(nrow).unwrap())
        })
        .collect();

    let mut res = Mat::zeros(ncol, ncol);
    for (i, j, dist) in results {
        res[(i, j)] = dist;
        res[(j, i)] = dist;
    }
    res
}

//////////////////////////////////
// Topological overlap measures //
//////////////////////////////////

/// Enum for the TOM function
#[derive(Debug, Default)]
pub enum TomType {
    /// Original TOM formulation. Computes overlap as:
    /// (a_ij + l_ij) / (min(k_i, k_j) + 1 - |a_ij|).
    #[default]
    Version1,
    /// Alternative formulation. Computes overlap as 0.5 * (a_ij + l_ij / (min(k_i, k_j) + |a_ij|))
    Version2,
}

/// Parsing the TOM type
pub fn parse_tom_types(s: &str) -> Option<TomType> {
    match s.to_lowercase().as_str() {
        "v1" => Some(TomType::Version1),
        "v2" => Some(TomType::Version2),
        _ => None,
    }
}

/// Calculates the topological overlap measure (TOM) for a given affinity matrix
///
/// The TOM quantifies the relative interconnectedness of two nodes in a network by measuring
/// how much they share neighbors relative to their connectivity. Higher TOM values indicate
/// nodes that are part of the same module or cluster.
///
/// ### Params
///
/// * `affinity_mat` - Symmetric affinity/adjacency matrix
/// * `signed` - Whether to use signed (absolute values) or unsigned connectivity
/// * `tom_type` - Algorithm version (Version1 or Version2)
///
/// ### Returns
/// Symmetric TOM matrix with values in `[0,1]` representing topological overlap
///
/// ### Mathematical Formulation
///
/// #### Connectivity
/// For node i: k_i = Σ_j |a_ij| (signed) or k_i = Σ_j a_ij (unsigned)
///
/// #### Shared Neighbors
/// For nodes i,j: l_ij = Σ_k (a_ik * a_kj) - a_ii*a_ij - a_ij*a_jj
///
/// #### TOM Calculation
///
/// **Version 1:**
/// - Numerator: a_ij + l_ij
/// - Denominator (unsigned): min(k_i, k_j) + 1 - a_ij
/// - Denominator (signed): min(k_i, k_j) + 1 - a_ij (if a_ij ≥ 0) or min(k_i, k_j) + 1 + a_ij (if a_ij < 0)
/// - TOM_ij = numerator / denominator
///
/// **Version 2:**
/// - Divisor (unsigned): min(k_i, k_j) + a_ij
/// - Divisor (signed): min(k_i, k_j) + a_ij (if a_ij ≥ 0) or min(k_i, k_j) - a_ij (if a_ij < 0)
/// - TOM_ij = 0.5 * (a_ij + l_ij/divisor)
pub fn calc_tom<T>(affinity_mat: MatRef<T>, signed: bool, tom_type: TomType) -> Mat<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = affinity_mat.nrows();
    let mut tom_mat = Mat::<T>::zeros(n, n);
    let connectivity = if signed {
        (0..n)
            .map(|i| (0..n).map(|j| affinity_mat.get(i, j).abs()).sum())
            .collect::<Vec<T>>()
    } else {
        col_sums(affinity_mat.as_ref())
    };

    let dot_products = affinity_mat.as_ref() * affinity_mat.as_ref();

    for i in 0..n {
        // set diagonal element to 1
        tom_mat[(i, i)] = T::one();
        for j in (i + 1)..n {
            let a_ij = affinity_mat.get(i, j);
            let shared_neighbours = *dot_products.get(i, j)
                - *affinity_mat.get(i, i) * *affinity_mat.get(i, j)
                - *affinity_mat.get(i, j) * *affinity_mat.get(j, j);
            let f_ki_kj = connectivity[i].min(connectivity[j]);

            let tom_value = match tom_type {
                TomType::Version1 => {
                    let numerator = *a_ij + shared_neighbours;
                    let denominator = if signed {
                        if *a_ij >= T::zero() {
                            f_ki_kj + T::one() - *a_ij
                        } else {
                            f_ki_kj + T::one() + *a_ij
                        }
                    } else {
                        f_ki_kj + T::one() - *a_ij
                    };
                    numerator / denominator
                }
                TomType::Version2 => {
                    let divisor = if signed {
                        if *a_ij >= T::zero() {
                            f_ki_kj + *a_ij
                        } else {
                            f_ki_kj - *a_ij
                        }
                    } else {
                        f_ki_kj + *a_ij
                    };
                    let neighbours = shared_neighbours / divisor;
                    T::from_f64(0.5).unwrap() * (*a_ij + neighbours)
                }
            };
            tom_mat[(i, j)] = tom_value;
            tom_mat[(j, i)] = tom_value;
        }
    }
    tom_mat
}

//////////////////////
// Set similarities //
//////////////////////

/// Calculate the set similarity.
///
/// ### Params
///
/// * `s_1` - The first HashSet.
/// * `s_2` - The second HashSet.
/// * `overlap_coefficient` - Shall the overlap coefficient be returned or the
///   Jaccard similarity
///
/// ### Return
///
/// The Jaccard similarity or overlap coefficient.
pub fn set_similarity<T, F>(s_1: &FxHashSet<T>, s_2: &FxHashSet<T>, overlap_coefficient: bool) -> F
where
    T: Borrow<String> + Hash + Eq,
    F: BixverseFloat,
{
    let i = s_1.intersection(s_2).count() as u64;
    let u = if overlap_coefficient {
        std::cmp::min(s_1.len(), s_2.len()) as u64
    } else {
        s_1.union(s_2).count() as u64
    };
    F::from_u64(i).unwrap() / F::from_u64(u).unwrap()
}

/// Calculate the Jaccard similarity between indices
///
/// Jaccard similarity between two integer slices via sorting
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// The Jaccard similarity
pub fn jaccard_sorted<T>(a: &[i32], b: &[i32]) -> T
where
    T: BixverseFloat,
{
    let mut sorted_a = a.to_vec();
    let mut sorted_b = b.to_vec();
    sorted_a.sort_unstable();
    sorted_b.sort_unstable();

    // Remove duplicates
    sorted_a.dedup();
    sorted_b.dedup();

    let mut intersection = 0;
    let mut i = 0;
    let mut j = 0;

    while i < sorted_a.len() && j < sorted_b.len() {
        match sorted_a[i].cmp(&sorted_b[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }

    let union = sorted_a.len() + sorted_b.len() - intersection;
    T::from_usize(intersection).unwrap() / T::from_usize(union).unwrap()
}

#[cfg(test)]
mod tests {
    // Tests focus mainly on API; the Rest was heavily tested within R

    use super::*;
    use faer::Mat;
    use rustc_hash::FxHashSet;

    // Helper to create a simple matrix for testing
    // 1.0 2.0
    // 3.0 4.0
    // 5.0 6.0
    fn get_test_mat() -> Mat<f64> {
        Mat::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 1) => 2.0,
            (1, 0) => 3.0,
            (1, 1) => 4.0,
            (2, 0) => 5.0,
            (2, 1) => 6.0,
            _ => 0.0,
        })
    }

    fn assert_approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{} != {}", a, b);
    }

    #[test]
    fn test_column_pairwise_cov_f64() {
        let mat = get_test_mat();
        let cov = column_pairwise_cov(&mat.as_ref());

        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);

        // Variance of column 0 (1,3,5) should be 4.0
        assert_approx_eq(*cov.get(0, 0), 4.0);
        // Covariance should be positive as they increase together
        assert!(*cov.get(0, 1) > 0.0);
    }

    #[test]
    fn test_column_pairwise_cor_pearson() {
        let mat = get_test_mat();
        let cor = column_pairwise_cor(&mat.as_ref(), false);

        // Diagonals must be 1.0
        assert_approx_eq(*cor.get(0, 0), 1.0);
        assert_approx_eq(*cor.get(1, 1), 1.0);

        // Correlation should be 1.0 for this linear relationship (x + 1 = y)
        assert_approx_eq(*cor.get(0, 1), 1.0);
    }

    #[test]
    fn test_column_pairwise_cor_spearman() {
        let mat = get_test_mat();
        let cor = column_pairwise_cor(&mat.as_ref(), true); // true for spearman

        // Ranks are identical, so correlation should be 1.0
        assert_approx_eq(*cor.get(0, 1), 1.0);
    }

    #[test]
    fn test_cor_two_matrices() {
        let mat_a = get_test_mat();
        let mat_b = get_test_mat();

        let cor = cor_two_matrices(&mat_a.as_ref(), &mat_b.as_ref(), false);

        // Correlating identical matrices should yield 1.0s on diagonal
        assert_approx_eq(*cor.get(0, 0), 1.0);
        assert_approx_eq(*cor.get(1, 1), 1.0);
    }

    #[test]
    fn test_cov2cor() {
        let mut cov = Mat::<f64>::zeros(2, 2);
        cov[(0, 0)] = 4.0;
        cov[(1, 1)] = 4.0;
        cov[(0, 1)] = 2.0; // correlation should be 0.5
        cov[(1, 0)] = 2.0;

        let cor = cov2cor(cov.as_ref());

        assert_approx_eq(*cor.get(0, 0), 1.0);
        assert_approx_eq(*cor.get(0, 1), 0.5);
    }

    #[test]
    fn test_distances_l2() {
        // Orthogonal vectors: (1, 0) and (0, 1)
        let mat = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let dist = column_pairwise_l2_norm(&mat.as_ref());

        assert_approx_eq(*dist.get(0, 0), 0.0);
        // Sqrt((1-0)^2 + (0-1)^2) = Sqrt(2)
        assert_approx_eq(*dist.get(0, 1), 2.0_f64.sqrt());
    }

    #[test]
    fn test_distances_l1() {
        let mat = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let dist = column_pairwise_l1_norm(&mat.as_ref());

        // |1-0| + |0-1| = 2
        assert_approx_eq(*dist.get(0, 1), 2.0);
    }

    #[test]
    fn test_calc_pmi() {
        // Simple case: 2 identical columns
        let data = vec![vec![true, false], vec![true, false]];
        let pmi = calc_pmi::<f64>(&data, true); // normalised

        assert_approx_eq(*pmi.get(0, 1), 1.0);
    }

    #[test]
    fn test_hamming_cat() {
        use faer::mat;

        // Col 0: 0, 0, 0
        // Col 1: 0, 1, 0
        let mat = mat![[0, 0], [0, 1], [0, 0]];

        let hamming = column_pairwise_hamming_cat::<f64>(&mat.as_ref());

        // 1 mismatch out of 3 rows = 0.333...
        assert_approx_eq(*hamming.get(0, 1), 1.0 / 3.0);
    }

    #[test]
    fn test_gower() {
        let mat = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let is_cat = vec![false, false];

        let gower = row_pairwise_gower(&mat.as_ref(), &is_cat, None);

        assert_approx_eq(*gower.get(0, 1), 1.0);
    }

    #[test]
    fn test_calc_tom() {
        use faer::mat;

        let adj = mat![[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]];

        let tom = calc_tom(adj.as_ref(), false, TomType::Version1);

        assert_approx_eq(*tom.get(0, 0), 1.0);
        assert_approx_eq(*tom.get(1, 1), 1.0);
        assert_approx_eq(*tom.get(2, 2), 1.0);
        assert_approx_eq(*tom.get(0, 1), 0.5);
        assert_approx_eq(*tom.get(1, 2), 1.0 / 3.0);
    }

    #[test]
    fn test_set_similarity() {
        let mut s1 = FxHashSet::default();
        s1.insert("A".to_string());
        s1.insert("B".to_string());

        let mut s2 = FxHashSet::default();
        s2.insert("B".to_string());
        s2.insert("C".to_string());

        // Jaccard: Intersection (1) / Union (3)
        let jaccard: f64 = set_similarity(&s1, &s2, false);
        assert_approx_eq(jaccard, 1.0 / 3.0);

        // Overlap: Intersection (1) / Min(2, 2)
        let overlap: f64 = set_similarity(&s1, &s2, true);
        assert_approx_eq(overlap, 0.5);
    }

    #[test]
    fn test_jaccard_sorted() {
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];

        let sim: f64 = jaccard_sorted(&a, &b);
        // Inter: {2,3} (2), Union: {1,2,3,4} (4) -> 0.5
        assert_approx_eq(sim, 0.5);
    }

    #[test]
    fn test_parse_helpers() {
        assert!(matches!(
            parse_distance_type("l2"),
            Some(DistanceType::L2Norm)
        ));
        assert!(matches!(parse_tom_types("v1"), Some(TomType::Version1)));
    }
}
