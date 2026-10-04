//! Contains correlation, co-variance, distance calculations and similarity
//! types.

use crate::utils::gemm;
use ann_search_rs::utils::dist::SimdDistance;
use faer::Accum;
use faer::linalg::matmul::triangular::BlockStructure;
use faer::{Mat, MatRef, Scale};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::borrow::Borrow;
use std::borrow::Cow;
use std::hash::Hash;

use crate::core::base::info::*;
use crate::core::math::matrix_helpers::*;
use crate::prelude::*;
use crate::utils::faer_parallelism;

/// Kernel from one query column to every target column
type ColumnKernel<'a, T> = &'a (dyn Fn(&[T], &[&[T]]) -> Vec<T> + Sync);

/// Bits per word of the packed boolean columns
const BITS_PER_WORD: usize = 64;

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
    let n_cols = mat.ncols();
    let centered = scale_matrix_col(mat, false);
    let alpha = T::from_f64(1.0 / (n_rows - 1) as f64).unwrap();

    let mut result = Mat::<T>::zeros(n_cols, n_cols);

    gemm::gram(
        result.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        centered.transpose(),
        centered.as_ref(),
        alpha,
        faer_parallelism(),
    );

    mirror_lower_to_upper(result.as_mut());

    result
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
    let n = normalised.ncols();
    let mut result = Mat::<T>::zeros(n, n);

    gemm::gram(
        result.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        normalised.transpose(),
        normalised.as_ref(),
        T::one(),
        faer_parallelism(),
    );

    mirror_lower_to_upper(result.as_mut());

    result
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
    let n = scaled.ncols();
    let nrow = T::from_usize(scaled.nrows()).unwrap();
    let alpha = T::one() / (nrow - T::one());

    let mut result = Mat::<T>::zeros(n, n);

    gemm::gram(
        result.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        scaled.transpose(),
        scaled.as_ref(),
        alpha,
        faer_parallelism(),
    );

    mirror_lower_to_upper(result.as_mut());

    result
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
    for j in 0..n {
        for i in 0..n {
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
    let n_bins_resolved = resolve_n_bins::<T>(n_bins, binned_mat.nrows());
    let mi_vals: Vec<((usize, usize), T)> = pairs
        .into_par_iter()
        .map_init(
            || MiScratch::new(n_bins_resolved),
            |scratch, (i, j)| {
                let (mi, joint_entropy) =
                    scratch.pair::<T>(binned_mat.col(i), binned_mat.col(j), normalised);
                let nmi = if normalised { mi / joint_entropy } else { mi };
                ((i, j), nmi)
            },
        )
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
    /// Correlation distance, i.e., `1 - Pearson r`
    Correlation,
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
        "correlation" | "pearson" => Some(DistanceType::Correlation),
        _ => None,
    }
}

/// Centre and L2-normalise every column
///
/// Columns with zero variance come back as all zeros, which puts their
/// correlation distance to anything at one.
///
/// ### Params
///
/// * `cols` - Column slices, each non-empty
///
/// ### Returns
///
/// The centred, unit-norm columns as a matrix.
fn centre_and_normalise<T>(cols: &[Cow<'_, [T]>]) -> Mat<T>
where
    T: BixverseFloat + SimdDistance,
{
    let n_rows = cols.first().map_or(0, |c| c.len());
    let mut out = Mat::<T>::zeros(n_rows, cols.len());
    out.par_col_iter_mut()
        .zip(cols.par_iter())
        .for_each(|(mut dst, c)| {
            let n = T::from_usize(c.len()).unwrap();
            let mean = c.iter().fold(T::zero(), |acc, x| acc + *x) / n;
            let dst = dst.as_mut().try_as_col_major_mut().unwrap().as_slice_mut();
            for (d, x) in dst.iter_mut().zip(c.iter()) {
                *d = *x - mean;
            }
            let norm = T::calculate_l2_norm(dst);
            if norm > T::zero() {
                dst.iter_mut().for_each(|x| *x /= norm);
            } else {
                dst.fill(T::zero());
            }
        });
    out
}

/// Apply a pairwise kernel between one query column and all target columns
///
/// Runs the batch-of-four kernel over full groups of four targets and the
/// single-pair kernel over the tail.
///
/// ### Params
///
/// * `q` - Query column
/// * `ys` - Target columns
/// * `f4` - Batch-of-four kernel
/// * `f1` - Single-pair kernel
///
/// ### Returns
///
/// One value per target column.
fn batched_kernel<T, F4, F1>(q: &[T], ys: &[&[T]], f4: F4, f1: F1) -> Vec<T>
where
    T: Copy,
    F4: Fn(&[T], [&[T]; 4]) -> [T; 4],
    F1: Fn(&[T], &[T]) -> T,
{
    let mut out = Vec::with_capacity(ys.len());
    let mut chunks = ys.chunks_exact(4);
    for c in &mut chunks {
        out.extend_from_slice(&f4(q, [c[0], c[1], c[2], c[3]]));
    }
    out.extend(chunks.remainder().iter().map(|y| f1(q, y)));
    out
}

/// Calculate distances between the columns of two matrices
///
/// All kernels go through [SimdDistance], parallel over the columns of
/// `mat_b`. The cosine distance is `1 - |cos|` as in
/// [column_pairwise_cosine_dist]; a zero-norm column gets distance one.
/// Correlation copies both matrices once to centre and normalise the columns.
///
/// ### Params
///
/// * `mat_a` - First matrix, columns are the samples
/// * `mat_b` - Second matrix with the same number of rows
/// * `dist` - The distance type
///
/// ### Returns
///
/// The `ncol(mat_a) x ncol(mat_b)` distance matrix, or an error if the row
/// counts differ or are zero.
pub fn two_matrices_dist<T>(
    mat_a: MatRef<T>,
    mat_b: MatRef<T>,
    dist: &DistanceType,
) -> Result<Mat<T>, BixverseErrors>
where
    T: BixverseFloat + SimdDistance,
{
    if mat_a.nrows() != mat_b.nrows() {
        return Err(BixverseErrors::NonMatchingFeatureDim {
            dim_x: mat_a.nrows(),
            dim_y: mat_b.nrows(),
        });
    }
    if mat_a.nrows() == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "The matrices have no rows.".to_string(),
        ));
    }

    let cols_a = column_slices(mat_a);
    let cols_b = column_slices(mat_b);
    let (n_a, n_b) = (cols_a.len(), cols_b.len());
    let ys: Vec<&[T]> = cols_a.iter().map(|c| c.as_ref()).collect();

    let mut res = Mat::<T>::zeros(n_a, n_b);

    // One output column per column of `mat_b`, so every write is contiguous.
    let fill = |res: &mut Mat<T>, kernel: ColumnKernel<T>| {
        res.par_col_iter_mut()
            .zip(cols_b.par_iter())
            .for_each(|(mut dst, q)| {
                let vals = kernel(q, &ys);
                dst.as_mut()
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .copy_from_slice(&vals);
            });
    };

    match dist {
        DistanceType::L2Norm => fill(&mut res, &|q, ys| {
            batched_kernel(q, ys, T::euclidean_simd_batch_4, T::euclidean_simd)
                .into_iter()
                .map(|d| d.max(T::zero()).sqrt())
                .collect()
        }),
        DistanceType::L1Norm => fill(&mut res, &|q, ys| {
            batched_kernel(q, ys, T::manhattan_simd_batch_4, T::manhattan_simd)
        }),
        DistanceType::Canberra => fill(&mut res, &|q, ys| {
            ys.iter().map(|y| T::canberra_simd(q, y)).collect()
        }),
        DistanceType::Cosine => {
            let norms_a: Vec<T> = ys.iter().map(|y| T::calculate_l2_norm(y)).collect();
            gemm::gemm(
                res.as_mut(),
                Accum::Replace,
                mat_a.transpose(),
                mat_b,
                T::one(),
                faer_parallelism(),
            );
            res.par_col_iter_mut()
                .zip(cols_b.par_iter())
                .for_each(|(mut dst, q)| {
                    let norm_b = T::calculate_l2_norm(q);
                    for (i, v) in dst
                        .as_mut()
                        .try_as_col_major_mut()
                        .unwrap()
                        .as_slice_mut()
                        .iter_mut()
                        .enumerate()
                    {
                        let denom = norms_a[i] * norm_b;
                        *v = if denom > T::zero() {
                            T::one() - (*v / denom).abs()
                        } else {
                            T::one()
                        };
                    }
                });
        }
        DistanceType::Correlation => {
            let a = centre_and_normalise(&cols_a);
            let b = centre_and_normalise(&cols_b);
            gemm::gemm(
                res.as_mut(),
                Accum::Replace,
                a.transpose(),
                b.as_ref(),
                T::one(),
                faer_parallelism(),
            );
            res.par_col_iter_mut().for_each(|mut dst| {
                for v in dst.as_mut().try_as_col_major_mut().unwrap().as_slice_mut() {
                    *v = T::one() - *v;
                }
            });
        }
    }

    Ok(res)
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
    for j in 0..ncols {
        for i in 0..ncols {
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

    let mut gram = Mat::<T>::zeros(ncols, ncols);
    gemm::gram(
        gram.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        mat.transpose(),
        *mat,
        T::one(),
        faer_parallelism(),
    );

    let col_norms_square: Vec<T> = (0..ncols).map(|i| *gram.get(i, i)).collect();

    let mut res: Mat<T> = Mat::zeros(ncols, ncols);
    let two = T::from_f64(2.0).unwrap();

    for i in 0..ncols {
        for j in (i + 1)..ncols {
            // lower triangle of gram, so read (j, i)
            let dist_sq = col_norms_square[i] + col_norms_square[j] - two * *gram.get(j, i);
            let dist = dist_sq.max(T::zero()).sqrt();
            res[(i, j)] = dist;
            res[(j, i)] = dist;
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
    T: BixverseFloat + SimdDistance,
{
    let (_, ncols) = mat.shape();
    let mut res: Mat<T> = Mat::zeros(ncols, ncols);

    let pairs: Vec<(usize, usize)> = (0..ncols)
        .flat_map(|i| ((i + 1)..ncols).map(move |j| (i, j)))
        .collect();

    let cols = column_slices(*mat);

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| (i, j, T::manhattan_simd(&cols[i], &cols[j])))
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
    T: BixverseFloat + SimdDistance,
{
    let (_, ncols) = mat.shape();
    let mut res: Mat<T> = Mat::zeros(ncols, ncols);

    let pairs: Vec<(usize, usize)> = (0..ncols)
        .flat_map(|i| ((i + 1)..ncols).map(move |j| (i, j)))
        .collect();

    let cols = column_slices(*mat);

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| (i, j, T::canberra_simd(&cols[i], &cols[j])))
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

    // Bit-packed columns: the pair counts become popcounts of ANDed words.
    let packed: Vec<Vec<u64>> = x
        .par_iter()
        .map(|col| {
            let mut words = vec![0u64; col.len().div_ceil(BITS_PER_WORD)];
            for (k, &b) in col.iter().enumerate() {
                words[k / BITS_PER_WORD] |= (b as u64) << (k % BITS_PER_WORD);
            }
            words
        })
        .collect();

    let p_values: Vec<T> = packed
        .par_iter()
        .zip(x.par_iter())
        .map(|(words, col)| {
            let sum = words.iter().map(|w| w.count_ones() as usize).sum::<usize>();
            T::from_usize(sum).unwrap() / T::from_usize(col.len()).unwrap()
        })
        .collect();

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let results: Vec<((usize, usize), T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let p_x = p_values[i];
            let p_y = p_values[j];
            let sum = packed[i]
                .iter()
                .zip(packed[j].iter())
                .map(|(a, b)| (a & b).count_ones() as usize)
                .sum::<usize>();
            let p_xy = T::from_usize(sum).unwrap() / T::from_usize(x[i].len()).unwrap();

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
    let cols: Vec<Cow<[i32]>> = (0..ncols)
        .map(|j| match mat.col(j).try_as_col_major() {
            Some(c) => Cow::Borrowed(c.as_slice()),
            None => Cow::Owned(mat.col(j).iter().cloned().collect()),
        })
        .collect();
    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mismatches = cols[i]
                .iter()
                .zip(cols[j].iter())
                .filter(|(a, b)| a != b)
                .count();
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

    // Samples as columns, so each sample's features are contiguous.
    let samples = mat.transpose().to_owned();
    let n_features = T::from_usize(ncol).unwrap();

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let row_i = samples.col(i).try_as_col_major().unwrap().as_slice();
            let row_j = samples.col(j).try_as_col_major().unwrap().as_slice();
            let mut total_dist = T::zero();
            for k in 0..ncol {
                let diff = (row_i[k] - row_j[k]).abs();
                total_dist += if is_cat[k] {
                    if diff < T::epsilon() {
                        T::zero()
                    } else {
                        T::one()
                    }
                } else {
                    diff / computed_ranges[k]
                };
            }
            (i, j, total_dist / n_features)
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
    let cols = column_slices(*mat);

    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(col_i, col_j)| {
            let (x, y) = (&cols[col_i], &cols[col_j]);
            let pairs_xy = x.iter().zip(y.iter());
            let total_dist = if is_cat[col_i] && is_cat[col_j] {
                let mut total = T::zero();
                for (a, b) in pairs_xy {
                    total += if (*a - *b).abs() < T::epsilon() {
                        T::zero()
                    } else {
                        T::one()
                    };
                }
                total
            } else if !is_cat[col_i] && !is_cat[col_j] {
                let range = computed_ranges[col_i].max(computed_ranges[col_j]);
                let mut total = T::zero();
                for (a, b) in pairs_xy {
                    total += (*a - *b).abs() / range;
                }
                total
            } else {
                T::from_usize(nrow).unwrap()
            };
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
    // affinity_mat is assumed symmetric: columns stand in for rows throughout
    // so every read and write below is contiguous.
    let connectivity = if signed {
        (0..n)
            .map(|j| affinity_mat.col(j).iter().map(|v| v.abs()).sum())
            .collect::<Vec<T>>()
    } else {
        col_sums(affinity_mat.as_ref())
    };

    let mut dot_products = Mat::<T>::zeros(n, n);
    gemm::gram(
        dot_products.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        affinity_mat,
        affinity_mat,
        T::one(),
        faer_parallelism(),
    );

    let half = T::from_f64(0.5).unwrap();
    tom_mat
        .par_col_iter_mut()
        .enumerate()
        .for_each(|(i, mut col)| {
            col[i] = T::one();
            let a_ii = *affinity_mat.get(i, i);
            for j in (i + 1)..n {
                let a_ij = *affinity_mat.get(j, i);
                let shared_neighbours =
                    *dot_products.get(j, i) - a_ii * a_ij - a_ij * *affinity_mat.get(j, j);
                let f_ki_kj = connectivity[i].min(connectivity[j]);

                col[j] = match tom_type {
                    TomType::Version1 => {
                        let numerator = a_ij + shared_neighbours;
                        let denominator = if signed && a_ij < T::zero() {
                            f_ki_kj + T::one() + a_ij
                        } else {
                            f_ki_kj + T::one() - a_ij
                        };
                        numerator / denominator
                    }
                    TomType::Version2 => {
                        let divisor = if signed && a_ij < T::zero() {
                            f_ki_kj - a_ij
                        } else {
                            f_ki_kj + a_ij
                        };
                        half * (a_ij + shared_neighbours / divisor)
                    }
                };
            }
        });

    mirror_lower_to_upper(tom_mat.as_mut());
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

    /// A 3 by 2 matrix whose two columns are perfectly linearly related.
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

    /// Covariance is column by column and square in the column count.
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

    /// Pearson normalisation puts ones on the diagonal and saturates on a linear pair.
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

    /// The Spearman flag ranks before correlating instead of being ignored.
    #[test]
    fn test_column_pairwise_cor_spearman() {
        let mat = get_test_mat();
        let cor = column_pairwise_cor(&mat.as_ref(), true); // true for spearman

        // Ranks are identical, so correlation should be 1.0
        assert_approx_eq(*cor.get(0, 1), 1.0);
    }

    /// The cross-matrix form lines columns up in order, so identical inputs give a unit diagonal.
    #[test]
    fn test_cor_two_matrices() {
        let mat_a = get_test_mat();
        let mat_b = get_test_mat();

        let cor = cor_two_matrices(&mat_a.as_ref(), &mat_b.as_ref(), false);

        // Correlating identical matrices should yield 1.0s on diagonal
        assert_approx_eq(*cor.get(0, 0), 1.0);
        assert_approx_eq(*cor.get(1, 1), 1.0);
    }

    /// Scaling a covariance by its diagonal recovers the correlation.
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

    /// L2 is a distance, not a similarity: zero on the diagonal, positive off it.
    #[test]
    fn test_distances_l2() {
        // Orthogonal vectors: (1, 0) and (0, 1)
        let mat = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let dist = column_pairwise_l2_norm(&mat.as_ref());

        assert_approx_eq(*dist.get(0, 0), 0.0);
        // Sqrt((1-0)^2 + (0-1)^2) = Sqrt(2)
        assert_approx_eq(*dist.get(0, 1), 2.0_f64.sqrt());
    }

    /// L1 sums absolute differences instead of squaring them.
    #[test]
    fn test_distances_l1() {
        let mat = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let dist = column_pairwise_l1_norm(&mat.as_ref());

        // |1-0| + |0-1| = 2
        assert_approx_eq(*dist.get(0, 1), 2.0);
    }

    /// Normalised PMI tops out at 1 for perfectly co-occurring boolean columns.
    #[test]
    fn test_calc_pmi() {
        // Simple case: 2 identical columns
        let data = vec![vec![true, false], vec![true, false]];
        let pmi = calc_pmi::<f64>(&data, true); // normalised

        assert_approx_eq(*pmi.get(0, 1), 1.0);
    }

    /// Categorical Hamming is normalised by the row count, not left as a raw mismatch count.
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

    /// Gower runs over rows, not columns, and handles the all-numeric case.
    #[test]
    fn test_gower() {
        let mat = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let is_cat = vec![false, false];

        let gower = row_pairwise_gower(&mat.as_ref(), &is_cat, None);

        assert_approx_eq(*gower.get(0, 1), 1.0);
    }

    /// TOM version 1 on a small adjacency: unit diagonal and shared-neighbour weighting off it.
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

    /// The overlap flag switches the denominator from the union to the smaller set.
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

    /// The sorted merge path gives the same Jaccard as the hash-set path.
    #[test]
    fn test_jaccard_sorted() {
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];

        let sim: f64 = jaccard_sorted(&a, &b);
        // Inter: {2,3} (2), Union: {1,2,3,4} (4) -> 0.5
        assert_approx_eq(sim, 0.5);
    }

    /// The R-facing string parsers still map onto the enum variants they name.
    #[test]
    fn test_parse_helpers() {
        assert!(matches!(
            parse_distance_type("l2"),
            Some(DistanceType::L2Norm)
        ));
        assert!(matches!(
            parse_distance_type("correlation"),
            Some(DistanceType::Correlation)
        ));
        assert!(matches!(parse_tom_types("v1"), Some(TomType::Version1)));
    }

    /// Two-matrix distances agree with the square pairwise versions on the
    /// concatenated matrix, across the batch-of-four path and its tail.
    #[test]
    fn test_two_matrices_dist_matches_pairwise() {
        let (nrow, n_a, n_b) = (50, 3, 5);
        let a = Mat::from_fn(nrow, n_a, |i, j| {
            ((i * 7 + j * 13) % 17) as f64 / 17.0 + 0.01
        });
        let b = Mat::from_fn(nrow, n_b, |i, j| {
            ((i * 5 + j * 11) % 19) as f64 / 19.0 + 0.02
        });
        let c = Mat::from_fn(nrow, n_a + n_b, |i, j| {
            if j < n_a { a[(i, j)] } else { b[(i, j - n_a)] }
        });

        let cor = column_pairwise_cor(&c.as_ref(), false);
        let cor_dist = Mat::from_fn(n_a + n_b, n_a + n_b, |i, j| 1.0 - cor[(i, j)]);
        let cases = [
            (DistanceType::L2Norm, column_pairwise_l2_norm(&c.as_ref())),
            (DistanceType::L1Norm, column_pairwise_l1_norm(&c.as_ref())),
            (
                DistanceType::Canberra,
                column_pairwise_canberra_dist(&c.as_ref()),
            ),
            (
                DistanceType::Cosine,
                column_pairwise_cosine_dist(&c.as_ref()),
            ),
            (DistanceType::Correlation, cor_dist),
        ];

        for (dist, square) in cases {
            let res = two_matrices_dist(a.as_ref(), b.as_ref(), &dist).unwrap();
            assert_eq!((res.nrows(), res.ncols()), (n_a, n_b));
            for i in 0..n_a {
                for j in 0..n_b {
                    let (got, want) = (res[(i, j)], square[(i, n_a + j)]);
                    assert!(
                        (got - want).abs() < 1e-10,
                        "{dist:?} ({i}, {j}): {got} vs {want}"
                    );
                }
            }
        }
    }

    /// Mismatched row counts are an error, not a panic.
    #[test]
    fn test_two_matrices_dist_rejects_row_mismatch() {
        let a = Mat::<f64>::zeros(4, 2);
        let b = Mat::<f64>::zeros(5, 2);
        assert!(two_matrices_dist(a.as_ref(), b.as_ref(), &DistanceType::L2Norm).is_err());
    }
}
