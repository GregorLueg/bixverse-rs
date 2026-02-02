use faer::Mat;
use rayon::prelude::*;

use crate::core::math::stats::z_scores_to_pval;
use crate::prelude::*;

////////////////
// Structures //
////////////////

/// Structure for DiffCor results
///
/// ### Fields
///
/// * `r_a` - Correlation coefficients of a
/// * `r_b` - Correlation coefficients of b
/// * `z_score` - Z-scores of the differential correlation
/// * `p_vals` - Calculated p-values from the Z-scores
#[derive(Clone, Debug)]
pub struct DiffCorRes<T> {
    pub r_a: Vec<T>,
    pub r_b: Vec<T>,
    pub z_score: Vec<T>,
    pub p_vals: Vec<T>,
}

///////////
// Other //
///////////

/// Calculate differential correlations
///
/// The function will panic if the two correlation matrices are not symmetric
/// and do not have the same dimensions.
///
/// ### Params
///
/// * `mat_a` - The first correlation matrix.
/// * `mat_b` - The second correlation matrix.
/// * `no_sample_a` - Number of samples that were present to calculate mat_a.
/// * `no_sample_b` - Number of samples that were present to calculate mat_b.
/// * `spearman` - Was Spearman correlation used.
///
/// ### Returns
///
/// The resulting differential correlation results as a structure.
pub fn calculate_diff_correlation<T>(
    mat_a: &Mat<T>,
    mat_b: &Mat<T>,
    no_sample_a: usize,
    no_sample_b: usize,
    spearman: bool,
) -> DiffCorRes<T>
where
    T: BixverseFloat,
{
    assert_symmetric_mat!(mat_a);
    assert_symmetric_mat!(mat_b);
    // assert_same_dims!(mat_a, mat_b);

    let mut cors_a: Vec<T> = Vec::new();
    let mut cors_b: Vec<T> = Vec::new();

    let upper_triangle_indices = upper_triangle_indices(mat_a.ncols(), 1);

    for (&r, &c) in upper_triangle_indices
        .0
        .iter()
        .zip(upper_triangle_indices.1.iter())
    {
        cors_a.push(*mat_a.get(r, c));
        cors_b.push(*mat_b.get(r, c));
    }

    // Maybe save the original correlations... Note to myself.
    let original_cor_a = cors_a.to_vec();
    let original_cor_b = cors_b.to_vec();

    cors_a.par_iter_mut().for_each(|x| *x = x.atanh());
    cors_b.par_iter_mut().for_each(|x| *x = x.atanh());

    // Constant will depend on if Spearman or Pearson
    let constant = if spearman {
        T::from_f32(1.06).unwrap()
    } else {
        T::from_f32(1.0).unwrap()
    };
    let denominator: T = ((constant
        / (T::from_usize(no_sample_a).unwrap() - T::from_f32(3.0).unwrap()))
        + (constant / (T::from_usize(no_sample_b).unwrap() - T::from_f32(3.0).unwrap())))
    .sqrt();

    let z_scores: Vec<T> = cors_a
        .par_iter()
        .zip(cors_b.par_iter())
        .map(|(a, b)| (*a - *b) / denominator)
        .collect();

    let p_values = z_scores_to_pval(&z_scores, "twosided");

    DiffCorRes {
        r_a: original_cor_a,
        r_b: original_cor_b,
        z_score: z_scores,
        p_vals: p_values,
    }
}
