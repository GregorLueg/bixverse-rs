//! Helpers to run the mitch multi-dimensional enrichment approach from
//! Kaspi and Ziemann, Bmc Genomics, 2020

use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::core::math::stats::*;
use crate::core::math::{matrix_helpers::*, vector_helpers::*};
use crate::prelude::*;

//////////////////
// Type aliases //
//////////////////

/// Type alias for processed MitchPathways
///
/// ### Slots
///
/// * `0` - Name of the pathway
/// * `1` - Index positions of the pathway relative to the input matrix
pub type MitchPathways = (Vec<String>, Vec<Vec<usize>>);

////////////////////
// Result structs //
////////////////////

/// Structure to store Mitch results
///
/// ### Fields
///
/// * `pathway_name` - Name of the pathway
/// * `pathway_size` - Size of the pathway
/// * `manova_pval` - P-value of the MANOVA test on the ranked data
/// * `scores` - The scores for each tested contrast for this pathway
/// * `anova_pvals` - The p-values for the individual contrasts based on the
///   ANOVA on top of the MANOVA results
/// * `s_dist` - Calculated distances from the hypotenuse.
/// * `mysd` - The standard deviation of the scores for this pathway.
#[derive(Clone, Debug)]
pub struct MitchResult<'a, T> {
    /// Name of the pathway
    pub pathway_name: &'a str,
    /// Size of the pathway
    pub pathway_size: usize,
    /// P-value of the MANOVA test on the ranked data
    pub manova_pval: T,
    /// The scores for each tested contrast for this pathway
    pub scores: Vec<T>,
    /// The p-values for the individual contrasts based on the ANOVA on top of
    /// the MANOVA results
    pub anova_pvals: Vec<T>,
    /// Calculated distances from the hypotenuse.
    pub s_dist: T,
    /// The standard deviation of the scores for this pathway.
    pub mysd: T,
}

///////////////
// Functions //
///////////////

/// Calculate the MANOVA results for a given pathway
///
/// ### Params
///
/// * `x` - The pre-ranked matrix.
/// * `group1_indices` - The row index positions for which genes belong to the
///   pathway.
///
/// ### Return
///
/// Returns the MANOVA results for this pathway.
pub fn manova_mitch<T: BixverseFloat>(x: MatRef<T>, group1_indices: &[usize]) -> ManovaResult<T> {
    let (n, p) = x.shape();

    assert!(
        group1_indices.iter().all(|&idx| idx < n),
        "All indices must be less than the number of rows"
    );

    let mut is_group1 = vec![false; n];
    for &idx in group1_indices {
        is_group1[idx] = true;
    }

    let group0_indices: Vec<usize> = (0..n).filter(|&i| !is_group1[i]).collect();

    let n1 = group1_indices.len();
    let n0 = group0_indices.len();

    let mut x0 = Mat::zeros(n0, p);
    let mut x1 = Mat::zeros(n1, p);

    for (new_i, &orig_i) in group0_indices.iter().enumerate() {
        for j in 0..p {
            x0[(new_i, j)] = x[(orig_i, j)];
        }
    }

    for (new_i, &orig_i) in group1_indices.iter().enumerate() {
        for j in 0..p {
            x1[(new_i, j)] = x[(orig_i, j)];
        }
    }

    let mean_0 = col_means(x0.as_ref());
    let mean_1 = col_means(x1.as_ref());
    let mean_overall = col_means(x);

    let x_centered = scale_matrix_col(&x.as_ref(), false);
    let x0_centered = scale_matrix_col(&x0.as_ref(), false);
    let x1_centered = scale_matrix_col(&x1.as_ref(), false);

    let sscp_within =
        x0_centered.transpose() * &x0_centered + x1_centered.transpose() * &x1_centered;
    let sscp_total = x_centered.transpose() * &x_centered;
    let sscp_between = &sscp_total - &sscp_within;

    ManovaResult {
        sscp_between,
        sscp_within,
        sscp_total,
        df_between: 1,
        df_within: n - 2,
        df_total: n - 1,
        n_vars: p,
        group_means: vec![mean_0, mean_1],
        overall_mean: mean_overall,
    }
}

/// Calculates the mitch-specific ranks for a given contrast based on the tied
/// method
///
/// ### Params
///
/// * `mat` - The matrix to rank
///
/// ### Returns
///
/// The mitched-ranked matrix
pub fn mitch_rank<T: BixverseFloat>(mat: &MatRef<T>) -> Mat<T> {
    let mut ranked_mat = Mat::zeros(mat.nrows(), mat.ncols());

    ranked_mat
        .par_col_iter_mut()
        .enumerate()
        .for_each(|(col_idx, mut col)| {
            let original_col: Vec<T> = mat.col(col_idx).iter().copied().collect();
            let mut zeros = T::zero();
            let mut neg = T::zero();
            for x in &original_col {
                if *x == T::zero() {
                    zeros += T::one();
                } else if *x < T::zero() {
                    neg += T::one();
                }
            }
            let adj = neg + (zeros / T::from_f64(2.0).unwrap());

            let ranks = rank_vector(&original_col);
            let ranks = ranks.iter().map(|x| *x - adj).collect::<Vec<T>>();

            for (row_idx, &rank) in ranks.iter().enumerate() {
                col[row_idx] = rank;
            }
        });

    ranked_mat
}

/// Wrapper function to process a given pathway
///
/// ### Params
///
/// * `ranked_mat` - The ranked matrix
/// * `pathway_name` - Name of the pathway/gene set that is being tested.
/// * `pathway_indices` - Index positions which genes (rows) belong to this given
///   pathway
///
/// ### Returns
///
/// A `MitchResult` structure with the results for this pathway.
pub fn process_mitch_pathway<'a, T>(
    ranked_mat: MatRef<T>,
    pathway_name: &'a str,
    pathway_indices: &[usize],
) -> MitchResult<'a, T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let nrow = T::from_usize(ranked_mat.nrows()).unwrap();
    let manova_res: ManovaResult<T> = manova_mitch(ranked_mat, pathway_indices);
    let sum_manova: ManovaSummary<T> = ManovaSummary::from_manova_res(&manova_res);
    let sum_anova = summary_aov(&manova_res);

    let p_manova = sum_manova.p_val_pillai;
    let mut p_aovs = Vec::with_capacity(sum_anova.len());

    for sum in sum_anova {
        p_aovs.push(sum.p_val);
    }

    let scores = &manova_res.group_means[0]
        .iter()
        .zip(&manova_res.group_means[1])
        .map(|(mean_0, mean_1)| (T::from_f64(2.0).unwrap() * (*mean_1 - *mean_0)) / nrow)
        .collect::<Vec<T>>();

    let sd_scores = standard_deviation(scores);
    let hypotenuse = scores
        .iter()
        .map(|x| x.powi(2))
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();

    MitchResult {
        pathway_name,
        pathway_size: pathway_indices.len(),
        manova_pval: p_manova,
        scores: scores.clone(),
        anova_pvals: p_aovs,
        s_dist: hypotenuse,
        mysd: sd_scores,
    }
}
