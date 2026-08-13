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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a rows x columns matrix from column-major data.
    fn mat_from_cols(cols: &[Vec<f64>]) -> Mat<f64> {
        Mat::from_fn(cols[0].len(), cols.len(), |i, j| cols[j][i])
    }

    /// Pins the offset `adj = n_negative + n_zero / 2` that mitch subtracts from
    /// the 1-based ranks. An exact zero therefore lands on +0.5, not 0.0, because
    /// the ranks are 1-based and the offset only recentres up to the midpoint of
    /// the zero block. This is not an off-by-one: upstream mitch does the same,
    /// see `rank_adj` inside `mitch_rank` at
    /// <https://github.com/markziemann/mitch/blob/master/R/mitch.R>
    ///
    /// ```r
    /// rank_adj <- function(x) {
    ///     xx <- rank(x, na.last = "keep")
    ///     num_neg <- length(which(x < 0))
    ///     num_zero <- length(which(x == 0))
    ///     num_adj <- num_neg + (num_zero/2)
    ///     adj <- xx - num_adj
    ///     adj
    /// }
    /// ```
    ///
    /// R's `rank()` defaults to 1-based average ties, which is exactly what
    /// `rank_vector` does, so both sides return `[-1.5, -0.5, 0.5, 1.5, 2.5]`
    /// here. Real DE statistics rarely contain exact zeros, so this case is the
    /// only place the convention is visible.
    #[test]
    fn mitch_rank_subtracts_the_negative_and_zero_offset() {
        let mat = mat_from_cols(&[vec![-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let ranked = mitch_rank(&mat.as_ref());

        // ranks 1..=5, adj = 2 + 1 / 2 = 2.5.
        let expected = [-1.5, -0.5, 0.5, 1.5, 2.5];
        for (i, &want) in expected.iter().enumerate() {
            assert_relative_eq!(ranked[(i, 0)], want, epsilon = 1e-12);
        }
    }

    /// Pins the tie handling: tied values share the average of their 1-based rank
    /// positions before the offset is subtracted.
    #[test]
    fn mitch_rank_averages_tied_ranks() {
        let mat = mat_from_cols(&[vec![1.0, 1.0, -1.0, 0.0, 0.0]]);
        let ranked = mitch_rank(&mat.as_ref());

        // sorted -1, 0, 0, 1, 1 => ranks 1, 2.5, 2.5, 4.5, 4.5; adj = 1 + 2 / 2 = 2.
        let expected = [2.5, 2.5, -1.0, 0.5, 0.5];
        for (i, &want) in expected.iter().enumerate() {
            assert_relative_eq!(ranked[(i, 0)], want, epsilon = 1e-12);
        }
    }

    /// Pins the hand-computable half of the pathway summary: the per-contrast
    /// scores, the Euclidean `s_dist` and the sample standard deviation `mysd`.
    #[test]
    fn process_mitch_pathway_scores_dist_and_sd_are_hand_computable() {
        let ranked = mat_from_cols(&[
            vec![3.0, 1.0, -1.0, -3.0, 2.0, -2.0],
            vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        ]);
        let res = process_mitch_pathway(ranked.as_ref(), "pathway_a", &[0, 1]);

        // n = 6. In-set means (2.0, 1.5), out-of-set means (-1.0, -0.75).
        // score = 2 * (mean_1 - mean_0) / n => 2 * 3 / 6 = 1.0 and 2 * 2.25 / 6 = 0.75.
        assert_eq!(res.pathway_name, "pathway_a");
        assert_eq!(res.pathway_size, 2);
        assert_eq!(res.scores.len(), 2);
        assert_relative_eq!(res.scores[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(res.scores[1], 0.75, epsilon = 1e-12);

        // sqrt(1.0^2 + 0.75^2) = sqrt(1.5625) = 1.25.
        assert_relative_eq!(res.s_dist, 1.25, epsilon = 1e-12);

        // Sample sd of [1.0, 0.75]: mean 0.875, sum of squares 0.03125 over n - 1 = 1.
        assert_relative_eq!(res.mysd, 0.03125_f64.sqrt(), epsilon = 1e-12);

        assert_eq!(res.anova_pvals.len(), 2);
        assert!(res.anova_pvals.iter().all(|p| (0.0..=1.0).contains(p)));
        assert!((0.0..=1.0).contains(&res.manova_pval));
    }

    /// Pins the sign flip: swapping the pathway for its complement negates every
    /// score but leaves `s_dist` and `mysd` unchanged.
    #[test]
    fn process_mitch_pathway_scores_flip_sign_with_the_complement() {
        let ranked = mat_from_cols(&[
            vec![3.0, 1.0, -1.0, -3.0, 2.0, -2.0],
            vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        ]);
        let a = process_mitch_pathway(ranked.as_ref(), "a", &[0, 1]);
        let b = process_mitch_pathway(ranked.as_ref(), "b", &[2, 3, 4, 5]);

        // mean_1 - mean_0 simply swaps sign, and the divisor n is unchanged.
        assert_relative_eq!(a.scores[0], -b.scores[0], epsilon = 1e-12);
        assert_relative_eq!(a.scores[1], -b.scores[1], epsilon = 1e-12);
        assert_relative_eq!(a.s_dist, b.s_dist, epsilon = 1e-12);
        assert_relative_eq!(a.mysd, b.mysd, epsilon = 1e-12);
        assert_eq!(b.pathway_size, 4);
    }
}
