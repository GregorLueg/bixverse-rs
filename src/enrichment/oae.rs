//! Helpers to run hypergeometric tests for standard overenrichment analysis

use rustc_hash::FxHashSet;

use crate::{
    core::math::stats::{calc_fdr, hypergeom_pval},
    prelude::BixverseFloat,
};

////////////////////
// Result structs //
////////////////////

/// A structure to store the hypergeometric enrichment results
#[derive(Clone, Debug)]
pub struct HypergeomResult<T> {
    /// P-value of the hypergeometric tests
    pub pval: Vec<T>,
    /// FDRs of the hypergeometric tests
    pub fdr: Vec<T>,
    /// Odds ratios of the hypergeometric tests
    pub odds_ratio: Vec<T>,
    /// Number of intersections/success
    pub hits: Vec<usize>,
    /// Length of the gene set
    pub gs_length: Vec<usize>,
}

///////////////
// Functions //
///////////////

/// Calculate odds ratios
///
/// ### Params
///
/// * `a1_b1` - In both gene set and target set
/// * `a0_b1` - In gene set, but not in target set
/// * `a1_b0` - In target set, but not in gene set
/// * `a0_b0` - Not in either
///
/// ### Return
///
/// The odds ratio. Pending values, can become infinity.
#[inline]
pub fn hypergeom_odds_ratio<T>(a1_b1: usize, a0_b1: usize, a1_b0: usize, a0_b0: usize) -> T
where
    T: BixverseFloat,
{
    T::from_f64((a1_b1 as f64 / a0_b1 as f64) / (a1_b0 as f64 / a0_b0 as f64)).unwrap()
}

/// Count the number of hits for the hypergeometric tests
///
/// ### Params
///
/// * `gene_set_list` - A slice of String vectors, representing the gene sets
///   you want to count the number of hits against
/// * `target_genes` - A string slice representing the target genes
///
/// ### Returns
///
/// A vector of hits, i.e., intersecting genes.
#[inline]
pub fn count_hits(
    gene_set_list: &[FxHashSet<String>],
    target_genes: &FxHashSet<String>,
) -> Vec<usize> {
    let hits: Vec<usize> = gene_set_list
        .iter()
        .map(|targets| targets.intersection(target_genes).count())
        .collect();

    hits
}

/// Helper function for the hypergeometric test
///
/// ### Params
///
/// - `target_genes` - The target genes for the test
/// - `gene_sets` - The list of vectors with the gene set genes
/// - `gene_universe` - Vector with the all genes of the universe
///
/// ### Returns
///
/// `HypergeomResult` - A tuple with the results.
pub fn hypergeom_helper<T>(
    target_genes: &FxHashSet<String>,
    gene_sets: &[FxHashSet<String>],
    gene_universe: &[String],
) -> HypergeomResult<T>
where
    T: BixverseFloat,
{
    let gene_universe_length = gene_universe.len();
    let trials = target_genes.len();
    let gene_set_lengths = gene_sets.iter().map(|s| s.len()).collect::<Vec<usize>>();

    let hits = count_hits(gene_sets, target_genes);

    let pvals: Vec<T> = hits
        .iter()
        .zip(gene_set_lengths.iter())
        .map(|(hit, gene_set_length)| {
            // A single hit gives q = 0, which is P(X >= 1) and perfectly
            // meaningful; only zero hits is the degenerate case.
            if *hit > 0 {
                hypergeom_pval(
                    *hit - 1,
                    *gene_set_length,
                    gene_universe_length - *gene_set_length,
                    trials,
                )
            } else {
                T::one()
            }
        })
        .collect();

    let odds_ratios: Vec<T> = hits
        .iter()
        .zip(gene_set_lengths.iter())
        .map(|(hit, gene_set_length)| {
            hypergeom_odds_ratio(
                *hit,
                *gene_set_length - *hit,
                trials - *hit,
                // Grouped so the addition lands before the subtractions. The
                // value is non-negative for any valid table, but left to right
                // `N - m - k` underflows whenever the target set is larger than
                // the universe minus the gene set.
                (gene_universe_length + *hit) - *gene_set_length - trials,
            )
        })
        .collect();

    let fdr = calc_fdr(&pvals);

    HypergeomResult {
        pval: pvals,
        fdr,
        odds_ratio: odds_ratios,
        hits,
        gs_length: gene_set_lengths,
    }
}

/// Helper function for the hypergeometric test
///
/// ### Params
///
/// - `res` - The `HypergeomResult` to filter.
/// - `min_overlap` - Optional minimum overlap in terms of hits.
/// - `fdr_threshold` - Optional threshold on the fdr.
///
/// ### Returns
///
/// `HypergeomResult` - The filtered results
pub fn filter_gse_results<T>(
    res: HypergeomResult<T>,
    min_overlap: Option<usize>,
    fdr_threshold: Option<T>,
) -> (HypergeomResult<T>, Vec<usize>)
where
    T: BixverseFloat,
{
    let to_keep: Vec<usize> = (0..res.pval.len())
        .filter(|i| {
            if let Some(min_overlap) = min_overlap
                && res.hits[*i] < min_overlap
            {
                return false;
            }
            if let Some(fdr_threshold) = fdr_threshold
                && res.fdr[*i] > fdr_threshold
            {
                return false;
            }
            true
        })
        .collect();

    (
        HypergeomResult {
            pval: to_keep.iter().map(|i| res.pval[*i]).collect(),
            fdr: to_keep.iter().map(|i| res.fdr[*i]).collect(),
            odds_ratio: to_keep.iter().map(|i| res.odds_ratio[*i]).collect(),
            hits: to_keep.iter().map(|i| res.hits[*i]).collect(),
            gs_length: to_keep.iter().map(|i| res.gs_length[*i]).collect(),
        },
        to_keep.iter().map(|x| *x + 1).collect(),
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `g0..g{n-1}` as a universe vector.
    fn universe(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("g{i}")).collect()
    }

    /// A gene set from explicit indices into that universe.
    fn set_of(indices: &[usize]) -> FxHashSet<String> {
        indices.iter().map(|i| format!("g{i}")).collect()
    }

    /// `(5/5) / (10/80)` reduces to exactly 8.
    #[test]
    fn odds_ratio_is_the_2x2_cross_product() {
        assert_relative_eq!(
            hypergeom_odds_ratio::<f64>(5, 5, 10, 80),
            8.0,
            epsilon = 1e-12
        );
    }

    /// Degenerate cells are pinned so a refactor cannot quietly change them:
    /// an empty second cell gives infinity, an empty first gives zero, and both
    /// empty gives NaN.
    #[test]
    fn odds_ratio_degenerate_cells_are_inf_zero_and_nan() {
        assert!(hypergeom_odds_ratio::<f64>(5, 0, 0, 95).is_infinite());
        assert_relative_eq!(hypergeom_odds_ratio::<f64>(0, 5, 5, 90), 0.0);
        assert!(hypergeom_odds_ratio::<f64>(0, 0, 5, 90).is_nan());
    }

    /// Hits are the plain set intersection, including the empty set.
    #[test]
    fn count_hits_counts_the_intersection() {
        let sets = [set_of(&[0, 1, 2]), set_of(&[5, 6]), FxHashSet::default()];
        assert_eq!(count_hits(&sets, &set_of(&[0, 2, 5])), vec![2, 1, 0]);
    }

    /// Regression: a single hit gives `q = 0`, which is P(X >= 1) and was being
    /// short-circuited to 1.0. For one white ball in a hundred and fifteen
    /// draws, P(X >= 1) = 1 - C(99,15)/C(100,15) = 1 - 85/100 = 0.15 exactly.
    #[test]
    fn single_hit_gene_sets_get_a_real_pvalue() {
        let res: HypergeomResult<f64> = hypergeom_helper(
            &set_of(&(0..15).collect::<Vec<_>>()),
            &[set_of(&[0])],
            &universe(100),
        );

        assert_eq!(res.hits, vec![1]);
        assert_relative_eq!(res.pval[0], 0.15, epsilon = 1e-9);
    }

    /// A well-enriched set must score below a barely-enriched one, and the
    /// odds ratio `(5/5) / (10/80)` is exactly 8.
    #[test]
    fn hypergeom_helper_ranks_enrichment_and_pins_the_odds_ratio() {
        let target: FxHashSet<String> = set_of(&(0..5).chain(50..60).collect::<Vec<_>>());
        let enriched = set_of(&(0..10).collect::<Vec<_>>());
        let background = set_of(&(80..90).collect::<Vec<_>>());

        let res: HypergeomResult<f64> =
            hypergeom_helper(&target, &[enriched, background], &universe(100));

        assert_eq!(res.hits, vec![5, 0]);
        assert_relative_eq!(res.odds_ratio[0], 8.0, epsilon = 1e-12);
        assert!(
            res.pval[0] < res.pval[1],
            "enriched {} should beat background {}",
            res.pval[0],
            res.pval[1]
        );
        assert!((0.0..=1.0).contains(&res.pval[0]));
    }

    /// Regression: the fourth cell was computed left to right as
    /// `(N - m) - k + hits`, which underflows once the target set is larger
    /// than the universe minus the gene set. Here `N - m = 5` against `k = 10`.
    #[test]
    fn hypergeom_helper_survives_a_target_larger_than_the_complement() {
        let res: HypergeomResult<f64> = hypergeom_helper(
            &set_of(&(0..8).chain(15..17).collect::<Vec<_>>()),
            &[set_of(&(0..15).collect::<Vec<_>>())],
            &universe(20),
        );

        assert_eq!(res.hits, vec![8]);
        // (8/7) / (2/3) = 12/7
        assert_relative_eq!(res.odds_ratio[0], 12.0 / 7.0, epsilon = 1e-12);
    }

    /// Regression: an empty gene-set slice reached `calc_fdr`, which indexed
    /// `[n - 1]` and panicked.
    #[test]
    fn hypergeom_helper_handles_no_gene_sets() {
        let res: HypergeomResult<f64> = hypergeom_helper(&set_of(&[0, 1]), &[], &universe(10));

        assert!(res.pval.is_empty() && res.fdr.is_empty() && res.hits.is_empty());
    }

    /// The returned indices are 1-based, which the R side depends on.
    #[test]
    fn filter_gse_results_returns_one_based_indices() {
        let res = HypergeomResult::<f64> {
            pval: vec![0.001, 0.2, 0.03],
            fdr: vec![0.003, 0.2, 0.045],
            odds_ratio: vec![8.0, 1.1, 3.0],
            hits: vec![5, 1, 4],
            gs_length: vec![10, 3, 8],
        };

        let (kept, idx) = filter_gse_results(res.clone(), Some(4), Some(0.05));
        assert_eq!(idx, vec![1, 3]);
        assert_eq!(kept.hits, vec![5, 4]);

        let (all, idx_all) = filter_gse_results(res, None, None);
        assert_eq!(idx_all, vec![1, 2, 3]);
        assert_eq!(all.hits, vec![5, 1, 4]);
    }
}
