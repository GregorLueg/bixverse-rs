//! Feature selection for ATAC: the most frequently accessible features.
//!
//! Signac's `FindTopFeatures` on raw counts: rank features by total counts
//! over the cells and keep those above a percentile or a count. There is no
//! variance model; at the depth of single cell ATAC most peaks are 0 or 1 in
//! a cell, so total counts carry the signal.

use indexmap::IndexSet;

use crate::prelude::*;

/// Features read per batch when summing totals.
const TOTALS_FEATURE_BATCH: usize = 1_024;

///////////
// Types //
///////////

/// Which features count as top features.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TopFeatureCutoff {
    /// Keep features whose empirical percentile of the total counts is
    /// strictly above this fraction (Signac `"q5"` is `Quantile(0.05)`).
    Quantile(f64),
    /// Keep features with strictly more total counts than this.
    MinCount(f64),
    /// Keep every feature.
    All,
}

impl Default for TopFeatureCutoff {
    /// Signac's default, `min.cutoff = "q5"`.
    fn default() -> Self {
        TopFeatureCutoff::Quantile(0.05)
    }
}

/// Parse a Signac style cutoff.
///
/// ### Params
///
/// * `s` - `"q<N>"` for the N-th percentile, a number for a minimum count, or
///   `"all"`
///
/// ### Returns
///
/// The cutoff, or `None` for anything else.
pub fn parse_top_feature_cutoff(s: &str) -> Option<TopFeatureCutoff> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("all") {
        return Some(TopFeatureCutoff::All);
    }
    if let Some(q) = s.strip_prefix('q') {
        let q: f64 = q.parse().ok()?;
        return (0.0..=100.0)
            .contains(&q)
            .then_some(TopFeatureCutoff::Quantile(q / 100.0));
    }
    s.parse().ok().map(TopFeatureCutoff::MinCount)
}

/// Total counts and percentile of every feature.
#[derive(Clone, Debug)]
pub struct FeatureTotals {
    /// Summed raw counts per feature over the cells, store order.
    pub count: Vec<f64>,
    /// `ecdf(count)(count)`: the fraction of features with at most this many
    /// counts.
    pub percentile: Vec<f64>,
}

///////////////
// Functions //
///////////////

/// Sum the raw counts of every feature over a set of cells.
///
/// ### Params
///
/// * `gene_reader` - Gene-based (feature-major) reader of the store
/// * `cells` - Cells to sum over, `None` for all
///
/// ### Returns
///
/// One total per feature in store order, or a read error.
pub fn feature_totals(
    gene_reader: &ParallelSparseReader,
    cells: Option<&[usize]>,
) -> Result<Vec<f64>, AtacErrors> {
    let n_features = gene_reader.dims().total_genes;
    let keep: Option<IndexSet<u32>> = cells.map(|c| c.iter().map(|&i| i as u32).collect());

    let mut totals = Vec::with_capacity(n_features);
    let features: Vec<usize> = (0..n_features).collect();
    for batch in features.chunks(TOTALS_FEATURE_BATCH) {
        let csc =
            gene_reader.read_genes_sparse::<u32>(batch, keep.as_ref(), &DataLayerReturn::Raw)?;
        totals.extend((0..batch.len()).map(|f| {
            let (s, e) = (csc.indptr[f] as usize, csc.indptr[f + 1] as usize);
            csc.data[s..e].iter().map(|&x| x as f64).sum::<f64>()
        }));
    }
    Ok(totals)
}

/// Empirical CDF of a sample evaluated at each of its points.
///
/// R's `ecdf(x)(x)`: for each value, the fraction of values that are less
/// than or equal to it, so ties share the highest rank.
///
/// ### Params
///
/// * `x` - The sample
///
/// ### Returns
///
/// One fraction in `(0, 1]` per value, same order as `x`.
fn ecdf_at_points(x: &[f64]) -> Vec<f64> {
    let mut sorted = x.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let n = x.len() as f64;
    x.iter()
        .map(|&v| sorted.partition_point(|&s| s <= v) as f64 / n)
        .collect()
}

/// Total counts and their percentiles, as Signac's `FindTopFeatures` reports
/// them.
///
/// ### Params
///
/// * `gene_reader` - Gene-based (feature-major) reader of the store
/// * `cells` - Cells to sum over, `None` for all
///
/// ### Returns
///
/// The [`FeatureTotals`], or a read error.
pub fn feature_total_stats(
    gene_reader: &ParallelSparseReader,
    cells: Option<&[usize]>,
) -> Result<FeatureTotals, AtacErrors> {
    let count = feature_totals(gene_reader, cells)?;
    let percentile = ecdf_at_points(&count);
    Ok(FeatureTotals { count, percentile })
}

/// Pick the top features from their totals.
///
/// ### Params
///
/// * `totals` - Output of [`feature_total_stats`]
/// * `cutoff` - Which features to keep
///
/// ### Returns
///
/// Indices of the kept features, ascending store order.
pub fn select_top_features(totals: &FeatureTotals, cutoff: TopFeatureCutoff) -> Vec<usize> {
    let keep = |i: usize| match cutoff {
        TopFeatureCutoff::Quantile(q) => totals.percentile[i] > q,
        TopFeatureCutoff::MinCount(c) => totals.count[i] > c,
        TopFeatureCutoff::All => true,
    };
    (0..totals.count.len()).filter(|&i| keep(i)).collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecdf_matches_r_with_ties() {
        // R: ecdf(c(3, 1, 2, 2, 5))(c(3, 1, 2, 2, 5)) = 0.8 0.2 0.6 0.6 1.0
        let p = ecdf_at_points(&[3.0, 1.0, 2.0, 2.0, 5.0]);
        assert_eq!(p, vec![0.8, 0.2, 0.6, 0.6, 1.0]);
    }

    #[test]
    fn test_select_top_features_quantile_is_strict() {
        let totals = FeatureTotals {
            count: vec![3.0, 1.0, 2.0, 2.0, 5.0],
            percentile: ecdf_at_points(&[3.0, 1.0, 2.0, 2.0, 5.0]),
        };
        assert_eq!(
            select_top_features(&totals, TopFeatureCutoff::Quantile(0.6)),
            vec![0, 4]
        );
        assert_eq!(
            select_top_features(&totals, TopFeatureCutoff::MinCount(2.0)),
            vec![0, 4]
        );
        assert_eq!(select_top_features(&totals, TopFeatureCutoff::All).len(), 5);
    }

    #[test]
    fn test_parse_top_feature_cutoff() {
        assert_eq!(
            parse_top_feature_cutoff("q5"),
            Some(TopFeatureCutoff::Quantile(0.05))
        );
        assert_eq!(
            parse_top_feature_cutoff("10"),
            Some(TopFeatureCutoff::MinCount(10.0))
        );
        assert_eq!(parse_top_feature_cutoff("all"), Some(TopFeatureCutoff::All));
        assert_eq!(parse_top_feature_cutoff("q150"), None);
        assert_eq!(parse_top_feature_cutoff("x"), None);
    }
}
