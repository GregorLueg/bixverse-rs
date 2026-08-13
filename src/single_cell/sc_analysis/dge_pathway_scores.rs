//! Differential gene expression for single cell via Mann Whitney U and AUCell
//! type enrichment of a bag of genes, see Aibar, et al., Nat Methods,
//! 2017.
//!
//! The Mann-Whitney path also reports the AUROC, which is the same U statistic
//! divided by `n1 * n2`, plus a one-vs-many entry point for marker discovery:
//! one reference group against each of several rival groups, with per-gene
//! summaries across the comparisons in the style of scran's `scoreMarkers`.

use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::time::Instant;

use crate::core::math::stats::{calc_fdr, z_scores_to_pval};
use crate::prelude::*;
use crate::single_cell::sc_analysis::fast_ranking::{
    append_cell_chunks, csr_rank_sum_stats_two_groups, rank_csr_chunk_vec,
};

/////////
// DGE //
/////////

////////////////
// Structures //
////////////////

/// Structure to store the Mann Whitney U-based DGE results in
///
/// Every vector except `genes_to_keep` has one entry per *kept* gene, in
/// increasing original gene index. `genes_to_keep` spans the full gene
/// universe and is the mask that maps the results back onto it.
#[derive(Clone, Debug)]
pub struct DgeMannWhitneyRes {
    /// The calculated LFCs
    pub lfc: Vec<f32>,
    /// AUROC of group 1 against group 2, from the same rank sums as the
    /// Z-scores. `0.5` means no separation, `1.0` that every group 1 cell sits
    /// above every group 2 cell.
    pub auroc: Vec<f32>,
    /// Proportions of cells in group 1 expressing the gene.
    pub prop1: Vec<f32>,
    /// Proportions of cells in group 2 expressing the gene.
    pub prop2: Vec<f32>,
    /// The Z-scores based on the Mann-Whitney U test.
    pub z_scores: Vec<f64>,
    /// The p-values from the Mann-Whitney U test.
    pub p_vals: Vec<f64>,
    /// FDR values given the p-values.
    pub fdr: Vec<f64>,
    /// Boolean indicating if the gene was included in the analysis, i.e.,
    /// passed the proportion thresholds.
    pub genes_to_keep: Vec<bool>,
}

/// Structure to store one-vs-many AUROC DGE results in
///
/// Per-comparison fields hold one inner vector per comparison group, in the
/// order the groups were supplied. Every inner vector, and every summary
/// vector, has one entry per kept gene. `genes_to_keep` spans the full gene
/// universe, as in [DgeMannWhitneyRes].
#[derive(Clone, Debug)]
pub struct DgeAurocMultiRes {
    /// AUROC of the reference against each comparison group.
    pub auroc: Vec<Vec<f32>>,
    /// LFC of the reference against each comparison group.
    pub lfc: Vec<Vec<f32>>,
    /// Proportion of cells expressing the gene in each comparison group.
    pub prop_other: Vec<Vec<f32>>,
    /// Tie-corrected Mann-Whitney Z-scores per comparison.
    pub z_scores: Vec<Vec<f64>>,
    /// P-values per comparison.
    pub p_vals: Vec<Vec<f64>>,
    /// FDR per comparison, over the kept gene set.
    pub fdr: Vec<Vec<f64>>,
    /// Proportion of reference cells expressing the gene.
    pub prop_ref: Vec<f32>,
    /// Median AUROC across comparisons. The statistic to sort markers on: it
    /// survives a single closely related rival, which `min_auroc` does not.
    pub median_auroc: Vec<f32>,
    /// Worst AUROC across comparisons, i.e. scran's `min.AUC`. Use it when the
    /// marker has to beat every rival unambiguously.
    pub min_auroc: Vec<f32>,
    /// Mean AUROC across comparisons.
    pub mean_auroc: Vec<f32>,
    /// Best AUROC across comparisons.
    pub max_auroc: Vec<f32>,
    /// Index of the comparison group achieving `min_auroc`, i.e. the rival the
    /// gene struggles hardest to separate the reference from.
    pub worst_comparison: Vec<usize>,
    /// Best rank the gene achieves in any single comparison when genes are
    /// ordered by descending AUROC. `1` means it is the single best
    /// discriminator against at least one rival.
    pub min_rank: Vec<usize>,
    /// Simes-combined p-value across comparisons, the "any" criterion.
    pub simes_p: Vec<f64>,
    /// FDR over `simes_p`.
    pub simes_fdr: Vec<f64>,
    /// Largest p-value across comparisons, the intersection-union test, i.e.
    /// the "all" criterion.
    pub max_p: Vec<f64>,
    /// FDR over `max_p`.
    pub max_p_fdr: Vec<f64>,
    /// Boolean indicating if the gene was included in the analysis, i.e.,
    /// passed the proportion thresholds in at least one group.
    pub genes_to_keep: Vec<bool>,
}

/////////////
// Helpers //
/////////////

/// Calculate the average expression and proportion for the genes
///
/// Takes in slice of CsrCellChunks and calculates the average expression
/// across the genes and proportions in which the gene is expressed.
///
/// ### Params
///
/// * `cells` - A vector of `CsrCellChunk`.
/// * `num_genes` - Number of represented genes in the data.
///
/// ### Returns
///
/// A tuple of `(average expression, proportion of cells expressing gene)`
fn calculate_avg_exp_prop(cells: &[CsrCellChunk], num_genes: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sum_exp = vec![0.0f32; num_genes];
    let mut count_exp = vec![0usize; num_genes];

    for cell in cells {
        for (&gene_idx, &norm_val) in cell.indices.iter().zip(cell.data_norm.iter()) {
            sum_exp[gene_idx as usize] += norm_val.to_f32();
            count_exp[gene_idx as usize] += 1;
        }
    }

    let total_cells = cells.len() as f32;
    let avg_exp: Vec<f32> = sum_exp.iter().map(|&sum| sum / total_cells).collect();
    let prop_exp: Vec<f32> = count_exp
        .iter()
        .map(|&count| count as f32 / total_cells)
        .collect();

    (avg_exp, prop_exp)
}

/// Calculates the AUROC and the tie-corrected Mann Whitney Z-score
///
/// Both fall out of the same U statistic, so they share one scan of the
/// pre-reduced rank sums produced by `csr_rank_sum_stats_two_groups`.
///
/// The variance carries the standard tie correction
/// `var = (n1 n2 / 12) * ((n + 1) - S / (n (n - 1)))` with `S = sum(t^3 - t)`.
/// This matters a great deal here: the data is f16 normalised counts, so the
/// implicit-zero block alone drops the variance by an order of magnitude on a
/// sparsely detected gene, and f16 spacing collapses plenty of the non-zero
/// values into ties on top of that.
///
/// One consequence worth knowing: with the correction in place, AUROC and the
/// Z-score are no longer rank-equivalent. Two genes with identical AUROC get
/// different Z-scores depending on their tie structure, so sorting by effect
/// size and sorting by significance will disagree.
///
/// No continuity correction, matching scran. It is irrelevant beyond a few
/// hundred cells.
///
/// ### Params
///
/// * `rank_sum_1` - Sum of group 1's midranks over the pooled ranking.
/// * `tie_term` - `sum(t^3 - t)` over the gene's tie groups.
/// * `n1` - Number of cells in group 1.
/// * `n2` - Number of cells in group 2.
///
/// ### Returns
///
/// A tuple of `(AUROC of group 1, tie-corrected Z-score)`. A degenerate gene,
/// i.e. one constant across both groups, yields `(0.5, 0.0)` rather than a NaN;
/// an empty group yields `(NaN, 0.0)`.
fn mann_whitney_stats(rank_sum_1: f64, tie_term: f64, n1: usize, n2: usize) -> (f32, f64) {
    if n1 == 0 || n2 == 0 {
        return (f32::NAN, 0.0);
    }

    let n1 = n1 as f64;
    let n2 = n2 as f64;
    let n = n1 + n2;

    let u1 = rank_sum_1 - n1 * (n1 + 1.0) / 2.0;
    let auroc = (u1 / (n1 * n2)) as f32;

    if n < 2.0 {
        return (auroc, 0.0);
    }

    let variance = (n1 * n2 / 12.0) * ((n + 1.0) - tie_term / (n * (n - 1.0)));

    // Hits whenever a gene is constant across both groups, which the shared
    // gene filter in the one-vs-many path guarantees will happen
    let z = if variance <= 0.0 {
        0.0
    } else {
        (u1 - n1 * n2 / 2.0) / variance.sqrt()
    };

    (auroc, z)
}

//////////
// Main //
//////////

/// Get differential expression based on Mann-Whitney
///
/// Reports the AUROC alongside the Z-score, since both derive from the same
/// rank sums. See [mann_whitney_stats] for the tie correction and why the two
/// do not order genes identically.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `grp_1_indices` - The cell indices of group 1.
/// * `grp_2_indices` - The cell indices of group 2.
/// * `min_proportion` - The minimum proportion that a gene needs to be
///   expressed in at least one of the two groups.
/// * `alternative` - The test alternative. One of `"twosided"`, `"greater"`,
///   or `"less"`
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `DgeMannWhitneyRes` structure with results
pub fn calculate_dge_grps_mann_whitney<S: SingleCellReading>(
    reader: &S,
    grp_1_indices: &[usize],
    grp_2_indices: &[usize],
    min_proportion: f32,
    alternative: &str,
    verbose: usize,
) -> Result<DgeMannWhitneyRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_read = Instant::now();

    let no_genes = reader.get_header().total_genes;

    let mut cell_chunks_1: Vec<CsrCellChunk> = reader.read_cells_parallel(grp_1_indices)?;
    let mut cell_chunks_2: Vec<CsrCellChunk> = reader.read_cells_parallel(grp_2_indices)?;

    let end_read = start_read.elapsed();

    if verbosity.normal_verbosity() {
        println!("Loaded in data: {:.2?}", end_read);
    }

    let (avg_exp_1, prop_1) = calculate_avg_exp_prop(&cell_chunks_1, no_genes);
    let (avg_exp_2, prop_2) = calculate_avg_exp_prop(&cell_chunks_2, no_genes);

    let genes_to_keep: Vec<bool> = prop_1
        .iter()
        .zip(prop_2.iter())
        .map(|(&p1, &p2)| p1 >= min_proportion || p2 >= min_proportion)
        .collect();

    let no_genes_new = genes_to_keep.iter().filter(|&&x| x).count();

    cell_chunks_1
        .par_iter_mut()
        .for_each(|cell| cell.filter_genes(&genes_to_keep));
    cell_chunks_2
        .par_iter_mut()
        .for_each(|cell| cell.filter_genes(&genes_to_keep));

    let genes_kept: Vec<usize> = genes_to_keep
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();

    let n1 = cell_chunks_1.len();
    let n2 = cell_chunks_2.len();

    let start_ranking = Instant::now();

    let mut indptr = vec![0_usize];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<F16> = Vec::new();

    append_cell_chunks(&cell_chunks_1, &mut indptr, &mut indices, &mut data);
    append_cell_chunks(&cell_chunks_2, &mut indptr, &mut indices, &mut data);
    drop(cell_chunks_1);
    drop(cell_chunks_2);

    let rank_stats =
        csr_rank_sum_stats_two_groups(&indptr, &indices, &data, n1, n1 + n2, no_genes_new);

    let end_ranking = start_ranking.elapsed();

    if verbosity.normal_verbosity() {
        println!("Finished the ranking across cells: {:.2?}", end_ranking);
    }

    let start_calculations = Instant::now();

    let res: Vec<(f32, f32, f32, f32, f64)> = genes_kept
        .par_iter()
        .enumerate()
        .map(|(new_idx, &original_idx)| {
            let log_fc = avg_exp_1[original_idx] - avg_exp_2[original_idx];
            let prop1 = prop_1[original_idx];
            let prop2 = prop_2[original_idx];

            let (rank_sum, tie_term) = rank_stats[new_idx];
            let (auroc, z) = mann_whitney_stats(rank_sum, tie_term, n1, n2);

            (log_fc, auroc, prop1, prop2, z)
        })
        .collect();

    let mut log_fc = Vec::with_capacity(res.len());
    let mut auroc = Vec::with_capacity(res.len());
    let mut prop1 = Vec::with_capacity(res.len());
    let mut prop2 = Vec::with_capacity(res.len());
    let mut z_scores = Vec::with_capacity(res.len());

    for (log_fc_i, auroc_i, prop1_i, prop2_i, z_i) in res {
        log_fc.push(log_fc_i);
        auroc.push(auroc_i);
        prop1.push(prop1_i);
        prop2.push(prop2_i);
        z_scores.push(z_i);
    }

    let p_vals = z_scores_to_pval(&z_scores, alternative);
    let fdr = calc_fdr(&p_vals);

    let end_calculations = start_calculations.elapsed();

    if verbosity.normal_verbosity() {
        println!("Finished DGE calculations: {:.2?}", end_calculations);
    }

    Ok(DgeMannWhitneyRes {
        lfc: log_fc,
        auroc,
        prop1,
        prop2,
        z_scores,
        p_vals,
        fdr,
        genes_to_keep,
    })
}

//////////////////
// One vs. many //
//////////////////

////////////////
// Structures //
////////////////

/// Per-gene summary across all comparisons, before transposing into
/// [DgeAurocMultiRes]'s column layout.
struct GeneSummary {
    /// Median AUROC across comparisons
    median_auroc: f32,
    /// Worst AUROC across comparisons
    min_auroc: f32,
    /// Mean AUROC across comparisons
    mean_auroc: f32,
    /// Best AUROC across comparisons
    max_auroc: f32,
    /// Comparison achieving `min_auroc`
    worst_comparison: usize,
    /// Best per-comparison AUROC rank the gene achieves anywhere
    min_rank: usize,
    /// Simes-combined p-value
    simes_p: f64,
    /// Largest p-value across comparisons
    max_p: f64,
}

/////////////
// Helpers //
/////////////

/// Median of a slice, taken by value
///
/// Uses `select_nth_unstable` rather than a full sort, and averages the two
/// central values for an even count.
///
/// ### Params
///
/// * `values` - The values to summarise. Reordered in place.
///
/// ### Returns
///
/// The median, or `NaN` for an empty slice.
fn median_in_place(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return f32::NAN;
    }

    let mid = n / 2;
    let (lo, &mut upper, _) = values.select_nth_unstable_by(mid, f32::total_cmp);

    if n % 2 == 1 {
        upper
    } else {
        let lower = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (lower + upper) / 2.0
    }
}

/// Competition ranks by descending value
///
/// Rank 1 is the largest value. Tied values all take the block's lowest rank
/// and the following rank skips accordingly (`1, 2, 2, 4`), matching R's
/// `ties.method = "min"` and therefore scran's `computeMinRank`.
///
/// Midranks would be wrong here: the shared gene filter leaves a large block of
/// genes at an AUROC of exactly 0.5 in some comparisons, and averaging over
/// that block would push their rank up by half its size.
///
/// ### Params
///
/// * `values` - One value per gene.
///
/// ### Returns
///
/// The 1-based competition rank of each gene, in input order.
fn competition_ranks_desc(values: &[f32]) -> Vec<usize> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));

    let mut ranks = vec![0_usize; n];
    let mut i = 0;
    while i < n {
        let start = i;
        while i < n && values[order[i]] == values[order[start]] {
            i += 1;
        }
        for &idx in &order[start..i] {
            ranks[idx] = start + 1;
        }
    }

    ranks
}

/// Simes combination of p-values
///
/// `min_k (m / k) * p_(k)` over the ascending-sorted p-values. This is the
/// "any" criterion: it is small when the gene separates the reference from at
/// least one rival, and it is what scran's `combineMarkers` defaults to.
///
/// ### Params
///
/// * `p_vals` - The p-values to combine. Reordered in place.
///
/// ### Returns
///
/// The combined p-value. The `k = m` term is `max(p)`, so the result can never
/// exceed 1 on valid input; the clamp is defence against a caller passing
/// something that is not a p-value.
fn simes_combine(p_vals: &mut [f64]) -> f64 {
    let m = p_vals.len();
    if m == 0 {
        return f64::NAN;
    }

    p_vals.sort_unstable_by(f64::total_cmp);

    let combined = p_vals
        .iter()
        .enumerate()
        .map(|(k, &p)| (m as f64 / (k + 1) as f64) * p)
        .fold(f64::INFINITY, f64::min);

    combined.min(1.0)
}

/// Reject empty, overlapping or missing cell groups
///
/// Overlap matters more than it looks: a cell present on both sides of a
/// comparison contributes to both rank sums and quietly drags the AUROC toward
/// 0.5 instead of failing.
///
/// ### Params
///
/// * `ref_indices` - The cell indices of the reference group.
/// * `other_indices` - The cell indices of each comparison group.
///
/// ### Returns
///
/// `Ok(())` if the grouping is usable.
fn validate_one_vs_many(
    ref_indices: &[usize],
    other_indices: &[Vec<usize>],
) -> Result<(), BixverseErrors> {
    if ref_indices.is_empty() {
        return Err(BixverseErrors::DgeEmptyReferenceGroup);
    }
    if other_indices.is_empty() {
        return Err(BixverseErrors::DgeNoComparisonGroups);
    }

    let ref_set: FxHashSet<usize> = ref_indices.iter().copied().collect();

    for (group, indices) in other_indices.iter().enumerate() {
        if indices.is_empty() {
            return Err(BixverseErrors::DgeEmptyComparisonGroup { group });
        }
        if let Some(&cell) = indices.iter().find(|c| ref_set.contains(c)) {
            return Err(BixverseErrors::DgeOverlappingGroups { group, cell });
        }
    }

    Ok(())
}

//////////
// Main //
//////////

/// Get one-vs-many AUROC-based differential expression
///
/// Scores one reference group against each comparison group separately and
/// summarises the results per gene. This is the marker question: a gene that
/// marks the reference has to hold up against every rival, which a single
/// pooled test cannot tell you because it is dominated by whichever rival
/// contributes the most cells.
///
/// AUROC rather than the p-value is what the summaries rank on. Group sizes
/// vary widely in practice, and p-values scale with `sqrt(n1 * n2)`, so a
/// large rival would otherwise crowd out a small one regardless of effect size.
///
/// Genes are filtered once, globally: a gene is kept if it clears
/// `min_proportion` in the reference or in any comparison group. Because each
/// gene is ranked independently of the others, the per-comparison statistics
/// are identical to what a per-pair filter would give; the only difference is
/// that every comparison's FDR is computed over the same gene set, which is
/// what makes the cross-comparison summaries well defined.
///
/// Memory is bounded by the reference group plus one comparison group at a
/// time. The comparison groups are read twice, once for the proportions that
/// decide the gene filter and once for the ranking.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `ref_indices` - The cell indices of the reference group.
/// * `other_indices` - The cell indices of each comparison group.
/// * `min_proportion` - The minimum proportion that a gene needs to be
///   expressed in at least one of the groups.
/// * `alternative` - The test alternative. One of `"twosided"`, `"greater"`,
///   or `"less"`, with `"greater"` being the natural choice for markers.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [DgeAurocMultiRes] structure with per-comparison and summary results.
///
/// ### References
///
/// Soneson and Robinson, Nat Methods, 2018 (AUROC for scRNA-seq DGE);
/// Lun, et al., F1000Research, 2016 (scran `scoreMarkers` summaries)
pub fn calculate_dge_one_vs_many_auroc<S: SingleCellReading>(
    reader: &S,
    ref_indices: &[usize],
    other_indices: &[Vec<usize>],
    min_proportion: f32,
    alternative: &str,
    verbose: usize,
) -> Result<DgeAurocMultiRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    validate_one_vs_many(ref_indices, other_indices)?;

    let no_genes = reader.get_header().total_genes;
    let n_comparisons = other_indices.len();

    // -- Pass 1: proportions and average expression, one group at a time --

    let start_pass_1 = Instant::now();

    let mut avg_exp_other: Vec<Vec<f32>> = Vec::with_capacity(n_comparisons);
    let mut prop_other: Vec<Vec<f32>> = Vec::with_capacity(n_comparisons);

    for indices in other_indices {
        let chunks = reader.read_cells_parallel(indices)?;
        let (avg_exp, prop) = calculate_avg_exp_prop(&chunks, no_genes);
        avg_exp_other.push(avg_exp);
        prop_other.push(prop);
    }

    // Read last so its chunks are the ones we hold on to
    let mut ref_chunks = reader.read_cells_parallel(ref_indices)?;
    let (avg_exp_ref, prop_ref_full) = calculate_avg_exp_prop(&ref_chunks, no_genes);

    if verbosity.normal_verbosity() {
        println!("Loaded in data: {:.2?}", start_pass_1.elapsed());
    }

    // -- Shared gene filter --

    let genes_to_keep: Vec<bool> = (0..no_genes)
        .map(|g| {
            prop_ref_full[g] >= min_proportion
                || prop_other.iter().any(|prop| prop[g] >= min_proportion)
        })
        .collect();

    let genes_kept: Vec<usize> = genes_to_keep
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();
    let no_genes_new = genes_kept.len();

    if no_genes_new == 0 {
        return Ok(empty_auroc_multi_res(n_comparisons, genes_to_keep));
    }

    ref_chunks
        .par_iter_mut()
        .for_each(|cell| cell.filter_genes(&genes_to_keep));

    let n_ref = ref_chunks.len();

    // Flatten the reference once; each comparison truncates back to this
    // prefix rather than re-copying it
    let mut indptr = vec![0_usize];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<F16> = Vec::new();
    append_cell_chunks(&ref_chunks, &mut indptr, &mut indices, &mut data);
    drop(ref_chunks);

    let ref_rows = indptr.len();
    let ref_nnz = indices.len();

    // -- Pass 2: one comparison at a time --

    let mut auroc: Vec<Vec<f32>> = Vec::with_capacity(n_comparisons);
    let mut lfc: Vec<Vec<f32>> = Vec::with_capacity(n_comparisons);
    let mut z_scores: Vec<Vec<f64>> = Vec::with_capacity(n_comparisons);
    let mut p_vals: Vec<Vec<f64>> = Vec::with_capacity(n_comparisons);
    let mut fdr: Vec<Vec<f64>> = Vec::with_capacity(n_comparisons);

    for (group, group_indices) in other_indices.iter().enumerate() {
        let start_group = Instant::now();

        let mut chunks = reader.read_cells_parallel(group_indices)?;
        chunks
            .par_iter_mut()
            .for_each(|cell| cell.filter_genes(&genes_to_keep));
        let n_other = chunks.len();

        indptr.truncate(ref_rows);
        indices.truncate(ref_nnz);
        data.truncate(ref_nnz);
        append_cell_chunks(&chunks, &mut indptr, &mut indices, &mut data);
        drop(chunks);

        let rank_stats = csr_rank_sum_stats_two_groups(
            &indptr,
            &indices,
            &data,
            n_ref,
            n_ref + n_other,
            no_genes_new,
        );

        let res: Vec<(f32, f32, f64)> = genes_kept
            .par_iter()
            .enumerate()
            .map(|(new_idx, &original_idx)| {
                let log_fc = avg_exp_ref[original_idx] - avg_exp_other[group][original_idx];
                let (rank_sum, tie_term) = rank_stats[new_idx];
                let (auroc_g, z) = mann_whitney_stats(rank_sum, tie_term, n_ref, n_other);

                (auroc_g, log_fc, z)
            })
            .collect();

        let mut auroc_g = Vec::with_capacity(no_genes_new);
        let mut lfc_g = Vec::with_capacity(no_genes_new);
        let mut z_g = Vec::with_capacity(no_genes_new);

        for (auroc_i, lfc_i, z_i) in res {
            auroc_g.push(auroc_i);
            lfc_g.push(lfc_i);
            z_g.push(z_i);
        }

        let p_g = z_scores_to_pval(&z_g, alternative);
        let fdr_g = calc_fdr(&p_g);

        auroc.push(auroc_g);
        lfc.push(lfc_g);
        z_scores.push(z_g);
        p_vals.push(p_g);
        fdr.push(fdr_g);

        if verbosity.normal_verbosity() {
            let pct_complete = ((group + 1) as f32 / n_comparisons as f32) * 100.0;
            println!(
                "Processed comparison {} out of {} (took {:.2?}, completed {:.1}%)",
                group + 1,
                n_comparisons,
                start_group.elapsed(),
                pct_complete
            );
        }
    }

    // -- Summaries across comparisons --

    let start_summary = Instant::now();

    let per_comparison_ranks: Vec<Vec<usize>> = auroc
        .par_iter()
        .map(|a| competition_ranks_desc(a))
        .collect();

    let summaries: Vec<GeneSummary> = (0..no_genes_new)
        .into_par_iter()
        .map(|gene| {
            let mut gene_auroc: Vec<f32> = auroc.iter().map(|a| a[gene]).collect();
            let mut gene_p: Vec<f64> = p_vals.iter().map(|p| p[gene]).collect();

            let (worst_comparison, &min_auroc) = gene_auroc
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .expect("at least one comparison group exists, checked on entry");
            let max_auroc = gene_auroc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean_auroc =
                gene_auroc.iter().map(|&a| a as f64).sum::<f64>() / n_comparisons as f64;

            let min_rank = per_comparison_ranks
                .iter()
                .map(|r| r[gene])
                .min()
                .expect("at least one comparison group exists, checked on entry");

            let max_p = gene_p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let simes_p = simes_combine(&mut gene_p);

            // Takes &mut and reorders, so it goes last
            let median_auroc = median_in_place(&mut gene_auroc);

            GeneSummary {
                median_auroc,
                min_auroc,
                mean_auroc: mean_auroc as f32,
                max_auroc,
                worst_comparison,
                min_rank,
                simes_p,
                max_p,
            }
        })
        .collect();

    let mut median_auroc = Vec::with_capacity(no_genes_new);
    let mut min_auroc = Vec::with_capacity(no_genes_new);
    let mut mean_auroc = Vec::with_capacity(no_genes_new);
    let mut max_auroc = Vec::with_capacity(no_genes_new);
    let mut worst_comparison = Vec::with_capacity(no_genes_new);
    let mut min_rank = Vec::with_capacity(no_genes_new);
    let mut simes_p = Vec::with_capacity(no_genes_new);
    let mut max_p = Vec::with_capacity(no_genes_new);

    for summary in summaries {
        median_auroc.push(summary.median_auroc);
        min_auroc.push(summary.min_auroc);
        mean_auroc.push(summary.mean_auroc);
        max_auroc.push(summary.max_auroc);
        worst_comparison.push(summary.worst_comparison);
        min_rank.push(summary.min_rank);
        simes_p.push(summary.simes_p);
        max_p.push(summary.max_p);
    }

    let simes_fdr = calc_fdr(&simes_p);
    let max_p_fdr = calc_fdr(&max_p);

    if verbosity.normal_verbosity() {
        println!(
            "Summarised across comparisons: {:.2?}",
            start_summary.elapsed()
        );
    }

    let prop_ref: Vec<f32> = genes_kept.iter().map(|&g| prop_ref_full[g]).collect();
    let prop_other: Vec<Vec<f32>> = prop_other
        .iter()
        .map(|prop| genes_kept.iter().map(|&g| prop[g]).collect())
        .collect();

    Ok(DgeAurocMultiRes {
        auroc,
        lfc,
        prop_other,
        z_scores,
        p_vals,
        fdr,
        prop_ref,
        median_auroc,
        min_auroc,
        mean_auroc,
        max_auroc,
        worst_comparison,
        min_rank,
        simes_p,
        simes_fdr,
        max_p,
        max_p_fdr,
        genes_to_keep,
    })
}

/// Empty result for the case where no gene clears the proportion filter
///
/// ### Params
///
/// * `n_comparisons` - Number of comparison groups.
/// * `genes_to_keep` - The (all false) gene mask.
///
/// ### Returns
///
/// A [DgeAurocMultiRes] whose per-comparison vectors exist but hold no genes.
fn empty_auroc_multi_res(n_comparisons: usize, genes_to_keep: Vec<bool>) -> DgeAurocMultiRes {
    DgeAurocMultiRes {
        auroc: vec![Vec::new(); n_comparisons],
        lfc: vec![Vec::new(); n_comparisons],
        prop_other: vec![Vec::new(); n_comparisons],
        z_scores: vec![Vec::new(); n_comparisons],
        p_vals: vec![Vec::new(); n_comparisons],
        fdr: vec![Vec::new(); n_comparisons],
        prop_ref: Vec::new(),
        median_auroc: Vec::new(),
        min_auroc: Vec::new(),
        mean_auroc: Vec::new(),
        max_auroc: Vec::new(),
        worst_comparison: Vec::new(),
        min_rank: Vec::new(),
        simes_p: Vec::new(),
        simes_fdr: Vec::new(),
        max_p: Vec::new(),
        max_p_fdr: Vec::new(),
        genes_to_keep,
    }
}

////////////
// AUCell //
////////////

///////////
// Enums //
///////////

/// Enum describing the type of AUC to calculate
///
/// All three consume the same within-cell ranking, but they weight it very
/// differently. [AucType::MannWhitney] is a pure function of the gene set's
/// rank sum, so it treats a gene at rank 2 and a gene at rank 200 as almost
/// interchangeable. The other two are top-heavy.
#[derive(Clone, Copy, Debug, Default)]
pub enum AucType {
    /// Recovery-curve AUC under a rank cutoff, i.e. the actual AUCell
    /// statistic of Aibar, et al., Nat Methods, 2017. Top-heavy: only genes
    /// inside the top `max_rank` of the cell contribute.
    #[default]
    Recovery,
    /// AUC derived from the Mann-Whitney U statistic over the full ranking.
    /// Answers "do genes in my set rank higher than genes not in my set?".
    /// Null sits at 0.5 for any gene set size.
    MannWhitney,
    /// Average precision, treating set membership as the positive label.
    /// The most top-heavy of the three, but its null tracks the gene set
    /// prevalence `m / n_genes`, so raw values are not comparable across gene
    /// sets of different size. See [AucellParams::standardise].
    AveragePrecision,
}

/// Parse the desired AUC type
///
/// Careful with `"auroc"`: it is a legacy alias for [AucType::Recovery], the
/// recovery-curve AUCell statistic, which is *not* an AUROC. The genuine
/// AUROC in this module is the gene-wise one reported by the DGE entry points.
/// The alias is kept because downstream callers use it.
///
/// ### Params
///
/// * `s` String specifying the desired AUC type.
///
/// ### Return
///
/// The Option of the `AucType`
pub fn parse_auc_type(s: &str) -> Option<AucType> {
    match s.to_lowercase().as_str() {
        "auroc" | "aucell" | "recovery" => Some(AucType::Recovery),
        "wilcox" | "mannwhitney" => Some(AucType::MannWhitney),
        "aupr" | "ap" => Some(AucType::AveragePrecision),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Default `aucMaxRank` as a fraction of the gene universe.
///
/// Aibar, et al., Nat Methods, 2017 default to the top 5% of the ranking. On a
/// 20k gene universe that is the top 1,000 genes, which for typical scRNA-seq
/// detection depths sits inside the detected genes and therefore never touches
/// the tied block of implicit zeros.
const AUC_MAX_RANK_FRAC: f64 = 0.05;

/// Parameters for the AUCell-type scoring functions
#[derive(Clone, Copy, Debug)]
pub struct AucellParams {
    /// Which statistic to compute.
    pub auc_type: AucType,
    /// Rank cutoff for [AucType::Recovery], counted from the top of the
    /// within-cell ranking. `None` resolves to the top 5% of the gene
    /// universe, see `AUC_MAX_RANK_FRAC`. Ignored
    /// by the other two variants.
    pub max_rank: Option<usize>,
    /// If true, z-score each gene set's scores across cells before returning.
    /// This is what makes [AucType::AveragePrecision] comparable across gene
    /// sets: its set-size dependency is a constant shift and scale per gene
    /// set, so standardising the row removes it exactly.
    pub standardise: bool,
}

impl AucellParams {
    /// Construct the parameters.
    ///
    /// ### Params
    ///
    /// * `auc_type` - Which statistic to compute.
    /// * `max_rank` - Rank cutoff for the recovery-curve AUC. `None` auto-picks.
    /// * `standardise` - Z-score each gene set's scores across cells.
    ///
    /// ### Returns
    ///
    /// The populated [AucellParams]
    pub fn new(auc_type: AucType, max_rank: Option<usize>, standardise: bool) -> Self {
        Self {
            auc_type,
            max_rank,
            standardise,
        }
    }
}

impl Default for AucellParams {
    fn default() -> Self {
        Self {
            auc_type: AucType::MannWhitney,
            max_rank: None,
            standardise: false,
        }
    }
}

/// Resolve the rank cutoff for the recovery-curve AUC
///
/// Heuristic lives here and nowhere else: absent an explicit value, take the
/// top [AUC_MAX_RANK_FRAC] of the gene universe, clamped into `1..=n_genes`.
///
/// ### Params
///
/// * `max_rank` - The user-supplied cutoff, if any.
/// * `n_genes` - Size of the gene universe.
///
/// ### Returns
///
/// The rank cutoff to use.
pub(crate) fn resolve_max_rank(max_rank: Option<usize>, n_genes: usize) -> usize {
    max_rank
        .unwrap_or_else(|| (AUC_MAX_RANK_FRAC * n_genes as f64).ceil() as usize)
        .clamp(1, n_genes.max(1))
}

/////////////
// Helpers //
/////////////

/// Calculate AUC based on ranks and gene set indices (Mann-Whitney version)
///
/// This uses the Mann Whitney statistic under the hood and calculates how
/// active the gene set is over a random gene set. Question asked:
/// "Do genes in my set rank higher than genes not in my set?"
///
/// ### Params
///
/// * `ranks` - The within cell ranked data.
/// * `gene_set` - Indices of the members of this gene set.
///
/// ### Returns
///
/// AUC for this gene set based on the Mann Whitney statistc.
pub fn calculate_auc_per_cell_mw(ranks: &[f32], gene_set: &[usize]) -> f32 {
    let n_genes = ranks.len();
    let n_in_set = gene_set.len();
    let n_not_in_set = n_genes - n_in_set;

    // f64 throughout: on a 20k gene universe the rank sum reaches ~4e8, where
    // f32 accumulation drifts far enough to move the third decimal of the AUC
    let rank_sum: f64 = gene_set.iter().map(|&idx| ranks[idx] as f64).sum();

    let n_in_set = n_in_set as f64;
    let u = rank_sum - n_in_set * (n_in_set + 1.0) / 2.0;

    (u / (n_in_set * n_not_in_set as f64)) as f32
}

/// Collect a gene set's ranks as descending ranks, sorted best-first
///
/// The ranking pipeline hands back ascending midranks (rank 1 = lowest
/// expression, `n_genes` = highest). Every top-heavy statistic here wants the
/// opposite convention, so flip via `n_genes + 1 - r`. Ranks stay `f32`
/// because averaged ties are not integral.
///
/// ### Params
///
/// * `ranks` - The within cell ranked data, ascending midranks.
/// * `gene_set` - Indices of the members of this gene set.
///
/// ### Returns
///
/// The set's descending ranks, sorted ascending (i.e. top-ranked gene first).
fn descending_ranks_sorted(ranks: &[f32], gene_set: &[usize]) -> Vec<f32> {
    let n_genes = ranks.len() as f32;
    let mut out: Vec<f32> = gene_set
        .iter()
        .map(|&idx| n_genes + 1.0 - ranks[idx])
        .collect();
    out.sort_unstable_by(|a, b| a.total_cmp(b));
    out
}

/// Calculate the recovery-curve AUC for one cell (AUCell proper)
///
/// Area under the step function counting how many gene set members have been
/// recovered by each rank, truncated at `max_rank` and normalised by
/// `max_rank * gene_set.len()`. The cutoff is what makes this top-heavy:
/// a gene sitting below `max_rank` contributes nothing at all, so the score
/// answers "how enriched is my gene set at the very top of this cell's
/// ranking?".
///
/// Same formula as `calculate_auc_single` in `crate::methods::cis_target`,
/// which is validated against RcisTarget. That one takes `i32` ranks from a
/// precomputed motif rank matrix, so it is not directly reusable here.
///
/// Note the normalisation constant is not attainable (a perfect set of `m`
/// genes tops out around `1 - m / (2 * max_rank)`), which matches the
/// reference implementation. Scores are meant to be compared across cells,
/// not read as a fraction.
///
/// ### Params
///
/// * `ranks` - The within cell ranked data, ascending midranks.
/// * `gene_set` - Indices of the members of this gene set.
/// * `max_rank` - Rank cutoff, counted from the top of the ranking.
///
/// ### Returns
///
/// Recovery-curve AUC for this gene set, `0.0` if no member clears the cutoff.
///
/// ### References
///
/// Aibar, et al., Nat Methods, 2017
pub fn calculate_auc_recovery(ranks: &[f32], gene_set: &[usize], max_rank: usize) -> f32 {
    if gene_set.is_empty() || max_rank == 0 {
        return 0.0;
    }

    let cutoff = max_rank as f32;
    let mut hits: Vec<f32> = descending_ranks_sorted(ranks, gene_set)
        .into_iter()
        .filter(|&d| d < cutoff)
        .collect();

    if hits.is_empty() {
        return 0.0;
    }

    // Close the curve at the cutoff so the final plateau contributes its width
    hits.push(cutoff);

    let area: f64 = hits
        .windows(2)
        .enumerate()
        .map(|(i, w)| (w[1] - w[0]) as f64 * (i + 1) as f64)
        .sum();

    (area / (cutoff as f64 * gene_set.len() as f64)) as f32
}

/// Calculate the average precision for one cell
///
/// Treats gene set membership as the positive label and the within-cell
/// ranking as the score, then averages the precision at every hit:
/// `AP = (1 / m) * sum_k k / d_k`, with `d_k` the descending rank of the k-th
/// hit. Strongly top-heavy without needing a cutoff, since a hit at rank 1
/// contributes `1` while a hit at rank 5,000 contributes `k / 5000`.
///
/// The null expectation is roughly the prevalence `m / n_genes`, so raw values
/// shrink with small gene sets and are only comparable across cells within one
/// gene set. Standardising each gene set's scores across cells removes this
/// exactly, see [AucellParams::standardise].
///
/// ### Params
///
/// * `ranks` - The within cell ranked data, ascending midranks.
/// * `gene_set` - Indices of the members of this gene set.
///
/// ### Returns
///
/// Average precision for this gene set.
pub fn calculate_ap_per_cell(ranks: &[f32], gene_set: &[usize]) -> f32 {
    if gene_set.is_empty() {
        return 0.0;
    }

    let hits = descending_ranks_sorted(ranks, gene_set);

    let sum: f64 = hits
        .iter()
        .enumerate()
        .map(|(k, &d)| (k + 1) as f64 / d as f64)
        .sum();

    (sum / hits.len() as f64) as f32
}

/// Z-score each gene set's scores across cells, in place
///
/// Operates on the `gene sets x cells` layout the entry points return. Rows
/// whose standard deviation is degenerate are zeroed rather than left as NaN.
///
/// ### Params
///
/// * `rows` - One row of per-cell scores per gene set, mutated in place.
pub(crate) fn standardise_rows(rows: &mut [Vec<f32>]) {
    rows.par_iter_mut().for_each(|row| {
        let n = row.len();
        if n < 2 {
            row.iter_mut().for_each(|v| *v = 0.0);
            return;
        }

        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = row
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1) as f64;
        let sd = var.sqrt();

        if sd < 1e-12 {
            row.iter_mut().for_each(|v| *v = 0.0);
        } else {
            row.iter_mut()
                .for_each(|v| *v = ((*v as f64 - mean) / sd) as f32);
        }
    });
}

//////////
// Main //
//////////

/// Score one cell against every gene set
///
/// Shared inner loop of the AUCell entry points, so the [AucType] dispatch
/// lives in one place.
///
/// ### Params
///
/// * `cell_ranks` - The within cell ranked data, ascending midranks.
/// * `gene_sets` - Slice of Vecs indicating the indices of the gene sets.
/// * `auc_type` - Which statistic to compute.
/// * `max_rank` - Resolved rank cutoff, only used by [AucType::Recovery].
///
/// ### Returns
///
/// One score per gene set, in input order.
pub(crate) fn score_cell(
    cell_ranks: &[f32],
    gene_sets: &[Vec<usize>],
    auc_type: AucType,
    max_rank: usize,
) -> Vec<f32> {
    gene_sets
        .par_iter()
        .map(|gene_set| match auc_type {
            AucType::Recovery => calculate_auc_recovery(cell_ranks, gene_set, max_rank),
            AucType::MannWhitney => calculate_auc_per_cell_mw(cell_ranks, gene_set),
            AucType::AveragePrecision => calculate_ap_per_cell(cell_ranks, gene_set),
        })
        .collect()
}

/// Calculate AUCell
///
/// Scores every cell against every gene set. See [AucType] for what the three
/// statistics measure and when to pick which.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `gene_sets` - Slice of Vecs indicating the indices of the gene sets
/// * `cells_to_keep` - Vector of indices with the cells to keep.
/// * `params` - Optional [AucellParams]. Defaults to the recovery-curve AUC
///   with an auto-picked rank cutoff.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// AUCell-type values in form gene set x cells.
///
/// ### References
///
/// Aibar, et al., Nat Methods, 2017
pub fn calculate_aucell<S: SingleCellReading>(
    reader: &S,
    gene_sets: &[Vec<usize>],
    cells_to_keep: &[usize],
    params: Option<AucellParams>,
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let params = params.unwrap_or_default();

    let start_read = Instant::now();
    let no_genes = reader.get_header().total_genes;
    let cell_chunks: Vec<CsrCellChunk> = reader.read_cells_parallel(cells_to_keep)?;
    let total_cells = cell_chunks.len();
    let end_read = start_read.elapsed();

    if verbosity.normal_verbosity() {
        println!("Loaded in data: {:.2?}", end_read);
    }

    let max_rank = resolve_max_rank(params.max_rank, no_genes);

    let start_ranking = Instant::now();
    let ranks = rank_csr_chunk_vec(cell_chunks, no_genes, true);
    let end_ranking = start_ranking.elapsed();

    if verbosity.normal_verbosity() {
        println!("Ranked gene expression within cells {:.2?}", end_ranking);
    }

    let start_auc = Instant::now();
    let mut all_results: Vec<Vec<f32>> = vec![Vec::with_capacity(total_cells); gene_sets.len()];

    for cell_ranks in ranks {
        let aucs = score_cell(&cell_ranks, gene_sets, params.auc_type, max_rank);

        for (gene_set_idx, auc) in aucs.into_iter().enumerate() {
            all_results[gene_set_idx].push(auc);
        }
    }

    if params.standardise {
        standardise_rows(&mut all_results);
    }

    let end_auc = start_auc.elapsed();

    if verbosity.normal_verbosity() {
        println!("Calulated AUCs {:.2?}", end_auc);
    }

    Ok(all_results)
}

/// Calculate AUCell (streaming)
///
/// As [calculate_aucell], but streams the data in chunks of 50,000 cells to
/// reduce memory pressure. Standardisation, if requested, is applied once at
/// the end across all chunks.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `gene_sets` - Slice of Vecs indicating the indices of the gene sets
/// * `cells_to_keep` - Vector of indices with the cells to keep.
/// * `params` - Optional [AucellParams]. Defaults to the recovery-curve AUC
///   with an auto-picked rank cutoff.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// AUCell-type values in form gene set x cells.
///
/// ### References
///
/// Aibar, et al., Nat Methods, 2017
pub fn calculate_aucell_streaming<S: SingleCellReading>(
    reader: &S,
    gene_sets: &[Vec<usize>],
    cells_to_keep: &[usize],
    params: Option<AucellParams>,
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let params = params.unwrap_or_default();

    const CHUNK_SIZE: usize = 50000;

    let no_genes = reader.get_header().total_genes;
    let total_chunks = cells_to_keep.len().div_ceil(CHUNK_SIZE);
    let max_rank = resolve_max_rank(params.max_rank, no_genes);
    let mut all_results: Vec<Vec<f32>> =
        vec![Vec::with_capacity(cells_to_keep.len()); gene_sets.len()];

    for (chunk_idx, cell_indices_chunk) in cells_to_keep.chunks(CHUNK_SIZE).enumerate() {
        let start_chunk = Instant::now();

        let cell_chunks = reader.read_cells_parallel(cell_indices_chunk)?;
        let ranks = rank_csr_chunk_vec(cell_chunks, no_genes, true);

        for cell_ranks in ranks {
            let aucs = score_cell(&cell_ranks, gene_sets, params.auc_type, max_rank);

            for (gene_set_idx, auc) in aucs.into_iter().enumerate() {
                all_results[gene_set_idx].push(auc);
            }
        }

        if verbosity.normal_verbosity() {
            let elapsed = start_chunk.elapsed();
            let pct_complete = ((chunk_idx + 1) as f32 / total_chunks as f32) * 100.0;
            println!(
                "Processing chunk {} out of {} (took {:.2?}, completed {:.1}%)",
                chunk_idx + 1,
                total_chunks,
                elapsed,
                pct_complete
            );
        }
    }

    if params.standardise {
        standardise_rows(&mut all_results);
    }

    Ok(all_results)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_data::data_io::CellGeneSparseWriter;
    use approx::assert_relative_eq;

    /// Scratch store that removes itself on drop.
    struct TempStore(std::path::PathBuf);

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempStore {
        /// Reserve a uniquely named scratch store in the system temp directory.
        ///
        /// ### Params
        ///
        /// * `name` - Test-unique suffix
        ///
        /// ### Returns
        ///
        /// The guard; the path is available via [`Self::path`].
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_dge_{name}.bin")))
        }

        /// Path of the guarded store as a `&str`.
        ///
        /// ### Returns
        ///
        /// The path.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Open a reader over a freshly written cell-based store.
    ///
    /// ### Params
    ///
    /// * `temp` - Guard owning the path
    /// * `dense` - `dense[cell][gene]` raw counts
    ///
    /// ### Returns
    ///
    /// The reader.
    fn reader_for(temp: &TempStore, dense: &[Vec<u32>]) -> ParallelSparseReader {
        let n_cells = dense.len();
        let n_genes = dense[0].len();
        let mut writer = CellGeneSparseWriter::new(temp.path(), true, n_cells, n_genes, 1e4)
            .expect("writer opens");

        for (cell_idx, cell) in dense.iter().enumerate() {
            let mut raw = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            for (gene, &value) in cell.iter().enumerate() {
                if value > 0 {
                    raw.push(value);
                    indices.push(gene as u32);
                }
            }

            writer
                .write_cell_chunk(CsrCellChunk::from_data(&raw, &indices, cell_idx, 1e4, true))
                .expect("write cell chunk");
        }

        writer.finalise().expect("finalise");
        ParallelSparseReader::new(temp.path()).expect("reader opens")
    }

    /// Identity ascending ranks: gene index `i` has ascending rank `i + 1`,
    /// i.e. descending rank `n - i`.
    fn identity_ranks(n: usize) -> Vec<f32> {
        (1..=n).map(|r| r as f32).collect()
    }

    /// Gene indices whose ascending ranks are the given values.
    fn genes_with_asc_ranks(asc: &[usize]) -> Vec<usize> {
        asc.iter().map(|&r| r - 1).collect()
    }

    /// A gene set sitting at the very top of the ranking scores 1.0, one at the bottom 0.0.
    #[test]
    fn test_mw_auc_perfect_separation() {
        let ranks = identity_ranks(10);

        let top = vec![7, 8, 9];
        assert_relative_eq!(calculate_auc_per_cell_mw(&ranks, &top), 1.0);

        let bottom = vec![0, 1, 2];
        assert_relative_eq!(calculate_auc_per_cell_mw(&ranks, &bottom), 0.0);
    }

    /// Pins the recovery AUC to a hand-computed area, plus the case where nothing clears the cutoff.
    #[test]
    fn test_recovery_auc_known_value() {
        let ranks = identity_ranks(10);

        // Descending ranks {1, 3}, cutoff 4: area = (3-1)*1 + (4-3)*2 = 4,
        // normalised by 4 * 2.
        let gene_set = genes_with_asc_ranks(&[10, 8]);
        assert_relative_eq!(calculate_auc_recovery(&ranks, &gene_set, 4), 0.5);

        // Descending ranks {8, 9}: nothing clears a cutoff of 4.
        let deep = genes_with_asc_ranks(&[3, 2]);
        assert_relative_eq!(calculate_auc_recovery(&ranks, &deep, 4), 0.0);
    }

    /// Regression: rank sum alone cannot separate a top-heavy set from a spread one, the cutoff can.
    #[test]
    fn test_recovery_auc_distinguishes_top_heavy() {
        // Regression guard. Both sets have an identical rank sum of 30, so the
        // Mann-Whitney AUC cannot tell them apart. `concentrated` holds the
        // single top-ranked gene, `spread` sits mid-table. Only a statistic
        // with a rank cutoff separates them.
        let ranks = identity_ranks(20);
        let spread = genes_with_asc_ranks(&[16, 14]);
        let concentrated = genes_with_asc_ranks(&[20, 10]);

        let mw_spread = calculate_auc_per_cell_mw(&ranks, &spread);
        let mw_concentrated = calculate_auc_per_cell_mw(&ranks, &concentrated);
        assert_relative_eq!(mw_spread, 0.75);
        assert_relative_eq!(mw_concentrated, 0.75);

        // Descending {5, 7} both clear the cutoff: area = 2*8 - 12 = 4.
        assert_relative_eq!(calculate_auc_recovery(&ranks, &spread, 8), 4.0 / 16.0);
        // Descending {1, 11}: only the top gene clears, area = 8 - 1 = 7.
        assert_relative_eq!(calculate_auc_recovery(&ranks, &concentrated, 8), 7.0 / 16.0);
    }

    /// Average precision weights hits by depth, so it splits the two sets the Mann-Whitney AUC ties.
    #[test]
    fn test_ap_distinguishes_top_heavy() {
        let ranks = identity_ranks(20);
        let spread = genes_with_asc_ranks(&[16, 14]);
        let concentrated = genes_with_asc_ranks(&[20, 10]);

        // (1/5 + 2/7) / 2
        assert_relative_eq!(
            calculate_ap_per_cell(&ranks, &spread),
            0.242_857_15,
            epsilon = 1e-6
        );
        // (1/1 + 2/11) / 2
        assert_relative_eq!(
            calculate_ap_per_cell(&ranks, &concentrated),
            0.590_909_1,
            epsilon = 1e-6
        );
    }

    /// The rank cutoff clamps to the gene universe and never drops below one.
    #[test]
    fn test_resolve_max_rank() {
        assert_eq!(resolve_max_rank(None, 20_000), 1_000);
        assert_eq!(resolve_max_rank(Some(50), 20_000), 50);
        assert_eq!(resolve_max_rank(Some(0), 20_000), 1);
        assert_eq!(resolve_max_rank(Some(99_999), 20_000), 20_000);
        // Ceiling keeps tiny universes at a usable cutoff
        assert_eq!(resolve_max_rank(None, 3), 1);
    }

    /// Rows are z-scored in place, with zero-variance rows collapsing to zero rather than NaN.
    #[test]
    fn test_standardise_rows() {
        let mut rows = vec![vec![1.0, 2.0, 3.0], vec![5.0, 5.0, 5.0], vec![7.0]];
        standardise_rows(&mut rows);

        assert_relative_eq!(rows[0][0], -1.0);
        assert_relative_eq!(rows[0][1], 0.0);
        assert_relative_eq!(rows[0][2], 1.0);
        // Degenerate rows collapse to zero rather than NaN
        assert_eq!(rows[1], vec![0.0, 0.0, 0.0]);
        assert_eq!(rows[2], vec![0.0]);
    }

    /// Every accepted alias maps to its variant and anything unknown returns `None`.
    #[test]
    fn test_auroc_perfect_separation() {
        // n1 = n2 = 3, group 1 holds the top ranks 4, 5, 6
        let (auroc, z) = mann_whitney_stats(15.0, 0.0, 3, 3);
        assert_relative_eq!(auroc, 1.0);
        assert!(z > 0.0);

        // Group 1 holds the bottom ranks 1, 2, 3
        let (auroc, z) = mann_whitney_stats(6.0, 0.0, 3, 3);
        assert_relative_eq!(auroc, 0.0);
        assert!(z < 0.0);
    }

    #[test]
    fn test_auroc_no_separation() {
        // n1 = n2 = 2, group 1 holds ranks 1 and 4
        let (auroc, _) = mann_whitney_stats(5.0, 0.0, 2, 2);
        assert_relative_eq!(auroc, 0.5);
    }

    #[test]
    fn test_auroc_z_orientation() {
        // On untied data the two are related by a function of n1 and n2 alone
        let (n1, n2) = (3_usize, 3_usize);
        let (auroc, z) = mann_whitney_stats(15.0, 0.0, n1, n2);

        let n = (n1 + n2) as f64;
        let expected = (auroc as f64 - 0.5) * (12.0 * n1 as f64 * n2 as f64 / (n + 1.0)).sqrt();

        assert_relative_eq!(z, expected, epsilon = 1e-12);
        assert!(z > 0.0 && auroc > 0.5);
    }

    #[test]
    fn test_tie_correction_known_value() {
        // Three tied plus two tied over n = 6: S = (27 - 3) + (8 - 2) = 30.
        // var = (9/12) * (7 - 30/30) = 4.5, against an untied 5.25.
        let (auroc, z) = mann_whitney_stats(12.0, 30.0, 3, 3);

        assert_relative_eq!(auroc, 6.0 / 9.0);
        assert_relative_eq!(z, 1.5 / 4.5_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_tie_correction_no_ties() {
        // Same rank sum, no ties: the variance falls back to n1 n2 (n + 1) / 12
        let (_, z) = mann_whitney_stats(12.0, 0.0, 3, 3);
        assert_relative_eq!(z, 1.5 / 5.25_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_constant_gene_degenerate() {
        // Every cell tied: rank sum is n1 (n + 1) / 2 and S = n^3 - n, so the
        // corrected variance is exactly zero. Must not produce a NaN.
        let (auroc, z) = mann_whitney_stats(10.5, 210.0, 3, 3);
        assert_relative_eq!(auroc, 0.5);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn test_mann_whitney_stats_empty_group() {
        let (auroc, z) = mann_whitney_stats(0.0, 0.0, 0, 5);
        assert!(auroc.is_nan());
        assert_eq!(z, 0.0);
    }

    #[test]
    fn test_min_rank_competition_ties() {
        // Competition ranking: the tied block takes the lowest rank and the
        // next rank skips over it
        let ranks = competition_ranks_desc(&[0.9, 0.7, 0.7, 0.5]);
        assert_eq!(ranks, vec![1, 2, 2, 4]);

        // Input order is preserved, not sorted order
        let ranks = competition_ranks_desc(&[0.5, 0.9, 0.7]);
        assert_eq!(ranks, vec![3, 1, 2]);
    }

    #[test]
    fn test_simes_and_max_p() {
        // m = 3: min(3/1 * 0.01, 3/2 * 0.04, 3/3 * 0.30) = 0.03
        let mut p = vec![0.01, 0.30, 0.04];
        let max_p = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_relative_eq!(simes_combine(&mut p), 0.03, epsilon = 1e-12);
        assert_relative_eq!(max_p, 0.30);

        // The k = m term is max(p), so a set of large p-values collapses to it
        let mut p = vec![0.9, 0.95];
        assert_relative_eq!(simes_combine(&mut p), 0.95, epsilon = 1e-12);
    }

    #[test]
    fn test_median_in_place() {
        let mut odd = [3.0, 1.0, 2.0];
        assert_relative_eq!(median_in_place(&mut odd), 2.0);

        let mut even = [4.0, 1.0, 3.0, 2.0];
        assert_relative_eq!(median_in_place(&mut even), 2.5);

        let mut single = [7.0];
        assert_relative_eq!(median_in_place(&mut single), 7.0);
    }

    #[test]
    fn test_validate_one_vs_many() {
        assert!(matches!(
            validate_one_vs_many(&[], &[vec![1]]),
            Err(BixverseErrors::DgeEmptyReferenceGroup)
        ));
        assert!(matches!(
            validate_one_vs_many(&[0], &[]),
            Err(BixverseErrors::DgeNoComparisonGroups)
        ));
        assert!(matches!(
            validate_one_vs_many(&[0], &[vec![1], vec![]]),
            Err(BixverseErrors::DgeEmptyComparisonGroup { group: 1 })
        ));
        assert!(matches!(
            validate_one_vs_many(&[0, 1], &[vec![2], vec![1]]),
            Err(BixverseErrors::DgeOverlappingGroups { group: 1, cell: 1 })
        ));
        assert!(validate_one_vs_many(&[0, 1], &[vec![2], vec![3]]).is_ok());
    }

    #[test]
    fn test_auc_per_cell_mw_precision() {
        // 20k genes, the top 1000 forming the set. The exact answer is 1.0,
        // but the rank sum reaches 1.95e7, past the point where f32 can
        // accumulate integers exactly, so an f32 sum drifts visibly.
        let n_genes = 20_000;
        let ranks = identity_ranks(n_genes);
        let gene_set: Vec<usize> = (n_genes - 1_000..n_genes).collect();

        assert_relative_eq!(calculate_auc_per_cell_mw(&ranks, &gene_set), 1.0);
    }

    /// Twelve cells over six genes: cells 0-3 are the reference, 4-7 rival A,
    /// 8-11 rival B. Gene 5 is filler that pins every library size to 100, so
    /// a normalised value depends only on the raw count and the intended ties
    /// survive the f16 quantisation exactly.
    ///
    /// * gene 0 - reference only, so a perfect marker against both rivals
    /// * gene 1 - shared with rival A, so it only separates the reference from B
    /// * gene 2 - rival A only, i.e. a down-marker
    /// * gene 3 - constant everywhere, the degenerate zero-variance case
    /// * gene 4 - never expressed, so it must be filtered out
    /// * gene 5 - filler, higher in B than in the reference
    ///
    /// ### Returns
    ///
    /// `dense[cell][gene]` raw counts.
    fn one_vs_many_counts() -> Vec<Vec<u32>> {
        let mut dense = Vec::with_capacity(12);

        // Reference: gene 0 expressed, gene 1 shared with rival A
        for &g0 in &[10_u32, 12, 11, 13] {
            dense.push(vec![g0, 20, 0, 5, 0, 75 - g0]);
        }
        // Rival A: gene 1 shared with the reference, gene 2 its own
        for &g2 in &[10_u32, 12, 11, 13] {
            dense.push(vec![0, 20, g2, 5, 0, 75 - g2]);
        }
        // Rival B: nothing but the constant gene and filler
        for _ in 0..4 {
            dense.push(vec![0, 0, 0, 5, 0, 95]);
        }

        dense
    }

    #[test]
    fn test_one_vs_many_auroc_end_to_end() {
        let temp = TempStore::new("one_vs_many");
        let reader = reader_for(&temp, &one_vs_many_counts());

        let res = calculate_dge_one_vs_many_auroc(
            &reader,
            &[0, 1, 2, 3],
            &[vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
            0.5,
            "greater",
            0,
        )
        .expect("one-vs-many runs");

        // Gene 4 is never expressed, so it is the only one dropped
        assert_eq!(res.genes_to_keep, vec![true, true, true, true, false, true]);

        // Kept gene order is 0, 1, 2, 3, 5
        assert_eq!(res.auroc[0], vec![1.0, 0.5, 0.0, 0.5, 0.5]);
        assert_eq!(res.auroc[1], vec![1.0, 1.0, 0.5, 0.5, 0.0]);

        assert_eq!(res.median_auroc, vec![1.0, 0.75, 0.25, 0.5, 0.25]);
        assert_eq!(res.min_auroc, vec![1.0, 0.5, 0.0, 0.5, 0.0]);
        assert_eq!(res.max_auroc, vec![1.0, 1.0, 0.5, 0.5, 0.5]);

        // Gene 1 is only let down by rival A, gene 5 only by rival B
        assert_eq!(res.worst_comparison[1], 0);
        assert_eq!(res.worst_comparison[4], 1);

        // Competition ranks per comparison are [1,2,5,2,2] and [1,1,3,3,5]
        assert_eq!(res.min_rank, vec![1, 1, 3, 2, 2]);

        // Genes constant across a pair must not produce NaN Z-scores
        assert_eq!(res.z_scores[0][1], 0.0);
        assert_eq!(res.z_scores[0][3], 0.0);
        assert_eq!(res.z_scores[1][2], 0.0);
        assert!(res.z_scores.iter().flatten().all(|z| z.is_finite()));

        // Gene 0 is the only unambiguous marker, and it is the top one
        assert!(res.median_auroc[0] > res.median_auroc[1]);
        assert!(res.simes_p[0] < res.simes_p[2]);
        assert!(res.max_p[0] < res.max_p[1]);

        // Detection proportions are reported over the kept genes
        assert_eq!(res.prop_ref, vec![1.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(res.prop_other[0], vec![0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_one_vs_many_matches_pairwise() {
        // The one-vs-many arm against a single rival must agree with the
        // pairwise entry point, which shares the same kernel but filters genes
        // on its own two groups.
        let temp = TempStore::new("vs_pairwise");
        let reader = reader_for(&temp, &one_vs_many_counts());

        let multi = calculate_dge_one_vs_many_auroc(
            &reader,
            &[0, 1, 2, 3],
            &[vec![4, 5, 6, 7]],
            0.5,
            "greater",
            0,
        )
        .expect("one-vs-many runs");

        let pairwise = calculate_dge_grps_mann_whitney(
            &reader,
            &[0, 1, 2, 3],
            &[4, 5, 6, 7],
            0.5,
            "greater",
            0,
        )
        .expect("pairwise runs");

        assert_eq!(multi.genes_to_keep, pairwise.genes_to_keep);
        assert_eq!(multi.auroc[0], pairwise.auroc);
        assert_eq!(multi.z_scores[0], pairwise.z_scores);
        assert_eq!(multi.lfc[0], pairwise.lfc);
        assert_eq!(multi.prop_ref, pairwise.prop1);
        assert_eq!(multi.prop_other[0], pairwise.prop2);
    }

    #[test]
    fn test_one_vs_many_all_genes_filtered() {
        let temp = TempStore::new("all_filtered");
        let reader = reader_for(&temp, &one_vs_many_counts());

        // Nothing clears a proportion above 1.0
        let res = calculate_dge_one_vs_many_auroc(
            &reader,
            &[0, 1, 2, 3],
            &[vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
            1.5,
            "greater",
            0,
        )
        .expect("one-vs-many runs");

        assert!(res.genes_to_keep.iter().all(|&keep| !keep));
        assert_eq!(res.auroc.len(), 2);
        assert!(res.auroc.iter().all(|a| a.is_empty()));
        assert!(res.median_auroc.is_empty());
    }

    #[test]
    fn test_parse_auc_type() {
        assert!(matches!(parse_auc_type("AUROC"), Some(AucType::Recovery)));
        assert!(matches!(parse_auc_type("aucell"), Some(AucType::Recovery)));
        assert!(matches!(
            parse_auc_type("wilcox"),
            Some(AucType::MannWhitney)
        ));
        assert!(matches!(
            parse_auc_type("aupr"),
            Some(AucType::AveragePrecision)
        ));
        assert!(parse_auc_type("nonsense").is_none());
    }
}
