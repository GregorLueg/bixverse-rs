//! Differential gene expression for single cell via Mann Whitney U and AUCell
//! type enrichment of a bag of genes, see Aibar, et al., Nat Methods,
//! 2017.

use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::stats::{calc_fdr, z_scores_to_pval};
use crate::prelude::*;
use crate::single_cell::sc_analysis::fast_ranking::rank_csr_chunk_vec;

/////////
// DGE //
/////////

////////////////
// Structures //
////////////////

/// Structure to store the Mann Whitney U-based DGE results in
#[derive(Clone, Debug)]
pub struct DgeMannWhitneyRes {
    /// The calculated LFCs
    pub lfc: Vec<f32>,
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

/// Calculates the Mann Whitney statistic
///
/// ### Params
///
/// * `ranks1` - Ranked data for the gene of group 1
/// * `ranks2` - Ranked data for the gene of group 2
///
/// ### Returns
///
/// Returns the Z-score based on Mann-Whitney test
fn mann_whitney_u_test(ranks1: &[f32], ranks2: &[f32]) -> f64 {
    let n1 = ranks1.len() as f64;
    let n2 = ranks2.len() as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return 0.0;
    }

    let r1: f64 = ranks1.iter().map(|&r| r as f64).sum();
    let u1 = n1 * n2 + n1 * (n1 + 1.0) / 2.0 - r1;

    let mean = n1 * n2 / 2.0;
    let variance = n1 * n2 * (n1 + n2 + 1.0) / 12.0;

    (mean - u1) / variance.sqrt()
}

//////////
// Main //
//////////

/// Get differential expression based on Mann-Whitney
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

    let mut combined_chunks = cell_chunks_1;
    combined_chunks.extend(cell_chunks_2);

    let start_ranking = Instant::now();

    let all_ranks = rank_csr_chunk_vec(combined_chunks, no_genes_new, false);

    let end_ranking = start_ranking.elapsed();

    if verbosity.normal_verbosity() {
        println!("Finished the ranking across cells: {:.2?}", end_ranking);
    }

    let start_calculations = Instant::now();

    let res: Vec<(f32, f32, f32, f64)> = genes_kept
        .par_iter()
        .enumerate()
        .map(|(new_idx, &original_idx)| {
            let log_fc = avg_exp_1[original_idx] - avg_exp_2[original_idx];
            let prop1 = prop_1[original_idx];
            let prop2 = prop_2[original_idx];
            let gene_ranks = &all_ranks[new_idx];

            let z = mann_whitney_u_test(&gene_ranks[..n1], &gene_ranks[n1..]);

            (log_fc, prop1, prop2, z)
        })
        .collect();

    let mut log_fc = Vec::with_capacity(res.len());
    let mut prop1 = Vec::with_capacity(res.len());
    let mut prop2 = Vec::with_capacity(res.len());
    let mut z_scores = Vec::with_capacity(res.len());

    for (log_fc_i, prop1_i, prop2_i, z_i) in res {
        log_fc.push(log_fc_i);
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
        prop1,
        prop2,
        z_scores,
        p_vals,
        fdr,
        genes_to_keep,
    })
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

    // Sum of ranks for genes in the set
    let rank_sum: f32 = gene_set.iter().map(|&idx| ranks[idx]).sum();

    // U statistic
    let u = rank_sum - (n_in_set * (n_in_set + 1)) as f32 / 2.0;

    // Convert to AUC
    u / (n_in_set * n_not_in_set) as f32
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
    use approx::assert_relative_eq;

    /// Identity ascending ranks: gene index `i` has ascending rank `i + 1`,
    /// i.e. descending rank `n - i`.
    fn identity_ranks(n: usize) -> Vec<f32> {
        (1..=n).map(|r| r as f32).collect()
    }

    /// Gene indices whose ascending ranks are the given values.
    fn genes_with_asc_ranks(asc: &[usize]) -> Vec<usize> {
        asc.iter().map(|&r| r - 1).collect()
    }

    #[test]
    fn test_mw_auc_perfect_separation() {
        let ranks = identity_ranks(10);

        let top = vec![7, 8, 9];
        assert_relative_eq!(calculate_auc_per_cell_mw(&ranks, &top), 1.0);

        let bottom = vec![0, 1, 2];
        assert_relative_eq!(calculate_auc_per_cell_mw(&ranks, &bottom), 0.0);
    }

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

    #[test]
    fn test_resolve_max_rank() {
        assert_eq!(resolve_max_rank(None, 20_000), 1_000);
        assert_eq!(resolve_max_rank(Some(50), 20_000), 50);
        assert_eq!(resolve_max_rank(Some(0), 20_000), 1);
        assert_eq!(resolve_max_rank(Some(99_999), 20_000), 20_000);
        // Ceiling keeps tiny universes at a usable cutoff
        assert_eq!(resolve_max_rank(None, 3), 1);
    }

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
